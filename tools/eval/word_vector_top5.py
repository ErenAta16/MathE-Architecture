"""Export separate Word2Vec and GloVe top-5 keyword scores for the 15 videos.

This script is intentionally offline: it does not call the VLM, does not
download videos, and does not read/write video caches. It takes the fixed VLM
title+summary/query text from a JSON reference file and scores the fixed
keyword pool with Word2Vec and GloVe separately.

Default input:
  data/gold/videos_15_graded_vlm_reference_nocache.json

Outputs:
  reports/word_vector_top5_separate_15videos.md
  reports/word_vector_top5_separate_15videos.csv
  reports/word_vector_top5_separate_15videos.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from keyword_eval import DEFAULT_KEYWORD_POOL  # noqa: E402
from reranker import default_rerank_config, rerank_pool  # noqa: E402
from similarity_compare import METRICS, encode_texts  # noqa: E402

DEFAULT_INPUT = _ROOT / "data" / "gold" / "videos_15_graded_vlm_reference_nocache.json"
DEFAULT_OUT_PREFIX = _ROOT / "reports" / "word_vector_top5_separate_15videos"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        key = text.lower()
        if text and key not in seen:
            out.append(text)
            seen.add(key)
    return out


def _load_items(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return [x for x in data["items"] if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("videos"), list):
        return [x for x in data["videos"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise ValueError(f"Could not find items/videos list in {path}")


def _load_pool(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_KEYWORD_POOL)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _dedupe([line.strip() for line in text.splitlines()])

    raw: list[Any]
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("keywords") or data.get("pool") or data.get("items") or []
    else:
        raw = []

    pool: list[str] = []
    for item in raw:
        if isinstance(item, str):
            pool.append(item)
        elif isinstance(item, dict):
            pool.append(str(item.get("keyword") or item.get("name") or "").strip())
    pool = _dedupe(pool)
    if not pool:
        raise ValueError(f"No keywords found in pool file: {path}")
    return pool


def _item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("item_id") or item.get("video_id") or "").strip()


def _query_text(item: dict[str, Any]) -> str:
    for key in ("query_text", "query", "text", "vlm_text"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    title = str(item.get("title") or "").strip()
    summary = str(item.get("summary") or "").strip()
    if title and summary:
        return f"{title}\n\n{summary}"
    return title or summary


def _llm_order(item: dict[str, Any]) -> list[str]:
    """Best available VLM keyword order for reproducing the UI hybrid step."""
    for key in ("reference_keywords", "keywords_llm", "keywords", "gold_keywords"):
        raw = item.get(key)
        if isinstance(raw, list):
            out = _dedupe([str(x) for x in raw])
            if out:
                return out
    return []


def _rank_batched(
    items: list[dict[str, Any]],
    pool: list[str],
    *,
    model: str,
    metric: str,
) -> dict[str, list[dict[str, Any]]]:
    metric_fn = METRICS[metric]
    queries = [_query_text(item) for item in items]
    embeddings = encode_texts(model, queries + list(pool))
    query_vecs = embeddings[: len(queries)]
    pool_vecs = embeddings[len(queries):]

    by_id: dict[str, list[dict[str, Any]]] = {}
    for i, item in enumerate(items):
        ranked: list[dict[str, Any]] = []
        for keyword, keyword_vec in zip(pool, pool_vecs):
            score = float(metric_fn(query_vecs[i], keyword_vec))
            ranked.append({"keyword": keyword, "score": round(score, 6)})
        ranked.sort(key=lambda row: (-float(row["score"]), str(row["keyword"]).lower()))
        by_id[_item_id(item)] = ranked
    return by_id


def _fmt_top(top: list[dict[str, Any]]) -> str:
    return "; ".join(f"{i}. {row['keyword']} ({float(row['score']):.3f})" for i, row in enumerate(top, 1))


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    if span <= 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / span for k, v in scores.items()}


def _ensemble_rank(
    pool: list[str],
    word2vec_ranked: list[dict[str, Any]],
    glove_ranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match semantic_similarity.py's word2vec_glove backend.

    Production first min-max normalizes Word2Vec and GloVe scores across the
    candidate pool, then averages the two normalized values.
    """
    w2v_raw = {str(r["keyword"]): float(r["score"]) for r in word2vec_ranked}
    glove_raw = {str(r["keyword"]): float(r["score"]) for r in glove_ranked}
    w2v_norm = _minmax(w2v_raw)
    glove_norm = _minmax(glove_raw)
    out: list[dict[str, Any]] = []
    for keyword in pool:
        w = w2v_norm.get(keyword, 0.0)
        g = glove_norm.get(keyword, 0.0)
        out.append({
            "keyword": keyword,
            "score": round((w + g) / 2.0, 6),
            "components": {
                "word2vec_norm": round(w, 6),
                "glove_norm": round(g, 6),
                "word2vec_raw": round(w2v_raw.get(keyword, 0.0), 6),
                "glove_raw": round(glove_raw.get(keyword, 0.0), 6),
            },
        })
    out.sort(key=lambda row: (-float(row["score"]), str(row["keyword"]).lower()))
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "video_no",
        "video_id",
        "url",
        "title",
        "topic",
        "subtopic",
        "metric",
        "rank",
        "keyword",
        "score",
        "embedding_raw",
        "embedding_norm",
        "vlm_rrf_norm",
        "vlm_rank",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary_rows: list[dict[str, Any]], *, metric: str, pool_size: int) -> None:
    lines: list[str] = []
    lines.append("# Separate Word2Vec and GloVe Top-5 Scores")
    lines.append("")
    lines.append(f"- Metric: `{metric}`")
    lines.append(f"- Keyword pool size: `{pool_size}`")
    lines.append("- `Raw` scores are pure per-model similarity scores.")
    lines.append("- `+VLM` scores reproduce the UI-style hybrid step with the available VLM keyword order.")
    lines.append("")
    lines.append("## Raw Embedding Top-5")
    lines.append("")
    lines.append("| # | Video | Topic / Subtopic | Word2Vec raw top-5 | GloVe raw top-5 | W2V+GloVe raw top-5 |")
    lines.append("|---:|---|---|---|---|---|")
    for row in summary_rows:
        video = f"{row['title']}<br>`{row['video_id']}`"
        tax = f"{row['topic']} / {row['subtopic']}"
        lines.append(
            "| {video_no} | {video} | {tax} | {w2v} | {glove} | {ensemble} |".format(
                video_no=row["video_no"],
                video=video.replace("|", "\\|"),
                tax=tax.replace("|", "\\|"),
                w2v=str(row["word2vec_raw_top5"]).replace("|", "\\|"),
                glove=str(row["glove_raw_top5"]).replace("|", "\\|"),
                ensemble=str(row["word2vec_glove_raw_top5"]).replace("|", "\\|"),
            )
        )
    lines.append("")
    lines.append("## VLM-Hybrid Top-5")
    lines.append("")
    lines.append("| # | Video | VLM+Word2Vec top-5 | VLM+GloVe top-5 | VLM+(W2V+GloVe) top-5 |")
    lines.append("|---:|---|---|---|---|")
    for row in summary_rows:
        video = f"{row['title']}<br>`{row['video_id']}`"
        lines.append(
            "| {video_no} | {video} | {w2v} | {glove} | {ensemble} |".format(
                video_no=row["video_no"],
                video=video.replace("|", "\\|"),
                w2v=str(row["word2vec_vlm_top5"]).replace("|", "\\|"),
                glove=str(row["glove_vlm_top5"]).replace("|", "\\|"),
                ensemble=str(row["word2vec_glove_vlm_top5"]).replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_scores(
    *,
    input_path: Path,
    pool_path: Path | None,
    out_prefix: Path,
    top_k: int,
    metric: str,
) -> dict[str, Any]:
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}. Choose one of: {', '.join(METRICS)}")
    items = _load_items(input_path)
    pool = _load_pool(pool_path)
    if not items:
        raise ValueError(f"No video items found in {input_path}")

    print(f"[INFO] Videos: {len(items)}")
    print(f"[INFO] Keyword pool: {len(pool)}")
    print("[INFO] Scoring Word2Vec...")
    w2v = _rank_batched(items, pool, model="word2vec", metric=metric)
    print("[INFO] Scoring GloVe...")
    glove = _rank_batched(items, pool, model="glove", metric=metric)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    json_items: list[dict[str, Any]] = []

    for idx, item in enumerate(items, 1):
        vid = _item_id(item)
        w2v_top = w2v.get(vid, [])[:top_k]
        glove_top = glove.get(vid, [])[:top_k]
        ensemble = _ensemble_rank(pool, w2v.get(vid, []), glove.get(vid, []))
        ensemble_top = ensemble[:top_k]
        w2v_keywords = [str(row["keyword"]) for row in w2v_top]
        glove_keywords = [str(row["keyword"]) for row in glove_top]
        raw_overlap = len({k.lower() for k in w2v_keywords} & {k.lower() for k in glove_keywords})

        llm_order = _llm_order(item)
        rcfg = default_rerank_config()
        w2v_hybrid = rerank_pool(
            cosine_scores={r["keyword"]: float(r["score"]) for r in w2v.get(vid, [])},
            llm_ranking=llm_order,
            cfg=rcfg,
        )[:top_k]
        glove_hybrid = rerank_pool(
            cosine_scores={r["keyword"]: float(r["score"]) for r in glove.get(vid, [])},
            llm_ranking=llm_order,
            cfg=rcfg,
        )[:top_k]
        ensemble_hybrid = rerank_pool(
            cosine_scores={r["keyword"]: float(r["score"]) for r in ensemble},
            llm_ranking=llm_order,
            cfg=rcfg,
        )[:top_k]
        w2v_hybrid_keywords = [str(row["keyword"]) for row in w2v_hybrid]
        glove_hybrid_keywords = [str(row["keyword"]) for row in glove_hybrid]
        vlm_overlap = len(
            {k.lower() for k in w2v_hybrid_keywords}
            & {k.lower() for k in glove_hybrid_keywords}
        )

        common = {
            "video_no": idx,
            "video_id": vid,
            "url": str(item.get("url") or ""),
            "title": str(item.get("title") or vid),
            "topic": str(item.get("topic") or ""),
            "subtopic": str(item.get("subtopic") or ""),
        }
        summary_rows.append({
            **common,
            "word2vec_raw_top5": _fmt_top(w2v_top),
            "glove_raw_top5": _fmt_top(glove_top),
            "word2vec_glove_raw_top5": _fmt_top(ensemble_top),
            "raw_overlap_count": raw_overlap,
            "word2vec_vlm_top5": _fmt_top(w2v_hybrid),
            "glove_vlm_top5": _fmt_top(glove_hybrid),
            "word2vec_glove_vlm_top5": _fmt_top(ensemble_hybrid),
            "vlm_overlap_count": vlm_overlap,
        })
        json_items.append({
            **common,
            "vlm_order": llm_order,
            "raw_word2vec": w2v_top,
            "raw_glove": glove_top,
            "raw_word2vec_glove": ensemble_top,
            "raw_overlap_count": raw_overlap,
            "vlm_word2vec": w2v_hybrid,
            "vlm_glove": glove_hybrid,
            "vlm_word2vec_glove": ensemble_hybrid,
            "vlm_overlap_count": vlm_overlap,
        })

        for model_name, top in (
            ("word2vec_raw", w2v_top),
            ("glove_raw", glove_top),
            ("word2vec_glove_raw", ensemble_top),
            ("word2vec_vlm_hybrid", w2v_hybrid),
            ("glove_vlm_hybrid", glove_hybrid),
            ("word2vec_glove_vlm_hybrid", ensemble_hybrid),
        ):
            for rank, row in enumerate(top, 1):
                comps = row.get("components") if isinstance(row.get("components"), dict) else {}
                detail_rows.append({
                    **common,
                    "metric": model_name,
                    "rank": rank,
                    "keyword": row["keyword"],
                    "score": row["score"],
                    "embedding_raw": comps.get("embedding_raw", ""),
                    "embedding_norm": comps.get("embedding", ""),
                    "vlm_rrf_norm": comps.get("rrf_llm", ""),
                    "vlm_rank": comps.get("llm_rank", ""),
                })

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    json_path = out_prefix.with_suffix(".json")

    _write_csv(csv_path, detail_rows)
    _write_markdown(md_path, summary_rows, metric=metric, pool_size=len(pool))
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "word-vector-top5-v1",
                "input_path": str(input_path),
                "pool_path": str(pool_path) if pool_path else "keyword_eval.DEFAULT_KEYWORD_POOL",
                "metric": metric,
                "top_k": top_k,
                "pool_size": len(pool),
                "items": json_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "json": str(json_path),
        "summary_rows": summary_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--metric", choices=sorted(METRICS), default="cosine")
    args = parser.parse_args()

    result = export_scores(
        input_path=args.input.resolve(),
        pool_path=args.pool.resolve() if args.pool else None,
        out_prefix=args.out_prefix.resolve(),
        top_k=max(1, args.top_k),
        metric=args.metric,
    )
    print(f"[OK] CSV: {result['csv']}")
    print(f"[OK] Markdown: {result['markdown']}")
    print(f"[OK] JSON: {result['json']}")
    print("")
    for row in result["summary_rows"]:
        print(f"{row['video_no']:02d}. {row['title']} [{row['video_id']}]")
        print(f"    Word2Vec raw : {row['word2vec_raw_top5']}")
        print(f"    GloVe raw    : {row['glove_raw_top5']}")
        print(f"    W2V+GloVe raw: {row['word2vec_glove_raw_top5']}")
        print(f"    Raw overlap  : {row['raw_overlap_count']}/5")
        print(f"    W2V + VLM    : {row['word2vec_vlm_top5']}")
        print(f"    GloVe + VLM  : {row['glove_vlm_top5']}")
        print(f"    VLM+(W2V+G)  : {row['word2vec_glove_vlm_top5']}")
        print(f"    VLM overlap  : {row['vlm_overlap_count']}/5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
