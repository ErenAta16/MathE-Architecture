"""
Compare semantic keyword-ranking backends on a fixed video test set.

This is the harness for Beatriz's second task: keep the video set, VLM text
output, gold keywords, and closed keyword pool fixed; replace only the
embedding backend used for semantic ranking.

Default experiment:
  - Models: BERT, Word2Vec, GloVe
  - Metric: cosine similarity
  - Output: P@5, R@5, F1@5, Jaccard, top-5 keywords, and OOV coverage

The script does not call Gemini and does not write video-analysis caches. It
reads query text from the gold/query file or from existing video cache files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from keyword_eval import DEFAULT_KEYWORD_POOL  # noqa: E402
from similarity_compare import (  # noqa: E402
    ENCODERS,
    METRICS,
    encode_texts,
    rank_pool_bm25,
    static_coverage,
    top_n_mmr,
)
from step_eval import aggregate, ndcg_at_k, score_item  # noqa: E402

_log = logging.getLogger("eval_multi")

DEFAULT_GOLD_PATH = _ROOT / "data" / "gold" / "videos.json"
DEFAULT_CACHE_DIR = _ROOT / "step_pipeline" / "video_cache"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _sha256_items(items: list[str]) -> str:
    blob = "\n".join(str(x).strip() for x in items).encode("utf-8")
    return _sha256_text(blob.decode("utf-8"))


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item).strip()
        key = s.lower()
        if not s or key in seen:
            continue
        out.append(s)
        seen.add(key)
    return out


def _load_pool(path: str | None) -> tuple[list[str], str]:
    """Load keyword pool from JSON/list/text, or fall back to DEFAULT_KEYWORD_POOL."""
    if not path:
        return list(DEFAULT_KEYWORD_POOL), "keyword_eval.DEFAULT_KEYWORD_POOL"

    fp = Path(path).resolve()
    if not fp.is_file():
        raise FileNotFoundError(f"Keyword pool file not found: {fp}")

    text = fp.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    raw: list[Any]
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        if isinstance(data.get("keywords"), list):
            raw = data["keywords"]
        elif isinstance(data.get("pool"), list):
            raw = data["pool"]
        elif isinstance(data.get("items"), list):
            raw = data["items"]
        else:
            raw = []
    else:
        raw = [line.strip() for line in text.splitlines()]

    pool: list[str] = []
    for item in raw:
        if isinstance(item, str):
            pool.append(item)
        elif isinstance(item, dict):
            pool.append(str(item.get("keyword") or item.get("name") or "").strip())
    pool = _dedupe(pool)
    if not pool:
        raise ValueError(f"No keywords found in pool file: {fp}")
    return pool, str(fp)


def _item_id(item: dict) -> str:
    return str(item.get("id") or item.get("item_id") or item.get("video_id") or "").strip()


def _title_summary_query(title: str, summary: str) -> str:
    title = (title or "").strip()
    summary = (summary or "").strip()
    if title and summary:
        return f"{title}\n\n{summary}"
    return title or summary


def _query_from_item(item: dict) -> tuple[str, str]:
    """Return explicit query text from a gold/query item if present."""
    for key in ("query_text", "query", "text", "vlm_text"):
        value = str(item.get(key) or "").strip()
        if value:
            return value, key

    query = _title_summary_query(str(item.get("title") or ""), str(item.get("summary") or ""))
    if query:
        return query, "title+summary"

    predicted = item.get("predicted")
    if isinstance(predicted, dict):
        query = _title_summary_query(
            str(predicted.get("title") or ""),
            str(predicted.get("summary") or ""),
        )
        if query:
            return query, "predicted.title+summary"

    return "", ""


def _latest_cache_payload(video_id: str, cache_dir: Path, mode: str) -> tuple[dict | None, str]:
    """Read the newest matching video-analysis cache for an id."""
    if not video_id or not cache_dir.exists():
        return None, ""

    patterns = []
    if mode != "any":
        patterns.append(f"yt_{video_id}_{mode}_*.json")
    patterns.extend([
        f"yt_{video_id}_quick_*.json",
        f"yt_{video_id}_deep_*.json",
        f"yt_{video_id}_*.json",
    ])

    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for fp in cache_dir.glob(pattern):
            if fp.is_file() and fp not in seen:
                files.append(fp)
                seen.add(fp)
    if not files:
        return None, ""

    files.sort(key=lambda p: p.stat().st_mtime)
    fp = files[-1]
    try:
        return _read_json(fp), str(fp)
    except Exception as e:
        _log.warning("Could not read cache %s: %s", fp, e)
        return None, ""


def _query_for_item(
    item: dict,
    *,
    cache_dir: Path,
    cache_mode: str,
    query_source: str,
) -> tuple[str, str]:
    explicit, explicit_source = _query_from_item(item)
    vid = _item_id(item)

    if query_source in ("gold", "auto") and explicit:
        return explicit, explicit_source

    if query_source in ("cache", "auto"):
        payload, fp = _latest_cache_payload(vid, cache_dir, cache_mode)
        if payload:
            query = _title_summary_query(
                str(payload.get("title") or ""),
                str(payload.get("summary") or ""),
            )
            if query:
                return query, f"cache:{Path(fp).name}"

    if query_source == "gold" and not explicit:
        return "", ""

    # Last-resort fallback keeps dry harness checks possible, but strict mode
    # can reject it for real experiments.
    fallback = str(item.get("title_hint") or vid).strip()
    return fallback, "title_hint_fallback"


def _score_combo(
    items: list[dict],
    queries: dict[str, dict],
    *,
    model: str,
    metric: str,
    pool: list[str],
    top_k: int,
    include_ranking: int,
) -> dict:
    rows: list[dict] = []
    metrics = []
    graded_metrics = []
    failures = 0
    metric_fn = METRICS[metric]
    item_ids = [_item_id(item) for item in items]
    query_texts = [str((queries.get(vid) or {}).get("query_text") or "") for vid in item_ids]
    batch_error = ""

    try:
        # Encode every query plus the keyword pool in a single model call. This
        # avoids recomputing the same 100+ pool vectors for every video.
        embeddings = encode_texts(model, query_texts + list(pool))
        query_vecs = embeddings[: len(query_texts)]
        pool_vecs = embeddings[len(query_texts):]
    except Exception as e:
        batch_error = str(e)[:240]
        _log.warning("[%s x %s] batch encoding failed: %s", model, metric, batch_error)
        query_vecs = []
        pool_vecs = []
        failures = len(items)

    for i, item in enumerate(items):
        vid = item_ids[i]
        qrec = queries.get(vid, {})

        try:
            if len(query_vecs) == 0 or len(pool_vecs) == 0:
                raise RuntimeError(batch_error or "embedding batch unavailable")
            ranked = []
            qv = query_vecs[i]
            for kw, kv in zip(pool, pool_vecs):
                score = metric_fn(qv, kv)
                ranked.append({"keyword": kw, "score": round(float(score), 6)})
            ranked.sort(key=lambda d: (-d["score"], d["keyword"].lower()))
            top = [r["keyword"] for r in ranked[:top_k]]
            ranking = ranked[:include_ranking]
        except Exception as e:
            if failures < len(items):
                failures += 1
            top = []
            ranking = []
            if not batch_error:
                _log.warning("[%s x %s] %s failed: %s", model, metric, vid, str(e)[:180])

        m = score_item(
            item_id=vid,
            predicted_keywords=top,
            gold_keywords=item.get("gold_keywords") or [],
            predicted_topic=item.get("topic"),
            predicted_subtopic=item.get("subtopic"),
            gold_topic=item.get("topic"),
            gold_subtopic=item.get("subtopic"),
            k=top_k,
        )
        metrics.append(m)
        graded = _score_graded(top, item, top_k=top_k)
        if graded:
            graded_metrics.append(graded)
        rows.append({
            **m.to_dict(),
            "query_source": qrec.get("source"),
            "query_hash": qrec.get("query_hash"),
            "gold_keywords": list(item.get("gold_keywords") or []),
            "graded_reference": list(item.get("graded_reference") or []),
            "graded_metrics": graded,
            "top_keywords": top,
            "ranking": ranking,
        })

    agg = aggregate(metrics)
    if graded_metrics:
        n = len(graded_metrics)
        agg.update({
            "graded_n": n,
            "ndcg_at_5_mean": round(sum(g["ndcg_at_5"] for g in graded_metrics) / n, 4),
            "mean_relevance_at_5": round(sum(g["mean_relevance_at_5"] for g in graded_metrics) / n, 4),
            "weighted_precision_at_5": round(sum(g["weighted_precision_at_5"] for g in graded_metrics) / n, 4),
            "mrr_relevance_ge_2": round(sum(g["mrr_relevance_ge_2"] for g in graded_metrics) / n, 4),
        })

    return {
        "model": model,
        "metric": metric,
        "aggregate": agg,
        "items": rows,
        "failures": failures,
    }


def _score_graded(predicted: list[str], item: dict, *, top_k: int) -> dict:
    """Score a ranked prediction list against item['graded_reference'] if present."""
    graded_raw = item.get("graded_reference") or []
    if not isinstance(graded_raw, list) or not graded_raw:
        return {}
    rel_map: dict[str, float] = {}
    ideal_rels: list[float] = []
    for row in graded_raw:
        if not isinstance(row, dict):
            continue
        kw = str(row.get("keyword") or "").strip()
        if not kw:
            continue
        try:
            rel = max(0.0, min(3.0, float(row.get("relevance", 0))))
        except (TypeError, ValueError):
            rel = 0.0
        if rel <= 0:
            continue
        rel_map[kw.lower()] = rel
        ideal_rels.append(rel)

    if not rel_map:
        return {}
    top = list(predicted or [])[:top_k]
    rels = [rel_map.get(str(kw).strip().lower(), 0.0) for kw in top]
    max_gain = 3.0 * max(1, min(top_k, len(top)))
    first_rank = 0.0
    for idx, rel in enumerate(rels, start=1):
        if rel >= 2.0:
            first_rank = 1.0 / idx
            break
    return {
        "ndcg_at_5": round(ndcg_at_k(rels, ideal_rels, k=top_k), 4),
        "mean_relevance_at_5": round(sum(rels) / max(1, len(rels)), 4),
        "weighted_precision_at_5": round(sum(rels) / max_gain, 4),
        "mrr_relevance_ge_2": round(first_rank, 4),
        "relevances": rels,
    }


def _coverage_for_model(model: str, queries: dict[str, dict], pool: list[str]) -> dict:
    if model == "bert":
        return {"model": model, "note": "BERT uses subword tokenization; OOV coverage is not measured as static word vectors."}

    query_rows = []
    for vid, qrec in queries.items():
        cov = static_coverage(model, str(qrec.get("query_text") or ""))
        query_rows.append({"item_id": vid, **cov})

    keyword_rows = []
    zero_keywords = []
    for kw in pool:
        cov = static_coverage(model, kw)
        keyword_rows.append({"keyword": kw, **cov})
        if cov["matched_tokens"] == 0:
            zero_keywords.append(kw)

    q_avg = 0.0
    if query_rows:
        q_avg = sum(float(r["coverage"]) for r in query_rows) / len(query_rows)
    kw_avg = 0.0
    if keyword_rows:
        kw_avg = sum(float(r["coverage"]) for r in keyword_rows) / len(keyword_rows)

    return {
        "model": model,
        "query_coverage_mean": round(q_avg, 4),
        "keyword_coverage_mean": round(kw_avg, 4),
        "zero_vector_keywords": zero_keywords,
        "n_zero_vector_keywords": len(zero_keywords),
        "queries": query_rows,
        "keywords": keyword_rows,
    }


def _run_bm25(items: list[dict], queries: dict[str, dict], pool: list[str], top_k: int) -> dict:
    rows = []
    metrics = []
    for item in items:
        vid = _item_id(item)
        query = str((queries.get(vid) or {}).get("query_text") or "")
        ranked = rank_pool_bm25(query, pool)
        top = [r["keyword"] for r in ranked[:top_k]]
        m = score_item(
            vid,
            top,
            item.get("gold_keywords") or [],
            predicted_topic=item.get("topic"),
            predicted_subtopic=item.get("subtopic"),
            gold_topic=item.get("topic"),
            gold_subtopic=item.get("subtopic"),
            k=top_k,
        )
        metrics.append(m)
        rows.append({**m.to_dict(), "gold_keywords": list(item.get("gold_keywords") or []), "top_keywords": top})
    return {"model": "bm25", "metric": "term_frequency", "aggregate": aggregate(metrics), "items": rows}


def _run_mmr(items: list[dict], queries: dict[str, dict], pool: list[str], top_k: int) -> dict:
    rows = []
    metrics = []
    for item in items:
        vid = _item_id(item)
        query = str((queries.get(vid) or {}).get("query_text") or "")
        top = top_n_mmr(query, pool, model="bert", n=top_k, diversity=0.3)
        m = score_item(
            vid,
            top,
            item.get("gold_keywords") or [],
            predicted_topic=item.get("topic"),
            predicted_subtopic=item.get("subtopic"),
            gold_topic=item.get("topic"),
            gold_subtopic=item.get("subtopic"),
            k=top_k,
        )
        metrics.append(m)
        rows.append({**m.to_dict(), "gold_keywords": list(item.get("gold_keywords") or []), "top_keywords": top})
    return {"model": "bert_mmr", "metric": "cosine_diversity", "aggregate": aggregate(metrics), "items": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(DEFAULT_GOLD_PATH),
                        help="Gold/query JSON with an items list.")
    parser.add_argument("--pool", default="",
                        help="Keyword pool JSON/list/text file. Defaults to DEFAULT_KEYWORD_POOL.")
    parser.add_argument("--models", default="bert,word2vec,glove",
                        help=f"Comma-separated models. Available: {','.join(ENCODERS)}")
    parser.add_argument("--metrics", default="cosine",
                        help=f"Comma-separated metrics. Available: {','.join(METRICS)}")
    parser.add_argument("--query-source", choices=("auto", "gold", "cache"), default="auto",
                        help="Where to read fixed VLM/query text from.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                        help="Existing video cache directory used only for reading title+summary.")
    parser.add_argument("--cache-mode", choices=("quick", "deep", "any"), default="quick",
                        help="Preferred cache mode when reading fixed VLM text.")
    parser.add_argument("--strict-queries", action="store_true",
                        help="Fail if any item only has the title_hint/id fallback as query text.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ranking-limit", type=int, default=10,
                        help="How many ranked keywords to store per item/model.")
    parser.add_argument("--include-bm25", action="store_true",
                        help="Also run a lexical BM25 baseline.")
    parser.add_argument("--include-mmr", action="store_true",
                        help="Also run BERT+MMR as a diversity-aware secondary baseline.")
    parser.add_argument("--report-out", default="reports/semantic_model_comparison.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    gold_path = Path(args.gold).resolve()
    if not gold_path.is_file():
        print(f"Gold file not found: {gold_path}", file=sys.stderr)
        return 2
    gold = _read_json(gold_path)
    items = list(gold.get("items") or [])
    if not items:
        print(f"No items found in {gold_path}", file=sys.stderr)
        return 2

    pool, pool_source = _load_pool(args.pool or None)
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    unknown_models = [m for m in models if m not in ENCODERS]
    unknown_metrics = [m for m in metrics if m not in METRICS]
    if unknown_models or unknown_metrics:
        print(f"Unknown models={unknown_models} metrics={unknown_metrics}", file=sys.stderr)
        return 2

    cache_dir = Path(args.cache_dir).resolve()
    queries: dict[str, dict] = {}
    fallback_items = []
    for item in items:
        vid = _item_id(item)
        if not vid:
            print("Every item needs an id/item_id/video_id.", file=sys.stderr)
            return 2
        query, source = _query_for_item(
            item,
            cache_dir=cache_dir,
            cache_mode=args.cache_mode,
            query_source=args.query_source,
        )
        if source == "title_hint_fallback":
            fallback_items.append(vid)
        queries[vid] = {
            "item_id": vid,
            "source": source,
            "query_text": query,
            "query_hash": _sha256_text(query),
            "n_chars": len(query),
        }

    if args.strict_queries and fallback_items:
        print(
            "Missing fixed VLM/query text for: " + ", ".join(fallback_items) +
            "\nAdd query_text/title+summary to the gold file or run with --query-source cache after generating cache.",
            file=sys.stderr,
        )
        return 2

    print(f"Gold: {gold_path}")
    print(f"Videos: {len(items)}   Pool: {len(pool)} ({pool_source})")
    print(f"Models: {models}   Metrics: {metrics}   top_k={args.top_k}")
    if fallback_items:
        print(f"Warning: {len(fallback_items)} item(s) use title_hint/id fallback query text.")
    print()

    results: list[dict] = []
    coverage: dict[str, dict] = {}

    for model in models:
        coverage[model] = _coverage_for_model(model, queries, pool)
        for metric in metrics:
            print(f"  Running {model} x {metric}...", end=" ", flush=True)
            t0 = time.time()
            combo = _score_combo(
                items,
                queries,
                model=model,
                metric=metric,
                pool=pool,
                top_k=max(1, args.top_k),
                include_ranking=max(args.top_k, args.ranking_limit),
            )
            combo["elapsed_s"] = round(time.time() - t0, 1)
            combo["coverage_summary"] = {
                k: v for k, v in coverage[model].items()
                if k not in ("queries", "keywords")
            }
            results.append(combo)
            agg = combo["aggregate"]
            graded_tail = ""
            if "ndcg_at_5_mean" in agg:
                graded_tail = (
                    f"  nDCG={agg['ndcg_at_5_mean']:.3f}  "
                    f"W-P@{args.top_k}={agg['weighted_precision_at_5']:.3f}"
                )
            print(
                f"P@{args.top_k}={agg['precision_at_5_mean']:.3f}  "
                f"R@{args.top_k}={agg['recall_at_5_mean']:.3f}  "
                f"F1={agg['f1_at_5_mean']:.3f}  "
                f"Jacc={agg['jaccard_mean']:.3f}  "
                f"{graded_tail}  "
                f"({combo['elapsed_s']}s)"
            )

    if args.include_bm25:
        print("  Running BM25 baseline...", end=" ", flush=True)
        t0 = time.time()
        bm25 = _run_bm25(items, queries, pool, max(1, args.top_k))
        bm25["elapsed_s"] = round(time.time() - t0, 1)
        results.append(bm25)
        agg = bm25["aggregate"]
        print(f"P@{args.top_k}={agg['precision_at_5_mean']:.3f}  F1={agg['f1_at_5_mean']:.3f}")

    if args.include_mmr:
        print("  Running BERT+MMR baseline...", end=" ", flush=True)
        t0 = time.time()
        mmr = _run_mmr(items, queries, pool, max(1, args.top_k))
        mmr["elapsed_s"] = round(time.time() - t0, 1)
        results.append(mmr)
        agg = mmr["aggregate"]
        print(f"P@{args.top_k}={agg['precision_at_5_mean']:.3f}  F1={agg['f1_at_5_mean']:.3f}")

    print("\n" + "=" * 78)
    print(f"{'Model':<12} {'Metric':<14} {'P@5':>7} {'R@5':>7} {'F1@5':>7} {'Jacc':>7} {'nDCG':>7} {'W-P@5':>7} {'Time':>7}")
    print("-" * 78)
    for row in results:
        a = row["aggregate"]
        print(
            f"{row['model']:<12} {row['metric']:<14} "
            f"{a['precision_at_5_mean']:>7.3f} {a['recall_at_5_mean']:>7.3f} "
            f"{a['f1_at_5_mean']:>7.3f} {a['jaccard_mean']:>7.3f} "
            f"{a.get('ndcg_at_5_mean', 0):>7.3f} {a.get('weighted_precision_at_5', 0):>7.3f} "
            f"{row.get('elapsed_s', 0):>6.1f}s"
        )
    print("=" * 78)

    if args.report_out:
        out = Path(args.report_out)
        if not out.is_absolute():
            out = (_ROOT / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": "semantic-model-comparison-v2",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": "Compare BERT, Word2Vec, and GloVe semantic keyword ranking on a fixed video set.",
            "inputs": {
                "gold_path": str(gold_path),
                "gold_hash": _sha256_text(gold_path.read_text(encoding="utf-8")),
                "pool_source": pool_source,
                "pool_size": len(pool),
                "pool_hash": _sha256_items(pool),
                "query_source": args.query_source,
                "cache_dir": str(cache_dir),
                "cache_mode": args.cache_mode,
                "top_k": max(1, args.top_k),
                "models": models,
                "metrics": metrics,
            },
            "n_videos": len(items),
            "queries": list(queries.values()),
            "coverage": coverage,
            "combos": results,
        }
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
