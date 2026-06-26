"""Evaluate summary-assisted keyword-ranking strategies on the fixed video set.

This harness answers a focused question: does using the VLM summary as an
extra context signal improve top-5 keyword selection, or does it over-generalize
the selected keywords?

It does not call Gemini and does not run YouTube analysis. It reuses the fixed
title/summary/reference-keyword records from the graded 15-video file and tests
the same closed keyword pool under several context strategies:

  - current: title + summary, matching the existing global UI query
  - scene_only: title only for this global-video harness
  - summary_late_fusion: title score fused with summary score
  - summary_shortlist: summary narrows candidate pool, title+summary scores it
  - summary_shortlist_fusion: summary shortlist plus title/summary late fusion
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reranker import RerankConfig, rerank_pool  # noqa: E402
from step_eval import aggregate, ndcg_at_k, score_item  # noqa: E402
from taxonomy import keywords_for_taxonomy  # noqa: E402


DEFAULT_GRADED = _ROOT / "data" / "gold" / "videos_15_graded_vlm_reference_nocache.json"
DEFAULT_OUT_PREFIX = _ROOT / "reports" / "summary_context_ablation_15videos"

MODE_ORDER = (
    "current",
    "scene_only",
    "summary_late_fusion",
    "summary_shortlist",
    "summary_shortlist_fusion",
)
MODEL_ORDER = ("bert", "word2vec", "glove", "word2vec_glove")
_TOKEN_RE = __import__("re").compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
_log = logging.getLogger("summary_context_ablation")


def _default_keyword_pool() -> list[str]:
    """Read DEFAULT_KEYWORD_POOL without requiring Gemini dependencies.

    ``keyword_eval.py`` imports ``google.genai`` for live Task-2 calls. Offline
    ablation should not need that package, so we parse the constant from source
    when a normal import is unavailable.
    """
    try:
        from keyword_eval import DEFAULT_KEYWORD_POOL  # type: ignore

        return list(DEFAULT_KEYWORD_POOL)
    except Exception:
        source = (_ROOT / "keyword_eval.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "DEFAULT_KEYWORD_POOL":
                        value = ast.literal_eval(node.value)
                        return [str(x) for x in value]
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "DEFAULT_KEYWORD_POOL"
            ):
                value = ast.literal_eval(node.value)
                return [str(x) for x in value]
        raise RuntimeError("Could not load DEFAULT_KEYWORD_POOL from keyword_eval.py")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        text = str(item or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _tokenize(text: str) -> list[str]:
    return [m.group(0) for m in _TOKEN_RE.finditer((text or "").replace("-", " "))]


def _lookup_keys(token: str, *, model: str) -> list[str]:
    token = str(token or "").strip()
    if not token:
        return []
    if model == "word2vec":
        return _dedupe([token, token.lower(), token.capitalize(), token.upper()])
    return _dedupe([token.lower(), token, token.capitalize(), token.upper()])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 / (1.0 + np.linalg.norm(a - b)))


def _manhattan_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 / (1.0 + np.sum(np.abs(a - b))))


def _metric_score(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "euclidean":
        return _euclidean_similarity(a, b)
    if metric == "manhattan":
        return _manhattan_similarity(a, b)
    return _cosine(a, b)


def _needed_tokens(texts: list[str], *, model: str) -> set[str]:
    needed: set[str] = set()
    for text in texts:
        for token in _tokenize(text):
            needed.update(_lookup_keys(token, model=model))
    return needed


def _read_word2vec_subset(path: Path, needed: set[str]) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    if not needed:
        return found
    with gzip.open(path, "rb") as f:
        header = f.readline().decode("utf-8", errors="ignore").strip().split()
        total = int(header[0])
        dim = int(header[1])
        vec_bytes = dim * 4
        for _ in range(total):
            token_bytes = bytearray()
            while True:
                ch = f.read(1)
                if not ch:
                    return found
                if ch == b" ":
                    break
                if ch != b"\n":
                    token_bytes.extend(ch)
            token = token_bytes.decode("utf-8", errors="ignore")
            raw = f.read(vec_bytes)
            if token in needed:
                found[token] = np.frombuffer(raw, dtype=np.float32).copy()
                if len(found) >= len(needed):
                    break
    return found


def _read_glove_subset(path: Path, needed: set[str]) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    if not needed:
        return found
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            parts = line.rstrip().split(" ")
            token = parts[0]
            if token in needed:
                found[token] = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
                if len(found) >= len(needed):
                    break
    return found


def _static_vectors(model: str, texts: list[str]) -> np.ndarray:
    data_root = Path.home() / "gensim-data"
    if model == "word2vec":
        path = data_root / "word2vec-google-news-300" / "word2vec-google-news-300.gz"
        vectors = _read_word2vec_subset(path, _needed_tokens(texts, model=model))
    elif model == "glove":
        path = data_root / "glove-wiki-gigaword-300" / "glove-wiki-gigaword-300.gz"
        vectors = _read_glove_subset(path, _needed_tokens(texts, model=model))
    else:
        raise ValueError(f"Unsupported static model: {model}")

    encoded: list[np.ndarray] = []
    zero = np.zeros(300, dtype=np.float32)
    for text in texts:
        found = []
        for token in _tokenize(text):
            for key in _lookup_keys(token, model=model):
                vec = vectors.get(key)
                if vec is not None:
                    found.append(vec)
                    break
        if found:
            encoded.append(np.mean(np.stack(found), axis=0))
        else:
            encoded.append(zero.copy())
    return np.vstack(encoded)


def _bert_vectors(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModel.from_pretrained(model_id, local_files_only=True)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            output = model(**inputs)
            token_embeddings = output.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            emb = summed / counts
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            batches.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(batches)


def _load_pool(path: Path | None) -> tuple[list[str], str]:
    if path is None:
        return _default_keyword_pool(), "keyword_eval.DEFAULT_KEYWORD_POOL"
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    raw: list[Any]
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("keywords") or data.get("pool") or data.get("items") or []
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
        raise ValueError(f"No keywords found in pool file: {path}")
    return pool, str(path)


def _item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("item_id") or item.get("video_id") or "").strip()


def _title_summary(title: str, summary: str) -> str:
    title = (title or "").strip()
    summary = (summary or "").strip()
    if title and summary:
        return f"{title}\n\n{summary}"
    return title or summary


def _vlm_order(item: dict[str, Any]) -> list[str]:
    for key in ("reference_keywords", "keywords_llm", "keywords", "gold_keywords"):
        raw = item.get(key)
        if isinstance(raw, list) and raw:
            return _dedupe([str(x) for x in raw])
    return []


def _score_map(ranked: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in ranked or []:
        kw = str(row.get("keyword") or "").strip()
        if not kw:
            continue
        try:
            out[kw] = float(row.get("score"))
        except Exception:
            out[kw] = 0.0
    return out


def _minmax(scores: dict[str, float], pool: list[str]) -> dict[str, float]:
    values = [float(scores.get(kw, 0.0)) for kw in pool]
    if not values:
        return {}
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-12:
        return {kw: 0.0 for kw in pool}
    return {kw: (float(scores.get(kw, 0.0)) - lo) / span for kw in pool}


def _fuse_scores(
    primary: dict[str, float],
    summary: dict[str, float],
    pool: list[str],
    *,
    primary_weight: float,
    summary_weight: float,
) -> dict[str, float]:
    p = _minmax(primary, pool)
    s = _minmax(summary, pool)
    return {
        kw: primary_weight * p.get(kw, 0.0) + summary_weight * s.get(kw, 0.0)
        for kw in pool
    }


class Ranker:
    def __init__(self, *, model: str, metric: str, preload_texts: list[str] | None = None) -> None:
        self.model = model
        self.metric = metric
        self._pool_cache: dict[tuple[str, ...], Any] = {}
        self._query_cache: dict[str, Any] = {}
        self._vector_cache: dict[str, np.ndarray] = {}
        if preload_texts:
            self._preload(preload_texts)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.model == "bert":
            return _bert_vectors(texts)
        if self.model == "word2vec":
            return _static_vectors("word2vec", texts)
        if self.model == "glove":
            return _static_vectors("glove", texts)
        if self.model == "word2vec_glove":
            w = _static_vectors("word2vec", texts)
            g = _static_vectors("glove", texts)
            return np.concatenate([w, g], axis=1)
        raise ValueError(f"Unknown model: {self.model}")

    def _preload(self, texts: list[str]) -> None:
        unique = _dedupe([str(t or "") for t in texts if str(t or "").strip()])
        if not unique:
            return
        _log.info("Encoding %d unique text(s) for %s", len(unique), self.model)
        vectors = self._encode_texts(unique)
        for text, vec in zip(unique, vectors):
            self._vector_cache[text] = vec

    def _pool_vectors(self, pool: list[str]):
        key = tuple(pool)
        if key not in self._pool_cache:
            missing = [text for text in pool if text not in self._vector_cache]
            if missing:
                vectors = self._encode_texts(missing)
                for text, vec in zip(missing, vectors):
                    self._vector_cache[text] = vec
            self._pool_cache[key] = np.vstack([self._vector_cache[text] for text in pool])
        return self._pool_cache[key]

    def _query_vector(self, query: str):
        key = query or ""
        if key not in self._query_cache:
            if key not in self._vector_cache:
                self._vector_cache[key] = self._encode_texts([key])[0]
            self._query_cache[key] = self._vector_cache[key]
        return self._query_cache[key]

    def rank(self, query: str, pool: list[str]) -> list[dict[str, Any]]:
        query = str(query or "").strip()
        pool = [str(kw).strip() for kw in pool if str(kw).strip()]
        if not query or not pool:
            return []
        qv = self._query_vector(query)
        pool_vecs = self._pool_vectors(pool)
        ranked = []
        for kw, kv in zip(pool, pool_vecs):
            score = float(_metric_score(qv, kv, self.metric))
            ranked.append({"keyword": kw, "score": round(score, 6)})
        ranked.sort(key=lambda row: (-float(row["score"]), str(row["keyword"]).lower()))
        return ranked


def _summary_shortlist(
    item: dict[str, Any],
    pool: list[str],
    ranker: Ranker,
    *,
    size: int,
) -> tuple[list[str], str]:
    query = _title_summary(str(item.get("title") or ""), str(item.get("summary") or ""))
    if not query:
        return pool, "full_pool"
    ranked = ranker.rank(query, pool)

    original = {kw.lower(): kw for kw in pool}
    selected: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        canonical = original.get(str(name or "").strip().lower())
        if canonical and canonical not in seen:
            seen.add(canonical)
            selected.append(canonical)

    for row in ranked[:size]:
        add(str(row.get("keyword") or ""))
    for kw in keywords_for_taxonomy(
        item.get("topic"),
        item.get("subtopic"),
        include_topic=True,
    ):
        add(kw)
    for kw in _vlm_order(item):
        add(kw)

    if len(selected) >= 5:
        return selected, f"summary_shortlist:{len(selected)}"
    return pool, "full_pool"


def _semantic_scores_for_mode(
    item: dict[str, Any],
    pool: list[str],
    ranker: Ranker,
    *,
    mode: str,
    shortlist_size: int,
    primary_weight: float,
    summary_weight: float,
) -> tuple[dict[str, float], list[str], str, str]:
    title = str(item.get("title") or "").strip()
    summary = str(item.get("summary") or "").strip()
    full_query = _title_summary(title, summary)
    candidate_pool = pool
    pool_scope = "full_pool"

    if mode in ("summary_shortlist", "summary_shortlist_fusion"):
        candidate_pool, pool_scope = _summary_shortlist(
            item, pool, ranker, size=shortlist_size
        )

    if mode == "scene_only":
        query = title or full_query
        ranked = ranker.rank(query, candidate_pool)
        return _score_map(ranked), candidate_pool, pool_scope, "title_only"

    if mode == "summary_late_fusion":
        primary_ranked = ranker.rank(title or full_query, candidate_pool)
        summary_ranked = ranker.rank(summary or full_query, candidate_pool)
        return (
            _fuse_scores(
                _score_map(primary_ranked),
                _score_map(summary_ranked),
                candidate_pool,
                primary_weight=primary_weight,
                summary_weight=summary_weight,
            ),
            candidate_pool,
            pool_scope,
            "title+summary_late_fusion",
        )

    if mode == "summary_shortlist_fusion":
        primary_ranked = ranker.rank(title or full_query, candidate_pool)
        summary_ranked = ranker.rank(summary or full_query, candidate_pool)
        return (
            _fuse_scores(
                _score_map(primary_ranked),
                _score_map(summary_ranked),
                candidate_pool,
                primary_weight=primary_weight,
                summary_weight=summary_weight,
            ),
            candidate_pool,
            pool_scope,
            "summary_shortlist+late_fusion",
        )

    query_source = "title+summary"
    if mode == "summary_shortlist":
        query_source = "title+summary_in_summary_shortlist"
    ranked = ranker.rank(full_query, candidate_pool)
    return _score_map(ranked), candidate_pool, pool_scope, query_source


def _score_graded(predicted: list[str], item: dict[str, Any], *, top_k: int) -> dict[str, Any]:
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


def _top_scores(ranking: list[dict[str, Any]], n: int = 5) -> str:
    bits = []
    for idx, row in enumerate(ranking[:n], start=1):
        bits.append(f"{idx}. {row['keyword']} ({float(row['score']):.3f})")
    return "; ".join(bits)


def evaluate(
    *,
    graded_path: Path,
    pool_path: Path | None,
    models: list[str],
    modes: list[str],
    metric: str,
    top_k: int,
    include_ranking: int,
    shortlist_size: int,
    primary_weight: float,
    summary_weight: float,
    rerank_weights: tuple[float, float],
) -> dict[str, Any]:
    graded = _load_json(graded_path)
    items = [row for row in graded.get("items") or [] if isinstance(row, dict)]
    pool, pool_source = _load_pool(pool_path)
    rcfg = RerankConfig(w_cos=rerank_weights[0], w_llm=rerank_weights[1], w_rule=0.0)

    combos: list[dict[str, Any]] = []
    preload_texts = list(pool)
    for item in items:
        title = str(item.get("title") or "")
        summary = str(item.get("summary") or "")
        preload_texts.extend([
            title,
            summary,
            _title_summary(title, summary),
        ])
    for model in models:
        ranker = Ranker(model=model, metric=metric, preload_texts=preload_texts)
        for mode in modes:
            rows: list[dict[str, Any]] = []
            metrics = []
            graded_metrics = []
            for item in items:
                vid = _item_id(item)
                scores, candidate_pool, pool_scope, query_source = _semantic_scores_for_mode(
                    item,
                    pool,
                    ranker,
                    mode=mode,
                    shortlist_size=shortlist_size,
                    primary_weight=primary_weight,
                    summary_weight=summary_weight,
                )
                hybrid = rerank_pool(
                    cosine_scores=scores,
                    llm_ranking=_vlm_order(item),
                    cfg=rcfg,
                )
                top = [str(row["keyword"]) for row in hybrid[:top_k]]
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
                graded_score = _score_graded(top, item, top_k=top_k)
                if graded_score:
                    graded_metrics.append(graded_score)
                rows.append({
                    **m.to_dict(),
                    "model": model,
                    "mode": mode,
                    "query_source": query_source,
                    "pool_scope": pool_scope,
                    "candidate_pool_size": len(candidate_pool),
                    "title": item.get("title") or "",
                    "url": item.get("url") or "",
                    "topic": item.get("topic") or "",
                    "subtopic": item.get("subtopic") or "",
                    "gold_keywords": list(item.get("gold_keywords") or []),
                    "vlm_order": _vlm_order(item),
                    "top_keywords": top,
                    "ranking": hybrid[:include_ranking],
                    "graded_metrics": graded_score,
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
            combos.append({
                "model": model,
                "mode": mode,
                "metric": metric,
                "aggregate": agg,
                "items": rows,
            })

    return {
        "schema_version": "summary-context-ablation-v1",
        "graded_path": str(graded_path),
        "pool_source": pool_source,
        "models": models,
        "modes": modes,
        "metric": metric,
        "top_k": top_k,
        "shortlist_size": shortlist_size,
        "summary_fusion_weights": {
            "primary": primary_weight,
            "summary": summary_weight,
        },
        "rerank_weights": {
            "embedding": rerank_weights[0],
            "vlm_rank": rerank_weights[1],
        },
        "combos": combos,
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "mode", "precision_at_k", "recall_at_k", "f1_at_k",
        "jaccard_at_k", "ndcg_at_5_mean", "mean_relevance_at_5",
        "weighted_precision_at_5", "mrr_relevance_ge_2",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for combo in report["combos"]:
            agg = combo.get("aggregate") or {}
            writer.writerow({
                "model": combo.get("model"),
                "mode": combo.get("mode"),
                "precision_at_k": agg.get("precision_at_5_mean"),
                "recall_at_k": agg.get("recall_at_5_mean"),
                "f1_at_k": agg.get("f1_at_5_mean"),
                "jaccard_at_k": agg.get("jaccard_at_5_mean"),
                "ndcg_at_5_mean": agg.get("ndcg_at_5_mean"),
                "mean_relevance_at_5": agg.get("mean_relevance_at_5"),
                "weighted_precision_at_5": agg.get("weighted_precision_at_5"),
                "mrr_relevance_ge_2": agg.get("mrr_relevance_ge_2"),
            })


def write_long_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "mode", "video_id", "title", "topic", "subtopic",
        "query_source", "pool_scope", "candidate_pool_size", "top5",
        "top5_with_scores", "gold_keywords", "vlm_order", "ndcg_at_5",
        "weighted_precision_at_5",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for combo in report["combos"]:
            for item in combo.get("items") or []:
                graded = item.get("graded_metrics") or {}
                writer.writerow({
                    "model": combo.get("model"),
                    "mode": combo.get("mode"),
                    "video_id": item.get("item_id"),
                    "title": item.get("title"),
                    "topic": item.get("topic"),
                    "subtopic": item.get("subtopic"),
                    "query_source": item.get("query_source"),
                    "pool_scope": item.get("pool_scope"),
                    "candidate_pool_size": item.get("candidate_pool_size"),
                    "top5": ", ".join(item.get("top_keywords") or []),
                    "top5_with_scores": _top_scores(item.get("ranking") or []),
                    "gold_keywords": ", ".join(item.get("gold_keywords") or []),
                    "vlm_order": ", ".join(item.get("vlm_order") or []),
                    "ndcg_at_5": graded.get("ndcg_at_5"),
                    "weighted_precision_at_5": graded.get("weighted_precision_at_5"),
                })


def write_md(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Summary Context Ablation",
        "",
        "Same 15 videos, same closed keyword pool, same fixed VLM title/summary.",
        "Scores below use the UI-style hybrid ranker: embedding score + VLM keyword rank.",
        "",
        "| Model | Mode | P@5 | R@5 | F1@5 | Jaccard | nDCG@5 | Weighted P@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for combo in report["combos"]:
        agg = combo.get("aggregate") or {}
        lines.append(
            f"| {combo.get('model')} | {combo.get('mode')} | "
            f"{agg.get('precision_at_5_mean', 0):.4f} | {agg.get('recall_at_5_mean', 0):.4f} | "
            f"{agg.get('f1_at_5_mean', 0):.4f} | {agg.get('jaccard_at_5_mean', 0):.4f} | "
            f"{agg.get('ndcg_at_5_mean', 0):.4f} | {agg.get('weighted_precision_at_5', 0):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graded", type=Path, default=DEFAULT_GRADED)
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER), choices=list(MODEL_ORDER))
    parser.add_argument("--modes", nargs="+", default=list(MODE_ORDER), choices=list(MODE_ORDER))
    parser.add_argument("--metric", default="cosine", choices=["cosine", "manhattan", "euclidean"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-ranking", type=int, default=10)
    parser.add_argument("--shortlist-size", type=int, default=25)
    parser.add_argument("--summary-weights", default="0.75,0.25",
                        help="primary,summary weights for late-fusion modes.")
    parser.add_argument("--rerank-weights", default="0.45,0.55",
                        help="embedding,vlm-rank weights for final hybrid ranking.")
    args = parser.parse_args()

    def parse_pair(raw: str, default: tuple[float, float]) -> tuple[float, float]:
        try:
            a, b = [float(x.strip()) for x in raw.split(",", 1)]
            if a >= 0 and b >= 0 and a + b > 0:
                total = a + b
                return a / total, b / total
        except Exception:
            pass
        return default

    primary_weight, summary_weight = parse_pair(args.summary_weights, (0.75, 0.25))
    rerank_weights = parse_pair(args.rerank_weights, (0.45, 0.55))
    out_prefix = args.out_prefix.resolve()

    report = evaluate(
        graded_path=args.graded.resolve(),
        pool_path=args.pool.resolve() if args.pool else None,
        models=list(args.models),
        modes=list(args.modes),
        metric=args.metric,
        top_k=max(1, args.top_k),
        include_ranking=max(args.top_k, args.include_ranking),
        shortlist_size=max(5, args.shortlist_size),
        primary_weight=primary_weight,
        summary_weight=summary_weight,
        rerank_weights=rerank_weights,
    )

    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    long_csv_path = out_prefix.with_name(out_prefix.name + "_long.csv")
    md_path = out_prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(report, csv_path)
    write_long_csv(report, long_csv_path)
    write_md(report, md_path)

    print(f"[OK] JSON: {json_path}")
    print(f"[OK] CSV : {csv_path}")
    print(f"[OK] LONG: {long_csv_path}")
    print(f"[OK] MD  : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
