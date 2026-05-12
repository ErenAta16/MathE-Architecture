"""
Hybrid keyword re-ranker for the STEP pipeline.

Two independent signals are produced for every candidate keyword in a
closed pool:

* **cosine** -- cosine similarity between the (normalized) problem text and
  the keyword in a sentence-embedding space (Layer 7 / per-scene
  similarity output, ``semantic_similarity.rank_pool``).
* **llm rank** -- the position the language model gave to the keyword in
  its own ordered Task 2 reply (``keyword_eval.evaluate_keywords``). Index
  1 is the model's most relevant pick. Keywords the model did not include
  receive rank ``+inf``.
The combined score is the weighted sum of these two signals after each
has been normalized to ``[0, 1]``::

    r(k) =  w1 * cos(q, k)
          + w2 * rrf(rank_LLM(k))    / Z_rank

The reciprocal-rank-fusion term ``rrf(r) = 1 / (r + 60)`` handles missing
keywords gracefully (returns 0 when ``r = +inf``) and is widely used in the
IR literature for combining heterogeneous rankers (Cormack et al., 2009).

The module is purely numeric: no LLM calls, no I/O, no embedding model
loaded here. ``run.py`` and ``run_video.py`` call ``rerank_pool`` after
they have collected the available ranking signals.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Mapping


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RerankConfig:
    """Default weights ``(w_cos, w_llm, w_rule) = (0.45, 0.55, 0.0)``.

    The production video path uses the grid-searched 45/55 split between
    embedding similarity and the LLM/VLM rank signal. ``w_rule`` is kept as a
    backward-compatible field, but the production video path no longer supplies
    rule-hit scores.

    The VLM (Gemini) is used elsewhere for video understanding (title +
    summary extraction) — this is the irreplaceable role of LLM/VLM in
    the pipeline. The selection mechanism, however, is grounded in
    classical similarity computation rather than LLM judgment.

    Weights are configurable via STEP_RERANK_WEIGHTS env var. Prefer a
    comma-separated pair (``w_cos,w_llm``); a legacy triplet is still accepted.
    """

    w_cos: float = 0.45
    w_llm: float = 0.55
    w_rule: float = 0.0
    rrf_k: int = 60  # reciprocal-rank-fusion smoothing constant


def default_rerank_config() -> RerankConfig:
    raw = (os.getenv("STEP_RERANK_WEIGHTS") or "").strip()
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        try:
            if len(parts) == 2:
                wc, wl = (float(parts[0]), float(parts[1]))
                if wc + wl > 0 and min(wc, wl) >= 0:
                    return RerankConfig(w_cos=wc, w_llm=wl, w_rule=0.0)
            if len(parts) >= 3:
                wc, wl, wr = (float(parts[0]), float(parts[1]), float(parts[2]))
                # Guard against negative or all-zero weights.
                if wc + wl + wr > 0 and min(wc, wl, wr) >= 0:
                    return RerankConfig(w_cos=wc, w_llm=wl, w_rule=wr)
        except ValueError:
            pass
    return RerankConfig()


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def _clip01(x: float) -> float:
    if x is None or not isinstance(x, (int, float)) or math.isnan(float(x)):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _rrf(rank: float, k: int) -> float:
    """Reciprocal-rank-fusion contribution. ``rank=+inf`` returns 0."""
    if rank is None or not math.isfinite(rank):
        return 0.0
    return 1.0 / (float(rank) + float(k))


def _llm_rank_map(llm_ranking: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, kw in enumerate(llm_ranking or []):
        if not kw:
            continue
        key = str(kw).strip().lower()
        if key and key not in out:
            out[key] = i + 1  # 1-indexed
    return out


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def _minmax_normalize(values: dict[str, float]) -> dict[str, float]:
    """Scale every value into [0, 1] using min-max normalisation across the pool.

    This step is critical for the weighted sum to mean what it says: without
    it, a metric whose raw range is [0.04, 0.07] (e.g. Manhattan similarity
    on L2-normalised BERT vectors) cannot compete with a metric whose range
    is [0, 1] (e.g. RRF_LLM), regardless of the assigned weight.
    """
    if not values:
        return {}
    vs = list(values.values())
    lo = min(vs)
    hi = max(vs)
    span = hi - lo
    if span < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


def rerank_pool(
    *,
    cosine_scores: Mapping[str, float],
    llm_ranking: Iterable[str] | None = None,
    rule_hits: Mapping[str, int] | None = None,
    cfg: RerankConfig | None = None,
) -> list[dict]:
    """Return the pool sorted by hybrid score (descending).

    Each signal is min-max normalised within the pool BEFORE the weighted
    sum, so the weights w_cos and w_llm reflect the actual proportional
    influence each signal has on the final ranking. This follows the score
    fusion methodology of COS-Mix (arXiv 2024) and the USMB benchmark
    (Berkeley 2024).

    Each output entry has::

        {
          "keyword": "...",
          "score": <combined>,
          "components": {
              "embedding_raw": <raw similarity score>,
              "embedding": <min-max normalised, in [0, 1]>,
              "rrf_llm": <RRF contribution, normalised to [0, 1]>,
              "llm_rank": <int | None>,
          },
        }
    """
    cfg = cfg or default_rerank_config()
    raw = {str(k).strip(): float(v) for k, v in (cosine_scores or {}).items() if str(k).strip()}
    if not raw:
        return []

    rank_map = _llm_rank_map(llm_ranking or [])
    hits = {str(k).strip(): int(v) for k, v in (rule_hits or {}).items() if str(k).strip()}

    # Normalise each signal across the pool so the weighted sum is meaningful.
    embed_norm = _minmax_normalize(raw)

    rrf_raw = {kw: _rrf(rank_map.get(kw.lower(), float("inf")), cfg.rrf_k) for kw in raw}
    rrf_norm = _minmax_normalize(rrf_raw) if any(v > 0 for v in rrf_raw.values()) else {kw: 0.0 for kw in raw}

    max_hits = max(hits.values()) if hits else 0
    rule_norm = {kw: (hits.get(kw, 0) / max_hits) if max_hits > 0 else 0.0 for kw in raw}

    out: list[dict] = []
    for keyword in raw:
        e = embed_norm.get(keyword, 0.0)
        l = rrf_norm.get(keyword, 0.0)
        r = rule_norm.get(keyword, 0.0)
        rank = rank_map.get(keyword.lower(), float("inf"))

        combined = cfg.w_cos * e + cfg.w_llm * l + cfg.w_rule * r

        out.append({
            "keyword": keyword,
            "score": round(combined, 6),
            "components": {
                "embedding_raw": round(raw[keyword], 6),
                "embedding": round(e, 6),
                "rrf_llm": round(l, 6),
                "rule": round(r, 6),
                "llm_rank": (int(rank) if math.isfinite(rank) else None),
                "hits": hits.get(keyword, 0),
            },
        })

    # Stable ordering: by combined score desc, then alphabetic on ties.
    out.sort(key=lambda d: (-d["score"], d["keyword"].lower()))
    return out


def top_n_keywords(
    *,
    cosine_scores: Mapping[str, float],
    llm_ranking: Iterable[str] | None = None,
    rule_hits: Mapping[str, int] | None = None,
    n: int = 5,
    cfg: RerankConfig | None = None,
) -> list[str]:
    """Convenience wrapper used by run.py / run_video.py."""
    ranked = rerank_pool(
        cosine_scores=cosine_scores,
        llm_ranking=llm_ranking,
        rule_hits=rule_hits,
        cfg=cfg,
    )
    return [r["keyword"] for r in ranked[: max(0, int(n))]]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    cosine = {
        "Partial fractions decomposition": 0.486,
        "Integration by parts":             0.359,
        "Substitution":                     0.292,
        "Region decomposition":             0.284,
        "Trigonometric substitution":       0.280,
        "Area of a region between two or more curves": 0.272,
        "Direct integrals":                 0.264,
        "Fundamental theorem of Calculus":  0.260,
        "Total area":                       0.090,
    }
    llm_order = [
        "Partial fractions decomposition",
        "Direct integrals",
        "Fundamental theorem of Calculus",
    ]
    hits = {
        "Partial fractions decomposition": 2,
        "Substitution": 1,
        "Trigonometric substitution": 1,
    }

    cfg = RerankConfig(w_cos=0.6, w_llm=0.3, w_rule=0.1)
    ranked = rerank_pool(
        cosine_scores=cosine, llm_ranking=llm_order, rule_hits=hits, cfg=cfg,
    )
    top5 = [r["keyword"] for r in ranked[:5]]
    print("self-test top-5:", top5)
    print("first row components:", ranked[0]["components"])

    # With min-max normalised signals, cosine values now occupy [0, 1] within
    # the pool; the highest-cosine keyword (Partial fractions decomposition)
    # plus its LLM rank 1 + 2 rule hits should still be the clear top-1.
    assert top5[0] == "Partial fractions decomposition", top5

    # All-zero or missing signals are handled gracefully.
    empty = rerank_pool(cosine_scores={}, llm_ranking=[], rule_hits={}, cfg=cfg)
    assert empty == []
    only_cos = rerank_pool(cosine_scores={"a": 0.9, "b": 0.1}, cfg=cfg)
    assert [r["keyword"] for r in only_cos] == ["a", "b"]

    print("reranker self-test OK")


if __name__ == "__main__":
    _self_test()
