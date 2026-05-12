"""
Multi-model, multi-metric semantic similarity comparison module.

Implements three embedding methods and two distance metrics as requested
by the supervisor (Beatriz, 2026-04):

Embedding models:
  - BERT (sentence-transformers/all-MiniLM-L6-v2) — default, already deployed
  - Word2Vec (Google News 300d, via gensim)
  - GloVe (glove-wiki-gigaword-300, via gensim)

Distance metrics:
  - Cosine Similarity: cos(θ) = ⟨v₁, v₂⟩ / (‖v₁‖ · ‖v₂‖)  → higher = more similar
  - Euclidean Distance: d = √Σ(xᵢ - yᵢ)² → LOWER = more similar (inverted for ranking)

Each combination produces a ranked list of pool keywords for a given query.
The module is designed for evaluation scripts (tools/eval_videos_multi.py)
and does not modify the live pipeline.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

# Model cache to avoid reloading
_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

def _get_bert():
    """Sentence-transformers BERT (all-MiniLM-L6-v2)."""
    if "bert" not in _CACHE:
        from sentence_transformers import SentenceTransformer
        _CACHE["bert"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _CACHE["bert"]


def _get_word2vec():
    """Google News Word2Vec (300d) via gensim downloader."""
    if "word2vec" not in _CACHE:
        import gensim.downloader as api
        _log.info("  [SIM] Loading Word2Vec (google-news-300)... this may take a minute")
        _CACHE["word2vec"] = api.load("word2vec-google-news-300")
    return _CACHE["word2vec"]


def _get_glove():
    """GloVe Wikipedia+Gigaword (300d) via gensim downloader."""
    if "glove" not in _CACHE:
        import gensim.downloader as api
        _log.info("  [SIM] Loading GloVe (glove-wiki-gigaword-300)... this may take a minute")
        _CACHE["glove"] = api.load("glove-wiki-gigaword-300")
    return _CACHE["glove"]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_bert(texts: list[str]) -> np.ndarray:
    model = _get_bert()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _encode_word2vec(texts: list[str]) -> np.ndarray:
    """Average word vectors for each text (standard approach for sentence-level Word2Vec)."""
    wv = _get_word2vec()
    dim = wv.vector_size
    out = []
    for text in texts:
        tokens = text.lower().replace("-", " ").split()
        vecs = [wv[t] for t in tokens if t in wv]
        if vecs:
            out.append(np.mean(vecs, axis=0))
        else:
            out.append(np.zeros(dim))
    return np.array(out, dtype=np.float32)


def _encode_glove(texts: list[str]) -> np.ndarray:
    """Average word vectors for each text using GloVe."""
    wv = _get_glove()
    dim = wv.vector_size
    out = []
    for text in texts:
        tokens = text.lower().replace("-", " ").split()
        vecs = [wv[t] for t in tokens if t in wv]
        if vecs:
            out.append(np.mean(vecs, axis=0))
        else:
            out.append(np.zeros(dim))
    return np.array(out, dtype=np.float32)


ENCODERS = {
    "bert": _encode_bert,
    "word2vec": _encode_word2vec,
    "glove": _encode_glove,
}


# ---------------------------------------------------------------------------
# BM25 baseline (lexical, no embeddings)
# ---------------------------------------------------------------------------

def rank_pool_bm25(query: str, pool: list[str]) -> list[dict]:
    """Rank pool keywords against query using BM25 (Okapi).

    BM25 is a term-frequency baseline (Robertson et al., 1995): it scores
    each keyword by how many of its tokens appear in the query, weighted by
    inverse document frequency. No semantic understanding — pure lexical
    overlap. Used here as the mandatory academic baseline that every
    embedding paper compares against (BEIR, Thakur et al. 2021).
    """
    from rank_bm25 import BM25Okapi

    tokenized_pool = [kw.lower().split() for kw in pool]
    bm25 = BM25Okapi(tokenized_pool)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    results = []
    for i, kw in enumerate(pool):
        results.append({"keyword": kw, "score": round(float(scores[i]), 6)})
    results.sort(key=lambda d: (-d["score"], d["keyword"].lower()))
    return results


# ---------------------------------------------------------------------------
# Distance / similarity metrics
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Higher = more similar. Range [-1, 1]."""
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Lower = more similar. Range [0, ∞)."""
    return float(np.linalg.norm(a - b))


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Convert euclidean distance to a similarity score in [0, 1].

    Uses 1 / (1 + d) so higher = more similar, matching cosine's direction.
    """
    d = euclidean_distance(a, b)
    return 1.0 / (1.0 + d)


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 norm. Lower = more similar."""
    return float(np.sum(np.abs(a - b)))


def manhattan_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Convert Manhattan distance to similarity in [0, 1]. 1/(1+d)."""
    return 1.0 / (1.0 + manhattan_distance(a, b))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Raw inner product. Higher = more similar.

    When vectors are L2-normalized (as in BERT), dot product = cosine.
    For non-normalized vectors (Word2Vec, GloVe) this captures magnitude.
    """
    return float(np.dot(a, b))


def soft_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Approximated soft cosine using element-wise minimum overlap.

    True soft cosine requires a word-word similarity matrix (expensive for
    dense vectors). This approximation measures the proportion of shared
    "direction" by computing min(|aᵢ|, |bᵢ|) overlap normalized by norms.
    For L2-normalized BERT vectors this closely tracks standard cosine;
    for sparse/unnormalized vectors it gives distinct rankings.
    """
    overlap = float(np.sum(np.minimum(np.abs(a), np.abs(b))))
    na = float(np.linalg.norm(a, 1))
    nb = float(np.linalg.norm(b, 1))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return overlap / ((na * nb) ** 0.5)


METRICS = {
    "cosine": cosine_similarity,
    "euclidean": euclidean_similarity,
    "manhattan": manhattan_similarity,
    "dot_product": dot_product,
    "soft_cosine": soft_cosine_similarity,
}


# ---------------------------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------------------------

def rank_pool(
    query: str,
    pool: list[str],
    *,
    model: str = "bert",
    metric: str = "cosine",
) -> list[dict]:
    """Rank every keyword in ``pool`` against ``query``.

    Returns a list of ``{keyword, score, raw_distance (for euclidean)}``
    sorted by score descending (highest = most relevant).
    """
    if not query or not pool:
        return []

    encoder = ENCODERS.get(model)
    if encoder is None:
        raise ValueError(f"Unknown model: {model}. Choose from: {list(ENCODERS)}")
    metric_fn = METRICS.get(metric)
    if metric_fn is None:
        raise ValueError(f"Unknown metric: {metric}. Choose from: {list(METRICS)}")

    all_texts = [query] + list(pool)
    embeddings = encoder(all_texts)
    q_vec = embeddings[0]
    pool_vecs = embeddings[1:]

    results: list[dict] = []
    for i, kw in enumerate(pool):
        score = metric_fn(q_vec, pool_vecs[i])
        entry: dict = {"keyword": kw, "score": round(float(score), 6)}
        if metric == "euclidean":
            entry["raw_distance"] = round(euclidean_distance(q_vec, pool_vecs[i]), 6)
        results.append(entry)

    results.sort(key=lambda d: (-d["score"], d["keyword"].lower()))
    return results


def top_n(query: str, pool: list[str], *, model: str = "bert",
          metric: str = "cosine", n: int = 5) -> list[str]:
    """Convenience: return just the top-N keyword names."""
    ranked = rank_pool(query, pool, model=model, metric=metric)
    return [r["keyword"] for r in ranked[:n]]


def top_n_mmr(query: str, pool: list[str], *, model: str = "bert",
              n: int = 5, diversity: float = 0.3) -> list[str]:
    """Select top-N keywords using Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

    MMR balances relevance to the query with diversity among selected keywords:

        MMR(k) = (1 - λ) · sim(query, k) - λ · max_{j ∈ S} sim(k, j)

    where λ = ``diversity`` (0 = pure relevance, 1 = pure diversity) and
    S is the set of already-selected keywords.

    This prevents selecting semantically redundant keywords (e.g. both
    "Substitution" and "Trigonometric substitution") when a more informative
    alternative exists.
    """
    if not query or not pool:
        return []

    encoder = ENCODERS.get(model)
    if encoder is None:
        return top_n(query, pool, model=model, n=n)

    all_texts = [query] + list(pool)
    embeddings = encoder(all_texts)
    q_vec = embeddings[0]
    pool_vecs = embeddings[1:]

    # Compute query-keyword similarities
    q_sims = np.array([cosine_similarity(q_vec, pv) for pv in pool_vecs])

    # Compute pairwise keyword-keyword similarities
    n_pool = len(pool)
    kw_sims = np.zeros((n_pool, n_pool))
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            s = cosine_similarity(pool_vecs[i], pool_vecs[j])
            kw_sims[i, j] = s
            kw_sims[j, i] = s

    # Greedy MMR selection
    selected_idx: list[int] = []
    remaining = set(range(n_pool))

    for _ in range(min(n, n_pool)):
        best_idx = -1
        best_score = -float("inf")
        for idx in remaining:
            relevance = float(q_sims[idx])
            if selected_idx:
                max_sim_to_selected = max(float(kw_sims[idx, j]) for j in selected_idx)
            else:
                max_sim_to_selected = 0.0
            mmr_score = (1 - diversity) * relevance - diversity * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx < 0:
            break
        selected_idx.append(best_idx)
        remaining.discard(best_idx)

    return [pool[i] for i in selected_idx]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def top_n_hybrid_manhattan(
    query: str, pool: list[str], *,
    llm_ranking: list[str] | None = None,
    rule_hits: dict[str, int] | None = None,
    n: int = 5,
    w_man: float = 0.6, w_llm: float = 0.3, w_rule: float = 0.1,
) -> list[dict]:
    """Hybrid re-ranker using Manhattan similarity as the primary signal.

    Same formula as the cosine hybrid but replaces cosine with Manhattan:
        r(k) = w_man · manhattan_sim(q,k) + w_llm · RRF(rank) + w_rule · hits/max

    Returns full ranking with scores and components.
    """
    if not query or not pool:
        return []

    embeddings = _encode_bert([query] + list(pool))
    q_vec = embeddings[0]
    pool_vecs = embeddings[1:]

    man_scores = {}
    for i, kw in enumerate(pool):
        man_scores[kw] = manhattan_similarity(q_vec, pool_vecs[i])

    from reranker import rerank_pool as _rr, RerankConfig
    cfg = RerankConfig(w_cos=w_man, w_llm=w_llm, w_rule=w_rule)
    return _rr(
        cosine_scores=man_scores,
        llm_ranking=llm_ranking or [],
        rule_hits=rule_hits or {},
        cfg=cfg,
    )


def _self_test() -> None:
    pool = [
        "Trigonometric substitution",
        "Partial fractions decomposition",
        "Fundamental theorem of Calculus",
        "Volume of revolution",
        "Direct integrals",
    ]
    query = "Evaluate the definite integral using trigonometric substitution"

    print("=== similarity_compare self-test ===\n")
    for metric_name in METRICS:
        ranked = rank_pool(query, pool, model="bert", metric=metric_name)
        top3 = [f"{r['keyword']}({r['score']:.3f})" for r in ranked[:3]]
        print(f"  bert × {metric_name:12} → {', '.join(top3)}")

    print("\nself-test OK")


if __name__ == "__main__":
    _self_test()
