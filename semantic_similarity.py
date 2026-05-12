"""
Embedding-based keyword selection for methodological grounding.

This module ranks a closed keyword pool against a query text using cosine
similarity in a sentence-embedding space. It is designed as a lightweight
post-processing layer for video tagging: Gemini still produces the summary /
OCR text; this module provides transparent similarity scores and deterministic
top-k selection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from config import VIDEO_CACHE_DIR, ensure_dirs

_log = logging.getLogger(__name__)

# One SentenceTransformer per model id per process (Deep mode calls rank_pool many times).
_MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class SimilarityConfig:
    model_id: str
    normalize: bool = True
    top_k: int = 5
    metric: str = "manhattan"  # "cosine" | "manhattan" | "euclidean"


def default_similarity_config() -> SimilarityConfig:
    model_id = os.getenv(
        "STEP_SIMILARITY_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ).strip()
    metric = os.getenv("STEP_SIMILARITY_METRIC", "manhattan").strip().lower()
    if metric not in ("cosine", "manhattan", "euclidean"):
        metric = "manhattan"
    try:
        top_k = int(os.getenv("STEP_SIMILARITY_TOPK", "5"))
    except ValueError:
        top_k = 5
    return SimilarityConfig(model_id=model_id, metric=metric, top_k=max(1, min(top_k, 10)))


def similarity_enabled() -> bool:
    return os.getenv("STEP_SIMILARITY_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _video_cache_disabled() -> bool:
    return os.getenv("STEP_DISABLE_VIDEO_CACHE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def normalize_embedding_query(text: str, *, max_chars: int = 2400) -> str:
    """Strip display-math noise and prose wrappers so embeddings stay stable.

    Frame OCR often returns ``$$``, stray backticks, or short English fragments
    mixed with LaTeX; those tokens skew cosine scores if left untouched.
    """
    s = (text or "").strip()
    if not s:
        return ""
    s = s.replace("`", " ")
    s = s.replace("$$", " ")
    s = s.replace("$", " ")
    s = s.replace("\\displaystyle", " ")
    s = s.replace("\\text", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars]
    return s


def pool_hash(pool: Iterable[str]) -> str:
    items = [str(p).strip() for p in (pool or []) if str(p).strip()]
    blob = "\n".join(items).lower().encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def query_hash(text: str) -> str:
    blob = (text or "").strip().encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _safe_name(s: str) -> str:
    out = []
    for ch in (s or ""):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "model"


def _emb_dir() -> Path:
    ensure_dirs()
    d = VIDEO_CACHE_DIR / "embeddings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _short_h(h: str, n: int = 16) -> str:
    """Trim ``"sha256:abc..."`` to a filename-safe prefix.

    Full 64-hex hashes pushed cache filenames over Windows MAX_PATH=260 in
    deeply-nested projects; 16 hex (64 bits) is more than enough to keep the
    pool/query cache collision-free for our scale.
    """
    s = (h or "").split(":", 1)[-1]
    return s[: max(1, n)]


def _pool_cache_path(cfg: SimilarityConfig, pool_h: str) -> Path:
    name = f"pool_{_safe_name(cfg.model_id)}_{_short_h(pool_h)}.json"
    return _emb_dir() / name


def _query_cache_path(
    cfg: SimilarityConfig, pool_h: str, qh: str
) -> Path:
    name = (
        f"q_{_safe_name(cfg.model_id)}_"
        f"{_short_h(pool_h)}_{_short_h(qh)}.json"
    )
    return _emb_dir() / name


def _load_pool_vectors(cfg: SimilarityConfig, pool_h: str) -> dict | None:
    if _video_cache_disabled():
        return None
    fp = _pool_cache_path(cfg, pool_h)
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_pool_vectors(cfg: SimilarityConfig, pool_h: str, data: dict) -> None:
    if _video_cache_disabled():
        return
    fp = _pool_cache_path(cfg, pool_h)
    try:
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        _log.info(f"  [SIM] [WARN] Pool cache write failed: {e}")


def _load_query_vector(
    cfg: SimilarityConfig, pool_h: str, qh: str
) -> list[float] | None:
    if _video_cache_disabled():
        return None
    fp = _query_cache_path(cfg, pool_h, qh)
    if not fp.exists():
        return None
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        v = data.get("vector")
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            return [float(x) for x in v]
    except Exception:
        return None
    return None


def _save_query_vector(
    cfg: SimilarityConfig, pool_h: str, qh: str, vector: list[float]
) -> None:
    if _video_cache_disabled():
        return
    fp = _query_cache_path(cfg, pool_h, qh)
    try:
        fp.write_text(
            json.dumps({"vector": vector}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        _log.info(f"  [SIM] [WARN] Query cache write failed: {e}")


def _get_embedder(cfg: SimilarityConfig):
    """
    Lazy import and reuse one model instance per model id.
    """
    if cfg.model_id in _MODEL_CACHE:
        return _MODEL_CACHE[cfg.model_id]
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(cfg.model_id)
    _MODEL_CACHE[cfg.model_id] = model
    return model


def _cosine(u: list[float], v: list[float]) -> float:
    du = 0.0
    dv = 0.0
    dot = 0.0
    for a, b in zip(u, v):
        dot += float(a) * float(b)
        du += float(a) * float(a)
        dv += float(b) * float(b)
    if du <= 0.0 or dv <= 0.0:
        return 0.0
    return dot / ((du**0.5) * (dv**0.5))


def rank_pool(query_text: str, pool: list[str], *, cfg: SimilarityConfig) -> list[dict]:
    q = normalize_embedding_query(query_text or "")
    pool = [p.strip() for p in (pool or []) if p and p.strip()]
    if not q or not pool:
        return []

    pool_h = pool_hash(pool)
    qh = query_hash(q)

    cached_qv = _load_query_vector(cfg, pool_h, qh)
    if cached_qv:
        qv = cached_qv
    else:
        model = _get_embedder(cfg)
        qv = model.encode([q], normalize_embeddings=cfg.normalize).tolist()[0]
        _save_query_vector(cfg, pool_h, qh, qv)

    cached = _load_pool_vectors(cfg, pool_h)
    if (
        cached
        and cached.get("pool") == pool
        and isinstance(cached.get("vectors"), list)
        and len(cached["vectors"]) == len(pool)
    ):
        vectors = cached["vectors"]
    else:
        model = _get_embedder(cfg)
        vectors = model.encode(pool, normalize_embeddings=cfg.normalize).tolist()
        _save_pool_vectors(
            cfg,
            pool_h,
            {"pool": pool, "vectors": vectors, "dim": len(vectors[0]) if vectors else 0},
        )

    scored: list[dict] = []
    for kw, kv in zip(pool, vectors):
        if cfg.metric == "manhattan":
            d = sum(abs(float(a) - float(b)) for a, b in zip(qv, kv))
            s = 1.0 / (1.0 + d)
        elif cfg.metric == "euclidean":
            d = sum((float(a) - float(b)) ** 2 for a, b in zip(qv, kv)) ** 0.5
            s = 1.0 / (1.0 + d)
        else:  # cosine (default for normalized vectors)
            if cfg.normalize:
                s = float(sum(float(a) * float(b) for a, b in zip(qv, kv)))
            else:
                s = float(_cosine(qv, kv))
        scored.append({"keyword": kw, "score": round(s, 6)})

    scored.sort(key=lambda d: (-d["score"], d["keyword"].lower()))
    return scored


def top_k_keywords(
    query_text: str, pool: list[str], *, cfg: SimilarityConfig, query_source: str
) -> dict:
    """Return cosine scores for every keyword in ``pool``, sorted by relevance.

    ``ranked`` is the full pool (so callers can look up scores for any keyword
    Gemini happens to pick, not just the cosine top-K). Consumers that only
    need the leaderboard can take ``ranked[:top_k]``; the UI's "Most relevant
    keywords" card uses exactly that slice.
    """
    qn = normalize_embedding_query(query_text or "")
    ranked = rank_pool(qn, pool, cfg=cfg)
    pool_h = pool_hash(pool)
    return {
        "method": f"embedding_{cfg.metric}",
        "metric": cfg.metric,
        "model_id": cfg.model_id,
        "query_source": query_source,
        "pool_hash": pool_h,
        "query_hash": query_hash(qn),
        "top_k": cfg.top_k,
        "ranked": ranked,
    }
