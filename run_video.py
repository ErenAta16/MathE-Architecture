"""
VideoAnalyzer — high-level entry point for single-video analysis.

Wraps ingestion + VLM extraction + taxonomy classification so the web layer
only has to call one method. Results are cached on disk under
``step_pipeline/video_cache/`` keyed by YouTube video id or file SHA-256, so
repeat analysis does not re-spend the LLM budget.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from config import VIDEO_CACHE_DIR, ensure_dirs
from layer0_video import (
    canonical_youtube_url,
    download_youtube_video,
    extract_frames,
    youtube_video_id,
)
from layer3_video_vlm import analyze_local_file, analyze_youtube
from layer3v_frames import analyze_frames_deep
from taxonomy import classify_taxonomy, refine_subtopic, topic_from_keywords
from semantic_similarity import (
    default_similarity_config,
    normalize_embedding_query,
    similarity_enabled,
    top_k_keywords,
)
from reranker import default_rerank_config, rerank_pool

_log = logging.getLogger(__name__)

# Bump whenever a schema-relevant change lands (per-scene similarity query,
# taxonomy refinement, displayed payload shape, etc.). Older cached payloads are
# silently ignored so users always see the current behaviour.
CACHE_SCHEMA = "v11"


def _video_cache_disabled() -> bool:
    return os.getenv("STEP_DISABLE_VIDEO_CACHE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class VideoAnalyzer:
    """Analyse a single video (YouTube URL or local file) into structured output."""

    def __init__(self) -> None:
        ensure_dirs()

    # ------------------------------------------------------------------
    # YouTube URL path
    # ------------------------------------------------------------------
    def analyze_youtube(self, url: str, *, mode: str = "quick",
                         use_cache: bool = True) -> dict:
        canon = canonical_youtube_url(url)
        if not canon:
            return {"media": "video", "error": "Invalid YouTube URL"}
        vid = youtube_video_id(canon) or "unknown"
        mode = (mode or "quick").lower()
        cache_key = f"yt_{vid}_{mode}_{CACHE_SCHEMA}.json"

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                cached["cached"] = True
                return cached

        t0 = time.time()
        try:
            vlm = analyze_youtube(canon)
        except Exception as e:
            return {
                "media": "video",
                "source": "youtube",
                "url": canon,
                "video_id": vid,
                "mode": mode,
                "error": str(e)[:200],
            }

        result = self._finalize(
            vlm,
            base={
                "media": "video",
                "source": "youtube",
                "url": canon,
                "video_id": vid,
                "mode": mode,
            },
            elapsed_s=round(time.time() - t0, 1),
        )

        if mode == "deep":
            self._attach_deep(result, source_hint=canon, vid=vid, is_youtube=True)
        self._attach_similarity(result)

        # Only cache successful runs so transient failures can be retried.
        if use_cache and not _video_cache_disabled() and not (mode == "deep" and result.get("deep_error")):
            self._save_cache(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Uploaded file path
    # ------------------------------------------------------------------
    def analyze_file(self, path: str | Path, *, mode: str = "quick",
                      use_cache: bool = True) -> dict:
        path = Path(path)
        if not path.exists():
            return {"media": "video", "error": f"File not found: {path.name}"}
        sha = self._sha256(path)
        mode = (mode or "quick").lower()
        cache_key = f"file_{sha}_{mode}_{CACHE_SCHEMA}.json"

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                cached["cached"] = True
                return cached

        t0 = time.time()
        try:
            vlm = analyze_local_file(path)
        except Exception as e:
            return {
                "media": "video",
                "source": "upload",
                "file": path.name,
                "file_sha": sha,
                "mode": mode,
                "error": str(e)[:200],
            }

        result = self._finalize(
            vlm,
            base={
                "media": "video",
                "source": "upload",
                "file": path.name,
                "file_sha": sha,
                "mode": mode,
            },
            elapsed_s=round(time.time() - t0, 1),
        )

        if mode == "deep":
            self._attach_deep(result, source_hint=path, vid=None, is_youtube=False)
        self._attach_similarity(result)

        if use_cache and not _video_cache_disabled() and not (mode == "deep" and result.get("deep_error")):
            self._save_cache(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Deep mode: frame sampling + per-scene keyword extraction
    # ------------------------------------------------------------------
    def _attach_deep(self, result: dict, *, source_hint, vid: str | None,
                      is_youtube: bool) -> None:
        """Run the Deep pipeline and mutate ``result`` in place with its output."""
        temp_dir: tempfile.TemporaryDirectory | None = None
        try:
            if is_youtube:
                if _video_cache_disabled():
                    temp_dir = tempfile.TemporaryDirectory(prefix="step_video_")
                    downloads_dir = Path(temp_dir.name)
                else:
                    downloads_dir = VIDEO_CACHE_DIR / "downloads"
                video_path = download_youtube_video(str(source_hint), downloads_dir)
            else:
                video_path = Path(source_hint)
            frames = extract_frames(video_path)
            _log.info(f"  [L0v] Extracted {len(frames)} frame(s) from {video_path.name}")
            if not frames:
                result["deep_error"] = "No frames could be extracted"
                result["problems"] = []
                return
            deep = analyze_frames_deep(
                frames,
                title_hint=str(result.get("title") or ""),
                summary_hint=str(result.get("summary") or ""),
            )
            result["problems"] = deep.get("problems", [])
            result["deep_elapsed_s"] = deep.get("elapsed_s")
            result["deep_frame_count"] = len(frames)
        except Exception as e:
            _log.info(f"  [L3vd] [FAIL] {str(e)[:120]}")
            result["deep_error"] = str(e)[:200]
            result["problems"] = []
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    @staticmethod
    def _short_query_text(title: str, summary: str) -> str:
        t = (title or "").strip()
        s = (summary or "").strip()
        if t and s:
            return f"{t}\n\n{s}"
        return t or s

    @staticmethod
    def _scene_similarity_query(scene_text: str, title: str, summary: str) -> tuple[str, str]:
        """Blend scene LaTeX with a short video context so sparse OCR does not
        collapse every scene to the same embedding neighborhood.
        """
        body = normalize_embedding_query(scene_text)
        ctx_full = VideoAnalyzer._short_query_text(title, summary).strip()
        ctx = ctx_full[:720] if ctx_full else ""
        if body and ctx:
            return f"{body}\n\n---\nContext: {ctx}", "scene_text+video_context"
        if body:
            return body, "scene_text"
        if ctx:
            return f"Context: {ctx}", "video_context"
        return "", "none"

    def _attach_similarity(self, result: dict) -> None:
        """Attach embedding-based cosine similarity scores for keyword selection.

        Gemini still produces the video summary / scene text. Similarity scoring
        ranks the fixed pool in an explicit vector space and returns the top-5
        along with scores for reporting.
        """
        pool = list(result.get("pool") or [])
        if not pool:
            result["keywords_source"] = "llm"
            return

        if not similarity_enabled():
            result["keywords_source"] = "llm"
            return

        cfg = default_similarity_config()

        # Preserve the original model-selected keyword list for debugging/ablation.
        if "keywords_llm" not in result and isinstance(result.get("keywords"), list):
            result["keywords_llm"] = list(result.get("keywords") or [])

        # Global selection (Quick or overall video in Deep): hybrid re-ranker.
        q = self._short_query_text(result.get("title", ""), result.get("summary", ""))
        if q:
            try:
                sim = top_k_keywords(q, pool, cfg=cfg, query_source="title+summary")
                result["keyword_similarity"] = sim
                ranked = sim.get("ranked") or []

                cosine_scores = {
                    r["keyword"]: float(r["score"])
                    for r in ranked
                    if isinstance(r, dict) and r.get("keyword")
                }
                llm_order = list(result.get("keywords_llm") or result.get("keywords") or [])
                rcfg = default_rerank_config()
                hybrid = rerank_pool(
                    cosine_scores=cosine_scores, llm_ranking=llm_order,
                    cfg=rcfg,
                )
                hybrid_top = [r["keyword"] for r in hybrid[: cfg.top_k]]
                result["keywords"] = hybrid_top
                result["keywords_source"] = "hybrid"
                result["hybrid_ranking"] = hybrid[:10]
                result["rerank_weights"] = {
                    "w_cos": rcfg.w_cos,
                    "w_llm": rcfg.w_llm,
                    "w_rule": rcfg.w_rule,
                }
                tax = result.get("taxonomy")
                if not isinstance(tax, dict):
                    tax = {}
                if result.get("keywords"):
                    tax["keywords"] = list(result["keywords"])[: cfg.top_k]
                    if not tax.get("topic") or not tax.get("subtopic"):
                        derived = topic_from_keywords(hybrid_top)
                        if derived:
                            tax["topic"] = derived[0]
                            tax["subtopic"] = derived[1]
                    result["taxonomy"] = tax
            except Exception as e:
                _log.info(f"  [SIM] [FAIL] global: {str(e)[:160]}")
                result["keyword_similarity"] = {
                    "method": "embedding_cosine",
                    "error": str(e)[:200],
                }
                result["keywords_source"] = "llm"

        # Deep mode: per-problem ranking; one bad scene must not abort the rest.
        problems = result.get("problems")
        if isinstance(problems, list) and problems:
            for p in problems:
                if not isinstance(p, dict):
                    continue
                if "keywords_llm" not in p and isinstance(p.get("keywords"), list):
                    p["keywords_llm"] = list(p.get("keywords") or [])
                scene_text = (p.get("text") or "").strip()
                q_sc, q_src = self._scene_similarity_query(
                    scene_text,
                    str(result.get("title") or ""),
                    str(result.get("summary") or ""),
                )
                if not (q_sc or "").strip():
                    p.setdefault("keywords_source", "llm")
                    continue
                try:
                    sim_p = top_k_keywords(
                        q_sc, pool, cfg=cfg, query_source=q_src
                    )
                    p["keyword_similarity"] = sim_p
                    ranked_p = sim_p.get("ranked") or []
                    cos_sc = {
                        r["keyword"]: float(r["score"])
                        for r in ranked_p
                        if isinstance(r, dict) and r.get("keyword")
                    }
                    scene_llm = list(p.get("keywords_llm") or p.get("keywords") or [])
                    rcfg = default_rerank_config()
                    scene_hybrid = rerank_pool(
                        cosine_scores=cos_sc, llm_ranking=scene_llm,
                        cfg=rcfg,
                    )
                    p["keywords"] = [r["keyword"] for r in scene_hybrid[: cfg.top_k]]
                    p["keywords_source"] = "hybrid"
                except Exception as e:
                    _log.info(f"  [SIM] [FAIL] scene: {str(e)[:120]}")
                    p["keyword_similarity"] = {
                        "method": "embedding_cosine",
                        "error": str(e)[:200],
                    }
                    p["keywords_source"] = "llm"

        if not result.get("keywords_source"):
            result["keywords_source"] = "llm"

    def _attach_pseudo_gold(self, result: dict) -> None:
        """Attach LLM-assisted pseudo-gold metrics for the global top-5 keywords."""
        pool = list(result.get("pool") or [])
        predicted = list(result.get("keywords") or [])
        if not pool or not predicted:
            return
        try:
            from pseudo_gold_eval import evaluate_pseudo_gold

            result["pseudo_gold_eval"] = evaluate_pseudo_gold(
                title=str(result.get("title") or ""),
                summary=str(result.get("summary") or ""),
                pool=pool,
                predicted_keywords=predicted[:5],
            )
        except Exception as e:
            result["pseudo_gold_eval"] = {"error": str(e)[:200]}

    # ------------------------------------------------------------------
    # Shared finalization / cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _finalize(vlm: dict, *, base: dict, elapsed_s: float) -> dict:
        summary_text = vlm.get("summary") or ""
        keywords = vlm.get("keywords", []) or []
        tax = classify_taxonomy(summary_text)

        # Regex taxonomy targets LaTeX / problem prose and often misses English
        # video summaries. Fall back to the already-chosen keywords (all from a
        # known subtopic pool) when the regex path finds nothing.
        if (not tax.get("topic") or not tax.get("subtopic")) and keywords:
            derived = topic_from_keywords(keywords)
            if derived:
                tax = {
                    "topic": derived[0],
                    "subtopic": derived[1],
                    "keywords": list(keywords[:5]),
                }

        tax = refine_subtopic(tax, summary_text, str(base.get("title") or ""))

        return {
            **base,
            "title": vlm.get("title", ""),
            "summary": summary_text,
            "keywords": keywords,
            "pool": vlm.get("pool", []),
            "model_used": vlm.get("model_used", ""),
            "vlm_elapsed_s": vlm.get("elapsed_s"),
            "elapsed_s": elapsed_s,
            "taxonomy": tax,
            "cached": False,
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _load_cache(name: str) -> dict | None:
        fp = VIDEO_CACHE_DIR / name
        if not fp.exists():
            return None
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _save_cache(name: str, data: dict) -> None:
        fp = VIDEO_CACHE_DIR / name
        # Stamp the payload with the schema in use so future readers can
        # double-check (the filename already encodes it via CACHE_SCHEMA).
        payload = dict(data)
        payload.setdefault("cache_schema", CACHE_SCHEMA)
        try:
            fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            _log.info(f"  [L3v] [WARN] Video cache write failed: {e}")
