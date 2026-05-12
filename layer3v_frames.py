"""
Layer 3v-deep — frame-level VLM for video "Deep" mode.

Runs Gemini Flash on each sampled frame to extract visible mathematical
problem statements, then groups near-duplicate outputs into scenes and (in a
single batch call) tags each scene with keywords from the closed MathE pool.

The goal is complementary to Layer 3v (native video): Layer 3v gives the
overall summary + top-5 keywords for the whole video, this module gives the
per-problem breakdown with timestamps.
"""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

from config import GEMINI_API_KEY
from keyword_eval import DEFAULT_KEYWORD_POOL
from taxonomy import classify_taxonomy


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame-level VLM: one call per sampled frame
# ---------------------------------------------------------------------------

_FRAME_SYSTEM = (
    "You are a careful visual OCR for mathematics education videos. Look at a "
    "single frame and extract the math-relevant content currently visible on "
    "screen. Include formulas/equations as clean LaTeX, and also include short "
    "visible concept labels, theorem names, axis labels, or slide/board text "
    "when they are mathematically meaningful (for example: KKT conditions, "
    "corner solution, feasible set, critical point). Do not solve, infer, or "
    "explain beyond what is visible. If no math-relevant formula or concept "
    "text is visible, output exactly NONE."
)

_FRAME_PROMPT = (
    "Return only the visible math-relevant content. Use one concise line per "
    "formula, concept label, or board/slide phrase. Use LaTeX for formulas, "
    "plain text for visible concept labels. No commentary, no invented steps, "
    "and no dollar-sign wrappers unless they appear as part of the on-screen "
    "math."
)


def _client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=GEMINI_API_KEY)


def _extract_one(client: genai.Client, ts: float, jpeg_bytes: bytes,
                   model: str) -> dict:
    try:
        response = client.models.generate_content(
            model=model,
            contents=types.Content(
                role="user",
                parts=[
                    types.Part(inline_data=types.Blob(
                        data=jpeg_bytes,
                        mime_type="image/jpeg",
                    )),
                    types.Part(text=_FRAME_PROMPT),
                ],
            ),
            config=types.GenerateContentConfig(
                system_instruction=_FRAME_SYSTEM,
                max_output_tokens=1024,
                temperature=0.0,
            ),
        )
        return {"t": ts, "text": (response.text or "").strip(), "ok": True}
    except Exception as e:
        return {"t": ts, "text": "", "ok": False, "error": str(e)[:140]}


def extract_frame_texts(frames: list[tuple[float, bytes]], *,
                          model: str = "gemini-2.5-flash",
                          max_workers: int | None = None,
                          verbose: bool = True) -> list[dict]:
    """Run the frame VLM on every sampled frame, in a bounded thread pool."""
    if not frames:
        return []
    client = _client()

    try:
        default_workers = int(os.environ.get("STEP_VIDEO_FRAME_WORKERS", "3"))
    except ValueError:
        default_workers = 3
    workers = max(1, min(max_workers or default_workers, len(frames), 8))

    out: list[dict] = [{} for _ in frames]
    if verbose:
        _log.info(f"  [L3vd] Frame VLM ({len(frames)} frames, workers={workers})...")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_extract_one, client, ts, jpg, model): idx
            for idx, (ts, jpg) in enumerate(frames)
        }
        for fut in as_completed(futs):
            idx = futs[fut]
            out[idx] = fut.result()
    return out


# ---------------------------------------------------------------------------
# Grouping near-duplicate frame outputs into scenes / problems
# ---------------------------------------------------------------------------

def _normalize_latex(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # Keep typical math punctuation but drop noise that varies between OCR passes.
    s = re.sub(r"[^\w\\{}^()+\-*/=.,| ]", "", s)
    return s


_LATEX_ERROR_CUES: tuple[str, ...] = (
    "missing argument",
    "missing superscript",
    "missing subscript",
    "undefined control sequence",
    "extra }, or forgotten",
    "missing }",
    "math processing error",
)


def _clean_scene_text(text: str) -> str:
    """Normalize frame OCR before grouping/rendering scene snippets."""
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("`", " ")
    cleaned = cleaned.replace("$$", " ")
    cleaned = cleaned.replace("$", " ")
    cleaned = cleaned.replace("\\displaystyle", " ")
    cleaned = re.sub(r"\\textcircled\s*\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\textcircled\s*([A-Za-z0-9])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _balanced_braces(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _has_dangling_script(text: str) -> bool:
    """Return True for TeX snippets ending with a bare ^ or _ script marker."""
    s = re.sub(r"\s+", "", text or "")
    return bool(re.search(r"[\^_](?:$|[+\-*/=,.;:|)\]}])", s))


def _trigram_set(s: str) -> set[str]:
    s = f"  {s}  "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _similarity(a: str, b: str) -> float:
    ta, tb = _trigram_set(a), _trigram_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _scene_text_meaningful(text: str, *, min_norm: int = 18) -> bool:
    """Drop OCR junk such as a lone dollar sign or ultra-short fragments."""
    raw = (text or "").strip()
    if not raw:
        return False
    lower = raw.lower()
    if any(cue in lower for cue in _LATEX_ERROR_CUES):
        return False
    cleaned = _clean_scene_text(raw)
    if not cleaned:
        return False
    if not _balanced_braces(cleaned):
        return False
    if _has_dangling_script(cleaned):
        return False
    if re.search(r"\\(?:text|operatorname)\b(?!\s*\{)", cleaned):
        return False
    norm = _normalize_latex(cleaned)
    if len(norm) < min_norm:
        return False
    stripped = norm.replace("$", "").replace("\\", "").strip()
    if len(stripped) < 8:
        return False
    return True


def group_scenes(frame_texts: list[dict], *,
                  min_chars: int = 20,
                  merge_threshold: float = 0.65) -> list[dict]:
    """Merge near-duplicate frame captures into scenes, keyed by earliest time.

    Each scene is a dict with ``earliest_t``, ``last_t``, ``timestamps``, and
    ``text`` (the longest representative LaTeX seen in the group).
    """
    items: list[tuple[float, str]] = []
    for row in frame_texts:
        text = (row.get("text") or "").strip()
        if not text or text.upper() == "NONE":
            continue
        # Drop obvious filler / short OCR noise.
        cleaned = _clean_scene_text(text)
        if len(cleaned) < min_chars or not _scene_text_meaningful(cleaned):
            continue
        items.append((row["t"], cleaned))

    items.sort(key=lambda x: x[0])

    scenes: list[dict] = []
    for ts, text in items:
        norm = _normalize_latex(text)
        best_idx = -1
        best_sim = 0.0
        for i, sc in enumerate(scenes):
            sim = _similarity(norm, sc["norm"])
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx >= 0 and best_sim >= merge_threshold:
            sc = scenes[best_idx]
            sc["timestamps"].append(ts)
            sc["last_t"] = ts
            if len(text) > len(sc["text"]):
                sc["text"] = text
                sc["norm"] = norm
        else:
            scenes.append({
                "earliest_t": ts,
                "last_t": ts,
                "timestamps": [ts],
                "text": text,
                "norm": norm,
            })

    # Strip internal ``norm`` field before returning.
    return [
        {
            "earliest_t": s["earliest_t"],
            "last_t": s["last_t"],
            "timestamps": s["timestamps"],
            "text": s["text"],
        }
        for s in scenes
        if _scene_text_meaningful(s["text"])
    ]


# ---------------------------------------------------------------------------
# Batch keyword evaluation: one LLM call for all scenes
# ---------------------------------------------------------------------------

_BATCH_SYSTEM = (
    "You classify mathematics problems against a closed keyword pool. "
    "For each numbered problem, output one line EXACTLY in the form:\n"
    "[N] kw1, kw2, kw3, kw4, kw5\n"
    "Use only keywords from the pool, copied verbatim. Pick the five most "
    "relevant, ordered from most to least relevant. If fewer than five truly "
    "fit, repeat the closest remaining ones to keep the count at five. Do "
    "not add explanations."
)


def batch_evaluate_keywords(scenes: list[dict], *,
                              pool: list[str] | None = None,
                              model: str = "gemini-2.5-pro",
                              verbose: bool = True) -> list[list[str]]:
    """Ask Gemini once for top-5 pool keywords per scene. Returns a list aligned with ``scenes``."""
    if not scenes:
        return []
    pool = list(pool) if pool else list(DEFAULT_KEYWORD_POOL)
    pool_lookup = {p.lower(): p for p in pool}
    client = _client()

    numbered = "\n".join(
        f"[{i + 1}] {s['text']}" for i, s in enumerate(scenes)
    )
    user = (
        "Closed keyword pool (use these exact phrasings only):\n"
        + ", ".join(pool)
        + "\n\nProblems:\n" + numbered
    )

    if verbose:
        _log.info(f"  [L3vd] Batch keyword eval for {len(scenes)} scene(s)...")

    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=_BATCH_SYSTEM,
            max_output_tokens=4096,
            temperature=0.0,
        ),
    )
    text = (response.text or "").strip()

    # Parse numbered lines back into per-scene keyword lists.
    out: list[list[str]] = [[] for _ in scenes]
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"\[(\d+)\]\s*(.*)", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if not (0 <= idx < len(scenes)):
            continue
        parts = [p.strip() for p in m.group(2).split(",") if p.strip()]
        seen: set[str] = set()
        accepted: list[str] = []
        for p in parts:
            canon = pool_lookup.get(p.lower())
            if canon and canon not in seen:
                accepted.append(canon)
                seen.add(canon)
            if len(accepted) >= 5:
                break
        out[idx] = accepted

    return out


# ---------------------------------------------------------------------------
# Public orchestrator: scenes + per-scene taxonomy + per-scene keywords
# ---------------------------------------------------------------------------

def _scene_taxonomy_with_video_hints(
    scene_text: str, title: str, summary: str, tax: dict,
) -> dict:
    """Nudge per-scene taxonomy when OCR picks ``\\int`` without limits but the
    video context clearly describes volumes of revolution or definite integrals.
    """
    if not isinstance(tax, dict):
        return tax
    blob = " ".join(
        (scene_text or "", title or "", summary or ""),
    ).lower()
    cues = (
        "volume", "washer", "disk", "shell", "revolution", "rotate",
        "solid of", "definite integral", "axis of rotation",
    )
    if any(c in blob for c in cues):
        if tax.get("topic") == "Integration" and tax.get("subtopic") == "Indefinite Integrals":
            return {**tax, "subtopic": "Definite Integrals"}
    return tax


def analyze_frames_deep(frames: list[tuple[float, bytes]], *,
                          frame_model: str = "gemini-2.5-flash",
                          batch_model: str = "gemini-2.5-pro",
                          pool: list[str] | None = None,
                          title_hint: str = "",
                          summary_hint: str = "",
                          verbose: bool = True) -> dict:
    """Run the full Deep pipeline on a list of sampled frames.

    Returns ``{scenes_raw, problems, elapsed_s}`` where ``problems`` is a list
    of ``{earliest_t, last_t, timestamps, text, taxonomy, keywords}`` dicts.
    """
    t0 = time.time()
    frame_texts = extract_frame_texts(frames, model=frame_model, verbose=verbose)
    scenes = group_scenes(frame_texts)
    if verbose:
        _log.info(f"  [L3vd] {len(scenes)} distinct scene(s) after grouping")

    keywords_per_scene = batch_evaluate_keywords(scenes, pool=pool,
                                                  model=batch_model,
                                                  verbose=verbose) if scenes else []

    problems: list[dict] = []
    for i, sc in enumerate(scenes):
        tax = classify_taxonomy(sc["text"])
        tax = _scene_taxonomy_with_video_hints(
            sc["text"], title_hint, summary_hint, tax,
        )
        kws = keywords_per_scene[i] if i < len(keywords_per_scene) else []
        problems.append({
            "earliest_t": sc["earliest_t"],
            "last_t": sc["last_t"],
            "timestamps": sc["timestamps"],
            "text": sc["text"],
            "taxonomy": tax,
            "keywords": kws,
        })

    return {
        "scenes_raw": frame_texts,
        "problems": problems,
        "elapsed_s": round(time.time() - t0, 1),
    }
