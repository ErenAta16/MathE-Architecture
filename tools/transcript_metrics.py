"""Shared transcript metadata and Beatriz-oriented quality metrics."""

from __future__ import annotations

import re
from pathlib import Path

_HEADER_KV = re.compile(r"^([a-z_]+):\s*(.*)$", re.IGNORECASE)
_INAUDIBLE = re.compile(r"\[inaudible\]", re.IGNORECASE)


def split_header_body(text: str) -> tuple[dict[str, str], str]:
    if "\n\n" not in text:
        return {}, text.strip()
    head, body = text.split("\n\n", 1)
    meta: dict[str, str] = {}
    for line in head.splitlines():
        line = line.strip()
        if line.startswith("="):
            continue
        m = _HEADER_KV.match(line)
        if m:
            meta[m.group(1).lower()] = m.group(2).strip()
    return meta, body.strip()


def content_metrics(body: str) -> dict:
    words = body.split()
    sentences = [s for s in re.split(r"[.!?]+", body) if s.strip()]
    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    return {
        "word_count": len(words),
        "char_count": len(body),
        "sentence_count": max(len(sentences), 1),
        "paragraph_count": max(len(paragraphs), 1),
        "avg_word_len": round(sum(len(w) for w in words) / max(len(words), 1), 2),
        "inaudible_spans": len(_INAUDIBLE.findall(body)),
    }


def keyword_hits(body: str, keywords: list[str]) -> dict[str, bool]:
    lower = body.lower()
    out: dict[str, bool] = {}
    for kw in keywords:
        k = (kw or "").strip()
        if not k:
            continue
        out[k] = k.lower() in lower
    return out


def keyword_recall(body: str, keywords: list[str]) -> dict:
    hits = keyword_hits(body, keywords)
    n = len(hits)
    found = sum(1 for v in hits.values() if v)
    return {
        "keyword_hits": hits,
        "keyword_hit_count": found,
        "keyword_total": n,
        "keyword_recall_pct": round(100.0 * found / n, 1) if n else 0.0,
    }


def timing_metrics(*, elapsed_s: float | None, duration_s: float | None) -> dict:
    out: dict[str, float | None] = {
        "elapsed_s": round(elapsed_s, 1) if elapsed_s is not None else None,
        "duration_s": round(duration_s, 1) if duration_s is not None else None,
        "real_time_factor": None,
        "words_per_minute": None,
    }
    if elapsed_s and duration_s and duration_s > 0:
        out["real_time_factor"] = round(elapsed_s / duration_s, 3)
    return out


def term_audit(body: str, checks: dict[str, list[str]]) -> dict[str, bool]:
    lower = body.lower()
    return {label: any(tok.lower() in lower for tok in tokens) for label, tokens in checks.items()}


# Per-video quality probes for Beatriz (beyond literal keyword-pool recall).
PILOT_QUALITY_CHECKS: dict[str, dict[str, list[str]]] = {
    "KTNcYYHuBTY": {
        "imaginary_unit_i": [" imaginary unit ", " unit imaginary", "powers of i", "power of i"],
        "sqrt_negative_one": ["square root of negative", "sqrt{-1}", r"\sqrt{-1}"],
    },
    "LVLuqNH5iWw": {
        "saddle_point_correct": ["saddle point"],
        "saddle_point_wrong": ["subtle point"],
        "critical_points": ["critical point"],
        "partial_derivatives": ["partial derivative", "partial derivatives"],
    },
    "uKfcS7-O6UE": {
        "quotient_rule": ["quotient rule"],
        "dy_dx_notation": ["dy dx", "dy/dx", r"\frac{dy}{dx}"],
    },
}


def read_transcript(path: Path) -> tuple[dict[str, str], str]:
    if not path.is_file():
        return {}, ""
    return split_header_body(path.read_text(encoding="utf-8"))


def merge_record(
    *,
    video_id: str,
    url: str,
    title: str,
    method: str,
    header: dict[str, str],
    body: str,
    extra: dict | None = None,
    reference_keywords: list[str] | None = None,
) -> dict:
    cm = content_metrics(body)
    kw = keyword_recall(body, reference_keywords or [])
    elapsed = header.get("elapsed_s")
    duration = header.get("duration_s")
    try:
        elapsed_f = float(elapsed) if elapsed else None
    except ValueError:
        elapsed_f = None
    try:
        duration_f = float(duration) if duration else None
    except ValueError:
        duration_f = None
    tm = timing_metrics(elapsed_s=elapsed_f, duration_s=duration_f)
    wpm = None
    if duration_f and duration_f > 0 and cm["word_count"]:
        wpm = round(cm["word_count"] / (duration_f / 60.0), 1)
    record = {
        "video_id": video_id,
        "url": url,
        "title": title,
        "method": method,
        **{k: v for k, v in header.items() if k not in ("method",)},
        **cm,
        **tm,
        **{k: v for k, v in kw.items() if k != "keyword_hits"},
        "keyword_hits": kw.get("keyword_hits", {}),
        "words_per_minute": wpm,
    }
    if extra:
        record.update(extra)
    return record
