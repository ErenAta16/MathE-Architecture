"""
LLM-assisted pseudo-gold evaluation for video keyword selection.

This module is for fast calibration when human gold labels are unavailable.
It asks an LLM judge to build a reference top-5 from the same closed keyword
pool and to grade the system's selected top-5. These scores are not a
replacement for human-validated evaluation in a paper.
"""

from __future__ import annotations

import json
import re
import time
from typing import Iterable, Sequence

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_FALLBACK_MODEL
from step_eval import (
    f1_at_k,
    jaccard,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


_SYSTEM = (
    "You are an independent evaluator for mathematics education video tags. "
    "Given a video title, summary, a closed keyword pool, and the system's "
    "predicted top-5 keywords, choose the ideal five keywords from the pool "
    "and grade each predicted keyword. Do not solve math problems. Use only "
    "the exact keyword strings from the closed pool."
)


def _client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=GEMINI_API_KEY)


def _json_from_text(text: str) -> dict:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _pool_lookup(pool: Iterable[str]) -> dict[str, str]:
    return {str(k).strip().lower(): str(k).strip() for k in pool if str(k).strip()}


def _canonicalize_keywords(items: Iterable, lookup: dict[str, str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        if isinstance(item, dict):
            raw = item.get("keyword", "")
        else:
            raw = item
        key = str(raw).strip().lower()
        canon = lookup.get(key)
        if canon and canon.lower() not in seen:
            out.append(canon)
            seen.add(canon.lower())
    return out


def _extract_ideal(data: dict, lookup: dict[str, str]) -> tuple[list[str], list[float]]:
    raw_items = data.get("pseudo_gold") or data.get("pseudo_gold_keywords") or []
    keywords: list[str] = []
    gains: list[float] = []
    seen: set[str] = set()
    for item in raw_items:
        if isinstance(item, dict):
            raw = item.get("keyword", "")
            rel = item.get("relevance", 3)
        else:
            raw = item
            rel = 3
        canon = lookup.get(str(raw).strip().lower())
        if not canon or canon.lower() in seen:
            continue
        try:
            gain = max(0.0, min(3.0, float(rel)))
        except (TypeError, ValueError):
            gain = 3.0
        keywords.append(canon)
        gains.append(gain)
        seen.add(canon.lower())
        if len(keywords) >= 5:
            break
    return keywords, gains


def _extract_predicted_relevance(data: dict, predicted: Sequence[str]) -> dict[str, float]:
    raw = data.get("predicted_relevance") or {}
    out: dict[str, float] = {}

    if isinstance(raw, list):
        pairs = []
        for item in raw:
            if isinstance(item, dict):
                pairs.append((item.get("keyword", ""), item.get("relevance", 0)))
    elif isinstance(raw, dict):
        pairs = list(raw.items())
    else:
        pairs = []

    pred_lc = {p.lower(): p for p in predicted}
    for key, value in pairs:
        canon = pred_lc.get(str(key).strip().lower())
        if not canon:
            continue
        try:
            out[canon] = max(0.0, min(3.0, float(value)))
        except (TypeError, ValueError):
            out[canon] = 0.0
    return out


def evaluate_pseudo_gold(
    *,
    title: str,
    summary: str,
    pool: Sequence[str],
    predicted_keywords: Sequence[str],
    model: str | None = None,
) -> dict:
    """Return pseudo-gold keywords and agreement metrics for one video."""
    predicted = [str(k).strip() for k in predicted_keywords[:5] if str(k).strip()]
    pool = [str(k).strip() for k in pool if str(k).strip()]
    if not pool or not predicted:
        return {"skipped": True, "error": "missing pool or predicted keywords"}

    lookup = _pool_lookup(pool)
    model = (model or GEMINI_MODEL or "gemini-2.5-flash").strip()
    prompt = (
        "Video title:\n"
        + (title or "(missing)")
        + "\n\nVideo summary:\n"
        + (summary or "(missing)")
        + "\n\nClosed keyword pool (copy exact strings only):\n"
        + ", ".join(pool)
        + "\n\nSystem predicted top-5 keywords:\n"
        + ", ".join(predicted)
        + "\n\nReturn JSON only with this schema:\n"
        "{\n"
        '  "pseudo_gold": [\n'
        '    {"keyword": "exact pool keyword", "relevance": 3}\n'
        "  ],\n"
        '  "predicted_relevance": {"predicted keyword": 0-3},\n'
        '  "rationale": "one short sentence"\n'
        "}\n"
        "Rules: pseudo_gold must contain exactly five distinct pool keywords, "
        "ordered from most to least central. Relevance scale: 3=central, "
        "2=relevant, 1=weakly related, 0=not relevant."
    )

    t0 = time.time()
    try:
        client = _client()
        cfg = types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            max_output_tokens=4096,
            temperature=0.0,
        )
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=cfg,
            )
            model_used = model
        except Exception as first_error:
            fallback = (GEMINI_FALLBACK_MODEL or "").strip()
            if not fallback or fallback == model:
                raise
            response = client.models.generate_content(
                model=fallback,
                contents=prompt,
                config=cfg,
            )
            model_used = fallback
        raw_text = response.text or ""
        data = _json_from_text(raw_text)
    except Exception as e:
        return {
            "error": str(e)[:200],
            "model_used": model,
            "elapsed_s": round(time.time() - t0, 1),
        }

    pseudo_gold, ideal_gains = _extract_ideal(data, lookup)
    pred_rel = _extract_predicted_relevance(data, predicted)
    # Fill missing judge grades from pseudo-gold membership so nDCG still works
    # when the model omits an entry in predicted_relevance.
    pseudo_set = {k.lower() for k in pseudo_gold}
    ranked_rels = [
        pred_rel.get(k, 3.0 if k.lower() in pseudo_set else 0.0)
        for k in predicted
    ]
    if not ideal_gains:
        ideal_gains = [3.0] * min(5, len(pseudo_gold))

    metrics = {
        "precision_at_5": round(precision_at_k(predicted, pseudo_gold, k=5), 4),
        "recall_at_5": round(recall_at_k(predicted, pseudo_gold, k=5), 4),
        "f1_at_5": round(f1_at_k(predicted, pseudo_gold, k=5), 4),
        "jaccard": round(jaccard(predicted, pseudo_gold), 4),
        "ndcg_at_5": round(ndcg_at_k(ranked_rels, ideal_gains, k=5), 4),
        "mean_relevance": round(sum(ranked_rels) / len(ranked_rels), 4) if ranked_rels else 0.0,
    }

    return {
        "method": "llm_assisted_pseudo_gold",
        "warning": "Pseudo-gold is model-generated and should not be treated as human-validated ground truth.",
        "model_used": model_used,
        "elapsed_s": round(time.time() - t0, 1),
        "pseudo_gold_keywords": pseudo_gold,
        "pseudo_gold_relevance": [
            {"keyword": k, "relevance": ideal_gains[i] if i < len(ideal_gains) else 3.0}
            for i, k in enumerate(pseudo_gold)
        ],
        "predicted_relevance": [
            {"keyword": k, "relevance": ranked_rels[i]}
            for i, k in enumerate(predicted)
        ],
        "metrics": metrics,
        "rationale": str(data.get("rationale", "")).strip(),
    }
