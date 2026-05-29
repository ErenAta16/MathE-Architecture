"""
Build a graded VLM reference from fixed video title/summary text.

Input should come from build_video_reference.py run with --no-cache. This
script does not analyze videos and does not read/write video caches; it asks a
separate text VLM judge to assign relevance grades to keywords from the closed
pool.

Relevance scale:
  3 = central / must-have keyword
  2 = clearly relevant keyword
  1 = weakly related supporting keyword
  0 = not relevant (not stored)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_FALLBACK_MODEL  # noqa: E402
from keyword_eval import DEFAULT_KEYWORD_POOL  # noqa: E402


SYSTEM = (
    "You are an independent evaluator for mathematics education video tags. "
    "Given a fixed video title and summary plus a closed keyword pool, assign "
    "graded relevance to the best matching pool keywords. Use only exact pool "
    "strings. Do not invent keywords. Do not solve math problems."
)


def _client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=GEMINI_API_KEY)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def _load_pool(path: str | None) -> tuple[list[str], str]:
    if not path:
        return list(DEFAULT_KEYWORD_POOL), "keyword_eval.DEFAULT_KEYWORD_POOL"
    fp = Path(path).resolve()
    data = _read_json(fp)
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("keywords") or data.get("pool") or data.get("items") or []
    else:
        raw = []
    pool = []
    for item in raw:
        if isinstance(item, str):
            pool.append(item.strip())
        elif isinstance(item, dict):
            pool.append(str(item.get("keyword") or item.get("name") or "").strip())
    pool = [p for p in pool if p]
    if not pool:
        raise ValueError(f"No keywords found in {fp}")
    return pool, str(fp)


def _canonicalize_graded(raw_items, pool: list[str], max_keywords: int) -> list[dict]:
    lookup = _pool_lookup(pool)
    by_key: dict[str, dict] = {}
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        raw_kw = str(item.get("keyword") or "").strip()
        canon = lookup.get(raw_kw.lower())
        if not canon:
            continue
        try:
            rel = int(round(float(item.get("relevance", 0))))
        except (TypeError, ValueError):
            rel = 0
        rel = max(0, min(3, rel))
        if rel <= 0:
            continue
        key = canon.lower()
        rationale = str(item.get("rationale") or "").strip()
        prev = by_key.get(key)
        if not prev or rel > int(prev["relevance"]):
            by_key[key] = {"keyword": canon, "relevance": rel, "rationale": rationale}
    items = list(by_key.values())
    items.sort(key=lambda d: (-int(d["relevance"]), d["keyword"].lower()))
    return items[:max_keywords]


def _judge_item(client: genai.Client, item: dict, pool: list[str], *,
                model: str, max_keywords: int) -> tuple[list[dict], str, str]:
    title = str(item.get("title") or "").strip()
    summary = str(item.get("summary") or "").strip()
    prompt = (
        "Video title:\n"
        + (title or "(missing)")
        + "\n\nVideo summary:\n"
        + (summary or "(missing)")
        + "\n\nClosed keyword pool. Copy exact strings only:\n"
        + ", ".join(pool)
        + "\n\nReturn JSON only with this schema:\n"
        "{\n"
        '  "graded_keywords": [\n'
        '    {"keyword": "exact pool keyword", "relevance": 3, "rationale": "short reason"}\n'
        "  ],\n"
        '  "summary": "one short sentence explaining the reference"\n'
        "}\n\n"
        f"Choose 10 to {max_keywords} distinct keywords if they are genuinely relevant. "
        "Use relevance 3 for central/must-have tags, 2 for clearly relevant tags, "
        "1 for weakly related supporting tags. Do not include irrelevant keywords."
    )
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        max_output_tokens=8192,
        temperature=0.0,
    )
    try:
        response = client.models.generate_content(model=model, contents=prompt, config=cfg)
        model_used = model
    except Exception:
        fallback = (GEMINI_FALLBACK_MODEL or "").strip()
        if not fallback or fallback == model:
            raise
        response = client.models.generate_content(model=fallback, contents=prompt, config=cfg)
        model_used = fallback
    data = _json_from_text(response.text or "")
    graded = _canonicalize_graded(data.get("graded_keywords") or [], pool, max_keywords)
    return graded, model_used, str(data.get("summary") or "").strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="No-cache video reference JSON")
    parser.add_argument("--out", default="data/gold/videos_15_graded_vlm_reference.json")
    parser.add_argument("--pool", default="", help="Optional keyword pool JSON")
    parser.add_argument("--model", default=GEMINI_MODEL or "gemini-2.5-flash")
    parser.add_argument("--max-keywords", type=int, default=15)
    parser.add_argument("--min-binary-relevance", type=int, default=2,
                        help="Keywords at or above this relevance are copied to gold_keywords for binary metrics.")
    args = parser.parse_args(argv)

    src = Path(args.input)
    if not src.is_absolute():
        src = (_ROOT / src).resolve()
    data = _read_json(src)
    items = list(data.get("items") or [])
    if not items:
        print(f"No items found in {src}", file=sys.stderr)
        return 2

    pool, pool_source = _load_pool(args.pool or None)
    client = _client()
    out_items = []
    failures = []
    t0 = time.time()

    for idx, item in enumerate(items, start=1):
        vid = str(item.get("id") or item.get("video_id") or f"video_{idx}")
        print(f"[{idx}/{len(items)}] judging {vid}...", flush=True)
        try:
            graded, model_used, judge_summary = _judge_item(
                client,
                item,
                pool,
                model=(args.model or GEMINI_MODEL or "gemini-2.5-flash").strip(),
                max_keywords=max(5, args.max_keywords),
            )
        except Exception as e:
            failures.append({"id": vid, "error": str(e)[:240]})
            print(f"  failed: {e}", flush=True)
            continue

        binary_gold = [
            row["keyword"]
            for row in graded
            if int(row.get("relevance", 0)) >= max(1, args.min_binary_relevance)
        ]
        out_item = dict(item)
        out_item["graded_reference"] = graded
        out_item["gold_keywords"] = binary_gold
        out_item["reference_type"] = "graded_vlm_reference_not_human_gold"
        out_item["reference_model_used"] = model_used
        out_item["reference_summary"] = judge_summary
        out_item["gold_warning"] = (
            "gold_keywords are derived from graded VLM reference with relevance >= "
            f"{args.min_binary_relevance}; not human ground truth."
        )
        out_items.append(out_item)
        print(
            "  graded: "
            + ", ".join(f"{r['keyword']}:{r['relevance']}" for r in graded[:8]),
            flush=True,
        )

    out = Path(args.out)
    if not out.is_absolute():
        out = (_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "graded-video-reference-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_path": str(src),
        "pool_source": pool_source,
        "pool_size": len(pool),
        "reference_type": "graded_vlm_reference_not_human_gold",
        "binary_gold_rule": f"relevance >= {args.min_binary_relevance}",
        "n_items": len(out_items),
        "n_failures": len(failures),
        "items": out_items,
        "failures": failures,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote: {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
