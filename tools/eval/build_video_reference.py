"""
Build a fixed video query/reference file for semantic-model comparison.

The output is a JSON file with one item per YouTube URL. Each item contains
the fixed VLM text (`title`, `summary`, `query_text`) used by
tools/eval/videos_multi.py.

By default, this script records VLM keywords as `reference_keywords` only. Use
`--gold-from-vlm` when you explicitly want an agreement-style report where VLM
keywords are copied into `gold_keywords`. That is useful for model-comparison
debugging, but it is not human-validated ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from layer0_video import canonical_youtube_url, youtube_video_id  # noqa: E402
from run_video import VideoAnalyzer  # noqa: E402


def _urls_from_args(args) -> list[str]:
    urls: list[str] = []
    if args.urls:
        urls.extend(args.urls)
    if args.urls_file:
        fp = Path(args.urls_file).resolve()
        lines = fp.read_text(encoding="utf-8").splitlines()
        urls.extend(line.strip() for line in lines if line.strip() and not line.strip().startswith("#"))
    out: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        canon = canonical_youtube_url(raw.strip())
        if not canon:
            continue
        vid = youtube_video_id(canon) or canon
        if vid in seen:
            continue
        seen.add(vid)
        out.append(canon)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="*", help="YouTube URLs")
    parser.add_argument("--urls-file", default="", help="Text file with one YouTube URL per line")
    parser.add_argument("--mode", choices=("quick", "deep"), default="quick")
    parser.add_argument("--out", default="data/gold/videos_15_vlm_reference.json")
    parser.add_argument("--gold-from-vlm", action="store_true",
                        help="Copy VLM keywords into gold_keywords for agreement-style metrics.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Do not read/write video analysis cache.")
    args = parser.parse_args(argv)

    urls = _urls_from_args(args)
    if not urls:
        print("No valid YouTube URLs provided.", file=sys.stderr)
        return 2

    analyzer = VideoAnalyzer()
    items = []
    failures = []
    t0 = time.time()

    for i, url in enumerate(urls, start=1):
        vid = youtube_video_id(url) or f"video_{i}"
        print(f"[{i}/{len(urls)}] {vid} ({args.mode})...", flush=True)
        result = analyzer.analyze_youtube(url, mode=args.mode, use_cache=not args.no_cache)
        if result.get("error"):
            failures.append({"id": vid, "url": url, "error": result.get("error")})
            print(f"  failed: {result.get('error')}", flush=True)
            continue

        title = str(result.get("title") or "").strip()
        summary = str(result.get("summary") or "").strip()
        query_text = f"{title}\n\n{summary}".strip()
        reference_keywords = list(result.get("keywords") or [])[:5]
        item = {
            "id": vid,
            "url": url,
            "mode": args.mode,
            "title": title,
            "summary": summary,
            "query_text": query_text,
            "reference_keywords": reference_keywords,
            "reference_type": "vlm_keywords_not_human_gold",
            "topic": (result.get("taxonomy") or {}).get("topic"),
            "subtopic": (result.get("taxonomy") or {}).get("subtopic"),
            "model_used": result.get("model_used"),
            "cached": bool(result.get("cached")),
        }
        if args.gold_from_vlm:
            item["gold_keywords"] = reference_keywords
            item["gold_warning"] = "VLM keywords copied as gold for agreement testing; not human ground truth."
        items.append(item)
        print(f"  title: {title}", flush=True)
        print(f"  refs: {', '.join(reference_keywords)}", flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = (_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "video-reference-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "n_requested": len(urls),
        "n_items": len(items),
        "n_failures": len(failures),
        "reference_type": "vlm_keywords_not_human_gold",
        "gold_from_vlm": bool(args.gold_from_vlm),
        "items": items,
        "failures": failures,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote: {out}")
    if failures:
        print(f"Failures: {len(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
