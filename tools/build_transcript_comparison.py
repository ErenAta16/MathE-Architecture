"""
Build a side-by-side transcript comparison table for pilot videos.

Reads Whisper + YouTube caption txt files and optional VLM summaries from gold JSON.

Usage:

    python tools/build_transcript_comparison.py \\
        --gold data/gold/videos_15_graded_vlm_reference_nocache.json \\
        --ids KTNcYYHuBTY LVLuqNH5iWw uKfcS7-O6UE \\
        --out reports/transcripts/pilot_3_comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _read_body(path: Path) -> str:
    if not path.is_file():
        return "(missing)"
    text = path.read_text(encoding="utf-8")
    if "\n\n" in text:
        return text.split("\n\n", 1)[1].strip()
    return text.strip()


def _preview(text: str, n: int = 600) -> str:
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript-dir", default=str(_ROOT / "data" / "transcripts"))
    parser.add_argument("--gold")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--out", default=str(_ROOT / "reports" / "transcripts" / "pilot_3_comparison.md"))
    args = parser.parse_args()

    tdir = Path(args.transcript_dir)
    gold_map: dict[str, dict] = {}
    if args.gold:
        data = json.loads(Path(args.gold).read_text(encoding="utf-8"))
        for item in data.get("items") or []:
            if isinstance(item, dict) and item.get("id"):
                gold_map[str(item["id"])] = item

    lines = [
        "# Pilot transcript comparison (3 videos)",
        "",
        "Sources: **Whisper** (`medium.en`, local GPU) vs **YouTube auto/manual captions** vs **VLM summary** (existing gold).",
        "Gemini audio transcription pending (multimodal API returned 503 during pilot run).",
        "",
    ]

    rows = []
    for vid in args.ids:
        item = gold_map.get(vid, {})
        title = item.get("title") or vid
        summary = item.get("summary") or "(no summary in gold)"
        whisper = _read_body(tdir / f"{vid}_whisper.txt")
        captions = _read_body(tdir / f"{vid}_youtube_captions.txt")
        rows.append((vid, title, summary, whisper, captions))

    for vid, title, summary, whisper, captions in rows:
        lines.extend([
            f"## {vid} — {title}",
            "",
            f"**VLM summary** ({len(summary.split())} words)",
            "",
            summary,
            "",
            f"**Whisper** ({len(whisper.split()) if whisper != '(missing)' else 0} words)",
            "",
            _preview(whisper, 1200),
            "",
            f"**YouTube captions** ({len(captions.split()) if captions != '(missing)' else 0} words)",
            "",
            _preview(captions, 1200),
            "",
            "---",
            "",
        ])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
