"""
Download YouTube auto/manual English captions and convert to plain text.

For side-by-side quality comparison with Whisper / Gemini transcripts.

Usage:

    python tools/transcribe_youtube_captions.py --urls-file data/transcripts/pilot_3_urls.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from layer0_video import canonical_youtube_url, youtube_video_id  # noqa: E402

_VTT_TS = re.compile(
    r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}"
)
_VTT_INLINE_TS = re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>")
_VTT_TAG = re.compile(r"</?c>|</?[^>]+>")


def _read_urls(path: Path) -> list[str]:
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def _vtt_to_text(vtt: str) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for raw in vtt.splitlines():
        line = raw.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if _VTT_TS.match(line) or line.isdigit():
            continue
        if line.startswith("Kind:") or line.startswith("Language:"):
            continue
        line = _VTT_INLINE_TS.sub("", line)
        line = _VTT_TAG.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line == "[Music]":
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return " ".join(lines)


def download_captions(url: str, out_dir: Path) -> tuple[Path, str, str]:
    import yt_dlp

    canon = canonical_youtube_url(url)
    if not canon:
        raise ValueError(f"Not a valid YouTube URL: {url!r}")
    vid = youtube_video_id(canon) or "captions"
    out_dir.mkdir(parents=True, exist_ok=True)

    for old in out_dir.glob(f"{vid}*.vtt"):
        old.unlink(missing_ok=True)

    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-orig", "en-US"],
        "subtitlesformat": "vtt",
        "outtmpl": str(out_dir / vid),
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(canon, download=False)
        ydl.download([canon])

    vtt_files = sorted(out_dir.glob(f"{vid}*.vtt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not vtt_files:
        raise RuntimeError(f"No English captions found for {vid}")

    vtt_path = vtt_files[0]
    caption_type = "manual" if ".en." in vtt_path.name and "auto" not in vtt_path.name else "auto"
    text = _vtt_to_text(vtt_path.read_text(encoding="utf-8", errors="replace"))
    if not text.strip():
        raise RuntimeError(f"Caption file empty for {vid}: {vtt_path.name}")
    return vtt_path, text, caption_type


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch YouTube captions as plain text")
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--urls-file")
    parser.add_argument("--out-dir", default=str(_ROOT / "data" / "transcripts"))
    args = parser.parse_args()

    urls = list(args.url)
    if args.urls_file:
        urls.extend(_read_urls(Path(args.urls_file)))
    if not urls:
        parser.error("Provide --url and/or --urls-file")

    out_dir = Path(args.out_dir)
    results: list[dict] = []

    for raw_url in urls:
        canon = canonical_youtube_url(raw_url)
        if not canon:
            continue
        vid = youtube_video_id(canon) or "unknown"
        print(f"\n=== {vid} (youtube captions) ===")
        try:
            vtt_path, text, caption_type = download_captions(canon, out_dir / "captions")
            txt_path = out_dir / f"{vid}_youtube_captions.txt"
            header = (
                f"video_id: {vid}\n"
                f"url: {canon}\n"
                f"method: youtube_captions\n"
                f"caption_type: {caption_type}\n"
                f"vtt_file: {vtt_path.name}\n"
                f"word_count: {len(text.split())}\n"
                f"char_count: {len(text)}\n"
                f"{'=' * 72}\n\n"
            )
            txt_path.write_text(header + text + "\n", encoding="utf-8")
            print(f"  type: {caption_type}, words: {len(text.split())}")
            print(f"  wrote: {txt_path}")
            results.append({"video_id": vid, "ok": True, "path": str(txt_path), "caption_type": caption_type})
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append({"video_id": vid, "ok": False, "error": str(e)})

    summary = out_dir / "youtube_captions_run_summary.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary}")
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
