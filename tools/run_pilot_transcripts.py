"""
Run optimized pilot transcription for all 3 methods and build the metrics report.

Usage (from Step_Project root):

    python tools/run_pilot_transcripts.py
    python tools/run_pilot_transcripts.py --skip-gemini
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
URLS = _ROOT / "data" / "transcripts" / "pilot_3_urls.txt"
IDS = ["KTNcYYHuBTY", "LVLuqNH5iWw", "uKfcS7-O6UE"]
GOLD = _ROOT / "data" / "gold" / "videos_15_graded_vlm_reference_nocache.json"


def _run(cmd: list[str]) -> int:
    print("\n>>", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 3-method pilot transcription + report")
    parser.add_argument("--urls-file", default=str(URLS))
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-captions", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    rc = 0

    if not args.skip_gemini:
        rc |= _run([
            py, "tools/transcribe_youtube_gemini.py",
            "--urls-file", args.urls_file,
            "--mode", "upload",
            "--model", "gemini-3-flash-preview",
        ])

    if not args.skip_whisper:
        rc |= _run([
            py, "tools/transcribe_youtube_whisper.py",
            "--urls-file", args.urls_file,
            "--model", "distil-large-v3",
            "--beam-size", "5",
        ])

    if not args.skip_captions:
        rc |= _run([
            py, "tools/transcribe_youtube_captions.py",
            "--urls-file", args.urls_file,
        ])

    if not args.skip_report:
        rc |= _run([
            py, "tools/build_transcript_report.py",
            "--gold", str(GOLD),
            "--ids", *IDS,
        ])

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
