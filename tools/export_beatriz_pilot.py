"""
Export final Beatriz pilot transcription package (Gemini primary + baselines).

Usage:

    python tools/export_beatriz_pilot.py
    python tools/export_beatriz_pilot.py --skip-gemini-run   # export only
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location("transcript_metrics", _HERE.parent / "transcript_metrics.py")
_tm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tm)

PILOT_IDS = ["KTNcYYHuBTY", "LVLuqNH5iWw", "uKfcS7-O6UE"]
URLS_FILE = _ROOT / "data" / "transcripts" / "pilot_3_urls.txt"
GOLD = _ROOT / "data" / "gold" / "videos_15_graded_vlm_reference_nocache.json"
TRANSCRIPT_DIR = _ROOT / "data" / "transcripts"
OUT = _ROOT / "reports" / "transcripts" / "beatriz_pilot"


def _load_gold() -> dict[str, dict]:
    data = json.loads(GOLD.read_text(encoding="utf-8"))
    return {str(it["id"]): it for it in data.get("items") or [] if isinstance(it, dict) and it.get("id")}


def _run_gemini() -> int:
    cmd = [
        sys.executable,
        str(_ROOT / "tools" / "transcribe_youtube_gemini.py"),
        "--urls-file",
        str(URLS_FILE),
        "--mode",
        "upload",
        "--model",
        "gemini-3-flash-preview",
        "--primary-only",
        "--prompt-profile",
        "readable",
    ]
    print(">>", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(_ROOT))


def _export_one(
    *,
    src: Path,
    dest_dir: Path,
    vid: str,
    prefix: str,
    title: str,
) -> dict | None:
    if not src.is_file():
        return None
    header, body = _tm.read_transcript(src)
    clean_name = f"{vid}_{prefix}_transcript.txt"
    full_name = f"{vid}_{prefix}_full.txt"
    (dest_dir / clean_name).write_text(
        f"# {title}\n# video_id: {vid}\n# url: {header.get('url', '')}\n\n{body}\n",
        encoding="utf-8",
    )
    shutil.copy2(src, dest_dir / full_name)
    gold = _load_gold().get(vid, {})
    ref_kw = [str(k) for k in (gold.get("reference_keywords") or [])]
    checks = _tm.PILOT_QUALITY_CHECKS.get(vid, {})
    row = _tm.merge_record(
        video_id=vid,
        url=header.get("url") or gold.get("url") or "",
        title=title,
        method=prefix,
        header=header,
        body=body,
        reference_keywords=ref_kw,
        extra={"quality_checks": _tm.term_audit(body, checks)} if checks else None,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Beatriz pilot transcription package")
    parser.add_argument("--skip-gemini-run", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args()

    if not args.skip_gemini_run:
        rc = _run_gemini()
        if rc != 0:
            print("WARNING: Gemini run returned non-zero; exporting latest files anyway.")

    gold_map = _load_gold()
    out = Path(args.out_dir)
    gemini_dir = out / "gemini_primary"
    whisper_dir = out / "baseline_whisper"
    captions_dir = out / "baseline_youtube_captions"
    for d in (gemini_dir, whisper_dir, captions_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for vid in PILOT_IDS:
        item = gold_map.get(vid, {})
        title = item.get("title") or vid
        for src_suffix, dest_dir, prefix in (
            ("gemini", gemini_dir, "gemini"),
            ("whisper", whisper_dir, "whisper"),
            ("youtube_captions", captions_dir, "youtube"),
        ):
            src = TRANSCRIPT_DIR / f"{vid}_{src_suffix}.txt"
            row = _export_one(src=src, dest_dir=dest_dir, vid=vid, prefix=prefix, title=title)
            if row:
                rows.append(row)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prompt_version = rows[0].get("prompt_version", "mathe-transcribe-v3-final") if rows else "n/a"

    # CSV for Beatriz
    csv_path = out / "transcript_metrics.csv"
    fields = [
        "video_id", "title", "method", "model", "prompt_version",
        "word_count", "char_count", "duration_s", "elapsed_s",
        "real_time_factor", "words_per_minute", "paragraph_count",
        "keyword_recall_pct", "keyword_hit_count", "keyword_total",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Summary markdown
    gemini_rows = [r for r in rows if r.get("method") == "gemini"]
    summary_lines = [
        "# MathE pilot — spoken transcripts for Beatriz",
        "",
        f"Generated: {generated}",
        "",
        "## Recommended files (primary deliverable)",
        "",
        "Use the **`gemini_primary/`** folder. These are the best-quality transcripts we can "
        "produce today with **Gemini 3 Flash** (`gemini-3-flash-preview`) and the optimized "
        f"math-education prompt (`{prompt_version}`).",
        "",
        "| File | Description |",
        "|------|-------------|",
    ]
    for vid in PILOT_IDS:
        title = gold_map.get(vid, {}).get("title") or vid
        summary_lines.append(
            f"| `gemini_primary/{vid}_gemini_transcript.txt` | {title} — clean transcript |"
        )
        summary_lines.append(
            f"| `gemini_primary/{vid}_gemini_full.txt` | Same video — transcript + run metadata |"
        )

    summary_lines.extend([
        "",
        "## Baselines (optional comparison)",
        "",
        "- `baseline_whisper/` — local **distil-large-v3** (optimized Whisper)",
        "- `baseline_youtube_captions/` — YouTube auto subtitles",
        "",
        "## How the Gemini transcripts were produced",
        "",
        "1. Download YouTube audio/video with yt-dlp.",
        "2. Upload media to Gemini Files API.",
        "3. Transcribe with `gemini-3-flash-preview`, temperature 0, math-aware system prompt.",
        "4. Formulas appear as inline LaTeX where helpful; spoken words are kept verbatim.",
        "",
        "## Quality snapshot (Gemini primary)",
        "",
        "| Video | Words | Duration (s) | Keyword recall* | Notes |",
        "|-------|------:|-------------:|----------------:|-------|",
    ])

    for row in gemini_rows:
        vid = row["video_id"]
        notes = []
        qc = row.get("quality_checks") or {}
        if qc.get("saddle_point_correct"):
            notes.append("saddle point ✓")
        if qc.get("saddle_point_wrong"):
            notes.append("subtle point ✗")
        if qc.get("quotient_rule"):
            notes.append("quotient rule ✓")
        if qc.get("imaginary_unit_i"):
            notes.append("imaginary unit ✓")
        summary_lines.append(
            f"| {vid} | {row.get('word_count')} | {row.get('duration_s')} | "
            f"{row.get('keyword_recall_pct')}% | {', '.join(notes) or '—'} |"
        )

    summary_lines.extend([
        "",
        "*Keyword recall = share of graded VLM reference keywords literally present in the "
        "transcript (proxy for MathE keyword-pool alignment, not human annotation).",
        "",
        "## Note on a separate pipeline step (not included here)",
        "",
        "These files are **spoken transcripts**. A different internal step feeds text into the "
        "MathE **keyword ranking** pipeline as `query_text` (today that input is usually a "
        "VLM video summary). Using transcripts instead is a possible follow-up experiment; "
        "this package is the transcription deliverable only.",
        "",
        "## Video URLs",
        "",
    ])
    for vid in PILOT_IDS:
        url = gold_map.get(vid, {}).get("url") or ""
        title = gold_map.get(vid, {}).get("title") or vid
        summary_lines.append(f"- **{title}** — {url}")

    (out / "README_FOR_BEATRIZ.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    manifest = {
        "generated_at": generated,
        "prompt_version": prompt_version,
        "primary_model": "gemini-3-flash-preview",
        "video_ids": PILOT_IDS,
        "records": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nBeatriz package ready: {out}")
    print(f"  README: {out / 'README_FOR_BEATRIZ.md'}")
    print(f"  Gemini transcripts: {gemini_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
