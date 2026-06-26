"""
Build a Beatriz-oriented pilot transcript report (Gemini + Whisper + YouTube captions).

Usage:

    python tools/build_transcript_report.py \\
        --gold data/gold/videos_15_graded_vlm_reference_nocache.json \\
        --ids KTNcYYHuBTY LVLuqNH5iWw uKfcS7-O6UE
"""

from __future__ import annotations

import argparse
import csv
import json
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
merge_record = _tm.merge_record
read_transcript = _tm.read_transcript
term_audit = _tm.term_audit
PILOT_QUALITY_CHECKS = _tm.PILOT_QUALITY_CHECKS


def _preview(text: str, n: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def _load_gold(path: Path | None) -> dict[str, dict]:
    if not path or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(it["id"]): it for it in (data.get("items") or []) if isinstance(it, dict) and it.get("id")}


def _collect_method(
    *,
    tdir: Path,
    vid: str,
    suffix: str,
    method: str,
    gold_item: dict,
) -> dict | None:
    path = tdir / f"{vid}_{suffix}.txt"
    header, body = read_transcript(path)
    if not body and not header:
        return None
    ref_kw = [str(k) for k in (gold_item.get("reference_keywords") or [])]
    checks = PILOT_QUALITY_CHECKS.get(vid, {})
    quality = term_audit(body, checks) if checks else {}
    row = merge_record(
        video_id=vid,
        url=gold_item.get("url") or header.get("url") or "",
        title=gold_item.get("title") or header.get("title_hint") or vid,
        method=method,
        header=header,
        body=body,
        reference_keywords=ref_kw,
        extra={"quality_checks": quality} if quality else None,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pilot transcript metrics report")
    parser.add_argument("--transcript-dir", default=str(_ROOT / "data" / "transcripts"))
    parser.add_argument("--gold", default=str(_ROOT / "data" / "gold" / "videos_15_graded_vlm_reference_nocache.json"))
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--out-md", default=str(_ROOT / "reports" / "transcripts" / "pilot_3_metrics_report.md"))
    parser.add_argument("--out-csv", default=str(_ROOT / "reports" / "transcripts" / "pilot_3_metrics_summary.csv"))
    parser.add_argument("--out-json", default=str(_ROOT / "reports" / "transcripts" / "pilot_3_metrics_summary.json"))
    args = parser.parse_args()

    tdir = Path(args.transcript_dir)
    gold_map = _load_gold(Path(args.gold))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    methods = [
        ("gemini", "Gemini (optimized)"),
        ("whisper", "Whisper (optimized)"),
        ("youtube_captions", "YouTube captions"),
    ]

    all_rows: list[dict] = []
    per_video: dict[str, list[dict]] = {vid: [] for vid in args.ids}

    for vid in args.ids:
        item = gold_map.get(vid, {"id": vid, "title": vid, "reference_keywords": []})
        for suffix, label in methods:
            row = _collect_method(tdir=tdir, vid=vid, suffix=suffix, method=label, gold_item=item)
            if row:
                all_rows.append(row)
                per_video[vid].append(row)

    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    for p in (out_md, out_csv, out_json):
        p.parent.mkdir(parents=True, exist_ok=True)

    # CSV summary
    csv_fields = [
        "video_id", "title", "method", "model", "word_count", "char_count",
        "duration_s", "elapsed_s", "real_time_factor", "words_per_minute",
        "segment_count", "paragraph_count", "keyword_hit_count", "keyword_total",
        "keyword_recall_pct", "inaudible_spans", "caption_type", "prompt_version",
        "device", "fallback_used", "quality_checks",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            out_row = dict(row)
            if isinstance(out_row.get("quality_checks"), dict):
                out_row["quality_checks"] = json.dumps(out_row["quality_checks"], ensure_ascii=False)
            if isinstance(out_row.get("keyword_hits"), dict):
                out_row["keyword_hits"] = json.dumps(out_row["keyword_hits"], ensure_ascii=False)
            w.writerow(out_row)

    payload = {"generated_at": generated_at, "video_ids": args.ids, "records": all_rows}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Pilot transcript metrics report (3 videos × 3 methods)",
        "",
        f"Generated: `{generated_at}` (UTC)",
        "",
        "Methods compared:",
        "- **Gemini** — `gemini-3-flash-preview`, optimized math-education prompts (`mathe-transcribe-v2`)",
        "- **Whisper** — `distil-large-v3` (best local ASR tier), beam search + math initial prompt",
        "- **YouTube captions** — platform auto/manual subtitles (baseline)",
        "",
        "> **Note:** YouTube captions for `KTNcYYHuBTY` could not be refreshed (HTTP 429 rate limit). "
        "LVL and uKfc caption rows are from this run.",
        "",
        "## Summary table",
        "",
        "| Video | Method | Model | Words | Chars | Duration (s) | Elapsed (s) | RTF | WPM | KW recall |",
        "|-------|--------|-------|------:|------:|-------------:|------------:|----:|----:|----------:|",
    ]

    for vid in args.ids:
        for row in per_video.get(vid, []):
            lines.append(
                f"| {vid} | {row.get('method', '')} | {row.get('model') or row.get('caption_type') or '—'} "
                f"| {row.get('word_count', '')} | {row.get('char_count', '')} "
                f"| {row.get('duration_s') or '—'} | {row.get('elapsed_s') or '—'} "
                f"| {row.get('real_time_factor') or '—'} | {row.get('words_per_minute') or '—'} "
                f"| {row.get('keyword_recall_pct', 0)}% ({row.get('keyword_hit_count', 0)}/{row.get('keyword_total', 0)}) |"
            )

    lines.extend([
        "",
        "RTF = real-time factor (elapsed ÷ audio duration; lower is faster). "
        "WPM = words per minute estimated from transcript length. "
        "KW recall = share of graded VLM reference keywords literally present in the transcript.",
        "",
    ])

    for vid in args.ids:
        item = gold_map.get(vid, {})
        title = item.get("title") or vid
        ref_kw = [str(k) for k in (item.get("reference_keywords") or [])]
        lines.extend([f"## {vid} — {title}", ""])
        if item.get("summary"):
            lines.extend(["**VLM summary (existing pipeline):**", "", item["summary"], ""])
        if ref_kw:
            lines.append(f"**Reference keywords (graded):** {', '.join(ref_kw)}")
            lines.append("")

        for row in per_video.get(vid, []):
            lines.append(f"### {row.get('method')}")
            lines.append("")
            meta_lines = [
                f"- model/config: `{row.get('model') or row.get('caption_type') or 'n/a'}`",
                f"- words / chars: **{row.get('word_count')}** / **{row.get('char_count')}**",
                f"- duration_s: {row.get('duration_s') or 'n/a'}",
                f"- elapsed_s: {row.get('elapsed_s') or 'n/a'}",
                f"- real_time_factor: {row.get('real_time_factor') or 'n/a'}",
                f"- words_per_minute: {row.get('words_per_minute') or 'n/a'}",
                f"- paragraphs: {row.get('paragraph_count')}",
                f"- keyword recall: **{row.get('keyword_recall_pct')}%** "
                f"({row.get('keyword_hit_count')}/{row.get('keyword_total')})",
            ]
            if row.get("segment_count") is not None:
                meta_lines.append(f"- segments: {row.get('segment_count')}")
            if row.get("prompt_version"):
                meta_lines.append(f"- prompt_version: `{row.get('prompt_version')}`")
            if row.get("device"):
                meta_lines.append(f"- device: `{row.get('device')}`")
            if row.get("beam_size"):
                meta_lines.append(f"- beam_size: {row.get('beam_size')}")
            hits = row.get("keyword_hits") or {}
            if row.get("quality_checks"):
                passed = [k for k, ok in row["quality_checks"].items() if ok and not k.endswith("_wrong")]
                failed = [k for k, ok in row["quality_checks"].items() if not ok and not k.endswith("_wrong")]
                wrong = [k for k, ok in row["quality_checks"].items() if k.endswith("_wrong") and ok]
                if passed:
                    meta_lines.append(f"- quality pass: {', '.join(passed)}")
                if failed:
                    meta_lines.append(f"- quality miss: {', '.join(failed)}")
                if wrong:
                    meta_lines.append(f"- **quality error detected:** {', '.join(wrong)}")
            if hits:
                found = [k for k, ok in hits.items() if ok]
                missing = [k for k, ok in hits.items() if not ok]
                if found:
                    meta_lines.append(f"- keywords found: {', '.join(found)}")
                if missing:
                    meta_lines.append(f"- keywords missing: {', '.join(missing)}")
            lines.extend(meta_lines)
            body_path = tdir / f"{vid}_{'gemini' if 'Gemini' in str(row.get('method')) else 'whisper' if 'Whisper' in str(row.get('method')) else 'youtube_captions'}.txt"
            _, body = read_transcript(body_path)
            lines.extend(["", "**Preview:**", "", _preview(body, 800), "", ""])

        lines.append("---")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
