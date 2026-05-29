"""
Transcribe YouTube video audio with faster-whisper (local GPU/CPU).

Reuses media downloads from transcribe_youtube_gemini.py when present.

Usage:

    python tools/transcribe_youtube_whisper.py --urls-file data/transcripts/pilot_3_urls.txt
    python tools/transcribe_youtube_whisper.py --url https://www.youtube.com/watch?v=KTNcYYHuBTY --model medium.en
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from layer0_video import canonical_youtube_url, youtube_video_id  # noqa: E402


def _load_download_helper():
    import importlib.util

    fp = _HERE.parent / "transcribe_youtube_gemini.py"
    spec = importlib.util.spec_from_file_location("transcribe_youtube_gemini", fp)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.download_youtube_audio


download_youtube_audio = _load_download_helper()


def _read_urls(path: Path) -> list[str]:
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def load_model(*, model_size: str = "medium.en", device: str = "auto", compute_type: str = "auto"):
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type), device, compute_type


def transcribe_file(
    media_path: Path,
    model,
    *,
    model_size: str = "medium.en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> tuple[str, list[dict], dict]:
    t0 = time.time()
    segments_iter, info = model.transcribe(
        str(media_path),
        language="en",
        vad_filter=True,
        word_timestamps=False,
    )
    segment_rows: list[dict] = []
    parts: list[str] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segment_rows.append({
            "start_s": round(float(seg.start), 2),
            "end_s": round(float(seg.end), 2),
            "text": text,
        })
        parts.append(text)
    elapsed = round(time.time() - t0, 1)
    meta = {
        "language": info.language,
        "duration_s": round(float(info.duration or 0.0), 2),
        "elapsed_s": elapsed,
        "model": model_size,
        "device": device,
        "compute_type": compute_type,
    }
    return " ".join(parts), segment_rows, meta


def _write_outputs(
    *,
    out_dir: Path,
    vid: str,
    url: str,
    transcript: str,
    segments: list[dict],
    meta: dict,
    media_path: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{vid}_whisper.txt"
    header = (
        f"video_id: {vid}\n"
        f"url: {url}\n"
        f"method: faster_whisper\n"
        f"model: {meta.get('model')}\n"
        f"device: {meta.get('device')}\n"
        f"media_file: {media_path.name}\n"
        f"duration_s: {meta.get('duration_s')}\n"
        f"elapsed_s: {meta.get('elapsed_s')}\n"
        f"word_count: {len(transcript.split())}\n"
        f"char_count: {len(transcript)}\n"
        f"{'=' * 72}\n\n"
    )
    txt_path.write_text(header + transcript + "\n", encoding="utf-8")
    payload = {
        "video_id": vid,
        "url": url,
        "transcript_path": str(txt_path),
        "media_file": media_path.name,
        "transcript": transcript,
        "segments": segments,
        **meta,
    }
    (out_dir / f"{vid}_whisper.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return txt_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Whisper transcription for YouTube videos")
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--urls-file")
    parser.add_argument("--out-dir", default=str(_ROOT / "data" / "transcripts"))
    parser.add_argument("--audio-dir", default=str(_ROOT / "data" / "transcripts" / "audio"))
    parser.add_argument("--model", default="medium.en")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    args = parser.parse_args()

    urls = list(args.url)
    if args.urls_file:
        urls.extend(_read_urls(Path(args.urls_file)))
    if not urls:
        parser.error("Provide --url and/or --urls-file")

    out_dir = Path(args.out_dir)
    audio_dir = Path(args.audio_dir)
    results: list[dict] = []

    print(f"Loading Whisper model {args.model}...")
    model, device, compute_type = load_model(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    print(f"  device={device} compute_type={compute_type}")

    for raw_url in urls:
        canon = canonical_youtube_url(raw_url)
        if not canon:
            print(f"SKIP invalid URL: {raw_url}")
            continue
        vid = youtube_video_id(canon) or "unknown"
        print(f"\n=== {vid} (whisper) ===")
        try:
            media_path = download_youtube_audio(canon, audio_dir)
            transcript, segments, meta = transcribe_file(
                media_path,
                model,
                model_size=args.model,
                device=device,
                compute_type=compute_type,
            )
            txt_path = _write_outputs(
                out_dir=out_dir,
                vid=vid,
                url=canon,
                transcript=transcript,
                segments=segments,
                meta=meta,
                media_path=media_path,
            )
            print(
                f"  {meta['model']} on {meta['device']} — "
                f"{meta['elapsed_s']}s, {len(transcript.split())} words"
            )
            print(f"  wrote: {txt_path}")
            results.append({"video_id": vid, "ok": True, "path": str(txt_path)})
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append({"video_id": vid, "ok": False, "error": str(e)})

    summary = out_dir / "whisper_run_summary.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary}")
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
