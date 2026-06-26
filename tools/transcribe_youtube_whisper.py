"""
Transcribe YouTube video audio with faster-whisper (local GPU/CPU).

Optimized defaults: distil-large-v3, beam search, math-education initial prompt.

Usage:

    python tools/transcribe_youtube_whisper.py --urls-file data/transcripts/pilot_3_urls.txt
    python tools/transcribe_youtube_whisper.py --url https://www.youtube.com/watch?v=KTNcYYHuBTY --model distil-large-v3
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

import importlib.util

_spec = importlib.util.spec_from_file_location("transcript_prompts", _HERE.parent / "transcript_prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_prompts)
WHISPER_INITIAL_PROMPT = _prompts.WHISPER_INITIAL_PROMPT
PROMPT_VERSION = _prompts.PROMPT_VERSION


def _load_download_helper():
    import importlib.util as iu

    fp = _HERE.parent / "transcribe_youtube_gemini.py"
    spec = iu.spec_from_file_location("transcribe_youtube_gemini", fp)
    mod = iu.module_from_spec(spec)
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


def load_model(*, model_size: str = "distil-large-v3", device: str = "auto", compute_type: str = "auto"):
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
    model_size: str = "distil-large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    initial_prompt: str = WHISPER_INITIAL_PROMPT,
) -> tuple[str, list[dict], dict]:
    t0 = time.time()
    segments_iter, info = model.transcribe(
        str(media_path),
        language="en",
        task="transcribe",
        beam_size=beam_size,
        best_of=beam_size,
        patience=1.0,
        initial_prompt=initial_prompt,
        condition_on_previous_text=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
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
    duration_s = round(float(info.duration or 0.0), 2)
    meta = {
        "language": info.language,
        "duration_s": duration_s,
        "elapsed_s": elapsed,
        "real_time_factor": round(elapsed / duration_s, 3) if duration_s > 0 else None,
        "model": model_size,
        "device": device,
        "compute_type": compute_type,
        "beam_size": beam_size,
        "prompt_version": PROMPT_VERSION,
        "segment_count": len(segment_rows),
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
    wc = len(transcript.split())
    cc = len(transcript)
    media_bytes = media_path.stat().st_size if media_path.is_file() else 0
    wpm = round(wc / (meta["duration_s"] / 60.0), 1) if meta.get("duration_s") else ""
    header = (
        f"video_id: {vid}\n"
        f"url: {url}\n"
        f"method: faster_whisper\n"
        f"model: {meta.get('model')}\n"
        f"device: {meta.get('device')}\n"
        f"compute_type: {meta.get('compute_type')}\n"
        f"beam_size: {meta.get('beam_size')}\n"
        f"prompt_version: {meta.get('prompt_version')}\n"
        f"media_file: {media_path.name}\n"
        f"media_bytes: {media_bytes}\n"
        f"duration_s: {meta.get('duration_s')}\n"
        f"elapsed_s: {meta.get('elapsed_s')}\n"
        f"real_time_factor: {meta.get('real_time_factor')}\n"
        f"segment_count: {meta.get('segment_count')}\n"
        f"words_per_minute: {wpm}\n"
        f"word_count: {wc}\n"
        f"char_count: {cc}\n"
        f"{'=' * 72}\n\n"
    )
    txt_path.write_text(header + transcript + "\n", encoding="utf-8")
    payload = {
        "video_id": vid,
        "url": url,
        "transcript_path": str(txt_path),
        "media_file": media_path.name,
        "media_bytes": media_bytes,
        "transcript": transcript,
        "segments": segments,
        "words_per_minute": wpm if wpm != "" else None,
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
    parser.add_argument("--model", default="distil-large-v3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--beam-size", type=int, default=5)
    args = parser.parse_args()

    urls = list(args.url)
    if args.urls_file:
        urls.extend(_read_urls(Path(args.urls_file)))
    if not urls:
        parser.error("Provide --url and/or --urls-file")

    out_dir = Path(args.out_dir)
    audio_dir = Path(args.audio_dir)
    results: list[dict] = []

    print(f"Loading Whisper model {args.model} (beam_size={args.beam_size})...")
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
                beam_size=args.beam_size,
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
                f"{meta['elapsed_s']}s, {len(transcript.split())} words, "
                f"rtf={meta.get('real_time_factor')}, segments={meta.get('segment_count')}"
            )
            print(f"  wrote: {txt_path}")
            results.append({"video_id": vid, "ok": True, "path": str(txt_path), **meta})
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append({"video_id": vid, "ok": False, "error": str(e)})

    summary = out_dir / "whisper_run_summary.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary}")
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
