"""
Transcribe YouTube video audio to plain text with Gemini.

Downloads audio via yt-dlp (no ffmpeg required — keeps native m4a/webm),
uploads to the Gemini Files API, and asks for a verbatim English transcript
with mathematical terms preserved.

Usage (from Step_Project root):

    python tools/transcribe_youtube_gemini.py \\
        --url https://www.youtube.com/watch?v=KTNcYYHuBTY \\
        --out-dir data/transcripts

    python tools/transcribe_youtube_gemini.py --urls-file data/transcripts/pilot_3_urls.txt
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

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from config import GEMINI_API_KEY, GEMINI_FALLBACK_MODEL, GEMINI_MODEL  # noqa: E402
from layer0_video import canonical_youtube_url, upload_local_video, youtube_video_id  # noqa: E402

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "transcript_prompts", _HERE.parent / "transcript_prompts.py"
)
_prompts = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_prompts)
get_prompt_profile = _prompts.get_prompt_profile
SYSTEM_INSTRUCTION, USER_PROMPT, PROMPT_VERSION = get_prompt_profile("readable")

GEMINI_MAX_OUTPUT_TOKENS = 16384
GEMINI_TEMPERATURE = 0.0

_MEDIA_SUFFIXES = {".m4a", ".mp3", ".opus", ".webm", ".ogg", ".wav", ".mp4", ".mkv", ".m4v"}


def _client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=GEMINI_API_KEY)


def _mime_for(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".opus": "audio/opus",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".m4v": "video/x-m4v",
    }.get(ext, "audio/mpeg")


def download_youtube_audio(url: str, out_dir: Path) -> Path:
    """Download a compact YouTube stream suitable for Gemini transcription.

    Prefers audio-only m4a when available; otherwise falls back to the same
    combined mp4 ladder used elsewhere in the project (format 18 / adaptive).
    """
    import yt_dlp

    canon = canonical_youtube_url(url)
    if not canon:
        raise ValueError(f"Not a valid YouTube URL: {url!r}")
    vid = youtube_video_id(canon) or "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    for existing in sorted(out_dir.glob(f"{vid}.*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if existing.is_file() and existing.suffix.lower() in _MEDIA_SUFFIXES:
            print(f"  reuse media: {existing.name}")
            return existing

    print(f"  downloading media for {vid}...")
    format_ladders = [
        "bestaudio[ext=m4a]/bestaudio/best",
        (
            "best[ext=mp4][height<=480]/"
            "best[height<=480]/best"
        ),
    ]
    base_opts = {
        "outtmpl": str(out_dir / f"{vid}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 60,
        "retries": 5,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    last_err: Exception | None = None
    for fmt in format_ladders:
        try:
            with yt_dlp.YoutubeDL({**base_opts, "format": fmt}) as ydl:
                ydl.download([canon])
            produced = sorted(
                (
                    p
                    for p in out_dir.glob(f"{vid}.*")
                    if p.is_file() and p.suffix.lower() in _MEDIA_SUFFIXES
                ),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if produced:
                print(f"  saved: {produced[0].name} ({produced[0].stat().st_size // 1024} KB)")
                return produced[0]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"yt-dlp did not produce a media file for {vid}"
        + (f"; last error: {last_err}" if last_err else "")
    )


def transcribe_youtube_url(
    url: str,
    *,
    model: str | None = None,
) -> tuple[str, str, float]:
    """Transcribe via YouTube file_uri (no local download/upload)."""
    canon = canonical_youtube_url(url)
    if not canon:
        raise ValueError(f"Not a valid YouTube URL: {url!r}")
    client = _client()
    candidates = []
    for m in (model, GEMINI_FALLBACK_MODEL, GEMINI_MODEL, "gemini-2.5-flash", "gemini-2.5-pro"):
        m = (m or "").strip()
        if m and m not in candidates:
            candidates.append(m)
    part_video = types.Part(file_data=types.FileData(file_uri=canon, mime_type="video/*"))
    part_prompt = types.Part(text=USER_PROMPT)
    cfg = _generation_config()
    last_err: Exception | None = None
    t0 = time.time()
    for used in candidates:
        for retry in range(3):
            try:
                response = client.models.generate_content(
                    model=used,
                    contents=types.Content(role="user", parts=[part_video, part_prompt]),
                    config=cfg,
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned an empty transcript")
                return text, used, round(time.time() - t0, 1)
            except Exception as e:
                last_err = e
                msg = str(e)
                if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                    time.sleep(5 * (retry + 1))
                    continue
                break
    raise RuntimeError(f"YouTube-uri transcription failed: {last_err}")


def _generation_config() -> types.GenerateContentConfig:
    kwargs: dict = {
        "system_instruction": SYSTEM_INSTRUCTION,
        "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
        "temperature": GEMINI_TEMPERATURE,
    }
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass
    return types.GenerateContentConfig(**kwargs)


def transcribe_audio_file(
    audio_path: Path,
    *,
    model: str | None = None,
    fallback: str | None = None,
    primary_only: bool = False,
) -> tuple[str, str, float]:
    """Upload media to Gemini and return (transcript, model_used, elapsed_s)."""
    client = _client()
    uploaded = upload_local_video(client, audio_path)
    primary = (model or "gemini-3-flash-preview").strip()
    if primary_only:
        candidates = [primary]
    else:
        candidates = []
        for m in (
            model,
            fallback,
            "gemini-3-flash-preview",
            "gemini-3.5-flash",
            GEMINI_MODEL,
            GEMINI_FALLBACK_MODEL,
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ):
            m = (m or "").strip()
            if m and m not in candidates:
                candidates.append(m)
    cfg = _generation_config()
    last_err: Exception | None = None
    t0 = time.time()
    for attempt, used in enumerate(candidates, start=1):
        for retry in range(3):
            try:
                response = client.models.generate_content(
                    model=used,
                    contents=[uploaded, USER_PROMPT],
                    config=cfg,
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned an empty transcript")
                elapsed = round(time.time() - t0, 1)
                return text, used, elapsed
            except Exception as e:
                last_err = e
                msg = str(e)
                if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                    wait = 5 * (retry + 1)
                    print(f"  retry {retry + 1}/3 on {used} after {wait}s ({msg[:80]})")
                    time.sleep(wait)
                    continue
                break
        print(f"  model failed: {used} ({last_err})")
    raise RuntimeError(f"All Gemini models failed for transcription: {last_err}")


def _probe_duration_s(media_path: Path) -> float | None:
    if not media_path.is_file():
        return None
    try:
        import av

        with av.open(str(media_path)) as container:
            if container.duration:
                return round(container.duration / 1_000_000.0, 2)
            stream = container.streams.audio[0] if container.streams.audio else None
            if stream and stream.duration and stream.time_base:
                return round(float(stream.duration * stream.time_base), 2)
    except Exception:
        pass
    try:
        from faster_whisper import decode_audio

        audio = decode_audio(str(media_path), sampling_rate=16000)
        return round(len(audio) / 16000.0, 2)
    except Exception:
        return None


def _load_title_hints(gold_path: Path | None = None) -> dict[str, str]:
    path = gold_path or (_ROOT / "data" / "gold" / "videos_15_graded_vlm_reference_nocache.json")
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            str(it["id"]): str(it.get("title") or "")
            for it in (data.get("items") or [])
            if isinstance(it, dict) and it.get("id")
        }
    except Exception:
        return {}


def _read_urls(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _write_outputs(
    *,
    out_dir: Path,
    vid: str,
    url: str,
    title_hint: str,
    transcript: str,
    model_used: str,
    elapsed_s: float,
    audio_path: Path,
    method: str,
    duration_s: float | None = None,
    primary_model: str = "",
    fallback_used: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{vid}_gemini.txt"
    wc = len(transcript.split())
    cc = len(transcript)
    media_bytes = audio_path.stat().st_size if audio_path.is_file() else 0
    rtf = round(elapsed_s / duration_s, 3) if duration_s and duration_s > 0 else ""
    header = (
        f"video_id: {vid}\n"
        f"url: {url}\n"
        f"title_hint: {title_hint}\n"
        f"method: {method}\n"
        f"model: {model_used}\n"
        f"primary_model: {primary_model or model_used}\n"
        f"fallback_used: {str(fallback_used).lower()}\n"
        f"prompt_version: {PROMPT_VERSION}\n"
        f"temperature: {GEMINI_TEMPERATURE}\n"
        f"max_output_tokens: {GEMINI_MAX_OUTPUT_TOKENS}\n"
        f"audio_file: {audio_path.name}\n"
        f"media_bytes: {media_bytes}\n"
        f"duration_s: {duration_s if duration_s is not None else ''}\n"
        f"elapsed_s: {elapsed_s}\n"
        f"real_time_factor: {rtf}\n"
        f"word_count: {wc}\n"
        f"char_count: {cc}\n"
        f"{'=' * 72}\n\n"
    )
    txt_path.write_text(header + transcript + "\n", encoding="utf-8")

    meta = {
        "video_id": vid,
        "url": url,
        "title_hint": title_hint,
        "method": method,
        "model_used": model_used,
        "primary_model": primary_model or model_used,
        "fallback_used": fallback_used,
        "prompt_version": PROMPT_VERSION,
        "temperature": GEMINI_TEMPERATURE,
        "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
        "elapsed_s": elapsed_s,
        "duration_s": duration_s,
        "real_time_factor": rtf if rtf != "" else None,
        "audio_file": audio_path.name,
        "media_bytes": media_bytes,
        "word_count": wc,
        "char_count": cc,
        "transcript_path": str(txt_path),
    }
    (out_dir / f"{vid}_gemini.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return txt_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini audio transcription for YouTube videos")
    parser.add_argument("--url", action="append", default=[], help="YouTube URL (repeatable)")
    parser.add_argument("--urls-file", help="Text file with one URL per line")
    parser.add_argument("--out-dir", default=str(_ROOT / "data" / "transcripts"))
    parser.add_argument("--audio-dir", default=str(_ROOT / "data" / "transcripts" / "audio"))
    parser.add_argument(
        "--mode",
        choices=("auto", "youtube-uri", "upload"),
        default="upload",
        help="auto tries youtube-uri first, then local media upload (upload recommended)",
    )
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model (default: gemini-3-flash-preview)")
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Use only --model (no fallback chain); fail if primary returns empty/errors",
    )
    parser.add_argument("--title-hint", default="", help="Optional title for metadata")
    parser.add_argument(
        "--prompt-profile",
        choices=("readable", "latex"),
        default="readable",
        help="readable = plain English for humans (default); latex = inline LaTeX formulas",
    )
    args = parser.parse_args()

    global SYSTEM_INSTRUCTION, USER_PROMPT, PROMPT_VERSION
    SYSTEM_INSTRUCTION, USER_PROMPT, PROMPT_VERSION = get_prompt_profile(args.prompt_profile)
    print(f"Prompt profile: {args.prompt_profile} ({PROMPT_VERSION})")

    urls: list[str] = list(args.url)
    if args.urls_file:
        urls.extend(_read_urls(Path(args.urls_file)))
    if not urls:
        parser.error("Provide --url and/or --urls-file")

    out_dir = Path(args.out_dir)
    audio_dir = Path(args.audio_dir)
    title_hints = _load_title_hints()
    results: list[dict] = []

    for raw_url in urls:
        canon = canonical_youtube_url(raw_url)
        if not canon:
            print(f"SKIP invalid URL: {raw_url}")
            continue
        vid = youtube_video_id(canon) or "unknown"
        title_hint = args.title_hint or title_hints.get(vid, "")
        print(f"\n=== {vid} ===")
        try:
            transcript = model_used = ""
            elapsed_s = 0.0
            audio_path: Path | None = None
            method = "gemini_youtube_uri_transcription"

            if args.mode in ("auto", "youtube-uri"):
                try:
                    print("  trying YouTube URI path...")
                    transcript, model_used, elapsed_s = transcribe_youtube_url(
                        canon,
                        model=args.model,
                    )
                except Exception as e:
                    print(f"  YouTube URI failed: {e}")
                    if args.mode == "youtube-uri":
                        raise

            if not transcript and args.mode in ("auto", "upload"):
                audio_path = download_youtube_audio(canon, audio_dir)
                transcript, model_used, elapsed_s = transcribe_audio_file(
                    audio_path,
                    model=args.model,
                    primary_only=args.primary_only,
                )
                method = "gemini_upload_transcription"

            if not audio_path:
                audio_path = audio_dir / f"{vid}.uri"
            duration_s = _probe_duration_s(audio_path)
            primary_model = (args.model or "gemini-3-flash-preview").strip()
            fallback_used = model_used != primary_model
            txt_path = _write_outputs(
                out_dir=out_dir,
                vid=vid,
                url=canon,
                title_hint=title_hint,
                transcript=transcript,
                model_used=model_used,
                elapsed_s=elapsed_s,
                audio_path=audio_path,
                method=method,
                duration_s=duration_s,
                primary_model=primary_model,
                fallback_used=fallback_used,
            )
            print(f"  model: {model_used} ({elapsed_s}s)")
            print(f"  words: {len(transcript.split())}  chars: {len(transcript)}")
            print(f"  wrote: {txt_path}")
            preview = transcript[:400].replace("\n", " ")
            print(f"  preview: {preview}...")
            results.append({"video_id": vid, "ok": True, "path": str(txt_path)})
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append({"video_id": vid, "ok": False, "error": str(e)})

    summary_path = out_dir / "transcription_run_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}")
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
