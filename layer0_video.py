"""
Layer 0v — video ingestion.

Two source types are supported:

* **YouTube URL.** Normalized to a canonical ``watch?v=<id>`` form and passed
  to Gemini as a ``FileData`` part. No download, no Files API upload.
* **Local video file.** Uploaded to the Gemini Files API and polled until it
  reaches the ``ACTIVE`` state. The returned ``File`` object can then be used
  directly as ``generate_content`` input.

The functions here are deliberately thin: no LLM call yet, only the plumbing
that lets Layer 3v talk about the video.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from google import genai

_log = logging.getLogger(__name__)


# Matches common YouTube URL shapes: /watch?v=, youtu.be/, /shorts/, /embed/.
_YT_ID_PATTERN = re.compile(
    r"(?:v=|/shorts/|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})"
)


def youtube_video_id(url: str) -> str | None:
    """Extract the 11-character YouTube video id from a URL (or return ``None``)."""
    if not url:
        return None
    m = _YT_ID_PATTERN.search(url)
    return m.group(1) if m else None


def canonical_youtube_url(url: str) -> str | None:
    """Rebuild the URL as ``https://www.youtube.com/watch?v=<id>``.

    Gemini rejects YouTube URLs that carry additional query parameters
    (``?app=desktop&v=...``), so the canonical form keeps only the video id.
    """
    vid = youtube_video_id(url)
    if not vid:
        return None
    return f"https://www.youtube.com/watch?v={vid}"


def _video_file_readable(path: Path) -> bool:
    """Return True if OpenCV can open the file and read at least one frame."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        cap.release()
        return bool(ok and frame is not None)
    except Exception:
        return False


def extract_frames(path: str | Path, *, interval_s: float = 15.0,
                    max_frames: int = 40,
                    jpeg_quality: int = 85) -> list[tuple[float, bytes]]:
    """Sample frames from a local video at roughly one per ``interval_s`` seconds.

    Returns a list of ``(timestamp_seconds, jpeg_bytes)`` tuples. Uses OpenCV so
    no separate ffmpeg binary is required. ``max_frames`` caps the number of
    samples for very long videos. The caller is responsible for deduplication;
    this function only hands back raw samples.
    """
    import cv2  # lazy import so the main pipeline does not pay the cost

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration_s = (total_frames / fps) if fps > 0 else 0.0

    targets: list[float] = []
    t = 0.0
    while True:
        if len(targets) >= max_frames:
            break
        if duration_s > 0 and t >= duration_s:
            break
        targets.append(t)
        t += interval_s

    out: list[tuple[float, bytes]] = []
    for ts in targets:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok_enc:
            out.append((round(ts, 1), bytes(buf)))

    cap.release()
    return out


def download_youtube_video(url: str, out_dir: str | Path, *,
                             max_height: int = 480) -> Path:
    """Download a YouTube video to ``out_dir`` using yt-dlp. Returns the path.

    The ``max_height`` cap keeps the file small and the download fast; 480p is
    already well above what frame OCR needs for on-screen LaTeX. Existing
    files with the same video id are reused so callers can use this as a
    simple disk cache. Socket timeouts and retries are raised for long
    videos whose CDN handshake can be slow.
    """
    import yt_dlp  # lazy import

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = youtube_video_id(url) or "video"

    # Reuse if already downloaded and the file decodes cleanly. Partial or
    # interrupted downloads reuse as-is would make OpenCV return zero frames.
    candidates = sorted(
        (
            p
            for p in out_dir.glob(f"{vid}.*")
            if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mkv", ".m4v"}
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for existing in candidates:
        if _video_file_readable(existing):
            _log.info(f"  [L0v] Reusing verified download {existing.name}")
            return existing
        try:
            existing.unlink(missing_ok=True)  # type: ignore[arg-type]
            _log.info(f"  [L0v] Removed unreadable cached file {existing.name}")
        except OSError:
            pass

    _log.info(f"  [L0v] Downloading YouTube video {vid} (max {max_height}p)...")

    # Format ladders, tried in order. YouTube has been serving format 18
    # (legacy 360p combined mp4) as a truncated stream for some videos: the
    # download "completes" but OpenCV can open the moov header yet decode
    # zero frames ("partial file"). The pipeline only needs frames (no
    # audio), so we prefer video-only adaptive streams (no merge required,
    # works even without ffmpeg) and fall back to combined legacy formats
    # only as a last resort.
    format_ladders = [
        # Adaptive video-only: no merge needed (works without ffmpeg).
        (
            f"bestvideo[ext=mp4][height<={max_height}]/"
            f"bestvideo[height<={max_height}]/"
            f"bestvideo[ext=mp4]/bestvideo"
        ),
        # Combined fallbacks (the path that fails for 7hCsQOKOYS8 today).
        (
            f"best[ext=mp4][height<={max_height}]/"
            f"best[height<={max_height}]/best"
        ),
    ]
    base_opts = {
        "outtmpl": str(out_dir / f"{vid}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 60,
        "retries": 5,
        "fragment_retries": 5,
        "file_access_retries": 5,
        "concurrent_fragment_downloads": 4,
        "merge_output_format": "mp4",
    }

    client_variants = (
        {},
        {"extractor_args": {"youtube": {"player_client": ["android"]}}},
        {"extractor_args": {"youtube": {"player_client": ["web", "ios", "android"]}}},
    )

    last_err: Exception | None = None
    attempt_ix = 0
    for client_opts in client_variants:
        client_label = (
            ",".join(client_opts.get("extractor_args", {}).get("youtube", {}).get("player_client", []))
            if client_opts else "default"
        )
        for fmt in format_ladders:
            attempt_ix += 1
            try:
                with yt_dlp.YoutubeDL({**base_opts, **client_opts, "format": fmt}) as ydl:
                    ydl.extract_info(url, download=True)
            except Exception as e:  # transient YouTube/yt-dlp errors
                last_err = e
                _log.info(
                    f"  [L0v] Download attempt {attempt_ix} ({client_label}) failed: {str(e)[:140]}"
                )
                continue

            produced_candidates = sorted(
                (
                    p
                    for p in out_dir.glob(f"{vid}.*")
                    if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mkv", ".m4v"}
                ),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for produced in produced_candidates:
                if _video_file_readable(produced):
                    _log.info(f"  [L0v] Download verified readable: {produced.name}")
                    return produced
                try:
                    produced.unlink(missing_ok=True)  # type: ignore[arg-type]
                except OSError:
                    pass
            _log.info(
                f"  [L0v] Attempt {attempt_ix} ({client_label}) produced an undecodable file; "
                "retrying with alternate format/client."
            )

    raise RuntimeError(
        f"yt-dlp did not produce a decodable video file for {vid} "
        "(file missing, corrupt, or open returned zero frames"
        + (f"; last error: {str(last_err)[:160]}" if last_err else "")
        + ")"
    )


def upload_local_video(client: genai.Client, path: str | Path,
                        *, poll_interval_s: float = 2.0,
                        timeout_s: float = 600.0) -> "genai.types.File":  # type: ignore[name-defined]
    """Upload a video to Gemini Files API and block until it is ``ACTIVE``.

    Raises ``RuntimeError`` if the upload fails or the file never becomes
    active within ``timeout_s``. Uses the provided ``client`` so callers can
    share one genai client across the request.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    _log.info(f"  [L0v] Uploading {path.name} to Gemini Files API...")
    uploaded = client.files.upload(file=str(path))
    t0 = time.time()
    while True:
        state = getattr(uploaded, "state", None)
        state_name = getattr(state, "name", str(state)) if state else ""
        if state_name == "ACTIVE":
            return uploaded
        if state_name == "FAILED":
            raise RuntimeError(f"Gemini video upload failed for {path.name}")
        if time.time() - t0 > timeout_s:
            raise RuntimeError(
                f"Gemini video upload did not become ACTIVE within {timeout_s:.0f}s "
                f"(last state: {state_name or 'unknown'})"
            )
        time.sleep(poll_interval_s)
        try:
            uploaded = client.files.get(name=uploaded.name)
        except Exception as e:
            raise RuntimeError(f"Failed to poll Gemini file status: {e}") from e
