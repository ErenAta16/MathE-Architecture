"""
STEP Pipeline — Web UI.

Flask backend with Server-Sent Events (SSE) for live log lines while a PDF is
solved. Solver output uses the same ``logging`` format as the CLI so
``_classify`` can tag ``[L0]`` … ``[L6]`` lines for the front end.

Run:
    python web_app.py
    Open http://127.0.0.1:5000
"""

import logging
import os
import threading
import queue
import sys
import time
import json
import re
import webbrowser
from pathlib import Path

# UI tests should measure the real end-to-end runtime; do not persist or reuse
# video-analysis artifacts from the web app.
os.environ.setdefault("STEP_DISABLE_VIDEO_CACHE", "1")

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    send_from_directory,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename

# Windows consoles often default to cp1252; force UTF-8 so box-drawing / checkmarks don't crash.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

app = Flask(__name__)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
VIDEO_UPLOAD_DIR = Path(__file__).parent / "uploads_video"
VIDEO_UPLOAD_DIR.mkdir(exist_ok=True)

# Keep uploads reasonable; Gemini's own limit is 2GB.
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB

_tasks: dict = {}
_tasks_lock = threading.Lock()
# Drop finished tasks after this many seconds (SSE client usually attaches immediately).
_TASK_RETENTION_SEC = 15 * 60
# Hard cap so _tasks cannot grow without bound if retention is misconfigured.
_TASK_MAX = 256

_thread_local = threading.local()
_original_stdout = sys.stdout
# Allow a few concurrent pipeline runs (API-bound). With Nougat/GPU, set
# STEP_WEB_MAX_CONCURRENT_SOLVES=1 to avoid CUDA OOM from overlapping inference.
try:
    _MAX_CONCURRENT_SOLVES = max(1, int(os.environ.get("STEP_WEB_MAX_CONCURRENT_SOLVES", "2")))
except ValueError:
    _MAX_CONCURRENT_SOLVES = 2
_solve_semaphore = threading.BoundedSemaphore(_MAX_CONCURRENT_SOLVES)

_WEB_USE_NOUGAT = os.environ.get("STEP_WEB_USE_NOUGAT", "").strip().lower() in ("1", "true", "yes")
_WEB_USE_VLM = os.environ.get("STEP_WEB_USE_VLM", "1").strip().lower() not in ("0", "false", "no")

# One ``STEPSolver`` is reused across requests so API clients (Gemini, Together,
# VLM) and the per-solver result cache are shared. Building a new solver per
# upload paid for repeated client handshakes and discarded every cache hit.
_shared_solver = None
_shared_solver_lock = threading.Lock()


def _get_shared_solver():
    """Lazily build one process-wide ``STEPSolver`` and return it."""
    global _shared_solver
    if _shared_solver is None:
        with _shared_solver_lock:
            if _shared_solver is None:
                from run import STEPSolver
                _shared_solver = STEPSolver(
                    use_nougat=_WEB_USE_NOUGAT,
                    use_vlm=_WEB_USE_VLM,
                )
    return _shared_solver


_shared_video_analyzer = None
_shared_video_lock = threading.Lock()


def _get_shared_video_analyzer():
    """Lazily build one process-wide ``VideoAnalyzer`` (shares the disk cache)."""
    global _shared_video_analyzer
    if _shared_video_analyzer is None:
        with _shared_video_lock:
            if _shared_video_analyzer is None:
                from run_video import VideoAnalyzer
                _shared_video_analyzer = VideoAnalyzer()
    return _shared_video_analyzer


def _short_problem_text(prompt: str) -> str:
    """Best short form of the problem statement for downstream tooling.

    Prefers the cleaned Layer 4 prompt section that directly describes the
    problem. Falls back to the original prompt the solver received.
    """
    if not prompt:
        return ""
    marker = "--- SOURCE"
    if marker in prompt:
        # Take the header plus the first source block; keeps the model focused
        # on the statement without the supplementary notes.
        head, _, rest = prompt.partition(marker)
        first_source = marker + rest.split("--- SOURCE", 1)[0]
        return (head + first_source).strip()
    # No multi-source block: the full prompt already reads as one statement.
    return prompt.strip()


def _prune_tasks() -> None:
    """Remove completed tasks past TTL and enforce a max entry count."""
    now = time.time()
    with _tasks_lock:
        for tid in list(_tasks.keys()):
            t = _tasks[tid]
            if not t.get("done"):
                continue
            finished = t.get("finished_at")
            if finished is None:
                continue
            if now - finished > _TASK_RETENTION_SEC:
                _tasks.pop(tid, None)

        if len(_tasks) <= _TASK_MAX:
            return
        done = [
            (tid, t["finished_at"])
            for tid, t in _tasks.items()
            if t.get("done") and t.get("finished_at") is not None
        ]
        done.sort(key=lambda x: x[1])
        for tid, _ in done[: max(0, len(_tasks) - _TASK_MAX)]:
            _tasks.pop(tid, None)


# ---------------------------------------------------------------------------
# Thread-safe stdout capture → SSE queue
# ---------------------------------------------------------------------------
class _SmartStdout:
    """ Tee stdout: mirror to the real console and buffer line-based events for SSE.

    When ``_thread_local.queue`` is set (worker thread), complete lines are
    classified and pushed for the event stream. ``logging`` uses the same
    ``sys.stdout``, so log records follow the same path as legacy prints.
    """

    def write(self, text):
        try:
            _original_stdout.write(text)
        except UnicodeEncodeError:
            enc = getattr(_original_stdout, "encoding", None) or "utf-8"
            try:
                _original_stdout.buffer.write(text.encode(enc, errors="replace"))
            except Exception:
                _original_stdout.write(text.encode("ascii", errors="replace").decode("ascii"))
        q = getattr(_thread_local, "queue", None)
        if q is not None:
            buf = getattr(_thread_local, "buf", "")
            buf += text
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                s = line.strip()
                if s:
                    q.put(_classify(s))
            _thread_local.buf = buf

    def flush(self):
        _original_stdout.flush()
        q = getattr(_thread_local, "queue", None)
        if q:
            buf = getattr(_thread_local, "buf", "")
            if buf.strip():
                q.put(_classify(buf.strip()))
            _thread_local.buf = ""

    # File-like protocol: many libraries (tqdm, huggingface_hub, sentence-transformers,
    # torch) probe these on sys.stdout. Missing any one of them raises AttributeError
    # mid-pipeline, which previously broke every [SIM] call.
    def isatty(self):
        try:
            return bool(_original_stdout.isatty())
        except Exception:
            return False

    def fileno(self):
        return _original_stdout.fileno()

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    @property
    def encoding(self):
        return getattr(_original_stdout, "encoding", "utf-8") or "utf-8"

    @property
    def errors(self):
        return getattr(_original_stdout, "errors", "replace")

    @property
    def closed(self):
        return getattr(_original_stdout, "closed", False)

    @property
    def buffer(self):
        return getattr(_original_stdout, "buffer", None)

    def __getattr__(self, name):
        # Last-resort proxy for any other attribute/method libraries expect.
        return getattr(_original_stdout, name)


sys.stdout = _SmartStdout()

# Import after stdout swap so ``configure_logging`` attaches to ``_SmartStdout``.
from config import ensure_dirs
from step_logging import configure_logging

ensure_dirs()
configure_logging()

_log = logging.getLogger(__name__)


@app.after_request
def _no_cache_for_ui(response):
    """Avoid stale HTML/JS when the template was updated (browser cache)."""
    response.headers.setdefault("Cache-Control", "no-store, max-age=0, must-revalidate, private")
    response.headers.setdefault("Pragma", "no-cache")
    response.headers.setdefault("Expires", "0")
    return response


def _classify(line: str) -> dict:
    """Classify a log line into a typed SSE event."""
    m = re.match(r"\[L(\d+b?)\]\s*(.*)", line)
    if m:
        return {"type": "layer", "layer": m.group(1), "detail": m.group(2)}
    if "LLM:" in line or "VLM" in line:
        return {"type": "config", "text": line}
    if "Consensus" in line or "[OK]" in line:
        return {"type": "success", "text": line}
    if (
        "ERR" in line
        or "HATA" in line
        or "[FAIL]" in line
        or line.strip().startswith("FAIL ")
    ):
        return {"type": "error", "text": line}
    if "[WARN]" in line or "[RETRY]" in line:
        return {"type": "warning", "text": line}
    return {"type": "log", "text": line}


def _strip_ui_hidden_fields(value):
    """Remove analysis-only fields that should never be rendered in the UI."""
    if isinstance(value, dict):
        return {
            k: _strip_ui_hidden_fields(v)
            for k, v in value.items()
            if k != "pseudo_gold_eval"
        }
    if isinstance(value, list):
        return [_strip_ui_hidden_fields(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/index")
@app.route("/index.html")
@app.route("/home")
def redirect_to_index():
    """Yanlis veya eski URL'ler ana sayfaya yonlensin."""
    return redirect(url_for("index"), code=302)


@app.route("/", strict_slashes=False)
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET"])
def upload_get_redirect():
    """GET /upload (bookmark) -> ana sayfa; yukleme sadece POST."""
    return redirect(url_for("index"), code=302)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("pdf")
    if not f or not f.filename:
        return jsonify(error="PDF file required"), 400

    safe = secure_filename(f.filename)
    if not safe:
        return jsonify(error="Invalid filename"), 400

    _prune_tasks()
    with _tasks_lock:
        for t in _tasks.values():
            if not t.get("done") and t.get("filename") == safe:
                return (
                    jsonify(
                        error="This file is already being processed",
                        detail=safe,
                    ),
                    409,
                )

    fp = UPLOAD_DIR / safe
    f.save(fp)

    # Optional free-form instruction to the solver LLM. Kept short so prompt
    # budgets stay predictable; longer notes are truncated rather than rejected.
    user_query = (request.form.get("user_query") or "").strip()
    if len(user_query) > 600:
        user_query = user_query[:600].rstrip() + "..."

    tid = f"t{int(time.time() * 1000)}"
    q = queue.Queue()
    with _tasks_lock:
        _tasks[tid] = {
            "queue": q,
            "done": False,
            "result": None,
            "finished_at": None,
            "filename": safe,
        }

    threading.Thread(target=_worker, args=(tid, fp, user_query), daemon=True).start()
    return jsonify(task_id=tid, filename=safe)


@app.route("/stream/<tid>")
def stream(tid):
    _prune_tasks()
    with _tasks_lock:
        task = _tasks.get(tid)
    if not task:
        return jsonify(error="not found"), 404

    def gen():
        while True:
            try:
                msg = task["queue"].get(timeout=180)
                if msg is None:
                    break
                yield f"data: {json.dumps(msg, ensure_ascii=False, default=str)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/pdf/<path:name>")
def serve_pdf(name):
    return send_from_directory(UPLOAD_DIR, name)


@app.route("/video/<path:name>")
def serve_video(name):
    return send_from_directory(VIDEO_UPLOAD_DIR, name)


@app.route("/analyze-video", methods=["POST"])
def analyze_video():
    """Kick off a video analysis: YouTube URL or uploaded file."""
    url = (request.form.get("youtube_url") or "").strip()
    upload = request.files.get("video")
    mode = (request.form.get("mode") or "quick").strip().lower()
    if mode not in ("quick", "deep"):
        mode = "quick"

    if not url and not upload:
        return jsonify(error="Provide youtube_url or a video file"), 400

    tid = f"v{int(time.time() * 1000)}"
    q = queue.Queue()
    with _tasks_lock:
        _tasks[tid] = {
            "queue": q,
            "done": False,
            "result": None,
            "finished_at": None,
            "media": "video",
        }

    if url:
        threading.Thread(
            target=_video_worker,
            args=(tid, "youtube", url, None, mode),
            daemon=True,
        ).start()
        return jsonify(task_id=tid, source="youtube", url=url, mode=mode)

    safe = secure_filename(upload.filename or "video.mp4")
    if not safe:
        with _tasks_lock:
            _tasks.pop(tid, None)
        return jsonify(error="Invalid filename"), 400
    fp = VIDEO_UPLOAD_DIR / safe
    upload.save(fp)
    with _tasks_lock:
        _tasks[tid]["filename"] = safe

    threading.Thread(
        target=_video_worker,
        args=(tid, "upload", None, fp, mode),
        daemon=True,
    ).start()
    return jsonify(task_id=tid, source="upload", filename=safe, mode=mode)


@app.route("/keyword-eval", methods=["POST"])
def keyword_eval():
    """Run Task 1 (free keywords) and Task 2 (closed pool) for a solved task.

    Body accepts ``task_id`` (required), optional ``pool`` (list of strings or
    comma-separated string) and optional ``model`` override (e.g.
    ``gemini-2.5-pro``). Returns JSON with ``task1``, ``task2``, ``model_used``,
    ``elapsed_s`` and the pool that was used.
    """
    from keyword_eval import evaluate_keywords, DEFAULT_KEYWORD_POOL

    payload = request.get_json(silent=True) or request.form
    tid = (payload.get("task_id") or "").strip()
    if not tid:
        return jsonify(error="task_id is required"), 400

    with _tasks_lock:
        task = _tasks.get(tid)
    if task is None:
        return jsonify(error="Unknown task_id"), 404
    if not task.get("done"):
        return jsonify(error="The solve is still running"), 409

    ctx = task.get("followup_context") or {}
    problem_text = (ctx.get("problem_text") or ctx.get("prompt") or "").strip()
    if not problem_text:
        return jsonify(error="No problem text stored for this task"), 409

    pool_raw = payload.get("pool")
    pool: list[str] | None
    if isinstance(pool_raw, list):
        pool = [str(x).strip() for x in pool_raw if str(x).strip()]
    elif isinstance(pool_raw, str) and pool_raw.strip():
        pool = [s.strip() for s in pool_raw.split(",") if s.strip()]
    else:
        pool = list(DEFAULT_KEYWORD_POOL)

    model = (payload.get("model") or "").strip() or None

    try:
        result = evaluate_keywords(problem_text, pool=pool, model=model)
    except Exception as e:
        return jsonify(error=str(e)[:200]), 500

    if result.get("error"):
        return jsonify(error=result["error"]), 500

    # Cache the last run on the task so the UI can fetch it on page reload.
    task["keyword_eval"] = result
    return jsonify(result)


@app.route("/followup", methods=["POST"])
def followup():
    """Ask a follow-up about an already-solved task.

    Runs one LLM round using the same shared solver and returns JSON.
    Kept synchronous because follow-ups are single passes (no OCR, no consensus).
    """
    payload = request.get_json(silent=True) or request.form
    tid = (payload.get("task_id") or "").strip()
    user_query = (payload.get("user_query") or "").strip()

    if not tid or not user_query:
        return jsonify(error="task_id and user_query are required"), 400
    if len(user_query) > 600:
        user_query = user_query[:600].rstrip() + "..."

    with _tasks_lock:
        task = _tasks.get(tid)
    if task is None:
        return jsonify(error="Unknown task_id"), 404
    if not task.get("done"):
        return jsonify(error="The solve is still running"), 409
    ctx = task.get("followup_context") or {}
    if not ctx.get("prompt") or not ctx.get("solution"):
        return jsonify(error="No follow-up context for this task"), 409

    try:
        solver = _get_shared_solver()
        reply = solver.ask_followup(
            prompt=ctx.get("prompt", ""),
            prior_solution=ctx.get("solution", ""),
            system_prompt=ctx.get("system_prompt", "") or None,
            user_query=user_query,
        )
    except Exception as e:
        return jsonify(error=str(e)[:200]), 500

    if reply.get("error"):
        return jsonify(error=reply["error"]), 500

    return jsonify(reply)


# ---------------------------------------------------------------------------
# Background solver
# ---------------------------------------------------------------------------
def _worker(tid: str, filepath: Path, user_query: str = ""):
    """Solve one uploaded PDF (bounded by ``_solve_semaphore``); push log lines to the task queue."""
    with _tasks_lock:
        task = _tasks.get(tid)
    if task is None:
        return
    q = task["queue"]
    _thread_local.queue = q
    _thread_local.buf = ""

    with _solve_semaphore:
        try:
            ensure_dirs()
            solver = _get_shared_solver()
            result = solver.solve(filepath, verbose=True, user_query=user_query or None)
            sys.stdout.flush()

            # Follow-up context lives server-side only. Read the keys without
            # mutating ``result`` — it is the same dict the solver caches, so
            # popping would erase the prompt from subsequent cache hits.
            full_prompt = result.get("_prompt", "") if isinstance(result, dict) else ""
            full_system = result.get("_system_prompt", "") if isinstance(result, dict) else ""
            follow_ctx = {
                "prompt": full_prompt,
                "system_prompt": full_system,
                "solution": result.get("solution", "") if isinstance(result, dict) else "",
                # Best-effort short problem text for downstream tools (e.g. the
                # keyword evaluator) that do not need the full L4 prompt wrapper.
                "problem_text": _short_problem_text(full_prompt),
            }

            # Build the payload for the UI, filtering server-only keys.
            clean = {}
            for k, v in result.items():
                if isinstance(k, str) and k.startswith("_"):
                    continue
                if isinstance(v, Path):
                    clean[k] = str(v)
                elif isinstance(v, list):
                    clean[k] = [str(x) for x in v]
                else:
                    clean[k] = v

            q.put({"type": "done", "data": clean})
            task["result"] = clean
            task["followup_context"] = follow_ctx

        except Exception as e:
            sys.stdout.flush()
            q.put({"type": "error", "text": str(e)})

        finally:
            _thread_local.queue = None
            q.put(None)
            with _tasks_lock:
                task["done"] = True
                task["finished_at"] = time.time()


def _video_worker(tid: str, kind: str, url: str | None, filepath: Path | None,
                   mode: str = "quick"):
    """Run a single video analysis (YouTube URL or uploaded file)."""
    with _tasks_lock:
        task = _tasks.get(tid)
    if task is None:
        return
    q = task["queue"]
    _thread_local.queue = q
    _thread_local.buf = ""

    with _solve_semaphore:
        try:
            ensure_dirs()
            analyzer = _get_shared_video_analyzer()
            if kind == "youtube":
                result = analyzer.analyze_youtube(url or "", mode=mode, use_cache=False)
            elif kind == "upload" and filepath is not None:
                result = analyzer.analyze_file(filepath, mode=mode, use_cache=False)
            else:
                result = {"media": "video", "error": "Invalid analysis request"}
            sys.stdout.flush()

            if result.get("error"):
                q.put({"type": "error", "text": result["error"]})
            else:
                clean_result = _strip_ui_hidden_fields(result)
                q.put({"type": "done", "data": clean_result})
                task["result"] = clean_result
        except Exception as e:
            sys.stdout.flush()
            q.put({"type": "error", "text": str(e)})
        finally:
            _thread_local.queue = None
            q.put(None)
            with _tasks_lock:
                task["done"] = True
                task["finished_at"] = time.time()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base = "http://127.0.0.1:5000/"
    _log.info("")
    _log.info("  " + "=" * 44)
    _log.info("  STEP Pipeline - Web UI")
    _log.info(f"  Max concurrent solves: {_MAX_CONCURRENT_SOLVES}")
    _log.info(f"  Web solver config: nougat={_WEB_USE_NOUGAT}, vlm={_WEB_USE_VLM}")
    _log.info("  " + "=" * 44)
    _log.info("")
    _log.info("  Tarayiciya TAM su adresi yapistirin (http:// sart):")
    _log.info(f"    {base}")
    _log.info("")
    _log.info("  Not: Sadece 127.0.0.1:5000 yazmak arama yapabilir; http:// ekleyin.")
    _log.info("  Kapatmak icin: Ctrl+C")
    _log.info("")

    def _open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(base)
        except Exception:
            pass

    if os.environ.get("STEP_NO_BROWSER", "").strip().lower() not in ("1", "true", "yes"):
        threading.Thread(target=_open_browser, daemon=True).start()

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
