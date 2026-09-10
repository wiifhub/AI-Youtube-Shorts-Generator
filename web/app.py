"""Web UI for the YouTube Shorts generator.

    python -m web.app
    # then open http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import io
import math
import os
import re
import shutil
import sys
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import File, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shorts_generator import generate_shorts  # noqa: E402
from shorts_generator.config import (  # noqa: E402
    LOCAL_BURN_CAPTIONS,
    LOCAL_OUTPUT_DIR,
    LOCAL_WHISPER_DEVICE,
    LOCAL_WHISPER_MODEL,
    gpu_status,
)

app = FastAPI(title="AI YouTube Shorts Generator", version="1.0")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
_cancel_events: Dict[str, threading.Event] = {}
_job_id_pattern = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer without letting a malformed .env break startup."""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(1, value)


_max_concurrent_jobs = _positive_int_env("SHORTS_MAX_CONCURRENT_JOBS", 2)
_job_slots = threading.Semaphore(_max_concurrent_jobs)
_output_root = Path(LOCAL_OUTPUT_DIR).expanduser().resolve()
_jobs_dir = _output_root / "jobs"
_uploads_dir = _output_root / "uploads"
_allowed_upload_extensions = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
_max_upload_mb = _positive_int_env("SHORTS_MAX_UPLOAD_MB", 2048)
_max_upload_bytes = _max_upload_mb * 1024 * 1024


def _safe_transcript_duration(transcript: Any) -> float:
    """Read a usable duration from persisted/API transcript data."""
    if not isinstance(transcript, dict):
        return 0.0
    try:
        duration = float(transcript.get("duration") or 0.0)
    except (TypeError, ValueError, OverflowError):
        duration = 0.0
    if math.isfinite(duration) and duration > 0:
        return duration
    ends = []
    raw_segments = transcript.get("segments")
    segments = raw_segments if isinstance(raw_segments, (list, tuple)) else []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            end = float(segment.get("end"))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(end) and end > 0:
            ends.append(end)
    return max(ends, default=0.0)


def _dict_items(value: Any) -> List[Dict[str, Any]]:
    """Return only object entries from persisted/API list data."""
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def _job_path(job_id: str) -> Path:
    safe_id = str(job_id)
    if not _job_id_pattern.fullmatch(safe_id):
        raise ValueError("invalid job id")
    return _jobs_dir / f"{safe_id}.json"


def _persist_job_locked(job: Dict[str, Any]) -> None:
    """Atomically persist one job; callers must hold ``_lock``."""
    _jobs_dir.mkdir(parents=True, exist_ok=True)
    path = _job_path(str(job["id"]))
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(job, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_persisted_jobs() -> None:
    """Restore recent jobs and mark in-flight work interrupted by a restart."""
    _jobs_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for path in _jobs_dir.glob("*.json"):
        try:
            files.append((path.stat().st_mtime, path))
        except OSError:
            continue
    for _, path in sorted(files, key=lambda item: item[0], reverse=True):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(job, dict) or not job.get("id"):
            continue
        job_id = str(job["id"])
        if not _job_id_pattern.fullmatch(job_id):
            continue
        job["id"] = job_id
        if not isinstance(job.get("logs"), list):
            job["logs"] = []
        try:
            created_at = float(job.get("created_at") or path.stat().st_mtime)
            if not math.isfinite(created_at):
                raise ValueError("created_at must be finite")
            job["created_at"] = created_at
        except (OSError, TypeError, ValueError, OverflowError):
            job["created_at"] = time.time()
        job.setdefault("status", "error")
        job.setdefault("stage", "unknown")
        job.setdefault("message", "Recovered project")
        if job.get("status") == "running":
            message = "Interrupted when Shorts Studio stopped; start a new job to retry."
            job["status"] = "error"
            job["stage"] = "error"
            job["message"] = message
            job["error"] = message
            job["logs"] = list(job["logs"])[-79:]
            job["logs"].append({"t": time.time(), "stage": "error", "message": message})
        _jobs[job_id] = job


_load_persisted_jobs()


class JobRequest(BaseModel):
    model_config = {"allow_inf_nan": False}
    url: str = Field(..., min_length=3)
    mode: str = "local"
    num_clips: int = Field(3, ge=1, le=12)
    aspect_ratio: str = "9:16"
    download_format: str = "720"
    language: Optional[str] = None
    caption_style: str = "bold"
    remove_silence: bool = False
    normalize_audio: bool = False
    denoise_audio: bool = False
    remove_filler_words: bool = False
    caption_position: str = "bottom"
    caption_font: str = "Arial"
    caption_size: int = Field(0, ge=0, le=120)
    caption_color: Optional[str] = None
    focus: str = "balanced"
    background_music: Optional[str] = None
    watermark: Optional[str] = None
    auto_reframe: bool = True
    crop_position: float = Field(0.5, ge=0.0, le=1.0)
    fit_mode: str = "crop"
    zoom: float = Field(1.0, ge=0.5, le=1.5)
    intro: Optional[str] = None
    outro: Optional[str] = None
    jump_cuts: bool = False
    layout: str = "single"
    whisper_model: Optional[str] = None
    whisper_device: Optional[str] = None
    save_folder: Optional[str] = None


class BatchRequest(JobRequest):
    url: str = ""
    urls: List[str] = Field(..., min_length=1, max_length=50)


class ClipUpdate(BaseModel):
    model_config = {"allow_inf_nan": False}
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., gt=0)
    caption_style: Optional[str] = None
    caption_position: str = "bottom"
    caption_font: str = "Arial"
    caption_size: int = Field(0, ge=0, le=120)
    caption_color: Optional[str] = None
    crop_position: float = Field(0.5, ge=0.0, le=1.0)
    zoom: float = Field(1.0, ge=0.5, le=1.5)
    fit_mode: str = "crop"
    layout: str = "single"


class ProjectUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


def _job_snapshot(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(job.get("id") or ""),
        "status": str(job.get("status") or "unknown"),
        "stage": str(job.get("stage") or "unknown"),
        "message": str(job.get("message") or ""),
        "logs": (job.get("logs") if isinstance(job.get("logs"), list) else [])[-80:],
        "error": job.get("error"),
        "result": job.get("result"),
        "created_at": job.get("created_at"),
        "request": job.get("request") if isinstance(job.get("request"), dict) else {},
        "output_dir": job.get("output_dir"),
    }


def _creator_metadata(short: Dict[str, Any]) -> Dict[str, str]:
    """Create ready-to-paste social metadata without another paid LLM call."""
    title = str(short.get("title") or "Untitled highlight").strip()
    hook = str(short.get("hook_sentence") or title).strip()
    reason = str(short.get("virality_reason") or "").strip()
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", f"{title} {hook}")
    tags = ["#Shorts", "#YouTubeShorts"]
    for word in words:
        tag = "#" + re.sub(r"[^A-Za-z0-9]", "", word)
        if tag.lower() not in {item.lower() for item in tags} and len(tag) <= 22:
            tags.append(tag)
        if len(tags) >= 6:
            break
    description = hook
    if reason:
        description += f"\n\n{reason}"
    description += "\n\n#Shorts #YouTubeShorts"
    return {
        "title": title[:100],
        "description": description[:5000],
        "hashtags": " ".join(tags),
        "thumbnail_text": hook[:70],
    }


def _public_shorts(shorts: List[Dict], job_id: str) -> List[Dict]:
    out = []
    for i, s in enumerate(_dict_items(shorts)):
        item = dict(s)
        clip = item.get("clip_url") or ""
        if clip and not str(clip).startswith("http"):
            item["play_url"] = f"/api/jobs/{job_id}/clip/{i}"
            item["local_path"] = clip
        else:
            item["play_url"] = clip
        thumbnail = item.get("thumbnail_path") or ""
        if thumbnail and not str(thumbnail).startswith("http"):
            item["thumbnail_url"] = f"/api/jobs/{job_id}/thumbnail/{i}"
        item["creator_metadata"] = _creator_metadata(item)
        out.append(item)
    return out


def _run_job(job_id: str, req: JobRequest) -> None:
    def progress(stage: str, message: str) -> None:
        with _lock:
            job = _jobs[job_id]
            if _cancel_events.get(job_id, threading.Event()).is_set():
                raise RuntimeError("Job cancelled")
            job["stage"] = stage
            job["message"] = message
            job["logs"].append({"t": time.time(), "stage": stage, "message": message})
            _persist_job_locked(job)

    try:
        _job_slots.acquire()
        progress("queued", "Starting pipeline...")
        if req.save_folder:
            base = Path(req.save_folder).expanduser().resolve()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            source_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", req.url.rsplit("/", 1)[-1])[:32] or "source"
            job_output_dir = base / f"{stamp}_shorts_{source_tag}_{job_id[:8]}"
        else:
            job_output_dir = _jobs_dir / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        with _lock:
            _jobs[job_id]["output_dir"] = str(job_output_dir)
            _persist_job_locked(_jobs[job_id])
        result = generate_shorts(
            youtube_url=req.url.strip(),
            num_clips=req.num_clips,
            aspect_ratio=req.aspect_ratio,
            download_format=req.download_format,
            language=req.language or None,
            mode=req.mode,
            progress=progress,
            output_dir=str(job_output_dir) if req.mode == "local" else None,
            caption_style=req.caption_style,
            remove_silence=req.remove_silence,
            normalize_audio=req.normalize_audio,
            denoise_audio=req.denoise_audio,
            remove_filler_words=req.remove_filler_words,
            caption_position=req.caption_position,
            caption_font=req.caption_font,
            caption_size=req.caption_size,
            caption_color=req.caption_color,
            focus=req.focus,
            background_music=req.background_music,
            watermark=req.watermark,
            auto_reframe=req.auto_reframe,
            crop_position=req.crop_position,
            fit_mode=req.fit_mode,
            zoom=req.zoom,
            intro=req.intro,
            outro=req.outro,
            jump_cuts=req.jump_cuts,
            layout=req.layout,
            whisper_model=req.whisper_model,
            whisper_device=req.whisper_device,
        )
        public = {
            "mode": result.get("mode"),
            "source_video_url": result.get("source_video_url"),
            "highlights": result.get("highlights", []),
            "shorts": _public_shorts(result.get("shorts", []), job_id),
            "transcript_duration": (result.get("transcript") or {}).get("duration"),
            "segment_count": len((result.get("transcript") or {}).get("segments") or []),
        }
        # Keep a portable manifest beside the rendered media, including when
        # the user selected a custom Save folder instead of output/jobs.
        try:
            with _lock:
                job_state = _jobs[job_id]
                created_at = job_state.get("created_at")
                request_snapshot = dict(job_state.get("request") or {})
            metadata = {
                "job_id": job_id,
                "created_at": created_at,
                "output_dir": str(job_output_dir),
                "request": request_snapshot,
                "result": public,
                "transcript": result.get("transcript") or {},
            }
            (job_output_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            progress("metadata", f"Could not write metadata.json: {exc}")
        with _lock:
            job = _jobs[job_id]
            if job.get("status") == "cancelled":
                return
            job["status"] = "done"
            job["stage"] = "done"
            job["message"] = f"Rendered {len(public['shorts'])} shorts"
            job["result"] = public
            job["raw_shorts"] = result.get("shorts", [])
            job["raw_transcript"] = result.get("transcript") or {}
            job["raw_source_video_url"] = result.get("source_video_url")
            job["logs"].append({"t": time.time(), "stage": "done", "message": job["message"]})
            _persist_job_locked(job)
    except Exception as exc:
        with _lock:
            job = _jobs[job_id]
            if job.get("status") == "cancelled":
                return
            job["status"] = "error"
            job["stage"] = "error"
            job["message"] = str(exc)
            job["error"] = str(exc)
            job["logs"].append({"t": time.time(), "stage": "error", "message": str(exc)})
            _persist_job_locked(job)
    finally:
        _job_slots.release()


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def _enqueue_job(req: JobRequest) -> Dict[str, Any]:
    if req.mode not in ("api", "local"):
        raise HTTPException(400, "mode must be api or local")
    if not req.url or not req.url.strip():
        raise HTTPException(400, "url/path cannot be blank")
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "running",
            "stage": "queued",
            "message": "Queued",
            "logs": [],
            "error": None,
            "result": None,
            "raw_shorts": [],
            "raw_transcript": {},
            "raw_source_video_url": None,
            "created_at": time.time(),
            "request": {
                "url": req.url.strip(),
                "mode": req.mode,
                "num_clips": req.num_clips,
                "aspect_ratio": req.aspect_ratio,
                "download_format": req.download_format,
                "language": req.language,
                "caption_style": req.caption_style,
                "remove_silence": req.remove_silence,
                "normalize_audio": req.normalize_audio,
                "denoise_audio": req.denoise_audio,
                "remove_filler_words": req.remove_filler_words,
                "caption_position": req.caption_position,
                "caption_font": req.caption_font,
                "caption_size": req.caption_size,
                "caption_color": req.caption_color,
                "focus": req.focus,
                "background_music": req.background_music,
                "watermark": req.watermark,
                "auto_reframe": req.auto_reframe,
                "crop_position": req.crop_position,
                "fit_mode": req.fit_mode,
                "zoom": req.zoom,
                "intro": req.intro,
                "outro": req.outro,
                "jump_cuts": req.jump_cuts,
                "layout": req.layout,
                "whisper_model": req.whisper_model,
                "whisper_device": req.whisper_device,
                "save_folder": req.save_folder,
            },
        }
        _cancel_events[job_id] = threading.Event()
        _persist_job_locked(_jobs[job_id])
    thread = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    thread.start()
    with _lock:
        return _job_snapshot(_jobs[job_id])


@app.post("/api/jobs")
def create_job(req: JobRequest) -> Dict[str, Any]:
    return _enqueue_job(req)


@app.post("/api/jobs/batch")
def create_batch_jobs(req: BatchRequest) -> Dict[str, Any]:
    jobs = []
    for url in req.urls:
        clean_url = url.strip()
        if len(clean_url) < 3:
            continue
        item = JobRequest(
            url=clean_url,
            mode=req.mode,
            num_clips=req.num_clips,
            aspect_ratio=req.aspect_ratio,
            download_format=req.download_format,
            language=req.language,
            caption_style=req.caption_style,
            remove_silence=req.remove_silence,
            normalize_audio=req.normalize_audio,
            denoise_audio=req.denoise_audio,
            remove_filler_words=req.remove_filler_words,
            caption_position=req.caption_position,
            caption_font=req.caption_font,
            caption_size=req.caption_size,
            caption_color=req.caption_color,
            focus=req.focus,
            background_music=req.background_music,
            watermark=req.watermark,
            auto_reframe=req.auto_reframe,
            crop_position=req.crop_position,
            fit_mode=req.fit_mode,
            zoom=req.zoom,
            intro=req.intro,
            outro=req.outro,
            jump_cuts=req.jump_cuts,
            layout=req.layout,
            whisper_model=req.whisper_model,
            whisper_device=req.whisper_device,
            save_folder=req.save_folder,
        )
        jobs.append(_enqueue_job(item))
    if not jobs:
        raise HTTPException(400, "batch did not contain any usable URLs or file paths")
    return {"jobs": jobs}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        if job.get("status") == "running":
            job["status"] = "cancelled"
            job["stage"] = "cancelled"
            job["message"] = "Cancellation requested"
            job["error"] = "Cancelled by user"
            job["logs"].append({"t": time.time(), "stage": "cancelled", "message": job["message"]})
            _cancel_events.setdefault(job_id, threading.Event()).set()
            _persist_job_locked(job)
        return _job_snapshot(job)


@app.get("/api/jobs")
def list_jobs() -> Dict[str, Any]:
    def created_at_value(job: Dict[str, Any]) -> float:
        try:
            value = float(job.get("created_at") or 0.0)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    with _lock:
        jobs = [
            _job_snapshot(j)
            for j in sorted(
                _jobs.values(),
                key=created_at_value,
                reverse=True,
            )
        ]
    return {"jobs": jobs[:20]}


@app.patch("/api/jobs/{job_id}")
def rename_job(job_id: str, update: ProjectUpdate) -> Dict[str, Any]:
    name = " ".join(update.name.split())
    if not name:
        raise HTTPException(400, "project name cannot be blank")
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        job["name"] = name
        _persist_job_locked(job)
        return _job_snapshot(job)


@app.post("/api/jobs/{job_id}/clips/{index}")
def update_clip(job_id: str, index: int, update: ClipUpdate) -> Dict[str, Any]:
    """Regenerate one clip after a manual timestamp/style adjustment."""
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        raw_shorts = _dict_items(job.get("raw_shorts"))
        if index < 0 or index >= len(raw_shorts):
            raise HTTPException(404, "clip not found")
        request = dict(job.get("request") or {})
        transcript = job.get("raw_transcript")
        transcript = dict(transcript) if isinstance(transcript, dict) else {}
        source = job.get("raw_source_video_url")
        mode = str((job.get("result") or {}).get("mode") or request.get("mode") or "local")

    duration = _safe_transcript_duration(transcript)
    if duration and update.start_time >= duration:
        raise HTTPException(400, f"start_time must be before the {duration:.1f}s source")
    if duration and update.end_time > duration + 0.25:
        raise HTTPException(400, f"end_time must be within the {duration:.1f}s source")
    if update.end_time <= update.start_time + 0.1:
        raise HTTPException(400, "end_time must be at least 0.1s after start_time")
    style = (update.caption_style or request.get("caption_style") or "bold").strip().lower()
    if style not in {"clean", "bold", "boxed", "karaoke"}:
        raise HTTPException(400, "caption_style must be clean, bold, boxed, or karaoke")
    if not source:
        raise HTTPException(400, "source video is unavailable for this job")

    old = dict(raw_shorts[index])
    try:
        if mode == "local":
            from shorts_generator.local.clipper import crop_clip_local

            old_path = str(old.get("clip_url") or "")
            out_path = old_path if old_path and not old_path.startswith("http") else str(
                _jobs_dir / job_id / f"short_{index + 1:02d}.mp4"
            )
            undo_path = out_path + ".undo.mp4"
            render_path = out_path + ".regenerate.mp4"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            crop_clip_local(
                str(source),
                update.start_time,
                update.end_time,
                str(request.get("aspect_ratio") or "9:16"),
                render_path,
                caption_segments=_dict_items(transcript.get("segments")),
                burn_captions=LOCAL_BURN_CAPTIONS,
                caption_style=style,
                remove_silence=bool(request.get("remove_silence")),
                normalize_audio=bool(request.get("normalize_audio")),
                denoise_audio=bool(request.get("denoise_audio")),
                remove_filler_words=bool(request.get("remove_filler_words")),
                caption_position=update.caption_position,
                caption_font=update.caption_font,
                caption_size=update.caption_size,
                caption_color=update.caption_color,
                background_music=request.get("background_music") or None,
                watermark=request.get("watermark") or None,
                auto_reframe=bool(request.get("auto_reframe", True)),
                crop_position=update.crop_position,
                fit_mode=update.fit_mode,
                zoom=update.zoom,
                layout=update.layout,
                intro=request.get("intro") or None,
                outro=request.get("outro") or None,
                jump_cuts=bool(request.get("jump_cuts")),
            )
            if not os.path.isfile(render_path):
                raise RuntimeError("clip renderer did not produce an output file")
            # Keep the current clip and its previous undo snapshot untouched
            # until the replacement render has completed successfully.
            if os.path.isfile(out_path):
                shutil.copyfile(out_path, undo_path)
            os.replace(render_path, out_path)
            replacement = {
                **old,
                "start_time": update.start_time,
                "end_time": update.end_time,
                "clip_url": out_path,
                "captions_burned": bool(LOCAL_BURN_CAPTIONS and transcript.get("segments")),
                "fit_mode": update.fit_mode,
                "zoom": update.zoom,
                "crop_position": update.crop_position,
                "caption_style": style,
                "caption_position": update.caption_position,
                "caption_font": update.caption_font,
                "caption_size": update.caption_size,
                "caption_color": update.caption_color,
                "layout": update.layout,
            }
            replacement["undo_path"] = undo_path if os.path.isfile(undo_path) else None
            replacement["undo_metadata"] = {
                key: value
                for key, value in old.items()
                if key not in {"undo_path", "undo_metadata"}
            }
            try:
                from shorts_generator.local.visual import extract_thumbnail

                thumb = str(_jobs_dir / job_id / f"short_{index + 1:02d}.jpg")
                extract_thumbnail(str(source), (update.start_time + update.end_time) / 2.0, thumb)
                replacement["thumbnail_path"] = thumb
            except Exception:
                pass
        elif mode == "api":
            from shorts_generator.clipper import crop_clip

            replacement = {
                **old,
                "start_time": update.start_time,
                "end_time": update.end_time,
                "fit_mode": update.fit_mode,
                "zoom": update.zoom,
                "crop_position": update.crop_position,
                "caption_style": style,
                "caption_position": update.caption_position,
                "caption_font": update.caption_font,
                "caption_size": update.caption_size,
                "caption_color": update.caption_color,
                "layout": update.layout,
                "clip_url": crop_clip(
                    str(source),
                    update.start_time,
                    update.end_time,
                    aspect_ratio=str(request.get("aspect_ratio") or "9:16"),
                ),
            }
        else:
            raise HTTPException(400, f"unsupported job mode: {mode}")
    except HTTPException:
        raise
    except Exception as exc:
        if mode == "local":
            render_path = locals().get("render_path")
            if render_path and os.path.isfile(render_path):
                try:
                    os.remove(render_path)
                except OSError:
                    pass
        raise HTTPException(500, f"could not regenerate clip: {exc}") from exc

    with _lock:
        job = _jobs[job_id]
        current = _dict_items(job.get("raw_shorts"))
        if index >= len(current):
            raise HTTPException(409, "job clips changed while regenerating")
        current[index] = replacement
        job["raw_shorts"] = current
        result = dict(job.get("result") or {})
        result["shorts"] = _public_shorts(current, job_id)
        job["result"] = result
        job["message"] = f"Regenerated clip {index + 1}"
        job["logs"].append({"t": time.time(), "stage": "edit", "message": job["message"]})
        _persist_job_locked(job)
        return _job_snapshot(job)


@app.post("/api/jobs/{job_id}/clips/{index}/undo")
def undo_clip(job_id: str, index: int) -> Dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        shorts = _dict_items(job.get("raw_shorts"))
        if index < 0 or index >= len(shorts):
            raise HTTPException(404, "clip not found")
        item = dict(shorts[index])
        undo_path = str(item.get("undo_path") or "")
        clip_path = str(item.get("clip_url") or "")
        if not undo_path or not os.path.isfile(undo_path) or not clip_path:
            raise HTTPException(400, "no previous clip version is available")
        shutil.copyfile(undo_path, clip_path)
        restored = dict(item.get("undo_metadata") or item)
        restored["clip_url"] = clip_path
        restored.pop("undo_path", None)
        restored.pop("undo_metadata", None)
        shorts[index] = restored
        job["raw_shorts"] = shorts
        if job.get("result"):
            job["result"]["shorts"] = _public_shorts(shorts, job_id)
        _persist_job_locked(job)
        return _job_snapshot(job)


@app.get("/api/jobs/{job_id}/timeline")
def get_timeline(job_id: str) -> Dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        transcript = job.get("raw_transcript")
        if not isinstance(transcript, dict):
            transcript = {}
        return {
            "duration": _safe_transcript_duration(transcript),
            "segments": _dict_items(transcript.get("segments")),
            "visual_events": _dict_items(transcript.get("visual_events")),
        }


@app.get("/api/jobs/{job_id}/export")
def export_job(job_id: str):
    """Download a ZIP containing local clips and creator metadata."""
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        result = dict(job.get("result") or {})
        raw_shorts = _dict_items(job.get("raw_shorts"))
        request = dict(job.get("request") or {})

    manifest = {
        "job_id": job_id,
        "request": request,
        "mode": result.get("mode"),
        "source_video_url": result.get("source_video_url"),
        "shorts": [
            {
                **{key: value for key, value in short.items() if key != "clip_url"},
                "creator_metadata": _creator_metadata(short),
            }
            for short in raw_shorts
        ],
    }
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("metadata.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        bundle.writestr("publishing/youtube_shorts.json", json.dumps({"platform": "youtube_shorts", "items": [s["creator_metadata"] for s in manifest["shorts"]]}, ensure_ascii=False, indent=2))
        bundle.writestr("publishing/tiktok.json", json.dumps({"platform": "tiktok", "items": [s["creator_metadata"] for s in manifest["shorts"]]}, ensure_ascii=False, indent=2))
        bundle.writestr("publishing/instagram_reels.json", json.dumps({"platform": "instagram_reels", "items": [s["creator_metadata"] for s in manifest["shorts"]]}, ensure_ascii=False, indent=2))
        for index, short in enumerate(raw_shorts, 1):
            path = str(short.get("clip_url") or "")
            if path and not path.startswith("http") and Path(path).is_file():
                bundle.write(path, arcname=f"clips/short_{index:02d}.mp4")
            thumbnail = str(short.get("thumbnail_path") or "")
            if thumbnail and Path(thumbnail).is_file():
                bundle.write(thumbnail, arcname=f"thumbnails/short_{index:02d}.jpg")
    archive.seek(0)
    return StreamingResponse(
        archive,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="shorts_{job_id}.zip"'},
    )


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return _job_snapshot(job)


@app.get("/api/jobs/{job_id}/clip/{index}")
def get_clip(job_id: str, index: int):
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        shorts = _dict_items(job.get("raw_shorts"))
    if index < 0 or index >= len(shorts):
        raise HTTPException(404, "clip not found")
    path = shorts[index].get("clip_url")
    if not path or str(path).startswith("http"):
        raise HTTPException(400, "clip is not a local file")
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, f"file missing: {path}")
    return FileResponse(p, media_type="video/mp4", filename=p.name)


@app.post("/api/jobs/{job_id}/preview")
def preview_clip(job_id: str, update: ClipUpdate) -> Dict[str, Any]:
    """Render a lightweight preview using the exact crop settings requested."""
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        request = dict(job.get("request") or {})
        source = job.get("raw_source_video_url")
        transcript = job.get("raw_transcript")
        transcript = dict(transcript) if isinstance(transcript, dict) else {}
        output_dir = str(job.get("output_dir") or (_jobs_dir / job_id))
    if not source or not Path(str(source)).is_file():
        raise HTTPException(400, "a completed local job is required for preview")
    duration = _safe_transcript_duration(transcript)
    if duration and update.start_time >= duration:
        raise HTTPException(400, f"start_time must be before the {duration:.1f}s source")
    if duration and update.end_time > duration + 0.25:
        raise HTTPException(400, f"end_time must be within the {duration:.1f}s source")
    if update.end_time <= update.start_time + 0.1:
        raise HTTPException(400, "end_time must be at least 0.1s after start_time")
    style = (update.caption_style or str(request.get("caption_style") or "bold")).strip().lower()
    if style not in {"clean", "bold", "boxed", "karaoke"}:
        raise HTTPException(400, "caption_style must be clean, bold, boxed, or karaoke")
    from shorts_generator.local.clipper import crop_clip_local
    preview = Path(output_dir) / "preview.mp4"
    preview.parent.mkdir(parents=True, exist_ok=True)
    render_path = preview.with_name("preview.render.mp4")
    try:
        crop_clip_local(
            str(source), update.start_time, update.end_time,
            str(request.get("aspect_ratio") or "9:16"), str(render_path),
            caption_segments=_dict_items(transcript.get("segments")),
            burn_captions=LOCAL_BURN_CAPTIONS,
            caption_style=style,
            caption_position=update.caption_position,
            caption_font=update.caption_font,
            caption_size=update.caption_size,
            caption_color=update.caption_color,
            remove_silence=bool(request.get("remove_silence")),
            normalize_audio=bool(request.get("normalize_audio")),
            denoise_audio=bool(request.get("denoise_audio")),
            remove_filler_words=bool(request.get("remove_filler_words")),
            background_music=request.get("background_music") or None,
            watermark=request.get("watermark") or None,
            auto_reframe=bool(request.get("auto_reframe", True)),
            crop_position=update.crop_position, fit_mode=update.fit_mode, zoom=update.zoom,
            layout=update.layout,
            intro=request.get("intro") or None,
            outro=request.get("outro") or None,
            jump_cuts=bool(request.get("jump_cuts")),
        )
        if not render_path.is_file():
            raise RuntimeError("preview renderer did not produce an output file")
        os.replace(render_path, preview)
    finally:
        if render_path.is_file():
            try:
                render_path.unlink()
            except OSError:
                pass
    return {"preview_url": f"/api/jobs/{job_id}/preview.mp4", "path": str(preview)}


@app.get("/api/jobs/{job_id}/preview.mp4")
def get_preview(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        path = Path(str(job.get("output_dir") or (_jobs_dir / job_id))) / "preview.mp4"
    if not path.is_file():
        raise HTTPException(404, "preview not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/jobs/{job_id}/thumbnail/{index}")
def get_thumbnail(job_id: str, index: int):
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        shorts = _dict_items(job.get("raw_shorts"))
    if index < 0 or index >= len(shorts):
        raise HTTPException(404, "thumbnail not found")
    path = shorts[index].get("thumbnail_path")
    if not path or not Path(path).is_file():
        raise HTTPException(404, "thumbnail not found")
    return FileResponse(path, media_type="image/jpeg", filename=Path(path).name)


@app.post("/api/uploads")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Store a browser-uploaded source in the local output area."""
    original_name = Path(file.filename or "video.mp4").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in _allowed_upload_extensions:
        allowed = ", ".join(sorted(_allowed_upload_extensions))
        raise HTTPException(400, f"unsupported video type; use one of: {allowed}")

    safe_stem = Path(original_name).stem.strip() or "video"
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in safe_stem)
    target = _uploads_dir / f"upload_{uuid.uuid4().hex[:12]}_{safe_stem}{suffix}"
    temporary = target.with_suffix(target.suffix + ".part")
    size = 0
    try:
        _uploads_dir.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > _max_upload_bytes:
                    raise HTTPException(
                        413,
                        f"file is larger than the {_max_upload_bytes // (1024 * 1024)} MB upload limit",
                    )
                output.write(chunk)
        os.replace(temporary, target)
    except HTTPException:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        raise
    except OSError as exc:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        raise HTTPException(500, f"could not save upload: {exc}") from exc
    except Exception as exc:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        raise HTTPException(500, f"could not save upload: {exc}") from exc
    finally:
        await file.close()

    return {
        "name": original_name,
        "path": str(target.resolve()),
        "url": f"/api/uploads/{target.name}",
        "size": size,
    }


@app.get("/api/uploads/{filename}")
def get_upload(filename: str):
    """Serve an uploaded source for an optional browser preview."""
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(404, "upload not found")
    path = _uploads_dir / safe_name
    if not path.is_file():
        raise HTTPException(404, "upload not found")
    return FileResponse(path, filename=path.name)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "output": str(Path(LOCAL_OUTPUT_DIR).resolve())}


@app.post("/api/shutdown")
def shutdown() -> Dict[str, str]:
    """Stop this local-only server (used by the portable launcher Quit button)."""
    threading.Timer(0.25, lambda: os._exit(0)).start()
    return {"status": "shutting_down"}


@app.get("/api/system")
def system_status() -> Dict[str, Any]:
    usage = shutil.disk_usage(_output_root)
    return {
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "whisper_model": LOCAL_WHISPER_MODEL,
        "whisper_device": LOCAL_WHISPER_DEVICE,
        "gpu": gpu_status(),
        "whisper_models": ["tiny", "base", "small", "medium", "large-v3"],
        "whisper_devices": ["auto", "cpu", "cuda"],
        "captions_enabled": LOCAL_BURN_CAPTIONS,
        "free_disk_gb": round(usage.free / (1024 ** 3), 2),
        "max_concurrent_jobs": _max_concurrent_jobs,
    }


@app.get("/api/diagnostics")
def diagnostics() -> Dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    with _lock:
        counts = {}
        for job in _jobs.values():
            counts[job.get("status", "unknown")] = counts.get(job.get("status", "unknown"), 0) + 1
    return {
        "python": sys.version,
        "ffmpeg_path": ffmpeg,
        "ffmpeg_ready": bool(ffmpeg),
        "output_root": str(_output_root),
        "free_disk_gb": round(shutil.disk_usage(_output_root).free / (1024 ** 3), 2),
        "job_counts": counts,
        "captions_enabled": LOCAL_BURN_CAPTIONS,
    }


@app.get("/api/update")
def update_check() -> Dict[str, Any]:
    """Check the public GitHub release without downloading or changing files."""
    import requests
    try:
        response = requests.get("https://api.github.com/repos/wiifhub/AI-Youtube-Shorts-Generator/releases/latest", timeout=8)
        response.raise_for_status()
        release = response.json()
        return {"available": True, "tag": release.get("tag_name"), "url": release.get("html_url"), "name": release.get("name")}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def main() -> None:
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=7860, reload=False)


if __name__ == "__main__":
    main()
