"""Process-level cancellation regression tests."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import pytest

import web.app as studio
from shorts_generator.config import runtime_job_control
from shorts_generator.local.clipper import _run_command
from web import job_routes


def test_ffmpeg_command_is_terminated_when_job_is_cancelled() -> None:
    cancelled = threading.Event()
    timer = threading.Timer(0.25, cancelled.set)
    timer.start()
    try:
        with runtime_job_control(cancel_check=cancelled.is_set):
            with pytest.raises(RuntimeError, match="Job cancelled"):
                _run_command([sys.executable, "-c", "import time; time.sleep(30)"])
    finally:
        timer.cancel()


@pytest.fixture()
def clean_job_state() -> Iterator[None]:
    """Leave no project record, process registration, or future behind."""
    yield
    with studio._lock:
        studio._jobs.clear()
        studio._cancel_events.clear()
        studio._job_credentials.clear()
        studio._job_futures.clear()
    with studio._process_lock:
        studio._job_processes.clear()
        getattr(studio, "_job_run_generations", {}).clear()


def _register_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str, job_id: str = "control-job") -> str:
    """Register one project whose local render can be cancelled or retried."""
    monkeypatch.setattr(studio, "_output_root", tmp_path)
    monkeypatch.setattr(studio, "_jobs_dir", tmp_path / "jobs")
    monkeypatch.setattr(studio, "_allow_external_paths", False)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output_dir = tmp_path / "jobs" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with studio._lock:
        studio._jobs[job_id] = {
            "id": job_id,
            "name": "Control",
            "status": status,
            "message": status,
            "progress": 0,
            "request": {"url": str(source), "mode": "local", "llm_provider": "ollama"},
            "result": None,
            "raw_shorts": [],
            "raw_transcript": {},
            "raw_source_video_url": str(source),
            "output_dir": str(output_dir),
            "logs": [],
            "created_at": time.time(),
            "updated_at": time.time(),
            "archived": False,
        }
        studio._cancel_events[job_id] = threading.Event()
        studio._persist_job_locked(studio._jobs[job_id])
    return job_id


def _sleeping_child() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])


def _reap(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.kill()


def _status(job_id: str) -> str:
    with studio._lock:
        return str(studio._jobs[job_id].get("status"))


def _wait_for_status(job_id: str, wanted: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _status(job_id) == wanted:
            return
        time.sleep(0.02)
    raise AssertionError(f"project never reached {wanted!r} (still {_status(job_id)!r})")


def _start_render(monkeypatch: pytest.MonkeyPatch, observed: dict, registered: threading.Event, release: threading.Event):
    """Stand in for a render: register a child, then hold it until released."""

    def stub_run_job(active_id: str, _req, _credentials) -> None:
        process = _sleeping_child()
        observed["process"] = process
        studio._register_job_process(active_id, process)
        registered.set()
        release.wait(timeout=30)
        observed["terminated"] = process.poll() is not None
        observed["returncode"] = process.returncode
        if process.poll() is None:
            process.kill()

    monkeypatch.setattr(studio, "_run_job", stub_run_job)


def test_cancel_does_not_terminate_a_retry_started_in_its_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_job_state: None
) -> None:
    """A retry accepted after the cancel decision keeps its own render.

    ``cancel_job`` commits ``cancelled`` under the lock but terminates the
    project's children after releasing it.  A retry that lands in that window is
    accepted and starts a fresh render, and the pending terminate used to kill
    it: the child died with a non-zero code while the project stayed ``running``.
    """
    job_id = _register_job(monkeypatch, tmp_path, "running")
    observed: dict = {}
    registered = threading.Event()
    release = threading.Event()
    _start_render(monkeypatch, observed, registered, release)

    real_terminate = studio._terminate_job_processes

    def terminate_after_the_retry_registers(active_id: str, generation: Optional[int] = None) -> int:
        # Hold the cancel exactly where the code leaves it: past the status
        # commit and inside the terminate, once a retry has started its render.
        registered.wait(timeout=15)
        if generation is None:
            return real_terminate(active_id)
        return real_terminate(active_id, generation)

    monkeypatch.setattr(studio, "_terminate_job_processes", terminate_after_the_retry_registers)

    cancelling = threading.Thread(target=lambda: job_routes.cancel_job(job_id), daemon=True)
    cancelling.start()
    _wait_for_status(job_id, "cancelled")
    job_routes.retry_job(job_id, None, None, None, None)
    try:
        assert registered.wait(timeout=15), "the retried render never started"
        cancelling.join(timeout=30)
        process = observed["process"]
        assert process.poll() is None, "the retry's fresh render was terminated by the cancel of the previous run"
        assert process.returncode is None
        assert _status(job_id) == "running"
    finally:
        release.set()


def test_cancel_after_a_retry_terminates_the_retried_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_job_state: None
) -> None:
    """Scoping must not protect a render the creator really cancelled."""
    job_id = _register_job(monkeypatch, tmp_path, "cancelled")
    previous_generation = studio._job_run_generation(job_id)
    observed: dict = {}
    registered = threading.Event()
    release = threading.Event()
    _start_render(monkeypatch, observed, registered, release)

    job_routes.retry_job(job_id, None, None, None, None)
    try:
        assert registered.wait(timeout=15), "the retried render never started"
        process = observed["process"]
        assert studio._job_run_generation(job_id) != previous_generation
        # The previous run's cancellation cannot reach this run's child ...
        assert studio._terminate_job_processes(job_id, previous_generation) == 0
        assert process.poll() is None
        # ... but cancelling the retried run still kills it.
        snapshot = job_routes.cancel_job(job_id)
        assert snapshot["terminated_processes"] == 1
        assert process.poll() is not None
    finally:
        release.set()


def test_cancel_terminates_the_run_it_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_job_state: None
) -> None:
    """The ordinary case is unchanged: cancelling kills the running render."""
    job_id = _register_job(monkeypatch, tmp_path, "running")
    process = _sleeping_child()
    studio._register_job_process(job_id, process)
    try:
        snapshot = job_routes.cancel_job(job_id)
        assert snapshot["terminated_processes"] == 1
        assert process.poll() is not None
    finally:
        _reap(process)


def test_second_cancel_cleans_up_a_lingering_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_job_state: None
) -> None:
    """A repeated cancel stays useful: it terminates whatever still runs."""
    job_id = _register_job(monkeypatch, tmp_path, "cancelled")
    process = _sleeping_child()
    studio._register_job_process(job_id, process)
    try:
        snapshot = job_routes.cancel_job(job_id)
        assert snapshot["terminated_processes"] == 1
        assert process.poll() is not None
        assert _status(job_id) == "cancelled"
    finally:
        _reap(process)


def test_retry_after_a_cancel_runs_to_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_job_state: None
) -> None:
    """A retried render still starts and completes on its own."""
    job_id = _register_job(monkeypatch, tmp_path, "cancelled")
    observed: dict = {}
    finished = threading.Event()

    def stub_run_job(active_id: str, _req, _credentials) -> None:
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"])
        studio._register_job_process(active_id, process)
        process.wait(timeout=20)
        observed["returncode"] = process.returncode
        with studio._lock:
            studio._jobs[active_id]["status"] = "done"
        finished.set()

    monkeypatch.setattr(studio, "_run_job", stub_run_job)
    job_routes.retry_job(job_id, None, None, None, None)
    assert finished.wait(timeout=20), "the retried render never completed"
    assert observed["returncode"] == 0
    assert _status(job_id) == "done"
