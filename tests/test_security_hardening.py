"""HTTP hardening checks that should stay cheap enough for every CI run."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web.app as studio

# The packaged app serves a loopback bind and refuses a foreign ``Host``, so the
# test client has to speak as a real browser on this machine would, and any
# same-origin header has to name that same authority.
LOOPBACK_BASE_URL = "http://127.0.0.1"
FOREIGN_ORIGIN = "https://evil.example"


@pytest.fixture()
def client() -> TestClient:
    with studio._lock:
        studio._jobs.clear()
        studio._job_credentials.clear()
        studio._cancel_events.clear()
        studio._job_futures.clear()
    with TestClient(studio.app, base_url=LOOPBACK_BASE_URL) as test_client:
        yield test_client
    with studio._lock:
        studio._jobs.clear()
        studio._job_credentials.clear()
        studio._cancel_events.clear()
        studio._job_futures.clear()


@pytest.fixture(autouse=True)
def _no_configured_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped desktop build runs with no token; that is the case under test."""
    monkeypatch.delenv("SHORTS_API_TOKEN", raising=False)
    monkeypatch.delenv("SHORTS_CORS_ORIGINS", raising=False)


def _fabricate_job(job_id: str = "cross-site-project") -> str:
    """Put one project into real state so a mutation is observable."""
    output_dir = Path(studio._jobs_dir) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with studio._lock:
        studio._jobs[job_id] = {
            "id": job_id,
            "name": "Creator project",
            "status": "running",
            "message": "rendering",
            "progress": 25,
            "request": {"url": "https://www.youtube.com/watch?v=abc123", "mode": "local", "llm_provider": "ollama"},
            "result": None,
            "raw_shorts": [],
            "raw_transcript": {},
            "raw_source_video_url": "https://www.youtube.com/watch?v=abc123",
            "output_dir": str(output_dir),
            "logs": [],
            "created_at": time.time(),
            "updated_at": time.time(),
            "archived": False,
        }
        studio._cancel_events[job_id] = threading.Event()
        studio._persist_job_locked(studio._jobs[job_id])
    return job_id


def _job_status(job_id: str) -> str:
    with studio._lock:
        return str(studio._jobs[job_id].get("status"))


def test_healthz_is_non_disclosing_and_security_headers_are_present(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "output" not in response.json()
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-security-policy"].startswith("default-src 'self'")


def test_cookie_authenticated_cross_site_mutation_is_blocked(client, monkeypatch) -> None:
    monkeypatch.setenv("SHORTS_API_TOKEN", "csrf-test-token")
    login = client.post("/api/auth/login", json={"token": "csrf-test-token"})
    assert login.status_code == 200
    csrf_token = client.cookies.get("shorts_csrf")
    assert csrf_token
    # The session cookie must not contain (or equal) the configured token.
    assert "csrf-test-token" not in client.cookies.get("shorts_token", "")

    # Cross-site mutation: rejected by the Origin check.
    blocked = client.post(
        "/api/setup/dismiss",
        headers={"Origin": FOREIGN_ORIGIN},
        json={"dismissed": True},
    )
    assert blocked.status_code == 403
    assert blocked.json()["code"] == "csrf_failed"

    # Same-origin but missing CSRF token: rejected by the double-submit check.
    blocked = client.post(
        "/api/setup/dismiss",
        headers={"Origin": LOOPBACK_BASE_URL},
        json={"dismissed": True},
    )
    assert blocked.status_code == 403
    assert blocked.json()["code"] == "csrf_failed"

    # Same-origin with the matching CSRF token: accepted.
    allowed = client.post(
        "/api/setup/dismiss",
        headers={"Origin": LOOPBACK_BASE_URL, "X-CSRF-Token": csrf_token},
        json={"dismissed": True},
    )
    assert allowed.status_code == 200


def test_cross_origin_mutation_is_blocked_without_a_configured_token(client) -> None:
    """The shipped default has no token, which is exactly when the browser rules
    used to be skipped entirely: a page the creator merely visits could cancel
    their render or stop the app with a bodyless form POST (no preflight)."""
    job_id = _fabricate_job()

    # A simple form-encoded POST, the shape a hostile page can send without a
    # preflight, carrying the Origin and Referer a browser always attaches.
    response = client.post(
        f"/api/jobs/{job_id}/cancel",
        headers={"Origin": FOREIGN_ORIGIN, "Referer": FOREIGN_ORIGIN + "/page.html"},
        data={},
    )
    assert response.status_code == 403, response.text
    assert response.json()["code"] == "csrf_failed"
    assert _job_status(job_id) == "running"

    # The same for a JSON POST and for a route that writes durable state.
    assert client.post("/api/setup/dismiss", headers={"Origin": FOREIGN_ORIGIN}, json={"dismissed": True}).status_code == 403

    # Stopping the app is the sharpest effect of all.
    studio._shutdown_requested.clear()
    try:
        stopped = client.post("/api/shutdown", headers={"Origin": FOREIGN_ORIGIN}, data={})
        assert stopped.status_code == 403, stopped.text
        assert not studio._shutdown_requested.is_set()
    finally:
        studio._shutdown_requested.clear()

    # An ordinary request from the app's own page carries no Origin at all in
    # some browsers, and must keep working.
    assert client.post("/api/setup/dismiss", json={"dismissed": True}).status_code == 200


def test_rebound_host_cannot_read_or_mutate_projects(client) -> None:
    """A DNS name the attacker owns, pointed at 127.0.0.1, is same-origin for the
    browser, so the server itself has to refuse the foreign name."""
    job_id = _fabricate_job()
    rebound = {"Host": "evil.example"}

    read = client.get("/api/jobs", headers=rebound)
    assert read.status_code == 403, read.text
    assert "Creator project" not in read.text

    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=rebound) as stream:
        assert stream.status_code == 403

    write = client.post(
        f"/api/jobs/{job_id}/cancel",
        headers={**rebound, "Origin": "http://evil.example"},
        data={},
    )
    assert write.status_code == 403, write.text
    assert _job_status(job_id) == "running"

    # The loopback names the app actually serves keep working.
    assert client.get("/api/jobs").status_code == 200


def test_cross_origin_update_apply_cannot_start_a_download(client, monkeypatch) -> None:
    """``/api/update/apply`` downloads an artifact and launches an installer, so
    a hostile page must not be able to reach it; the control request proves the
    route really would have started the update."""
    started: list[str] = []
    monkeypatch.setattr(studio._update_service, "package_root_writable", lambda: True)
    monkeypatch.setattr(studio._update_service, "download_and_apply", lambda asset, latest: started.append(str(asset["name"])))
    monkeypatch.setattr(
        studio,
        "_github_release",
        lambda: {
            "tag_name": "v99.0.0",
            "html_url": "https://github.com/example/shorts/releases/tag/v99.0.0",
            "name": "Shorts Studio 99",
            "assets": [
                {
                    "name": "ShortsStudio-windows.zip",
                    "browser_download_url": "https://github.com/example/shorts/releases/download/v99.0.0/ShortsStudio-windows.zip",
                    "digest": "sha256:" + "0" * 64,
                }
            ],
        },
    )
    with studio._update_lock:
        saved_state = dict(studio._update_state)
        studio._update_state.clear()
        studio._update_state["status"] = "idle"
    try:
        blocked = client.post("/api/update/apply", headers={"Origin": FOREIGN_ORIGIN}, data={})
        assert blocked.status_code == 403, blocked.text
        assert started == []

        with studio._update_lock:
            studio._update_state["status"] = "idle"
        control = client.post("/api/update/apply")
        assert control.status_code == 200, control.text
        assert control.json()["status"] == "starting"
        assert started == ["ShortsStudio-windows.zip"]
    finally:
        with studio._update_lock:
            studio._update_state.clear()
            studio._update_state.update(saved_state)


def test_a_token_presented_in_a_header_is_not_a_browser_request(client, monkeypatch) -> None:
    """A page cannot read the token, so a caller that presents it is not the
    browser context these rules exist to constrain; cancelling a running render
    therefore still works for API clients."""
    monkeypatch.setenv("SHORTS_API_TOKEN", "api-client-token")
    job_id = _fabricate_job()

    response = client.post(
        f"/api/jobs/{job_id}/cancel",
        headers={"Authorization": "Bearer api-client-token", "Origin": FOREIGN_ORIGIN},
        data={},
    )
    assert response.status_code == 200, response.text
    assert _job_status(job_id) == "cancelled"


def test_a_network_bind_keeps_serving_its_real_hostname(monkeypatch) -> None:
    """Only the loopback build refuses a foreign Host; a deployment bound to a
    network address is authenticated and keeps the name its users type."""
    monkeypatch.setenv("SHORTS_BIND_HOST", "0.0.0.0")
    monkeypatch.setenv("SHORTS_API_TOKEN", "deploy-token")
    with TestClient(studio.app, base_url="http://studio.internal") as deployment:
        response = deployment.get("/api/jobs", headers={"Authorization": "Bearer deploy-token"})
    assert response.status_code == 200, response.text


def test_backup_safe_url_helper_removes_signed_media_links() -> None:
    from web.feature_routes import _safe_backup_value

    value = _safe_backup_value(
        {
            "raw_source_video_url": "https://cdn.example/source?token=secret",
            "clip_url": "https://cdn.example/clip?signature=secret",
            "request_url": "https://www.youtube.com/watch?v=abc&expires=10",
        }
    )
    assert value["raw_source_video_url"] is None
    assert value["clip_url"] is None
    assert "secret" not in value["request_url"]
