"""Regression tests for the 2026 security remediation pass.

Covers: credential-looking URL query redaction, provider error sanitisation,
persisted-URL scrubbing, hashed session cookies, CSRF double-submit,
job-limiter coverage for factory/batch/retry routes, bounded render queue,
and the fail-closed remote-bind startup guard.
"""

from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

import web.app as studio
from web import feature_routes
from web.security import SlidingWindowLimiter, issue_session_token, redact_text, redact_url_query, verify_session_token


@pytest.fixture()
def client() -> TestClient:
    with studio._lock:
        studio._jobs.clear()
        studio._job_credentials.clear()
        studio._cancel_events.clear()
        studio._job_futures.clear()
    with TestClient(studio.app) as test_client:
        yield test_client
    with studio._lock:
        studio._jobs.clear()
        studio._job_credentials.clear()
        studio._cancel_events.clear()
        studio._job_futures.clear()


def _login(client: TestClient, token: str) -> str:
    response = client.post("/api/auth/login", json={"token": token})
    assert response.status_code == 200
    csrf = client.cookies.get("shorts_csrf")
    assert csrf
    return csrf


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_redact_text_strips_credential_query_parameters() -> None:
    leaked = "GET https://graph.facebook.com/v25.0/oauth/access_token?client_id=1&client_secret=SUPERSECRET&code=abc failed"
    result = redact_text(leaked)
    assert "SUPERSECRET" not in result
    assert "client_secret" not in result
    assert "code=abc" in result


def test_redact_url_query_is_case_insensitive_and_recursive() -> None:
    scrubbed = redact_url_query("https://cdn.example/v.m3u8?AccessToken=abc123&X-Amz-Signature=sig&Expires=900&keep=1")
    assert "abc123" not in scrubbed
    assert "sig" not in scrubbed
    assert "900" not in scrubbed
    assert "keep=1" in scrubbed


def test_redact_text_handles_s3_signed_urls() -> None:
    url = "https://bucket.s3.amazonaws.com/key?X-Amz-Credential=AKIAEXAMPLE%2F20260924&X-Amz-Signature=deadbeef&X-Amz-Date=20260924"
    result = redact_text(url)
    assert "deadbeef" not in result
    assert "AKIAEXAMPLE" not in result


def test_backup_helper_scrubs_signed_and_secret_params() -> None:
    value = feature_routes._safe_backup_value(
        {
            "nested": "https://cdn.example/file.mp4?X-Amz-Signature=sigvalue&Client_Secret=sec&Expires=900&id=42",
            "plain": "https://www.youtube.com/watch?v=abc",
        }
    )
    assert "sigvalue" not in value["nested"]
    assert "sec" not in value["nested"]
    assert "id=42" in value["nested"]
    assert value["plain"] == "https://www.youtube.com/watch?v=abc"


# ---------------------------------------------------------------------------
# Session cookies
# ---------------------------------------------------------------------------


def test_session_cookie_does_not_contain_configured_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORTS_API_TOKEN", "super-secret-bearer-token")
    cookie_value, csrf_token, max_age = issue_session_token()
    assert "super-secret-bearer-token" not in cookie_value
    assert "super-secret-bearer-token" not in csrf_token
    assert max_age == 86400
    assert verify_session_token(cookie_value, csrf_token)
    assert not verify_session_token(cookie_value, "tampered:" + csrf_token.split(":", 1)[1])
    assert not verify_session_token("other-cookie", csrf_token)


def test_session_is_revoked_when_token_rotates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORTS_API_TOKEN", "before-rotation")
    cookie_value, csrf_token, _ = issue_session_token()
    assert verify_session_token(cookie_value, csrf_token)
    monkeypatch.setenv("SHORTS_API_TOKEN", "after-rotation")
    assert not verify_session_token(cookie_value, csrf_token)


def test_login_does_not_set_raw_token_cookie(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORTS_API_TOKEN", "raw-token-value")
    response = client.post("/api/auth/login", json={"token": "raw-token-value"})
    assert response.status_code == 200
    set_cookie = response.headers.get("set-cookie", "")
    assert "raw-token-value" not in set_cookie
    assert client.cookies.get("shorts_token") not in (None, "raw-token-value")


def test_bearer_token_still_accepted_and_cookie_is_not_bearer(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORTS_API_TOKEN", "bearer-only-token")
    _login(client, "bearer-only-token")
    assert client.get("/api/storage").status_code == 200
    # The session cookie value must not double as a bearer credential: strip
    # the cookie jar and replay the value through the Authorization header.
    cookie = client.cookies.get("shorts_token") or ""
    assert cookie != "bearer-only-token"
    client.cookies.clear()
    assert client.get("/api/storage", headers={"Authorization": f"Bearer {cookie}"}).status_code == 401
    # The configured token itself remains valid as a bearer credential.
    assert client.get("/api/storage", headers={"Authorization": "Bearer bearer-only-token"}).status_code == 200


# ---------------------------------------------------------------------------
# CSRF double-submit
# ---------------------------------------------------------------------------


def test_csrf_double_submit_required_for_cookie_mutations(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORTS_API_TOKEN", "csrf-token")
    csrf = _login(client, "csrf-token")
    origin = {"Origin": "http://testserver"}

    missing = client.post("/api/setup/dismiss", headers=dict(origin), json={"dismissed": True})
    assert missing.status_code == 403
    assert missing.json()["code"] == "csrf_failed"

    wrong = client.post("/api/setup/dismiss", headers={**origin, "X-CSRF-Token": "nope"}, json={"dismissed": True})
    assert wrong.status_code == 403

    okay = client.post("/api/setup/dismiss", headers={**origin, "X-CSRF-Token": csrf}, json={"dismissed": True})
    assert okay.status_code == 200


def test_csrf_null_origin_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORTS_API_TOKEN", "csrf-token")
    csrf = _login(client, "csrf-token")
    response = client.post(
        "/api/setup/dismiss",
        headers={"Origin": "null", "X-CSRF-Token": csrf},
        json={"dismissed": True},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "csrf_failed"


def test_bearer_mutations_skip_csrf(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORTS_API_TOKEN", "csrf-token")
    response = client.post(
        "/api/setup/dismiss",
        headers={"Authorization": "Bearer csrf-token"},
        json={"dismissed": True},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Job limiter coverage
# ---------------------------------------------------------------------------


def test_job_limiter_covers_factory_and_retry_paths(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    monkeypatch.setattr(studio, "_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_upload_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limiter", SlidingWindowLimiter(limit=1, window_seconds=60))
    # The middleware passes this per-minute value explicitly, so patch it too.
    monkeypatch.setattr(studio, "_job_rate_limit_per_minute", 1)

    first = client.post("/api/factory/jobs", json={"url": "https://youtu.be/aBcD1234", "mode": "api"})
    assert first.status_code == 200
    second = client.post("/api/factory/jobs", json={"url": "https://youtu.be/eFgH5678", "mode": "api"})
    assert second.status_code == 429
    assert second.json()["code"] == "rate_limited"


def test_versioned_factory_jobs_share_job_limiter(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    monkeypatch.setattr(studio, "_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_upload_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limiter", SlidingWindowLimiter(limit=1, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limit_per_minute", 1)

    first = client.post("/api/v1/factory/jobs", json={"url": "https://youtu.be/aBcD1234", "mode": "api"})
    assert first.status_code == 200
    second = client.post("/api/v1/factory/jobs", json={"url": "https://youtu.be/eFgH5678", "mode": "api"})
    assert second.status_code == 429


def test_batch_counts_each_url_against_job_budget(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(studio, "_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_upload_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limiter", SlidingWindowLimiter(limit=2, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limit_per_minute", 2)
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")

    response = client.post(
        "/api/jobs/batch",
        json={"urls": [f"https://youtu.be/aBcD{n:04d}" for n in range(3)], "mode": "api"},
    )
    assert response.status_code == 429
    assert "remaining job budget" in response.json()["error"]

    with studio._lock:
        assert not [job for job in studio._jobs.values() if str(job.get("status")) in {"queued", "running"}]

    okay = client.post(
        "/api/jobs/batch",
        json={"urls": [f"https://youtu.be/aBcD{n:04d}" for n in range(2)], "mode": "api"},
    )
    assert okay.status_code == 200
    assert len(okay.json()["jobs"]) == 2

    followup = client.post("/api/jobs", json={"url": "https://youtu.be/overflow9", "mode": "api"})
    assert followup.status_code == 429


def test_batch_request_model_caps_urls() -> None:
    from web.models import BatchRequest

    with pytest.raises(Exception):
        BatchRequest.model_validate({"urls": [f"https://example.com/{n}" for n in range(11)]})
    model = BatchRequest.model_validate({"urls": [f"https://youtu.be/aBcD{n:04d}" for n in range(10)]})
    assert len(model.urls) == 10


def test_bounded_queue_rejects_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """A saturated budget must turn submissions into 429, not unbounded growth."""
    release = threading.Event()

    def blocker(*_args: object, **_kwargs: object) -> None:
        release.wait(5)

    monkeypatch.setattr(studio, "_run_job", blocker)
    # Drain the shared semaphore so the budget is deterministic: fully busy.
    drained = []
    while studio._job_queue_slots.acquire(blocking=False):
        drained.append(True)
    try:
        req = studio.JobRequest(url="https://youtu.be/aBcD9999", mode="api")
        with pytest.raises(studio.HTTPException) as exc_info:
            studio._start_job_thread("bounded-job", req)
        assert exc_info.value.status_code == 429
        with studio._lock:
            assert "bounded-job" not in studio._job_futures
    finally:
        for _ in drained:
            studio._job_queue_slots.release()
        release.set()


# ---------------------------------------------------------------------------
# Persisted URL scrubbing
# ---------------------------------------------------------------------------


def test_enqueue_job_persists_scrubbed_url(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    url = "https://youtu.be/aBcD4242?access_token=leakvalue&X-Amz-Signature=sigvalue"
    response = client.post("/api/jobs", json={"url": url, "mode": "api"})
    assert response.status_code == 200
    payload = response.json()
    assert "leakvalue" not in json.dumps(payload)
    assert "sigvalue" not in json.dumps(payload)
    job_id = payload["id"]
    with studio._lock:
        stored = studio._jobs[job_id]["request"]["url"]
    assert "leakvalue" not in stored
    assert stored.startswith("https://youtu.be/aBcD4242")
    mirror = json.loads((studio._jobs_dir / f"{job_id}.json").read_text(encoding="utf-8"))
    assert "leakvalue" not in mirror["request"]["url"]


# ---------------------------------------------------------------------------
# Fail-closed remote bind
# ---------------------------------------------------------------------------


def test_remote_bind_without_token_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHORTS_API_TOKEN", raising=False)
    monkeypatch.delenv("SHORTS_ALLOW_UNAUTHENTICATED_REMOTE", raising=False)
    monkeypatch.setenv("SHORTS_BIND_HOST", "0.0.0.0")
    with pytest.raises(RuntimeError, match="SHORTS_API_TOKEN"):
        studio._validate_remote_security_configuration()


def test_remote_bind_with_token_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORTS_API_TOKEN", "remote-token")
    monkeypatch.setenv("SHORTS_BIND_HOST", "0.0.0.0")
    studio._validate_remote_security_configuration()


def test_loopback_bind_without_token_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHORTS_API_TOKEN", raising=False)
    monkeypatch.setenv("SHORTS_BIND_HOST", "127.0.0.1")
    studio._validate_remote_security_configuration()
    monkeypatch.setenv("SHORTS_BIND_HOST", "[::1]")
    studio._validate_remote_security_configuration()


def test_explicit_remote_opt_out_warns_and_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHORTS_API_TOKEN", raising=False)
    monkeypatch.setenv("SHORTS_BIND_HOST", "0.0.0.0")
    monkeypatch.setenv("SHORTS_ALLOW_UNAUTHENTICATED_REMOTE", "true")
    studio._validate_remote_security_configuration()


# ---------------------------------------------------------------------------
# Publishing request hygiene
# ---------------------------------------------------------------------------


def test_rejected_submission_cleans_up_job_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 from the bounded queue must not leave a stuck queued/running record."""
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    drained = []
    while studio._job_queue_slots.acquire(blocking=False):
        drained.append(True)
    try:
        with pytest.raises(studio.HTTPException) as exc_info:
            studio._enqueue_job(
                studio.JobRequest(url="https://youtu.be/aBcD7777", mode="api"),
                {"muapi": "mu_test_key_1234567890"},
            )
        assert exc_info.value.status_code == 429
        with studio._lock:
            stuck = [
                job
                for job in studio._jobs.values()
                if str(job.get("status")) in {"queued", "running"} and not studio._job_futures.get(str(job.get("id")))
            ]
            assert not stuck
            assert "aBcD7777" not in json.dumps([job.get("request") for job in studio._jobs.values()])
    finally:
        for _ in drained:
            studio._job_queue_slots.release()


def test_unsafe_open_urls_are_rejected() -> None:
    from web.security import is_safe_open_url

    assert is_safe_open_url("https://github.com/wiifhub/AI-Youtube-Shorts-Generator/releases/tag/v1.0.0")
    assert not is_safe_open_url("javascript:alert(1)")
    assert not is_safe_open_url("file:///C:/Windows/System32")
    assert not is_safe_open_url("https://user:pass@example.com/x")
    assert not is_safe_open_url("https://example.com/\r\nSet-Cookie: x=1")
    assert not is_safe_open_url("")


def test_publishing_validates_returned_authorization_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    from web import publishing

    monkeypatch.setenv("YOUTUBE_CLIENT_ID", "client-id")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "client-secret-value")
    monkeypatch.setenv("YOUTUBE_OAUTH_REDIRECT_URI", "http://127.0.0.1:7860/api/youtube/oauth/callback")
    info = publishing.start_youtube_oauth()
    assert info["authorization_url"].startswith("https://accounts.google.com/")

    with publishing._oauth_lock:
        publishing._oauth_states.clear()


def test_instagram_token_exchange_uses_post_body(monkeypatch: pytest.MonkeyPatch) -> None:
    from web import publishing

    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"access_token": "returned-token"}

    def fake_request(method: str, url: str, **kwargs: object) -> FakeResponse:
        captured["method"] = method
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setenv("INSTAGRAM_APP_ID", "app-id")
    monkeypatch.setenv("INSTAGRAM_APP_SECRET", "app-secret-value")
    monkeypatch.setattr(publishing.requests, "request", fake_request)

    with publishing._oauth_lock:
        publishing._social_oauth_states["state-value"] = {"provider": "instagram", "created_at": time.time()}
    publishing.complete_instagram_oauth("oauth-code", "state-value")

    assert captured["method"] == "POST"
    data = captured.get("data") or {}
    assert data.get("client_secret") == "app-secret-value"
    params = captured.get("params") or {}
    assert not params or "client_secret" not in params


# ---------------------------------------------------------------------------
# Session cookie Secure attribute (plain-http LAN access)
# ---------------------------------------------------------------------------


def test_plain_http_lan_bind_login_keeps_a_usable_cookie(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 0.0.0.0 deployment reached over plain http must still authenticate.

    Deriving ``Secure`` from the bind host made browsers silently drop the
    session cookie on ``http://<lan-ip>``: login answered 200 and every later
    request stayed unauthenticated.
    """
    monkeypatch.setenv("SHORTS_API_TOKEN", "lan-token")
    monkeypatch.delenv("SHORTS_COOKIE_SECURE", raising=False)
    monkeypatch.setenv("SHORTS_BIND_HOST", "0.0.0.0")

    response = client.post("/api/auth/login", json={"token": "lan-token"})
    assert response.status_code == 200
    set_cookie = response.headers.get("set-cookie", "")
    assert "shorts_token=" in set_cookie
    assert "Secure" not in set_cookie
    # The browser can now send the cookie back, so the session actually works.
    assert client.get("/api/storage").status_code == 200


def test_https_login_cookie_is_secure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORTS_API_TOKEN", "tls-token")
    monkeypatch.delenv("SHORTS_COOKIE_SECURE", raising=False)
    with TestClient(studio.app, base_url="https://testserver") as tls_client:
        response = tls_client.post("/api/auth/login", json={"token": "tls-token"})
    assert response.status_code == 200
    assert "Secure" in response.headers.get("set-cookie", "")


def test_forwarded_https_keeps_login_cookie_secure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """TLS terminated at a proxy arrives as http plus X-Forwarded-Proto: https."""
    monkeypatch.setenv("SHORTS_API_TOKEN", "proxy-token")
    monkeypatch.delenv("SHORTS_COOKIE_SECURE", raising=False)
    response = client.post(
        "/api/auth/login",
        headers={"X-Forwarded-Proto": "https"},
        json={"token": "proxy-token"},
    )
    assert response.status_code == 200
    assert "Secure" in response.headers.get("set-cookie", "")


def test_cookie_secure_override_forces_secure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORTS_API_TOKEN", "override-token")
    monkeypatch.setenv("SHORTS_COOKIE_SECURE", "true")
    response = client.post("/api/auth/login", json={"token": "override-token"})
    assert response.status_code == 200
    assert "Secure" in response.headers.get("set-cookie", "")


# ---------------------------------------------------------------------------
# Job limiter covers mutations, never reads
# ---------------------------------------------------------------------------


def test_job_reads_never_consume_the_job_budget(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The status poll, SSE stream and project list stay on the general limiter.

    Charging them here capped the 1200ms SSE fallback poll (~50/min) and the
    project list against the far smaller render budget.
    """
    monkeypatch.setattr(studio, "_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_upload_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limiter", SlidingWindowLimiter(limit=2, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limit_per_minute", 2)
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    monkeypatch.setattr(studio, "_start_job_thread", lambda *_args, **_kwargs: None)

    for _ in range(40):
        assert client.get("/api/jobs").status_code == 200
    for _ in range(40):
        assert client.get("/api/jobs/deadbeef0000").status_code in {200, 404}

    # Reads spent nothing, so both render slots are still available.
    assert client.post("/api/jobs", json={"url": "https://youtu.be/aBcD1111", "mode": "api"}).status_code == 200
    assert client.post("/api/jobs", json={"url": "https://youtu.be/aBcD2222", "mode": "api"}).status_code == 200
    exhausted = client.post("/api/jobs", json={"url": "https://youtu.be/aBcD3333", "mode": "api"})
    assert exhausted.status_code == 429


def test_library_refresh_does_not_drain_the_batch_budget(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(studio, "_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_upload_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limiter", SlidingWindowLimiter(limit=2, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limit_per_minute", 2)
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    monkeypatch.setattr(studio, "_start_job_thread", lambda *_args, **_kwargs: None)

    for _ in range(28):
        assert client.get("/api/jobs").status_code == 200

    # The batch handler spends the same per-client bucket, so the refreshes
    # above must not have consumed any of it.
    response = client.post(
        "/api/jobs/batch",
        json={"urls": ["https://youtu.be/aBcD0001", "https://youtu.be/aBcD0002"], "mode": "api"},
    )
    assert response.status_code == 200
    assert len(response.json()["jobs"]) == 2


# ---------------------------------------------------------------------------
# Source-URL fidelity versus credential scrubbing
# ---------------------------------------------------------------------------


def _capture_started(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Record the URL each render is actually handed."""
    started: dict = {}
    monkeypatch.setattr(
        studio,
        "_start_job_thread",
        lambda job_id, req, *args, **kwargs: started.__setitem__(job_id, req.url),
    )
    return started


def test_query_secret_matching_is_whole_name_only() -> None:
    """Substring matching rewrote functional source URLs; whole names do not."""
    for url in (
        "https://cdn.example/v.mp4?key=abc123",
        "https://cdn.example/v.mp4?author=bob&design=1",
        "https://cdn.example/v.mp4?si=share&size=large",
        "https://cdn.example/v.mp4?t=60#t=60",
        "https://youtu.be/aBcD1234?si=SHARE&list=PL123",
    ):
        assert redact_url_query(url) == url

    assert "X-Amz-Signature" not in redact_url_query("https://b.s3.amazonaws.com/k?X-Amz-Signature=deadbeef&id=1")
    assert "SEKRET" not in redact_url_query("https://g.example/o?access_token=SEKRET&code=abc")
    assert "SEKRET" not in redact_url_query("https://g.example/o?client_secret=SEKRET&client_id=1")
    assert "FRAGSECRET" not in redact_url_query("https://x.example/cb#access_token=FRAGSECRET")
    assert redact_text("open https://cdn.example/v.mp4?key=abc&author=bob").endswith("?key=abc&author=bob")


def test_functional_source_url_survives_enqueue_persist_and_resume(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A working source URL must not be rewritten just for looking secret-ish."""
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    url = "https://cdn.example/video.mp4?key=abc123&author=bob&design=1&si=share&size=large"
    started = _capture_started(monkeypatch)

    response = client.post("/api/jobs", json={"url": url, "mode": "api"})
    assert response.status_code == 200
    job_id = response.json()["id"]

    # 1. handed to the renderer byte-for-byte
    assert started[job_id] == url
    assert response.json()["request"]["url"] == url

    # 2. persisted byte-for-byte, in memory and in the portable mirror
    with studio._lock:
        assert studio._jobs[job_id]["request"]["url"] == url
        assert studio._jobs[job_id]["source_url_redacted"] is False
    mirror = json.loads((studio._jobs_dir / f"{job_id}.json").read_text(encoding="utf-8"))
    assert mirror["request"]["url"] == url

    # 3. resumed byte-for-byte
    with studio._lock:
        studio._jobs[job_id]["status"] = "interrupted"
        studio._persist_job_locked(studio._jobs[job_id])
    resumed = _capture_started(monkeypatch)
    studio._resume_interrupted_jobs()
    assert resumed[job_id] == url


def test_provider_query_credentials_are_stripped_from_every_saved_copy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    url = (
        "https://bucket.s3.amazonaws.com/video.mp4"
        "?X-Amz-Signature=deadbeefcafe&X-Amz-Credential=AKIAEXAMPLEKEY&access_token=OAUTHSECRET&id=7"
        "#access_token=FRAGMENTSECRET"
    )
    started = _capture_started(monkeypatch)

    response = client.post("/api/jobs", json={"url": url, "mode": "api"})
    assert response.status_code == 200
    job_id = response.json()["id"]
    secrets = ("deadbeefcafe", "AKIAEXAMPLEKEY", "OAUTHSECRET", "FRAGMENTSECRET")

    # The renderer still gets the URL it needs for the download to succeed.
    assert started[job_id] == url

    # No returned copy carries a credential value or name.
    returned = json.dumps(response.json())
    for secret in secrets:
        assert secret not in returned
    assert "access_token" not in returned

    # Nor does the persisted record, which keeps the functional parameter.
    with studio._lock:
        job = studio._jobs[job_id]
        stored = job["request"]["url"]
        assert job["source_url_redacted"] is True
    for secret in secrets:
        assert secret not in stored
    assert "access_token" not in stored
    assert stored.endswith("?id=7")

    mirror = (studio._jobs_dir / f"{job_id}.json").read_text(encoding="utf-8")
    for secret in secrets:
        assert secret not in mirror


def test_redacted_source_url_is_never_reused_for_resume_or_retry(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    _capture_started(monkeypatch)
    response = client.post(
        "/api/jobs",
        json={"url": "https://cdn.example/video.mp4?access_token=LEAKEDSECRET", "mode": "api"},
    )
    assert response.status_code == 200
    job_id = response.json()["id"]

    # Resume must refuse rather than fetch the stripped URL.
    with studio._lock:
        studio._jobs[job_id]["status"] = "interrupted"
        studio._persist_job_locked(studio._jobs[job_id])
    resumed = _capture_started(monkeypatch)
    studio._resume_interrupted_jobs()
    assert job_id not in resumed
    with studio._lock:
        assert studio._jobs[job_id]["status"] == "error"
        assert "credentials" in str(studio._jobs[job_id]["message"])

    # Retry must refuse for the same reason.
    retry = client.post(f"/api/jobs/{job_id}/retry")
    assert retry.status_code == 409
    assert "credentials" in json.dumps(retry.json())
