"""Regression tests for the 2026 security remediation pass.

Covers: credential-looking URL query redaction, provider error sanitisation,
persisted-URL scrubbing, hashed session cookies, CSRF double-submit,
job-limiter coverage for factory/batch/retry routes, bounded render queue,
and the fail-closed remote-bind startup guard.
"""

from __future__ import annotations

import io
import json
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web.app as studio
from shorts_generator.local import clipper
from web import feature_routes
from web.security import (
    JOB_SUBMISSION_PATH,
    MUTATING_METHODS,
    SlidingWindowLimiter,
    issue_session_token,
    job_budget_bucket,
    rate_limit_shape,
    redact_record_urls,
    redact_text,
    redact_url_query,
    verify_session_token,
)


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


def test_record_url_scrub_covers_every_field_without_rewriting_prose() -> None:
    """The durable write walks the whole record, but only whole-string URLs."""
    record = {
        "request": {"url": "https://cdn.example/video.mp4?key=abc123"},
        "raw_source_video_url": "https://b.s3.amazonaws.com/v.mp4?X-Amz-Signature=deadbeefcafe",
        "result": {
            "source_video_url": "https://x.example/cb#access_token=FRAGSECRET",
            "shorts": [{"play_url": "/api/jobs/1/clip/0", "note": "see https://docs.example/a b for help"}],
        },
        "tuple_url": ("https://x.example/o?access_token=TUPLESECRET",),
    }

    redact_record_urls(record)
    assert "deadbeefcafe" not in json.dumps(record)
    assert "FRAGSECRET" not in json.dumps(record)
    assert "TUPLESECRET" not in json.dumps(record)
    # Functional parameters survive, prose is not treated as a URL, and a tuple
    # keeps its type.
    assert record["request"]["url"] == "https://cdn.example/video.mp4?key=abc123"
    assert record["result"]["shorts"][0]["note"] == "see https://docs.example/a b for help"
    assert isinstance(record["tuple_url"], tuple)

    # A record with nothing to strip stays byte-for-byte identical, so nothing
    # downstream sees a change it must react to.
    clean = {
        "request": {"url": "https://cdn.example/v.mp4?key=abc"},
        "logs": [{"message": "rendered 3 clips at https://docs.example/a b"}],
    }
    before = json.dumps(clean)
    redact_record_urls(clean)
    assert json.dumps(clean) == before


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
            "stream": (
                f"https://customer-abc.cloudflarestream.com/{_STREAM_TOKEN}/manifest/video.m3u8"
            ),
        }
    )
    assert "sigvalue" not in value["nested"]
    assert "sec" not in value["nested"]
    assert "id=42" in value["nested"]
    assert value["plain"] == "https://www.youtube.com/watch?v=abc"
    assert _STREAM_TOKEN not in value["stream"]
    assert value["stream"].endswith("/manifest/video.m3u8")


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
# The render budget is keyed by route shape, not by the concrete URL
# ---------------------------------------------------------------------------


def _widen_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup work should not spend the budget under test."""
    monkeypatch.setattr(studio, "_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_upload_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limit_per_minute", 1000)
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")


def _tighten_render_budget(monkeypatch: pytest.MonkeyPatch, limit: int = 1) -> None:
    monkeypatch.setattr(studio, "_job_rate_limiter", SlidingWindowLimiter(limit=limit, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limit_per_minute", limit)


def _record_started(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record every render that actually reaches the worker."""
    started: list = []
    monkeypatch.setattr(
        studio,
        "_start_job_thread",
        lambda job_id, req, *args, **kwargs: started.append((job_id, req.url)),
    )
    return started


def _park_retryable_projects(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, started: list, count: int
) -> list:
    """Create real API-mode projects through the endpoint, then park them failed."""
    _widen_budgets(monkeypatch)
    job_ids = []
    for index in range(count):
        response = client.post("/api/jobs", json={"url": f"https://youtu.be/aBcD{index:04d}", "mode": "api"})
        assert response.status_code == 200, response.text
        job_ids.append(response.json()["id"])
    with studio._lock:
        for job_id in job_ids:
            job = studio._jobs[job_id]
            job["status"] = "error"
            job["stage"] = "error"
            studio._persist_job_locked(job)
    started.clear()
    return job_ids


def _local_project(client: TestClient, monkeypatch: pytest.MonkeyPatch, index: int) -> str:
    """One local-mode project whose source file really exists inside the output.

    The projects exist to exercise the editor routes, not to render, so the
    worker is parked and the fake source is never handed to the pipeline.
    """
    _widen_budgets(monkeypatch)
    monkeypatch.setattr(studio, "_start_job_thread", lambda *_args, **_kwargs: None)
    source = studio._uploads_dir / f"budget-shape-{index}.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"not really a video")
    response = client.post("/api/jobs", json={"url": str(source), "mode": "local"})
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    with studio._lock:
        job = studio._jobs[job_id]
        job["status"] = "done"
        job["stage"] = "done"
        job["raw_source_video_url"] = str(source)
        studio._persist_job_locked(job)
    return job_id


def _middleware_path(path: str) -> str:
    """The path the limiter sees: the middleware rewrites /api/v1 to /api."""
    if path.startswith("/api/v1/"):
        return "/api" + path[len("/api/v1") :]
    return path


def _concrete_route(path: str, job_id: str) -> str:
    """Fill a registered route template with a project id and dummy parameters."""
    return (
        _middleware_path(path).replace("{job_id}", job_id)
        .replace("{index}", "3")
        .replace("{filename}", "clip.mp4")
        .replace("{name}", "preset")
        .replace("{variant_id}", "abc123def456")
    )


def test_no_mutating_route_keys_its_budget_on_the_project_id() -> None:
    """A bucket built from the URL hands every project a private quota.

    This walks the registered route table instead of a hand-written list, so a
    route added later cannot quietly reintroduce the multiplier: two different
    projects hitting the same route must land in the same bucket.
    """
    paths = studio.app.openapi()["paths"]
    assert paths, "route table is empty"
    checked = 0
    for path, operations in paths.items():
        for method in operations:
            if method.upper() not in MUTATING_METHODS:
                continue
            first = job_budget_bucket(method.upper(), _concrete_route(path, "aBcD1234ef56"))
            second = job_budget_bucket(method.upper(), _concrete_route(path, "zzTt99887766"))
            assert first == second, f"{method.upper()} {path} keys its render budget on the project id"
            first_shape = rate_limit_shape(_concrete_route(path, "aBcD1234ef56"))
            second_shape = rate_limit_shape(_concrete_route(path, "zzTt99887766"))
            assert first_shape == second_shape, f"{method.upper()} {path} is limiter-bucketed per project"
            checked += 1
    assert checked > 20


def test_limiter_buckets_are_route_shapes_not_concrete_urls() -> None:
    """A read that runs ffmpeg must not get a private quota per project either."""
    assert rate_limit_shape("/api/jobs/aBcD1234ef56/waveform") == "/api/jobs/{job_id}/waveform"
    assert rate_limit_shape("/api/jobs/zzTt99887766/waveform") == "/api/jobs/{job_id}/waveform"
    assert rate_limit_shape("/api/jobs/aBcD1234ef56/clips/3") == "/api/jobs/{job_id}/clips/{index}"
    assert rate_limit_shape("/api/jobs/zzTt99887766/clips/9") == "/api/jobs/{job_id}/clips/{index}"
    assert rate_limit_shape("/api/jobs/aBcD1234ef56") == "/api/jobs/{job_id}"
    # Paths with no per-project id are their own shape.
    assert rate_limit_shape("/api/jobs") == "/api/jobs"
    assert rate_limit_shape("/api/storage") == "/api/storage"
    assert rate_limit_shape("/api/uploads") == "/api/uploads"


def test_job_budget_buckets_are_the_route_shapes() -> None:
    """The shapes that exist, and the ones that do not, decide the budget."""
    assert job_budget_bucket("POST", "/api/jobs") == JOB_SUBMISSION_PATH
    assert job_budget_bucket("POST", "/api/factory/jobs") == JOB_SUBMISSION_PATH
    # The batch handler spends one submission slot per URL itself, so the
    # middleware keeps it on its own request bucket rather than double-charging.
    assert job_budget_bucket("POST", "/api/jobs/batch") == "/api/jobs/batch"
    # A retry starts a render, so it spends the shared submission slot.
    assert job_budget_bucket("POST", "/api/jobs/aBcD1234ef56/retry") == JOB_SUBMISSION_PATH
    assert job_budget_bucket("POST", "/api/jobs/aBcD1234ef56/clips/3") == "/api/jobs/{job_id}/clips/{index}"
    assert job_budget_bucket("POST", "/api/jobs/zzTt99887766/clips/9") == "/api/jobs/{job_id}/clips/{index}"
    assert job_budget_bucket("POST", "/api/jobs/aBcD1234ef56/preview") == "/api/jobs/{job_id}/preview"
    # Reads never spend the render budget, and neither does an unrelated route.
    assert job_budget_bucket("GET", "/api/jobs/aBcD1234ef56/preview") is None
    assert job_budget_bucket("POST", "/api/open-folder") is None


def test_a_retry_spends_the_shared_render_slot_not_a_private_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry starts a render, so N failed projects must not start N renders."""
    started = _record_started(monkeypatch)
    job_ids = _park_retryable_projects(client, monkeypatch, started, 3)
    _tighten_render_budget(monkeypatch)

    codes = [client.post(f"/api/jobs/{job_id}/retry").status_code for job_id in job_ids]
    assert codes == [200, 429, 429]
    assert len(started) == 1


def test_a_create_and_a_retry_compete_for_the_same_render_slot(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The policy says retry spends the slots a plain create would."""
    started = _record_started(monkeypatch)
    job_ids = _park_retryable_projects(client, monkeypatch, started, 1)
    _tighten_render_budget(monkeypatch)

    assert client.post("/api/jobs", json={"url": "https://youtu.be/slot0001", "mode": "api"}).status_code == 200
    assert client.post("/api/jobs", json={"url": "https://youtu.be/slot0002", "mode": "api"}).status_code == 429

    started.clear()
    assert client.post(f"/api/jobs/{job_ids[0]}/retry").status_code == 429
    assert started == []


def test_waveform_read_does_not_get_a_private_quota_per_project(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The waveform GET runs ffmpeg, so it must share one bucket too."""
    calls: list = []

    def fake_media_command(job_id: str, args: list, timeout: float = 90.0) -> object:
        calls.append(job_id)
        return subprocess.CompletedProcess(args, 0, b"\x00\x40" * 400, b"")

    monkeypatch.setattr(studio, "_run_media_command", fake_media_command)
    job_ids = [_local_project(client, monkeypatch, index) for index in range(2)]
    monkeypatch.setattr(studio, "_rate_limiter", SlidingWindowLimiter(limit=2, window_seconds=60))

    # A new ``bins`` value misses the cache, so each accepted request is a
    # second ffmpeg run rather than a served file.
    codes = [
        client.get(f"/api/jobs/{job_ids[0]}/waveform", params={"bins": bins}).status_code
        for bins in (32, 64, 96)
    ]
    codes.append(client.get(f"/api/jobs/{job_ids[1]}/waveform", params={"bins": 32}).status_code)
    assert codes == [200, 200, 429, 429]
    assert len(calls) == 2


def test_ffmpeg_draft_work_does_not_multiply_the_budget_by_project(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preview per project would otherwise be three ffmpeg runs per budget."""
    renders: list = []

    def fake_crop(*args: object, **_kwargs: object) -> None:
        renders.append(args[4])
        Path(str(args[4])).write_bytes(b"preview")

    monkeypatch.setattr(clipper, "crop_clip_local", fake_crop)
    job_ids = [_local_project(client, monkeypatch, index) for index in range(3)]
    _tighten_render_budget(monkeypatch)

    codes = [
        client.post(
            f"/api/jobs/{job_id}/preview",
            json={"start_time": 0.0, "end_time": 1.5, "caption_style": "bold"},
        ).status_code
        for job_id in job_ids
    ]
    assert codes == [200, 429, 429]
    assert len(renders) == 1


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


# A Cloudflare Stream signed URL keeps the token where the public video id sits
# (docs: customer-<CODE>.cloudflarestream.com/<TOKEN>/manifest/video.m3u8,
# /<TOKEN>/iframe).  The token is always a JWS compact serialization: three
# base64url segments whose header decodes to a JSON object.
_STREAM_TOKEN = (
    "eyJhbGciOiJSUzI1NiIsImtpZCI6ImFiY2RlZjEyMzQ1NiJ9"
    ".eyJzdWIiOiJ2aWRlbyIsImV4cCI6MTc1MDAwMDAwMH0"
    ".c2lnbmF0dXJlLXNpZ25hdHVyZS1zaWduYXR1cmUtc2lnbmF0dXJl"
)

# Realistic source URLs whose parameters *look* credential-ish but are part of
# how the source is fetched.  Each one is a shape an over-broad rule already
# broke once: ``si``/``t``/``list`` are share and seek parameters, ``size`` is a
# rendition hint, a self-hosted ``key`` is an application-level accessor, and the
# 32-character id in a public Stream URL is a video id, not a token.
_FUNCTIONAL_SOURCE_URLS = (
    pytest.param("https://www.youtube.com/watch?v=aBcD1234&si=SHARE_TOKEN&t=60", id="youtube-share"),
    pytest.param("https://youtu.be/aBcD1234?si=SHARE&list=PL123&size=large", id="youtube-playlist"),
    pytest.param("https://vimeo.com/123456789?share=copy&size=large", id="vimeo-share"),
    pytest.param("https://vimeo.com/123456789#t=60", id="vimeo-time-fragment"),
    pytest.param("https://self-hosted.example/media/clip.mp4?key=abc123&author=bob", id="self-hosted-key"),
    pytest.param(
        "https://customer-abc.cloudflarestream.com/ea95132c15732412d22c1476fa83f27a/manifest/video.m3u8",
        id="stream-public-video-id",
    ),
    pytest.param("https://cdn.example/a1b2c3d4e5f6a7b8c9d0/manifest/video.m3u8", id="opaque-path-segment"),
)

# The same download shapes carrying a real credential: the renderer must still
# receive the signed URL, while every saved/returned copy loses the secret but
# keeps any functional parameter.  ``gone`` are substrings that must appear in
# no returned or persisted copy (parameter names as well as values).
_CREDENTIAL_SOURCE_URLS = (
    pytest.param(
        "https://bucket.s3.us-east-1.amazonaws.com/video.mp4"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Credential=AKIAEXAMPLE%2F20260924%2Fus-east-1%2Fs3%2Faws4_request"
        "&X-Amz-Date=20260924T000000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host"
        "&X-Amz-Signature=deadbeefcafe",
        "https://bucket.s3.us-east-1.amazonaws.com/video.mp4",
        ("X-Amz-", "AKIAEXAMPLE", "deadbeefcafe"),
        id="presigned-s3",
    ),
    pytest.param(
        "https://storage.googleapis.com/b/video.mp4"
        "?X-Goog-Algorithm=GOOG4-RSA-SHA256"
        "&X-Goog-Credential=svc%40proj.iam.gserviceaccount.com%2F20260924"
        "&X-Goog-Date=20260924T000000Z&X-Goog-Expires=900&X-Goog-Signature=feedfacecafe",
        "https://storage.googleapis.com/b/video.mp4",
        ("X-Goog-", "iam.gserviceaccount.com", "feedfacecafe"),
        id="signed-gcs",
    ),
    pytest.param(
        "https://customer-abc.cloudflarestream.com/manifest/video.m3u8?token=CFSTREAMTOKEN123",
        "https://customer-abc.cloudflarestream.com/manifest/video.m3u8",
        ("token=", "CFSTREAMTOKEN123"),
        id="cloudflare-stream-token",
    ),
    pytest.param(
        "https://self-hosted.example/media/clip.mp4?key=abc123&viewer=bob&X-Amz-Signature=deadbeefcafe",
        "https://self-hosted.example/media/clip.mp4?key=abc123&viewer=bob",
        ("X-Amz-Signature", "deadbeefcafe"),
        id="self-hosted-key-plus-signature",
    ),
    pytest.param(
        f"https://customer-abc.cloudflarestream.com/{_STREAM_TOKEN}/manifest/video.m3u8",
        "https://customer-abc.cloudflarestream.com/manifest/video.m3u8",
        (_STREAM_TOKEN,),
        id="stream-signed-path-token",
    ),
)


@pytest.mark.parametrize("url", _FUNCTIONAL_SOURCE_URLS)
def test_realistic_functional_source_urls_survive_byte_for_byte(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """A functional source URL must reach the renderer and persist untouched."""
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    started = _capture_started(monkeypatch)

    response = client.post("/api/jobs", json={"url": url, "mode": "api"})
    assert response.status_code == 200
    job_id = response.json()["id"]

    assert started[job_id] == url
    assert response.json()["request"]["url"] == url

    with studio._lock:
        job = studio._jobs[job_id]
        assert job["request"]["url"] == url
        assert job["source_url_redacted"] is False

    mirror = json.loads((studio._jobs_dir / f"{job_id}.json").read_text(encoding="utf-8"))
    assert mirror["request"]["url"] == url


@pytest.mark.parametrize("url, persisted, gone", _CREDENTIAL_SOURCE_URLS)
def test_realistic_signed_source_urls_lose_credentials_in_every_saved_copy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, url: str, persisted: str, gone: tuple
) -> None:
    """Signed download URLs keep working but persist without their credential."""
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    started = _capture_started(monkeypatch)

    response = client.post("/api/jobs", json={"url": url, "mode": "api"})
    assert response.status_code == 200
    job_id = response.json()["id"]

    # The renderer still gets the URL the download needs to succeed.
    assert started[job_id] == url

    returned = json.dumps(response.json())
    with studio._lock:
        job = studio._jobs[job_id]
        stored = job["request"]["url"]
        assert job["source_url_redacted"] is True
    mirror = (studio._jobs_dir / f"{job_id}.json").read_text(encoding="utf-8")

    for secret in gone:
        assert secret not in returned
        assert secret not in stored
        assert secret not in mirror
    # The functional remainder is kept, not emptied out with the credential.
    assert stored == persisted
    assert persisted in returned
    assert json.dumps(persisted)[1:-1] in mirror


def test_signed_path_tokens_are_scrubbed_by_shape_only() -> None:
    """A path-signing provider loses its token; every other path is left alone."""
    host = "https://customer-abc.cloudflarestream.com"
    assert redact_url_query(f"{host}/{_STREAM_TOKEN}/manifest/video.m3u8") == f"{host}/manifest/video.m3u8"
    assert redact_url_query(f"{host}/{_STREAM_TOKEN}/iframe") == f"{host}/iframe"
    # The route survives next to the stripped token, and an expiry is not a
    # credential on its own, so ``exp`` stays.
    downloads = redact_url_query(f"{host}/{_STREAM_TOKEN}/downloads/default.mp4?exp=1750000000")
    assert downloads == f"{host}/downloads/default.mp4?exp=1750000000"

    # A public Stream URL puts the video id in the same position: not a credential.
    public = f"{host}/ea95132c15732412d22c1476fa83f27a/manifest/video.m3u8"
    assert redact_url_query(public) == public
    # An opaque-looking segment on an unrecognised host is still a path.
    unknown = "https://cdn.example/a1b2c3d4e5f6a7b8c9d0/manifest/video.m3u8"
    assert redact_url_query(unknown) == unknown

    assert _STREAM_TOKEN not in redact_text(f"GET {host}/{_STREAM_TOKEN}/manifest/video.m3u8 failed")
    scrubbed = redact_url_query(f"{host}/{_STREAM_TOKEN}/manifest/video.m3u8")
    assert redact_url_query(scrubbed) == scrubbed


def test_factory_manifest_source_reads_the_shared_credential_policy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest must not carry a narrower credential list of its own."""
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    _capture_started(monkeypatch)

    # A seek offset in the fragment is functional, not a credential.
    functional = "https://www.youtube.com/watch?v=aBcD1234&si=SHARE&t=60#t=60"
    shared_id = client.post("/api/jobs", json={"url": functional, "mode": "api"}).json()["id"]
    shared = client.get(f"/api/v1/jobs/{shared_id}/factory")
    assert shared.status_code == 200
    assert shared.json()["source"] == {"kind": "url", "value": functional}

    # An implicit-flow token living only in the fragment is a credential.
    fragment = "https://x.example/cb#access_token=FRAGSECRET"
    fragment_id = client.post("/api/jobs", json={"url": fragment, "mode": "api"}).json()["id"]
    fragment_source = client.get(f"/api/v1/jobs/{fragment_id}/factory").json()["source"]
    assert fragment_source == {"kind": "url", "value": "https://x.example/cb"}

    signed = (
        "https://bucket.s3.amazonaws.com/video.mp4"
        "?X-Amz-Signature=deadbeefcafe&X-Amz-Credential=AKIAEXAMPLE&X-Amz-Expires=900"
    )
    signed_id = client.post("/api/jobs", json={"url": signed, "mode": "api"}).json()["id"]
    package = client.get(f"/api/v1/jobs/{signed_id}/factory")
    assert package.status_code == 200
    body = json.dumps(package.json())
    for leaked in ("X-Amz-", "deadbeefcafe", "AKIAEXAMPLE"):
        assert leaked not in body
    assert package.json()["source"] == {
        "kind": "url",
        "value": "https://bucket.s3.amazonaws.com/video.mp4",
    }


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


def test_completed_render_leaves_no_credential_in_any_persisted_field(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drives the real completion path, which stores its own source URL.

    Every other URL test asserts against a record the enqueue path built, which
    is why the completion fields could leak unnoticed: after the pipeline hands
    back the URL it was given, the worker stores that value in
    ``raw_source_video_url`` and inside ``result``, so scrubbing only the
    request URL leaves a credential on disk.
    """
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    signed = f"https://customer-abc.cloudflarestream.com/{_STREAM_TOKEN}/manifest/video.m3u8"
    scrubbed = "https://customer-abc.cloudflarestream.com/manifest/video.m3u8"
    handed: dict = {}
    monkeypatch.setattr(
        studio,
        "_start_job_thread",
        lambda job_id, req, *args, **kwargs: handed.__setitem__(job_id, req),
    )
    # The API-mode pipeline echoes back the URL it was handed, as it does live.
    monkeypatch.setattr(
        studio,
        "generate_shorts",
        lambda **kwargs: {
            "mode": "api",
            "source_video_url": kwargs["youtube_url"],
            "transcript": {},
            "highlights": [],
            "shorts": [],
            "llm": {},
        },
    )

    response = client.post("/api/jobs", json={"url": signed, "mode": "api"})
    assert response.status_code == 200
    job_id = response.json()["id"]

    # The downloader must still be handed the URL exactly as supplied.
    assert handed[job_id].url == signed
    studio._run_job(job_id, handed[job_id], None)

    with studio._lock:
        job = studio._jobs[job_id]
        assert job["status"] == "done"
        assert job["source_url_redacted"] is True
        assert job["request"]["url"] == scrubbed
        # The field the completion path adds, and its twin inside the result.
        assert job["raw_source_video_url"] == scrubbed
        assert job["result"]["source_video_url"] == scrubbed

    # Every durable artifact this path produces: the portable mirror, the
    # manifest beside the rendered media, and anything added later that lands
    # under the job directory.
    artifacts = sorted(path for path in studio._jobs_dir.rglob("*") if path.is_file())
    assert (studio._jobs_dir / f"{job_id}.json") in artifacts
    assert (studio._jobs_dir / job_id / "metadata.json") in artifacts
    for path in artifacts:
        assert _STREAM_TOKEN not in path.read_text(encoding="utf-8", errors="ignore"), f"token in {path}"
    database = studio._jobs_db_path
    for path in (database, database.with_name(database.name + "-wal")):
        if path.is_file():
            assert _STREAM_TOKEN.encode() not in path.read_bytes(), f"token in {path}"

    # A downloaded copy must not carry it either.
    export = client.get(f"/api/v1/jobs/{job_id}/export")
    assert export.status_code == 200
    assert _STREAM_TOKEN.encode() not in export.content

    # The flagged record must refuse to fetch its stripped address again.
    with studio._lock:
        studio._jobs[job_id]["status"] = "interrupted"
        studio._persist_job_locked(studio._jobs[job_id])
    studio._resume_interrupted_jobs()
    with studio._lock:
        assert studio._jobs[job_id]["status"] == "error"
        assert "credentials" in str(studio._jobs[job_id]["message"])
    retry = client.post(f"/api/jobs/{job_id}/retry")
    assert retry.status_code == 409


def test_signed_hosted_media_url_does_not_make_a_project_unresumable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """API mode stores the hosted media URL, which is not what resume fetches."""
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    source = "https://www.youtube.com/watch?v=aBcD1234&si=SHARE&t=60"
    hosted = f"https://customer-abc.cloudflarestream.com/{_STREAM_TOKEN}/manifest/video.m3u8"
    handed: dict = {}
    monkeypatch.setattr(
        studio,
        "_start_job_thread",
        lambda job_id, req, *args, **kwargs: handed.__setitem__(job_id, req),
    )
    monkeypatch.setattr(
        studio,
        "generate_shorts",
        lambda **kwargs: {
            "mode": "api",
            "source_video_url": hosted,
            "transcript": {},
            "highlights": [],
            "shorts": [],
            "llm": {},
        },
    )

    response = client.post("/api/jobs", json={"url": source, "mode": "api"})
    job_id = response.json()["id"]
    studio._run_job(job_id, handed[job_id], None)

    with studio._lock:
        job = studio._jobs[job_id]
        # The hosted URL is scrubbed for storage, but it is not the fetch URL.
        assert job["request"]["url"] == source
        assert job["raw_source_video_url"] == "https://customer-abc.cloudflarestream.com/manifest/video.m3u8"
        assert job["source_url_redacted"] is False
        studio._jobs[job_id]["status"] = "interrupted"
        studio._persist_job_locked(studio._jobs[job_id])

    resumed: dict = {}
    monkeypatch.setattr(
        studio,
        "_start_job_thread",
        lambda job_id, req, *args, **kwargs: resumed.__setitem__(job_id, req),
    )
    studio._resume_interrupted_jobs()
    assert job_id in resumed
    with studio._lock:
        assert studio._jobs[job_id]["status"] != "error"


def test_credential_shaped_project_name_does_not_block_retry(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project name is free text, not the source URL the fetch depends on."""
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    _capture_started(monkeypatch)
    response = client.post(
        "/api/jobs", json={"url": "https://youtu.be/aBcD1234", "mode": "api"}
    )
    job_id = response.json()["id"]

    renamed = client.patch(
        f"/api/jobs/{job_id}", json={"name": "https://cdn.example/v.mp4?token=abc123"}
    )
    assert renamed.status_code == 200

    with studio._lock:
        job = studio._jobs[job_id]
        assert "abc123" not in json.dumps(job)  # the credential is not stored
        assert job["source_url_redacted"] is False
        assert job["request"]["url"] == "https://youtu.be/aBcD1234"
        job["status"] = "error"
        studio._persist_job_locked(job)

    retry = client.post(f"/api/jobs/{job_id}/retry")
    assert retry.status_code == 200


@pytest.mark.parametrize(
    "url",
    (
        pytest.param("https://cdn.example/video.mp4?access_token=LEAKEDSECRET", id="query-credential"),
        pytest.param(
            f"https://customer-abc.cloudflarestream.com/{_STREAM_TOKEN}/manifest/video.m3u8",
            id="signed-path-token",
        ),
    ),
)
def test_redacted_source_url_is_never_reused_for_resume_or_retry(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    _capture_started(monkeypatch)
    response = client.post("/api/jobs", json={"url": url, "mode": "api"})
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


# ---------------------------------------------------------------------------
# The retry affordance must agree with the refusal it is paired with
# ---------------------------------------------------------------------------

# Pinned literally so a wording change has to break a test on purpose.  The
# project card paraphrases this text, so the two move together.
_PINNED_SOURCE_URL_REDACTED_MESSAGE = (
    "This project's source URL carried credentials that are not stored, so it cannot be fetched again. "
    "Submit the URL to start a new render."
)

_RETRYABLE_STATUSES = ("error", "cancelled", "interrupted", "draft")


def _enqueue_and_park(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, url: str, status: str
) -> str:
    """Enqueue through the real endpoint, then park the record in a retryable state.

    The record itself is always built by ``POST /api/jobs`` so the fetch/stored
    URL split is the real one; only the terminal status is forced, because the
    suite never runs a render.  The limiters are raised so the shared per-minute
    job budget other tests spend cannot decide whether these assertions hold.
    """
    monkeypatch.setattr(studio, "_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_upload_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    _capture_started(monkeypatch)
    response = client.post("/api/jobs", json={"url": url, "mode": "api"})
    assert response.status_code == 200
    job_id = response.json()["id"]
    with studio._lock:
        job = studio._jobs[job_id]
        job["status"] = status
        job["stage"] = status
        studio._persist_job_locked(job)
    return job_id


def _card_payload(client: TestClient, job_id: str) -> dict:
    """The project grid renders from the list route, not the single-job route."""
    listing = client.get("/api/jobs")
    assert listing.status_code == 200
    cards = {job["id"]: job for job in listing.json()["jobs"]}
    assert job_id in cards, "the queued project is missing from the list the grid renders"
    return cards[job_id]


@pytest.mark.parametrize("status", _RETRYABLE_STATUSES)
def test_stripped_source_is_never_offered_retry_by_any_endpoint(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """A stripped source URL cannot be fetched, so no snapshot may advertise Retry.

    Regression: ``can_retry`` ignored the flag, so the card offered a Retry/Run
    button whose only possible outcome was a 409 -- the dead end this gate
    exists to remove.  Both endpoints are checked because the grid renders from
    the list route while the workspace renders from the single-job route.
    """
    signed = f"https://customer-abc.cloudflarestream.com/{_STREAM_TOKEN}/manifest/video.m3u8"
    job_id = _enqueue_and_park(client, monkeypatch, signed, status)

    single = client.get(f"/api/jobs/{job_id}")
    assert single.status_code == 200
    assert single.json()["source_url_redacted"] is True
    assert single.json()["can_retry"] is False

    card = _card_payload(client, job_id)
    assert card["source_url_redacted"] is True
    assert card["can_retry"] is False


@pytest.mark.parametrize("status", ("error", "interrupted"))
def test_ordinary_failure_with_intact_url_stays_retryable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """The honesty gate must not turn every failure into an unretryable record."""
    url = "https://www.youtube.com/watch?v=aBcD1234&si=SHARE&t=60"
    job_id = _enqueue_and_park(client, monkeypatch, url, status)

    single = client.get(f"/api/jobs/{job_id}")
    assert single.status_code == 200
    assert single.json()["source_url_redacted"] is False
    assert single.json()["can_retry"] is True

    card = _card_payload(client, job_id)
    assert card["source_url_redacted"] is False
    assert card["can_retry"] is True

    # And the advertised action really works rather than 409ing.
    retry = client.post(f"/api/jobs/{job_id}/retry")
    assert retry.status_code == 200


def test_flagged_record_refusal_wording_is_unchanged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume and retry refuse a flagged record with the exact published wording."""
    assert studio._SOURCE_URL_REDACTED_MESSAGE == _PINNED_SOURCE_URL_REDACTED_MESSAGE

    job_id = _enqueue_and_park(
        client, monkeypatch, "https://cdn.example/video.mp4?access_token=LEAKEDSECRET", "interrupted"
    )

    # Resume must leave the record alone rather than fetch the stripped address.
    resumed: dict = {}
    monkeypatch.setattr(
        studio,
        "_start_job_thread",
        lambda job_id, req, *args, **kwargs: resumed.__setitem__(job_id, req),
    )
    studio._resume_interrupted_jobs()
    assert job_id not in resumed
    with studio._lock:
        job = studio._jobs[job_id]
        assert job["status"] == "error"
        assert job["message"] == _PINNED_SOURCE_URL_REDACTED_MESSAGE
        assert job["error"] == _PINNED_SOURCE_URL_REDACTED_MESSAGE

    retry = client.post(f"/api/jobs/{job_id}/retry")
    assert retry.status_code == 409
    assert retry.json() == {"error": _PINNED_SOURCE_URL_REDACTED_MESSAGE, "code": "http_409"}

    # The refusal survives the snapshot without leaking the credential either.
    returned = client.get(f"/api/jobs/{job_id}")
    assert returned.status_code == 200
    assert returned.json()["error"] == _PINNED_SOURCE_URL_REDACTED_MESSAGE
    assert returned.json()["can_retry"] is False
    assert "LEAKEDSECRET" not in json.dumps(returned.json())


# ---------------------------------------------------------------------------
# Credentials in the URL authority (basic-auth userinfo)
# ---------------------------------------------------------------------------

# RFC 3986 puts basic-auth credentials in the authority, which is how a
# self-hosted media server or a signed direct-download link usually carries
# them, so a policy that only knows query parameters and signed paths never
# sees the secret at all.
_BASIC_AUTH_SECRET = "SUPERSECRETBASICAUTH"
_BASIC_AUTH_URL = f"https://alice:{_BASIC_AUTH_SECRET}@self-hosted.example/media/clip.mp4?key=abc123"
_BASIC_AUTH_SCRUBBED = "https://self-hosted.example/media/clip.mp4?key=abc123"


def test_basic_auth_userinfo_loses_its_credential_without_rewriting_other_urls() -> None:
    """The credential lives in the authority, not in a query parameter."""
    assert redact_url_query(_BASIC_AUTH_URL) == _BASIC_AUTH_SCRUBBED
    assert _BASIC_AUTH_SECRET not in redact_text(_BASIC_AUTH_URL)
    assert _BASIC_AUTH_SECRET not in redact_text(f"GET {_BASIC_AUTH_URL} failed")

    # The durable write scrubs the whole-string URL field; a log line is prose,
    # and the log API hands every message through ``redact_text`` when serving.
    record = {
        "request": {"url": _BASIC_AUTH_URL},
        "raw_source_video_url": _BASIC_AUTH_URL,
        "logs": [{"message": f"GET {_BASIC_AUTH_URL} failed"}],
    }
    redact_record_urls(record)
    assert record["request"]["url"] == _BASIC_AUTH_SCRUBBED
    assert record["raw_source_video_url"] == _BASIC_AUTH_SCRUBBED
    assert _BASIC_AUTH_SECRET not in str(studio._redact_log_text(record["logs"][0]["message"]))

    # A username without a password is not a credential, and a URL that has no
    # credential keeps every byte, so scrubbing cannot break a live source.
    for untouched in (
        "https://viewer@self-hosted.example/media/clip.mp4",
        "https://self-hosted.example:8443/media/clip.mp4",
        "https://self-hosted.example/media/clip.mp4?key=abc&author=bob",
        "https://self-hosted.example/media/clip.mp4#t=60",
    ):
        assert redact_url_query(untouched) == untouched


def test_basic_auth_source_never_reaches_a_saved_or_returned_copy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drives the real API, the durable stores, the archives and the refusal."""
    monkeypatch.setattr(studio, "_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_upload_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")
    handed: dict = {}
    monkeypatch.setattr(
        studio,
        "_start_job_thread",
        lambda job_id, req, *args, **kwargs: handed.__setitem__(job_id, req),
    )
    monkeypatch.setattr(
        studio,
        "generate_shorts",
        lambda **kwargs: {
            "mode": "api",
            "source_video_url": kwargs["youtube_url"],
            "transcript": {},
            "highlights": [],
            "shorts": [],
            "llm": {},
        },
    )

    response = client.post("/api/jobs", json={"url": _BASIC_AUTH_URL, "mode": "api"})
    assert response.status_code == 200
    job_id = response.json()["id"]

    # The renderer is still handed the URL that actually needs the credential.
    assert handed[job_id].url == _BASIC_AUTH_URL
    assert _BASIC_AUTH_SECRET not in json.dumps(response.json())

    studio._run_job(job_id, handed[job_id], None)

    with studio._lock:
        job = studio._jobs[job_id]
        assert job["status"] == "done"
        # The stored copy keeps the functional remainder and is flagged, so a
        # retry or resume cannot fetch the stripped address by accident.
        assert job["request"]["url"] == _BASIC_AUTH_SCRUBBED
        assert job["raw_source_video_url"] == _BASIC_AUTH_SCRUBBED
        assert job["result"]["source_video_url"] == _BASIC_AUTH_SCRUBBED
        assert job["source_url_redacted"] is True

    # Every durable artifact this path produces: the portable mirror, the
    # manifest beside the rendered media, and anything added later.
    for path in sorted(candidate for candidate in studio._jobs_dir.rglob("*") if candidate.is_file()):
        assert _BASIC_AUTH_SECRET not in path.read_text(encoding="utf-8", errors="ignore"), f"secret in {path}"
    database = studio._jobs_db_path
    for path in (database, database.with_name(database.name + "-wal")):
        if path.is_file():
            assert _BASIC_AUTH_SECRET.encode() not in path.read_bytes(), f"secret in {path}"

    # A returned copy, the project grid, and the two archives a creator shares.
    assert _BASIC_AUTH_SECRET not in json.dumps(client.get(f"/api/jobs/{job_id}").json())
    assert _BASIC_AUTH_SECRET not in json.dumps(client.get(f"/api/v1/jobs/{job_id}/factory").json())
    card = _card_payload(client, job_id)
    assert card["source_url_redacted"] is True
    assert card["can_retry"] is False
    for endpoint in ("/api/backup", f"/api/v1/jobs/{job_id}/export"):
        archive_response = client.get(endpoint)
        assert archive_response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
            for name in archive.namelist():
                assert _BASIC_AUTH_SECRET not in archive.read(name).decode("utf-8", "ignore"), (
                    f"secret in {endpoint}:{name}"
                )

    # The two log surfaces are prose, so the text policy has to catch a line
    # that quotes the address it was fetching.
    with studio._lock:
        studio._append_job_log(studio._jobs[job_id], "download", f"fetching {_BASIC_AUTH_URL}")
        studio._persist_job_locked(studio._jobs[job_id])
    listing = client.get("/api/logs", params={"job_id": job_id})
    assert listing.status_code == 200
    assert _BASIC_AUTH_SECRET not in json.dumps(listing.json())
    job_log = client.get(f"/api/jobs/{job_id}/logs")
    assert job_log.status_code == 200
    assert _BASIC_AUTH_SECRET not in job_log.text

    # The credential is never replayed to fetch the stripped address again.
    with studio._lock:
        studio._jobs[job_id]["status"] = "interrupted"
        studio._persist_job_locked(studio._jobs[job_id])
    studio._resume_interrupted_jobs()
    with studio._lock:
        assert studio._jobs[job_id]["status"] == "error"
    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 409


# ---------------------------------------------------------------------------
# A credential the app itself holds is not always URL-shaped
# ---------------------------------------------------------------------------

# Session keys arrive in headers, live only in memory, and can be echoed back
# by the dependency that rejected them -- the exact case the text policy exists
# for.  They are fake, and recognisable, so a byte scan can find one anywhere.
_SESSION_MUAPI_KEY = "session_muapi_credential_1234567890"
_SESSION_OPENAI_KEY = "session_openai_credential_0987654321"


def _run_render(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    generator,
    key: str,
    narration: str,
) -> str:
    """Drive one real render through the API, with only the pipeline replaced.

    Everything around the pipeline is production: the worker thread, the
    ``progress`` writer, the completion write, the durable store and the
    portable manifest.  The limiters are raised so the shared per-minute job
    budget other tests spend cannot decide whether these assertions hold.
    """
    monkeypatch.setattr(studio, "_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_upload_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "_job_rate_limiter", SlidingWindowLimiter(limit=1000, window_seconds=60))
    monkeypatch.setattr(studio, "MUAPI_API_KEY", "mu_test_key_1234567890")

    def pipeline(**kwargs):
        progress = kwargs.get("progress")
        if progress:
            progress("download", narration)
        return generator()

    monkeypatch.setattr(studio, "generate_shorts", pipeline)
    response = client.post(
        "/api/jobs",
        headers={"X-MuAPI-Key": key, "X-OpenAI-Key": _SESSION_OPENAI_KEY},
        json={"url": "https://youtu.be/aBcD1234", "mode": "api"},
    )
    assert response.status_code == 200
    job_id = response.json()["id"]
    deadline = time.time() + 30.0
    status = ""
    while time.time() < deadline:
        with studio._lock:
            status = str((studio._jobs.get(job_id) or {}).get("status"))
        if status in {"done", "error", "cancelled"}:
            return job_id
        time.sleep(0.02)
    raise AssertionError(f"render never reached a terminal state (status={status})")


def _durable_copies(job_id: str) -> dict:
    """Every copy a credential must not reach: record, mirror, store, manifest."""
    copies = {
        "record": json.dumps(studio._jobs.get(job_id), default=str, ensure_ascii=False).encode("utf-8"),
        "mirror": studio._job_path(job_id).read_bytes(),
    }
    database = studio._jobs_db_path
    for suffix in ("", "-wal", "-shm"):
        path = database.with_name(database.name + suffix)
        if path.is_file():
            copies[f"sqlite{suffix or '-main'}"] = path.read_bytes()
    manifest = studio._jobs_dir / job_id / "metadata.json"
    if manifest.is_file():
        copies["metadata"] = manifest.read_bytes()
    return copies


def _assert_credential_never_returned_or_stored(client: TestClient, job_id: str, key: str) -> None:
    """Scan every returned view and every durable copy for the session key."""
    for name, path in (
        ("single", f"/api/jobs/{job_id}"),
        ("grid", "/api/jobs"),
        ("factory", f"/api/jobs/{job_id}/factory"),
        ("timeline", f"/api/jobs/{job_id}/timeline"),
        ("logs", f"/api/logs?job_id={job_id}"),
        ("job_logs", f"/api/jobs/{job_id}/logs"),
        ("export", f"/api/jobs/{job_id}/export"),
        ("backup", "/api/backup?include_media=true"),
    ):
        response = client.get(path)
        assert response.status_code == 200, name
        assert key not in response.content.decode("utf-8", "ignore"), f"credential returned by {name}"
    for name, blob in _durable_copies(job_id).items():
        assert key not in blob.decode("utf-8", "ignore"), f"credential stored in {name}"


def test_dependency_echo_in_a_progress_message_is_scrubbed_everywhere(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A narration that quotes the key must not persist one copy of it raw."""
    key = _SESSION_MUAPI_KEY
    narration = f"GET https://api.muapi.ai/v1/render?api_key={key} failed"
    job_id = _run_render(
        client,
        monkeypatch,
        lambda: {
            "mode": "api",
            "source_video_url": "https://cdn.example/v.mp4",
            "shorts": [],
            "transcript": {},
        },
        key,
        narration,
    )

    with studio._lock:
        checkpoint = json.dumps(studio._jobs[job_id]["checkpoint"], default=str)
    assert key not in checkpoint
    # Scrubbed, not dropped: the narration the user needs is still there.
    assert "api_key=[redacted]" in checkpoint
    _assert_credential_never_returned_or_stored(client, job_id, key)


def test_dependency_payload_echo_is_scrubbed_from_every_copy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider payload that echoes the key must not reach a result field raw."""
    key = _SESSION_OPENAI_KEY
    job_id = _run_render(
        client,
        monkeypatch,
        lambda: {
            "mode": "api",
            "source_video_url": "https://cdn.example/v.mp4",
            "highlights": [{"start": 0.0, "end": 1.0, "note": f"api_key={key}"}],
            "shorts": [
                {
                    "index": 0,
                    "title": f"clip echoing {key}",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "virality_reason": f"provider said {key}",
                }
            ],
            "transcript": {"segments": [{"start": 0.0, "end": 1.0, "text": f"said {key}"}]},
            "llm": {"provider": "openai", "model": "x", "usage": {"api_key": key}},
        },
        key,
        "Fetching source video...",
    )

    with studio._lock:
        stored = json.dumps(studio._jobs[job_id], default=str)
    assert key not in stored
    # The payload is still readable; only the credential inside it is gone.
    assert "clip echoing [redacted]" in stored
    _assert_credential_never_returned_or_stored(client, job_id, key)


def test_ordinary_narration_is_left_byte_for_byte(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The value policy must not rewrite a record that holds no credential."""
    narration = "Cropping 3 of 10 candidates..."
    job_id = _run_render(
        client,
        monkeypatch,
        lambda: {
            "mode": "api",
            "source_video_url": "https://cdn.example/v.mp4",
            "shorts": [],
            "transcript": {},
        },
        _SESSION_MUAPI_KEY,
        narration,
    )

    with studio._lock:
        job = studio._jobs[job_id]
        checkpoint_message = str(job["checkpoint"]["message"])
        log_messages = [entry["message"] for entry in job["logs"]]
    assert checkpoint_message == narration
    assert narration in log_messages
    mirror = studio._job_path(job_id).read_bytes()
    assert narration.encode("utf-8") in mirror
    # Nothing was rewritten, because nothing in the record was a credential.
    assert b"[redacted]" not in mirror
