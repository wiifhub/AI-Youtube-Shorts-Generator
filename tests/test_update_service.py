"""Release update policy tests (no GitHub or process restart required)."""

from __future__ import annotations

import hashlib
import types
from pathlib import Path

from web.update_service import UpdateService


def _service(tmp_path: Path, state: dict) -> UpdateService:
    return UpdateService(
        repo="wiifhub/AI-Youtube-Shorts-Generator",
        current_version="0.10.0",
        data_root=tmp_path,
        state_setter=lambda **values: (state.update(values) or dict(state)),
        state_getter=lambda: dict(state),
    )


def test_release_policy_parses_versions_and_never_installs_source_checkout(tmp_path: Path) -> None:
    service = _service(tmp_path, {})
    assert service.version_tuple("v1.2") == (1, 2, 0)
    assert service.version_tuple("not-a-version") == (0, 0, 0)

    info = service.release_info(
        {
            "tag_name": "v0.11.0",
            "html_url": "https://github.com/wiifhub/AI-Youtube-Shorts-Generator/releases/tag/v0.11.0",
            "assets": [{"name": "ShortsStudio-v0.11.0-windows.zip", "browser_download_url": "https://github.com/a.zip"}],
        }
    )
    assert info["update_available"] is True
    assert info["can_install"] is False
    assert service.select_release_asset({"assets": []}) is None


def test_download_rejects_non_github_urls_and_records_safe_error(tmp_path: Path) -> None:
    state = {}
    service = _service(tmp_path, state)

    service.download_and_apply({"name": "update.zip", "url": "http://example.com/update.zip", "kind": "zip"}, "0.11.0")

    assert state["status"] == "error"
    assert "approved GitHub HTTPS host" in state["error"]


def test_release_digest_is_checked_before_install(tmp_path: Path, monkeypatch) -> None:
    state = {}
    service = _service(tmp_path, state)
    service.require_digest = True
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: type(
            "Response",
            (),
            {
                "headers": {"content-length": "4"},
                # A real streamed response always reports the URL it ended on.
                "url": "https://github.com/example/update.zip",
                "raise_for_status": lambda self: None,
                "iter_content": lambda self, chunk_size: [b"data"],
                "__enter__": lambda self: self,
                "__exit__": lambda self, *exc: None,
            },
        )(),
    )
    service.download_and_apply(
        {"name": "update.zip", "url": "https://github.com/example/update.zip", "kind": "zip"},
        "0.11.0",
    )
    assert "release digest" in state["error"]


class _StubResponse:
    """A streamed response whose final URL need not match the one requested."""

    def __init__(self, body: bytes, url: str) -> None:
        self._body = body
        self.url = url
        self.headers = {"content-length": str(len(body))}
        self.history: list = []

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 1024):
        yield self._body

    def __enter__(self) -> "_StubResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _NoOpThread:
    """Keeps the restart thread from ending the test run."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        return None

    def start(self) -> None:
        return None


def _armed_service(tmp_path: Path, state: dict, monkeypatch) -> tuple[UpdateService, list]:
    """A service that records what it would install instead of installing it."""
    service = _service(tmp_path, state)
    launched: list = []
    monkeypatch.setattr(service, "launch", lambda asset, path: launched.append(Path(path)))
    monkeypatch.setattr("web.update_service.threading", types.SimpleNamespace(Thread=_NoOpThread))
    return service, launched


def test_download_rejects_a_redirect_to_a_non_github_host(tmp_path: Path, monkeypatch) -> None:
    """The allowlist has to cover where the bytes came from, not just the URL asked for.

    ``requests`` follows redirects by default, so an approved ``github.com`` URL
    can answer with bytes served from anywhere; before this the redirect target
    was never inspected and the download was installed.
    """
    state: dict = {}
    service, launched = _armed_service(tmp_path, state, monkeypatch)
    service.require_digest = False  # no digest, so nothing else pins the bytes
    monkeypatch.setattr(
        "requests.get", lambda *args, **kwargs: _StubResponse(b"payload", "https://evil.example/payload.zip")
    )

    service.download_and_apply(
        {"name": "update.zip", "url": "https://github.com/example/update.zip", "kind": "zip"},
        "0.11.0",
    )

    assert launched == []
    assert state["status"] == "error"
    assert "redirected outside" in state["error"]
    assert not list((tmp_path / "updates").glob("*.zip"))


def test_redirect_check_holds_even_when_the_digest_matches(tmp_path: Path, monkeypatch) -> None:
    """A digest that agrees with the bytes cannot vouch for where they came from."""
    state: dict = {}
    service, launched = _armed_service(tmp_path, state, monkeypatch)
    service.require_digest = True
    body = b"payload"
    monkeypatch.setattr(
        "requests.get", lambda *args, **kwargs: _StubResponse(body, "https://evil.example/payload.zip")
    )

    service.download_and_apply(
        {
            "name": "update.zip",
            "url": "https://github.com/example/update.zip",
            "kind": "zip",
            "digest": "sha256:" + hashlib.sha256(body).hexdigest(),
        },
        "0.11.0",
    )

    assert launched == []
    assert state["status"] == "error"
    assert "redirected outside" in state["error"]


def test_a_redirect_inside_the_approved_hosts_still_installs(tmp_path: Path, monkeypatch) -> None:
    """GitHub hands release downloads to its own asset hosts, so that has to keep working."""
    state: dict = {}
    service, launched = _armed_service(tmp_path, state, monkeypatch)
    body = b"payload"
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: _StubResponse(body, "https://release-assets.githubusercontent.com/github-production-release-asset/1/update.zip"),
    )

    service.download_and_apply(
        {
            "name": "update.zip",
            "url": "https://github.com/example/update.zip",
            "kind": "zip",
            "digest": "sha256:" + hashlib.sha256(body).hexdigest(),
        },
        "0.11.0",
    )

    assert state["status"] == "restarting"
    assert [path.name for path in launched] == ["update.zip"]
