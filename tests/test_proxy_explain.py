"""Tests for requests.proxy_explain.

These tests use mock environments (via ``monkeypatch``) so they are
deterministic and never depend on the host's proxy configuration.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from requests import Session, explain_proxy
from requests.adapters import HTTPAdapter
from requests.proxy_explain import (
    LAYER_ENVIRONMENT,
    LAYER_REQUEST,
    LAYER_SESSION,
    ProxyExplanation,
)
from requests.utils import select_proxy

# A clean environment with no proxy variables, used as a baseline.
_CLEAN_ENV = {
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "ALL_PROXY": "",
    "NO_PROXY": "",
    "http_proxy": "",
    "https_proxy": "",
    "all_proxy": "",
    "no_proxy": "",
}


@pytest.fixture
def mock_env(monkeypatch):
    """Provide a clean environment with no proxy variables."""
    for key, value in _CLEAN_ENV.items():
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


class RecordingAdapter(HTTPAdapter):
    """Adapter that records the proxies mapping it would use, without
    sending anything."""

    def __init__(self):
        super().__init__()
        self.recorded_proxies = None

    def send(self, request, **kwargs):
        self.recorded_proxies = kwargs.get("proxies")
        raise RuntimeError("recording adapter: no request sent")


def _make_session_with_recording_adapter():
    session = Session()
    adapter = RecordingAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session, adapter


def _actual_send_path_proxy(session, url, proxies=None):
    """Run the real send path up to adapter.send and return the proxy the
    adapter would select."""
    try:
        session.request("GET", url, proxies=proxies)
    except RuntimeError as exc:
        if str(exc) != "recording adapter: no request sent":
            raise
    return select_proxy(url, session._recording_adapter.recorded_proxies)


def test_three_layers_source_accurate(mock_env):
    """When request, session and environment all configure proxies, the
    reported source must identify the layer that provided the value."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")
    mock_env.setenv("HTTPS_PROXY", "http://env-https:8080")
    mock_env.setenv("ALL_PROXY", "socks5://env-all:1080")

    session = Session()
    session.proxies = {
        "http": "http://sess-http:8080",
        "all": "socks5://sess-all:1080",
    }

    # Request-level wins for https.
    explanation = explain_proxy(
        "https://example.com",
        proxies={"https": "http://req-https:9090"},
        session=session,
    )
    assert explanation.proxy == "http://req-https:9090"
    assert explanation.proxy_source == LAYER_REQUEST

    # No request-level https -> environment fills it (env > session in the
    # request flow).
    explanation = explain_proxy("https://example.com", session=session)
    assert explanation.proxy == "http://env-https:8080"
    assert explanation.proxy_source == LAYER_ENVIRONMENT

    # http: environment provides it, beating session.
    explanation = explain_proxy("http://example.com", session=session)
    assert explanation.proxy == "http://env-http:8080"
    assert explanation.proxy_source == LAYER_ENVIRONMENT

    # all: environment provides it, beating session.  ftp has no scheme
    # proxy, so the "all" key is what matches.
    explanation = explain_proxy("ftp://example.com", session=session)
    assert explanation.proxy == "socks5://env-all:1080"
    assert explanation.proxy_source == LAYER_ENVIRONMENT


def test_no_proxy_loopback_shows_direct(mock_env):
    """no_proxy covering the loopback target must report a direct
    connection."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")
    mock_env.setenv("NO_PROXY", "127.0.0.1,localhost")

    explanation = explain_proxy("http://127.0.0.1/")
    assert explanation.bypassed is True
    assert explanation.proxy is None
    assert explanation.proxy_source is None
    assert explanation.no_proxy == "127.0.0.1,localhost"
    assert explanation.no_proxy_source == LAYER_ENVIRONMENT

    # localhost also matches.
    explanation = explain_proxy("http://localhost/")
    assert explanation.bypassed is True
    assert explanation.proxy is None


def test_no_proxy_from_request_layer(mock_env):
    """no_proxy supplied on the request layer is reported as such."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")

    explanation = explain_proxy("http://127.0.0.1/", proxies={"no_proxy": "127.0.0.1"})
    assert explanation.bypassed is True
    assert explanation.proxy is None
    assert explanation.no_proxy_source == LAYER_REQUEST


def test_explicit_empty_no_proxy_disables_env(mock_env):
    """An explicitly empty no_proxy must shadow the environment's no_proxy
    (and therefore not bypass)."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")
    mock_env.setenv("NO_PROXY", "127.0.0.1")

    explanation = explain_proxy("http://127.0.0.1/", proxies={"no_proxy": ""})
    assert explanation.bypassed is False
    assert explanation.proxy == "http://env-http:8080"
    assert explanation.no_proxy == ""
    assert explanation.no_proxy_source == LAYER_REQUEST


def test_authenticated_url_does_not_expose_password(mock_env):
    """Proxy URLs embedding credentials must be redacted."""
    explanation = explain_proxy(
        "http://example.com",
        proxies={"http": "http://user:secret@proxy.example.com:8080"},
    )
    assert explanation.proxy_auth is True
    assert "secret" not in explanation.proxy
    assert "***@proxy.example.com:8080" in explanation.proxy
    assert "secret" not in str(explanation)
    assert "secret" not in json.dumps(explanation.to_dict())

    # A proxy without credentials reports auth=False.
    explanation = explain_proxy(
        "http://example.com", proxies={"http": "http://proxy.example.com:8080"}
    )
    assert explanation.proxy_auth is False
    assert explanation.proxy == "http://proxy.example.com:8080"


def test_missing_not_treated_as_empty_override(mock_env):
    """Missing request proxies (None) must not override session/environment;
    an explicit empty mapping behaves the same for selection but is
    distinguished in the explanation."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")

    session = Session()
    session.proxies = {"http": "http://sess-http:8080"}

    # Missing (None): environment fills the key, selection identical to
    # explicit empty.
    missing = explain_proxy("http://example.com", proxies=None, session=session)
    empty = explain_proxy("http://example.com", proxies={}, session=session)
    assert missing.proxy == empty.proxy == "http://env-http:8080"
    assert missing.request_proxies_missing is True
    assert empty.request_proxies_missing is False

    # An explicit None value on the request layer blocks both environment
    # and session.
    blocked = explain_proxy(
        "http://example.com",
        proxies={"http": None},
        session_proxies={"http": "http://sess-http:8080"},
    )
    assert blocked.proxy is None
    assert blocked.proxy_source is None


def test_trust_env_false(mock_env):
    """When trust_env is disabled, environment proxies are ignored."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")

    explanation = explain_proxy(
        "http://example.com",
        session_proxies={"http": "http://sess-http:8080"},
        trust_env=False,
    )
    assert explanation.trust_env is False
    assert explanation.proxy == "http://sess-http:8080"
    assert explanation.proxy_source == LAYER_SESSION


def test_explicit_environment_mapping(mock_env):
    """The ``env`` parameter allows a fully controlled mock environment."""
    env = {
        "HTTP_PROXY": "http://mocked-http:8080",
        "NO_PROXY": "internal.example.com",
    }
    explanation = explain_proxy("http://example.com", env=env)
    assert explanation.proxy == "http://mocked-http:8080"
    assert explanation.proxy_source == LAYER_ENVIRONMENT

    explanation = explain_proxy("http://internal.example.com", env=env)
    assert explanation.bypassed is True
    assert explanation.proxy is None


def test_no_environment_mutation(mock_env):
    """The explainer must not mutate the process environment."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")
    mock_env.setenv("NO_PROXY", "127.0.0.1")

    before = dict(os.environ)
    explain_proxy("http://example.com")
    explain_proxy("http://127.0.0.1/")
    assert dict(os.environ) == before


def test_consistency_with_send_path(mock_env):
    """The explainer's selection must match the real Session send path."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")
    mock_env.setenv("HTTPS_PROXY", "http://env-https:8080")
    mock_env.setenv("ALL_PROXY", "socks5://env-all:1080")
    mock_env.setenv("NO_PROXY", "127.0.0.1,localhost")

    session, adapter = _make_session_with_recording_adapter()
    session.proxies = {"http": "http://sess-http:8080"}
    # Attach the adapter to the session for the helper.
    session._recording_adapter = adapter

    cases = [
        ("http://example.com", None),
        ("https://example.com", None),
        ("http://127.0.0.1/", None),
        ("http://example.com", {"http": "http://req-http:9090"}),
        ("https://example.com", {"https": "http://req-https:9090"}),
        ("http://example.com", {"http": None}),
        ("http://127.0.0.1/", {"no_proxy": "127.0.0.1"}),
    ]
    for url, req_proxies in cases:
        actual = _actual_send_path_proxy(session, url, req_proxies)
        explanation = explain_proxy(url, proxies=req_proxies, session=session)
        assert actual == explanation.proxy, (
            f"mismatch for {url}: send path selected {actual!r}, "
            f"explainer selected {explanation.proxy!r}"
        )


def test_session_method(mock_env):
    """Session.explain_proxy uses the session's proxies and trust_env."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")

    session = Session()
    session.proxies = {"http": "http://sess-http:8080"}
    explanation = session.explain_proxy("http://example.com")
    assert isinstance(explanation, ProxyExplanation)
    # Environment wins over session in the request flow.
    assert explanation.proxy == "http://env-http:8080"
    assert explanation.proxy_source == LAYER_ENVIRONMENT


def test_to_dict_and_str(mock_env):
    """The explanation renders to a dict and a human-readable string."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")

    explanation = explain_proxy(
        "http://example.com",
        proxies={"http": "http://user:secret@proxy:8080"},
    )
    data = explanation.to_dict()
    assert data["url"] == "http://example.com"
    assert data["proxy_auth"] is True
    assert "secret" not in json.dumps(data)

    text = str(explanation)
    assert "Proxy selection for http://example.com" in text
    assert "***@proxy:8080" in text
    assert "secret" not in text


def test_cli_human_readable(mock_env):
    """The command-line interface prints a human-readable explanation."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")
    mock_env.setenv("NO_PROXY", "127.0.0.1")

    result = subprocess.run(
        [sys.executable, "-m", "requests.proxy_explain", "http://example.com"],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert result.returncode == 0, result.stderr
    assert "Proxy selection for http://example.com" in result.stdout
    assert "http://env-http:8080" in result.stdout


def test_cli_json(mock_env):
    """The --json flag emits a JSON explanation."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "requests.proxy_explain",
            "http://example.com",
            "--json",
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["proxy"] == "http://env-http:8080"
    assert data["proxy_source"] == LAYER_ENVIRONMENT


def test_cli_no_trust_env(mock_env):
    """The --no-trust-env flag disables environment proxies."""
    mock_env.setenv("HTTP_PROXY", "http://env-http:8080")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "requests.proxy_explain",
            "http://example.com",
            "--no-trust-env",
            "--session-proxies",
            '{"http": "http://sess-http:8080"}',
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert result.returncode == 0, result.stderr
    assert "http://sess-http:8080" in result.stdout
