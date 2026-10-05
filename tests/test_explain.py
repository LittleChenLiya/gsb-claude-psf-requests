"""Tests for the explicit proxy-selection explainer (requests.explain)."""

import os

import pytest

import requests
from requests.explain import (
    ProxyExplanation,
    _redact_proxy_url,
    explain_proxies,
)
from requests.models import PreparedRequest
from requests.utils import resolve_proxies, select_proxy


def send_path_proxy(session, url, request_proxies):
    """Compute the proxy the real Session send path would select."""
    # Session.request normalizes the argument with ``proxies or {}`` before
    # merging environment settings.
    merged = session.merge_environment_settings(
        url, request_proxies or {}, None, None, None
    )["proxies"]
    prep = PreparedRequest()
    prep.prepare(method="GET", url=url)
    resolved = resolve_proxies(prep, merged, session.trust_env)
    return select_proxy(url, resolved)


class TestThreeLayers:
    """With request, session and environment proxies all configured, the
    reported source of the final proxy must be accurate."""

    ENV = {
        "http_proxy": "http://env-proxy:3128",
        "https_proxy": "http://env-proxy:3129",
    }

    def make_session(self):
        session = requests.Session()
        session.proxies = {
            "http": "http://session-proxy:8080",
            "ftp": "http://session-ftp:8021",
        }
        return session

    def test_request_layer_wins(self):
        session = self.make_session()
        explanation = session.explain_proxies(
            "http://example.org/",
            proxies={"http": "http://req-proxy:3128"},
            env=self.ENV,
        )
        assert explanation.final_proxy == "http://req-proxy:3128"
        assert explanation.source == "request"
        assert explanation.matched_key == "http"
        assert not explanation.direct

    def test_environment_layer_fills_missing_request_keys(self):
        # The send path fills environment proxies into missing per-request
        # keys *before* merging over the session layer, so for a scheme the
        # request does not set, the environment beats the session.
        session = self.make_session()
        explanation = session.explain_proxies(
            "https://example.org/",
            proxies={"http": "http://req-proxy:3128"},
            env=self.ENV,
        )
        assert explanation.final_proxy == "http://env-proxy:3129"
        assert explanation.source == "environment"
        assert explanation.matched_key == "https"

    def test_session_layer_used_when_request_and_env_silent(self):
        session = self.make_session()
        explanation = session.explain_proxies(
            "ftp://example.org/",
            proxies={"http": "http://req-proxy:3128"},
            env=self.ENV,
        )
        assert explanation.final_proxy == "http://session-ftp:8021"
        assert explanation.source == "session"
        assert explanation.matched_key == "ftp"

    def test_all_three_layers_reported(self):
        session = self.make_session()
        explanation = session.explain_proxies(
            "http://example.org/",
            proxies={"http": "http://req-proxy:3128"},
            env=self.ENV,
        )
        layers = {layer.name: layer for layer in explanation.layers}
        assert layers["request"].provided
        assert layers["session"].provided
        assert layers["environment"].provided
        assert layers["request"].entries == {"http": "http://req-proxy:3128"}
        assert layers["environment"].entries["http"] == "http://env-proxy:3128"


class TestSendPathConsistency:
    """The explained selection must match what the Session send path would
    actually use."""

    @pytest.mark.parametrize(
        "url",
        (
            "http://example.org/",
            "https://example.org/",
            "http://127.0.0.1:8080/",
            "http://localhost/",
        ),
    )
    @pytest.mark.parametrize(
        "request_proxies",
        (
            None,
            {},
            {"http": "http://req-proxy:3128"},
            {"http": None},
            {"http": "http://user:pass@req-proxy:3128"},
        ),
    )
    def test_matches_send_path(self, monkeypatch, url, request_proxies):
        monkeypatch.setenv("http_proxy", "http://env-proxy:3128")
        monkeypatch.setenv("https_proxy", "http://env-proxy:3129")
        monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")

        session = requests.Session()
        session.proxies = {"http": "http://session-proxy:8080"}

        explanation = session.explain_proxies(url, proxies=request_proxies)
        expected = send_path_proxy(session, url, request_proxies)

        if expected is None:
            assert explanation.direct
            assert explanation.final_proxy is None
            assert explanation.source is None
        else:
            assert not explanation.direct
            assert explanation.final_proxy == _redact_proxy_url(expected)


class TestNoProxyBypass:
    def test_env_no_proxy_loopback_is_direct(self):
        env = {
            "http_proxy": "http://env-proxy:3128",
            "no_proxy": "localhost,127.0.0.1",
        }
        session = requests.Session()
        explanation = session.explain_proxies("http://127.0.0.1:8080/", env=env)
        assert explanation.direct
        assert explanation.bypassed
        assert explanation.bypass_match == "127.0.0.1"
        assert explanation.no_proxy_source == "environment"
        assert "direct connection" in explanation.explain()

    def test_env_no_proxy_localhost_name(self):
        env = {
            "http_proxy": "http://env-proxy:3128",
            "NO_PROXY": "localhost",
        }
        explanation = explain_proxies("http://localhost:5000/", env=env)
        assert explanation.direct
        assert explanation.bypassed
        assert explanation.bypass_match == "localhost"

    def test_request_no_proxy_overrides_environment_proxy(self):
        env = {"http_proxy": "http://env-proxy:3128"}
        explanation = explain_proxies(
            "http://example.org/",
            proxies={"no_proxy": "example.org"},
            env=env,
        )
        assert explanation.direct
        assert explanation.bypassed
        assert explanation.bypass_match == "example.org"

    def test_no_bypass_means_environment_proxy_used(self):
        env = {
            "http_proxy": "http://env-proxy:3128",
            "no_proxy": "localhost",
        }
        explanation = explain_proxies("http://example.org/", env=env)
        assert not explanation.direct
        assert not explanation.bypassed
        assert explanation.source == "environment"


class TestProxyAuthRedaction:
    def test_credentials_not_exposed(self):
        secret_url = "http://alice:s3cret-p4ss@proxy.local:3128"
        explanation = explain_proxies(
            "http://example.org/",
            proxies={"http": secret_url},
            env={},
        )
        assert explanation.auth_configured
        assert explanation.final_proxy == "http://***:***@proxy.local:3128"
        rendered = explanation.explain()
        assert "s3cret-p4ss" not in rendered
        assert "alice" not in rendered
        assert "***:***@" in rendered
        assert "authentication configured" in rendered

    def test_env_credentials_not_exposed(self):
        env = {"http_proxy": "http://bob:hunter2@env-proxy:3128"}
        explanation = explain_proxies("http://example.org/", env=env)
        assert explanation.auth_configured
        rendered = explanation.explain()
        assert "hunter2" not in rendered
        assert "bob" not in rendered

    def test_no_auth_flagged_when_no_credentials(self):
        explanation = explain_proxies(
            "http://example.org/",
            proxies={"http": "http://proxy.local:3128"},
            env={},
        )
        assert not explanation.auth_configured
        assert explanation.final_proxy == "http://proxy.local:3128"


class TestMissingVsExplicitEmpty:
    def test_missing_request_layer(self):
        explanation = explain_proxies("http://example.org/", proxies=None, env={})
        request_layer = explanation.layers[0]
        assert request_layer.name == "request"
        assert not request_layer.provided
        assert "not provided" in request_layer.describe()

    def test_explicit_empty_request_layer(self):
        explanation = explain_proxies("http://example.org/", proxies={}, env={})
        request_layer = explanation.layers[0]
        assert request_layer.provided
        assert "explicitly empty" in request_layer.describe()

    def test_missing_layer_does_not_override_session(self):
        session = requests.Session()
        session.proxies = {"http": "http://session-proxy:8080"}
        explanation = session.explain_proxies(
            "http://example.org/", proxies=None, env={}
        )
        assert explanation.final_proxy == "http://session-proxy:8080"
        assert explanation.source == "session"

    def test_explicit_none_value_clears_session_key(self):
        session = requests.Session()
        session.proxies = {"http": "http://session-proxy:8080"}
        explanation = session.explain_proxies(
            "http://example.org/", proxies={"http": None}, env={}
        )
        assert explanation.direct
        request_layer = explanation.layers[0]
        assert request_layer.cleared == ("http",)
        assert "clears lower layers" in request_layer.describe()


class TestPurity:
    """The explainer must not send requests nor modify the process
    environment, not even temporarily."""

    def test_os_environ_not_modified(self):
        before = dict(os.environ)
        env = {
            "http_proxy": "http://env-proxy:3128",
            "no_proxy": "localhost",
        }
        explain_proxies("http://example.org/", env=env)
        explain_proxies("http://localhost/", env=env)
        session = requests.Session()
        session.explain_proxies("http://example.org/", env=env)
        assert dict(os.environ) == before

    def test_mock_env_not_modified(self):
        env = {
            "http_proxy": "http://env-proxy:3128",
            "no_proxy": "localhost",
        }
        snapshot = dict(env)
        explain_proxies("http://example.org/", env=env)
        assert env == snapshot

    def test_no_request_sent(self):
        # An unroutable URL would raise if anything were actually sent;
        # explaining it must succeed purely.
        explanation = explain_proxies(
            "http://192.0.2.1:1/", proxies={"http": "http://proxy:3128"}, env={}
        )
        assert isinstance(explanation, ProxyExplanation)
        assert explanation.final_proxy == "http://proxy:3128"

    def test_mock_env_ignores_real_os_environ(self, monkeypatch):
        monkeypatch.setenv("http_proxy", "http://real-env-proxy:3128")
        explanation = explain_proxies("http://example.org/", env={})
        assert explanation.direct
        env_layer = explanation.layers[2]
        assert not env_layer.provided


class TestApiSurface:
    def test_module_level_explain_proxies_with_session(self):
        session = requests.Session()
        session.proxies = {"http": "http://session-proxy:8080"}
        explanation = requests.explain_proxies(
            "http://example.org/", session=session, env={}
        )
        assert explanation.source == "session"

    def test_module_level_without_session(self):
        explanation = requests.explain_proxies(
            "http://example.org/",
            proxies={"http": "http://req-proxy:3128"},
            env={},
        )
        assert explanation.source == "request"

    def test_session_method_uses_trust_env(self):
        session = requests.Session()
        session.trust_env = False
        env = {"http_proxy": "http://env-proxy:3128"}
        explanation = session.explain_proxies("http://example.org/", env=env)
        assert explanation.direct
        assert not explanation.trust_env
        env_layer = explanation.layers[2]
        assert not env_layer.provided

    def test_str_renders_explanation(self):
        explanation = explain_proxies("http://example.org/", env={})
        assert str(explanation) == explanation.explain()
        assert "no request was sent" in str(explanation)

    def test_scheme_host_key_precedence(self):
        explanation = explain_proxies(
            "http://example.org/",
            proxies={
                "http": "http://generic:3128",
                "http://example.org": "http://specific:3128",
            },
            env={},
        )
        assert explanation.final_proxy == "http://specific:3128"
        assert explanation.matched_key == "http://example.org"


class TestCommandLine:
    def test_main_prints_explanation(self, capsys):
        from requests.explain import main

        assert main(["http://example.org/"]) == 0
        out = capsys.readouterr().out
        assert "Proxy selection for 'http://example.org/'" in out
        assert "no request was sent" in out

    def test_python_m_requests_explain(self):
        import subprocess
        import sys

        env = dict(os.environ)
        env["http_proxy"] = "http://cli-proxy:3128"
        env.pop("no_proxy", None)
        env.pop("NO_PROXY", None)
        result = subprocess.run(
            [sys.executable, "-m", "requests.explain", "http://example.org/"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0
        assert "http://cli-proxy:3128" in result.stdout
        assert "from environment" in result.stdout
