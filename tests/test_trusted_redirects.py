"""Tests for TrustedRedirectSession: redirect-scoped cookie trust policy.

Two loopback HTTP servers impersonate ``a.test`` and ``b.test`` by
monkeypatching ``socket.getaddrinfo`` so both names resolve to 127.0.0.1.
No external network access is required.
"""

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import requests
from requests.sessions import TrustedRedirectSession


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        srv = self.server
        srv.received.append(
            {
                "path": self.path,
                "cookie": self.headers.get("Cookie"),
                "authorization": self.headers.get("Authorization"),
            }
        )
        status, headers = srv.routes.get(self.path, (404, {}))
        body = b"ok"
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _Site:
    """A loopback HTTP server reachable under a fake hostname."""

    def __init__(self, name):
        self.name = name
        self.routes = {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.server.routes = self.routes
        self.server.received = []
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def received(self):
        return self.server.received

    def url(self, path):
        return f"http://{self.name}:{self.port}{path}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


@pytest.fixture
def sites(monkeypatch):
    real_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host in ("a.test", "b.test"):
            host = "127.0.0.1"
        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    a = _Site("a.test")
    b = _Site("b.test")
    try:
        yield a, b
    finally:
        a.close()
        b.close()


def _parse_cookie_header(header):
    assert header is not None
    return dict(part.split("=", 1) for part in header.split("; "))


def test_untrusted_hop_drops_cookie_even_when_jar_has_cookies(sites):
    a, b = sites
    a.routes["/start"] = (302, {"Location": b.url("/landing")})
    b.routes["/landing"] = (200, {})

    session = TrustedRedirectSession()
    # The jar already holds a cookie for b.test before the redirect happens.
    session.cookies.set("b_session", "topsecret", domain="b.test", path="/")

    resp = session.get(a.url("/start"), cookies={"caller": "first-hop"})

    assert resp.status_code == 200
    # First hop: caller-specified cookie is sent to the original target.
    assert _parse_cookie_header(a.received[0]["cookie"]) == {"caller": "first-hop"}
    # Untrusted hop: no Cookie header at all, despite the jar holding
    # cookies for b.test -- the jar is not allowed to re-attach them.
    assert b.received[0]["cookie"] is None
    # Audit records host and decision only.
    assert session.cookie_redirect_audit == [{"host": "b.test", "decision": "deny"}]
    # The jar still retains the b.test cookie for later same-origin use.
    assert session.cookies.get("b_session", domain="b.test") == "topsecret"


def test_trusted_hop_sends_cookies_by_path_rules(sites):
    a, b = sites
    a.routes["/start"] = (302, {"Location": b.url("/sub/landing")})
    b.routes["/sub/landing"] = (200, {})
    b.routes["/other"] = (200, {})

    session = TrustedRedirectSession(trusted_hosts=["B.TEST"])  # case-insensitive
    session.cookies.set("root", "r1", domain="b.test", path="/")
    session.cookies.set("sub", "s1", domain="b.test", path="/sub")

    session.get(a.url("/start"))

    # Trusted hop: cookies matching the domain *and* path are attached.
    assert _parse_cookie_header(b.received[0]["cookie"]) == {"root": "r1", "sub": "s1"}
    assert session.cookie_redirect_audit == [{"host": "b.test", "decision": "allow"}]

    # The path-scoped cookie is not sent outside its path.
    session.get(b.url("/other"))
    assert _parse_cookie_header(b.received[1]["cookie"]) == {"root": "r1"}


def test_return_to_origin_judged_by_original_trust_set(sites):
    a, b = sites
    a.routes["/start"] = (302, {"Location": b.url("/hop")})
    b.routes["/hop"] = (302, {"Location": a.url("/back")})
    a.routes["/back"] = (200, {})

    session = TrustedRedirectSession()
    session.cookies.set("a_session", "aaa", domain="a.test", path="/")
    session.cookies.set("b_session", "bbb", domain="b.test", path="/")

    session.get(a.url("/start"))

    # b.test is untrusted: no cookies cross over.
    assert b.received[0]["cookie"] is None
    # Returning to the original host is allowed by the original trust set.
    assert _parse_cookie_header(a.received[1]["cookie"]) == {"a_session": "aaa"}
    assert session.cookie_redirect_audit == [
        {"host": "b.test", "decision": "deny"},
        {"host": "a.test", "decision": "allow"},
    ]


def test_multi_hop_same_hostname_different_port_stays_trusted(sites):
    a, b = sites
    a2 = _Site("a.test")  # same hostname, different port
    try:
        a.routes["/start"] = (302, {"Location": a2.url("/relay")})
        a2.routes["/relay"] = (302, {"Location": b.url("/landing")})
        b.routes["/landing"] = (200, {})

        session = TrustedRedirectSession()
        session.cookies.set("a_session", "aaa", domain="a.test", path="/")

        session.get(a.url("/start"))

        # Same hostname on a different port remains trusted (cookies ignore
        # ports), so the a.test cookie is carried across the port change.
        assert _parse_cookie_header(a2.received[0]["cookie"]) == {"a_session": "aaa"}
        # The final cross-host hop is still denied.
        assert b.received[0]["cookie"] is None
        assert session.cookie_redirect_audit == [
            {"host": "a.test", "decision": "allow"},
            {"host": "b.test", "decision": "deny"},
        ]
    finally:
        a2.close()


def test_authorization_stripping_still_applies_on_cross_host_redirect(sites):
    a, b = sites
    a.routes["/start"] = (302, {"Location": b.url("/landing")})
    b.routes["/landing"] = (200, {})

    session = TrustedRedirectSession()
    session.cookies.set("b_session", "bbb", domain="b.test", path="/")

    session.get(a.url("/start"), headers={"Authorization": "Bearer token"})

    # Existing behavior: Authorization is stripped on cross-host redirects.
    assert b.received[0]["authorization"] is None
    # New policy: cookies are stripped as well.
    assert b.received[0]["cookie"] is None


def test_same_host_redirect_keeps_authorization_and_cookies(sites):
    a, _ = sites
    a.routes["/old"] = (302, {"Location": a.url("/new")})
    a.routes["/new"] = (200, {})

    session = TrustedRedirectSession()
    session.cookies.set("a_session", "aaa", domain="a.test", path="/")

    session.get(a.url("/old"), headers={"Authorization": "Bearer token"})

    assert a.received[1]["authorization"] == "Bearer token"
    assert _parse_cookie_header(a.received[1]["cookie"]) == {"a_session": "aaa"}
    assert session.cookie_redirect_audit == [{"host": "a.test", "decision": "allow"}]


def test_default_session_behavior_is_unchanged(sites):
    """Plain Session still lets the jar attach cookies on cross-host redirects."""
    a, b = sites
    a.routes["/start"] = (302, {"Location": b.url("/landing")})
    b.routes["/landing"] = (200, {})

    session = requests.Session()
    session.cookies.set("b_session", "bbb", domain="b.test", path="/")

    session.get(a.url("/start"))

    assert _parse_cookie_header(b.received[0]["cookie"]) == {"b_session": "bbb"}
