"""
tests.test_trusted_cookies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Targeted tests for :class:`requests.sessions.TrustedCookieSession`, an
optional :class:`~requests.Session` subclass that scopes which hosts may
receive cookies across redirects.

Policy under test:

* The first hop always carries the caller-supplied cookies.
* On each redirect hop the target host is checked against the trusted-host
  set (seeded from the original target and extended via ``trusted_hosts``).
* Untrusted hops get the ``Cookie`` header stripped and the cookie jar must
  not replenish it.
* Trusted hops send cookies per the standard domain and path rules.
* The existing ``Authorization`` stripping logic is unaffected.
* :attr:`TrustedCookieSession.cookie_audit` records only ``host`` and
  ``decision``; cookie values are never recorded.

Distinct loopback addresses stand in for distinct hosts (a=127.0.0.1,
b=127.0.0.2, c=127.0.0.3). No Docker or external services are used.
"""

from __future__ import annotations

import socketserver
import threading
from http.server import BaseHTTPRequestHandler

import pytest

from requests.sessions import TrustedCookieSession

# ---------------------------------------------------------------------------
# Loopback test fixtures
# ---------------------------------------------------------------------------


class _RecordingHandler(BaseHTTPRequestHandler):
    """A request handler that records received requests and responds 200."""

    protocol_version = "HTTP/1.1"
    received: list[dict] = []

    def log_message(self, format: str, *args) -> None:  # silence stderr
        return

    def _record(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        record = {
            "host": self.headers.get("Host"),
            "method": self.command,
            "path": self.path,
            "cookie": self.headers.get("Cookie"),
            "authorization": self.headers.get("Authorization"),
            "body": body,
        }
        self.received.append(record)
        return record

    def do_GET(self) -> None:
        self._record()
        self._send_response()

    def do_POST(self) -> None:
        self._record()
        self._send_response()

    def _send_response(self) -> None:
        self.send_response_only(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


class _RedirectHandler(_RecordingHandler):
    """Redirects requests to ``redirect_to`` (mutable class attr).

    By default every request is redirected; set ``redirect_paths`` to a
    tuple of paths to redirect only those (others get a normal 200).
    """

    redirect_to: str = ""
    status: int = 302
    redirect_paths: tuple | None = None

    def _send_response(self) -> None:
        if self.redirect_paths is not None and self.path not in self.redirect_paths:
            super()._send_response()
            return
        self.send_response_only(self.status)
        self.send_header("Content-Length", "0")
        self.send_header("Location", self.redirect_to)
        self.end_headers()


class _SameHostRedirectHandler(_RecordingHandler):
    """Redirects ``/`` to ``/other`` on the same server, then responds 200."""

    def _send_response(self) -> None:
        if self.path == "/":
            self.send_response_only(302)
            self.send_header("Content-Length", "0")
            self.send_header("Location", "/other")
            self.end_headers()
        else:
            super()._send_response()


def _make_server(handler_class: type, host: str = "127.0.0.1", **attrs):
    """Start a loopback server on ``host`` with a fresh ``received`` list.

    Returns ``(server, handler_class, received)``.
    """
    received: list[dict] = []
    handler = type(
        handler_class.__name__,
        (handler_class,),
        {"received": received, **attrs},
    )
    # Use ThreadingTCPServer directly (not ThreadingHTTPServer) to avoid
    # HTTPServer.server_bind's getfqdn() reverse-DNS lookup, which times
    # out after ~5s for loopback addresses other than 127.0.0.1.
    server = socketserver.ThreadingTCPServer((host, 0), handler)
    server.allow_reuse_address = True
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, handler, received


@pytest.fixture
def loopback_servers():
    """Provide a factory for loopback servers; shut them all down after."""
    servers: list[socketserver.ThreadingTCPServer] = []

    def start(handler_class, host="127.0.0.1", **attrs):
        server, handler, received = _make_server(handler_class, host=host, **attrs)
        servers.append(server)
        return server, handler, received

    yield start

    for server in servers:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_untrusted_host_receives_no_cookies(loopback_servers):
    """b is untrusted: the request to b must carry no Cookie header, even
    though the jar already holds a cookie for b."""
    b_server, _, b_received = loopback_servers(_RecordingHandler, host="127.0.0.2")
    b_port = b_server.server_address[1]

    a_server, _, a_received = loopback_servers(
        _RedirectHandler, redirect_to=f"http://127.0.0.2:{b_port}/"
    )
    a_port = a_server.server_address[1]

    session = TrustedCookieSession(trusted_hosts=[f"127.0.0.1:{a_port}"])
    session.cookies.set("a_cookie", "a-val", domain="127.0.0.1", path="/")
    session.cookies.set("b_cookie", "b-val", domain="127.0.0.2", path="/")

    r = session.get(f"http://127.0.0.1:{a_port}/")

    assert r.status_code == 200
    assert a_received[0]["cookie"] == "a_cookie=a-val"
    assert b_received[0]["cookie"] is None
    decisions = [(e["host"], e["decision"]) for e in session.cookie_audit]
    assert (f"127.0.0.2:{b_port}", "deny") in decisions
    assert all(set(e.keys()) == {"host", "decision"} for e in session.cookie_audit)


def test_trusted_host_sends_cookies_per_path(loopback_servers):
    """Once b is trusted, cookies are sent per their path rules."""
    b_server, _, b_received = loopback_servers(_RecordingHandler, host="127.0.0.2")
    b_port = b_server.server_address[1]

    a_server, _, _ = loopback_servers(
        _RedirectHandler, redirect_to=f"http://127.0.0.2:{b_port}/deep/path"
    )
    a_port = a_server.server_address[1]

    session = TrustedCookieSession(
        trusted_hosts=[f"127.0.0.1:{a_port}", f"127.0.0.2:{b_port}"]
    )
    session.cookies.set("b_root", "root-val", domain="127.0.0.2", path="/")
    session.cookies.set("b_path", "path-val", domain="127.0.0.2", path="/deep")

    r = session.get(f"http://127.0.0.1:{a_port}/")

    assert r.status_code == 200
    assert b_received[0]["cookie"] is not None
    assert "b_root=root-val" in b_received[0]["cookie"]
    assert "b_path=path-val" in b_received[0]["cookie"]
    assert {"host": f"127.0.0.2:{b_port}", "decision": "allow"} in session.cookie_audit


def test_trusted_host_path_filtering(loopback_servers):
    """A cookie scoped to /deep is not sent to b's root path."""
    b_server, _, b_received = loopback_servers(_RecordingHandler, host="127.0.0.2")
    b_port = b_server.server_address[1]

    a_server, _, _ = loopback_servers(
        _RedirectHandler, redirect_to=f"http://127.0.0.2:{b_port}/"
    )
    a_port = a_server.server_address[1]

    session = TrustedCookieSession(
        trusted_hosts=[f"127.0.0.1:{a_port}", f"127.0.0.2:{b_port}"]
    )
    session.cookies.set("b_path", "path-val", domain="127.0.0.2", path="/deep")

    r = session.get(f"http://127.0.0.1:{a_port}/")

    assert r.status_code == 200
    assert b_received[0]["cookie"] is None


def test_return_to_original_host_sends_own_cookies(loopback_servers):
    """After a -> b -> a, the return to a is judged by the original trust
    set and a's cookies are sent again (b's are not leaked to a)."""
    a_server, a_handler, a_received = loopback_servers(_RedirectHandler, redirect_to="")
    b_server, b_handler, b_received = loopback_servers(
        _RedirectHandler, host="127.0.0.2", redirect_to=""
    )
    a_port = a_server.server_address[1]
    b_port = b_server.server_address[1]
    a_handler.redirect_to = f"http://127.0.0.2:{b_port}/"
    a_handler.redirect_paths = ("/",)
    b_handler.redirect_to = f"http://127.0.0.1:{a_port}/done"
    b_handler.redirect_paths = ("/",)

    session = TrustedCookieSession(trusted_hosts=[f"127.0.0.1:{a_port}"])
    session.cookies.set("a_cookie", "a-val", domain="127.0.0.1", path="/")
    session.cookies.set("b_cookie", "b-val", domain="127.0.0.2", path="/")

    r = session.get(f"http://127.0.0.1:{a_port}/")

    assert r.status_code == 200
    assert a_received[0]["path"] == "/"
    assert a_received[0]["cookie"] == "a_cookie=a-val"
    assert b_received[0]["cookie"] is None
    assert a_received[1]["path"] == "/done"
    assert a_received[1]["cookie"] == "a_cookie=a-val"
    decisions = [(e["host"], e["decision"]) for e in session.cookie_audit]
    assert decisions == [
        (f"127.0.0.1:{a_port}", "allow"),
        (f"127.0.0.2:{b_port}", "deny"),
        (f"127.0.0.1:{a_port}", "allow"),
    ]


def test_multi_hop_audit(loopback_servers):
    """A multi-hop chain records an allow/deny decision for every hop."""
    c_server, _, c_received = loopback_servers(_RecordingHandler, host="127.0.0.3")
    c_port = c_server.server_address[1]

    b_server, _, b_received = loopback_servers(
        _RedirectHandler, host="127.0.0.2", redirect_to=f"http://127.0.0.3:{c_port}/"
    )
    b_port = b_server.server_address[1]

    a_server, _, a_received = loopback_servers(
        _RedirectHandler, redirect_to=f"http://127.0.0.2:{b_port}/"
    )
    a_port = a_server.server_address[1]

    session = TrustedCookieSession(
        trusted_hosts=[f"127.0.0.1:{a_port}", f"127.0.0.3:{c_port}"]
    )
    session.cookies.set("a_cookie", "a-val", domain="127.0.0.1", path="/")
    session.cookies.set("b_cookie", "b-val", domain="127.0.0.2", path="/")
    session.cookies.set("c_cookie", "c-val", domain="127.0.0.3", path="/")

    r = session.get(f"http://127.0.0.1:{a_port}/")

    assert r.status_code == 200
    assert a_received[0]["cookie"] == "a_cookie=a-val"
    assert b_received[0]["cookie"] is None
    assert c_received[0]["cookie"] == "c_cookie=c-val"
    decisions = [(e["host"], e["decision"]) for e in session.cookie_audit]
    assert decisions == [
        (f"127.0.0.1:{a_port}", "allow"),
        (f"127.0.0.2:{b_port}", "deny"),
        (f"127.0.0.3:{c_port}", "allow"),
    ]


def test_port_change_same_host_trusted(loopback_servers):
    """A redirect to a different port on the same host stays trusted."""
    b_server, _, b_received = loopback_servers(_RecordingHandler)
    b_port = b_server.server_address[1]

    a_server, _, a_received = loopback_servers(
        _RedirectHandler, redirect_to=f"http://127.0.0.1:{b_port}/"
    )
    a_port = a_server.server_address[1]

    session = TrustedCookieSession(trusted_hosts=["127.0.0.1"])
    session.cookies.set("host_cookie", "host-val", domain="127.0.0.1", path="/")

    r = session.get(f"http://127.0.0.1:{a_port}/")

    assert r.status_code == 200
    assert a_received[0]["cookie"] == "host_cookie=host-val"
    assert b_received[0]["cookie"] == "host_cookie=host-val"
    assert {"host": f"127.0.0.1:{b_port}", "decision": "allow"} in session.cookie_audit


def test_port_change_different_host_untrusted(loopback_servers):
    """A redirect to a different host (even on the same port) is untrusted."""
    b_server, _, b_received = loopback_servers(_RecordingHandler, host="127.0.0.2")
    b_port = b_server.server_address[1]

    a_server, _, a_received = loopback_servers(
        _RedirectHandler, redirect_to=f"http://127.0.0.2:{b_port}/"
    )
    a_port = a_server.server_address[1]

    session = TrustedCookieSession(trusted_hosts=["127.0.0.1"])
    session.cookies.set("b_cookie", "b-val", domain="127.0.0.2", path="/")

    r = session.get(f"http://127.0.0.1:{a_port}/")

    assert r.status_code == 200
    assert a_received[0]["cookie"] is None  # b's cookie does not match a's host
    assert b_received[0]["cookie"] is None
    assert {"host": f"127.0.0.2:{b_port}", "decision": "deny"} in session.cookie_audit


def test_authorization_stripped_on_cross_host_redirect(loopback_servers):
    """The existing Authorization stripping is preserved: a cross-host
    redirect removes Authorization, and the untrusted hop also gets no
    cookies."""
    b_server, _, b_received = loopback_servers(_RecordingHandler, host="127.0.0.2")
    b_port = b_server.server_address[1]

    a_server, _, a_received = loopback_servers(
        _RedirectHandler, redirect_to=f"http://127.0.0.2:{b_port}/"
    )
    a_port = a_server.server_address[1]

    session = TrustedCookieSession(trusted_hosts=[f"127.0.0.1:{a_port}"])
    session.cookies.set("a_cookie", "a-val", domain="127.0.0.1", path="/")

    r = session.get(
        f"http://127.0.0.1:{a_port}/",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert r.status_code == 200
    assert a_received[0]["authorization"] == "Bearer secret-token"
    assert b_received[0]["authorization"] is None
    assert b_received[0]["cookie"] is None


def test_authorization_preserved_same_host_redirect(loopback_servers):
    """A same-host redirect keeps Authorization (existing behavior) and
    cookies are sent per the trust decision."""
    a_server, _, a_received = loopback_servers(_SameHostRedirectHandler)
    a_port = a_server.server_address[1]

    session = TrustedCookieSession(trusted_hosts=[f"127.0.0.1:{a_port}"])
    session.cookies.set("host_cookie", "host-val", domain="127.0.0.1", path="/")

    r = session.get(
        f"http://127.0.0.1:{a_port}/",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert r.status_code == 200
    # Both hops hit the same host:port, so Authorization is preserved.
    assert a_received[0]["authorization"] == "Bearer secret-token"
    assert a_received[1]["authorization"] == "Bearer secret-token"
    # The trusted same-host redirect carries the cookie.
    assert a_received[1]["cookie"] == "host_cookie=host-val"


def test_first_hop_always_sends_caller_cookies(loopback_servers):
    """Even with no explicit trusted hosts, the first hop sends the
    caller-supplied cookies (the original target is always trusted)."""
    a_server, _, a_received = loopback_servers(_RecordingHandler)
    a_port = a_server.server_address[1]

    session = TrustedCookieSession()
    session.cookies.set("a_cookie", "a-val", domain="127.0.0.1", path="/")

    r = session.get(f"http://127.0.0.1:{a_port}/")

    assert r.status_code == 200
    assert a_received[0]["cookie"] == "a_cookie=a-val"
    assert session.cookie_audit == [
        {"host": f"127.0.0.1:{a_port}", "decision": "allow"}
    ]


def test_audit_contains_no_cookie_values(loopback_servers):
    """The audit records only host and decision, never cookie values."""
    b_server, _, _ = loopback_servers(_RecordingHandler, host="127.0.0.2")
    b_port = b_server.server_address[1]

    a_server, _, _ = loopback_servers(
        _RedirectHandler, redirect_to=f"http://127.0.0.2:{b_port}/"
    )
    a_port = a_server.server_address[1]

    session = TrustedCookieSession(trusted_hosts=[f"127.0.0.1:{a_port}"])
    session.cookies.set("secret", "super-secret-value", domain="127.0.0.1", path="/")

    session.get(f"http://127.0.0.1:{a_port}/")

    for entry in session.cookie_audit:
        assert set(entry.keys()) == {"host", "decision"}
        assert "super-secret-value" not in str(entry)


def test_default_session_unchanged(loopback_servers):
    """The default Session still sends cookies to redirect targets
    (no filtering) -- the new behavior is opt-in only."""
    b_server, _, b_received = loopback_servers(_RecordingHandler, host="127.0.0.2")
    b_port = b_server.server_address[1]

    a_server, _, a_received = loopback_servers(
        _RedirectHandler, redirect_to=f"http://127.0.0.2:{b_port}/"
    )
    a_port = a_server.server_address[1]

    import requests

    session = requests.Session()
    # Cookie scoped to b's domain: the default session sends it on redirect
    # (no trust filtering), proving the new behavior is opt-in only.
    session.cookies.set("b_cookie", "b-val", domain="127.0.0.2", path="/")

    r = session.get(f"http://127.0.0.1:{a_port}/")

    assert r.status_code == 200
    assert b_received[0]["cookie"] == "b_cookie=b-val"
