"""Deterministic tests for the HTTPAdapter retry policy.

All tests run against in-process socket servers (tests.testserver) or
monkeypatched pool managers, so they never touch the network and produce
the same results on every run.
"""

import io
import threading
import time
from email.utils import formatdate

import pytest
from urllib3.exceptions import NewConnectionError
from urllib3.util.retry import Retry

import requests
from requests.adapters import (
    DEFAULT_BACKOFF_MAX,
    HTTPAdapter,
    _parse_retry_after,
    _RetryState,
)
from requests.exceptions import ConnectionError, ReadTimeout, RetryError

from .testserver.server import Server, consume_socket_content


def make_response(status, reason=None, body=b"", extra_headers=""):
    reason = reason or {
        200: "OK",
        302: "Found",
        400: "Bad Request",
        503: "Service Unavailable",
    }.get(status, "X")
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        f"{extra_headers}"
        "\r\n"
    )
    return head.encode("utf-8") + body


def sequenced_server(responses, requests_to_handle=None):
    """A Server that answers each connection with the next response in the
    sequence (repeating the last one) and records what it received."""
    seen = []
    lock = threading.Lock()

    def handler(sock):
        request = consume_socket_content(sock, timeout=0.1)
        with lock:
            index = len(seen)
            seen.append(request)
        sock.sendall(responses[min(index, len(responses) - 1)])
        return request

    server = Server(
        handler,
        host="127.0.0.1",
        requests_to_handle=requests_to_handle or len(responses),
    )
    return server, seen


def retry_session(**retry_kwargs):
    session = requests.Session()
    session.trust_env = False
    adapter = HTTPAdapter(max_retries=Retry(**retry_kwargs))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session, adapter


class TestParseRetryAfter:
    def test_seconds(self):
        assert _parse_retry_after("5", DEFAULT_BACKOFF_MAX) == 5.0

    def test_http_date(self):
        value = formatdate(time.time() + 2, usegmt=True)
        delay = _parse_retry_after(value, DEFAULT_BACKOFF_MAX)
        assert 0.0 <= delay <= 2.0

    def test_seconds_capped_by_backoff_max(self):
        assert _parse_retry_after("9999", 0.05) == 0.05

    def test_http_date_capped_by_backoff_max(self):
        value = formatdate(time.time() + 9999, usegmt=True)
        assert _parse_retry_after(value, 0.05) == 0.05

    def test_past_date_clamps_to_zero(self):
        value = formatdate(time.time() - 60, usegmt=True)
        assert _parse_retry_after(value, DEFAULT_BACKOFF_MAX) == 0.0

    def test_unparseable_returns_none(self):
        assert _parse_retry_after("not-a-date", DEFAULT_BACKOFF_MAX) is None


class TestRetryStateBackoff:
    def test_exponential_backoff_capped(self):
        request = requests.Request("GET", "http://example.com/").prepare()
        state = _RetryState(Retry(total=5, backoff_factor=10), request)
        delays = []
        for _ in range(5):
            state.increment_status()
            delays.append(state.retry_delay())
        assert delays == [0.0, 20.0, 40.0, 80.0, DEFAULT_BACKOFF_MAX]

    def test_default_retryable_status_codes(self):
        request = requests.Request("GET", "http://example.com/").prepare()
        state = _RetryState(Retry(total=1), request)
        for status in (500, 502, 503, 504):
            assert state.is_retryable_status(status)
        assert not state.is_retryable_status(400)

    def test_no_status_retries_when_retries_disabled(self):
        request = requests.Request("GET", "http://example.com/").prepare()
        state = _RetryState(Retry(0, read=False), request)
        assert not state.is_retryable_status(503)


class TestConnectFailures:
    def test_retries_then_raises_with_exception_chain(self, monkeypatch):
        session, adapter = retry_session(total=2, backoff_factor=0)
        calls = []

        def fail(*args, **kwargs):
            calls.append(1)
            raise NewConnectionError(adapter.poolmanager, "connection refused")

        monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", fail)
        with pytest.raises(ConnectionError) as exc_info:
            session.get("http://localhost/")
        assert len(calls) == 3  # 1 initial attempt + 2 retries
        assert isinstance(exc_info.value.__cause__, NewConnectionError)

    def test_non_idempotent_method_never_retried(self, monkeypatch):
        session, adapter = retry_session(total=3, backoff_factor=0)
        calls = []

        def fail(*args, **kwargs):
            calls.append(1)
            raise NewConnectionError(adapter.poolmanager, "connection refused")

        monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", fail)
        with pytest.raises(ConnectionError):
            session.post("http://localhost/", data=b"payload")
        assert len(calls) == 1

    def test_connection_refused_raises_connection_error(self):
        # Bind and immediately close a socket to get a reliably refused port.
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        session, _ = retry_session(total=1, backoff_factor=0)
        with pytest.raises(ConnectionError) as exc_info:
            session.get(f"http://127.0.0.1:{port}/", timeout=1)
        assert exc_info.value.__cause__ is not None


class TestStatusRetries:
    def test_retryable_5xx_retried_until_success(self):
        responses = [
            make_response(503, extra_headers="Retry-After: 0\r\n"),
            make_response(503, extra_headers="Retry-After: 0\r\n"),
            make_response(200, body=b"recovered"),
        ]
        server, seen = sequenced_server(responses)
        with server as (host, port):
            session, _ = retry_session(total=3, backoff_factor=0)
            response = session.get(f"http://{host}:{port}/")
        assert response.status_code == 200
        assert response.content == b"recovered"
        assert len(seen) == 3

    def test_exhaustion_raises_retry_error_with_last_response(self):
        responses = [make_response(503, body=b"still down")]
        server, seen = sequenced_server(responses, requests_to_handle=3)
        with server as (host, port):
            session, _ = retry_session(
                total=2, backoff_factor=0, status_forcelist=[503]
            )
            with pytest.raises(RetryError) as exc_info:
                session.get(f"http://{host}:{port}/")
        assert len(seen) == 3
        assert exc_info.value.response is not None
        assert exc_info.value.response.status_code == 503

    def test_non_retryable_status_not_retried(self):
        responses = [make_response(400)]
        server, seen = sequenced_server(responses)
        with server as (host, port):
            session, _ = retry_session(total=3, backoff_factor=0)
            response = session.get(f"http://{host}:{port}/")
        assert response.status_code == 400
        assert len(seen) == 1

    def test_retry_after_header_caps_sleep(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("requests.adapters.time.sleep", sleeps.append)
        responses = [
            make_response(503, extra_headers="Retry-After: 30\r\n"),
            make_response(
                503,
                extra_headers="Retry-After: "
                + formatdate(time.time() + 30, usegmt=True)
                + "\r\n",
            ),
            make_response(200, body=b"ok"),
        ]
        server, seen = sequenced_server(responses)
        with server as (host, port):
            session, _ = retry_session(total=2, backoff_factor=0, backoff_max=0.05)
            response = session.get(f"http://{host}:{port}/")
        assert response.status_code == 200
        assert len(seen) == 3
        # Both the delta-seconds and the HTTP-date form are clamped to
        # backoff_max instead of sleeping the requested 30 seconds.
        assert sleeps == [0.05, 0.05]


class TestNonRetryableRequests:
    def test_post_with_bytes_body_not_retried(self):
        responses = [make_response(503)]
        server, seen = sequenced_server(responses)
        with server as (host, port):
            session, _ = retry_session(
                total=3, backoff_factor=0, status_forcelist=[503]
            )
            response = session.post(f"http://{host}:{port}/", data=b"payload")
        assert response.status_code == 503
        assert len(seen) == 1

    def test_streaming_body_not_retried(self):
        responses = [make_response(503)]
        server, seen = sequenced_server(responses)
        with server as (host, port):
            session, _ = retry_session(
                total=3, backoff_factor=0, status_forcelist=[503]
            )
            response = session.post(
                f"http://{host}:{port}/", data=(chunk for chunk in [b"a", b"b"])
            )
        assert response.status_code == 503
        assert len(seen) == 1

    def test_rewindable_body_resent_on_retry(self):
        responses = [make_response(503), make_response(200, body=b"ok")]
        server, seen = sequenced_server(responses)
        with server as (host, port):
            session, _ = retry_session(
                total=1, backoff_factor=0, status_forcelist=[503]
            )
            response = session.put(
                f"http://{host}:{port}/", data=io.BytesIO(b"rewind-me")
            )
        assert response.status_code == 200
        assert len(seen) == 2
        assert b"rewind-me" in seen[1]


class TestTimeoutsAndRedirects:
    def test_read_timeout_not_retried(self):
        close_event = threading.Event()
        seen = []

        def silent_handler(sock):
            seen.append(consume_socket_content(sock, timeout=0.1))
            close_event.wait(5)

        server = Server(silent_handler, host="127.0.0.1", requests_to_handle=1)
        try:
            with server as (host, port):
                session, _ = retry_session(total=3, backoff_factor=0)
                with pytest.raises(ReadTimeout):
                    session.get(f"http://{host}:{port}/", timeout=(1, 0.3))
                close_event.set()
        finally:
            close_event.set()
        assert len(seen) == 1

    def test_redirects_work_with_retries_enabled(self):
        responses = [
            make_response(302, extra_headers="Location: /target\r\n"),
            make_response(200, body=b"redirected"),
        ]
        server, seen = sequenced_server(responses)
        with server as (host, port):
            session, _ = retry_session(total=2, backoff_factor=0)
            response = session.get(f"http://{host}:{port}/")
        assert response.status_code == 200
        assert response.content == b"redirected"
        assert response.history[0].status_code == 302
        assert len(seen) == 2


class TestConcurrency:
    def test_concurrent_requests_do_not_share_retry_state(self):
        thread_count = 4
        lock = threading.Lock()
        counts = {"total": 0}

        def handler(sock):
            consume_socket_content(sock, timeout=0.1)
            with lock:
                counts["total"] += 1
                number = counts["total"]
            if number == 1:
                sock.sendall(make_response(503, extra_headers="Retry-After: 0\r\n"))
            else:
                sock.sendall(make_response(200, body=b"ok"))

        server = Server(handler, host="127.0.0.1", requests_to_handle=thread_count + 1)
        with server as (host, port):
            session, adapter = retry_session(total=3, backoff_factor=0)
            barrier = threading.Barrier(thread_count)
            results = []

            def worker():
                barrier.wait(timeout=5)
                results.append(session.get(f"http://{host}:{port}/").status_code)

            threads = [threading.Thread(target=worker) for _ in range(thread_count)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

        # Exactly one request hit the single 503 and was retried; every
        # thread observed success and the shared adapter configuration was
        # never mutated.
        assert sorted(results) == [200] * thread_count
        assert counts["total"] == thread_count + 1
        assert adapter.max_retries.total == 3
