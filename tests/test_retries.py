"""Deterministic tests for the HTTPAdapter retry policy."""

import email.utils
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import urllib3.connection
import urllib3.connectionpool
from urllib3.exceptions import InvalidHeader as Urllib3InvalidHeader
from urllib3.exceptions import MaxRetryError, NewConnectionError
from urllib3.util.retry import Retry

import requests
from requests.adapters import HTTPAdapter, _is_rewindable_body
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout, RetryError, TooManyRedirects
from tests.testserver.server import Server, consume_socket_content


def _http_response(status, reason, headers=(), body=b""):
    lines = [f"HTTP/1.1 {status} {reason}"]
    lines += [f"{name}: {value}" for name, value in headers]
    lines.append(f"Content-Length: {len(body)}")
    # Each attempt uses a fresh connection; the server closes its socket
    # after every response.
    lines.append("Connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode() + body


def _status_server(status, reason, requests_to_handle, headers=()):
    response = _http_response(status, reason, headers=headers)

    def handler(sock):
        request_content = consume_socket_content(sock, timeout=0.1)
        sock.send(response)
        return request_content

    return Server(handler, host="127.0.0.1", requests_to_handle=requests_to_handle)


def _sequencing_server(responses, requests_to_handle):
    responses_iter = iter(responses)

    def handler(sock):
        request_content = consume_socket_content(sock, timeout=0.1)
        sock.send(next(responses_iter))
        return request_content

    return Server(handler, host="127.0.0.1", requests_to_handle=requests_to_handle)


def _make_session(max_retries):
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=max_retries)
    session.mount("http://", adapter)
    return session, adapter


def _capture_urlopen(monkeypatch):
    """Stub out HTTPConnectionPool.urlopen, recording the keyword arguments."""
    calls = []
    lock = threading.Lock()

    def fake_urlopen(self, **kwargs):
        with lock:
            calls.append(kwargs)
        raise RuntimeError("stop after capturing urlopen arguments")

    monkeypatch.setattr(
        urllib3.connectionpool.HTTPConnectionPool, "urlopen", fake_urlopen
    )
    return calls


class TestRewindableBodyDetection:
    def test_rewindable_bodies(self):
        assert _is_rewindable_body(None)
        assert _is_rewindable_body(b"bytes")
        assert _is_rewindable_body("text")
        assert _is_rewindable_body(io.BytesIO(b"stream"))

    def test_non_rewindable_bodies(self):
        assert not _is_rewindable_body(iter([b"a", b"b"]))
        assert not _is_rewindable_body(chunk for chunk in [b"a"])

        class NonSeekableStream:
            def read(self, size=-1):
                return b""

        assert not _is_rewindable_body(NonSeekableStream())


class TestRetryAfterParsing:
    def test_seconds_and_total_cap(self):
        retry = Retry(retry_after_max=10)
        assert retry.parse_retry_after("5") == 5
        # Bounded by the total retry-after cap.
        assert retry.parse_retry_after("999") == 10

    def test_http_date(self):
        retry = Retry(retry_after_max=60)
        future = email.utils.formatdate(time.time() + 5, usegmt=True)
        assert 0 <= retry.parse_retry_after(future) <= 5
        # Dates in the past clamp to zero instead of failing.
        assert retry.parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0

    def test_invalid_header_rejected(self):
        with pytest.raises(Urllib3InvalidHeader):
            Retry().parse_retry_after("not-a-date")

    def test_backoff_capped_by_backoff_max(self):
        retry = Retry(backoff_factor=10, backoff_max=0.5)
        for _ in range(3):
            retry = retry.increment("GET", "/", error=Exception("boom"))
        assert retry.get_backoff_time() == 0.5


class TestPerRequestRetryState:
    def test_each_request_gets_fresh_retry_state(self, monkeypatch):
        calls = _capture_urlopen(monkeypatch)
        adapter = HTTPAdapter(max_retries=Retry(total=3, status_forcelist=[503]))
        request = requests.Request("GET", "http://example.com/").prepare()

        for _ in range(2):
            with pytest.raises(RuntimeError, match="stop after capturing"):
                adapter.send(request)

        first, second = (call["retries"] for call in calls)
        assert first is not second
        assert first is not adapter.max_retries
        assert second is not adapter.max_retries
        # The adapter-level configuration is never mutated by a request.
        assert adapter.max_retries.history == ()

    def test_concurrent_requests_do_not_share_retry_state(self, monkeypatch):
        calls = _capture_urlopen(monkeypatch)
        adapter = HTTPAdapter(max_retries=Retry(total=3))
        request = requests.Request("GET", "http://example.com/").prepare()

        def send_quietly():
            with pytest.raises(RuntimeError, match="stop after capturing"):
                adapter.send(request)

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: send_quietly(), range(8)))

        retries_seen = [call["retries"] for call in calls]
        assert len(retries_seen) == 8
        assert len({id(retries) for retries in retries_seen}) == 8
        assert adapter.max_retries.history == ()


class TestStreamingBodyRetryGuard:
    def test_streaming_body_disables_resend_retries(self, monkeypatch):
        calls = _capture_urlopen(monkeypatch)
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=3, connect=2, read=3, status=3, status_forcelist=[503]
            )
        )
        request = requests.Request(
            "POST", "http://example.com/", data=iter([b"a", b"b"])
        ).prepare()

        with pytest.raises(RuntimeError, match="stop after capturing"):
            adapter.send(request)

        retries = calls[0]["retries"]
        # Only connection-establishment failures may be retried.
        assert retries.connect == 2
        assert retries.read is False
        assert retries.status == 0
        assert not retries.status_forcelist
        assert retries.redirect == 0
        assert retries.respect_retry_after_header is False

    def test_rewindable_body_keeps_retry_configuration(self, monkeypatch):
        calls = _capture_urlopen(monkeypatch)
        adapter = HTTPAdapter(
            max_retries=Retry(total=3, status=3, status_forcelist=[503])
        )
        request = requests.Request(
            "POST", "http://example.com/", data=b"payload"
        ).prepare()

        with pytest.raises(RuntimeError, match="stop after capturing"):
            adapter.send(request)

        retries = calls[0]["retries"]
        assert retries.status == 3
        assert list(retries.status_forcelist) == [503]


class TestConnectionFailureRetries:
    def test_connection_failure_retried_then_raises(self, monkeypatch):
        attempts = []

        def fail_connect(self):
            attempts.append(1)
            raise NewConnectionError(self, "simulated connection failure")

        monkeypatch.setattr(urllib3.connection.HTTPConnection, "connect", fail_connect)
        session, _ = _make_session(Retry(total=2, connect=2, backoff_factor=0))

        with pytest.raises(RequestsConnectionError) as exc_info:
            session.get("http://localhost/")

        # 1 initial attempt + 2 retries.
        assert len(attempts) == 3
        # The exception chain back to urllib3 is preserved.
        assert isinstance(exc_info.value.args[0], MaxRetryError)
        assert isinstance(exc_info.value.args[0].reason, NewConnectionError)


class TestStatusCodeRetries:
    def test_retryable_status_exhaustion_raises_with_chain(self):
        server = _status_server(503, "Service Unavailable", requests_to_handle=3)
        session, _ = _make_session(
            Retry(total=2, status=2, status_forcelist=[503], backoff_factor=0)
        )

        with server as (host, port):
            with pytest.raises(RetryError) as exc_info:
                session.get(f"http://{host}:{port}/")

        assert len(server.handler_results) == 3
        assert isinstance(exc_info.value.__context__, MaxRetryError)

    def test_retryable_status_exhaustion_returns_last_response(self):
        server = _status_server(503, "Service Unavailable", requests_to_handle=3)
        session, _ = _make_session(
            Retry(
                total=2,
                status=2,
                status_forcelist=[503],
                backoff_factor=0,
                raise_on_status=False,
            )
        )

        with server as (host, port):
            response = session.get(f"http://{host}:{port}/")

        # The last response is preserved and returned to the caller.
        assert response.status_code == 503
        assert len(server.handler_results) == 3
        response.close()

    def test_non_forcelisted_status_not_retried(self):
        server = _status_server(500, "Internal Server Error", requests_to_handle=1)
        session, _ = _make_session(
            Retry(total=2, status=2, status_forcelist=[503], backoff_factor=0)
        )

        with server as (host, port):
            response = session.get(f"http://{host}:{port}/")

        assert response.status_code == 500
        assert len(server.handler_results) == 1

    def test_non_idempotent_method_not_retried(self):
        server = _status_server(503, "Service Unavailable", requests_to_handle=1)
        session, _ = _make_session(
            Retry(total=2, status=2, status_forcelist=[503], backoff_factor=0)
        )

        with server as (host, port):
            response = session.post(f"http://{host}:{port}/", data=b"payload")

        # POST is not idempotent, so the failure is surfaced without resending.
        assert response.status_code == 503
        assert len(server.handler_results) == 1

    def test_retry_after_seconds_header_honored(self):
        server = _sequencing_server(
            [
                _http_response(
                    503, "Service Unavailable", headers=(("Retry-After", "0"),)
                ),
                _http_response(200, "OK"),
            ],
            requests_to_handle=2,
        )
        session, _ = _make_session(
            Retry(total=2, status=2, status_forcelist=[503], backoff_factor=0)
        )

        with server as (host, port):
            response = session.get(f"http://{host}:{port}/")

        assert response.status_code == 200
        assert len(server.handler_results) == 2

    def test_retry_after_http_date_header_honored(self):
        server = _sequencing_server(
            [
                _http_response(
                    503,
                    "Service Unavailable",
                    headers=(("Retry-After", "Wed, 21 Oct 2015 07:28:00 GMT"),),
                ),
                _http_response(200, "OK"),
            ],
            requests_to_handle=2,
        )
        session, _ = _make_session(
            Retry(total=2, status=2, status_forcelist=[503], backoff_factor=0)
        )

        with server as (host, port):
            response = session.get(f"http://{host}:{port}/")

        assert response.status_code == 200
        assert len(server.handler_results) == 2


class TestTimeoutAndRedirectBehavior:
    def test_read_timeout_retried_for_idempotent_method(self):
        def sleepy_handler(sock):
            request_content = consume_socket_content(sock, timeout=0.1)
            time.sleep(1)
            return request_content

        server = Server(sleepy_handler, host="127.0.0.1", requests_to_handle=2)
        session, _ = _make_session(Retry(total=1, read=1, backoff_factor=0))

        with server as (host, port):
            # Exhausted read retries surface as a ConnectionError wrapping
            # the urllib3 MaxRetryError.
            with pytest.raises(RequestsConnectionError):
                session.get(f"http://{host}:{port}/", timeout=(1, 0.2))

        # The read timeout is retried once for the idempotent GET.
        assert len(server.handler_results) == 2

    def test_read_timeout_raises_without_retries(self):
        def sleepy_handler(sock):
            request_content = consume_socket_content(sock, timeout=0.1)
            time.sleep(1)
            return request_content

        server = Server(sleepy_handler, requests_to_handle=1)
        session, _ = _make_session(0)

        with server as (host, port):
            with pytest.raises(ReadTimeout):
                session.get(f"http://{host}:{port}/", timeout=(1, 0.2))

        assert len(server.handler_results) == 1

    def test_redirect_limit_raises_too_many_redirects(self):
        session, _ = _make_session(0)
        session.max_redirects = 2
        server = _status_server(
            302,
            "Found",
            requests_to_handle=3,
            headers=(("Location", "/loop"),),
        )

        with server as (host, port):
            with pytest.raises(TooManyRedirects):
                session.get(f"http://{host}:{port}/loop")

        # 1 initial request + 2 followed redirects.
        assert len(server.handler_results) == 3


class TestStreamingBodyEndToEnd:
    def test_streaming_body_never_resent(self):
        server = _status_server(503, "Service Unavailable", requests_to_handle=1)
        session, _ = _make_session(
            Retry(total=3, status=3, status_forcelist=[503], backoff_factor=0)
        )

        with server as (host, port):
            response = session.post(
                f"http://{host}:{port}/", data=iter([b"a", b"b", b"c"])
            )

        # The streaming body must not be re-sent: exactly one request was made
        # and the 503 response is surfaced to the caller.
        assert response.status_code == 503
        assert len(server.handler_results) == 1
        response.close()


class TestConcurrentSessionUse:
    def test_concurrent_requests_on_shared_session(self):
        request_count = 8
        server = _status_server(200, "OK", requests_to_handle=request_count)
        session, adapter = _make_session(
            Retry(total=3, status=3, status_forcelist=[503], backoff_factor=0)
        )

        with server as (host, port):
            url = f"http://{host}:{port}/"
            with ThreadPoolExecutor(max_workers=4) as executor:
                responses = list(
                    executor.map(lambda _: session.get(url), range(request_count))
                )

        assert [r.status_code for r in responses] == [200] * request_count
        assert len(server.handler_results) == request_count
        # No request mutated the shared retry configuration.
        assert adapter.max_retries.history == ()
        for response in responses:
            response.close()
