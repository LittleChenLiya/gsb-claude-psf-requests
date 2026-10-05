"""Tests for the exclusive raw-stream consumption session on Response.

State machine (``Response.stream_state``):

* ``idle``     -- no consumer holds the raw stream; it may be read from
                  the start.
* ``active``   -- one consumer session exclusively holds the raw stream;
                  any other consumer is rejected.
* ``consumed`` -- the stream was fully or partially consumed (including
                  cancelled or failed iterations) and cannot be re-read
                  from the start.
* ``closed``   -- ``close()`` was called and the underlying connection
                  was released.
"""

import pickle

import pytest

from requests.exceptions import StreamConsumedError
from requests.models import Response


class FakeRaw:
    """A minimal file-like raw stream recording its lifecycle."""

    def __init__(self, chunks=(b"aa", b"bb", b"cc"), fail_after_reads=None):
        self._chunks = list(chunks)
        self._fail_after_reads = fail_after_reads
        self.reads = 0
        self.closed = False
        self.released = False

    def read(self, chunk_size):
        if self.closed:
            raise ValueError("I/O operation on a closed raw stream")
        self.reads += 1
        if self._fail_after_reads is not None and self.reads > self._fail_after_reads:
            raise OSError("simulated read failure")
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def close(self):
        self.closed = True

    def release_conn(self):
        self.released = True


def make_response(chunks=(b"aa", b"bb", b"cc"), **raw_kwargs):
    r = Response()
    r.status_code = 200
    r.raw = FakeRaw(chunks, **raw_kwargs)
    return r


def test_default_single_consumer_behaviour_unchanged():
    r = make_response()
    assert r.stream_state == Response.STREAM_IDLE
    assert list(r.iter_content(1)) == [b"aa", b"bb", b"cc"]
    assert r.stream_state == Response.STREAM_CONSUMED
    assert r._content_consumed is True


def test_whole_reads_work_and_cache_replay_still_allowed():
    r = make_response(chunks=(b'{"a": ', b"1}"))
    assert r.content == b'{"a": 1}'
    assert r.text == '{"a": 1}'
    assert r.json() == {"a": 1}
    # cached content can still be replayed from memory
    assert b"".join(r.iter_content(1)) == b'{"a": 1}'


def test_second_consumer_rejected_while_first_holds_stream():
    r = make_response()
    first = r.iter_content(1)
    assert next(first) == b"aa"
    assert r.stream_state == Response.STREAM_ACTIVE

    second = r.iter_content(1)
    with pytest.raises(RuntimeError, match="already being consumed"):
        next(second)

    # the first consumer is undisturbed and can finish
    assert list(first) == [b"bb", b"cc"]
    assert r.stream_state == Response.STREAM_CONSUMED


def test_finished_stream_is_consumed_not_restartable():
    r = make_response()
    assert list(r.iter_content(2)) == [b"aa", b"bb", b"cc"]
    assert r.stream_state == Response.STREAM_CONSUMED
    with pytest.raises(StreamConsumedError):
        r.iter_content(1)


def test_early_close_releases_connection():
    r = make_response()
    first = r.iter_content(1)
    assert next(first) == b"aa"

    r.close()
    assert r.raw.closed is True
    assert r.raw.released is True
    assert r.stream_state == Response.STREAM_CLOSED

    with pytest.raises(StreamConsumedError, match="closed"):
        next(r.iter_content(1))


def test_exception_marks_stream_consumed_not_idle():
    r = make_response(fail_after_reads=1)
    first = r.iter_content(1)
    assert next(first) == b"aa"
    with pytest.raises(OSError, match="simulated read failure"):
        next(first)

    # the failure must not release the stream as unconsumed
    assert r.stream_state == Response.STREAM_CONSUMED
    assert r.stream_state != Response.STREAM_IDLE
    with pytest.raises(StreamConsumedError):
        next(r.iter_content(1))

    # whole-body reads follow the occupancy check too
    with pytest.raises(StreamConsumedError):
        r.content

    # close() still releases the underlying connection
    r.close()
    assert r.raw.closed is True
    assert r.raw.released is True
    assert r.stream_state == Response.STREAM_CLOSED


def test_cancelled_iteration_marks_stream_consumed():
    r = make_response()
    first = r.iter_content(1)
    assert next(first) == b"aa"
    first.close()  # GeneratorExit: cancelled mid-stream
    assert r.stream_state == Response.STREAM_CONSUMED
    with pytest.raises(StreamConsumedError):
        next(r.iter_content(1))


def test_text_and_json_rejected_while_stream_busy():
    r = make_response(chunks=(b'{"a":', b" 1}"))
    first = r.iter_content(1)
    assert next(first) == b'{"a":'

    with pytest.raises(RuntimeError, match="already being consumed"):
        r.content
    with pytest.raises(RuntimeError, match="already being consumed"):
        r.text
    with pytest.raises(RuntimeError, match="already being consumed"):
        r.json()

    first.close()


def test_explicit_session_allows_holder_and_releases():
    r = make_response()
    with r.stream_session() as session:
        assert session is r
        assert r.stream_state == Response.STREAM_ACTIVE
        # the holder may iterate; a second explicit acquire is rejected
        with pytest.raises(RuntimeError):
            r.acquire_stream()
        assert list(r.iter_content(2)) == [b"aa", b"bb", b"cc"]
    assert r.stream_state == Response.STREAM_CONSUMED


def test_explicit_session_release_without_reading_returns_to_idle():
    r = make_response()
    r.acquire_stream()
    assert r.stream_state == Response.STREAM_ACTIVE
    r.release_stream()
    assert r.stream_state == Response.STREAM_IDLE
    # nothing was consumed, so the stream can still be read from the start
    assert list(r.iter_content(3)) == [b"aa", b"bb", b"cc"]


def test_release_stream_without_session_raises():
    r = make_response()
    with pytest.raises(RuntimeError, match="No explicit stream"):
        r.release_stream()


def test_pickled_response_is_consumed():
    r = make_response()
    _ = r.content
    r2 = pickle.loads(pickle.dumps(r))
    assert r2.stream_state == Response.STREAM_CONSUMED
    assert r2.content == b"aabbcc"
