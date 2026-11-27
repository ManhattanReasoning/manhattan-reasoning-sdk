"""Short job ids in `mrg job/logs/cancel`, and poll_job riding out outages."""

import pytest
import requests

from manhattan_reasoning_gym import _client

FULL_A = "075dab40-1111-4222-8333-444455556666"
FULL_B = "075dab99-1111-4222-8333-444455556666"
FULL_C = "9f00aa00-1111-4222-8333-444455556666"


@pytest.fixture
def jobs(monkeypatch):
    listed = [{"job_id": j} for j in (FULL_A, FULL_B, FULL_C)]
    calls = []

    def fake_list(*a, **k):
        calls.append(a)
        return listed

    monkeypatch.setattr(_client, "list_jobs", fake_list)
    return calls


def test_full_id_is_used_without_listing(jobs):
    assert _client.resolve_job_id(FULL_C.upper(), "k", "u") == FULL_C
    assert jobs == []


@pytest.mark.parametrize("given", ["9f00aa00", "9f00aa00…", "9F00", "075dab4"])
def test_unique_prefix_resolves(jobs, given):
    expected = FULL_C if given.lower().startswith("9f") else FULL_A
    assert _client.resolve_job_id(given, "k", "u") == expected


def test_ambiguous_prefix_is_refused(jobs):
    with pytest.raises(ValueError, match="2 of your jobs"):
        _client.resolve_job_id("075dab", "k", "u")


def test_unknown_prefix_is_refused(jobs):
    with pytest.raises(ValueError, match="no job"):
        _client.resolve_job_id("deadbeef", "k", "u")


class _Resp:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self.ok = status_code < 400
        self._body = body or {}
        self.text = ""

    def json(self):
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(response=self)


@pytest.fixture
def fast(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(_client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        _client.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s)
    )
    return clock


def test_poll_job_survives_connection_errors(monkeypatch, fast):
    replies = [
        requests.ConnectionError("reset by peer"),
        _Resp(502),
        requests.Timeout("slow"),
        _Resp(200, {"status": "complete", "session_id": "s"}),
    ]

    def fake_get(*a, **k):
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(_client.requests, "get", fake_get)
    job = _client.poll_job("j", "k", "u", timeout=600)
    assert job["status"] == "complete"
    assert replies == []


def test_poll_job_gives_up_after_a_long_outage(monkeypatch, fast):
    def down(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(_client.requests, "get", down)
    with pytest.raises(requests.ConnectionError):
        _client.poll_job("j", "k", "u", timeout=10_000)
    assert _client._POLL_OUTAGE_GRACE < fast[0] < _client._POLL_OUTAGE_GRACE + 10


def test_poll_job_does_not_retry_client_errors(monkeypatch, fast):
    monkeypatch.setattr(_client.requests, "get", lambda *a, **k: _Resp(404))
    with pytest.raises(requests.HTTPError):
        _client.poll_job("j", "k", "u", timeout=600)


class _LogResp:
    def __init__(self, ctype, text, body):
        self.headers = {"content-type": ctype}
        self.text = text
        self._body = body

    def json(self):
        return self._body


def test_logs_sent_as_a_json_string_print_as_text():
    resp = _LogResp("application/json", '"a\\nb\\n"', "a\nb\n")
    assert _client._log_text(resp) == "a\nb\n"


def test_plain_text_logs_pass_through():
    assert _client._log_text(_LogResp("text/plain", "a\nb", None)) == "a\nb"
