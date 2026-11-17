"""Unit tests for _client.submit()'s multipart form construction.

No network is used -- requests.post is faked to capture what would be sent.
"""

from __future__ import annotations

from manhattan_reasoning_gym import _client


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"job_id": "job-123"}


def test_submit_omits_top_when_not_given(monkeypatch, tmp_path):
    design = tmp_path / "design.py"
    design.write_text("# design\n")
    seen = {}

    def fake_post(url, headers, files, data):
        seen["data"] = data
        return _FakeResponse()

    monkeypatch.setattr(_client.requests, "post", fake_post)
    _client.submit(str(design), "key", "https://api.example")
    assert seen["data"] is None


def test_submit_includes_top_when_given(monkeypatch, tmp_path):
    design = tmp_path / "design.v"
    design.write_text("module m; endmodule\n")
    seen = {}

    def fake_post(url, headers, files, data):
        seen["data"] = data
        return _FakeResponse()

    monkeypatch.setattr(_client.requests, "post", fake_post)
    _client.submit(str(design), "key", "https://api.example", top="echo_slave")
    assert seen["data"] == {"top": "echo_slave"}


def test_submit_combines_top_with_other_fields(monkeypatch, tmp_path):
    design = tmp_path / "design.v"
    design.write_text("module m; endmodule\n")
    seen = {}

    def fake_post(url, headers, files, data):
        seen["data"] = data
        return _FakeResponse()

    monkeypatch.setattr(_client.requests, "post", fake_post)
    _client.submit(
        str(design), "key", "https://api.example",
        top="echo_slave", sys_clk_freq=90_000_000,
    )
    assert seen["data"] == {"top": "echo_slave", "sys_clk_freq": "90000000"}


class _Resp:
    def __init__(self, status_code=202, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _client.requests.HTTPError(str(self.status_code))

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def test_reclock_posts_the_rate_and_returns_the_snapped_variant(monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, json=json)
        return _Resp(202, {"job_id": "j1", "fpga_id": 3, "sys_clk_freq": 85_714_286})

    monkeypatch.setattr(_client.requests, "post", fake_post)
    out = _client.reclock(3, "key", 86e6, "https://api.example")
    assert seen == {
        "url": "https://api.example/fpga/3/reclock",
        "json": {"sys_clk_freq": 86_000_000},
    }
    assert out["sys_clk_freq"] == 85_714_286


def test_reclock_surfaces_the_servers_reason(monkeypatch):
    detail = {"detail": {"error": "clock_not_available", "message": "Available: 100"}}
    monkeypatch.setattr(
        _client.requests, "post", lambda url, headers, json, timeout: _Resp(422, detail)
    )
    try:
        _client.reclock(3, "key", 110_000_000, "https://api.example")
    except _client.ReclockError as exc:
        assert str(exc) == "Available: 100"
    else:
        raise AssertionError("expected ReclockError")


def test_wait_job_returns_once_the_job_is_terminal(monkeypatch):
    statuses = iter(["queued", "running", "complete"])
    monkeypatch.setattr(
        _client, "get_job", lambda job_id, key, url: {"status": next(statuses)}
    )
    monkeypatch.setattr(_client.time, "sleep", lambda _s: None)
    assert _client.wait_job("j1", "key", "https://api.example")["status"] == "complete"
