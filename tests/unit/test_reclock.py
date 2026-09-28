"""App.reclock: switch a programmed board's clock without rebuilding."""

from __future__ import annotations

import pytest

from manhattan_reasoning_gym import _client
from manhattan_reasoning_gym._app import App


@pytest.fixture
def app():
    a = App("t", design="d.py", api_key="key", api_url="https://api.example")
    a.fpga_id, a._programmed = 3, True
    return a


def _queued(monkeypatch, status, fpga_state="reserved"):
    monkeypatch.setattr(
        _client,
        "reclock",
        lambda fpga_id, key, hz, url: {"job_id": "j1", "sys_clk_freq": 100_000_000},
    )
    monkeypatch.setattr(
        _client, "wait_job", lambda job_id, key, url: {"status": status}
    )
    monkeypatch.setattr(
        _client, "get_fpga", lambda fpga_id, key, url: {"state": fpga_state}
    )


def test_success_returns_and_records_the_exact_clock(app, monkeypatch):
    _queued(monkeypatch, "complete")
    assert app.reclock(99.5e6) == 100_000_000
    assert app.sys_clk_freq == 100_000_000
    assert app._programmed


def test_a_failed_flash_raises_and_forgets_the_board_if_it_was_released(
    app, monkeypatch
):
    _queued(monkeypatch, "failed", fpga_state="idle")
    with pytest.raises(_client.ReclockError, match="now idle"):
        app.reclock(100e6)
    assert not app._programmed


def test_a_refused_flash_that_left_the_board_reserved_keeps_it(app, monkeypatch):
    _queued(monkeypatch, "failed", fpga_state="reserved")
    with pytest.raises(_client.ReclockError):
        app.reclock(100e6)
    assert app._programmed


def test_nothing_programmed_is_refused_without_a_request(monkeypatch):
    a = App("t", design="d.py", api_key="key", api_url="https://api.example")
    monkeypatch.setattr(_client, "reclock", lambda *a: pytest.fail("no request"))
    with pytest.raises(_client.ReclockError, match="no board programmed"):
        a.reclock(100e6)
