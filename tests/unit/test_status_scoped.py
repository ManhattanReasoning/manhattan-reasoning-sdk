"""`mrg status` / `mrg reset` now that board status is scoped to the signed-in user.

The server answers 401 without a key, lists only the caller's boards (often
none), and gives 404 for a board that isn't theirs.
"""

from __future__ import annotations

import argparse
from unittest.mock import Mock

import pytest
import requests

from manhattan_reasoning_gym import _cli, _client
from manhattan_reasoning_gym._app import App


def _http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} error", response=response)


@pytest.fixture
def quiet_server(monkeypatch):
    monkeypatch.setattr(_client, "get_health", Mock(return_value={}))
    monkeypatch.setattr(_client, "list_jobs", Mock(return_value=[]))


def _status(fpga_id=None, json=False):
    return argparse.Namespace(fpga_id=fpga_id, json=json)


def test_an_empty_list_says_you_hold_no_board(monkeypatch, capsys, quiet_server):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("key", "https://x"))
    monkeypatch.setattr(_client, "list_fpgas", lambda *args: [])
    _cli.cmd_status(_status())
    out = capsys.readouterr().out
    assert "You don't hold a board right now" in out
    assert "STATE" not in out  # no empty table
    assert "online" not in out.lower()  # nothing about the fleet


def test_an_empty_list_as_json_is_an_empty_array(monkeypatch, capsys):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("key", "https://x"))
    monkeypatch.setattr(_client, "list_fpgas", lambda *args: [])
    _cli.cmd_status(_status(json=True))
    assert capsys.readouterr().out.strip() == "[]"


@pytest.mark.parametrize("fpga_id", [None, 3])
def test_not_signed_in_is_a_clear_error_not_a_traceback(monkeypatch, fpga_id):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("", "https://x"))
    monkeypatch.setattr(_client, "list_fpgas", Mock(side_effect=_http_error(401)))
    monkeypatch.setattr(_client, "get_fpga", Mock(side_effect=_http_error(401)))
    with pytest.raises(SystemExit) as exc:
        _cli.cmd_status(_status(fpga_id))
    assert "mrg login" in str(exc.value)


def test_someone_elses_board_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("key", "https://x"))
    monkeypatch.setattr(_client, "get_fpga", Mock(side_effect=_http_error(404)))
    with pytest.raises(SystemExit) as exc:
        _cli.cmd_status(_status(3))
    assert "isn't one of your boards" in str(exc.value)


def test_reset_of_a_board_that_isnt_yours_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(_cli, "_creds", lambda args: ("key", "https://x"))
    monkeypatch.setattr(_client, "reset_fpga", Mock(side_effect=_http_error(404)))
    with pytest.raises(SystemExit) as exc:
        _cli.cmd_reset(argparse.Namespace(fpga_id=3))
    assert "isn't one of your boards" in str(exc.value)


def test_other_server_errors_still_raise(monkeypatch):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("key", "https://x"))
    monkeypatch.setattr(_client, "list_fpgas", Mock(side_effect=_http_error(500)))
    with pytest.raises(requests.HTTPError):
        _cli.cmd_status(_status())


def test_a_failed_reclock_on_a_board_already_returned_still_raises_reclock_error(
    monkeypatch,
):
    app = App("t", design="d.py", api_key="key", api_url="https://api.example")
    app.fpga_id, app._programmed = 3, True
    monkeypatch.setattr(
        _client,
        "reclock",
        lambda *args: {"job_id": "j1", "sys_clk_freq": 100_000_000},
    )
    monkeypatch.setattr(_client, "wait_job", lambda *args: {"status": "failed"})
    monkeypatch.setattr(_client, "get_fpga", Mock(side_effect=_http_error(404)))
    with pytest.raises(_client.ReclockError, match="released"):
        app.reclock(100e6)
    assert not app._programmed
