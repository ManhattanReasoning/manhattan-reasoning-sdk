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


SESSION = "5f0c6f1e-2b7a-4c1e-9d8f-3a2b1c0d9e8f"


def _status(session=None, json=False, board=None, boards=False):
    return argparse.Namespace(session=session, json=json, board=board, boards=boards)


def test_an_empty_list_says_you_hold_no_board(monkeypatch, capsys, quiet_server):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("key", "https://x"))
    monkeypatch.setattr(_client, "list_sessions", lambda *args: [])
    _cli.cmd_status(_status())
    out = capsys.readouterr().out
    assert "You don't hold a board right now" in out
    assert "STATE" not in out  # no empty table
    assert "online" not in out.lower()  # nothing about the fleet


def test_an_empty_list_as_json_is_an_empty_array(monkeypatch, capsys):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("key", "https://x"))
    monkeypatch.setattr(_client, "list_sessions", lambda *args: [])
    _cli.cmd_status(_status(json=True))
    assert capsys.readouterr().out.strip() == "[]"


@pytest.mark.parametrize("session", [None, SESSION])
def test_not_signed_in_is_a_clear_error_not_a_traceback(monkeypatch, session):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("", "https://x"))
    monkeypatch.setattr(_client, "list_sessions", Mock(side_effect=_http_error(401)))
    monkeypatch.setattr(_client, "get_session", Mock(side_effect=_http_error(401)))
    with pytest.raises(SystemExit) as exc:
        _cli.cmd_status(_status(session))
    assert "mrg login" in str(exc.value)


def test_an_ended_or_unknown_session_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("key", "https://x"))
    monkeypatch.setattr(_client, "get_session", Mock(side_effect=_http_error(404)))
    with pytest.raises(SystemExit) as exc:
        _cli.cmd_status(_status(SESSION))
    assert "it has ended, or it isn't yours" in str(exc.value)


def test_reset_of_a_session_that_isnt_yours_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(_cli, "_creds", lambda args: ("key", "https://x"))
    monkeypatch.setattr(_client, "reset_session", Mock(side_effect=_http_error(404)))
    with pytest.raises(SystemExit) as exc:
        _cli.cmd_reset(
            argparse.Namespace(session=SESSION, legacy_board=None, board=None)
        )
    assert "it has ended, or it isn't yours" in str(exc.value)


def test_reset_takes_the_session_positionally(monkeypatch):
    monkeypatch.setattr(_cli, "_creds", lambda args: ("key", "https://x"))
    reset = Mock(return_value={"job_id": "j"})
    monkeypatch.setattr(_client, "reset_session", reset)
    _cli.cmd_reset(argparse.Namespace(session=None, legacy_board=SESSION, board=None))
    assert reset.call_args.args[0] == SESSION


def test_board_flags_are_refused_for_regular_users(monkeypatch):
    """The server answers 410 to anyone but an admin."""
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("key", "https://x"))
    monkeypatch.setattr(_client, "get_fpga", Mock(side_effect=_http_error(410)))
    monkeypatch.setattr(_client, "list_fpgas", Mock(side_effect=_http_error(410)))
    for args in (_status(board=3), _status(boards=True)):
        with pytest.raises(SystemExit) as exc:
            _cli.cmd_status(args)
        assert "operators" in str(exc.value)


def test_sessions_table_shows_no_board_numbers(monkeypatch, capsys, quiet_server):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("key", "https://x"))
    monkeypatch.setattr(
        _client, "list_sessions",
        lambda *a: [{"session_id": SESSION, "state": "reserved"}],
    )
    _cli.cmd_status(_status())
    out = capsys.readouterr().out
    assert SESSION[:8] in out and "FPGA" not in out and "BOARD" not in out


def test_other_server_errors_still_raise(monkeypatch):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("key", "https://x"))
    monkeypatch.setattr(_client, "list_sessions", Mock(side_effect=_http_error(500)))
    with pytest.raises(requests.HTTPError):
        _cli.cmd_status(_status())


def test_a_failed_reclock_on_an_ended_session_still_raises_reclock_error(
    monkeypatch,
):
    app = App("t", design="d.py", api_key="key", api_url="https://api.example")
    app.session_id, app._programmed = SESSION, True
    monkeypatch.setattr(
        _client,
        "reclock",
        lambda *args: {"job_id": "j1", "sys_clk_freq": 100_000_000},
    )
    monkeypatch.setattr(_client, "wait_job", lambda *args: {"status": "failed"})
    monkeypatch.setattr(_client, "get_session", Mock(side_effect=_http_error(404)))
    with pytest.raises(_client.ReclockError, match="session has ended"):
        app.reclock(100e6)
    assert not app._programmed
