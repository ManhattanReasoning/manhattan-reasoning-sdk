"""`mrg admin`: the physical-stack view and board online/offline toggles."""

import argparse
from unittest.mock import Mock

import pytest
import requests

from manhattan_reasoning_gym import _cli, _client

STACK = [
    {"fpga_id": 4, "state": "error", "online": False, "jtag_present": True,
     "network_up": False, "owner": None},
    {"fpga_id": 2, "state": "idle", "online": True, "jtag_present": True,
     "network_up": None, "owner": None},
    {"fpga_id": 0, "state": "reserved", "online": True, "jtag_present": None,
     "network_up": True, "owner": "alice"},
]


def _args(action=None, fpga_ids=(), json=False):
    return argparse.Namespace(action=action, fpga_ids=list(fpga_ids), json=json)


@pytest.fixture(autouse=True)
def creds(monkeypatch):
    monkeypatch.setattr(_cli, "_creds", lambda args: ("key", "https://example"))


def _http_error(status: int, body=None) -> requests.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    resp._content = b"{}" if body is None else body
    return requests.HTTPError(response=resp)


def test_stack_prints_boards_top_first_with_offline_marked(monkeypatch, capsys):
    monkeypatch.setattr(_client, "admin_list_fpgas", lambda *a: STACK)
    _cli.cmd_admin(_args())  # not a tty under pytest: prints and exits
    out = capsys.readouterr().out
    assert out.index("FPGA 4") < out.index("FPGA 2") < out.index("FPGA 0")
    assert "OFFLINE" in out
    assert "offline (no new builds): 4" in out
    # every row of the box is the same width
    widths = {len(line) for line in out.splitlines() if line.startswith("  │")}
    assert len(widths) == 1


@pytest.mark.parametrize("action,online", [("offline", False), ("online", True)])
def test_online_offline_toggle_each_board(monkeypatch, capsys, action, online):
    calls = Mock(side_effect=lambda i, o, *a: {"fpga_id": i, "state": "idle"})
    monkeypatch.setattr(_client, "set_fpga_online", calls)
    _cli.cmd_admin(_args(action, [4, 2]))
    assert [c.args[:2] for c in calls.call_args_list] == [(4, online), (2, online)]
    assert capsys.readouterr().out.count(action) == 2


def test_offline_of_a_busy_board_says_the_session_finishes(monkeypatch, capsys):
    monkeypatch.setattr(
        _client, "set_fpga_online", lambda *a: {"fpga_id": 0, "state": "reserved"}
    )
    _cli.cmd_admin(_args("offline", [0]))
    assert "current session finishes first" in capsys.readouterr().out


def test_action_needs_ids():
    with pytest.raises(SystemExit, match="needs at least one FPGA id"):
        _cli.cmd_admin(_args("offline"))


def test_non_admin_gets_a_clean_refusal(monkeypatch):
    def refuse(*a):
        raise _http_error(403)

    monkeypatch.setattr(_client, "admin_list_fpgas", refuse)
    with pytest.raises(SystemExit, match="admins only"):
        _cli.cmd_admin(_args())


def test_unknown_board_reports_the_servers_message(monkeypatch):
    def missing(*a):
        raise _http_error(
            404, b'{"detail": {"message": "FPGA 9 is not a live board."}}'
        )

    monkeypatch.setattr(_client, "set_fpga_online", missing)
    with pytest.raises(SystemExit, match="FPGA 9 is not a live board"):
        _cli.cmd_admin(_args("offline", [9]))


def test_interactive_toggle_confirms_then_flips(monkeypatch, capsys):
    monkeypatch.setattr(_client, "admin_list_fpgas", lambda *a: STACK)
    calls = Mock(return_value={"fpga_id": 4, "state": "error"})
    monkeypatch.setattr(_client, "set_fpga_online", calls)
    answers = iter(["4", "y", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    _cli._admin_interactive("key", "https://example")
    calls.assert_called_once_with(4, True, "key", "https://example")  # 4 was offline


def test_interactive_declined_confirmation_changes_nothing(monkeypatch):
    monkeypatch.setattr(_client, "admin_list_fpgas", lambda *a: STACK)
    calls = Mock()
    monkeypatch.setattr(_client, "set_fpga_online", calls)
    answers = iter(["2", "n", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    _cli._admin_interactive("key", "https://example")
    calls.assert_not_called()
