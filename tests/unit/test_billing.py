"""Unit tests for the membership commands: subscribe, billing, usage.

No network is used -- requests is faked, and nothing opens a browser.
"""

from __future__ import annotations

import json
import sys

import pytest

from manhattan_reasoning_gym import _cli, _client

API = "https://api.example"


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _client.requests.HTTPError(str(self.status_code))

    def json(self):
        return self._body


def _detail(error, message="server message"):
    return {"detail": {"error": error, "message": message}}


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["mrg", *argv, "--api-key", "k", "--api-url", API])
    _cli.main()


@pytest.mark.parametrize(
    ("call", "path"),
    [(_client.create_checkout, "/billing/checkout"),
     (_client.create_portal, "/billing/portal")],
)
def test_session_calls_post_and_return_url(monkeypatch, call, path):
    seen = {}

    def fake_post(url, headers, timeout):
        seen.update(url=url, headers=headers)
        return _Resp(200, {"url": "https://stripe.example/s"})

    monkeypatch.setattr(_client.requests, "post", fake_post)
    assert call("key", API) == "https://stripe.example/s"
    assert seen == {"url": API + path, "headers": {"X-API-Key": "key"}}


@pytest.mark.parametrize(
    ("status", "code", "hint"),
    [(409, "no_stripe_customer", "mrg login"),
     (409, "already_subscribed", "mrg billing"),
     (503, "billing_unavailable", None)],
)
def test_session_refusals_become_billing_errors(monkeypatch, status, code, hint):
    monkeypatch.setattr(
        _client.requests, "post",
        lambda url, headers, timeout: _Resp(status, _detail(code)),
    )
    with pytest.raises(_client.BillingError) as exc:
        _client.create_checkout("key", API)
    assert exc.value.code == code
    assert "server message" in str(exc.value)
    if hint:
        assert hint in str(exc.value)


@pytest.mark.parametrize(
    ("code", "hint"),
    [("subscription_required", "mrg subscribe"),
     ("billing_blocked", "mrg billing"),
     ("quota_exhausted", "mrg usage"),
     ("build_in_flight", "mrg jobs")],
)
def test_submit_402_names_the_command_that_fixes_it(monkeypatch, tmp_path, code, hint):
    design = tmp_path / "design.py"
    design.write_text("# design\n")
    monkeypatch.setattr(
        _client.requests, "post",
        lambda url, headers, files, data: _Resp(402, _detail(code)),
    )
    with pytest.raises(_client.BillingError) as exc:
        _client.submit(str(design), "key", API)
    assert exc.value.code == code
    assert hint in str(exc.value)


def test_submit_402_with_unparseable_body_still_raises_billing_error(
    monkeypatch, tmp_path
):
    design = tmp_path / "design.py"
    design.write_text("# design\n")
    monkeypatch.setattr(
        _client.requests, "post",
        lambda url, headers, files, data: _Resp(402, {"detail": "nope"}),
    )
    with pytest.raises(_client.BillingError) as exc:
        _client.submit(str(design), "key", API)
    assert exc.value.code == "unknown"


@pytest.mark.parametrize(
    ("cmd", "fn"), [("subscribe", "create_checkout"), ("billing", "create_portal")]
)
def test_url_commands_print_the_url_without_opening_a_browser(
    monkeypatch, capsys, cmd, fn
):
    import webbrowser

    monkeypatch.setattr(_client, fn, lambda key, url: "https://stripe.example/s")
    monkeypatch.setattr(
        webbrowser, "open", lambda url: pytest.fail("opened a browser")
    )
    _main(monkeypatch, cmd)  # stdout is not a TTY under pytest
    assert "https://stripe.example/s" in capsys.readouterr().out


def test_subscribe_exits_with_the_server_reason(monkeypatch):
    def refuse(key, url):
        raise _client.BillingError("no_stripe_customer", "No customer yet.")

    monkeypatch.setattr(_client, "create_checkout", refuse)
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "subscribe")
    assert "No customer yet." in str(exc.value)
    assert "mrg login" in str(exc.value)


_USAGE = {
    "month": "2026-11",
    "build_minutes_used": 42,
    "build_minutes_limit": 1000,
    "build_minutes_remaining": 958,
    "builds": [{
        "job_id": "0b6f3c1e-0000-0000-0000-000000000000",
        "build_minutes": 7,
        "started_at": "2026-11-16T10:00:00Z",
        "ended_at": "2026-11-16T10:07:00Z",
        "source": "build_complete",
    }],
}


def test_usage_table(monkeypatch, capsys):
    monkeypatch.setattr(_client, "get_usage", lambda key, url: _USAGE)
    _main(monkeypatch, "usage")
    out = capsys.readouterr().out
    assert "42 of 1000 build-minutes" in out
    assert "958" in out
    assert "0b6f3c1e" in out


def test_usage_json_is_the_raw_response(monkeypatch, capsys):
    monkeypatch.setattr(_client, "get_usage", lambda key, url: _USAGE)
    _main(monkeypatch, "usage", "--json")
    assert json.loads(capsys.readouterr().out) == _USAGE
