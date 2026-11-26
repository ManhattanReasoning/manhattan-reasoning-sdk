import pytest

import manhattan_reasoning_gym
from manhattan_reasoning_gym import _client


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("MRG_API_KEY", "test-key")
    return tmp_path


# ── App session discovery ────────────────────────────────────────────────────
#
# A build never picks a board: the server assigns one once the build finishes.
# The App learns its session id from the completed job, and never a board id.


def _complete(session_id="sess-a"):
    return lambda *a, **kw: {"status": "complete", "session_id": session_id}


def test_session_id_is_none_until_programmed(monkeypatch):
    monkeypatch.setattr(_client, "submit", lambda *a, **kw: "job")
    monkeypatch.setattr(_client, "poll_job", _complete("sess-2"))

    app = manhattan_reasoning_gym.App("x", design="d.py")
    assert app.session_id is None
    app._program()

    assert app.session_id == "sess-2"


def test_submit_is_never_given_a_board(monkeypatch):
    calls = {}

    def fake_submit(design_path, api_key, api_url, **kw):
        calls["design_path"] = design_path
        calls.update(kw)
        return "job"

    monkeypatch.setattr(_client, "submit", fake_submit)
    monkeypatch.setattr(_client, "poll_job", _complete())

    app = manhattan_reasoning_gym.App("x", design="d.py")
    app._program()

    assert calls["design_path"] == "d.py"
    assert not any("fpga" in k or "board" in k for k in calls)


def test_a_completed_job_without_a_session_is_a_clear_error(monkeypatch):
    """A server too old to hand out sessions: say so rather than guess."""
    monkeypatch.setattr(_client, "submit", lambda *a, **kw: "job")
    monkeypatch.setattr(
        _client, "poll_job", lambda *a, **kw: {"status": "complete", "fpga_id": 2}
    )
    app = manhattan_reasoning_gym.App("x", design="d.py")
    with pytest.raises(RuntimeError, match="no session id"):
        app._program()


def test_builds_tell_the_server_this_client_uses_sessions(monkeypatch, tmp_path):
    design = tmp_path / "d.py"
    design.write_text("# design\n")
    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"job_id": "job"}

    def fake_post(url, headers, files, data):
        seen["headers"] = headers
        return _Resp()

    monkeypatch.setattr(_client.requests, "post", fake_post)
    _client.submit(str(design), "k", "u")
    assert seen["headers"]["X-MRG-Client-Features"] == "sessions"


# ── timing-target plumbing ───────────────────────────────────────────────────

def test_app_forwards_timing_target_to_submit(monkeypatch):
    calls = {}

    def fake_submit(design_path, api_key, api_url, **kw):
        calls.update(kw)
        return "job"

    monkeypatch.setattr(_client, "submit", fake_submit)
    monkeypatch.setattr(_client, "poll_job", _complete())

    app = manhattan_reasoning_gym.App("x", design="d.py", timing_target_mhz=90)
    app._program()
    assert calls["timing_target_mhz"] == 90


def test_app_timing_target_from_env(monkeypatch):
    monkeypatch.setenv("MRG_TIMING_TARGET_MHZ", "75")
    app = manhattan_reasoning_gym.App("x", design="d.py")
    assert app.timing_target_mhz == 75.0


def test_submit_sends_timing_target_form_field(monkeypatch, tmp_path):
    design = tmp_path / "d.py"
    design.write_text("# design\n")
    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"job_id": "job"}

    def fake_post(url, headers, files, data):
        seen["data"] = data
        return _Resp()

    monkeypatch.setattr(_client.requests, "post", fake_post)

    _client.submit(str(design), "k", "u", timing_target_mhz=90)
    assert seen["data"] == {"timing_target_mhz": "90"}

    # Nothing set => no form body at all (older servers get an unchanged request).
    _client.submit(str(design), "k", "u")
    assert seen["data"] is None
