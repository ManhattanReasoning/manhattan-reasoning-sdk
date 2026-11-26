import io

import pytest

from manhattan_reasoning_gym import _client, _progress


def _bar(color):
    s = io.StringIO()
    b = _progress.BuildProgress("demo", stream=s)
    b.color = color
    return b, s


# ── BuildProgress: non-TTY fallback ──────────────────────────────────────────

def test_non_tty_prints_each_phase_once():
    b, s = _bar(False)
    for st in ["queued", "queued", "running", "running"]:
        b.update(st)
    out = s.getvalue()
    assert out.count("queued") == 1            # no duplicate spam
    assert "building and flashing" in out
    assert "\r" not in out                     # no animation control chars

def test_the_progress_line_never_names_a_board():
    b, s = _bar(False)
    b.update("running", 3)  # a stray second argument is ignored
    b.finish(3)
    assert "fpga" not in s.getvalue().lower() and "3" not in s.getvalue()

def test_unknown_status_passes_through():
    b, s = _bar(False)
    b.update("weird")
    assert "weird" in s.getvalue()

def test_finish_reports_ready():
    b, s = _bar(False)
    b.finish()
    out = s.getvalue()
    assert "ready" in out and "demo" in out


# ── BuildProgress: TTY rendering ─────────────────────────────────────────────

def test_tty_renders_spinner_clock_and_ansi():
    b, s = _bar(True)
    b.update("running")
    out = s.getvalue()
    assert "\r" in out and "\033[" in out                  # redraw + color
    assert any(f in out for f in _progress._FRAMES)        # a spinner frame
    assert "building and flashing" in out

def test_abort_clears_line_on_tty():
    b, s = _bar(True)
    b.abort()
    assert "\033[K" in s.getvalue()

def test_quip_shimmer_uses_256color_and_cycles():
    b, s = _bar(True)
    out = b._shimmer("hi", elapsed=0.0)
    assert "\033[38;5;" in out                       # 256-colour escape
    # Different times pick different colours (it shimmers).
    early = b._shimmer("hi", 0.0)
    later = b._shimmer("hi", 2.0)
    assert early != later

def test_rainbow_code_wraps():
    n = len(_progress._RAINBOW)
    assert _progress._rainbow_code(0) == _progress._rainbow_code(n)


# ── poll_job + on_poll integration ───────────────────────────────────────────

class _Resp:
    def __init__(self, status=None, session_id=None, text=""):
        self._status = status
        self._session_id = session_id
        self.text = text
        self.ok = True

    def raise_for_status(self):
        pass

    def json(self):
        return {"status": self._status, "session_id": self._session_id}


def test_poll_job_invokes_on_poll_and_returns_on_complete(monkeypatch):
    monkeypatch.setattr(
        _client.requests, "get", lambda *a, **k: _Resp("complete", session_id="s")
    )
    seen = []

    def on_poll(status):
        seen.append(status)

    job = _client.poll_job("j", "k", "u", on_poll=on_poll)
    assert seen == ["complete"]
    assert job == {"status": "complete", "session_id": "s"}

def test_poll_job_raises_with_logs_on_failure(monkeypatch):
    responses = iter([_Resp(status="failed"), _Resp(text="synth blew up")])
    monkeypatch.setattr(_client.requests, "get", lambda *a, **k: next(responses))
    with pytest.raises(RuntimeError, match="synth blew up"):
        _client.poll_job("j", "k", "u")

def test_poll_job_still_works_without_callback(monkeypatch):
    monkeypatch.setattr(_client.requests, "get", lambda *a, **k: _Resp("complete"))
    # Should simply return the final job record when on_poll is omitted.
    assert _client.poll_job("j", "k", "u") == {"status": "complete", "session_id": None}


# ── poll deadline vs. the server's build ceiling ─────────────────────────────
#
# The client deadline has to outlast the server's, or we abandon builds that go
# on to succeed. That invariant lived only in a docstring and silently broke
# when the build image raised BUILD_TIMEOUT_SECONDS 1800 -> 5400 while the SDK
# stayed at 2400 s, discarding good 40-55 min builds. These pin it down.

# What the deployed build image (infra/fargate/Dockerfile in the cloud repo)
# actually allows: three pre-gateware stages at BUILD_SYNTH_TIMEOUT_SECONDS,
# then place-and-route at BUILD_TIMEOUT_SECONDS. Update alongside that image.
_SERVER_BUILD_CEILING = 3 * 1800 + 5400  # 10800 s = 3 h


def test_default_poll_timeout_outlasts_server_build_ceiling():
    assert _client.DEFAULT_POLL_TIMEOUT > _SERVER_BUILD_CEILING, (
        "poll deadline is shorter than the server's own build ceiling, so the "
        "client will abandon builds that later succeed"
    )


def test_poll_job_default_timeout_is_the_module_default(monkeypatch):
    monkeypatch.setattr(_client.requests, "get", lambda *a, **k: _Resp("building"))
    monkeypatch.setattr(_client.time, "sleep", lambda *_: None)
    # Freeze the clock past the deadline so we can read which one was used.
    clock = iter([0.0, 0.0, 0.0, _client.DEFAULT_POLL_TIMEOUT + 1])
    monkeypatch.setattr(_client.time, "monotonic", lambda: next(clock, 1e9))
    with pytest.raises(TimeoutError, match=str(int(_client.DEFAULT_POLL_TIMEOUT))):
        _client.poll_job("j", "k", "u", timeout=None)


def test_timeout_message_says_the_build_was_not_cancelled(monkeypatch):
    monkeypatch.setattr(_client.requests, "get", lambda *a, **k: _Resp("building"))
    monkeypatch.setattr(_client.time, "sleep", lambda *_: None)
    clock = iter([0.0, 0.0, 0.0, 99.0])
    monkeypatch.setattr(_client.time, "monotonic", lambda: next(clock, 1e9))
    with pytest.raises(TimeoutError, match="NOT been cancelled"):
        _client.poll_job("j", "k", "u", timeout=1.0)


def test_app_passes_its_poll_timeout_through(monkeypatch):
    import manhattan_reasoning_gym as mrg

    seen = {}

    def fake_poll(job_id, api_key, api_url, timeout=None, on_poll=None):
        seen["timeout"] = timeout
        return {"status": "complete", "session_id": "s"}

    monkeypatch.setattr(_client, "submit", lambda *a, **k: "job-1")
    monkeypatch.setattr(_client, "poll_job", fake_poll)

    app = mrg.App("t", design="d.py", api_key="k", poll_timeout=7200.0)
    app._program()
    assert seen["timeout"] == 7200.0
