from unittest.mock import Mock

from manhattan_reasoning_gym._build_report import print_build_report


def test_old_server_is_silent(capsys):
    print_build_report(None)
    assert not capsys.readouterr().out


def test_timing_miss_warns_even_if_user_clock_is_fast(capsys):
    print_build_report(
        {
            "schema_version": 1,
            "sys_clk_freq_hz": 80_000_000,
            "fmax_mhz": 100,
            "timing_target_mhz": 90,
            "timing_met": False,
            "utilization": {"lut": {"used": 100, "available": 1000, "percent": 10}},
        }
    )
    text = capsys.readouterr().out
    assert "80 MHz" in text and "100 MHz" in text and "90 MHz" in text
    assert "WARNING: timing NOT MET" in text
    assert "100/1000 (10%)" in text


def test_unknown_is_not_reported_as_success(capsys):
    print_build_report({"schema_version": 1, "timing_met": None})
    assert "UNKNOWN" in capsys.readouterr().out


def test_app_retains_and_prints_report(monkeypatch, capsys):
    from manhattan_reasoning_gym import _app

    report = {"schema_version": 1, "timing_met": False}
    monkeypatch.setattr(_app._client, "submit", Mock(return_value="job"))
    monkeypatch.setattr(
        _app._client,
        "poll_job",
        Mock(
            return_value={
                "fpga_id": 2,
                "build_report": report,
            }
        ),
    )
    app = _app.App("test", design="design.py", api_key="key")
    app._program()
    assert app.build_report == report
    assert app._programmed
    assert "WARNING" in capsys.readouterr().out


def test_job_command_shows_report_and_preserves_json(monkeypatch, capsys):
    import argparse
    import json

    from manhattan_reasoning_gym import _cli

    report = {"schema_version": 1, "timing_met": False}
    job = {
        "job_id": "job", "fpga_id": 2, "type": "build_and_program",
        "status": "complete", "created_at": "2026-10-16T00:00:00+00:00",
        "updated_at": "2026-10-16T00:01:00+00:00", "build_report": report,
    }
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: ("key", "url"))
    monkeypatch.setattr(_cli._client, "get_job", Mock(return_value=job))
    args = argparse.Namespace(job_id="job", json=False)
    _cli.cmd_job(args)
    assert "WARNING" in capsys.readouterr().out
    args.json = True
    _cli.cmd_job(args)
    assert json.loads(capsys.readouterr().out) == job
