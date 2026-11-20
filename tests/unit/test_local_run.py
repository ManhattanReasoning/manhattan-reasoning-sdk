"""Unit tests for `mrg run --local`: reserve -> local bitstream build -> upload.

No network, Docker, or toolchain is used -- requests and subprocess are faked.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from manhattan_reasoning_gym import _app, _client, _local_bitstream
from manhattan_reasoning_gym._local_build import SandboxUnavailableError

BIT = b"fake-bitstream" * 10


class _Resp:
    def __init__(self, status=200, body=None, reason="OK"):
        self.status_code = status
        self._body = body or {}
        self.reason = reason

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._body


# --- _client -----------------------------------------------------------------


def test_reserve_local_build_posts_with_key(monkeypatch):
    seen = {}

    def fake_post(url, headers):
        seen.update(url=url, headers=headers)
        return _Resp(201, {"job_id": "j1", "slot_id": 3})

    monkeypatch.setattr(_client.requests, "post", fake_post)
    res = _client.reserve_local_build("key", "https://api.example")
    assert res["slot_id"] == 3
    assert seen == {
        "url": "https://api.example/builds/local",
        "headers": {"X-API-Key": "key"},
    }


def test_reserve_maps_server_error_codes(monkeypatch):
    body = {"detail": {"error": "build_in_flight", "message": "busy"}}
    monkeypatch.setattr(
        _client.requests, "post", lambda url, headers: _Resp(402, body)
    )
    with pytest.raises(_client.LocalBuildAPIError) as info:
        _client.reserve_local_build("key", "https://api.example")
    assert info.value.status == 402 and info.value.code == "build_in_flight"


def test_upload_bitstream_sends_file_and_manifest(monkeypatch, tmp_path):
    bit = tmp_path / "design.bit"
    bit.write_bytes(BIT)
    seen = {}

    def fake_put(url, headers, files, data):
        seen.update(url=url, name=files["file"][0], content=files["file"][1].read())
        seen["manifest"] = json.loads(data["manifest"])
        return _Resp(202, {"job_id": "j1", "fpga_id": None, "status": "queued"})

    monkeypatch.setattr(_client.requests, "put", fake_put)
    res = _client.upload_bitstream(
        "j1", bit, {"slot_id": 3}, "key", "https://api.example"
    )
    assert res["status"] == "queued"
    assert seen["url"] == "https://api.example/jobs/j1/bitstream"
    assert seen["name"] == "design.bit" and seen["content"] == BIT
    assert seen["manifest"] == {"slot_id": 3}


# --- _local_bitstream ----------------------------------------------------------


def _fake_docker(monkeypatch, *, run_rc=0, write_bit=True, digest="sha256:img"):
    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0 if digest else 1, digest, "")
        if cmd[:2] == ["git", "-C"]:
            return subprocess.CompletedProcess(cmd, 0, "a1b2c3d\n", "")
        if cmd[:2] == ["docker", "run"]:
            out = next(
                part.split(":")[0] for part in cmd if part.endswith(":/output")
            )
            if write_bit:
                (Path(out) / "design.bit").write_bytes(BIT)
            return subprocess.CompletedProcess(cmd, run_rc, "ok", "stage log")
        raise AssertionError(cmd)

    monkeypatch.setattr(_local_bitstream.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(_local_bitstream.subprocess, "run", fake_run)
    return calls


def test_build_bitstream_bakes_slot_and_writes_manifest(monkeypatch, tmp_path):
    design = tmp_path / "design.py"
    design.write_text("# design\n")
    calls = _fake_docker(monkeypatch)

    built = _local_bitstream.build_bitstream(
        design, slot_id=4, out_dir=tmp_path / "out", sys_clk_freq=60_000_000
    )

    assert built.path.read_bytes() == BIT
    assert built.manifest == {
        "target": "ecp5",
        "slot_id": 4,
        "bitstream_sha256": hashlib.sha256(BIT).hexdigest(),
        "image_digest": "sha256:img",
        "sys_clk_freq": 60_000_000,
        "built_at": built.manifest["built_at"],
        "source_revision": "a1b2c3d",
    }
    on_disk = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert on_disk == built.manifest
    run = next(c for c in calls if c[:2] == ["docker", "run"])
    assert "--network=none" in run and "sha256:img" in run
    assert run[-5:] == ["/design/design.py", "4", "60000000", "", ""]


def test_build_bitstream_defaults_to_50mhz(monkeypatch, tmp_path):
    design = tmp_path / "design.v"
    design.write_text("module m; endmodule\n")
    _fake_docker(monkeypatch)
    built = _local_bitstream.build_bitstream(design, slot_id=0, out_dir=tmp_path)
    assert built.manifest["sys_clk_freq"] == 50_000_000


def test_build_failure_raises_with_log(monkeypatch, tmp_path):
    design = tmp_path / "design.py"
    design.write_text("# design\n")
    _fake_docker(monkeypatch, run_rc=1, write_bit=False)
    with pytest.raises(_local_bitstream.LocalBuildError, match="stage log"):
        _local_bitstream.build_bitstream(design, slot_id=0, out_dir=tmp_path / "o")
    assert "stage log" in (tmp_path / "o" / "build.log").read_text()


def test_missing_image_explains_how_to_build_it(monkeypatch, tmp_path):
    design = tmp_path / "design.py"
    design.write_text("# design\n")
    _fake_docker(monkeypatch, digest="")
    with pytest.raises(SandboxUnavailableError, match="infra/fargate/Dockerfile"):
        _local_bitstream.build_bitstream(design, slot_id=0, out_dir=tmp_path)


# --- App._program_local ---------------------------------------------------------


@pytest.fixture
def local_app(tmp_path):
    design = tmp_path / "design.py"
    design.write_text("# design\n")
    app = _app.App("t", design=str(design), api_key="key", local=True)
    yield app
    _app._registry.remove(app)


def _wire(monkeypatch, tmp_path, events, *, upload_exc=None, build_exc=None):
    reservation = {
        "job_id": "job-0123456789", "slot_id": 3, "ip": "192.168.1.104",
        "mac": "02:00:00:00:00:04", "expires_at": "2026-12-02T15:00:00Z",
    }

    def reserve(api_key, api_url):
        events.append("reserve")
        return reservation

    def build(design, *, slot_id, out_dir, **kw):
        events.append(("build", slot_id))
        if build_exc:
            raise build_exc
        bit = tmp_path / "design.bit"
        bit.write_bytes(BIT)
        return _local_bitstream.LocalBitstream(bit, {"slot_id": slot_id})

    def upload(job_id, path, manifest, api_key, api_url):
        events.append(("upload", job_id, manifest["slot_id"]))
        if upload_exc:
            raise upload_exc
        return {"status": "queued"}

    monkeypatch.setattr(_client, "reserve_local_build", reserve)
    monkeypatch.setattr(_local_bitstream, "build_bitstream", build)
    monkeypatch.setattr(_client, "upload_bitstream", upload)
    monkeypatch.setattr(
        _client, "poll_job",
        lambda job_id, *a, **k: events.append("poll") or {"fpga_id": 2},
    )
    monkeypatch.setattr(
        _client, "cancel_job", lambda job_id, *a: events.append(("cancel", job_id))
    )


def test_program_local_reserves_builds_uploads_then_waits(
    monkeypatch, tmp_path, local_app
):
    events = []
    _wire(monkeypatch, tmp_path, events)
    local_app._program()
    assert events == [
        "reserve", ("build", 3), ("upload", "job-0123456789", 3), "poll",
    ]
    assert local_app.fpga_id == 2 and local_app._programmed


def test_failed_local_build_cancels_the_reservation(monkeypatch, tmp_path, local_app):
    events = []
    _wire(monkeypatch, tmp_path, events,
          build_exc=_local_bitstream.LocalBuildError("boom"))
    with pytest.raises(_local_bitstream.LocalBuildError):
        local_app._program()
    assert events[-1] == ("cancel", "job-0123456789")


def test_expired_reservation_tells_the_user_to_rerun(monkeypatch, tmp_path, local_app):
    events = []
    expired = _client.LocalBuildAPIError(410, "reservation_expired", "expired")
    _wire(monkeypatch, tmp_path, events, upload_exc=expired)
    with pytest.raises(_client.LocalBuildAPIError, match="run `mrg run --local`"):
        local_app._program()


def test_local_rejects_multi_file_projects(local_app):
    local_app.sources = ["rtl/a.v"]
    with pytest.raises(ValueError, match="single design file"):
        local_app._program()
