"""Unit tests for `mrg run --local`: local bitstream build + reserve + upload.

Flash-time-identity images build first and reserve afterwards; older
build-time-identity images reserve a slot first and bake it in.

No network, Docker, or toolchain is used -- requests and subprocess are faked.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

import pytest

from manhattan_reasoning_gym import _app, _client, _local_bitstream
from manhattan_reasoning_gym._local_build import SandboxUnavailableError

BIT = b"fake-bitstream" * 10
PATCHED = b"patched-bitstream" * 10


def _placeholder(seed=1296191305):
    """A random 512-word pattern; random columns are distinct in practice."""
    rng = random.Random(seed)
    return [rng.getrandbits(32) for _ in range(_local_bitstream.IDENTITY_WORDS)]


PLACEHOLDER = _placeholder()

# The real, board-tested placeholder (mrg-sandbox
# firmware/src/cloud_fpga_firmware/identity_placeholder.hex).
REAL_PLACEHOLDER_TEXT = (
    Path(__file__).parent / "data" / "identity_placeholder.hex"
).read_text()
REAL_PLACEHOLDER = _local_bitstream.parse_identity_hex(REAL_PLACEHOLDER_TEXT)

# The tail of a real nextpnr run: a pre-route estimate, then the final report.
_FMAX = "Info: Max frequency for clock {:>40}: {} MHz (PASS at {} MHz)"
_ETH = "'$glbnet$eth_clocks_ref_clk$TRELLIS_IO_IN'"
LITEX_LOG = "\n".join([
    "Info: Device utilisation:",
    "Info: \t          TRELLIS_IO:      15/    365     4%",
    "Info: \t              DP16KD:     191/    208    91%",
    "Info: \t              ALU54B:       0/     78     0%",
    "Info: \t        TRELLIS_COMB:   12551/  83640    15%",
    "",
    _FMAX.format(_ETH, "130.38", "50.00"),
    _FMAX.format("'$glbnet$crg_clkout1'", "60.98", "59.00"),
    "Info: Routing..",
    _FMAX.format(_ETH, "120.96", "50.00"),
    _FMAX.format("'$glbnet$crg_clkout1'", "85.35", "59.00"),
    _FMAX.format("'$glbnet$crg_clkout0'", "124.46", "50.00"),
    "Info: Program finished normally.",
]) + "\n"
TIMELINE = "\n".join(
    json.dumps({"stage": st, "event": ev, "at": at, "detail": None})
    for st, ev, at in [
        ("export_design", "started", 100.0), ("export_design", "finished", 104.5),
        ("build_gateware", "started", 105.0),
        ("build_gateware/yosys", "started", 106.0),
        ("build_gateware/yosys", "finished", 206.0),
        ("build_gateware", "finished", 425.0),
    ]
) + "\n"


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
        "headers": {"X-API-Key": "key", "X-MRG-Client-Features": "sessions"},
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


def _fake_docker(
    monkeypatch, *, run_rc=0, write_bit=True, digest="sha256:img", litex_log=None
):
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
            if litex_log is not None:
                (Path(out) / "litex.log").write_text(litex_log)
                (Path(out) / "stage_timeline.jsonl").write_text(TIMELINE)
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


# --- build summary --------------------------------------------------------------


def test_summary_uses_the_final_reports_and_skips_unused_resources():
    text = _local_bitstream.summarize_build(LITEX_LOG, TIMELINE)
    assert text.splitlines() == [
        "Build summary:",
        "  stages: export_design 4s, build_gateware 5m20s (total 5m24s)",
        "  tools: yosys 1m40s",
        "  timing:",
        "    eth_clocks_ref_clk: 120.96 MHz (PASS at 50.00 MHz)",
        "    crg_clkout1: 85.35 MHz (PASS at 59.00 MHz)",
        "    crg_clkout0: 124.46 MHz (PASS at 50.00 MHz)",
        "  utilisation:",
        "    TRELLIS_IO: 15/365 (4%)",
        "    DP16KD: 191/208 (91%)",
        "    TRELLIS_COMB: 12551/83640 (15%)",
    ]


def test_summary_is_empty_without_reports():
    assert _local_bitstream.summarize_build("Info: nothing here\n", TIMELINE) == ""


def test_successful_build_logs_and_returns_the_summary(monkeypatch, tmp_path):
    design = tmp_path / "design.py"
    design.write_text("# design\n")
    _fake_docker(monkeypatch, litex_log=LITEX_LOG)
    built = _local_bitstream.build_bitstream(design, out_dir=tmp_path / "out")
    assert "DP16KD: 191/208 (91%)" in built.summary
    log = (tmp_path / "out" / "build.log").read_text()
    assert "ok" in log and built.summary in log


def test_failed_build_log_keeps_the_stage_log(monkeypatch, tmp_path):
    design = tmp_path / "design.py"
    design.write_text("# design\n")
    _fake_docker(monkeypatch, run_rc=1, write_bit=False, litex_log=LITEX_LOG)
    with pytest.raises(_local_bitstream.LocalBuildError):
        _local_bitstream.build_bitstream(design, out_dir=tmp_path / "o")
    log = (tmp_path / "o" / "build.log").read_text()
    assert "stage log" in log and "Build summary" in log


# --- flash-time identity ----------------------------------------------------------


def test_build_without_slot_bakes_none_and_omits_it_from_the_manifest(
    monkeypatch, tmp_path
):
    design = tmp_path / "design.py"
    design.write_text("# design\n")
    calls = _fake_docker(monkeypatch)
    built = _local_bitstream.build_bitstream(
        design, out_dir=tmp_path / "out", image="sha256:pinned"
    )
    assert "slot_id" not in built.manifest
    assert built.manifest["image_digest"] == "sha256:pinned"
    run = next(c for c in calls if c[:2] == ["docker", "run"])
    assert run[-5:] == ["/design/design.py", "", "50000000", "", ""]
    # A digest handed in is used as is: no second `docker image inspect`.
    assert not any(c[:3] == ["docker", "image", "inspect"] for c in calls)


def test_identity_words_match_the_documented_example():
    words = _local_bitstream.identity_words(
        0, "192.168.1.101", "02:00:00:00:00:01", PLACEHOLDER
    )
    assert [f"{w:08x}" for w in words[:7]] == [
        "4d524749", "00000001", "00000000", "c0a80165",
        "00000200", "00000001", "7205bbd3",
    ]
    assert len(words) == 512
    # Padding is the placeholder inverted, so its columns stay distinct.
    assert words[7:] == [~w & 0xFFFFFFFF for w in PLACEHOLDER[7:]]
    _local_bitstream.parse_identity_hex("".join(f"{w:08x}\n" for w in words))


def test_identity_words_for_another_slot():
    words = _local_bitstream.identity_words(
        3, "192.168.1.104", "02:00:00:00:00:04", PLACEHOLDER
    )
    assert words[2:6] == [3, 0xC0A80168, 0x0200, 0x00000004]
    check = 0
    for w in words[:6]:
        check ^= w
    assert words[6] == ~check & 0xFFFFFFFF


@pytest.mark.parametrize("ip, mac", [
    ("192.168.1", "02:00:00:00:00:01"),
    ("192.168.1.101", "02:00:00:00:01"),
])
def test_identity_words_rejects_malformed_addresses(ip, mac):
    with pytest.raises(ValueError):
        _local_bitstream.identity_words(0, ip, mac, PLACEHOLDER)


def test_parse_identity_hex_validates_shape_and_columns():
    good = "".join(f"{w:08x}\n" for w in PLACEHOLDER)
    assert _local_bitstream.parse_identity_hex(good) == PLACEHOLDER
    with pytest.raises(ValueError, match="511 words"):
        _local_bitstream.parse_identity_hex("".join(good.splitlines(True)[:-1]))
    with pytest.raises(ValueError, match="not 32-bit hex"):
        _local_bitstream.parse_identity_hex(good.replace(good[:8], "zz", 1))
    with pytest.raises(ValueError, match="distinct"):
        _local_bitstream.parse_identity_hex("00000000\n" * 512)


def _fake_probe(monkeypatch, rc, stdout=""):
    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, rc, stdout, "probe err")

    monkeypatch.setattr(_local_bitstream.subprocess, "run", fake_run)
    return calls


def test_probe_reads_the_placeholder_from_the_image(monkeypatch):
    calls = _fake_probe(monkeypatch, 0, REAL_PLACEHOLDER_TEXT)
    assert _local_bitstream.identity_placeholder("sha256:img") == REAL_PLACEHOLDER
    (cmd,) = calls
    assert "--network=none" in cmd and "sha256:img" in cmd
    assert "identity_placeholder.hex" in cmd[-1]


def test_probe_refuses_a_placeholder_that_is_not_the_pinned_one(monkeypatch):
    # Valid shape and distinct columns, but not the placeholder the board
    # host pins: its bitstreams would be refused at flash time.
    _fake_probe(monkeypatch, 0, "".join(f"{w:08x}\n" for w in PLACEHOLDER))
    with pytest.raises(_local_bitstream.LocalBuildError, match="not the pinned"):
        _local_bitstream.identity_placeholder("sha256:img")


def test_pinned_placeholder_matches_the_firmware_copy():
    assert (
        hashlib.sha256(REAL_PLACEHOLDER_TEXT.encode()).hexdigest()
        == _local_bitstream.PLACEHOLDER_SHA256
    )
    assert REAL_PLACEHOLDER[0] == 0x75064692


def test_identity_image_for_board_0_matches_the_board_tested_one():
    # identity_0.hex as flashed and run on a real board (prototype), byte
    # for byte: header as the firmware's identity.h decodes it, then the
    # NOT-placeholder padding.
    words = _local_bitstream.identity_words(
        0, "192.168.1.101", "02:00:00:00:00:01", REAL_PLACEHOLDER
    )
    assert words[:7] == [
        0x4D524749, 0x00000001, 0x00000000, 0xC0A80165,
        0x00000200, 0x00000001, 0x7205BBD3,
    ]
    text = "".join(f"{w:08x}\n" for w in words)
    assert hashlib.sha256(text.encode()).hexdigest() == (
        "6aa82af676931c93f666836e8768c06bc908304a2ee43d902d60df8812fe4a35"
    )
    # Its columns are distinct too, so ecpbram can find it again (verify).
    _local_bitstream.parse_identity_hex(text)


def test_probe_returns_none_for_a_build_time_identity_image(monkeypatch):
    _fake_probe(monkeypatch, 3)
    assert _local_bitstream.identity_placeholder("sha256:img") is None


def test_probe_failure_is_reported(monkeypatch):
    _fake_probe(monkeypatch, 125)
    with pytest.raises(SandboxUnavailableError, match="probe err"):
        _local_bitstream.identity_placeholder("sha256:img")


def _built(tmp_path):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    bit = out / "design.bit"
    bit.write_bytes(BIT)
    manifest = {"target": "ecp5", "bitstream_sha256": "x", "image_digest": "d"}
    return _local_bitstream.LocalBitstream(bit, manifest, "summary")


def test_patch_identity_writes_slot_identity_and_manifest(monkeypatch, tmp_path):
    built = _built(tmp_path)
    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        out = Path(next(p.split(":")[0] for p in cmd if p.endswith(":/output")))
        (out / cmd[-1]).write_bytes(PATCHED)
        return subprocess.CompletedProcess(cmd, 0, "identity patched", "")

    monkeypatch.setattr(_local_bitstream.subprocess, "run", fake_run)
    patched = _local_bitstream.patch_identity(
        built, image="sha256:img", placeholder=PLACEHOLDER,
        slot_id=3, ip="192.168.1.104", mac="02:00:00:00:00:04",
    )
    (cmd,) = calls
    assert cmd[-4:] == [
        "design.bit", "identity_placeholder.hex", "identity_3.hex",
        "design.slot3.bit",
    ]
    assert "--network=none" in cmd and "sha256:img" in cmd
    out = tmp_path / "out"
    assert built.path.read_bytes() == BIT  # pristine build kept
    assert patched.path == out / "design.slot3.bit"
    assert patched.manifest["slot_id"] == 3
    assert patched.manifest["bitstream_sha256"] == hashlib.sha256(PATCHED).hexdigest()
    assert json.loads((out / "manifest.slot3.json").read_text()) == patched.manifest
    assert not (out / "manifest.json").exists()  # design.bit's, left alone
    ident = _local_bitstream.parse_identity_hex((out / "identity_3.hex").read_text())
    assert ident[3] == 0xC0A80168
    on_disk = (out / "identity_placeholder.hex").read_text()
    assert _local_bitstream.parse_identity_hex(on_disk) == PLACEHOLDER
    assert "identity patched" in (out / "build.log").read_text()


def test_patch_identity_refuses_when_placeholder_is_missing(monkeypatch, tmp_path):
    built = _built(tmp_path)
    monkeypatch.setattr(
        _local_bitstream.subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 2, "", "placeholder identity not found in the bitstream"
        ),
    )
    with pytest.raises(_local_bitstream.IdentityPatchError, match="not found"):
        _local_bitstream.patch_identity(
            built, image="i", placeholder=PLACEHOLDER,
            slot_id=0, ip="192.168.1.101", mac="02:00:00:00:00:01",
        )


def test_patch_script_checks_presence_by_diff_not_exit_code():
    script = _local_bitstream._PATCH_SCRIPT
    assert "--bootaddr 0 --compress" in script
    assert '-gt 256' in script and '-eq 0' in script
    assert "cmp -s" in script


def test_reserve_advertises_flash_identity(monkeypatch):
    seen = {}

    def fake_post(url, headers):
        seen.update(headers=headers)
        return _Resp(201, {"job_id": "j1"})

    monkeypatch.setattr(_client.requests, "post", fake_post)
    _client.reserve_local_build("key", "https://api.example", flash_identity=True)
    assert seen["headers"]["X-MRG-Client-Features"] == "sessions,flash-identity"


# --- App._program_local ---------------------------------------------------------


@pytest.fixture
def local_app(tmp_path):
    design = tmp_path / "design.py"
    design.write_text("# design\n")
    app = _app.App("t", design=str(design), api_key="key", local=True)
    yield app
    _app._registry.remove(app)


OLD_RESERVATION = {
    "job_id": "job-0123456789", "slot_id": 3, "ip": "192.168.1.104",
    "mac": "02:00:00:00:00:04", "expires_at": "2026-12-02T15:00:00Z",
}
NEW_RESERVATION = {"job_id": "job-0123456789", "slot_id": None}


def _wire(
    monkeypatch, tmp_path, events, *, placeholder=None, reservation=None,
    reserve_excs=(), upload_exc=None, build_exc=None, patch_exc=None,
):
    reservation = OLD_RESERVATION if reservation is None else reservation
    reserve_excs = list(reserve_excs)

    def reserve(api_key, api_url, flash_identity=False):
        events.append(("reserve", flash_identity))
        if reserve_excs:
            raise reserve_excs.pop(0)
        return reservation

    def build(design, *, slot_id, out_dir, image, **kw):
        events.append(("build", slot_id, image))
        if build_exc:
            raise build_exc
        bit = tmp_path / "design.bit"
        bit.write_bytes(BIT)
        manifest = {} if slot_id is None else {"slot_id": slot_id}
        return _local_bitstream.LocalBitstream(bit, manifest, "Build summary: x")

    def patch(built, *, image, placeholder, slot_id, ip, mac):
        events.append(("patch", slot_id, ip, mac, image))
        if patch_exc:
            raise patch_exc
        return _local_bitstream.LocalBitstream(
            tmp_path / "design.slot.bit", {"slot_id": slot_id}
        )

    def upload(job_id, path, manifest, api_key, api_url):
        events.append(("upload", job_id, manifest.get("slot_id"), Path(path).name))
        if upload_exc:
            raise upload_exc
        return {"status": "queued"}

    monkeypatch.setattr(_local_bitstream, "resolve_image", lambda: "sha256:img")
    monkeypatch.setattr(
        _local_bitstream, "identity_placeholder", lambda image: placeholder
    )
    monkeypatch.setattr(_client, "reserve_local_build", reserve)
    monkeypatch.setattr(_local_bitstream, "build_bitstream", build)
    monkeypatch.setattr(_local_bitstream, "patch_identity", patch)
    monkeypatch.setattr(_client, "upload_bitstream", upload)
    monkeypatch.setattr(
        _client, "poll_job",
        lambda job_id, *a, **k: events.append("poll") or {"session_id": "sess-2"},
    )
    monkeypatch.setattr(
        _client, "cancel_job", lambda job_id, *a: events.append(("cancel", job_id))
    )
    monkeypatch.setattr(_app.time, "sleep", lambda s: events.append(("sleep", s)))


# Build-time-identity image (old flow).


def test_program_local_reserves_builds_uploads_then_waits(
    monkeypatch, tmp_path, local_app, capsys
):
    events = []
    _wire(monkeypatch, tmp_path, events)
    local_app._program()
    assert events == [
        ("reserve", False), ("build", 3, "sha256:img"),
        ("upload", "job-0123456789", 3, "design.bit"), "poll",
    ]
    assert local_app.session_id == "sess-2" and local_app._programmed
    assert "Build summary: x" in capsys.readouterr().out


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


def test_old_image_against_flash_time_server_says_to_rebuild_the_image(
    monkeypatch, tmp_path, local_app
):
    events = []
    _wire(monkeypatch, tmp_path, events, reservation=NEW_RESERVATION)
    with pytest.raises(_local_bitstream.LocalBuildError, match="Rebuild the image"):
        local_app._program()
    assert events == [("reserve", False), ("cancel", "job-0123456789")]


# Flash-time-identity image (build first).


def test_build_first_then_upload_to_a_flash_time_server(
    monkeypatch, tmp_path, local_app, capsys
):
    events = []
    _wire(monkeypatch, tmp_path, events, placeholder=PLACEHOLDER,
          reservation=NEW_RESERVATION)
    local_app._program()
    assert events == [
        ("build", None, "sha256:img"), ("reserve", True),
        ("upload", "job-0123456789", None, "design.bit"), "poll",
    ]
    assert local_app.session_id == "sess-2"
    out = capsys.readouterr().out
    assert "no board reserved until the build finishes" in out
    assert "Build summary: x" in out


def test_build_first_against_a_slot_server_patches_the_slot_identity(
    monkeypatch, tmp_path, local_app
):
    events = []
    _wire(monkeypatch, tmp_path, events, placeholder=PLACEHOLDER)
    local_app._program()
    assert events == [
        ("build", None, "sha256:img"), ("reserve", True),
        ("patch", 3, "192.168.1.104", "02:00:00:00:00:04", "sha256:img"),
        ("upload", "job-0123456789", 3, "design.slot.bit"), "poll",
    ]


def test_build_first_failure_reserves_nothing(monkeypatch, tmp_path, local_app):
    events = []
    _wire(monkeypatch, tmp_path, events, placeholder=PLACEHOLDER,
          build_exc=_local_bitstream.LocalBuildError("boom"))
    with pytest.raises(_local_bitstream.LocalBuildError):
        local_app._program()
    assert events == [("build", None, "sha256:img")]


@pytest.mark.parametrize("where", ["patch", "upload"])
def test_build_first_cancels_the_job_on_a_later_failure(
    monkeypatch, tmp_path, local_app, where
):
    events = []
    exc = (
        _local_bitstream.IdentityPatchError("refused") if where == "patch"
        else _client.LocalBuildAPIError(422, "sha256_mismatch", "bad")
    )
    _wire(monkeypatch, tmp_path, events, placeholder=PLACEHOLDER,
          **{f"{where}_exc": exc})
    with pytest.raises(type(exc)):
        local_app._program()
    assert events[-1] == ("cancel", "job-0123456789")
    assert "poll" not in events


def test_build_first_waits_out_a_full_server(monkeypatch, tmp_path, local_app):
    events = []
    full = _client.LocalBuildAPIError(503, "no_build_capacity", "full")
    _wire(monkeypatch, tmp_path, events, placeholder=PLACEHOLDER,
          reservation=NEW_RESERVATION, reserve_excs=[full, full])
    local_app._program()
    assert events[:6] == [
        ("build", None, "sha256:img"),
        ("reserve", True), ("sleep", _app._RESERVE_RETRY_EVERY),
        ("reserve", True), ("sleep", _app._RESERVE_RETRY_EVERY),
        ("reserve", True),
    ]
    assert events[-1] == "poll"


def test_build_first_waits_for_its_own_other_build(monkeypatch, tmp_path, local_app):
    # Builds hold nothing on the server, so two local runs can build at once;
    # the second must not throw its bitstream away on 402 build_in_flight.
    events = []
    busy = _client.LocalBuildAPIError(402, "build_in_flight", "busy")
    _wire(monkeypatch, tmp_path, events, placeholder=PLACEHOLDER,
          reservation=NEW_RESERVATION, reserve_excs=[busy])
    local_app._program()
    assert events[:4] == [
        ("build", None, "sha256:img"),
        ("reserve", True), ("sleep", _app._RESERVE_RETRY_EVERY),
        ("reserve", True),
    ]
    assert events[-1] == "poll"


def test_build_first_gives_up_after_the_retry_window(
    monkeypatch, tmp_path, local_app
):
    events = []
    full = _client.LocalBuildAPIError(503, "no_build_capacity", "full")
    _wire(monkeypatch, tmp_path, events, placeholder=PLACEHOLDER,
          reserve_excs=[full] * 3)
    monkeypatch.setattr(_app, "_RESERVE_RETRY_FOR", 0.0)
    with pytest.raises(_client.LocalBuildAPIError) as info:
        local_app._program()
    assert info.value.code == "no_build_capacity"
    assert str(tmp_path / "design.bit") in str(info.value)
    assert events == [("build", None, "sha256:img"), ("reserve", True)]


def test_build_first_does_not_retry_other_reserve_errors(
    monkeypatch, tmp_path, local_app
):
    events = []
    spent = _client.LocalBuildAPIError(402, "quota_exhausted", "spent")
    _wire(monkeypatch, tmp_path, events, placeholder=PLACEHOLDER, reserve_excs=[spent])
    with pytest.raises(_client.LocalBuildAPIError, match="spent") as info:
        local_app._program()
    assert "kept at" in str(info.value)
    assert events == [("build", None, "sha256:img"), ("reserve", True)]


def test_local_rejects_multi_file_projects(local_app):
    local_app.sources = ["rtl/a.v"]
    with pytest.raises(ValueError, match="single design file"):
        local_app._program()
