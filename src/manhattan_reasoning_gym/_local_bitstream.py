"""Build a full, flashable ECP5 bitstream on this machine (``mrg run --local``).

Unlike ``synth``/``pnr`` (reports only), this produces the real ``.bit`` the
Cloud flashes: the same four-stage pipeline the build server runs (design
export, SoC headers, firmware, gateware).

A board's network identity (MAC/IP) gets into the bitstream one of two ways,
depending on the build image:

* **Flash-time identity** (newer images). The SoC carries a small identity
  ROM initialised with a fixed placeholder pattern, so every build is the same
  for every board and needs no slot. Whoever flashes the board swaps the
  placeholder for that board's identity (``ecpbram``): a few seconds of work
  on the finished bitstream, not a rebuild. The image ships the placeholder as
  ``cloud_fpga_firmware/identity_placeholder.hex``.
* **Build-time identity** (older images). The reserved build slot is compiled
  into the firmware as ``FPGA_ID``, so the slot must be known before building.

``identity_placeholder`` tells the two apart. With a flash-time image the SDK
builds first and reserves afterwards; if the server still hands out a build
slot (it predates flash-time identity), ``patch_identity`` writes that slot's
identity into the finished bitstream here -- the same swap the server's board
host makes once it assigns identity at flash time.

It runs in the build image, never on the host toolchain: the image pins the
toolchain the Cloud itself uses, and the manifest records that image's digest
so any run can be traced back to the exact build environment. The image is
``$MRG_LOCAL_BUILD_IMAGE`` (default ``mrg-local-build:latest``, built from the
Cloud repo's ``infra/fargate/Dockerfile``).
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ._local_build import SandboxUnavailableError

DEFAULT_LOCAL_BUILD_IMAGE = "mrg-local-build:latest"
DEFAULT_SYS_CLK_FREQ = 50_000_000

#: Size of the identity ROM: one DP16KD block in 512 x 36 mode.
IDENTITY_WORDS = 512
#: Identity image header (see ``identity_words``).
IDENTITY_MAGIC = 0x4D524749  # "MRGI"
IDENTITY_VERSION = 1
_IDENTITY_HEADER_WORDS = 7

# Where a flash-time-identity image keeps the placeholder. The SoC package is
# installed into the LiteX venv; the source tree is the fallback.
_PLACEHOLDER_GLOBS = (
    "/opt/venv/lib/python3*/site-packages/cloud_fpga_firmware/identity_placeholder.hex",
    "/opt/app/firmware/src/cloud_fpga_firmware/identity_placeholder.hex",
)
_PROBE_SCRIPT = (
    "for f in " + " ".join(_PLACEHOLDER_GLOBS) + "; do "
    '[ -f "$f" ] && { cat "$f"; exit 0; }; done; exit 3'
)
_PROBE_ABSENT = 3

# Runs inside the image. Calls the build server's own pipeline; prints the
# failing stage and its log on error so the user sees why a build broke.
# Passes the slot only to a pipeline that still takes one (an empty slot means
# "none": a flash-time-identity build ignores it). Copies the PnR log and the
# stage timeline out either way so the SDK can summarise or explain the build.
_BUILD_SCRIPT = r"""
import inspect, shutil, sys
from pathlib import Path
from tempfile import TemporaryDirectory
from cloud_fpga_orchestrator.compiler import BuildError, run_pipeline

design, slot, clk, target, top = sys.argv[1:6]
kwargs = dict(
    sys_clk_freq=int(clk),
    timing_target_mhz=float(target) if target else None,
    top=top or None,
)
if "fpga_id" in inspect.signature(run_pipeline).parameters:
    kwargs["fpga_id"] = int(slot or 0)

def keep_logs(scratch):
    for name in ("litex.log", "stage_timeline.jsonl"):
        src = Path(scratch) / name
        if src.exists():
            shutil.copyfile(src, Path("/output") / name)

with TemporaryDirectory(prefix="mrg-local-") as scratch:
    try:
        result = run_pipeline(Path(design), Path(scratch), **kwargs)
    except BuildError as exc:
        keep_logs(scratch)
        print(f"Build failed at {exc.stage}: {exc}", file=sys.stderr)
        print(exc.log, file=sys.stderr)
        sys.exit(1)
    keep_logs(scratch)
    shutil.copyfile(result.bitstream_path, "/output/design.bit")
    print(result.logs.get("timeline", ""))
"""

# Runs inside the image: swap the placeholder identity for a board's identity.
# ecpbram exits 0 and changes nothing when the --from pattern is absent, so
# presence is checked by diffing the configs, never by its exit code. One
# DP16KD .bram_init block is 256 config lines: a correct patch touches at most
# that one block. Exit 2 = refused (placeholder absent or ambiguous, or the
# patched bitstream failed verification).
_PATCH_SCRIPT = r"""
set -euo pipefail
cd /output
IN=$1; PH=$2; ID=$3; OUT=$4
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
changed() { diff "$1" "$2" | grep -c '^>' || true; }
ecpunpack "$IN" "$T/in.config" >/dev/null 2>&1
ecpbram -i "$T/in.config" -o "$T/out.config" -f "$PH" -t "$ID"
n=$(changed "$T/in.config" "$T/out.config")
refuse() { echo "$1" >&2; exit 2; }
[ "$n" -eq 0 ] && refuse "placeholder identity not found in the bitstream"
[ "$n" -gt 256 ] && refuse "placeholder matched more than one BRAM ($n lines)"
ecppack --bootaddr 0 --compress "$T/out.config" "$OUT"
ecpunpack "$OUT" "$T/re.config" >/dev/null 2>&1
cmp -s "$T/out.config" "$T/re.config" || refuse "repacked bitstream does not round-trip"
ecpbram -i "$T/re.config" -o "$T/v1.config" -f "$ID" -t "$PH"
ecpbram -i "$T/re.config" -o "$T/v2.config" -f "$PH" -t "$ID"
v1=$(changed "$T/re.config" "$T/v1.config")
v2=$(changed "$T/re.config" "$T/v2.config")
[ "$v1" -gt 0 ] && [ "$v2" -eq 0 ] || refuse "verify failed (id=$v1 placeholder=$v2)"
echo "identity patched: $n config lines; verified"
"""


class LocalBuildError(RuntimeError):
    """The local bitstream build ran but did not produce a bitstream."""


class IdentityPatchError(RuntimeError):
    """Writing a board identity into a flash-time-identity bitstream failed."""


@dataclass
class LocalBitstream:
    path: Path
    manifest: dict
    #: Human-readable timing/utilisation summary ("" if the log had none).
    summary: str = field(default="")


def _image() -> str:
    return os.environ.get("MRG_LOCAL_BUILD_IMAGE", DEFAULT_LOCAL_BUILD_IMAGE)


def _require_docker() -> None:
    if shutil.which("docker") is None:
        raise SandboxUnavailableError(
            "a local bitstream build needs Docker to run the pinned build image."
        )


def _image_digest(image: str) -> str:
    proc = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True, text=True,
    )
    digest = proc.stdout.strip()
    if proc.returncode != 0 or not digest:
        raise SandboxUnavailableError(
            f"local build image {image!r} not found. Build it from the Cloud repo "
            f"(docker build --platform linux/amd64 -f infra/fargate/Dockerfile "
            f"-t {DEFAULT_LOCAL_BUILD_IMAGE} .) or set $MRG_LOCAL_BUILD_IMAGE."
        )
    return digest


def resolve_image() -> str:
    """Digest of the local build image (``$MRG_LOCAL_BUILD_IMAGE``).

    Every container in one ``mrg run --local`` runs this digest, so a re-tag
    mid-run can't mix two toolchains.
    """
    _require_docker()
    return _image_digest(_image())


def _source_revision(design_dir: Path) -> str | None:
    """Git commit of the user's design, if it lives in a git checkout."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(design_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
    except OSError:
        return None
    rev = proc.stdout.strip()
    return rev if proc.returncode == 0 and rev else None


# -- identity --------------------------------------------------------------------


def parse_identity_hex(text: str) -> list[int]:
    """Parse a 512-line, 32-bit-per-line identity hex file.

    Raises ``ValueError`` unless it is exactly ``IDENTITY_WORDS`` words of at
    most 8 hex digits and every bit-column is distinct -- the property that
    lets ``ecpbram`` find the pattern in a bitstream unambiguously.
    """
    tokens = text.split()
    if len(tokens) != IDENTITY_WORDS:
        raise ValueError(
            f"identity placeholder has {len(tokens)} words, expected {IDENTITY_WORDS}"
        )
    words = []
    for tok in tokens:
        if not re.fullmatch(r"[0-9a-fA-F]{1,8}", tok):
            raise ValueError(f"identity placeholder word {tok!r} is not 32-bit hex")
        words.append(int(tok, 16))
    columns = {tuple((w >> bit) & 1 for w in words) for bit in range(32)}
    if len(columns) != 32:
        raise ValueError("identity placeholder bit-columns are not all distinct")
    return words


def identity_placeholder(image: str) -> list[int] | None:
    """The image's identity placeholder, or None for a build-time-identity image.

    ``image`` is a digest from ``resolve_image``. A flash-time-identity image
    ships the placeholder its SoC bakes into every bitstream; an older image
    has none, and bakes the build slot into the firmware instead.
    """
    proc = subprocess.run(
        [
            "docker", "run", "--rm", "--platform", "linux/amd64", "--network=none",
            "--entrypoint", "/bin/sh", image, "-c", _PROBE_SCRIPT,
        ],
        capture_output=True, text=True,
    )
    if proc.returncode == _PROBE_ABSENT:
        return None
    if proc.returncode != 0:
        raise SandboxUnavailableError(
            f"could not inspect the local build image (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-300:]}"
        )
    return parse_identity_hex(proc.stdout)


def identity_words(
    board_id: int, ip: str, mac: str, placeholder: list[int]
) -> list[int]:
    """The identity ROM image for one board (``identity_<board>.hex``).

    Layout (512 x 32-bit words): magic "MRGI", version 1, board id, IPv4
    (big-endian), MAC bytes 0..1 (low 16 bits), MAC bytes 2..5 (big-endian),
    a check word ``~(w0 ^ ... ^ w5)``; words 7..511 are the bitwise NOT of
    the placeholder's, so the image's bit-columns are distinct too (``ecpbram``
    can't locate an image padded with zeros) and no half of it equals the
    placeholder. The firmware reads words 0..6 at boot and stays off the
    network unless magic, version and check word are right.
    """
    if len(placeholder) != IDENTITY_WORDS:
        raise ValueError(f"placeholder must have {IDENTITY_WORDS} words")
    if not 0 <= board_id <= 0xFFFFFFFF:
        raise ValueError(f"board id {board_id} out of range")
    ip_word = int(ipaddress.IPv4Address(ip))
    parts = mac.split(":")
    if len(parts) != 6:
        raise ValueError(f"MAC {mac!r} is not 6 bytes")
    mac_bytes = bytes(int(part, 16) for part in parts)
    header = [
        IDENTITY_MAGIC,
        IDENTITY_VERSION,
        board_id,
        ip_word,
        int.from_bytes(mac_bytes[:2], "big"),
        int.from_bytes(mac_bytes[2:], "big"),
    ]
    check = 0
    for word in header:
        check ^= word
    header.append(~check & 0xFFFFFFFF)
    return header + [~w & 0xFFFFFFFF for w in placeholder[_IDENTITY_HEADER_WORDS:]]


def _hex_lines(words: list[int]) -> str:
    return "".join(f"{w:08x}\n" for w in words)


def patch_identity(
    built: LocalBitstream,
    *,
    image: str,
    placeholder: list[int],
    slot_id: int,
    ip: str,
    mac: str,
) -> LocalBitstream:
    """Write a build slot's identity into a flash-time-identity bitstream.

    For a server that still reserves a build slot: the same swap its board
    host makes once identity is assigned at flash time. Leaves the pristine
    ``design.bit`` in place and writes ``design.slot<N>.bit`` beside it;
    returns that, with a manifest for the slot. Refuses (``IdentityPatchError``)
    unless the placeholder is in the bitstream exactly once.
    """
    out_dir = built.path.parent
    (out_dir / "identity_placeholder.hex").write_text(_hex_lines(placeholder))
    ident = identity_words(slot_id, ip, mac, placeholder)
    ident_name = f"identity_{slot_id}.hex"
    (out_dir / ident_name).write_text(_hex_lines(ident))
    patched = out_dir / f"design.slot{slot_id}.bit"
    proc = subprocess.run(
        [
            "docker", "run", "--rm", "--platform", "linux/amd64", "--network=none",
            "--entrypoint", "/bin/bash",
            "-v", f"{out_dir}:/output",
            image, "-c", _PATCH_SCRIPT, "patch",
            built.path.name, "identity_placeholder.hex", ident_name, patched.name,
        ],
        capture_output=True, text=True,
    )
    with open(out_dir / "build.log", "a") as log:
        log.write(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not patched.exists():
        raise IdentityPatchError(
            f"could not write build slot {slot_id}'s identity into the bitstream "
            f"(rc={proc.returncode}): {(proc.stderr or proc.stdout).strip()[-500:]}"
        )
    manifest = dict(built.manifest)
    manifest["slot_id"] = slot_id
    manifest["bitstream_sha256"] = hashlib.sha256(patched.read_bytes()).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return LocalBitstream(path=patched, manifest=manifest, summary=built.summary)


# -- build -----------------------------------------------------------------------


def build_bitstream(
    design: str | Path,
    *,
    slot_id: int | None = None,
    out_dir: str | Path,
    sys_clk_freq: int | None = None,
    timing_target_mhz: float | None = None,
    top: str | None = None,
    image: str | None = None,
) -> LocalBitstream:
    """Build ``design`` into ``out_dir``.

    ``slot_id`` is the build slot to bake in as ``FPGA_ID`` for a
    build-time-identity image; leave it None for a flash-time-identity image
    (the bitstream then carries the placeholder and fits any board).
    ``image`` is a digest from ``resolve_image`` (resolved here if omitted).

    Returns the ``.bit`` path, the upload manifest (target, slot_id when
    given, bitstream_sha256, image_digest, sys_clk_freq, source_revision,
    built_at) and a timing/utilisation summary. ``build.log`` in ``out_dir``
    holds the stage timeline and summary on success, the failing stage's log
    on failure.
    """
    _require_docker()
    design = Path(design).resolve()
    if not design.exists():
        raise FileNotFoundError(f"design not found: {design}")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = image or _image_digest(_image())
    clk = sys_clk_freq or DEFAULT_SYS_CLK_FREQ

    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64", "--network=none",
        "--entrypoint", "/opt/runner-venv/bin/python",
        "-v", f"{design.parent}:/design:ro",
        "-v", f"{out_dir}:/output",
        digest, "-c", _BUILD_SCRIPT,
        f"/design/{design.name}", "" if slot_id is None else str(slot_id),
        str(clk), str(timing_target_mhz or ""), top or "",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    bit = out_dir / "design.bit"
    summary = build_summary(out_dir)
    log = proc.stdout + proc.stderr
    if summary:
        log += ("\n" if log and not log.endswith("\n") else "") + summary + "\n"
    (out_dir / "build.log").write_text(log)
    if proc.returncode != 0 or not bit.exists():
        tail = (proc.stderr or proc.stdout).strip()[-2000:]
        raise LocalBuildError(
            f"local build failed (rc={proc.returncode}); full log in "
            f"{out_dir / 'build.log'}\n{tail}"
        )

    manifest: dict = {"target": "ecp5"}
    if slot_id is not None:
        manifest["slot_id"] = slot_id
    manifest.update(
        bitstream_sha256=hashlib.sha256(bit.read_bytes()).hexdigest(),
        image_digest=digest,
        sys_clk_freq=clk,
        built_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    if (rev := _source_revision(design.parent)) is not None:
        manifest["source_revision"] = rev
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return LocalBitstream(path=bit, manifest=manifest, summary=summary)


# -- build summary -----------------------------------------------------------------

_UTIL_RE = re.compile(r"^Info:\s+(\w+):\s+(\d+)/\s*(\d+)\s+(\d+)%")
_FMAX_RE = re.compile(
    r"^Info: Max frequency for clock\s+'([^']+)':\s+([\d.]+) MHz "
    r"\((\w+) at ([\d.]+) MHz\)"
)


def _clock_name(raw: str) -> str:
    return raw.replace("$glbnet$", "").replace("$TRELLIS_IO_IN", "")


def _last_block(lines: list[str], pattern: re.Pattern) -> list[re.Match]:
    """Matches of ``pattern`` in the last run of consecutive matching lines."""
    block: list[re.Match] = []
    current: list[re.Match] = []
    for line in lines:
        m = pattern.match(line)
        if m:
            current.append(m)
        elif current:
            block, current = current, []
    return current or block


def _fmt_seconds(s: float) -> str:
    return f"{s:.0f}s" if s < 90 else f"{int(s // 60)}m{int(s % 60):02d}s"


def _stage_times(timeline: str) -> list[tuple[str, float]]:
    """(stage, seconds) for each finished stage in a stage_timeline.jsonl."""
    started: dict[str, float] = {}
    out: list[tuple[str, float]] = []
    for line in timeline.splitlines():
        try:
            ev = json.loads(line)
            stage, event, at = ev["stage"], ev["event"], float(ev["at"])
        except (ValueError, KeyError, TypeError):
            continue
        if event == "started":
            started[stage] = at
        elif event == "finished" and stage in started:
            out.append((stage, at - started.pop(stage)))
    return out


def summarize_build(litex_log: str, timeline: str = "") -> str:
    """Timing and utilisation summary from a build's nextpnr log and timeline.

    Uses the *last* utilisation and Fmax reports in the log -- nextpnr prints
    a pre-route estimate first and the post-route result last. Lists only the
    resources the design uses. Returns "" when the log has neither report.
    """
    lines = litex_log.splitlines()
    fmax = _last_block(lines, _FMAX_RE)
    util = [m for m in _last_block(lines, _UTIL_RE) if int(m[2]) > 0]
    if not (fmax or util):
        return ""
    parts = ["Build summary:"]
    # Pipeline stages are top-level names; the tool shims time the tools
    # inside a stage as "<stage>/<tool>".
    times = _stage_times(timeline)
    stages = [(name, t) for name, t in times if "/" not in name]
    tools = [(name.split("/", 1)[1], t) for name, t in times if "/" in name]
    if stages:
        total = sum(t for _, t in stages)
        parts.append(
            "  stages: "
            + ", ".join(f"{name} {_fmt_seconds(t)}" for name, t in stages)
            + f" (total {_fmt_seconds(total)})"
        )
    if tools:
        parts.append(
            "  tools: " + ", ".join(f"{name} {_fmt_seconds(t)}" for name, t in tools)
        )
    if fmax:
        parts.append("  timing:")
        parts += [
            f"    {_clock_name(m[1])}: {m[2]} MHz ({m[3]} at {m[4]} MHz)"
            for m in fmax
        ]
    if util:
        parts.append("  utilisation:")
        parts += [f"    {m[1]}: {m[2]}/{m[3]} ({m[4]}%)" for m in util]
    return "\n".join(parts)


def build_summary(out_dir: Path) -> str:
    """``summarize_build`` over the logs a build copied into ``out_dir``."""
    log = out_dir / "litex.log"
    if not log.exists():
        return ""
    timeline = out_dir / "stage_timeline.jsonl"
    return summarize_build(
        log.read_text(errors="replace"),
        timeline.read_text(errors="replace") if timeline.exists() else "",
    )
