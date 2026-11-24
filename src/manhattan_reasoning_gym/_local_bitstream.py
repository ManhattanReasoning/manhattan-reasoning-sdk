"""Build a full, flashable ECP5 bitstream on this machine (``mrg run --local``).

Unlike ``synth``/``pnr`` (reports only), this produces the real ``.bit`` the
Cloud flashes: the same four-stage pipeline the build server runs (design
export, SoC headers, firmware, gateware), with the reserved build slot baked in
as ``FPGA_ID`` -- that slot is the board's MAC/IP, so the bitstream only works
under the slot the server reserved for it.

It runs in the build image, never on the host toolchain: the image pins the
toolchain the Cloud itself uses, and the manifest records that image's digest
so any run can be traced back to the exact build environment. The image is
``$MRG_LOCAL_BUILD_IMAGE`` (default ``mrg-local-build:latest``, built from the
Cloud repo's ``infra/fargate/Dockerfile``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ._local_build import SandboxUnavailableError

DEFAULT_LOCAL_BUILD_IMAGE = "mrg-local-build:latest"
DEFAULT_SYS_CLK_FREQ = 50_000_000

# Runs inside the image. Calls the build server's own pipeline; prints the
# failing stage and its log on error so the user sees why a build broke.
_BUILD_SCRIPT = r"""
import shutil, sys
from pathlib import Path
from tempfile import TemporaryDirectory
from cloud_fpga_orchestrator.compiler import BuildError, run_pipeline

design, slot, clk, target, top = sys.argv[1:6]
with TemporaryDirectory(prefix="mrg-local-") as scratch:
    try:
        result = run_pipeline(
            Path(design), Path(scratch), int(slot), int(clk),
            float(target) if target else None, top or None,
        )
    except BuildError as exc:
        print(f"Build failed at {exc.stage}: {exc}", file=sys.stderr)
        print(exc.log, file=sys.stderr)
        sys.exit(1)
    shutil.copyfile(result.bitstream_path, "/output/design.bit")
"""


class LocalBuildError(RuntimeError):
    """The local bitstream build ran but did not produce a bitstream."""


@dataclass
class LocalBitstream:
    path: Path
    manifest: dict


def _image() -> str:
    return os.environ.get("MRG_LOCAL_BUILD_IMAGE", DEFAULT_LOCAL_BUILD_IMAGE)


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


def build_bitstream(
    design: str | Path,
    *,
    slot_id: int,
    out_dir: str | Path,
    sys_clk_freq: int | None = None,
    timing_target_mhz: float | None = None,
    top: str | None = None,
) -> LocalBitstream:
    """Build ``design`` for build slot ``slot_id`` into ``out_dir``.

    Returns the ``.bit`` path and the upload manifest (target, slot_id,
    bitstream_sha256, image_digest, sys_clk_freq, source_revision, built_at).
    """
    if shutil.which("docker") is None:
        raise SandboxUnavailableError(
            "a local bitstream build needs Docker to run the pinned build image."
        )
    design = Path(design).resolve()
    if not design.exists():
        raise FileNotFoundError(f"design not found: {design}")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image = _image()
    digest = _image_digest(image)
    clk = sys_clk_freq or DEFAULT_SYS_CLK_FREQ

    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64", "--network=none",
        "--entrypoint", "/opt/runner-venv/bin/python",
        "-v", f"{design.parent}:/design:ro",
        "-v", f"{out_dir}:/output",
        digest, "-c", _BUILD_SCRIPT,
        f"/design/{design.name}", str(slot_id), str(clk),
        str(timing_target_mhz or ""), top or "",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "build.log").write_text(proc.stdout + proc.stderr)
    bit = out_dir / "design.bit"
    if proc.returncode != 0 or not bit.exists():
        tail = (proc.stderr or proc.stdout).strip()[-2000:]
        raise LocalBuildError(
            f"local build failed (rc={proc.returncode}); full log in "
            f"{out_dir / 'build.log'}\n{tail}"
        )

    manifest = {
        "target": "ecp5",
        "slot_id": slot_id,
        "bitstream_sha256": hashlib.sha256(bit.read_bytes()).hexdigest(),
        "image_digest": digest,
        "sys_clk_freq": clk,
        "built_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    if (rev := _source_revision(design.parent)) is not None:
        manifest["source_revision"] = rev
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return LocalBitstream(path=bit, manifest=manifest)
