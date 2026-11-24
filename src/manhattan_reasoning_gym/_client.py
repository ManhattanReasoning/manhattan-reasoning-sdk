"""HTTP client for the Cloud FPGA orchestrator API (WebSocket side: _transport)."""

from __future__ import annotations

import json
import os
import time

import requests

DEFAULT_API_URL = "https://api.manhattanreasoning.com"
_RUN_POLL_INTERVAL = 0.5
_BUILD_POLL_INTERVAL = 2.0
_ANIM_INTERVAL = 0.08  # spinner tick when a progress callback is attached

# Wall-clock deadline for poll_job. This has to outlast the *server's* own build
# ceiling, or the client abandons builds that go on to succeed -- throwing away
# a good bitstream and, if a board was already assigned, programming it for
# nothing.
#
# The server's ceiling is not one number. A build runs four sequentially timed
# stages (orchestrator compiler/stages.py): design export, SoC header
# generation, and firmware compile each get BUILD_SYNTH_TIMEOUT_SECONDS, then
# place-and-route gets BUILD_TIMEOUT_SECONDS. The deployed build image
# (infra/fargate/Dockerfile) sets 1800 and 5400, so the real worst case is
# 3*1800 + 5400 = 10800 s (3 h) of build alone, before any time spent queued
# for one of the finite build slots or waiting for a board to free up to flash.
#
# 4 h leaves ~1 h of margin over that for queueing and flashing. Being too
# generous only costs a wedged client some extra waiting -- with a live status
# in the progress bar and Ctrl-C always available -- whereas being too tight
# silently discards completed work, which is the failure this default exists to
# prevent. Override per call, per App(poll_timeout=...), or with $MRG_POLL_TIMEOUT.
#
# NOTE: if you change BUILD_TIMEOUT_SECONDS or BUILD_SYNTH_TIMEOUT_SECONDS in
# the build image, this is the other half of that change.
DEFAULT_POLL_TIMEOUT = float(os.environ.get("MRG_POLL_TIMEOUT", 14400.0))


def exchange_github_token(github_token: str, api_url: str) -> dict:
    """Exchange a GitHub token for an API key.

    Returns the decoded {"api_key", "github_username"} response.
    """
    resp = requests.post(
        f"{api_url}/auth/github/exchange",
        json={"github_token": github_token},
    )
    resp.raise_for_status()
    return resp.json()


def revoke_key(api_key: str, api_url: str) -> None:
    """Revoke the calling user's API key (DELETE /auth/keys/me)."""
    resp = requests.delete(
        f"{api_url}/auth/keys/me",
        headers={"X-API-Key": api_key},
    )
    resp.raise_for_status()


def submit(
    design_path: str,
    api_key: str,
    api_url: str,
    top: str | None = None,
    sys_clk_freq: int | None = None,
    timing_target_mhz: float | None = None,
    *,
    sources: list[str] | None = None,
    include_dir: str | None = None,
    command: str | None = None,
    project: str | None = None,
) -> str:
    """Submit a design for build_and_program and return its job id.

    Never touches a board: the server claims a build slot (a network identity
    baked into the bitstream) from a pool sized larger than the physical
    fleet, so many builds run concurrently on Fargate. Which board ends up
    running the design isn't known until the build finishes and some board's
    worker claims it -- see ``poll_job``.

    ``design_path``'s extension picks the language server-side (.py ->
    Amaranth, .v -> plain Verilog); ``top`` is a Verilog-only top-module
    disambiguator (ignored for Amaranth), only needed when the file has more
    than one module exposing the required Wishbone contract.

    ``command`` and ``project`` are only for the web dashboard, which shows
    them with the build: the command line that started it and the project
    name. ``App`` sends both, with secret values in the command hidden
    (``_app._command_line``); a build submitted without them shows neither.
    Both are cut to the length the server keeps.
    """
    # Inspect local dependencies without importing or executing the design.
    url = f"{api_url}/submit"
    # Optional multipart form fields; older servers simply ignore extra fields.
    # top disambiguates a Verilog design's top module; sys_clk_freq re-clocks
    # the SoC (Hz); timing_target_mhz is the PnR/grading constraint (MHz),
    # sent only when the caller overrides the sys-clock default.
    data = {}
    if top:
        data["top"] = top
    if sys_clk_freq:
        data["sys_clk_freq"] = str(sys_clk_freq)
    if timing_target_mhz:
        data["timing_target_mhz"] = str(timing_target_mhz)
    # Shown on the web dashboard; older servers ignore them.
    if command:
        data["command"] = command[:1000]
    if project:
        data["project"] = project[:100]
    data = data or None
    from ._project import source_bundle

    bundle = source_bundle(design_path, sources, include_dir, single_file_raw=True)
    if bundle is not None:
        resp = requests.post(
            url, headers={"X-API-Key": api_key},
            files={"file": ("project.zip", bundle, "application/zip")}, data=data,
        )
    else:
        with open(design_path, "rb") as f:
            resp = requests.post(
                url, headers={"X-API-Key": api_key}, files={"file": f}, data=data
            )
    resp.raise_for_status()
    return resp.json()["job_id"]


class LocalBuildAPIError(RuntimeError):
    """The server refused a local-build call; ``code`` is its error code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{message} ({status} {code})")
        self.status = status
        self.code = code


def _raise_api_error(resp) -> None:
    if resp.ok:
        return
    try:
        detail = resp.json().get("detail") or {}
    except ValueError:
        detail = {}
    if not isinstance(detail, dict):
        detail = {"message": str(detail)}
    raise LocalBuildAPIError(
        resp.status_code,
        detail.get("error", "http_error"),
        detail.get("message", resp.reason or "request failed"),
    )


def reserve_local_build(api_key: str, api_url: str) -> dict:
    """Reserve a build slot for a build that runs on this machine.

    Returns ``job_id``, ``slot_id`` (bake it in as FPGA_ID), ``ip``, ``mac``
    and ``expires_at`` (the upload deadline).
    """
    resp = requests.post(
        f"{api_url}/builds/local", headers={"X-API-Key": api_key}
    )
    _raise_api_error(resp)
    return resp.json()


def upload_bitstream(
    job_id: str, bitstream_path, manifest: dict, api_key: str, api_url: str
) -> dict:
    """Upload a locally built ``.bit`` and its manifest; the server queues a flash."""
    with open(bitstream_path, "rb") as f:
        resp = requests.put(
            f"{api_url}/jobs/{job_id}/bitstream",
            headers={"X-API-Key": api_key},
            files={"file": (os.path.basename(str(bitstream_path)), f)},
            data={"manifest": json.dumps(manifest)},
        )
    _raise_api_error(resp)
    return resp.json()


def poll_job(
    job_id: str,
    api_key: str,
    api_url: str,
    timeout: float | None = None,
    on_poll=None,
) -> dict:
    """Block until a job completes, and return the final job record.

    ``timeout`` is the wall-clock deadline in seconds; ``None`` uses
    ``DEFAULT_POLL_TIMEOUT`` (see there for how it's derived from the server's
    build ceiling). The point of the default is that the client outlasts the
    server, so this surfaces the real ``complete``/``failed`` status rather
    than giving up on a build that is still running and may yet succeed.

    The server is the authority on when a build has taken too long -- it
    enforces its own per-stage ceilings and marks the job ``failed``. This
    deadline is only a backstop for a server that has gone away or a job that
    has wedged, so erring long is deliberate.

    If ``on_poll`` is given it's called as ``on_poll(status, fpga_id)`` on a
    fast tick (~12/s) so callers can animate a spinner, while the job itself
    is only queried every ``_BUILD_POLL_INTERVAL`` seconds. ``fpga_id`` is
    None until a board is assigned (some board's worker has claimed the
    finished build) -- for a build_and_program job that's anywhere from
    submit through the build finishing on Fargate.
    """
    if timeout is None:
        timeout = DEFAULT_POLL_TIMEOUT
    url = f"{api_url}/jobs/{job_id}"
    headers = {"X-API-Key": api_key}
    tick = _ANIM_INTERVAL if on_poll else _BUILD_POLL_INTERVAL
    deadline = time.monotonic() + timeout
    next_check = 0.0
    job = {"status": "queued", "fpga_id": None}
    while time.monotonic() < deadline:
        if time.monotonic() >= next_check:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            job = resp.json()
            next_check = time.monotonic() + _BUILD_POLL_INTERVAL
            if job["status"] == "complete":
                if on_poll:
                    on_poll(job["status"], job["fpga_id"])
                return job
            if job["status"] in ("failed", "cancelled"):
                logs_resp = requests.get(f"{url}/logs", headers=headers)
                logs = logs_resp.text if logs_resp.ok else "no logs available"
                raise RuntimeError(
                    f"Job {job_id!r} ended with status {job['status']!r}.\n{logs}"
                )
        if on_poll:
            on_poll(job["status"], job["fpga_id"])
        time.sleep(tick)
    # Giving up here says nothing about the job: the server keeps building, and
    # this one may still complete. Say so, so a timeout isn't misread as a
    # failed build (and so the id needed to recover it isn't buried in a log).
    raise TimeoutError(
        f"Job {job_id!r} was still {job['status']!r} after {timeout}s, so this "
        f"client stopped waiting. The build has NOT been cancelled and may "
        f"still complete -- check `mrg jobs` or GET /jobs/{job_id}. If builds "
        f"legitimately take this long, raise the deadline with "
        f"$MRG_POLL_TIMEOUT or App(poll_timeout=...)."
    )


def read(
    fpga_id: int, api_key: str, address: int, count: int, api_url: str
) -> list[int]:
    url = f"{api_url}/fpga/{fpga_id}/run"
    resp = requests.post(
        url,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={"op": 2, "address": address, "data": [], "count": count},
    )
    resp.raise_for_status()
    return _poll_run(resp.json()["job_id"], api_key, api_url)


def write(
    fpga_id: int,
    api_key: str,
    address: int,
    words: list[int],
    api_url: str,
    fixed_address: bool = False,
) -> None:
    url = f"{api_url}/fpga/{fpga_id}/run"
    resp = requests.post(
        url,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={
            "op": 1,
            "address": address,
            "data": words,
            "count": 0,
            "fixed_address": fixed_address,
        },
    )
    resp.raise_for_status()
    _poll_run(resp.json()["job_id"], api_key, api_url)


def list_fpgas(api_key: str, api_url: str) -> list[dict]:
    resp = requests.get(f"{api_url}/fpga", headers={"X-API-Key": api_key})
    resp.raise_for_status()
    return resp.json()


def get_fpga(fpga_id: int, api_key: str, api_url: str) -> dict:
    resp = requests.get(f"{api_url}/fpga/{fpga_id}", headers={"X-API-Key": api_key})
    resp.raise_for_status()
    return resp.json()


def get_job(job_id: str, api_key: str, api_url: str) -> dict:
    resp = requests.get(
        f"{api_url}/jobs/{job_id}",
        headers={"X-API-Key": api_key},
    )
    resp.raise_for_status()
    return resp.json()


def list_jobs(
    api_key: str,
    api_url: str,
    status: str | None = None,
    *,
    timeout: float | None = None,
) -> list[dict]:
    """Return every job the caller (the owner of ``api_key``) has submitted,
    newest first. Pass ``status`` (e.g. ``"running"``) to filter."""
    params = {"status": status} if status else None
    resp = requests.get(
        f"{api_url}/jobs", headers={"X-API-Key": api_key}, params=params,
        **({"timeout": timeout} if timeout is not None else {}),
    )
    resp.raise_for_status()
    return resp.json()


def get_logs(job_id: str, api_key: str, api_url: str) -> str:
    resp = requests.get(
        f"{api_url}/jobs/{job_id}/logs",
        headers={"X-API-Key": api_key},
    )
    resp.raise_for_status()
    return resp.text


def cancel_job(job_id: str, api_key: str, api_url: str) -> None:
    resp = requests.delete(
        f"{api_url}/jobs/{job_id}",
        headers={"X-API-Key": api_key},
    )
    resp.raise_for_status()


def get_session(fpga_id: int, api_key: str, api_url: str) -> dict:
    resp = requests.get(
        f"{api_url}/fpga/{fpga_id}/session",
        headers={"X-API-Key": api_key},
    )
    resp.raise_for_status()
    return resp.json()


def release_session(fpga_id: int, api_key: str, api_url: str) -> str:
    resp = requests.post(
        f"{api_url}/fpga/{fpga_id}/session/release",
        headers={"X-API-Key": api_key},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["job_id"]


def reset_fpga(fpga_id: int, api_key: str, api_url: str) -> dict:
    resp = requests.post(
        f"{api_url}/fpga/{fpga_id}/reset",
        headers={"X-API-Key": api_key},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()


class ReclockError(RuntimeError):
    """A reclock the server refused, or one whose flash failed."""


def reclock(fpga_id: int, api_key: str, sys_clk_freq: int, api_url: str) -> dict:
    """Queue a reflash of a reserved board's build at another user clock (Hz).

    Every build is also packed at each clock its PLL reaches without moving the
    control plane (VCO / n); the server snaps ``sys_clk_freq`` to the nearest
    of those within 1%. Returns the server's response, whose ``sys_clk_freq`` is
    the exact rate that will be flashed.

    Raises:
        ReclockError: The server refused (409/422) -- its message says why,
            e.g. listing the rates this build can reach.
    """
    resp = requests.post(
        f"{api_url}/fpga/{fpga_id}/reclock",
        headers={"X-API-Key": api_key},
        json={"sys_clk_freq": int(sys_clk_freq)},
        timeout=10.0,
    )
    if resp.status_code in (409, 422):
        try:
            message = resp.json()["detail"]["message"]
        except (ValueError, KeyError, TypeError):
            message = resp.text
        raise ReclockError(message)
    resp.raise_for_status()
    return resp.json()


def wait_job(
    job_id: str, api_key: str, api_url: str, timeout: float = 600.0
) -> dict:
    """Block until a board-side job (reset, reclock) finishes; return its record."""
    deadline = time.monotonic() + timeout
    while True:
        job = get_job(job_id, api_key, api_url)
        if job["status"] in ("complete", "failed", "cancelled"):
            return job
        if time.monotonic() > deadline:
            raise TimeoutError(f"job {job_id!r} did not finish in {timeout:.0f}s")
        time.sleep(_BUILD_POLL_INTERVAL)


def _poll_run(job_id: str, api_key: str, api_url: str) -> list[int]:
    status_url = f"{api_url}/jobs/{job_id}"
    result_url = f"{status_url}/result"
    headers = {"X-API-Key": api_key}
    for _ in range(120):
        resp = requests.get(status_url, headers=headers)
        resp.raise_for_status()
        status = resp.json()["status"]
        if status == "complete":
            resp = requests.get(result_url, headers=headers)
            resp.raise_for_status()
            result = resp.json()
            if result.get("ok"):
                return result.get("data", [])
            raise RuntimeError(f"Run op failed: {result}")
        if status in ("failed", "cancelled"):
            raise RuntimeError(f"Run job {job_id!r} ended with status {status!r}")
        time.sleep(_RUN_POLL_INTERVAL)
    raise TimeoutError(f"Run job {job_id!r} did not complete in time")


def _ws_url(api_url: str, fpga_id: int) -> str:
    """Convert the orchestrator's http(s):// API URL to its /stream ws(s):// form."""
    if api_url.startswith("https://"):
        base = "wss://" + api_url[len("https://") :]
    elif api_url.startswith("http://"):
        base = "ws://" + api_url[len("http://") :]
    else:
        base = api_url
    return f"{base}/fpga/{fpga_id}/stream"


def get_health(api_url: str) -> dict:
    """Read public server health with a bounded wait for status display."""
    response = requests.get(f"{api_url.rstrip('/')}/health", timeout=3)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Invalid health response")
    return data

