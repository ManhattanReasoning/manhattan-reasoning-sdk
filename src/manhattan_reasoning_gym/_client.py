"""HTTP client for the Cloud FPGA orchestrator API (WebSocket side: _transport)."""

from __future__ import annotations

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
    if resp.status_code == 402:
        raise _billing_error(resp)
    resp.raise_for_status()
    return resp.json()["job_id"]


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


class BillingError(RuntimeError):
    """A request the server refused for a membership or quota reason.

    ``code`` is the server's machine-readable reason. From ``submit`` (HTTP
    402): ``subscription_required``, ``billing_blocked``, ``quota_exhausted``
    or ``build_in_flight``. From the billing calls (409): ``already_subscribed``,
    ``subscription_needs_attention``, ``no_membership`` or
    ``no_stripe_customer``; or ``billing_unavailable`` (503). The message ends
    with the ``mrg`` command that resolves it, when there is one.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        hint = _BILLING_HINTS.get(code)
        super().__init__(f"{message}\n  {hint}" if hint else message)


_BILLING_HINTS = {
    "subscription_required": "run 'mrg subscribe' to start a membership",
    "billing_blocked": "run 'mrg billing' to fix your payment method, or "
                       "'mrg subscribe' if the membership was cancelled",
    "quota_exhausted": "run 'mrg usage' to see this month's build-minutes",
    "build_in_flight": "run 'mrg jobs --status running' to see it, "
                       "'mrg cancel <job_id>' to stop it",
    "already_subscribed": "run 'mrg billing' to manage it",
    "subscription_needs_attention": "run 'mrg billing' to update your payment "
                                    "method",
    "no_membership": "run 'mrg subscribe' to start one",
}


def _billing_error(resp) -> BillingError:
    """Build a BillingError from an error response's ``detail`` body."""
    try:
        detail = resp.json()["detail"]
        return BillingError(str(detail["error"]), str(detail["message"]))
    except (ValueError, KeyError, TypeError):
        return BillingError("unknown", resp.text or f"HTTP {resp.status_code}")


def _billing_session_url(path: str, api_key: str, api_url: str) -> str:
    resp = requests.post(
        f"{api_url}{path}", headers={"X-API-Key": api_key}, timeout=30.0
    )
    if resp.status_code in (409, 503):
        raise _billing_error(resp)
    resp.raise_for_status()
    return resp.json()["url"]


def create_checkout(api_key: str, api_url: str) -> str:
    """Return a Stripe Checkout URL where the caller can start a membership.

    Nothing is charged by this call: the URL is a Stripe-hosted page to open
    in a browser. The membership becomes active when Stripe confirms payment.

    Raises:
        BillingError: The account has no billing record yet (log in again),
            or the billing provider is unreachable.
    """
    return _billing_session_url("/billing/checkout", api_key, api_url)


def create_portal(api_key: str, api_url: str) -> str:
    """Return a Stripe Customer Portal URL for managing the membership.

    The portal is where a member updates their card, downloads invoices, or
    cancels. Raises BillingError as ``create_checkout`` does.
    """
    return _billing_session_url("/billing/portal", api_key, api_url)


def get_usage(api_key: str, api_url: str, *, timeout: float = 30.0) -> dict:
    """Return the caller's build-minute usage for the current UTC month.

    Keys: ``month`` (YYYY-MM), ``build_minutes_used``, ``build_minutes_limit``,
    ``build_minutes_remaining``, and ``builds`` (recent metered builds, newest
    first). Newer servers add the membership: ``billing_status`` (``none``,
    ``active``, ``blocked`` or ``comp``), ``subscription_required`` and
    ``billing_enforced`` -- see ``membership_action``.
    """
    resp = requests.get(
        f"{api_url}/usage", headers={"X-API-Key": api_key}, timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()


def membership_action(usage: dict) -> str | None:
    """Return what the caller must do before their next build, if anything.

    ``"subscribe"`` when a membership is required and they have none,
    ``"fix"`` when their membership is blocked (cancelled, or a payment is
    failing), otherwise ``None``. Also ``None`` whenever the server isn't
    enforcing billing -- nothing is refused then, so there is nothing to
    prompt for -- and for older servers that don't report a membership.
    """
    if not usage.get("billing_enforced"):
        return None
    status = usage.get("billing_status")
    if status == "blocked":
        return "fix"
    if status == "none" and usage.get("subscription_required"):
        return "subscribe"
    return None


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

