from __future__ import annotations

import os
import shlex
import signal
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import requests

from . import _client, _credentials, _progress, _transport

# All App instances created at module import time are registered here so the
# CLI can discover them without the user needing to export anything explicitly.
_registry: list[App] = []


#: The longest command line sent; the server keeps at most this much.
_MAX_COMMAND = 1000
#: A flag or ``NAME=value`` whose name contains one of these has its value
#: hidden: ``--api-key``, ``--token``, ``--db-password``, ``SECRET=...``.
_SECRET_WORDS = ("key", "token", "secret", "passw")


def _is_api_key_flag(name: str) -> bool:
    """``--api-key`` or a shortened form of it, which argparse also accepts."""
    return len(name) >= len("--api") and "--api-key".startswith(name)


def _is_secret_name(name: str) -> bool:
    lowered = name.lower()
    return _is_api_key_flag(name) or any(w in lowered for w in _SECRET_WORDS)


def _redact(args: list[str]) -> list[str]:
    """``args`` with the value of every secret-looking flag replaced by ``***``."""
    out: list[str] = []
    hide_next = False
    for arg in args:
        # A word starting with "-" is the next flag, not the secret's value.
        if hide_next and not arg.startswith("-"):
            out.append("***")
            hide_next = False
            continue
        hide_next = False
        name, eq, _ = arg.partition("=")
        if eq and _is_secret_name(name):
            out.append(f"{name}=***")
        elif arg.startswith("-") and _is_secret_name(arg):
            out.append(arg)
            hide_next = True
        else:
            out.append(arg)
    return out


def _join(words: list[str]) -> str:
    """Shell-quote ``words``, dropping whole words at the end (marked ``...``)
    rather than cutting a quoted one in half."""
    line = ""
    for word in words:
        quoted = shlex.quote(word)
        longer = f"{line} {quoted}" if line else quoted
        if len(longer) > _MAX_COMMAND - len(" ..."):
            return f"{line} ..." if line else quoted[:_MAX_COMMAND]
        line = longer
    return line


def _main_module() -> str | None:
    """``pkg.mod`` when this program was started as ``python -m pkg.mod``."""
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return getattr(spec, "name", None)


def _command_line(argv: list[str] | None = None, module: str | None = None) -> str:
    """How this program was started, for the dashboard: ``mrg run sat.py``,
    ``python sat.py --n 64`` or ``python -m pkg.mod``.

    Secret-looking values are replaced by ``***`` (``_redact``). A notebook
    sends only ``jupyter``, because the kernel's arguments are a local file
    path, and ``python -c`` sends no code. ``module`` is the ``-m`` module;
    with no ``argv`` both are read from this process.
    """
    if argv is None:
        argv, module = sys.argv, _main_module()
    args = list(argv)
    if not args or not args[0]:
        return "python"
    # Split on both separators: a Windows path must work when tests run on Linux.
    prog = args[0].replace("\\", "/").rsplit("/", 1)[-1]
    if prog.lower().endswith(".exe"):
        prog = prog[:-4]
    if module == "ipykernel_launcher" or prog.startswith("ipykernel_launcher"):
        return "jupyter"
    if prog == "-c":
        return "python -c"
    if module and module != "__main__":
        head = ["python", "-m", module.removesuffix(".__main__")]
    elif prog.endswith(".py"):
        head = ["python", prog]
    else:
        head = [prog]
    return _join(head + _redact(args[1:]))


def _warn_fpga_id() -> None:
    warnings.warn(
        "fpga_id is no longer used: the server picks the board, and you reach "
        "it through your session (App.session_id). To reconnect to a session "
        "you hold, pass App(session=...) or run `mrg run --no-program`.",
        FutureWarning,
        stacklevel=3,
    )


class RegisterMap:
    """Base class for FPGA register address maps.

    Subclass and define integer class attributes for named register byte addresses:

        class Regs(manhattan_reasoning_gym.RegisterMap):
            CTRL     = 0x0000
            DATA_IN  = 0x0004
            DATA_OUT = 0x0008
    """


class App:
    """A Manhattan Reasoning Gym application: design source + register map + API config.

    The API key is resolved automatically (explicit ``api_key`` arg >
    ``$MRG_API_KEY`` > the key stored by ``mrg login``), so it
    usually doesn't need to be passed explicitly.

    You never pick a board: the server assigns one once the build finishes,
    and ``session_id`` -- your handle for that reservation -- fills in when
    ``_program()`` completes. Pass ``session=`` to reconnect to a session you
    already hold without rebuilding (``mrg run --no-program`` finds it for
    you; ``mrg status`` lists your sessions). ``fpga_id=`` is no longer used.

    ``read()``/``write()`` go over one WebSocket, opened on the first call and
    kept open for the whole session (the server closes it when the session
    ends; the next call after that raises a clear error). Pass
    ``transport="http"`` -- or set ``$MRG_TRANSPORT=http`` -- to make each call
    its own HTTP job instead: much slower, but it works on networks that block
    WebSockets.

    Typical usage::

        app = manhattan_reasoning_gym.App(
            "my_design",
            design="path/to/design.py",
        )

        @app.local_entrypoint()
        def main():
            app.write(Regs.DATA_IN, 0xDEADBEEF)
            print(hex(app.read(Regs.DATA_OUT)))

    Run with::

        mrg run myfile.py
    """

    def __init__(
        self,
        name: str,
        *,
        design: str,
        sources: list[str] | None = None,
        include_dir: str | None = None,
        top: str | None = None,
        session: str | None = None,
        fpga_id: int | None = None,
        registers: type[RegisterMap] | None = None,
        api_key: str | None = None,
        api_url: str = _client.DEFAULT_API_URL,
        sys_clk_freq: int | None = None,
        timing_target_mhz: float | None = None,
        poll_timeout: float | None = None,
        transport: str | None = None,
        local: bool = False,
    ) -> None:
        self.name = name
        # Build the bitstream on this machine instead of the build server
        # (`mrg run --local`): reserve a slot, build in the pinned image,
        # upload, then the server flashes it exactly like a cloud build.
        self.local = local
        self.design = design
        self.sources = None if sources is None else list(sources)
        self.include_dir = include_dir
        # How long _program() waits for a build before giving up. None takes
        # _client.DEFAULT_POLL_TIMEOUT, which is sized to outlast the server's
        # own build ceiling. Raise it only if your designs build slower than
        # that; lowering it risks abandoning builds that would have succeeded.
        self.poll_timeout = poll_timeout
        # How read()/write() reach the board: "websocket" (default) keeps one
        # socket open across calls; "http" makes each call its own queued job
        # -- much slower, for networks that block WebSockets. Falls back to
        # $MRG_TRANSPORT; `mrg run --transport` overrides it.
        self.transport = _transport.resolve_transport(transport)
        # Verilog-only top-module disambiguator (ignored for an Amaranth
        # design); only needed when the file has more than one module
        # exposing the Wishbone contract -- otherwise auto-detected.
        self.top = top
        if fpga_id is not None:
            _warn_fpga_id()
        # The session this App drives: set by _program(), or given to reconnect
        # to one you already hold (then there is nothing to program).
        self.session_id: str | None = session
        self.registers = registers
        self.api_url = api_url
        # Optional override for the SoC compute-domain clock (Hz). None lets the
        # build server use its default (50 MHz). Falls back to $MRG_SYS_CLK_FREQ
        # so the same knob works from the CLI, the env, or an explicit arg.
        _env_clk = os.environ.get("MRG_SYS_CLK_FREQ")
        self.sys_clk_freq = sys_clk_freq or (int(_env_clk) if _env_clk else None)
        # Optional PnR/grading timing target (MHz); defaults server-side to the
        # sys clock. Falls back to $MRG_TIMING_TARGET_MHZ, mirroring sys_clk_freq.
        _env_tt = os.environ.get("MRG_TIMING_TARGET_MHZ")
        self.timing_target_mhz = timing_target_mhz or (
            float(_env_tt) if _env_tt else None
        )
        # Resolve the key: explicit arg > $MRG_API_KEY > stored login.
        # Left empty if none found; the CLI re-checks before running, and
        # programmatic callers can set .api_key later.
        self.api_key = (
            api_key or os.environ.get("MRG_API_KEY") or _credentials.load(api_url) or ""
        )
        self._entrypoint: Callable | None = None
        self._programmed = session is not None
        self._connection: _transport.Transport | None = None
        _registry.append(self)

    def local_entrypoint(self) -> Callable:
        """Decorator that marks a function as the CLI entrypoint for this app.

        The decorated function is called by ``mrg run`` after the FPGA
        has been programmed.  It is NOT called automatically when the file is
        imported or run with ``python``.
        """

        def decorator(fn: Callable) -> Callable:
            self._entrypoint = fn
            return fn

        return decorator

    def read(self, addr: int, count: int = 1) -> int | list[int]:
        """Read ``count`` 32-bit words starting at byte address ``addr``.

        Returns a single ``int`` when *count* is 1, or a ``list[int]`` otherwise.
        Programs the FPGA first if not already done.
        """
        self._ensure_programmed()
        data = self._open_connection().read(addr, count)
        return data[0] if count == 1 else data

    def write(
        self, addr: int, value: int | list[int], fixed_address: bool = False
    ) -> None:
        """Write one or more 32-bit words to byte address ``addr``.

        *value* may be a single ``int`` or a ``list[int]`` for a burst write.
        ``fixed_address=True`` repeats ``addr`` for every word instead of
        incrementing it -- for a hardware FIFO/push-register port (a design
        that keeps its own internal write_idx), where a normal burst would
        scatter words across whatever registers happen to sit at
        address+1, address+2, ... Programs the FPGA first if not already done.
        """
        self._ensure_programmed()
        words = [value] if isinstance(value, int) else value
        self._open_connection().write(addr, words, fixed_address=fixed_address)

    @property
    def fpga_id(self) -> None:
        """Removed: boards aren't addressed by id any more. Use ``session_id``."""
        _warn_fpga_id()
        return None

    @fpga_id.setter
    def fpga_id(self, _value: int | None) -> None:
        _warn_fpga_id()

    def _program(self) -> None:
        """Submit the design and block until it's built and flashed.

        No board is chosen: the server assigns one once the build finishes.
        ``self.session_id`` is set from the completed job.
        """
        if self.local:
            self._program_local()
            return
        job_id = _client.submit(
            self.design,
            self.api_key,
            self.api_url,
            top=self.top,
            sys_clk_freq=self.sys_clk_freq,
            timing_target_mhz=self.timing_target_mhz,
            command=_command_line(),
            project=self.name,
            **({"sources": self.sources} if self.sources is not None else {}),
            **(
                {"include_dir": self.include_dir}
                if self.include_dir is not None else {}
            ),
        )
        self._wait_for_board(job_id)

    def _program_local(self) -> None:
        """Build the bitstream here, upload it, and block until it's flashed.

        The build runs in the pinned image on this machine, and the upload
        hands the job to the same flash queue a cloud build uses. How the
        board's network identity gets in depends on the image (see
        ``_local_bitstream``):

        - Flash-time identity: build first, holding nothing on the server, so
          no reservation can expire however long the build takes; then create
          the job and upload. If the server still reserves a build slot, that
          slot's identity is patched into the finished bitstream first.
        - Build-time identity (older image): reserve a slot, build for it,
          upload before the reservation expires.

        Any failure after the job exists cancels it, so a reserved slot goes
        straight back to the pool.
        """
        if self.sources is not None or self.include_dir is not None:
            raise ValueError(
                "mrg run --local builds a single design file; "
                "sources/include_dir are not supported yet"
            )
        from . import _local_bitstream

        image = _local_bitstream.resolve_image()
        placeholder = _local_bitstream.identity_placeholder(image)
        if placeholder is None:
            self._program_local_slot_first(image)
        else:
            self._program_local_build_first(image, placeholder)

    def _local_build(self, out_dir: Path, image: str, slot_id: int | None):
        from . import _local_bitstream

        built = _local_bitstream.build_bitstream(
            self.design,
            slot_id=slot_id,
            out_dir=out_dir,
            sys_clk_freq=self.sys_clk_freq,
            timing_target_mhz=self.timing_target_mhz,
            top=self.top,
            image=image,
        )
        if built.summary:
            print(f"[{self.name}] {built.summary}", flush=True)
        return built

    def _program_local_build_first(self, image: str, placeholder: list[int]) -> None:
        from . import _local_bitstream

        out_dir = Path(tempfile.mkdtemp(prefix="mrg-local-"))
        print(
            f"[{self.name}] building locally (no board reserved until the "
            f"build finishes). Output: {out_dir}",
            flush=True,
        )
        built = self._local_build(out_dir, image, None)
        res = _reserve_after_build(self.name, self.api_key, self.api_url)
        job_id = res["job_id"]
        try:
            slot_id = res.get("slot_id")
            if slot_id is not None:
                # A server from before flash-time identity: it flashes what it
                # gets, and talks to the board at the reserved slot's address.
                print(
                    f"[{self.name}] server reserved build slot {slot_id} "
                    f"(IP {res['ip']}); writing its identity into the bitstream",
                    flush=True,
                )
                built = _local_bitstream.patch_identity(
                    built, image=image, placeholder=placeholder,
                    slot_id=slot_id, ip=res["ip"], mac=res["mac"],
                )
            _client.upload_bitstream(
                job_id, built.path, built.manifest, self.api_key, self.api_url
            )
        except BaseException:
            _cancel_quietly(job_id, self.api_key, self.api_url)
            raise
        self._wait_for_board(job_id)

    def _program_local_slot_first(self, image: str) -> None:
        res = _client.reserve_local_build(self.api_key, self.api_url)
        job_id = res["job_id"]
        try:
            slot_id = res.get("slot_id")
            if slot_id is None:
                from ._local_bitstream import LocalBuildError

                raise LocalBuildError(
                    "the server assigns each board its network identity at "
                    "flash time, but this local build image bakes it in at "
                    "build time. Rebuild the image from the current Cloud repo "
                    "(docker build --platform linux/amd64 -f "
                    "infra/fargate/Dockerfile -t mrg-local-build:latest .)."
                )
            out_dir = Path(tempfile.mkdtemp(prefix=f"mrg-local-{job_id[:8]}-"))
            print(
                f"[{self.name}] reserved build slot {slot_id} "
                f"(IP {res['ip']}); building locally, upload due by "
                f"{res['expires_at']}. Output: {out_dir}",
                flush=True,
            )
            built = self._local_build(out_dir, image, slot_id)
            try:
                _client.upload_bitstream(
                    job_id, built.path, built.manifest, self.api_key, self.api_url
                )
            except _client.LocalBuildAPIError as exc:
                if exc.code == "reservation_expired":
                    raise _client.LocalBuildAPIError(
                        exc.status,
                        exc.code,
                        "the build finished after the reservation expired; "
                        "run `mrg run --local` again to reserve a new slot",
                    ) from exc
                raise
        except BaseException:
            _cancel_quietly(job_id, self.api_key, self.api_url)
            raise
        self._wait_for_board(job_id)

    def _wait_for_board(self, job_id: str) -> None:
        """Block until ``job_id`` is flashed onto a board; keep its session."""
        bar = _progress.BuildProgress(self.name)
        try:
            job = _client.poll_job(
                job_id,
                self.api_key,
                self.api_url,
                timeout=self.poll_timeout,
                on_poll=bar.update,
            )
        except BaseException:
            bar.abort()
            try:
                _client.cancel_job(job_id, self.api_key, self.api_url)
            except Exception:
                pass
            raise
        session_id = job.get("session_id")
        if not session_id:
            raise RuntimeError(
                f"Job {job_id!r} completed but the server sent no session id. "
                "It may be older than this SDK; see `mrg job "
                f"{job_id}`."
            )
        self.session_id = session_id
        bar.finish()
        self._programmed = True

    def reclock(self, sys_clk_freq: float) -> int:
        """Switch the design to another clock without rebuilding it.

        Reflashes this board with the same build re-packed at ``sys_clk_freq``
        (Hz): identical placement and routing, only the user-clock PLL divider
        differs, so a sweep over clocks is a true overclock test at a few
        seconds per point. Reachable rates are VCO / n (e.g. 600 MHz / n: 50,
        60, 75, 85.7, 100, 120, 150, ...); anything within 1% of one snaps to it.

        The board, session, and network identity carry over, but reflashing
        resets the design -- re-write any registers you had set. This App's
        WebSocket is closed first (an open socket holds the board's link lock,
        which the reflash needs); the next ``read()``/``write()`` reopens it.

        Returns:
            The exact clock flashed, in Hz (also stored as ``self.sys_clk_freq``).

        Raises:
            ReclockError: The server refused the rate (the message lists the
                ones this build can reach), or the flash failed. A failed flash
                usually means the design didn't come up at that clock; the
                server then recovers the board and ends the session.
        """
        if not self._programmed or self.session_id is None:
            raise _client.ReclockError("no board programmed yet; nothing to reclock")
        self._close_connection()
        queued = _client.reclock(
            self.session_id, self.api_key, int(sys_clk_freq), self.api_url
        )
        job = _client.wait_job(queued["job_id"], self.api_key, self.api_url)
        if job["status"] != "complete":
            try:
                state = _client.get_session(
                    self.session_id, self.api_key, self.api_url
                ).get("state")
            except requests.HTTPError as exc:
                # 404: the session is over (the failed flash ended it and the
                # board went back to be wiped).
                if exc.response is None or exc.response.status_code != 404:
                    raise
                state = "ended"
            self._programmed = state == "reserved"
            where = "still yours" if self._programmed else "your session has ended"
            raise _client.ReclockError(
                f"reclock to {queued['sys_clk_freq'] / 1e6:g} MHz "
                f"{job['status']} ({where}); "
                f"see `mrg logs {queued['job_id']}`"
            )
        self.sys_clk_freq = queued["sys_clk_freq"]
        return self.sys_clk_freq

    def release(self) -> str:
        """Release the active session, returning the reset job_id.

        Closes this App's WebSocket first: an open socket holds the board's
        link lock, which the reset job needs.
        """
        if self.session_id is None:
            return ""

        self._close_connection()

        orig_handler = None
        try:
            # Shield the network call from SIGINT so mashing Ctrl-C does not abort
            # the release request while it's in flight.
            if threading.current_thread() is threading.main_thread():
                try:
                    orig_handler = signal.getsignal(signal.SIGINT)
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                except (ValueError, TypeError):
                    orig_handler = None
            try:
                job_id = _client.release_session(
                    self.session_id, self.api_key, self.api_url
                )
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 404:
                    raise
                # 404: the session already ended (timeout, reset); the server
                # is already wiping the board, so there is nothing to release.
                job_id = ""
            self._programmed = False
            return job_id
        finally:
            if orig_handler is not None:
                signal.signal(signal.SIGINT, orig_handler)

    def __enter__(self) -> App:
        return self

    def __exit__(self, exc_type, *_) -> None:
        try:
            self.release()
        except Exception:
            # Do not let release errors mask exceptions raised inside the with-block
            if exc_type is None:
                raise

    def _ensure_programmed(self) -> None:
        if not self._programmed:
            self._program()

    def _open_connection(self) -> _transport.Transport:
        # Made on first use, not in __init__: the session is only known once
        # the board has been programmed.
        if self._connection is None:
            self._connection = _transport.open_transport(
                self.transport, self.session_id, self.api_key, self.api_url
            )
        return self._connection

    def _close_connection(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


#: After a local build, how long to keep asking for a job while the server
#: has no build capacity, and how often. The bitstream is already built, so
#: waiting a while beats throwing it away.
_RESERVE_RETRY_FOR = 600.0
_RESERVE_RETRY_EVERY = 15.0


def _reserve_after_build(name: str, api_key: str, api_url: str) -> dict:
    """``reserve_local_build`` for a finished flash-time-identity bitstream,
    retrying while the server answers ``503 no_build_capacity``."""
    deadline = time.monotonic() + _RESERVE_RETRY_FOR
    while True:
        try:
            return _client.reserve_local_build(
                api_key, api_url, flash_identity=True
            )
        except _client.LocalBuildAPIError as exc:
            if exc.code != "no_build_capacity" or time.monotonic() >= deadline:
                raise
        print(
            f"[{name}] no build capacity on the server yet; retrying in "
            f"{_RESERVE_RETRY_EVERY:.0f}s",
            flush=True,
        )
        time.sleep(_RESERVE_RETRY_EVERY)


def _cancel_quietly(job_id: str, api_key: str, api_url: str) -> None:
    try:
        _client.cancel_job(job_id, api_key, api_url)
    except Exception:
        pass
