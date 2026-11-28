"""How App.read/App.write and ``mrg read``/``mrg write`` reach the board.

Two transports share one small interface -- ``read``, ``write``, ``close``:

- ``WebSocketTransport`` (the default): one WebSocket to the orchestrator's
  ``/sessions/{session_id}/stream`` relay, opened on the first call, reused by
  every call after it, and kept open until closed or until the server closes it.
- ``HttpTransport``: every call is its own job on the orchestrator's ``/run``
  queue (``_client.read``/``_client.write``). Much slower -- the SDK polls each
  job every 0.5 s -- but plain HTTPS, for networks that block WebSockets.

Users pick HTTP with ``App(..., transport="http")``, ``$MRG_TRANSPORT``, or
``--transport http`` on the CLI. There is no automatic fallback: a WebSocket
that can't open raises, and the message says how to switch.
"""

from __future__ import annotations

import os
import threading

import websockets.exceptions
import websockets.sync.client

from . import _client, _wire

WEBSOCKET = "websocket"
HTTP = "http"
TRANSPORTS = (WEBSOCKET, HTTP)

# How long to wait for the WebSocket handshake. A network that blocks
# WebSockets often stalls the upgrade rather than refusing it outright.
_WS_OPEN_TIMEOUT = 10.0

# Close code the orchestrator's /stream uses after accepting a socket it then
# can't serve because another connection holds the board's link lock. (Its
# refusals before accepting -- board not reserved, not your session -- reach
# the client as an HTTP status on the handshake instead.)
_CLOSE_BUSY = 4409

# The session ended (idle timeout, maximum length, released, or reset). Final:
# the stream must not reconnect into a refusal.
_CLOSE_SESSION_ENDED = 4410

# The board stayed busy with another operation past the server's wait; the
# call did not run. Trying again is fine.
_CLOSE_BOARD_BUSY = 4503

#: Largest write burst sent as one transaction. A bigger ``write`` is split
#: into consecutive bursts of at most this many words (a 4096-word burst makes
#: the board host fail with 4502 "Board link failed"; 256 is known to work).
#: With ``fixed_address`` every chunk targets the same address; otherwise each
#: chunk starts where the previous one ended. Words reach the bus in order
#: either way.
MAX_WRITE_BURST_WORDS = 256

_HTTP_HINT = (
    "If your network blocks WebSockets, use HTTP instead: "
    'App(..., transport="http") or MRG_TRANSPORT=http.'
)


def resolve_transport(value: str | None = None) -> str:
    """Return the transport to use: *value* > ``$MRG_TRANSPORT`` > websocket.

    Case and surrounding whitespace are ignored; anything other than
    ``"websocket"`` or ``"http"`` raises ``ValueError``.
    """
    raw = value or os.environ.get("MRG_TRANSPORT") or WEBSOCKET
    chosen = raw.strip().lower()
    if chosen not in TRANSPORTS:
        raise ValueError(
            f"unknown transport {raw!r}: expected 'websocket' or 'http'"
        )
    return chosen


def open_transport(
    kind: str | None, target: _client.Target, api_key: str, api_url: str
) -> Transport:
    """Return a transport of *kind* (resolved via ``resolve_transport``) to
    ``target``: your session id (or, for an admin, a board id).

    Never touches the network: a WebSocket opens on its first call.
    """
    if resolve_transport(kind) == HTTP:
        return HttpTransport(target, api_key, api_url)
    return WebSocketTransport(target, api_key, api_url)


def write_chunks(
    addr: int, words: list[int], fixed_address: bool
) -> list[tuple[int, list[int]]]:
    """Split a write into ``(address, words)`` bursts of at most
    ``MAX_WRITE_BURST_WORDS``. An empty write stays one (empty) burst."""
    if len(words) <= MAX_WRITE_BURST_WORDS:
        return [(addr, list(words))]
    chunks = []
    for start in range(0, len(words), MAX_WRITE_BURST_WORDS):
        chunk_addr = addr if fixed_address else addr + 4 * start
        chunks.append((chunk_addr, list(words[start:start + MAX_WRITE_BURST_WORDS])))
    return chunks


class HttpTransport:
    """Each call is its own job on the orchestrator's ``/run`` queue."""

    def __init__(self, target: _client.Target, api_key: str, api_url: str) -> None:
        self._target = target
        self._api_key = api_key
        self._api_url = api_url

    def read(self, addr: int, count: int = 1) -> list[int]:
        return _client.read(self._target, self._api_key, addr, count, self._api_url)

    def write(
        self, addr: int, words: list[int], fixed_address: bool = False
    ) -> None:
        for chunk_addr, chunk in write_chunks(addr, words, fixed_address):
            _client.write(
                self._target, self._api_key, chunk_addr, chunk, self._api_url,
                fixed_address=fixed_address,
            )

    def close(self) -> None:
        """Nothing to close: every call made and finished its own request."""


class WebSocketTransport:
    """One WebSocket to the FPGA: opened on first use and kept open until closed.

    The server keeps it open for the whole session and closes it with 4410
    when the session ends; a socket the server closed while idle is replaced
    on the next call. Every call holds ``self._lock`` for its whole
    send/receive, so two threads sharing one App can't interleave frames on
    the wire -- the relay matches replies to requests purely by order.
    """

    def __init__(self, target: _client.Target, api_key: str, api_url: str) -> None:
        self._target = target
        self._api_key = api_key
        self._api_url = api_url
        self._lock = threading.Lock()
        self._conn: websockets.sync.client.ClientConnection | None = None

    def read(self, addr: int, count: int = 1) -> list[int]:
        """Read ``count`` 32-bit words starting at byte address ``addr``."""
        what = f"read of {count} words at {addr:#x}"
        request = _wire.WishboneRequest(
            op=_wire.WishboneOp.READ, address=addr, count=count
        )
        response = self._transact(request, what)
        if not response.ok:
            raise RuntimeError(f"{what} failed")
        return response.data

    def write(
        self, addr: int, words: list[int], fixed_address: bool = False
    ) -> None:
        """Write ``words`` from ``addr`` up (or all at ``addr`` if fixed).

        Sent as bursts of at most ``MAX_WRITE_BURST_WORDS``; a failure part
        way through means the earlier bursts were written.
        """
        chunks = write_chunks(addr, words, fixed_address)
        for i, (chunk_addr, chunk) in enumerate(chunks):
            what = f"write to {addr:#x}"
            if len(chunks) > 1:
                what += f" (burst {i + 1}/{len(chunks)}, {len(chunk)} words)"
            request = _wire.WishboneRequest(
                op=_wire.WishboneOp.WRITE,
                address=chunk_addr,
                data=chunk,
                fixed_address=fixed_address,
            )
            response = self._transact(request, what)
            if not response.ok:
                raise RuntimeError(f"{what} failed")

    def close(self) -> None:
        """Close the socket. Safe to call twice."""
        with self._lock:
            self._close_locked()

    # -- internals; every *_locked method expects self._lock to be held -------

    def _transact(
        self, request: _wire.WishboneRequest, what: str
    ) -> _wire.WishboneResponse:
        payload = request.to_bytes()
        with self._lock:
            try:
                conn = self._send_locked(payload)
                raw = conn.recv()
            except websockets.exceptions.ConnectionClosed as exc:
                self._close_locked()
                raise _closed_error(exc, self._target, what) from exc
            except BaseException:
                # Interrupted mid-exchange (e.g. Ctrl-C while waiting for the
                # reply): the server may still answer, and the relay pairs
                # replies with requests purely by order, so this socket would
                # hand the next call a stale reply. Drop it.
                self._close_locked()
                raise
        if isinstance(raw, str):
            raise RuntimeError("received an unexpected text frame")
        return _wire.WishboneResponse.from_bytes(raw)

    def _send_locked(self, payload: bytes) -> websockets.sync.client.ClientConnection:
        """Send *payload*, replacing a socket that closed while idle.

        Returns the connection the payload went out on.
        """
        conn = self._open_locked()
        try:
            conn.send(payload)
        except websockets.exceptions.ConnectionClosed as exc:
            # The socket closed while idle (server restart, network drop,
            # session ended). websockets refuses to send on a socket that
            # isn't open, so nothing reached the server and a fresh socket
            # can safely carry this call -- unless the session ended, which is
            # final: say so instead of reconnecting into a refusal.
            # Only once: a second failure is a real one.
            if exc.rcvd is not None and exc.rcvd.code == _CLOSE_SESSION_ENDED:
                self._close_locked()
                raise _closed_error(exc, self._target, "this call") from exc
            self._close_locked()
            conn = self._open_locked()
            conn.send(payload)
        return conn

    def _open_locked(self) -> websockets.sync.client.ClientConnection:
        if self._conn is None:
            self._conn = _connect(self._target, self._api_key, self._api_url)
        return self._conn

    def _close_locked(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()


def _connect(
    target: _client.Target, api_key: str, api_url: str
) -> websockets.sync.client.ClientConnection:
    try:
        conn = websockets.sync.client.connect(
            _client._ws_url(api_url, target),
            additional_headers={"X-API-Key": api_key},
            open_timeout=_WS_OPEN_TIMEOUT,
        )
    except websockets.exceptions.InvalidStatus as exc:
        status = exc.response.status_code
        if status == 404:
            raise ConnectionError(
                "Couldn't reach your board: the session has ended or isn't "
                "yours (HTTP 404). 'mrg status' lists your sessions."
            ) from exc
        raise ConnectionError(
            f"Couldn't open a WebSocket to your board (HTTP {status}). "
            f"{_HTTP_HINT}"
        ) from exc
    except (OSError, websockets.exceptions.InvalidHandshake) as exc:
        # OSError covers DNS failures, refused connections, and the
        # TimeoutError a stalled upgrade raises after _WS_OPEN_TIMEOUT.
        raise ConnectionError(
            f"Couldn't open a WebSocket to your board ({exc}). {_HTTP_HINT}"
        ) from exc
    # websockets >= 16 warns when a connection is used outside
    # `with connect(...)`. This one deliberately outlives any single block --
    # _close_locked closes it -- so enter its context explicitly.
    return conn.__enter__()


def _closed_error(
    exc: websockets.exceptions.ConnectionClosed, target: _client.Target, what: str
) -> ConnectionError:
    """Explain a socket that closed while a call was waiting for its reply."""
    rcvd = exc.rcvd
    if rcvd is not None and rcvd.code == _CLOSE_SESSION_ENDED:
        return ConnectionError(
            f"Your session has ended ({rcvd.reason}). "
            "Submit your design again to get a board."
        )
    if rcvd is not None and rcvd.code == _CLOSE_BOARD_BUSY:
        return ConnectionError(
            f"Your board is busy ({rcvd.reason}). Try again in a moment."
        )
    if rcvd is not None and rcvd.code == _CLOSE_BUSY:
        # The server never forwarded this call: it closes before reading.
        return ConnectionError(
            "Your board is in use by another connection "
            f"(server: {rcvd.reason}). Close other scripts or terminals "
            "using this board, then try again."
        )
    detail = f" (server closed with {rcvd.code}: {rcvd.reason})" if rcvd else ""
    return ConnectionError(
        f"Connection to your board lost during {what}{detail}. "
        "The operation may or may not have run on the FPGA."
    )


# What open_transport returns; App and the CLI only use read/write/close.
Transport = HttpTransport | WebSocketTransport
