# Changelog

## 0.2.1

### Fixed

- `mrg job`, `mrg logs` and `mrg cancel` accept the short job id that
  `mrg jobs` prints (any unique prefix), instead of failing with a 422.
- Waiting for a build no longer gives up on the first dropped connection or
  gateway error: it keeps polling through up to 5 minutes of outage (for
  example a server restart), since the build keeps running server-side.
- `mrg logs`, and the log shown when a build fails, print as text instead of
  one line of escaped `\n`s.

## 0.2.0

Boards are now reached through **sessions**, not board numbers. You never
choose a board or learn which one you got: the server assigns one when your
build is flashed, and each reservation gets a random session id that stops
working when the session ends.

### Changed

- `App(session=...)` reconnects to a session you hold without rebuilding.
  After a build, `app.session_id` holds the session's id.
- `App(fpga_id=...)` and `app.fpga_id` are no longer used. They emit a
  `FutureWarning`, are ignored, and will be removed in a later release.
- `mrg run --session ID` reconnects to that session; `mrg run --no-program`
  reconnects to the one you hold. `--fpga-id` was removed and exits with a
  message explaining the change.
- `mrg status` lists your sessions (no board numbers); `mrg status SESSION`
  shows one. A session id can be shortened to its first 8 characters.
- `mrg reset [SESSION]`, `mrg read ADDRESS [--count N]` and
  `mrg write ADDRESS VALUE` act on your session; pick one with `--session ID`
  if you hold several. The old forms that led with a board number exit with a
  message.
- `mrg jobs` shows a SESSION column instead of FPGA; `mrg job` shows the
  session instead of the board.
- The build progress line and connection errors no longer name a board.
- New `cloud.list_sessions()`. `get_session`, `release_session`, `reset_session`,
  `reclock`, `read` and `write` in the client take a session id.
- Builds send `X-MRG-Client-Features: sessions` so the server knows this client
  reaches its board by session.

### Operators

Admin accounts can still address a board by number: `mrg status --board N`,
`mrg status --boards`, and `--board N` on `reset`, `read` and `write`. The
client functions accept an `int` board id in place of a session id. The server
refuses these to anyone else.

### Upgrading

Older SDKs keep working until the server stops accepting board-addressed
requests; after that they get an "upgrade required" error. Upgrade with
`pip install -U manhattan-reasoning-gym`.
