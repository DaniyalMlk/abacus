"""The stdio transport: one JSON message per line, in and out.

The loop deliberately keeps nothing between iterations. An open stdio process is
not a session and not a conversation — a client may interleave requests from
unrelated conversations on the same pipe, so treating process identity as a
proxy for conversation continuity would be wrong even though nothing would
immediately break.

Logs go to stderr. Stdout carries the protocol, and a stray print into it
corrupts the stream; the ``logging`` feature that used to carry diagnostics is
deprecated in this revision, with stderr named as the replacement.
"""

from __future__ import annotations

import json
import sys
from typing import IO

from ..server import Server


def serve_stdio(
    server: Server,
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
) -> None:
    """Read messages from ``stdin`` until it closes, writing responses to ``stdout``."""
    source = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout

    for line in source:
        text = line.strip()
        if not text:
            continue
        response = server.handle_json(text)
        if response is None:
            # A notification. The protocol forbids answering one, and emitting
            # anything here would desynchronise a client counting responses.
            continue
        sink.write(json.dumps(response, separators=(",", ":")) + "\n")
        sink.flush()
