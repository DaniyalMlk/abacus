"""Server-minted handles for state the protocol has nowhere to keep.

Revision ``2026-07-28`` is stateless. There is no handshake, no session and no
``Mcp-Session-Id``, and the protocol module is written so that nothing can hold
state between requests. A position book, however, is state by definition: the
caller assembles one and then asks several questions of it.

The obvious implementation is a dictionary on the server, keyed by a random
string handed back to the caller. It is also the wrong one, for three reasons
that have nothing to do with taste:

*It reintroduces the session.* A server-side book is reachable by whoever
presents the key, and there is no session scope to confine it to. Two requests
on one stdio pipe may come from unrelated conversations, so "the caller who
opened it" is not a thing the server can identify.

*It grows without bound.* Nothing in the protocol tells a server that a caller
has finished. There is no close, no disconnect that means anything, and a
language model is under no obligation to release what it allocated. The store
would only ever get larger, and any eviction policy would amount to expiry
implemented badly.

*It breaks on the second process.* Two workers behind a load balancer do not
share a dictionary, and a handle minted by one would be unrecognised by the
other on the very next request.

So the handle carries the book instead of pointing at one. The payload is
canonical JSON, compressed, and authenticated with a keyed MAC; the server
stores nothing and verifies everything. The three problems disappear: there is
no store to scope, nothing to grow, and any process holding the same key can
redeem a handle any other minted.

What this buys and what it costs, stated plainly because it belongs in the
design rather than in a footnote:

- The caller *can* read the payload. It is compressed and signed, not encrypted.
  That is acceptable because the payload is the caller's own book, which it just
  sent. It would not be acceptable for anything the caller should not see, and
  nothing of that kind may be put in here.
- The caller *cannot* alter it. The MAC is over the compressed bytes and is
  checked in constant time, so an edited handle is refused rather than decoded
  into a book nobody asked for.
- The handle is bounded in size, and therefore so is the book. A book that will
  not fit is refused when it is opened, naming the limit, rather than minting
  something that fails on use.
- By default the key lives for the life of the process. A restart invalidates
  every outstanding handle, which is why "not recognised" is reported separately
  from "expired": the caller's recovery is the same, but a caller that sees the
  first after a working call has learned something true about the server.
- A deployment that runs more than one process, or restarts one, gives every
  process the same key through ``ABACUS_HANDLE_KEY``. Then a handle minted by
  one worker is redeemed by the next, which is the property that made carrying
  the book inside the handle worth doing in the first place.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time
import zlib
from collections.abc import Mapping
from typing import Any, Final

#: Version tag on every handle. It is inside the authenticated payload as well
#: as in front of it, so the prefix cannot be rewritten to make an old handle
#: look like a new one.
HANDLE_VERSION: Final = "abk1"

#: How long a handle stays redeemable, in seconds. An hour is longer than any
#: single exchange with a model and short enough that a handle pasted into a
#: transcript is useless by the time anyone reads it back.
DEFAULT_TTL_S: Final = 3600

#: Longest handle this server will mint, in characters. A handle travels inside
#: a JSON-RPC result and then back inside the next request's arguments, and a
#: model has to carry it through its context in between. Eight kilobytes is
#: generous for a book of any size a person would hold in their head, and small
#: enough that a handle never dominates the message it rides in.
MAX_HANDLE_CHARS: Final = 8192

#: Bytes of MAC kept. A truncated HMAC-SHA256 is still an HMAC-SHA256; 16 bytes
#: leaves a forgery probability of 2^-128, and the 16 saved are 22 characters of
#: handle that do not have to be carried around.
MAC_BYTES: Final = 16

_SEPARATOR: Final = "."

#: The environment variable a shared key is read from, as hexadecimal.
KEY_VARIABLE: Final = "ABACUS_HANDLE_KEY"

#: Shortest key accepted, in bytes. The MAC is HMAC-SHA256, whose key is
#: as strong as its entropy up to the hash's own block; thirty-two random bytes
#: is the size the server draws for itself, and a shared key should not be the
#: weaker of the two.
MIN_KEY_BYTES: Final = 32

_GENERATE: Final = "python -c 'import secrets; print(secrets.token_hex(32))'"


class BadHandleKey(ValueError):
    """A shared key was supplied and cannot be used.

    The message describes the key's shape and never the key, because it is
    printed to a terminal and lands in whatever collects the server's logs.
    """


def parse_key(text: str) -> bytes:
    """Read a shared handle key written as hexadecimal.

    Surrounding whitespace is ignored, since a key copied from a file usually
    carries a newline. Anything else that is not a hex digit is refused, as is a
    key shorter than :data:`MIN_KEY_BYTES`.
    """
    stripped = text.strip()
    if not stripped:
        raise BadHandleKey(
            f"{KEY_VARIABLE} is set but empty. Unset it to have the server draw "
            f"its own key, or set it to {MIN_KEY_BYTES * 2} hex digits: {_GENERATE}"
        )
    try:
        raw = bytes.fromhex(stripped)
    except ValueError:
        raise BadHandleKey(
            f"{KEY_VARIABLE} must be hexadecimal, and the {len(stripped)} characters "
            f"it holds are not all hex digits. Generate one with: {_GENERATE}"
        ) from None
    if len(raw) < MIN_KEY_BYTES:
        raise BadHandleKey(
            f"{KEY_VARIABLE} holds a {len(raw)}-byte key; it must be at least "
            f"{MIN_KEY_BYTES} bytes, which is {MIN_KEY_BYTES * 2} hex digits. "
            f"Generate one with: {_GENERATE}"
        )
    return raw


def key_from_environment(environ: Mapping[str, str] | None = None) -> bytes | None:
    """The shared key, if one is configured; ``None`` if the variable is unset.

    Set but unusable is an error rather than a quiet fallback to a private key,
    because the fallback would look like it worked until the first handle
    crossed between processes.
    """
    source = os.environ if environ is None else environ
    text = source.get(KEY_VARIABLE)
    if text is None:
        return None
    return parse_key(text)


class HandleError(Exception):
    """A handle was presented and could not be used.

    Deliberately not a subclass of :class:`ValueError`. The tool layer converts
    a library ``ValueError`` into a generic domain error, and a handle failure
    needs to stay distinguishable from one so the caller can be told the one
    thing that actually helps: open the book again.
    """

    #: Machine-readable discriminator, surfaced as the error ``kind``.
    kind = "handle_invalid"

    def __init__(self, message: str, *, handle: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.handle = handle


class HandleUnrecognised(HandleError):
    """The handle is malformed, was not minted here, or has been altered.

    One class for all three because the server cannot tell them apart, and
    should not pretend to: a failed MAC check says the bytes are not ones this
    key authenticated, and whether that is because they were edited, because
    they came from another server, or because this process restarted since
    minting them is not information the check contains.
    """

    kind = "handle_unrecognised"


class HandleExpired(HandleError):
    """The handle was minted here and its lifetime has run out."""

    kind = "handle_expired"

    def __init__(self, message: str, *, handle: str | None = None, age_s: float) -> None:
        super().__init__(message, handle=handle)
        self.age_s = age_s


class HandleTooLarge(Exception):
    """The payload will not fit in a handle.

    Raised when minting, never when redeeming. It is a statement about what the
    caller asked to store, so it is reported at the point where the caller can
    still do something about it.
    """

    def __init__(self, message: str, *, size: int, limit: int) -> None:
        super().__init__(message)
        self.message = message
        self.size = size
        self.limit = limit


def _b64encode(raw: bytes) -> str:
    """URL-safe base64 without padding, which is not needed and looks like noise."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


class Minter:
    """Mints and redeems handles under one key.

    Without a key one is generated per instance and never leaves the process.
    Passing the same key to several instances -- in separate processes, or the
    same process after a restart -- makes each redeem the others' handles;
    :func:`key_from_environment` is how the command line finds one.
    """

    def __init__(
        self,
        *,
        key: bytes | None = None,
        ttl_s: int = DEFAULT_TTL_S,
        max_chars: int = MAX_HANDLE_CHARS,
    ) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self._key = key if key is not None else secrets.token_bytes(32)
        self.ttl_s = ttl_s
        self.max_chars = max_chars

    def _mac(self, body: bytes) -> bytes:
        return hmac.new(self._key, body, hashlib.sha256).digest()[:MAC_BYTES]

    def mint(self, payload: dict[str, Any], *, now: float | None = None) -> str:
        """Encode ``payload`` into a handle.

        ``payload`` must be JSON-serialisable. It is written with sorted keys and
        no incidental whitespace so that the same book always produces the same
        handle within a lifetime window, which makes the encoding testable and
        keeps a model's context stable across a repeated call.
        """
        issued = time.time() if now is None else now
        envelope = {
            "v": HANDLE_VERSION,
            "iat": round(issued, 3),
            "exp": round(issued + self.ttl_s, 3),
            "payload": payload,
        }
        body = zlib.compress(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8"), 9
        )
        handle = _SEPARATOR.join(
            (HANDLE_VERSION, _b64encode(body), _b64encode(self._mac(body)))
        )
        if len(handle) > self.max_chars:
            raise HandleTooLarge(
                f"this book encodes to a {len(handle)}-character handle, over the "
                f"{self.max_chars}-character limit. The handle carries the book itself "
                "rather than pointing at stored state, so the book has to fit inside it. "
                "Split the positions across two books and aggregate the results.",
                size=len(handle),
                limit=self.max_chars,
            )
        return handle

    def redeem(self, handle: object, *, now: float | None = None) -> dict[str, Any]:
        """Decode and authenticate a handle, returning the payload it carries."""
        if not isinstance(handle, str) or not handle:
            raise HandleUnrecognised("a handle must be a non-empty string")

        parts = handle.split(_SEPARATOR)
        if len(parts) != 3:
            raise HandleUnrecognised(
                "this is not a handle this server minted: a handle has three "
                f"{_SEPARATOR!r}-separated parts, this one has {len(parts)}",
                handle=handle,
            )
        version, body_text, mac_text = parts
        if version != HANDLE_VERSION:
            raise HandleUnrecognised(
                f"handle version {version!r} is not one this server mints "
                f"(it mints {HANDLE_VERSION!r})",
                handle=handle,
            )

        try:
            body = _b64decode(body_text)
            presented = _b64decode(mac_text)
        except (binascii.Error, ValueError) as exc:
            raise HandleUnrecognised(
                "handle is not decodable; it has been truncated or altered in transit",
                handle=handle,
            ) from exc

        # Constant time, and before decompression: a MAC checked after parsing
        # would mean feeding unauthenticated bytes to the decompressor, which is
        # how a handle turns into a way of spending the server's memory.
        if not hmac.compare_digest(presented, self._mac(body)):
            raise HandleUnrecognised(
                "handle failed its integrity check. It was altered, it came from a "
                "different server, or this server has restarted since it was minted. "
                "Open the book again to get a fresh handle.",
                handle=handle,
            )

        try:
            envelope = json.loads(zlib.decompress(body).decode("utf-8"))
        except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            # Unreachable for a handle this server minted, since the MAC has
            # already matched. Kept because "authenticated" is not "well-formed"
            # if the key is ever shared with something that mints differently.
            raise HandleUnrecognised(
                "handle carried a payload this server cannot read", handle=handle
            ) from exc

        if not isinstance(envelope, dict) or envelope.get("v") != HANDLE_VERSION:
            raise HandleUnrecognised("handle payload is not in a known shape", handle=handle)

        moment = time.time() if now is None else now
        expiry = envelope.get("exp")
        issued = envelope.get("iat")
        if not isinstance(expiry, int | float) or not isinstance(issued, int | float):
            raise HandleUnrecognised("handle payload has no usable lifetime", handle=handle)
        if moment > expiry:
            age = moment - float(issued)
            raise HandleExpired(
                f"this handle was minted {age:.0f}s ago and expired "
                f"{moment - float(expiry):.0f}s ago; handles last {self.ttl_s}s. "
                "Open the book again to get a fresh one. The positions are unchanged — "
                "it is the handle that has a lifetime, not the book.",
                handle=handle,
                age_s=age,
            )

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise HandleUnrecognised("handle payload is not an object", handle=handle)
        return payload


__all__ = [
    "DEFAULT_TTL_S",
    "HANDLE_VERSION",
    "KEY_VARIABLE",
    "MAC_BYTES",
    "MAX_HANDLE_CHARS",
    "MIN_KEY_BYTES",
    "BadHandleKey",
    "HandleError",
    "HandleExpired",
    "HandleTooLarge",
    "HandleUnrecognised",
    "Minter",
    "key_from_environment",
    "parse_key",
]
