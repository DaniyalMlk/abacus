"""One key per server, and a way to give several servers the same one.

A handle carries its book rather than pointing at stored state, and the reason
given for that is that a stored book breaks across processes. That reason only
holds if the processes can verify each other's handles, which needs them to
share a key. Each server used to draw its own -- three of them, one per family
of tool -- so a handle minted by one worker was refused by the next, and by the
same worker after a restart.

``ABACUS_HANDLE_KEY`` gives every process the same key. These tests check the
key is parsed strictly and never echoed, that one server signs every family
with one key, and -- over stdio, between two real processes -- that a handle
opened on one is redeemed by the other exactly when they share a key.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
from typing import Any

import pytest

from abacus.analytics import default_registry
from abacus.cli import build_server, main
from abacus.client import Client, InProcessTransport, StdioTransport
from abacus.handles import (
    KEY_VARIABLE,
    MIN_KEY_BYTES,
    BadHandleKey,
    Minter,
    key_from_environment,
    parse_key,
)

KEY = "5a" * MIN_KEY_BYTES
LEG = {"instrument": "call", "quantity": 1.0, "strike": 100.0, "time": 0.5, "vol": 0.2}
BOOK = {"spot": 100.0, "rate": 0.04, "legs": [LEG]}


# -- parsing ------------------------------------------------------------------


def test_a_hex_key_parses_to_its_bytes() -> None:
    assert parse_key(KEY) == bytes.fromhex(KEY)


def test_surrounding_whitespace_is_ignored() -> None:
    """A key read out of a file arrives with its newline."""
    assert parse_key(f"  {KEY}\n") == parse_key(KEY)


def test_the_shortest_accepted_key_is_the_size_the_server_draws() -> None:
    assert parse_key("00" * MIN_KEY_BYTES) == bytes(MIN_KEY_BYTES)
    with pytest.raises(BadHandleKey, match=f"31-byte key; it must be at least {MIN_KEY_BYTES}"):
        parse_key("00" * (MIN_KEY_BYTES - 1))


@pytest.mark.parametrize(
    ("text", "complaint"),
    [
        ("", "set but empty"),
        ("   \n", "set but empty"),
        ("not hexadecimal at all, but long enough to be a key of sorts", "not all hex digits"),
        ("ab" * MIN_KEY_BYTES + "g", "not all hex digits"),
        ("abc", "not all hex digits"),
    ],
)
def test_an_unusable_key_is_refused(text: str, complaint: str) -> None:
    with pytest.raises(BadHandleKey, match=complaint):
        parse_key(text)


@pytest.mark.parametrize(
    "text",
    [
        "ab" * (MIN_KEY_BYTES - 1),
        "secret-" + "ab" * MIN_KEY_BYTES,
        "ab" * MIN_KEY_BYTES + "z",
    ],
)
def test_a_refusal_never_contains_the_key(text: str) -> None:
    """The message goes to a terminal and on into logs: shape only, never value."""
    with pytest.raises(BadHandleKey) as caught:
        parse_key(text)
    message = str(caught.value)
    assert text not in message
    assert text[:16] not in message


def test_an_unset_variable_means_no_shared_key() -> None:
    assert key_from_environment({}) is None


def test_a_set_variable_is_parsed() -> None:
    assert key_from_environment({KEY_VARIABLE: KEY}) == bytes.fromhex(KEY)


def test_a_set_but_unusable_variable_is_an_error_not_a_fallback() -> None:
    """Falling back to a private key would look like success until a handle moved."""
    with pytest.raises(BadHandleKey):
        key_from_environment({KEY_VARIABLE: "nonsense"})


def test_the_process_environment_is_read_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_VARIABLE, KEY)
    assert key_from_environment() == bytes.fromhex(KEY)
    monkeypatch.delenv(KEY_VARIABLE)
    assert key_from_environment() is None


# -- in process -----------------------------------------------------------------


def open_and_describe(opener: Client, reader: Client) -> dict[str, Any]:
    opened = opener.call_tool("open_position_book", BOOK).result
    assert opened is not None and opened["isError"] is False, opened
    handle = opened["structuredContent"]["handle"]
    read = reader.call_tool("describe_position_book", {"handle": handle}).result
    assert read is not None
    return read


def in_process(minter: Minter | None) -> Client:
    return Client(InProcessTransport(build_server(minter=minter)))


def test_two_servers_with_one_key_read_each_other_s_handles() -> None:
    key = secrets.token_bytes(MIN_KEY_BYTES)
    read = open_and_describe(in_process(Minter(key=key)), in_process(Minter(key=key)))
    assert read["isError"] is False
    assert read["structuredContent"]["book"]["legs"] == [LEG]


def test_two_servers_with_their_own_keys_do_not() -> None:
    read = open_and_describe(in_process(None), in_process(None))
    assert read["isError"] is True
    assert read["structuredContent"]["error"]["kind"] == "handle_unrecognised"


def test_one_server_signs_every_family_with_one_key() -> None:
    """A handle from any family verifies under the minter the registry was given.

    Redeeming through a bare minter -- no tool, no kind check -- isolates the
    key: a family that drew its own would fail here whatever its kind.
    """
    key = secrets.token_bytes(MIN_KEY_BYTES)
    registry = default_registry(minter=Minter(key=key))
    returns = [[0.01 * ((i * 7 + j * 3) % 5 - 2) for j in range(3)] for i in range(40)]
    quotes = {
        "reference": "2021-01-05",
        "basis": "ACT_365F",
        "deposits": [{"maturity": "2021-07-05", "rate": 0.002, "basis": "ACT_360"}],
    }
    minted = {
        "open_position_book": BOOK,
        "estimate_return_moments": {"returns": returns, "periodsPerYear": 252},
        "bootstrap_discount_curve": quotes,
    }
    outsider = Minter(key=key)
    for name, args in minted.items():
        result = registry.call(name, args)
        assert result["isError"] is False, (name, result["structuredContent"])
        outsider.redeem(result["structuredContent"]["handle"])


# -- over the wire, between two processes -------------------------------------


def launched() -> StdioTransport:
    return StdioTransport([sys.executable, "-m", "abacus.cli", "stdio"])


def test_a_handle_crosses_between_two_processes_that_share_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The restart case too: the second process is a server that was not there."""
    monkeypatch.setenv(KEY_VARIABLE, secrets.token_hex(MIN_KEY_BYTES))
    with launched() as first, launched() as second:
        read = open_and_describe(Client(first), Client(second))
    assert read["isError"] is False, read["structuredContent"]
    assert read["structuredContent"]["book"]["spot"] == BOOK["spot"]


def test_without_a_shared_key_the_second_process_refuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(KEY_VARIABLE, raising=False)
    with launched() as first, launched() as second:
        read = open_and_describe(Client(first), Client(second))
    assert read["isError"] is True
    assert read["structuredContent"]["error"]["kind"] == "handle_unrecognised"


@pytest.mark.parametrize("command", ["stdio", "http"])
def test_a_bad_key_stops_the_server_before_it_serves(command: str) -> None:
    """Exit 2 with the shape of the problem on stderr, and nothing on stdout.

    The HTTP server is given a port it could bind, so a run that got past the
    key would hang serving rather than exit; the timeout is what would say so.
    """
    secret = "secret-" + "cd" * MIN_KEY_BYTES
    done = subprocess.run(
        [sys.executable, "-m", "abacus.cli", command],
        env={**os.environ, KEY_VARIABLE: secret},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert done.returncode == 2
    assert done.stdout == ""
    assert done.stderr.startswith(f"abacus: {KEY_VARIABLE} must be hexadecimal")
    assert secret not in done.stderr


def test_the_tool_listing_ignores_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the serving commands mint handles, so only they read the key."""
    monkeypatch.setenv(KEY_VARIABLE, "nonsense")
    assert main(["tools"]) == 0
