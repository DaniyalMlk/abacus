"""Handles: round trip, integrity, lifetime and the size bound.

The tests here are mostly adversarial, which is the point. A handle is the one
thing this server hands out and takes back, and everything it protects against
is invisible when the input is well behaved.
"""

from __future__ import annotations

import time

import pytest

from abacus.handles import (
    HANDLE_VERSION,
    MAC_BYTES,
    HandleExpired,
    HandleTooLarge,
    HandleUnrecognised,
    Minter,
    _b64decode,
    _b64encode,
)

PAYLOAD = {
    "spot": 100.0,
    "rate": 0.04,
    "legs": [
        {"instrument": "call", "quantity": 3.0, "strike": 105.0, "time": 0.5, "vol": 0.2},
        {"instrument": "underlying", "quantity": -1.5},
    ],
}


@pytest.fixture
def minter() -> Minter:
    return Minter(key=b"a fixed key, so these tests do not depend on entropy")


# -- round trip -----------------------------------------------------------


def test_a_handle_round_trips_to_the_payload_it_carried(minter: Minter) -> None:
    assert minter.redeem(minter.mint(PAYLOAD)) == PAYLOAD


def test_a_handle_carries_the_version_it_was_minted_under(minter: Minter) -> None:
    assert minter.mint(PAYLOAD).split(".")[0] == HANDLE_VERSION


def test_the_same_payload_at_the_same_moment_gives_the_same_handle(minter: Minter) -> None:
    # Canonical JSON with sorted keys. Worth pinning: a handle that changed on
    # every call would be a fresh string in a model's context each turn, which
    # is both unnecessary and a prompt cache miss.
    moment = 1_700_000_000.0
    assert minter.mint(PAYLOAD, now=moment) == minter.mint(dict(PAYLOAD), now=moment)


def test_key_order_in_the_payload_does_not_change_the_handle(minter: Minter) -> None:
    moment = 1_700_000_000.0
    reversed_keys = {k: PAYLOAD[k] for k in reversed(list(PAYLOAD))}
    assert minter.mint(reversed_keys, now=moment) == minter.mint(PAYLOAD, now=moment)


def test_floats_survive_the_round_trip_exactly(minter: Minter) -> None:
    payload = {"awkward": [0.1 + 0.2, 1e-17, -1.5e300, 3.141592653589793]}
    assert minter.redeem(minter.mint(payload)) == payload


def test_non_ascii_labels_survive_the_round_trip(minter: Minter) -> None:
    payload = {"label": "Écart de volatilité — 日経225"}
    assert minter.redeem(minter.mint(payload)) == payload


# -- integrity ------------------------------------------------------------


def test_altering_any_byte_of_the_body_is_detected(minter: Minter) -> None:
    # Every byte, not a sampled one. A MAC that happened to be computed over a
    # prefix would pass a spot check and fail here.
    #
    # The tampering is done on the decoded bytes rather than on the characters,
    # because base64 is not injective at the tail: a 16-byte value ends in a
    # character carrying two significant bits, and several distinct characters
    # decode to the same bytes. Editing text would therefore produce an edit
    # that is sometimes not an edit at all, which is a flaky test rather than a
    # weak MAC.
    handle = minter.mint(PAYLOAD, now=1_700_000_000.0)
    version, body, mac = handle.split(".")
    raw = _b64decode(body)
    for index in range(len(raw)):
        edited = bytearray(raw)
        edited[index] ^= 0x01
        with pytest.raises(HandleUnrecognised):
            minter.redeem(f"{version}.{_b64encode(bytes(edited))}.{mac}")


def test_altering_any_byte_of_the_mac_is_detected(minter: Minter) -> None:
    handle = minter.mint(PAYLOAD, now=1_700_000_000.0)
    version, body, mac = handle.split(".")
    raw = _b64decode(mac)
    assert len(raw) == MAC_BYTES
    for index in range(len(raw)):
        edited = bytearray(raw)
        edited[index] ^= 0x01
        with pytest.raises(HandleUnrecognised):
            minter.redeem(f"{version}.{body}.{_b64encode(bytes(edited))}")


def test_a_mac_of_the_wrong_length_is_detected(minter: Minter) -> None:
    # Dropping a byte rather than changing one, since compare_digest is only
    # equal for equal lengths and a truncated MAC must not slip through as a
    # prefix match.
    handle = minter.mint(PAYLOAD, now=1_700_000_000.0)
    version, body, mac = handle.split(".")
    raw = _b64decode(mac)
    with pytest.raises(HandleUnrecognised):
        minter.redeem(f"{version}.{body}.{_b64encode(raw[:-1])}")


def test_truncating_a_handle_is_detected(minter: Minter) -> None:
    handle = minter.mint(PAYLOAD)
    with pytest.raises(HandleUnrecognised):
        minter.redeem(handle[: len(handle) - 4])


def test_a_handle_from_another_key_is_not_recognised() -> None:
    # The multi-process case, and the restart case, are the same case: a key
    # that did not mint this handle cannot redeem it.
    handle = Minter(key=b"one key").mint(PAYLOAD)
    with pytest.raises(HandleUnrecognised, match="integrity check"):
        Minter(key=b"another key").redeem(handle)


def test_the_version_prefix_cannot_be_rewritten(minter: Minter) -> None:
    handle = minter.mint(PAYLOAD)
    _, body, mac = handle.split(".")
    with pytest.raises(HandleUnrecognised, match="version"):
        minter.redeem(f"abk9.{body}.{mac}")


@pytest.mark.parametrize(
    "handle",
    ["", "not-a-handle", "abk1.only-two-parts", "a.b.c.d", "abk1..", "..."],
)
def test_a_malformed_handle_is_refused(minter: Minter, handle: str) -> None:
    with pytest.raises(HandleUnrecognised):
        minter.redeem(handle)


@pytest.mark.parametrize("handle", [None, 17, 3.5, True, [], {}])
def test_a_handle_that_is_not_a_string_is_refused(minter: Minter, handle: object) -> None:
    with pytest.raises(HandleUnrecognised, match="non-empty string"):
        minter.redeem(handle)


def test_a_body_that_is_not_base64_is_refused(minter: Minter) -> None:
    handle = minter.mint(PAYLOAD)
    _, _, mac = handle.split(".")
    with pytest.raises(HandleUnrecognised):
        minter.redeem(f"{HANDLE_VERSION}.not valid base64 at all!.{mac}")


# -- lifetime -------------------------------------------------------------


def test_a_handle_is_live_within_its_window() -> None:
    minter = Minter(key=b"k", ttl_s=100)
    issued = 1_700_000_000.0
    handle = minter.mint(PAYLOAD, now=issued)
    assert minter.redeem(handle, now=issued + 99.0) == PAYLOAD


def test_a_handle_is_still_live_at_the_instant_it_expires() -> None:
    # The boundary is inclusive. Which way it falls matters less than its being
    # decided, since an off-by-one here is a handle that fails a nanosecond
    # early for reasons nobody would find.
    minter = Minter(key=b"k", ttl_s=100)
    issued = 1_700_000_000.0
    handle = minter.mint(PAYLOAD, now=issued)
    assert minter.redeem(handle, now=issued + 100.0) == PAYLOAD


def test_a_handle_past_its_window_is_expired_not_unrecognised() -> None:
    minter = Minter(key=b"k", ttl_s=100)
    issued = 1_700_000_000.0
    handle = minter.mint(PAYLOAD, now=issued)
    with pytest.raises(HandleExpired) as caught:
        minter.redeem(handle, now=issued + 100.001)
    # The distinction is the whole point: this one was ours and simply aged out,
    # so the caller is told to reopen rather than to suspect its own plumbing.
    assert not isinstance(caught.value, HandleUnrecognised)


def test_an_expired_handle_reports_its_age_and_what_to_do() -> None:
    minter = Minter(key=b"k", ttl_s=60)
    issued = 1_700_000_000.0
    handle = minter.mint(PAYLOAD, now=issued)
    with pytest.raises(HandleExpired) as caught:
        minter.redeem(handle, now=issued + 300.0)
    assert caught.value.age_s == pytest.approx(300.0, abs=1e-3)
    assert caught.value.handle == handle
    assert "Open the book again" in caught.value.message


def test_expiry_is_measured_against_the_wall_clock_by_default() -> None:
    minter = Minter(key=b"k", ttl_s=1)
    handle = minter.mint(PAYLOAD, now=time.time() - 10.0)
    with pytest.raises(HandleExpired):
        minter.redeem(handle)


# -- bounds ---------------------------------------------------------------


def test_a_payload_that_will_not_fit_is_refused_at_mint_time(minter: Minter) -> None:
    # Refused where the caller can still act on it, rather than minted and
    # discovered later by whoever tries to carry it.
    oversized = {"legs": [{"label": f"leg {i} " + "x" * 64, "q": i * 1.5} for i in range(4000)]}
    with pytest.raises(HandleTooLarge) as caught:
        minter.mint(oversized)
    assert caught.value.size > caught.value.limit == minter.max_chars
    assert "Split the positions" in caught.value.message


def test_the_size_limit_is_configurable() -> None:
    tight = Minter(key=b"k", max_chars=32)
    with pytest.raises(HandleTooLarge):
        tight.mint(PAYLOAD)


def test_compression_means_a_repetitive_book_still_fits(minter: Minter) -> None:
    # The bound is on the encoded handle, not the payload, and books are
    # repetitive by nature — sixty near-identical legs is a normal book.
    repetitive = {
        "spot": 100.0,
        "rate": 0.04,
        "legs": [
            {"instrument": "call", "quantity": 1.0, "strike": 100.0 + i, "time": 0.5, "vol": 0.2}
            for i in range(64)
        ],
    }
    assert minter.redeem(minter.mint(repetitive)) == repetitive


@pytest.mark.parametrize("ttl", [0, -1])
def test_a_non_positive_lifetime_is_rejected(ttl: int) -> None:
    with pytest.raises(ValueError, match="ttl_s"):
        Minter(ttl_s=ttl)


def test_a_non_positive_size_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        Minter(max_chars=0)


def test_two_minters_without_an_explicit_key_do_not_share_one() -> None:
    handle = Minter().mint(PAYLOAD)
    with pytest.raises(HandleUnrecognised):
        Minter().redeem(handle)
