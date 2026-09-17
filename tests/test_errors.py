"""The error-code allocation policy, which is enforced rather than documented."""

from __future__ import annotations

import pytest

from abacus.errors import (
    HEADER_MISMATCH,
    INVALID_PARAMS,
    MISSING_REQUIRED_CLIENT_CAPABILITY,
    RESERVED_RANGE,
    RETIRED_CODES,
    SPEC_DEFINED_CODES,
    UNSUPPORTED_PROTOCOL_VERSION,
    InvalidParams,
    MethodNotFound,
    MissingRequiredClientCapabilityError,
    ProtocolError,
    UnsupportedProtocolVersionError,
    is_reserved,
)


def test_spec_codes_have_the_values_the_specification_assigns() -> None:
    assert HEADER_MISMATCH == -32020
    assert MISSING_REQUIRED_CLIENT_CAPABILITY == -32021
    assert UNSUPPORTED_PROTOCOL_VERSION == -32022


@pytest.mark.parametrize("code", sorted(SPEC_DEFINED_CODES))
def test_every_spec_defined_code_lies_inside_the_reserved_range(code: int) -> None:
    assert is_reserved(code)


@pytest.mark.parametrize("code", [-32019, -32000, -32100, -32603, 0, 42])
def test_codes_outside_the_reserved_range_are_not_reserved(code: int) -> None:
    assert not is_reserved(code)


def test_reserved_range_bounds() -> None:
    low, high = RESERVED_RANGE
    assert (low, high) == (-32099, -32020)
    assert is_reserved(low) and is_reserved(high)
    assert not is_reserved(low - 1) and not is_reserved(high + 1)


def test_an_undefined_code_in_the_reserved_range_is_refused() -> None:
    # -32050 sits in the range the specification keeps for itself but is not a
    # code it defines, so emitting it would be a conformance violation.
    with pytest.raises(ValueError, match="reserved for the MCP specification"):
        ProtocolError(-32050, "invented")


def test_codes_below_the_reserved_range_remain_available() -> None:
    # -32019 and below are the legacy sub-range: grandfathered, not forbidden.
    error = ProtocolError(-32019, "legacy code")
    assert error.code == -32019


@pytest.mark.parametrize("code", sorted(RETIRED_CODES))
def test_codes_retired_by_earlier_revisions_are_refused(code: int) -> None:
    with pytest.raises(ValueError, match="earlier protocol revision"):
        ProtocolError(code, "retired")


def test_resource_not_found_now_uses_invalid_params() -> None:
    # It was -32002 before this revision; that code is now retired.
    assert InvalidParams("no such resource").code == INVALID_PARAMS


def test_error_renders_without_data_when_none_is_given() -> None:
    assert ProtocolError(-32603, "boom").to_dict() == {"code": -32603, "message": "boom"}


def test_unsupported_version_error_carries_the_supported_list() -> None:
    error = UnsupportedProtocolVersionError("1900-01-01", ["2026-07-28"])
    assert error.to_dict() == {
        "code": -32022,
        "message": "Unsupported protocol version",
        "data": {"supported": ["2026-07-28"], "requested": "1900-01-01"},
    }


def test_missing_capability_error_names_what_was_missing() -> None:
    error = MissingRequiredClientCapabilityError(["elicitation", "roots"])
    assert error.data == {"requiredCapabilities": ["elicitation", "roots"]}
    assert "elicitation, roots" in error.message


def test_method_not_found_names_the_method() -> None:
    error = MethodNotFound("tools/nope")
    assert error.code == -32601
    assert error.data == {"method": "tools/nope"}
