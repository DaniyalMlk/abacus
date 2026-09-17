"""Message parsing, `_meta` handling and result construction."""

from __future__ import annotations

from typing import Any

import pytest

from abacus.errors import InvalidParams, InvalidRequest, UnsupportedProtocolVersionError
from abacus.protocol import (
    META_CLIENT_CAPABILITIES,
    META_CLIENT_INFO,
    META_LOG_LEVEL,
    META_PROGRESS_TOKEN,
    META_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    Capabilities,
    Implementation,
    Notification,
    Request,
    RequestMeta,
    cacheable,
    complete,
    input_required,
    is_reserved_meta_key,
    is_valid_meta_key,
    parse_message,
    split_meta_key,
)

from .conftest import meta, request


# -- `_meta` key naming ---------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "progressToken",
        "simple",
        "with-hyphen",
        "with_underscore",
        "with.dot",
        "a",
        "0start",
        "io.modelcontextprotocol/protocolVersion",
        "com.example/key",
        "com.example.sub/key",
        "x-y-z.q/name",
        "traceparent",
        "com.example/",  # an empty name is permitted
    ],
)
def test_valid_meta_keys(key: str) -> None:
    assert is_valid_meta_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "-leading-hyphen",
        "trailing-hyphen-",
        ".leading-dot",
        "trailing-dot.",
        "has space",
        "9bad.prefix/name",  # a label must start with a letter
        "bad-.prefix/name",  # a label must end with a letter or digit
        "/name",  # an empty prefix label
        "a..b/name",  # an empty interior label
    ],
)
def test_invalid_meta_keys(key: str) -> None:
    assert not is_valid_meta_key(key)


def test_split_meta_key_uses_only_the_first_slash() -> None:
    assert split_meta_key("com.example/a/b") == ("com.example", "a/b")
    assert split_meta_key("bare") == (None, "bare")


@pytest.mark.parametrize(
    "key",
    [
        "io.modelcontextprotocol/protocolVersion",
        "dev.mcp/anything",
        "org.modelcontextprotocol.api/thing",
        "com.mcp.tools/thing",
        "traceparent",
        "tracestate",
        "baggage",
    ],
)
def test_reserved_meta_prefixes(key: str) -> None:
    assert is_reserved_meta_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "com.example.mcp/thing",  # second label is `example`, so not reserved
        "com.example/thing",
        "mcp/thing",  # a single label has no second label to reserve on
        "modelcontextprotocol/thing",
        "bare",
    ],
)
def test_unreserved_meta_prefixes(key: str) -> None:
    assert not is_reserved_meta_key(key)


# -- request metadata -----------------------------------------------------


def test_meta_round_trips_every_field() -> None:
    parsed = RequestMeta.from_params(
        {
            "_meta": meta(
                client_capabilities={"elicitation": {}, "extensions": {"io.example/x": {}}},
                client_info={"name": "probe", "version": "2.0.0", "title": "Probe"},
                **{META_LOG_LEVEL: "warning", META_PROGRESS_TOKEN: "tok-1"},
            )
        }
    )
    assert parsed.protocol_version == PROTOCOL_VERSION
    assert parsed.client_info == Implementation("probe", "2.0.0", "Probe")
    assert parsed.client_capabilities.declares("elicitation")
    assert parsed.client_capabilities.extensions() == {"io.example/x": {}}
    assert parsed.log_level == "warning"
    assert parsed.progress_token == "tok-1"
    assert parsed.wants_logs


def test_absent_log_level_means_no_logs_are_emitted() -> None:
    # `logging/setLevel` was removed; a request that does not ask for logs must
    # not receive `notifications/message` at all.
    parsed = RequestMeta.from_params({"_meta": meta()})
    assert parsed.log_level is None
    assert not parsed.wants_logs


def test_missing_meta_is_rejected() -> None:
    with pytest.raises(InvalidParams, match="missing _meta"):
        RequestMeta.from_params({})


def test_missing_protocol_version_is_rejected() -> None:
    block = meta()
    del block[META_PROTOCOL_VERSION]
    with pytest.raises(InvalidParams, match="protocolVersion is required"):
        RequestMeta.from_params({"_meta": block})


def test_missing_client_capabilities_is_rejected() -> None:
    # Required even when empty: the server must be able to tell "declared
    # nothing" from "did not say", and only the first permits a capability check.
    block = meta()
    del block[META_CLIENT_CAPABILITIES]
    with pytest.raises(InvalidParams, match="clientCapabilities is required"):
        RequestMeta.from_params({"_meta": block})


def test_empty_client_capabilities_is_accepted() -> None:
    parsed = RequestMeta.from_params({"_meta": meta(client_capabilities={})})
    assert parsed.client_capabilities.to_dict() == {}


@pytest.mark.parametrize("bad", [[], "x", 3, None])
def test_client_capabilities_must_be_an_object(bad: Any) -> None:
    block = meta()
    block[META_CLIENT_CAPABILITIES] = bad
    with pytest.raises(InvalidParams):
        RequestMeta.from_params({"_meta": block})


def test_client_info_must_be_well_formed_when_present() -> None:
    block = meta(client_info={"name": "probe"})
    with pytest.raises(InvalidParams, match="version must be a non-empty string"):
        RequestMeta.from_params({"_meta": block})


def test_client_info_is_optional() -> None:
    assert RequestMeta.from_params({"_meta": meta()}).client_info is None


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(InvalidParams, match="must be one of"):
        RequestMeta.from_params({"_meta": meta(**{META_LOG_LEVEL: "loud"})})


def test_malformed_meta_key_is_rejected() -> None:
    with pytest.raises(InvalidParams, match="not a valid key name"):
        RequestMeta.from_params({"_meta": meta(**{"has space": 1})})


@pytest.mark.parametrize("token", [True, False, 1.5, [], {}])
def test_progress_token_must_be_a_string_or_integer(token: Any) -> None:
    # `True` is an `int` in Python; a JSON boolean is not a valid token, so the
    # check has to exclude it explicitly.
    with pytest.raises(InvalidParams, match="progressToken"):
        RequestMeta.from_params({"_meta": meta(**{META_PROGRESS_TOKEN: token})})


def test_integer_progress_token_is_accepted() -> None:
    parsed = RequestMeta.from_params({"_meta": meta(**{META_PROGRESS_TOKEN: 7})})
    assert parsed.progress_token == 7


def test_require_version_accepts_a_supported_version() -> None:
    RequestMeta.from_params({"_meta": meta()}).require_version()


def test_require_version_rejects_an_unsupported_version() -> None:
    parsed = RequestMeta.from_params({"_meta": meta(version="2025-11-25")})
    with pytest.raises(UnsupportedProtocolVersionError) as caught:
        parsed.require_version()
    assert caught.value.data == {"supported": [PROTOCOL_VERSION], "requested": "2025-11-25"}


# -- capabilities ---------------------------------------------------------


def test_capabilities_declares_supports_nested_paths() -> None:
    caps = Capabilities.from_dict({"elicitation": {"form": {}}}, where="caps")
    assert caps.declares("elicitation")
    assert caps.declares("elicitation.form")
    assert not caps.declares("elicitation.url")
    assert not caps.declares("roots")


def test_capabilities_declared_as_null_still_count_as_declared() -> None:
    caps = Capabilities.from_dict({"tools": None}, where="caps")
    assert caps.declares("tools")
    assert not caps.declares("tools.listChanged")


def test_capabilities_keep_names_they_do_not_recognise() -> None:
    caps = Capabilities.from_dict({"somethingNew": {}}, where="caps")
    assert caps.declares("somethingNew")


# -- message parsing ------------------------------------------------------


def test_parse_request() -> None:
    parsed = parse_message(request("test/echo", {"value": 3}))
    assert isinstance(parsed, Request)
    assert (parsed.id, parsed.method) == (1, "test/echo")
    assert parsed.argument("value") == 3


def test_parse_notification_needs_no_meta() -> None:
    parsed = parse_message({"jsonrpc": "2.0", "method": "notifications/progress"})
    assert isinstance(parsed, Notification)
    assert parsed.method == "notifications/progress"


def test_string_ids_are_accepted() -> None:
    parsed = parse_message(request("test/echo", ident="discover-1"))
    assert isinstance(parsed, Request)
    assert parsed.id == "discover-1"


def test_null_id_is_rejected() -> None:
    # Legal in base JSON-RPC, forbidden by MCP.
    payload = request("test/echo")
    payload["id"] = None
    with pytest.raises(InvalidRequest, match="must not be null"):
        parse_message(payload)


@pytest.mark.parametrize("ident", [True, 1.5, [], {}])
def test_non_scalar_ids_are_rejected(ident: Any) -> None:
    payload = request("test/echo")
    payload["id"] = ident
    with pytest.raises(InvalidRequest, match="string or an integer"):
        parse_message(payload)


def test_batches_are_rejected() -> None:
    with pytest.raises(InvalidRequest, match="batch"):
        parse_message([request("test/echo")])


def test_positional_params_are_rejected() -> None:
    with pytest.raises(InvalidRequest, match="positional"):
        parse_message({"jsonrpc": "2.0", "id": 1, "method": "m", "params": [1, 2]})


@pytest.mark.parametrize(
    "payload",
    [
        {"id": 1, "method": "m"},  # no jsonrpc member
        {"jsonrpc": "1.0", "id": 1, "method": "m"},
        {"jsonrpc": "2.0", "id": 1},  # no method
        {"jsonrpc": "2.0", "id": 1, "method": ""},
        {"jsonrpc": "2.0", "id": 1, "method": 5},
        "not an object",
        7,
    ],
)
def test_malformed_messages_are_rejected(payload: Any) -> None:
    with pytest.raises(InvalidRequest):
        parse_message(payload)


def test_null_params_are_treated_as_absent_and_then_fail_on_meta() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "m", "params": None}
    with pytest.raises(InvalidParams, match="missing _meta"):
        parse_message(payload)


# -- results --------------------------------------------------------------


def test_complete_marks_the_result_type() -> None:
    assert complete(value=1) == {"resultType": "complete", "value": 1}


def test_input_required_carries_requests_and_state() -> None:
    result = input_required(
        {"ask": {"method": "elicitation/create", "params": {}}}, request_state="abc"
    )
    assert result["resultType"] == "input_required"
    assert result["inputRequests"]["ask"]["method"] == "elicitation/create"
    assert result["requestState"] == "abc"


def test_input_required_omits_absent_state() -> None:
    assert "requestState" not in input_required({"ask": {}})


def test_cacheable_attaches_freshness_hints() -> None:
    result = cacheable(complete(x=1), ttl_ms=1000, cache_scope="private")
    assert result["ttlMs"] == 1000
    assert result["cacheScope"] == "private"
    assert result["resultType"] == "complete"


@pytest.mark.parametrize(
    ("ttl", "scope"),
    [(-1, "public"), (0, "shared"), (0, "PUBLIC")],
)
def test_cacheable_rejects_nonsense_hints(ttl: int, scope: str) -> None:
    with pytest.raises(ValueError):
        cacheable(complete(), ttl_ms=ttl, cache_scope=scope)
