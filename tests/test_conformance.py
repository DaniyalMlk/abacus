"""The conformance suite, tested against servers with known defects.

A suite that passes a healthy server proves very little; what proves it works is
that it fails a broken one, and fails on the right check. So most of this file
takes the real server, injects one specific defect into its responses, and
asserts that exactly the check written for that requirement turns red.

The defects are the ones a server written from stale guidance would actually
have: the handshake still implemented, `resultType` missing, a retired error
code, an execution failure reported as a JSON-RPC error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from abacus.cli import build_server
from abacus.client import Client, InProcessTransport
from abacus.conformance import CHECKS, Report, run
from abacus.protocol import META_SERVER_INFO, complete
from abacus.server import Server

Mutation = Callable[[Any, Any], Any]

#: Checks that cannot run without a Streamable HTTP endpoint. They are skipped,
#: not failed — a server reached over stdio is not required to have one.
HTTP_CHECKS = {check.name for check in CHECKS if check.scope == "http"}


class Defective:
    """The real server with one defect injected into its responses.

    A wrapper rather than a bespoke server, so the only difference between this
    and a conforming server is the defect under test. A hand-written stub would
    differ in a dozen other ways and the checks would have nothing to isolate.
    """

    def __init__(self, mutate: Mutation, server: Server | None = None) -> None:
        self.server = server if server is not None else build_server()
        self.mutate = mutate

    def handle(self, payload: Any) -> Any:
        return self.mutate(payload, self.server.handle(payload))


def report_for(mutate: Mutation | None = None) -> Report:
    server: Any = build_server() if mutate is None else Defective(mutate)
    return run(Client(InProcessTransport(server)))


def failing(report: Report) -> set[str]:
    return {outcome.name for outcome in report.failures}


def method_of(payload: Any) -> str | None:
    return payload.get("method") if isinstance(payload, dict) else None


# -- the healthy case ------------------------------------------------------


def test_a_conforming_server_passes_every_check_it_can_run() -> None:
    report = report_for()
    assert failing(report) == set()
    # Everything skipped is skipped for a stated reason, never silently.
    assert all(outcome.detail for outcome in report.skipped)
    assert {outcome.name for outcome in report.skipped} >= HTTP_CHECKS


def test_the_suite_reports_a_requirement_alongside_every_check() -> None:
    report = report_for()
    assert len(report.outcomes) == len(CHECKS)
    for outcome in report.outcomes:
        assert outcome.requirement
        assert outcome.scope in ("core", "http")
        assert outcome.status in ("pass", "fail", "skip")


def test_a_healthy_report_is_ok_and_renders() -> None:
    report = report_for()
    assert report.ok is True
    rendered = report.render()
    assert "0 failed" in rendered
    assert report.to_dict()["ok"] is True


# -- one defect at a time --------------------------------------------------


def test_a_missing_result_type_is_caught() -> None:
    def drop_result_type(_payload: Any, response: Any) -> Any:
        if isinstance(response, dict) and isinstance(response.get("result"), dict):
            response["result"].pop("resultType", None)
        return response

    assert "result/typed-and-attributed" in failing(report_for(drop_result_type))


def test_an_invented_result_type_is_caught() -> None:
    def invent(_payload: Any, response: Any) -> Any:
        if isinstance(response, dict) and isinstance(response.get("result"), dict):
            response["result"]["resultType"] = "partial"
        return response

    assert "result/typed-and-attributed" in failing(report_for(invent))


def test_a_server_that_does_not_identify_itself_is_caught() -> None:
    # There is no handshake in which a server could have named itself once, so
    # it has to do so on every result.
    def strip_identity(_payload: Any, response: Any) -> Any:
        if isinstance(response, dict) and isinstance(response.get("result"), dict):
            meta = response["result"].get("_meta")
            if isinstance(meta, dict):
                meta.pop(META_SERVER_INFO, None)
        return response

    assert "result/typed-and-attributed" in failing(report_for(strip_identity))


def test_a_listing_without_cache_hints_is_caught() -> None:
    def strip_hints(payload: Any, response: Any) -> Any:
        if method_of(payload) == "tools/list" and isinstance(response.get("result"), dict):
            response["result"].pop("ttlMs", None)
        return response

    assert failing(report_for(strip_hints)) == {"tools/list-cache-hints"}


def test_a_discovery_without_a_cache_scope_is_caught() -> None:
    def strip_scope(payload: Any, response: Any) -> Any:
        if method_of(payload) == "server/discover" and isinstance(response.get("result"), dict):
            response["result"].pop("cacheScope", None)
        return response

    assert failing(report_for(strip_scope)) == {"discovery/cache-hints"}


def test_a_retired_error_code_is_caught() -> None:
    # -32002 was resource-not-found before it moved to -32602. A server carried
    # over from an older revision is exactly where this shows up.
    def retire(_payload: Any, response: Any) -> Any:
        if isinstance(response, dict) and isinstance(response.get("error"), dict):
            response["error"]["code"] = -32002
        return response

    assert "errors/allocation-policy" in failing(report_for(retire))


def test_a_code_squatting_in_the_reserved_range_is_caught() -> None:
    def squat(_payload: Any, response: Any) -> Any:
        if isinstance(response, dict) and isinstance(response.get("error"), dict):
            response["error"]["code"] = -32055
        return response

    assert "errors/allocation-policy" in failing(report_for(squat))


def test_an_unsupported_version_answered_without_alternatives_is_caught() -> None:
    def strip_data(_payload: Any, response: Any) -> Any:
        if isinstance(response, dict) and isinstance(response.get("error"), dict):
            response["error"].pop("data", None)
        return response

    assert "version/unsupported-refused" in failing(report_for(strip_data))


def test_a_server_still_implementing_the_handshake_is_caught() -> None:
    server = build_server()

    @server.method("initialize")
    def _initialize(_request: Any) -> dict[str, Any]:
        return complete(protocolVersion="2025-06-18", capabilities={})

    report = run(Client(InProcessTransport(server)))
    assert failing(report) == {"version/handshake-removed"}


def test_a_server_that_answers_a_notification_is_caught() -> None:
    def answer_everything(payload: Any, response: Any) -> Any:
        if response is None and isinstance(payload, dict):
            return {"jsonrpc": "2.0", "id": None, "result": {"resultType": "complete"}}
        return response

    assert "framing/notification-unanswered" in failing(report_for(answer_everything))


def test_a_server_that_accepts_a_batch_is_caught() -> None:
    def accept_batches(payload: Any, response: Any) -> Any:
        if isinstance(payload, list):
            return {"jsonrpc": "2.0", "id": 1, "result": {"resultType": "complete"}}
        return response

    assert failing(report_for(accept_batches)) == {"framing/batch-refused"}


def test_an_execution_failure_reported_as_a_protocol_error_is_caught() -> None:
    # The distinction the suite is really guarding: a caller can correct bad
    # arguments and retry, so the refusal has to come back as a result.
    def escalate(payload: Any, response: Any) -> Any:
        if method_of(payload) != "tools/call":
            return response
        result = response.get("result") if isinstance(response, dict) else None
        if isinstance(result, dict) and result.get("isError") is True:
            return {
                "jsonrpc": "2.0",
                "id": response.get("id"),
                "error": {"code": -32602, "message": "bad arguments"},
            }
        return response

    assert "tools/execution-error-is-a-result" in failing(report_for(escalate))


def test_a_non_deterministic_tool_order_is_caught() -> None:
    state = {"flipped": False}

    def shuffle(payload: Any, response: Any) -> Any:
        if method_of(payload) != "tools/list":
            return response
        result = response.get("result") if isinstance(response, dict) else None
        if isinstance(result, dict) and isinstance(result.get("tools"), list):
            if state["flipped"]:
                result["tools"] = list(reversed(result["tools"]))
            state["flipped"] = not state["flipped"]
        return response

    assert "tools/list-deterministic" in failing(report_for(shuffle))


def test_a_tool_listed_without_a_schema_is_caught() -> None:
    def strip_schemas(payload: Any, response: Any) -> Any:
        if method_of(payload) != "tools/list":
            return response
        result = response.get("result") if isinstance(response, dict) else None
        if isinstance(result, dict):
            for tool in result.get("tools", []):
                tool.pop("inputSchema", None)
        return response

    assert "tools/declared-schemas" in failing(report_for(strip_schemas))


def test_an_unechoed_request_id_is_caught() -> None:
    def renumber(_payload: Any, response: Any) -> Any:
        if isinstance(response, dict) and response.get("id") == "a-string-id":
            response["id"] = 1
        return response

    assert "framing/string-id-echoed" in failing(report_for(renumber))


def test_a_server_that_reports_success_for_an_unknown_tool_is_caught() -> None:
    def pretend(payload: Any, response: Any) -> Any:
        params = payload.get("params") if isinstance(payload, dict) else None
        if isinstance(params, dict) and params.get("name") == "no_such_tool_exists_anywhere":
            return {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "result": {"resultType": "complete", "isError": False, "content": []},
            }
        return response

    assert "tools/unknown-tool-refused" in failing(report_for(pretend))


# -- the suite's own failure handling --------------------------------------


def test_a_check_that_crashes_is_recorded_as_a_failure_not_lost() -> None:
    # A suite that died partway through would report fewer problems than it
    # found, which is the wrong direction for this to fail in.
    def explode(_payload: Any, _response: Any) -> Any:
        raise RuntimeError("the server fell over")

    report = report_for(explode)
    assert report.ok is False
    assert len(report.outcomes) == len(CHECKS)
    assert any("RuntimeError" in outcome.detail for outcome in report.failures)


@pytest.mark.parametrize("status", ["pass", "fail", "skip"])
def test_every_status_renders_a_line(status: str) -> None:
    from abacus.conformance import Outcome

    line = Outcome("a/check", "a requirement", "core", status, "some detail").line()
    assert "a/check" in line
    assert "some detail" in line
