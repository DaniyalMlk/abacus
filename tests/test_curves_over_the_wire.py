"""Every curve and bond tool driven through the protocol, not through a Python call.

Same argument as ``test_risk_over_the_wire``: a tool can be right in process and
unusable over the protocol. The failure that nearly happened here is the reason the
file exists — the bootstrap result carried the solver's ``Root`` objects rather than
their fields, and `json.dumps` refused the payload. Nothing in the registry tests
would have caught it, because the registry never serialises.

So each of the five is called twice over a transport, once with arguments that
should work and once with arguments that should not, and every refusal is checked
for arriving as a result with ``isError`` set rather than a JSON-RPC error.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from abacus.cli import build_server
from abacus.client import Client, InProcessTransport, StdioTransport

REFERENCE = "2021-01-05"

QUOTES: dict[str, Any] = {
    "reference": REFERENCE,
    "basis": "ACT_365F",
    "deposits": [
        {"maturity": "2021-07-05", "rate": 0.0020, "basis": "ACT_360", "label": "6m"}
    ],
    "swaps": [
        {
            "maturity": maturity,
            "rate": rate,
            "frequency": "SEMI_ANNUAL",
            "basis": "THIRTY_360_BOND",
            "label": label,
        }
        for maturity, rate, label in (
            ("2023-01-05", 0.0060, "2y"),
            ("2026-01-05", 0.0150, "5y"),
            ("2031-01-05", 0.0220, "10y"),
        )
    ],
}

BOND: dict[str, Any] = {
    "effective": REFERENCE,
    "maturity": "2031-01-05",
    "coupon": 0.05,
    "frequency": "SEMI_ANNUAL",
    "basis": "THIRTY_360_BOND",
    "label": "10y 5s",
}


@pytest.fixture
def client() -> Iterator[Client]:
    with Client(InProcessTransport(build_server())) as connected:
        yield connected


def succeed(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    exchange = client.call_tool(name, arguments)
    assert exchange.error is None, exchange.error
    result = exchange.result
    assert result is not None
    assert result["isError"] is False, result["structuredContent"]

    # The text block is the same payload serialised. A value JSON cannot carry —
    # a date, an enum, a solver object — fails here and only here.
    structured = result["structuredContent"]
    assert json.loads(result["content"][0]["text"]) == structured
    assert isinstance(structured, dict)
    return structured


def be_refused(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    exchange = client.call_tool(name, arguments)
    assert exchange.error is None, f"a fixable mistake became a protocol error: {exchange.error}"
    result = exchange.result
    assert result is not None
    assert result["isError"] is True, result["structuredContent"]
    error = result["structuredContent"]["error"]
    assert isinstance(error, dict)
    assert error["message"]
    assert result["content"][0]["text"] == error["message"]
    return error


def test_the_curve_tools_are_all_listed(client: Client) -> None:
    names = client.tool_names()
    for name in (
        "bootstrap_discount_curve",
        "discount_curve_rates",
        "bond_analytics",
        "bond_curve_risk",
        "bond_spreads",
    ):
        assert name in names


def test_bootstrapping_over_the_wire(client: Client) -> None:
    """Including the solver's own diagnostics, which is where serialisation broke.

    ``Bootstrapped.solutions`` holds ``Root`` objects. The first draft put them in
    the payload whole and `json.dumps` refused it, so the tool raised rather than
    answering — invisible to a direct handler call.
    """
    payload = succeed(client, "bootstrap_discount_curve", QUOTES)
    assert payload["repricesItsInstruments"] is True
    assert isinstance(payload["handle"], str)
    solutions = payload["solutions"]
    assert isinstance(solutions, list)
    assert all(isinstance(entry["residual"], float) for entry in solutions)
    assert all(isinstance(entry["converged"], bool) for entry in solutions)

    error = be_refused(
        client,
        "bootstrap_discount_curve",
        # A basis nobody supports, which is what the enum is for.
        {"reference": REFERENCE, "basis": "ACT_ACT_ICMA", "swaps": QUOTES["swaps"]},
    )
    assert error["kind"] == "invalid_input"


def test_reading_a_curve_over_the_wire(client: Client) -> None:
    handle = succeed(client, "bootstrap_discount_curve", QUOTES)["handle"]
    payload = succeed(
        client,
        "discount_curve_rates",
        {"handle": handle, "dates": ["2022-01-05", "2026-01-05"]},
    )
    assert len(payload["points"]) == 2

    error = be_refused(
        client, "discount_curve_rates", {"handle": handle, "dates": ["2041-01-05"]}
    )
    assert error["kind"] == "domain"
    assert "horizon" in error["message"]


def test_bond_analytics_over_the_wire(client: Client) -> None:
    handle = succeed(client, "bootstrap_discount_curve", QUOTES)["handle"]
    payload = succeed(client, "bond_analytics", {"handle": handle, "bond": BOND, "price": 120.0})
    assert payload["fromYield"]["yieldToMaturity"] > 0.0
    assert payload["fromCurve"]["price"] > 0.0
    # Dates in the cashflow table are strings, not date objects, which is the other
    # half of the serialisation question.
    assert all(isinstance(flow["date"], str) for flow in payload["cashflows"])

    error = be_refused(
        client,
        "bond_analytics",
        {"handle": handle, "bond": BOND, "price": 120.0, "yieldToMaturity": 0.03},
    )
    assert "not both" in error["message"]


def test_curve_risk_over_the_wire(client: Client) -> None:
    handle = succeed(client, "bootstrap_discount_curve", QUOTES)["handle"]
    payload = succeed(client, "bond_curve_risk", {"handle": handle, "bond": BOND})
    assert abs(float(payload["sumAgainstEffectiveDuration"])) < 1e-6
    assert payload["instrumentRisk"]["instruments"][-1]["instrument"]["label"] == "10y"

    error = be_refused(
        client,
        "bond_curve_risk",
        {"handle": handle, "bond": BOND, "buckets": [5.0, 2.0]},
    )
    assert "increasing order" in error["message"]


def test_spreads_over_the_wire(client: Client) -> None:
    handle = succeed(client, "bootstrap_discount_curve", QUOTES)["handle"]
    payload = succeed(
        client,
        "bond_spreads",
        {
            "handle": handle,
            "bond": BOND,
            "price": 110.0,
            "volatility": 0.15,
            "firstCall": "2026-01-05",
        },
    )
    option = payload["embeddedOption"]
    assert option["latticeRepricesCurve"] is True
    assert float(option["optionCost"]) > 0.0

    error = be_refused(
        client,
        "bond_spreads",
        {
            "handle": handle,
            "bond": BOND,
            "price": 110.0,
            "volatility": 0.15,
            "firstCall": "2031-01-05",
        },
    )
    assert "not an option" in error["message"]


def test_one_curve_serves_every_tool_over_the_wire(client: Client) -> None:
    """The workflow the handle exists for, bootstrapped once and spent four times.

    And a cross-tool identity to go with it: the effective duration reported by
    ``bond_analytics`` and the one reported by ``bond_curve_risk`` are the same
    number, reached through the same curve rebuilt twice from the same handle.
    """
    handle = succeed(client, "bootstrap_discount_curve", QUOTES)["handle"]

    rates = succeed(client, "discount_curve_rates", {"handle": handle, "dates": ["2026-01-05"]})
    analytics = succeed(client, "bond_analytics", {"handle": handle, "bond": BOND})
    risk = succeed(client, "bond_curve_risk", {"handle": handle, "bond": BOND})
    spreads = succeed(
        client, "bond_spreads", {"handle": handle, "bond": BOND, "price": 110.0}
    )

    assert rates["points"][0]["discount"] < 1.0
    assert float(risk["effectiveDuration"]) == pytest.approx(
        float(analytics["fromCurve"]["effectiveDuration"]), rel=1e-12
    )
    assert float(risk["price"]) == pytest.approx(float(analytics["fromCurve"]["price"]), rel=1e-12)
    assert float(spreads["curvePrice"]) == pytest.approx(
        float(analytics["fromCurve"]["price"]), rel=1e-12
    )
    # The bond is cheaper than the curve says, so the spread is positive.
    assert float(spreads["zSpread"]) > 0.0


def test_the_curve_tools_answer_a_launched_server_over_stdio() -> None:
    """Through framing, a pipe and a separate interpreter.

    The curve handle round trip is the reason this matters: a handle that did not
    survive serialisation would fail its integrity check on the next call rather
    than return a wrong number, and in-process transports share objects instead of
    bytes.
    """
    command = [sys.executable, "-m", "abacus.cli", "stdio"]
    with StdioTransport(command) as transport, Client(transport) as client:
        assert client.discover().ok
        built = succeed(client, "bootstrap_discount_curve", QUOTES)
        spreads = succeed(
            client,
            "bond_spreads",
            {
                "handle": built["handle"],
                "bond": BOND,
                "price": 110.0,
                "volatility": 0.15,
                "firstCall": "2026-01-05",
            },
        )
        assert spreads["embeddedOption"]["latticeRepricesCurve"] is True

        error = be_refused(
            client,
            "bond_curve_risk",
            {"handle": built["handle"], "bond": BOND, "buckets": [0.5, 1.0, 2.0]},
        )
        assert "covers no pillar" in error["message"]

        # Still serving, so the refusal was a refusal and not a crash.
        assert client.discover().ok
