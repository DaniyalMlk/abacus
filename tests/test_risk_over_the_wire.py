"""Every risk tool driven through the protocol, not through a Python call.

The tests in ``test_risk.py`` call the handlers through the registry, which checks
the arithmetic and the refusals and proves nothing about the wire. A tool can be
correct in process and unusable over the protocol in several ways that nothing
else here would catch: a payload holding a value JSON cannot carry, a schema that
rejects the arguments its own description asks for, a handle that does not survive
a round trip through serialisation, an error that escapes as a stack trace instead
of a result a model can act on.

So each of the five is called twice over a transport — once with arguments that
should work and once with arguments that should not — and the second case is
checked for being a *result* with ``isError`` set rather than a JSON-RPC error,
because that is the distinction that decides whether a model can repair the call
and try again.

One of them runs against a launched subprocess rather than in process, which is
the only configuration that exercises the framing as well.
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from abacus.cli import build_server
from abacus.client import Client, InProcessTransport, StdioTransport

PERIODS = 252
ASSETS = ["EQ", "CR", "GV", "CM"]
WEIGHTS = [0.4, 0.25, 0.15, 0.2]


def one_factor(periods: int = 260, assets: int = 4, *, seed: int = 5) -> list[list[float]]:
    rng = random.Random(seed)
    factor = [rng.gauss(0.0003, 0.008) for _ in range(periods)]
    return [
        [factor[t] * (0.7 + 0.15 * a) + rng.gauss(0.0, 0.004) for a in range(assets)]
        for t in range(periods)
    ]


RETURNS = one_factor()


@pytest.fixture
def client() -> Iterator[Client]:
    with Client(InProcessTransport(build_server())) as connected:
        yield connected


def succeed(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call over the wire and insist on a usable success result."""
    exchange = client.call_tool(name, arguments)
    assert exchange.error is None, exchange.error
    result = exchange.result
    assert result is not None
    assert result["isError"] is False, result["structuredContent"]

    # Both representations, because a client that reads only text blocks has to
    # get the answer too. The text block is the same payload serialised, so it has
    # to parse and agree — a payload holding a value JSON cannot carry would fail
    # here and nowhere else.
    structured = result["structuredContent"]
    text = json.loads(result["content"][0]["text"])
    assert text == structured
    assert isinstance(structured, dict)
    return structured


def be_refused(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call over the wire and insist the refusal arrives as a recoverable result.

    A JSON-RPC error here would mean the caller is told the request was
    unprocessable, when in fact the arguments were wrong and fixable — which is
    the difference between a model retrying successfully and giving up.
    """
    exchange = client.call_tool(name, arguments)
    assert exchange.error is None, f"a fixable mistake became a protocol error: {exchange.error}"
    result = exchange.result
    assert result is not None
    assert result["isError"] is True, result["structuredContent"]
    error = result["structuredContent"]["error"]
    assert isinstance(error, dict)
    assert error["message"], "a refusal with no message tells the caller nothing"
    # The text block carries the message, so a client that renders only text shows
    # the caller something actionable rather than an empty failure.
    assert result["content"][0]["text"] == error["message"]
    return error


# -- the five tools, each with a good call and a bad one ---------------------


def test_the_risk_tools_are_all_listed(client: Client) -> None:
    names = client.tool_names()
    for name in (
        "estimate_return_moments",
        "portfolio_tail_risk",
        "portfolio_risk_contributions",
        "risk_parity_weights",
        "portfolio_drawdown",
    ):
        assert name in names


def test_estimating_moments_over_the_wire(client: Client) -> None:
    payload = succeed(
        client,
        "estimate_return_moments",
        {"returns": RETURNS, "assets": ASSETS, "periodsPerYear": PERIODS},
    )
    assert payload["observations"] == len(RETURNS)
    assert isinstance(payload["handle"], str)
    assert len(payload["correlation"]) == 4

    error = be_refused(
        client,
        "estimate_return_moments",
        # Percentages rather than fractions: the commonest input mistake there is,
        # and a hundredfold error that every downstream number would inherit.
        {"returns": [[3.0, 250.0, -1.5, 0.4]] * 20, "periodsPerYear": PERIODS},
    )
    assert error["kind"] == "invalid_input"


def test_tail_risk_over_the_wire(client: Client) -> None:
    payload = succeed(
        client,
        "portfolio_tail_risk",
        {
            "returns": RETURNS,
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
            "method": "historical",
        },
    )
    assert payload["method"] == "historical"
    assert payload["expectedShortfall"] >= payload["valueAtRisk"]

    error = be_refused(
        client,
        "portfolio_tail_risk",
        {
            "returns": RETURNS,
            "assets": ASSETS,
            "weights": [0.4, 0.25],
            "periodsPerYear": PERIODS,
        },
    )
    assert error["kind"] == "domain"
    assert "2 weights for 4 assets" in error["message"]


def test_contributions_over_the_wire(client: Client) -> None:
    payload = succeed(
        client,
        "portfolio_risk_contributions",
        {
            "returns": RETURNS,
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
        },
    )
    assert [entry["name"] for entry in payload["assets"]] == ASSETS

    error = be_refused(
        client,
        "portfolio_risk_contributions",
        {
            "returns": RETURNS,
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
            "measure": "sharpe",
        },
    )
    assert error["kind"] == "invalid_input"
    assert "enum" in json.dumps(error["details"])


def test_risk_parity_over_the_wire(client: Client) -> None:
    payload = succeed(
        client,
        "risk_parity_weights",
        {"returns": RETURNS, "assets": ASSETS, "periodsPerYear": PERIODS},
    )
    assert payload["converged"] is True
    assert len(payload["weights"]) == 4

    error = be_refused(
        client,
        "risk_parity_weights",
        {
            "returns": RETURNS,
            "assets": ASSETS,
            "periodsPerYear": PERIODS,
            "budgets": [0.5, 0.5],
        },
    )
    assert "2 budgets for 4 assets" in error["message"]


def test_drawdown_over_the_wire(client: Client) -> None:
    payload = succeed(
        client,
        "portfolio_drawdown",
        {
            "returns": RETURNS,
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
        },
    )
    assert payload["maximumDrawdown"]["depth"] > 0.0

    error = be_refused(
        client,
        "portfolio_drawdown",
        {
            "returns": RETURNS,
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
            "index": [f"d{i}" for i in range(len(RETURNS))],
        },
    )
    assert "wealth curve" in error["message"]


# -- the handle, across calls -------------------------------------------------


def test_a_handle_survives_the_wire_and_is_reusable(client: Client) -> None:
    """Minted in one call, spent in three others, all over the protocol.

    This is the workflow the handle exists for, and it only means anything end to
    end: the handle is a string in a result, carried back as a string in the next
    request's arguments, with nothing on the server remembering it.
    """
    moments = succeed(
        client,
        "estimate_return_moments",
        {"returns": RETURNS, "assets": ASSETS, "periodsPerYear": PERIODS},
    )
    handle = moments["handle"]

    risk = succeed(client, "portfolio_tail_risk", {"handle": handle, "weights": WEIGHTS})
    contributions = succeed(
        client, "portfolio_risk_contributions", {"handle": handle, "weights": WEIGHTS}
    )
    parity = succeed(client, "risk_parity_weights", {"handle": handle})

    assert risk["observations"] == len(RETURNS)
    assert risk["periodsPerYear"] == PERIODS
    assert [entry["name"] for entry in contributions["assets"]] == ASSETS
    assert parity["converged"] is True

    # The portfolio volatility is one number computed from one covariance matrix,
    # so two tools reading the same handle must agree on it exactly. They do not
    # share a code path to it: one goes through the parametric estimator and the
    # other through the Euler decomposition.
    assert contributions["total"] == pytest.approx(risk["volatility"], rel=1e-12)


def test_a_handle_refused_for_a_path_method_over_the_wire(client: Client) -> None:
    moments = succeed(
        client,
        "estimate_return_moments",
        {"returns": RETURNS, "assets": ASSETS, "periodsPerYear": PERIODS},
    )
    error = be_refused(
        client,
        "portfolio_tail_risk",
        {"handle": moments["handle"], "weights": WEIGHTS, "method": "historical"},
    )
    assert error["kind"] == "handle_lacks_path"
    assert "`returns`" in error["message"]


# -- against a launched process ----------------------------------------------


def test_the_risk_tools_answer_a_launched_server_over_stdio() -> None:
    """The same calls through framing, a pipe and a separate interpreter.

    In-process transports share objects; this one shares only bytes. A payload
    that is not serialisable, or a number that does not survive the round trip,
    fails here and passes everywhere else — and the handle round trip is exactly
    the case where that would matter, because a handle spent after a lossy encode
    fails its integrity check rather than returning a wrong number.
    """
    command = [sys.executable, "-m", "abacus.cli", "stdio"]
    with StdioTransport(command) as transport, Client(transport) as client:
        assert client.discover().ok
        moments = succeed(
            client,
            "estimate_return_moments",
            {"returns": RETURNS, "assets": ASSETS, "periodsPerYear": PERIODS},
        )
        risk = succeed(
            client,
            "portfolio_tail_risk",
            {"handle": moments["handle"], "weights": WEIGHTS, "method": "student-t"},
        )
        assert risk["method"] == "student-t"
        assert risk["degreesOfFreedom"] == 5.0
        assert risk["expectedShortfall"] >= risk["valueAtRisk"]

        # And a refusal still arrives as a result over a real pipe, rather than
        # killing the process or escaping as a protocol error.
        error = be_refused(
            client,
            "portfolio_drawdown",
            {"returns": RETURNS, "weights": [9.0, 0.0, 0.0, 0.0], "periodsPerYear": PERIODS},
        )
        assert error["kind"] == "domain"

        # The server is still serving afterwards, which is what makes the refusal a
        # refusal rather than a crash that happened to be reported.
        assert client.discover().ok
