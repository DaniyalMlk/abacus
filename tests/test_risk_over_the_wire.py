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
import math
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


def fat_tailed(periods: int = 1200, *, seed: int = 17) -> list[list[float]]:
    """Returns with a genuinely heavy tail, so a fitted shape has something to find."""
    rng = random.Random(seed)

    def student_t() -> float:
        chi_square = 2.0 * rng.gammavariate(2.0, 1.0)
        return rng.gauss(0.0, 1.0) / math.sqrt(chi_square / 4.0)

    factor = [0.004 * student_t() for _ in range(periods)]
    return [
        [factor[t] * (0.7 + 0.15 * a) + rng.gauss(0.0, 0.002) for a in range(4)]
        for t in range(periods)
    ]


def test_the_fitted_tail_over_the_wire(client: Client) -> None:
    """The sixth method, with a good call and two bad ones.

    The first refusal is the one this method has and no other does: a confidence
    below the threshold the fit was taken above, where the fit deliberately knows
    nothing. It has to arrive as a result a model can repair — with the lowest legal
    confidence in the message — rather than as a protocol error, because the repair
    is to ask for a higher confidence and the caller cannot work that out from
    "invalid arguments".

    The payload is also the largest this tool produces, carrying a mean excess curve
    of eight points and two fields that are legitimately null, so the round trip
    through the text block is worth having here specifically.
    """
    payload = succeed(
        client,
        "portfolio_tail_risk",
        {
            "returns": fat_tailed(),
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
            "method": "extreme-value",
            "confidence": 0.999,
            "tailFraction": 0.05,
        },
    )
    assert payload["method"] == "extreme-value"
    assert payload["expectedShortfall"] >= payload["valueAtRisk"]
    # 5% of 1,200 is the 60th largest loss, which 59 strictly exceed: the threshold
    # lands on an observation rather than between two, so the count is pinned.
    assert payload["exceedances"] == 59
    assert payload["lowestConfidence"] == pytest.approx(1.0 - 59 / 1200)
    assert payload["shape"] > 0.0
    assert payload["upperEndpoint"] is None
    assert len(payload["meanExcessCurve"]) == 8
    # Nothing here may be a bare Infinity or NaN: json.dumps writes those and
    # json.loads reads them back, so only a strict re-encode catches it.
    assert json.dumps(payload, allow_nan=False)

    inside = be_refused(
        client,
        "portfolio_tail_risk",
        {
            "returns": fat_tailed(),
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
            "method": "extreme-value",
            # Legal for every other method on this tool and inside the body for
            # this one, which is the distinction the refusal has to make.
            "confidence": 0.9,
        },
    )
    assert inside["kind"] == "domain"
    assert "inside the body" in inside["message"]
    assert "0.95" in inside["message"]

    schema = be_refused(
        client,
        "portfolio_tail_risk",
        {
            "returns": fat_tailed(),
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
            "method": "extreme-value",
            # Over the schema's maximum: a tail fraction above a half is not a tail.
            "tailFraction": 0.9,
        },
    )
    assert schema["kind"] == "invalid_input"


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


def tail_dependent(periods: int = 700, *, seed: int = 23) -> list[list[float]]:
    """Returns with tail dependence, not merely heavy marginal tails.

    One Student-t factor shared across the columns with small idiosyncratic noise,
    so the chi-square mixing variable the copula estimates survives into the ranks.
    Heavy marginals alone would not distinguish this from independent heavy-tailed
    columns, and that distinction is the method.
    """
    rng = random.Random(seed)

    def student_t() -> float:
        chi_square = 2.0 * rng.gammavariate(2.0, 1.0)
        return rng.gauss(0.0, 1.0) / math.sqrt(chi_square / 4.0)

    factor = [0.004 * student_t() for _ in range(periods)]
    return [
        [factor[t] * (0.7 + 0.15 * a) + rng.gauss(0.0, 0.0008) for a in range(4)]
        for t in range(periods)
    ]


def test_the_copula_over_the_wire(client: Client) -> None:
    """The seventh method, with a good call and three bad ones.

    Worth its own wire test for two reasons beyond the usual. The payload carries a
    list of six nested objects and two fields that are legitimately null under the
    Gaussian family, so the round trip through the text block is where a
    serialisation mistake would show. And the refusals are of three different
    kinds — a sample too short for a rank estimate, a path count outside the
    schema's range, and a family that does not exist — which should arrive as three
    distinguishable results a model can act on rather than as one protocol error.
    """
    payload = succeed(
        client,
        "portfolio_tail_risk",
        {
            "returns": tail_dependent(),
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
            "method": "copula",
            "confidence": 0.99,
            "paths": 4000,
            "seed": 1,
        },
    )
    assert payload["method"] == "copula"
    assert payload["degreesOfFreedomFitted"] is True
    assert 3.0 < payload["degreesOfFreedom"] < 12.0
    assert payload["likelihoodRatio"] > 20.0
    assert payload["expectedShortfall"] >= payload["valueAtRisk"]
    assert payload["gaussianExpectedShortfall"] > 0.0
    assert len(payload["tailDependence"]) == 6
    assert all(
        -1.0 <= pair["correlation"] <= 1.0 and pair["coefficient"] > 0.0
        for pair in payload["tailDependence"]
    )
    assert payload["standardError"] > 0.0
    # Nothing here may be a bare Infinity or NaN: json.dumps writes those and
    # json.loads reads them back, so only a strict re-encode catches it.
    assert json.dumps(payload, allow_nan=False)

    gaussian = succeed(
        client,
        "portfolio_tail_risk",
        {
            "returns": tail_dependent(),
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
            "method": "copula",
            "copulaFamily": "gaussian",
            "paths": 4000,
        },
    )
    # A null in the payload has to survive as a null rather than as a string or a
    # dropped key, which is the case a round trip is most likely to lose.
    assert gaussian["degreesOfFreedom"] is None
    assert gaussian["valueAtRisk"] == gaussian["gaussianValueAtRisk"]
    assert json.dumps(gaussian, allow_nan=False)

    short = be_refused(
        client,
        "portfolio_tail_risk",
        {
            # Legal for every other method on this tool, and too short for a rank
            # estimate of every pair, which is the distinction this refusal makes.
            "returns": tail_dependent(periods=150),
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
            "method": "copula",
        },
    )
    assert short["kind"] == "domain"
    assert "at least 200" in short["message"]

    paths = be_refused(
        client,
        "portfolio_tail_risk",
        {
            "returns": tail_dependent(),
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
            "method": "copula",
            "paths": 10,
        },
    )
    assert paths["kind"] == "invalid_input"

    family = be_refused(
        client,
        "portfolio_tail_risk",
        {
            "returns": tail_dependent(),
            "assets": ASSETS,
            "weights": WEIGHTS,
            "periodsPerYear": PERIODS,
            "method": "copula",
            "copulaFamily": "clayton",
        },
    )
    assert family["kind"] == "invalid_input"
