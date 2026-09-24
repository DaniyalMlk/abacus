"""Every validation tool driven through the protocol, not through a Python call.

``test_validation.py`` calls the handlers directly, which checks the statistics
and the refusals and proves nothing about the wire. This group has two hazards
the others do not. Its results carry numbers that can legitimately be ``nan`` —
the degradation regression returns one when the in-sample metric does not vary —
and ``json.dumps`` writes a bare ``NaN`` for that, which is not JSON: a strict
parser rejects the whole message and every number beside it. And its inputs are
matrices large enough that a schema mistake shows up as a refusal of perfectly
good arguments rather than as anything obvious.

So each of the five is called over a transport with arguments that should work
and arguments that should not, every result is re-serialised with
``allow_nan=False``, and every refusal is checked for arriving as a result with
``isError`` set rather than as a JSON-RPC error.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from abacus.cli import build_server
from abacus.client import Client, InProcessTransport, StdioTransport


def correlated(seed: int = 11, n: int = 1000, k: int = 40) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, 0.01, size=(n, 1))
    return [list(row) for row in factor + rng.normal(0.0, 0.002, size=(n, k))]


def independent(seed: int = 23, n: int = 1000, k: int = 40) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return [list(row) for row in rng.normal(0.0, 0.01, size=(n, k))]


@pytest.fixture
def client() -> Iterator[Client]:
    with Client(InProcessTransport(build_server())) as connected:
        yield connected


def succeed(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call over the wire and insist on a usable, strictly serialisable result."""
    exchange = client.call_tool(name, arguments)
    assert exchange.error is None, exchange.error
    result = exchange.result
    assert result is not None
    assert result["isError"] is False, result["structuredContent"]

    structured = result["structuredContent"]
    text = json.loads(result["content"][0]["text"])
    assert text == structured
    assert isinstance(structured, dict)
    # `json.loads` accepts NaN and Infinity quite happily, so the round trip
    # above would not notice either. A strict reader is what this asserts.
    json.dumps(structured, allow_nan=False)
    return structured


def be_refused(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    exchange = client.call_tool(name, arguments)
    assert exchange.error is None, f"a fixable mistake became a protocol error: {exchange.error}"
    result = exchange.result
    assert result is not None
    assert result["isError"] is True, result["structuredContent"]
    error = result["structuredContent"]["error"]
    assert isinstance(error, dict)
    assert error["message"], "a refusal with no message tells the caller nothing"
    assert result["content"][0]["text"] == error["message"]
    return error


def test_the_validation_tools_are_all_listed(client: Client) -> None:
    names = client.tool_names()
    for name in (
        "deflated_sharpe_ratio",
        "minimum_track_record_length",
        "effective_trial_count",
        "backtest_overfitting_probability",
        "superior_predictive_ability",
    ):
        assert name in names


# -- the five tools, each with a good call and a bad one ---------------------


def test_the_deflated_sharpe_survives_the_wire(client: Client) -> None:
    result = succeed(
        client, "deflated_sharpe_ratio", {"trials": correlated(), "periodsPerYear": 252}
    )
    assert result["trials"] == 40
    assert result["effectiveTrials"]["eigenvalue"] == pytest.approx(3.0, abs=0.1)
    assert result["deflatedSharpe"] > result["deflatedSharpeAgainstRawTrials"]
    assert result["sharpe"]["annualised"]["periodsPerYear"] == 252


def test_too_short_a_history_comes_back_as_a_refusal(client: Client) -> None:
    """The schema's own minimum catches it before the handler does.

    A matrix with too few rows is a fixable mistake, so it must arrive as a
    result rather than as a protocol error — the caller can send more history.
    """
    error = be_refused(
        client, "deflated_sharpe_ratio", {"trials": correlated(n=12, k=5)}
    )
    assert error["kind"] in {"invalid_input", "domain"}


def test_the_track_record_length_survives_the_wire(client: Client) -> None:
    result = succeed(
        client,
        "minimum_track_record_length",
        {"sharpe": 1.0 / math.sqrt(252), "periodsPerYear": 252},
    )
    assert result["minimumYears"] == pytest.approx(2.71, abs=0.05)
    assert result["annualisedSharpe"] == pytest.approx(1.0, abs=1e-6)


def test_a_sharpe_under_the_benchmark_comes_back_as_a_refusal(client: Client) -> None:
    error = be_refused(client, "minimum_track_record_length", {"sharpe": -0.01})
    assert error["kind"] == "domain"
    assert "no track record length" in error["message"]


def test_impossible_moments_come_back_as_a_refusal(client: Client) -> None:
    """kurtosis >= 1 + skewness**2, which the formula would ignore."""
    error = be_refused(
        client,
        "minimum_track_record_length",
        {"sharpe": 0.06, "skewness": -1.5, "excessKurtosis": 0.0},
    )
    assert error["kind"] == "domain"
    assert "kurtosis" in error["message"]


def test_the_effective_count_survives_the_wire(client: Client) -> None:
    result = succeed(client, "effective_trial_count", {"trials": independent()})
    assert result["effectiveTrials"]["eigenvalue"] == pytest.approx(40.0, abs=1e-6)
    assert result["reduction"] == pytest.approx(0.0, abs=1e-6)


def test_a_ragged_matrix_comes_back_as_a_refusal(client: Client) -> None:
    trials = independent(n=60, k=4)
    trials[9] = trials[9][:2]
    error = be_refused(client, "effective_trial_count", {"trials": trials})
    assert error["kind"] in {"invalid_input", "domain"}


def test_the_overfitting_probability_survives_the_wire(client: Client) -> None:
    """The result carrying a possible nan, checked as JSON a strict reader accepts."""
    result = succeed(
        client, "backtest_overfitting_probability", {"trials": correlated(), "blocks": 10}
    )
    assert result["partitions"] == 252
    assert 0.0 <= result["probabilityOfBacktestOverfitting"] <= 1.0
    assert set(result["degradation"]) == {"slope", "intercept", "rSquared"}


def test_too_few_strategies_come_back_as_a_refusal(client: Client) -> None:
    error = be_refused(
        client,
        "backtest_overfitting_probability",
        {"trials": [row[:2] for row in independent()]},
    )
    assert error["kind"] == "domain"
    assert "grid" in error["message"]


def test_superior_predictive_ability_survives_the_wire(client: Client) -> None:
    result = succeed(
        client, "superior_predictive_ability", {"trials": independent(k=10), "bootstrap": 300}
    )
    bounds = result["pValue"]
    assert bounds["lower"] <= bounds["consistent"] <= bounds["upper"] + 1e-12
    assert result["seed"] == 0
    assert result["candidates"] == list(range(1, 10))


def test_the_same_call_twice_over_the_wire_gives_the_same_p_value(client: Client) -> None:
    """Idempotent is annotated on these tools, and a bootstrap is not by default."""
    arguments = {"trials": independent(k=8), "bootstrap": 300}
    first = succeed(client, "superior_predictive_ability", dict(arguments))
    second = succeed(client, "superior_predictive_ability", dict(arguments))
    assert first == second


def test_a_benchmark_column_out_of_range_comes_back_as_a_refusal(client: Client) -> None:
    error = be_refused(
        client, "superior_predictive_ability", {"trials": independent(k=4), "benchmark": 11}
    )
    assert error["kind"] == "domain"


def test_the_tools_work_over_a_real_pipe() -> None:
    """The same calls against a launched subprocess, which exercises the framing.

    These payloads are the largest the server takes — a forty-column matrix over
    a thousand periods — so this is also the only place the framing is driven
    with a body of that size.
    """
    command = [sys.executable, "-m", "abacus.cli", "stdio"]
    with StdioTransport(command) as transport, Client(transport) as client:
        assert client.discover().ok

        deflated = succeed(
            client, "deflated_sharpe_ratio", {"trials": correlated(), "periodsPerYear": 252}
        )
        assert deflated["effectiveTrials"]["eigenvalue"] == pytest.approx(3.0, abs=0.1)

        overfitting = succeed(
            client, "backtest_overfitting_probability", {"trials": correlated(), "blocks": 8}
        )
        assert overfitting["trials"] == 40

        error = be_refused(
            client,
            "backtest_overfitting_probability",
            {"trials": [row[:2] for row in independent()]},
        )
        assert error["kind"] == "domain"

        assert client.discover().ok
