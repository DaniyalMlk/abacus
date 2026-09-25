"""The two new tools driven through the protocol rather than through Python.

``test_model_validation.py`` calls the handlers through the registry, which
proves the arithmetic and the refusals and nothing about the wire. A tool can
be correct in process and unusable over the protocol in ways nothing else here
would catch: a payload holding a value JSON cannot carry, a schema that rejects
the arguments its own description asks for, an error escaping as a stack trace
instead of a result a model can repair.

So each is called twice — once with arguments that should work and once with
arguments that should not — and the refusal is checked for being a *result*
with ``isError`` set rather than a JSON-RPC error. That distinction decides
whether a model fixes its call and retries or gives up.

One of them runs against a launched subprocess, which is the only configuration
that exercises the framing as well.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from shortfall.distributions import normal_pdf, normal_ppf

from abacus.cli import build_server
from abacus.client import Client, InProcessTransport, StdioTransport

CONFIDENCE = 0.99
TAIL = 1.0 - CONFIDENCE
OBSERVATIONS = 250


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
    structured = result["structuredContent"]
    # The text block is the same payload serialised, so it has to parse and
    # agree. A value JSON cannot carry would fail here and nowhere else.
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
    assert error["message"], "a refusal with no message tells the caller nothing"
    assert result["content"][0]["text"] == error["message"]
    return error


def forecasts(volatility: float) -> tuple[float, float]:
    quantile = -normal_ppf(TAIL)
    return volatility * quantile, volatility * normal_pdf(quantile) / TAIL


def returns(volatility: float, count: int = OBSERVATIONS, *, seed: int = 5) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, volatility) for _ in range(count)]


def validation_call(
    true_volatility: float = 0.012, forecast_volatility: float = 0.012, **extra: Any
) -> dict[str, Any]:
    var, shortfall = forecasts(forecast_volatility)
    args: dict[str, Any] = {
        "returns": returns(true_volatility),
        "valueAtRisk": [var] * OBSERVATIONS,
        "expectedShortfall": [shortfall] * OBSERVATIONS,
        "confidence": CONFIDENCE,
    }
    args.update(extra)
    return args


def swap(maturity: str, rate: float) -> dict[str, Any]:
    return {
        "maturity": maturity,
        "rate": rate,
        "frequency": "SEMI_ANNUAL",
        "basis": "THIRTY_360_BOND",
    }


def horizon_call(**extra: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "reference": "2021-01-05",
        "basis": "ACT_365F",
        "deposits": [{"maturity": "2021-07-05", "rate": 0.0020, "basis": "ACT_360"}],
        "swaps": [
            swap("2023-01-05", 0.0060),
            swap("2026-01-05", 0.0150),
            swap("2031-01-05", 0.0220),
            swap("2041-01-07", 0.0270),
        ],
        "bond": {
            "effective": "2021-01-05",
            "maturity": "2031-01-05",
            "coupon": 0.03,
            "frequency": "SEMI_ANNUAL",
            "basis": "THIRTY_360_BOND",
        },
        "horizon": "2022-01-05",
    }
    args.update(extra)
    return args


def test_both_tools_are_listed(client: Client) -> None:
    names = client.tool_names()
    assert "validate_risk_model" in names
    assert "bond_carry_rolldown" in names


# -- validate_risk_model -----------------------------------------------------


def test_validation_answers_over_the_wire(client: Client) -> None:
    payload = succeed(client, "validate_risk_model", validation_call())
    assert payload["observations"] == OBSERVATIONS
    assert payload["trafficLight"]["zone"] == "green"
    assert payload["rejectedAt5Percent"] == []
    assert len(payload["tests"]) == 3


def test_a_condemned_model_answers_over_the_wire(client: Client) -> None:
    payload = succeed(
        client,
        "validate_risk_model",
        validation_call(true_volatility=0.03, forecast_volatility=0.01),
    )
    assert payload["trafficLight"]["zone"] == "red"
    assert "unconditional coverage" in payload["rejectedAt5Percent"]
    assert payload["expectedShortfallTest"]["realisedOverForecast"] > 1.0


def test_a_simulated_p_value_survives_the_round_trip(client: Client) -> None:
    payload = succeed(
        client,
        "validate_risk_model",
        validation_call(true_volatility=0.022, replications=300, seed=3),
    )
    found = payload["expectedShortfallTest"]
    assert found["replications"] == 300
    assert 0.0 < found["unconditionalPValue"] <= 1.0


def test_a_misaligned_forecast_series_is_a_recoverable_refusal(client: Client) -> None:
    args = validation_call()
    args["valueAtRisk"] = args["valueAtRisk"][:-1]
    error = be_refused(client, "validate_risk_model", args)
    assert "one-step-ahead" in error["message"]


def test_a_negative_forecast_is_a_recoverable_refusal(client: Client) -> None:
    """The sign convention crossed. The schema's exclusiveMinimum catches it
    first, and either way the caller gets something it can fix."""
    args = validation_call()
    args["valueAtRisk"] = [-0.02] * OBSERVATIONS
    be_refused(client, "validate_risk_model", args)


def test_an_unknown_field_is_a_recoverable_refusal(client: Client) -> None:
    args = validation_call()
    args["confidenceLevel"] = 0.99
    be_refused(client, "validate_risk_model", args)


def test_the_note_reaches_the_caller_intact(client: Client) -> None:
    """The note carries the sign convention and the scale of the shortfall
    statistics, and it is the only place a model is told either."""
    payload = succeed(client, "validate_risk_model", validation_call())
    note = payload["note"]
    assert "POSITIVE losses" in note
    assert "CLUSTER" in note
    assert "-0.245" in note


def test_clustering_is_reported_over_the_wire(client: Client) -> None:
    rng = random.Random(46)
    observed: list[float] = []
    turbulent = False
    for _ in range(1500):
        turbulent = rng.random() < (0.90 if turbulent else 0.02)
        observed.append(rng.gauss(0.0, 0.030 if turbulent else 0.006))
    level = math.sqrt(sum(value * value for value in observed) / len(observed))
    var, _ = forecasts(level)
    payload = succeed(
        client,
        "validate_risk_model",
        {"returns": observed, "valueAtRisk": [var] * 1500, "confidence": CONFIDENCE},
    )
    by_name = {one["name"]: one for one in payload["tests"]}
    assert by_name["independence"]["rejectsAt1Percent"] is True


# -- bond_carry_rolldown -----------------------------------------------------


def test_the_horizon_tool_answers_over_the_wire(client: Client) -> None:
    payload = succeed(client, "bond_carry_rolldown", horizon_call())
    assert payload["arbitrageFree"] is True
    assert payload["carry"] == pytest.approx(payload["financingCost"], abs=1e-9)
    assert payload["totalReturnBasisPoints"] == pytest.approx(279.0, abs=1.0)


def test_the_horizon_tool_works_from_a_curve_handle(client: Client) -> None:
    """A caller bootstraps once and asks several questions, which is the whole
    reason handles exist. The handle has to survive serialisation to get here.
    """
    built = succeed(
        client,
        "bootstrap_discount_curve",
        {
            "reference": "2021-01-05",
            "basis": "ACT_365F",
            "deposits": [{"maturity": "2021-07-05", "rate": 0.0020, "basis": "ACT_360"}],
            "swaps": [
                swap("2023-01-05", 0.0060),
                swap("2026-01-05", 0.0150),
                swap("2031-01-05", 0.0220),
                swap("2041-01-07", 0.0270),
            ],
        },
    )
    direct = succeed(client, "bond_carry_rolldown", horizon_call())
    from_handle = succeed(
        client,
        "bond_carry_rolldown",
        {
            "handle": built["handle"],
            "bond": horizon_call()["bond"],
            "horizon": "2022-01-05",
        },
    )
    assert from_handle["rollDown"] == pytest.approx(direct["rollDown"])
    assert from_handle["arbitrageFree"] is True


def test_a_horizon_past_maturity_is_a_recoverable_refusal(client: Client) -> None:
    error = be_refused(client, "bond_carry_rolldown", horizon_call(horizon="2032-01-05"))
    assert "no price at the horizon" in error["message"]


def test_a_malformed_horizon_date_is_a_recoverable_refusal(client: Client) -> None:
    be_refused(client, "bond_carry_rolldown", horizon_call(horizon="not-a-date"))


def test_a_missing_horizon_is_a_recoverable_refusal(client: Client) -> None:
    args = horizon_call()
    del args["horizon"]
    be_refused(client, "bond_carry_rolldown", args)


# -- and once against a real subprocess --------------------------------------


def test_both_tools_answer_a_launched_server() -> None:
    """The same calls through framing, a pipe and a separate interpreter.

    In-process transports share objects; this one shares only bytes. The
    curve handle is the case where that matters most — a handle spent after a
    lossy encode fails its integrity check rather than returning a wrong
    number — so it is round-tripped here rather than only in process.
    """
    command = [sys.executable, "-m", "abacus.cli", "stdio"]
    with StdioTransport(command) as transport, Client(transport) as connected:
        assert connected.discover().ok
        validation = succeed(connected, "validate_risk_model", validation_call())
        assert validation["trafficLight"]["zone"] == "green"

        built = succeed(
            connected,
            "bootstrap_discount_curve",
            {
                "reference": "2021-01-05",
                "basis": "ACT_365F",
                "deposits": [
                    {"maturity": "2021-07-05", "rate": 0.0020, "basis": "ACT_360"}
                ],
                "swaps": [
                    swap("2023-01-05", 0.0060),
                    swap("2026-01-05", 0.0150),
                    swap("2031-01-05", 0.0220),
                    swap("2041-01-07", 0.0270),
                ],
            },
        )
        horizon = succeed(
            connected,
            "bond_carry_rolldown",
            {
                "handle": built["handle"],
                "bond": horizon_call()["bond"],
                "horizon": "2022-01-05",
            },
        )
        assert horizon["arbitrageFree"] is True
        assert horizon["totalReturnBasisPoints"] == pytest.approx(279.0, abs=1.0)

        # And a refusal, which has to cross the same framing.
        error = be_refused(
            connected, "bond_carry_rolldown", horizon_call(horizon="2032-01-05")
        )
        assert "maturity" in error["message"]


# -- conditional_volatility, and the workflow over the wire ------------------


def garch_path(count: int = 1200, *, seed: int = 8) -> list[float]:
    rng = random.Random(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    variance = omega / (1.0 - alpha - beta)
    out: list[float] = []
    for _ in range(count + 500):
        value = math.sqrt(variance) * rng.gauss(0.0, 1.0)
        out.append(value)
        variance = omega + alpha * value * value + beta * variance
    return out[500:]


def regime_path(count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    out: list[float] = []
    turbulent = False
    for _ in range(count):
        turbulent = rng.random() < (0.90 if turbulent else 0.02)
        out.append(rng.gauss(0.0, 0.030 if turbulent else 0.006))
    return out


def test_the_volatility_tool_is_listed(client: Client) -> None:
    assert "conditional_volatility" in client.tool_names()


def test_the_fit_answers_over_the_wire(client: Client) -> None:
    payload = succeed(client, "conditional_volatility", {"returns": garch_path()})
    assert payload["converged"] is True
    assert 0.0 < payload["persistence"] < 1.0
    assert len(payload["volatilityForecasts"]) == 1200


def test_a_long_forecast_series_survives_serialisation(client: Client) -> None:
    """Twelve hundred floats through JSON and back. A value the encoder could
    not carry would fail here and nowhere else."""
    payload = succeed(client, "conditional_volatility", {"returns": garch_path(1200)})
    series = payload["volatilityForecasts"]
    assert all(isinstance(value, float) and value > 0.0 for value in series)
    assert all(math.isfinite(value) for value in series)


def test_a_sample_too_short_is_a_recoverable_refusal(client: Client) -> None:
    be_refused(client, "conditional_volatility", {"returns": garch_path(40)})


def test_an_unknown_field_is_a_recoverable_refusal_here_too(client: Client) -> None:
    be_refused(
        client, "conditional_volatility", {"returns": garch_path(400), "decay": 0.94}
    )


def test_the_workflow_runs_end_to_end_over_the_wire(client: Client) -> None:
    """Fit the process with one tool, score it with the other, and watch the
    clustering verdict change — with no offsetting between them, which is the
    property the first tool exists to guarantee.
    """
    # Seed 40 is representative rather than lucky: over the ten seeds 40-49
    # the constant forecast is rejected on nine and the conditional one on a
    # single seed. The statistical claim is asserted in aggregate in
    # test_model_validation.py; this is one instance of it, carried over the
    # wire, and it is the eight-in-ten case rather than the exception.
    observed = regime_path(2000, seed=40)
    quantile = -normal_ppf(TAIL)
    level = math.sqrt(math.fsum(value * value for value in observed) / len(observed))

    flat = succeed(
        client,
        "validate_risk_model",
        {
            "returns": observed,
            "valueAtRisk": [quantile * level] * len(observed),
            "confidence": CONFIDENCE,
        },
    )
    fitted = succeed(client, "conditional_volatility", {"returns": observed})
    conditional = succeed(
        client,
        "validate_risk_model",
        {
            "returns": observed,
            "valueAtRisk": [quantile * v for v in fitted["volatilityForecasts"]],
            "confidence": CONFIDENCE,
        },
    )

    before = {t["name"]: t for t in flat["tests"]}["independence"]
    after = {t["name"]: t for t in conditional["tests"]}["independence"]
    assert before["rejectsAt5Percent"] is True
    assert after["rejectsAt5Percent"] is False
    assert after["pValue"] > before["pValue"]
    assert conditional["breaches"] < flat["breaches"]
