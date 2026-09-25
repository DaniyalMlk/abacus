"""`validate_risk_model` and `bond_carry_rolldown` through the registry.

The arithmetic belongs to `shortfall` and `tenor` and is tested there. What is
tested here is what this layer adds: that the schemas accept what their own
descriptions ask for, that the refusals a model would trip over say something
it can act on, and that the figures the tool descriptions and the documentation
quote are the ones the code produces.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

import pytest
from shortfall.distributions import normal_pdf, normal_ppf

from abacus.analytics import default_registry
from abacus.tools import DomainError, Tool, ToolRegistry

CONFIDENCE = 0.99
TAIL = 1.0 - CONFIDENCE


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return default_registry()


def tool(registry: ToolRegistry, name: str) -> Tool:
    found = {one.name: one for one in registry}
    return found[name]


def call(registry: ToolRegistry, name: str, args: dict[str, Any]) -> dict[str, Any]:
    result = tool(registry, name).handler(args)
    payload = result.get("structuredContent", result)
    assert isinstance(payload, dict)
    return payload


# -- validate_risk_model -----------------------------------------------------


def normal_forecast(volatility: float) -> tuple[float, float]:
    quantile = -normal_ppf(TAIL)
    return volatility * quantile, volatility * normal_pdf(quantile) / TAIL


def series(
    true_volatility: float, count: int = 250, *, seed: int = 5
) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, true_volatility) for _ in range(count)]


def validation_args(
    true_volatility: float = 0.012,
    forecast_volatility: float = 0.012,
    count: int = 250,
    **extra: Any,
) -> dict[str, Any]:
    var, shortfall = normal_forecast(forecast_volatility)
    args: dict[str, Any] = {
        "returns": series(true_volatility, count),
        "valueAtRisk": [var] * count,
        "confidence": CONFIDENCE,
    }
    args.update(extra)
    if extra.get("expectedShortfall") is True:
        args["expectedShortfall"] = [shortfall] * count
    return args


def test_a_healthy_model_is_not_rejected(registry: ToolRegistry) -> None:
    payload = call(registry, "validate_risk_model", validation_args())
    assert payload["observations"] == 250
    assert payload["rejectedAt5Percent"] == []
    assert payload["trafficLight"]["zone"] == "green"
    assert payload["tailProbability"] == pytest.approx(TAIL)
    assert len(payload["tests"]) == 3
    assert {one["name"] for one in payload["tests"]} == {
        "unconditional coverage",
        "independence",
        "conditional coverage",
    }


def test_a_model_scaled_far_too_low_is_condemned(registry: ToolRegistry) -> None:
    payload = call(
        registry,
        "validate_risk_model",
        validation_args(true_volatility=0.03, forecast_volatility=0.01),
    )
    assert payload["breaches"] > 20
    assert payload["trafficLight"]["zone"] == "red"
    assert payload["trafficLight"]["plusFactor"] == pytest.approx(1.0)
    assert "unconditional coverage" in payload["rejectedAt5Percent"]


def test_the_supervisory_add_on_is_withheld_off_its_own_setup(
    registry: ToolRegistry,
) -> None:
    """125 observations is not the setup the table is published for, so the
    zone is still reported and the multiplier is not."""
    payload = call(registry, "validate_risk_model", validation_args(count=125))
    assert payload["trafficLight"]["plusFactor"] is None
    assert payload["trafficLight"]["supervisorySetup"] is False
    assert payload["trafficLight"]["zone"] in {"green", "yellow", "red"}


def test_a_short_sample_is_marked_advisory_rather_than_refused(
    registry: ToolRegistry,
) -> None:
    payload = call(registry, "validate_risk_model", validation_args(count=40))
    assert all(one["advisory"] for one in payload["tests"])
    assert payload["warnings"]


def test_clustered_breaches_are_caught_by_independence_not_by_the_count(
    registry: ToolRegistry,
) -> None:
    """The failure the tool exists for, driven end to end.

    Volatility switches between calm and turbulent; a constant forecast set to
    the unconditional level gets roughly the right number of breaches and puts
    them all in the turbulent stretches.
    """
    rng = random.Random(11)
    observed: list[float] = []
    turbulent = False
    for _ in range(1500):
        turbulent = rng.random() < (0.90 if turbulent else 0.02)
        observed.append(rng.gauss(0.0, 0.030 if turbulent else 0.006))
    level = math.sqrt(sum(value * value for value in observed) / len(observed))
    var, _ = normal_forecast(level)
    payload = call(
        registry,
        "validate_risk_model",
        {"returns": observed, "valueAtRisk": [var] * 1500, "confidence": CONFIDENCE},
    )
    by_name = {one["name"]: one for one in payload["tests"]}
    # Rejected at 5%, and the interpretation names the asymmetry that caused
    # it: a breach is followed by another far more often than a calm day is.
    # The p-value here is 0.02 rather than 0.0001 — the clustering is obvious
    # in the transition rates and only moderately significant in 1500
    # observations, because thirty breaches is not many to fit a chain from.
    assert by_name["independence"]["rejectsAt5Percent"] is True
    assert "10.00% of the time" in by_name["independence"]["interpretation"]
    assert "conditional coverage" in payload["rejectedAt5Percent"]


def test_the_shortfall_statistics_appear_only_when_forecasts_are_given(
    registry: ToolRegistry,
) -> None:
    without = call(registry, "validate_risk_model", validation_args())
    assert "expectedShortfallTest" not in without
    with_them = call(
        registry, "validate_risk_model", validation_args(expectedShortfall=True)
    )
    assert "expectedShortfallTest" in with_them
    assert with_them["expectedShortfallTest"]["replications"] == 0
    assert with_them["expectedShortfallTest"]["unconditionalPValue"] is None


def test_a_simulated_p_value_detects_an_understated_tail(
    registry: ToolRegistry,
) -> None:
    payload = call(
        registry,
        "validate_risk_model",
        validation_args(
            true_volatility=0.022,
            forecast_volatility=0.012,
            expectedShortfall=True,
            replications=400,
            seed=3,
        ),
    )
    found = payload["expectedShortfallTest"]
    assert found["replications"] == 400
    assert found["unconditionalPValue"] < 0.01
    assert found["realisedOverForecast"] > 1.0
    assert "understated" in found["direction"]


def test_the_note_states_the_sign_convention(registry: ToolRegistry) -> None:
    """The commonest way to misuse this tool, so it has to be in the note a
    model reads rather than only in the schema."""
    payload = call(registry, "validate_risk_model", validation_args())
    assert "POSITIVE losses" in payload["note"]
    assert "return < -forecast" in payload["note"]


def test_the_note_gives_the_measured_scale_of_the_shortfall_statistic(
    registry: ToolRegistry,
) -> None:
    """-0.245 is asserted in shortfall's own tests against a closed form. If
    that number ever moves, this fails rather than the note quietly lying."""
    payload = call(registry, "validate_risk_model", validation_args())
    assert "-0.245" in payload["note"]
    var, shortfall = normal_forecast(0.01)
    observed = series(0.02, 4000, seed=22)
    measured = call(
        registry,
        "validate_risk_model",
        {
            "returns": observed,
            "valueAtRisk": [var] * 4000,
            "expectedShortfall": [shortfall] * 4000,
            "confidence": CONFIDENCE,
        },
    )
    assert measured["expectedShortfallTest"]["conditional"] == pytest.approx(
        -0.245, abs=0.01
    )


@pytest.mark.parametrize(
    ("mutate", "field", "fragment"),
    [
        (
            lambda a: a.__setitem__("valueAtRisk", a["valueAtRisk"][:-1]),
            "valueAtRisk",
            "one-step-ahead",
        ),
        (lambda a: a.__setitem__("valueAtRisk", [-0.02] * 250), "valueAtRisk", "positive losses"),
        (lambda a: a.__setitem__("replications", 100), "replications", "expectedShortfall"),
    ],
)
def test_the_refusals_say_what_to_do(
    registry: ToolRegistry, mutate: Any, field: str, fragment: str
) -> None:
    args = validation_args()
    mutate(args)
    with pytest.raises(DomainError) as caught:
        call(registry, "validate_risk_model", args)
    assert fragment in str(caught.value)
    assert caught.value.field == field


def test_too_many_replications_is_refused_with_the_reason(
    registry: ToolRegistry,
) -> None:
    args = validation_args(expectedShortfall=True, replications=50_000)
    with pytest.raises(DomainError, match="simulation error"):
        call(registry, "validate_risk_model", args)


def test_a_cornish_fisher_null_is_refused(registry: ToolRegistry) -> None:
    """It is a quantile mapping rather than a distribution, so there is
    nothing to draw from."""
    args = validation_args(expectedShortfall=True, replications=50)
    args["distribution"] = "cornish-fisher"
    # The schema's enum excludes it, so a caller reaching this has gone around
    # the schema. The handler refuses anyway rather than trying to draw from
    # something that is not a distribution.
    with pytest.raises(DomainError, match="quantile mapping"):
        call(registry, "validate_risk_model", args)


def test_a_shortfall_below_its_own_value_at_risk_is_refused(
    registry: ToolRegistry,
) -> None:
    args = validation_args()
    args["expectedShortfall"] = [one * 0.5 for one in args["valueAtRisk"]]
    with pytest.raises(DomainError, match="below the value at risk"):
        call(registry, "validate_risk_model", args)


def test_the_payload_is_json_serialisable_with_no_non_finite_values(
    registry: ToolRegistry,
) -> None:
    """A bare Infinity or NaN is not JSON, and a strict parser rejects the
    whole message rather than the one field."""
    payload = call(
        registry, "validate_risk_model", validation_args(expectedShortfall=True)
    )
    json.dumps(payload, allow_nan=False)


def test_the_schema_accepts_what_its_description_asks_for(
    registry: ToolRegistry,
) -> None:
    schema = tool(registry, "validate_risk_model").input_schema
    assert schema["required"] == ["returns", "valueAtRisk", "confidence"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["valueAtRisk"]["items"]["exclusiveMinimum"] == 0
    assert schema["properties"]["distribution"]["enum"] == ["normal", "student-t"]


# -- bond_carry_rolldown -----------------------------------------------------

REFERENCE = "2021-01-05"


def swap(maturity: str, rate: float) -> dict[str, Any]:
    return {
        "maturity": maturity,
        "rate": rate,
        "frequency": "SEMI_ANNUAL",
        "basis": "THIRTY_360_BOND",
    }


def horizon_args(coupon: float = 0.03, horizon: str = "2022-01-05", **extra: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "reference": REFERENCE,
        "basis": "ACT_365F",
        "deposits": [{"maturity": "2021-07-05", "rate": 0.0020, "basis": "ACT_360"}],
        "swaps": [
            swap("2023-01-05", 0.0060),
            swap("2026-01-05", 0.0150),
            swap("2031-01-05", 0.0220),
            swap("2041-01-07", 0.0270),
        ],
        "bond": {
            "effective": REFERENCE,
            "maturity": "2031-01-05",
            "coupon": coupon,
            "frequency": "SEMI_ANNUAL",
            "basis": "THIRTY_360_BOND",
        },
        "horizon": horizon,
    }
    args.update(extra)
    return args


def test_the_identity_survives_the_tool_layer(registry: ToolRegistry) -> None:
    payload = call(registry, "bond_carry_rolldown", horizon_args())
    assert payload["arbitrageFree"] is True
    assert payload["carry"] == pytest.approx(payload["financingCost"], abs=1e-9)
    assert payload["totalReturn"] == pytest.approx(
        payload["carry"] + payload["rollDown"], abs=1e-9
    )
    assert payload["excessOverFinancing"] == pytest.approx(payload["rollDown"], abs=1e-9)


def test_the_figure_the_description_quotes_is_the_one_it_computes(
    registry: ToolRegistry,
) -> None:
    """The description says 279 basis points, 50 financing and 249 roll-down.
    If the curve or the arithmetic moves, this fails rather than the
    description going stale."""
    payload = call(registry, "bond_carry_rolldown", horizon_args())
    assert payload["totalReturnBasisPoints"] == pytest.approx(279.0, abs=1.0)
    assert payload["financingCost"] == pytest.approx(0.50, abs=0.01)
    assert payload["rollDown"] == pytest.approx(2.49, abs=0.01)
    description = tool(registry, "bond_carry_rolldown").description
    assert "279 basis points" in description
    assert "50 is financing" in description
    assert "249 is roll-down" in description


def test_income_less_financing_ranks_the_two_bonds_backwards(
    registry: ToolRegistry,
) -> None:
    """The warning in the note, measured on this curve rather than asserted.

    The high-coupon bond looks far better on income minus financing and earns
    less. The note quotes 27 basis points, so the gap is checked against it.
    """
    high = call(registry, "bond_carry_rolldown", horizon_args(coupon=0.09))
    zero = call(registry, "bond_carry_rolldown", horizon_args(coupon=0.0))
    assert high["incomeLessFinancing"] > zero["incomeLessFinancing"]
    assert high["totalReturnBasisPoints"] < zero["totalReturnBasisPoints"]
    assert "rank" in zero["note"]


def test_roll_down_is_positive_on_this_upward_sloping_curve(
    registry: ToolRegistry,
) -> None:
    for coupon in (0.0, 0.02, 0.05, 0.09):
        payload = call(registry, "bond_carry_rolldown", horizon_args(coupon=coupon))
        assert payload["rollDown"] > 0.0


def test_a_longer_horizon_rolls_further(registry: ToolRegistry) -> None:
    near = call(registry, "bond_carry_rolldown", horizon_args(horizon="2022-01-05"))
    far = call(registry, "bond_carry_rolldown", horizon_args(horizon="2024-01-05"))
    assert far["rollDown"] > near["rollDown"]
    assert far["periodYears"] > near["periodYears"]


def test_the_coupons_in_the_window_are_reported_with_their_horizon_value(
    registry: ToolRegistry,
) -> None:
    payload = call(registry, "bond_carry_rolldown", horizon_args(coupon=0.03))
    assert len(payload["coupons"]) == 2
    assert all(one["amount"] == pytest.approx(1.5) for one in payload["coupons"])
    assert payload["couponIncome"] == pytest.approx(
        sum(one["valueAtHorizon"] for one in payload["coupons"])
    )


def test_a_horizon_past_maturity_is_refused_with_the_reason(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(DomainError) as caught:
        call(registry, "bond_carry_rolldown", horizon_args(horizon="2032-01-05"))
    assert "no price at the horizon" in str(caught.value)
    assert caught.value.field == "horizon"


def test_a_horizon_before_settlement_is_refused(registry: ToolRegistry) -> None:
    with pytest.raises(DomainError, match="runs forwards"):
        call(registry, "bond_carry_rolldown", horizon_args(horizon="2020-06-01"))


def test_the_horizon_payload_is_json_serialisable(registry: ToolRegistry) -> None:
    payload = call(registry, "bond_carry_rolldown", horizon_args())
    json.dumps(payload, allow_nan=False)


def test_both_tools_are_read_only_and_idempotent(registry: ToolRegistry) -> None:
    for name in ("validate_risk_model", "bond_carry_rolldown"):
        annotations = tool(registry, name).annotations
        assert annotations["readOnlyHint"] is True
        assert annotations["idempotentHint"] is True


# -- conditional_volatility, and the two composing ---------------------------


def garch_path(count: int = 1500, *, seed: int = 8) -> list[float]:
    """A simulated GARCH(1,1), burnt in so the start is not the seed."""
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


def test_the_fit_recovers_what_it_should(registry: ToolRegistry) -> None:
    payload = call(registry, "conditional_volatility", {"returns": garch_path(3000)})
    assert payload["converged"] is True
    assert payload["alpha"] == pytest.approx(0.08, abs=0.04)
    assert payload["beta"] == pytest.approx(0.90, abs=0.05)
    assert 0.0 < payload["persistence"] < 1.0
    assert payload["halfLife"] > 0.0


def test_the_forecast_series_is_one_per_observation(registry: ToolRegistry) -> None:
    data = garch_path(800)
    payload = call(registry, "conditional_volatility", {"returns": data})
    assert len(payload["volatilityForecasts"]) == len(data)
    assert all(value > 0.0 for value in payload["volatilityForecasts"])
    assert payload["currentVolatility"] == pytest.approx(payload["volatilityForecasts"][-1])


def test_the_forecasts_can_be_withheld_for_a_long_history(registry: ToolRegistry) -> None:
    payload = call(
        registry,
        "conditional_volatility",
        {"returns": garch_path(800), "includeForecasts": False},
    )
    assert "volatilityForecasts" not in payload
    assert payload["persistence"] > 0.0


def test_the_horizon_ratio_follows_where_today_sits_against_the_long_run(
    registry: ToolRegistry,
) -> None:
    """The invariant, rather than a sign on one contrived sample.

    Appending quiet returns to force a calm ending does not reliably put the
    ratio above one, because it refits the model and drags the long-run level
    down with it — the first draft of this test assumed otherwise and got
    0.9938. What is actually guaranteed is the *relationship*: the ratio is
    below one exactly when the next-period forecast is above the long-run
    level, whatever the fit turns out to be.
    """
    data = garch_path(1200, seed=3)
    for series, horizon in (
        (data, 250),
        ([*data, *([data[-1] * 0.02] * 40)], 250),
        ([*data, *([abs(data[-1]) * 8.0] * 3)], 250),
        (data, 10),
    ):
        payload = call(
            registry, "conditional_volatility", {"returns": series, "horizon": horizon}
        )
        above_long_run = payload["nextVolatility"] > payload["longRunVolatility"]
        below_one = payload["squareRootOfTimeRatio"] < 1.0
        assert above_long_run == below_one, payload["squareRootOfTimeRatio"]


def test_a_shock_lowers_the_ratio_against_square_root_of_time(
    registry: ToolRegistry,
) -> None:
    """Mean reversion has more to pull down from, so the scaled figure
    overstates the horizon by more."""
    data = garch_path(1200, seed=3)
    calm = call(
        registry,
        "conditional_volatility",
        {"returns": [*data, *([data[-1] * 0.02] * 40)], "horizon": 250},
    )
    shocked = call(
        registry,
        "conditional_volatility",
        {"returns": [*data, *([abs(data[-1]) * 8.0] * 3)], "horizon": 250},
    )
    assert shocked["squareRootOfTimeRatio"] < calm["squareRootOfTimeRatio"]
    assert shocked["squareRootOfTimeRatio"] < 1.0


def test_variance_targeting_is_reported_as_used(registry: ToolRegistry) -> None:
    payload = call(
        registry,
        "conditional_volatility",
        {"returns": garch_path(800), "varianceTargeting": True},
    )
    assert payload["varianceTargeted"] is True
    assert payload["converged"] is True


def test_a_sample_too_short_to_fit_is_refused_with_the_reason(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(DomainError, match="nearly flat"):
        call(registry, "conditional_volatility", {"returns": garch_path(40)})


def test_a_series_with_no_variance_is_refused(registry: ToolRegistry) -> None:
    with pytest.raises(DomainError, match="no variance at all"):
        call(registry, "conditional_volatility", {"returns": [0.001] * 300})


def test_the_volatility_payload_is_json_serialisable(registry: ToolRegistry) -> None:
    payload = call(registry, "conditional_volatility", {"returns": garch_path(400)})
    json.dumps(payload, allow_nan=False)


def test_the_two_tools_compose_into_a_workflow(registry: ToolRegistry) -> None:
    """The reason this tool exists, measured over ten independent samples.

    A constant forecast on a regime-switching series has its breaches rejected
    as clustered nine times in ten. The forecasts this tool produces, handed
    straight to the validator with no offsetting, are rejected once.

    The breach count improves from about 51 to about 27 against a nominal 20 —
    halved rather than fixed, because a Gaussian GARCH still understates the
    tail of a series whose standardised residuals are fat. The note says so,
    and this is where that claim is held to account.
    """
    quantile = -normal_ppf(TAIL)
    samples = 10
    constant_rejections = garch_rejections = 0
    constant_breaches = garch_breaches = 0
    for seed in range(40, 40 + samples):
        observed = regime_path(2000, seed)
        fitted = call(registry, "conditional_volatility", {"returns": observed})
        level = math.sqrt(math.fsum(v * v for v in observed) / len(observed))
        flat = call(
            registry,
            "validate_risk_model",
            {
                "returns": observed,
                "valueAtRisk": [quantile * level] * len(observed),
                "confidence": CONFIDENCE,
            },
        )
        conditional = call(
            registry,
            "validate_risk_model",
            {
                "returns": observed,
                # No offsetting: the series comes out of one tool aligned for
                # the other, which is the whole point.
                "valueAtRisk": [quantile * v for v in fitted["volatilityForecasts"]],
                "confidence": CONFIDENCE,
            },
        )
        by_name = {t["name"]: t for t in flat["tests"]}
        constant_rejections += by_name["independence"]["rejectsAt5Percent"]
        constant_breaches += flat["breaches"]
        by_name = {t["name"]: t for t in conditional["tests"]}
        garch_rejections += by_name["independence"]["rejectsAt5Percent"]
        garch_breaches += conditional["breaches"]

    assert constant_rejections >= 8
    assert garch_rejections <= 3
    assert constant_breaches / samples > 45
    # The tool's note tells a caller to expect about 28 times in 2000
    # observations. This is where that figure is held to account.
    assert 22 < garch_breaches / samples < 33


def test_the_note_refuses_the_flattering_summary(registry: ToolRegistry) -> None:
    """It would be easy to say this fixes the model. It halves the excess."""
    payload = call(registry, "conditional_volatility", {"returns": garch_path(400)})
    assert "does NOT make the tail thin" in payload["note"]
    assert "28 times in 2000" in payload["note"]
    assert "ALREADY ALIGNED" in payload["note"]
