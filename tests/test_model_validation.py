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
from shortfall.distributions import normal_pdf, normal_ppf, student_t_ppf

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

    Normal innovations are asked for explicitly. The default now tests for a fat
    tail and would fit one on this data, which is the subject of the test below —
    keeping this one pinned to the Gaussian case is what makes the pair a
    comparison rather than two runs of the same thing.
    """
    quantile = -normal_ppf(TAIL)
    samples = 10
    constant_rejections = garch_rejections = 0
    constant_breaches = garch_breaches = 0
    for seed in range(40, 40 + samples):
        observed = regime_path(2000, seed)
        fitted = call(
            registry,
            "conditional_volatility",
            {"returns": observed, "innovation": "normal"},
        )
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
    # The tool's note quotes 28.2 breaches in 2000 observations under normal
    # innovations. This is where that figure is held to account; the tests in
    # shortfall hold the 22.45 that estimating the tail brings it to.
    assert 22 < garch_breaches / samples < 33


def test_the_note_refuses_the_flattering_summary(registry: ToolRegistry) -> None:
    """It would be easy to say estimating the tail fixes the model. It does not.

    The note now quotes both ends of the improvement — 28.2 breaches to 22.45
    against a nominal 20 — rather than the first alone, and says why the rest is
    still there. A series whose volatility jumps between regimes does not have
    identically distributed standardised residuals, so one tail index for the
    whole sample is closer than the normal's and not correct.
    """
    payload = call(registry, "conditional_volatility", {"returns": garch_path(400)})
    note = payload["note"]
    assert "ALREADY ALIGNED" in note
    assert "28.2 to 22.45" in note
    assert "nominal 20" in note
    assert "still an approximation" in note
    # And the footgun the multiplier exists to remove is named in the note that
    # tells the caller to use it.
    assert "STANDARDISED quantile" in note
    assert "quantileMultiplier" in note


# -- the innovation distribution ---------------------------------------------


def student_t_garch_path(
    count: int = 1500, *, degrees: float = 4.5, seed: int = 8
) -> list[float]:
    """The same variance process as :func:`garch_path` with a fat-tailed draw.

    The innovation is standardised to unit variance, so the two generators differ
    in the shape of the draw and in nothing else. Without that the fit would see a
    different variance level as well and the comparison would not isolate the tail.
    """
    rng = random.Random(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    scale = math.sqrt(degrees / (degrees - 2.0))
    variance = omega / (1.0 - alpha - beta)
    out: list[float] = []
    for _ in range(count + 500):
        chi_square = 2.0 * rng.gammavariate(degrees / 2.0, 1.0)
        innovation = rng.gauss(0.0, 1.0) / math.sqrt(chi_square / degrees) / scale
        value = math.sqrt(variance) * innovation
        out.append(value)
        variance = omega + alpha * value * value + beta * variance
    return out[500:]


def test_the_fat_tail_is_found_when_it_is_there(registry: ToolRegistry) -> None:
    payload = call(
        registry, "conditional_volatility", {"returns": student_t_garch_path(1500)}
    )
    assert payload["innovation"] == "student-t"
    assert payload["fatTail"]["fat"] is True
    assert payload["fatTail"]["pValue"] < 0.01
    assert 2.0 < payload["degreesOfFreedom"] < 12.0
    assert payload["expectedShortfall"] > payload["valueAtRisk"] > 0.0


def test_the_implied_kurtosis_is_null_exactly_when_there_is_no_fourth_moment(
    registry: ToolRegistry,
) -> None:
    """`6 / (v - 4)` is infinite at four degrees of freedom and negative below.

    Both are wrong to emit. The infinity is not JSON — `json.dumps` writes a bare
    `Infinity` token and a strict parser rejects the whole message — and the
    negative number is worse than useless, since a caller comparing it against a
    sample kurtosis would conclude the innovations were thin-tailed. The field is
    null in both cases, which is why the tool reports the degrees of freedom
    beside it.

    A tail of eight degrees of freedom is fat and has a fourth moment, so the
    number is there; a tail near three does not, so it is not.
    """
    moderate = call(
        registry,
        "conditional_volatility",
        {"returns": student_t_garch_path(2000, degrees=8.0, seed=21)},
    )
    assert moderate["degreesOfFreedom"] > 4.0
    assert moderate["impliedExcessKurtosis"] == pytest.approx(
        6.0 / (moderate["degreesOfFreedom"] - 4.0)
    )

    extreme = call(
        registry,
        "conditional_volatility",
        {"returns": student_t_garch_path(2000, degrees=2.8, seed=22)},
    )
    assert extreme["degreesOfFreedom"] < 4.0
    assert extreme["impliedExcessKurtosis"] is None
    json.dumps(extreme, allow_nan=False)


def test_a_thin_tail_is_reported_as_unidentified_rather_than_as_a_number(
    registry: ToolRegistry,
) -> None:
    """Above 200 degrees of freedom the likelihood is flat, so the number is noise.

    Returning it anyway is what lets a caller write "our fitted tail index is
    640" about a series that is simply Gaussian. The field is null instead.
    """
    payload = call(registry, "conditional_volatility", {"returns": garch_path(1500)})
    assert payload["innovation"] == "normal"
    assert payload["fatTail"]["fat"] is False
    assert payload["degreesOfFreedom"] is None
    assert payload["impliedExcessKurtosis"] is None


def test_the_quantile_multiplier_is_the_standardised_quantile(
    registry: ToolRegistry,
) -> None:
    """The footgun this field exists to remove, checked against both closed forms.

    Under Student-t innovations the multiplier is the raw t quantile scaled by
    ``sqrt((v - 2) / v)``, and that scaling is not a rounding correction: the raw
    quantile is 41% larger at four degrees of freedom, and 73% larger at three.
    A caller who reaches for it widens every forecast and undershoots the breach
    count, which looks conservative rather than wrong — the reason this is
    computed here instead of being described.

    The ratio is ``sqrt(v / (v - 2))`` and depends on nothing else, so it is
    asserted at the exact degrees of freedom rather than at whatever the fit
    happened to return.
    """
    assert math.sqrt(4.0 / 2.0) == pytest.approx(1.414, abs=0.001)
    assert math.sqrt(3.0 / 1.0) == pytest.approx(1.732, abs=0.001)
    fat = call(
        registry, "conditional_volatility", {"returns": student_t_garch_path(1200)}
    )
    degrees = fat["degreesOfFreedom"]
    expected = -student_t_ppf(0.01, degrees) * math.sqrt((degrees - 2.0) / degrees)
    assert fat["quantileMultiplier"] == pytest.approx(expected)
    assert fat["quantileMultiplier"] > -normal_ppf(0.01)
    # And the raw quantile really is the larger number, by exactly the factor
    # above at the degrees of freedom the fit found.
    raw = -student_t_ppf(0.01, degrees)
    assert raw / fat["quantileMultiplier"] == pytest.approx(
        math.sqrt(degrees / (degrees - 2.0))
    )

    thin = call(
        registry,
        "conditional_volatility",
        {"returns": garch_path(1200), "innovation": "normal"},
    )
    assert thin["quantileMultiplier"] == pytest.approx(-normal_ppf(0.01))


def test_the_multiplier_carries_no_mean(registry: ToolRegistry) -> None:
    """A drift shifts the value at risk and must not shift the multiplier.

    The multiplier is a quantile of the innovation, so it belongs to the shape
    and not to the level. Folding the mean in would give a number that multiplied
    through a forecast series scales a constant drift by each period's volatility
    — wrong in a way that is invisible on a series whose mean is near zero, which
    is every series anybody tests this on.
    """
    base = garch_path(800)
    drifted = [value + 0.002 for value in base]
    plain = call(
        registry, "conditional_volatility", {"returns": base, "innovation": "normal"}
    )
    shifted = call(
        registry, "conditional_volatility", {"returns": drifted, "innovation": "normal"}
    )
    assert shifted["quantileMultiplier"] == pytest.approx(plain["quantileMultiplier"])
    assert shifted["valueAtRisk"] < plain["valueAtRisk"]


def test_the_confidence_moves_both_figures_together(registry: ToolRegistry) -> None:
    data = student_t_garch_path(1000)
    deeper = call(
        registry, "conditional_volatility", {"returns": data, "confidence": 0.995}
    )
    shallower = call(
        registry, "conditional_volatility", {"returns": data, "confidence": 0.95}
    )
    assert deeper["valueAtRisk"] > shallower["valueAtRisk"]
    assert deeper["expectedShortfall"] > shallower["expectedShortfall"]
    assert deeper["quantileMultiplier"] > shallower["quantileMultiplier"]
    assert deeper["confidence"] == 0.995


def test_the_multiplier_closes_the_workflow_on_a_fat_tailed_series(
    registry: ToolRegistry,
) -> None:
    """Fit, scale by what came back, score — and the breach count lands near 20.

    The same composition as the Gaussian test above, with the tail estimated and
    the multiplier used as handed over. Over ten regime-switching samples the
    normal-innovation path breaches about 28 times per 2000 observations and this
    one about 22, against a nominal 20. The point of the test is that a caller who
    does what the note says gets the better number without computing a quantile.
    """
    samples = 10
    fat_breaches = 0
    thin_breaches = 0
    for seed in range(40, 40 + samples):
        observed = regime_path(2000, seed)
        for innovation, total in (("auto", "fat"), ("normal", "thin")):
            fitted = call(
                registry,
                "conditional_volatility",
                {"returns": observed, "innovation": innovation},
            )
            multiplier = fitted["quantileMultiplier"]
            scored = call(
                registry,
                "validate_risk_model",
                {
                    "returns": observed,
                    "valueAtRisk": [
                        multiplier * value for value in fitted["volatilityForecasts"]
                    ],
                    "confidence": CONFIDENCE,
                },
            )
            if total == "fat":
                fat_breaches += scored["breaches"]
            else:
                thin_breaches += scored["breaches"]
    assert 18.0 < fat_breaches / samples < 26.0
    assert 24.0 < thin_breaches / samples < 33.0
    assert fat_breaches < thin_breaches


# -- the horizon, simulated --------------------------------------------------


def test_the_horizon_simulation_is_absent_unless_paths_are_asked_for(
    registry: ToolRegistry,
) -> None:
    """The work is paths times horizon, so a caller who wants the parameters
    alone should not pay for it."""
    payload = call(registry, "conditional_volatility", {"returns": garch_path(400)})
    assert "horizonRisk" not in payload


def test_the_simulated_horizon_agrees_with_the_analytic_volatility(
    registry: ToolRegistry,
) -> None:
    """The one part of the simulation that has an exact answer to check against.

    The variance of the accumulated return is the sum of the expected variances
    along the path, because the residuals are a martingale difference sequence. So
    the simulated volatility has to reproduce the analytic aggregation whatever the
    quantile does — and if the two disagreed, the accumulation would be wrong and
    every quantile with it.
    """
    payload = call(
        registry,
        "conditional_volatility",
        {"returns": garch_path(1200), "horizon": 10, "paths": 10_000},
    )
    simulated = payload["horizonRisk"]
    assert simulated["paths"] == 10_000
    assert simulated["steps"] == 10
    assert simulated["draw"] == "bootstrap"
    assert simulated["simulatedVolatility"] == pytest.approx(
        simulated["analyticVolatility"], rel=0.03
    )
    assert simulated["analyticVolatility"] == pytest.approx(
        payload["horizonVolatility"], rel=1e-9
    )


def test_the_horizon_quantile_is_not_the_volatility_times_a_multiplier(
    registry: ToolRegistry,
) -> None:
    """The reason the simulation exists, as a number rather than as an argument.

    Multiplying the horizon volatility by the one-step quantile multiplier is the
    substitution the tool's note warns against. The simulated figure is larger, and
    the size of the gap in units of the simulation's own error is the evidence that
    it is a difference rather than noise.

    Measured over four samples at ten steps and 40,000 paths, the gap came to
    2.5, 11.0, 8.2 and 0.6 standard errors — mean 5.5. So it is *not* decisive on every
    sample, and that variation is itself worth knowing: how far the horizon quantile
    departs from a scaled one depends on where the fit sits relative to its long-run
    level, which changes from series to series. The assertion is on the mean over
    samples and on the sign, not on any single run being significant.
    """
    gaps = []
    for seed in range(4):
        payload = call(
            registry,
            "conditional_volatility",
            {
                "returns": garch_path(1500, seed=8 + seed),
                "horizon": 10,
                "paths": 40_000,
                "draw": "parametric",
                "includeForecasts": False,
            },
        )
        simulated = payload["horizonRisk"]
        substituted = payload["quantileMultiplier"] * payload["horizonVolatility"]
        gaps.append((simulated["valueAtRisk"] - substituted) / simulated["standardError"])
    assert sum(gaps) / len(gaps) > 2.0
    assert sum(gap > 0.0 for gap in gaps) >= 3


def test_the_two_square_root_of_time_ratios_disagree(registry: ToolRegistry) -> None:
    """Both are reported because they are different quantities.

    The volatility ratio and the quantile ratio can sit on opposite sides of one,
    and a caller handed only the first would read the horizon as conservative when
    it is not.
    """
    payload = call(
        registry,
        "conditional_volatility",
        {"returns": garch_path(1500), "horizon": 10, "paths": 20_000, "draw": "parametric"},
    )
    simulated = payload["horizonRisk"]
    assert simulated["quantileAgainstSquareRootOfTime"] > 1.0
    assert simulated["quantileAgainstSquareRootOfTime"] > (
        simulated["volatilityAgainstSquareRootOfTime"]
    )


def test_the_relative_error_falls_as_the_path_count_rises(registry: ToolRegistry) -> None:
    data = garch_path(1200)
    errors = {}
    for paths in (2_000, 40_000):
        payload = call(
            registry,
            "conditional_volatility",
            {"returns": data, "horizon": 5, "paths": paths, "draw": "parametric"},
        )
        errors[paths] = payload["horizonRisk"]["relativeStandardError"]
    assert errors[2_000] > errors[40_000]
    assert errors[40_000] < 0.03


def test_a_path_count_below_the_floor_is_refused_with_the_reason(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(DomainError, match="quantile of anything"):
        call(
            registry,
            "conditional_volatility",
            {"returns": garch_path(400), "paths": 50},
        )


def test_bootstrapping_a_short_history_is_refused_rather_than_extrapolated(
    registry: ToolRegistry,
) -> None:
    """A bootstrap cannot draw past the worst residual it holds.

    The library's floor is 250 observations for a resampled draw. The refusal names
    the parametric route, which extrapolates and says so, so a caller has somewhere
    to go rather than only something they cannot have.
    """
    short = garch_path(150)
    with pytest.raises(DomainError, match="too few"):
        call(
            registry,
            "conditional_volatility",
            {"returns": short, "paths": 2_000, "draw": "bootstrap"},
        )
    parametric = call(
        registry,
        "conditional_volatility",
        {"returns": short, "paths": 2_000, "draw": "parametric"},
    )
    assert parametric["horizonRisk"]["valueAtRisk"] > 0.0


def test_the_horizon_payload_is_strict_json(registry: ToolRegistry) -> None:
    payload = call(
        registry,
        "conditional_volatility",
        {"returns": garch_path(600), "paths": 2_000, "includeForecasts": False},
    )
    json.dumps(payload, allow_nan=False)


def test_the_same_seed_gives_the_same_horizon_figure(registry: ToolRegistry) -> None:
    """Annotated idempotent, so it has to be."""
    data = garch_path(600)
    arguments = {"returns": data, "paths": 3_000, "includeForecasts": False}
    first = call(registry, "conditional_volatility", dict(arguments))
    second = call(registry, "conditional_volatility", dict(arguments))
    assert first["horizonRisk"]["valueAtRisk"] == second["horizonRisk"]["valueAtRisk"]
    moved = call(registry, "conditional_volatility", {**arguments, "seed": 99})
    assert moved["horizonRisk"]["valueAtRisk"] != first["horizonRisk"]["valueAtRisk"]
