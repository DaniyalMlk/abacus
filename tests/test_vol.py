"""Implied volatility, slice calibration, surfaces and local volatility."""

from __future__ import annotations

import math
from typing import Any

import pytest
from moneyness import SVI

from abacus.analytics import default_registry
from abacus.tools import ToolRegistry
from abacus.vol import log_moneyness_on_forward

FORWARD = 100.0 * math.exp(0.04 * 1.0)

#: A slice known to be admissible, used to generate quotes that a fit should be
#: able to recover. Taking quotes from a real SVI slice rather than from an
#: invented smile means a failure to recover it is a failure of the fit rather
#: than an artefact of quotes no slice could match.
ADMISSIBLE = SVI(a=0.02, b=0.12, rho=-0.3, m=0.0, s=0.15)


@pytest.fixture
def registry() -> ToolRegistry:
    return default_registry()


def ok(registry: ToolRegistry, name: str, args: dict[str, Any]) -> Any:
    result = registry.call(name, args)
    assert result["isError"] is False, result["content"][0]["text"]
    return result["structuredContent"]


def refused(registry: ToolRegistry, name: str, args: dict[str, Any]) -> dict[str, Any]:
    result = registry.call(name, args)
    assert result["isError"] is True, "expected the call to be refused"
    error = result["structuredContent"]["error"]
    assert isinstance(error, dict)
    return error


def quotes_from(slice_: SVI, time: float, forward: float, strikes: list[float]) -> list[Any]:
    """Quotes whose *volatility* smile is the same at every maturity.

    The slice is read at a reference maturity of one year and the resulting
    volatilities are used unchanged, so total variance grows linearly with
    maturity. Reading the slice at `time` instead would hold total variance
    constant across maturities, which makes dw/dT zero — no calendar arbitrage,
    but no admissible Dupire local volatility either, since the identity needs a
    positive dw/dT. That is a property of the fixture, not of the surface code,
    and it is worth not baking into the tests.
    """
    return [
        {"strike": strike, "vol": slice_.volatility(math.log(strike / forward), 1.0)}
        for strike in strikes
    ]


STRIKES = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 135.0]


# -- implied volatility ---------------------------------------------------


@pytest.mark.parametrize("target", [0.08, 0.15, 0.27, 0.45, 0.9])
@pytest.mark.parametrize("side", ["call", "put"])
def test_a_price_round_trips_back_to_the_volatility_that_made_it(
    registry: ToolRegistry, target: float, side: str
) -> None:
    market = {"spot": 100.0, "strike": 110.0, "time": 0.75, "rate": 0.04, "type": side}
    price = ok(registry, "price_european_option", {**market, "vol": target})["price"]
    recovered = ok(registry, "implied_volatility", {**market, "price": price})
    assert recovered["impliedVol"] == pytest.approx(target, rel=1e-10)


def test_the_residual_travels_with_the_answer(registry: ToolRegistry) -> None:
    # A residual that is not tiny means the answer has not converged, whatever
    # the solver reported, so it is reported rather than dropped.
    market = {"spot": 100.0, "strike": 95.0, "time": 1.0, "rate": 0.03, "type": "call"}
    price = ok(registry, "price_european_option", {**market, "vol": 0.3})["price"]
    diagnostics = ok(registry, "implied_volatility", {**market, "price": price})["diagnostics"]
    assert abs(diagnostics["residual"]) < 1e-9
    assert diagnostics["iterations"] >= 1
    assert diagnostics["method"] in ("newton", "brent", "bound")


def test_deep_out_of_the_money_still_recovers(registry: ToolRegistry) -> None:
    # Where vega is nearly zero a naive Newton step diverges, so this is the
    # case that exercises the fallback.
    market = {"spot": 100.0, "strike": 250.0, "time": 0.25, "rate": 0.02, "type": "call"}
    price = ok(registry, "price_european_option", {**market, "vol": 0.6})["price"]
    assert ok(registry, "implied_volatility", {**market, "price": price})[
        "impliedVol"
    ] == pytest.approx(0.6, rel=1e-6)


def test_a_price_above_the_range_is_refused_with_the_range(registry: ToolRegistry) -> None:
    error = refused(
        registry,
        "implied_volatility",
        {"spot": 100.0, "strike": 110.0, "time": 0.75, "rate": 0.04, "price": 500.0,
         "type": "call"},
    )
    assert error["kind"] == "domain"
    assert "above the attainable range" in error["message"]
    assert "no implied volatility" in error["message"]


def test_a_price_below_the_range_is_refused(registry: ToolRegistry) -> None:
    # Below intrinsic. No non-negative volatility produces it.
    error = refused(
        registry,
        "implied_volatility",
        {"spot": 200.0, "strike": 100.0, "time": 1.0, "rate": 0.05, "price": 1.0,
         "type": "call"},
    )
    assert "below the attainable range" in error["message"]


def test_the_reported_log_moneyness_is_measured_on_the_forward(
    registry: ToolRegistry,
) -> None:
    # log(K / F), not log(S / K). A strike above the forward is positive here;
    # under the other convention it would be negative, mirroring every smile.
    market = {"spot": 100.0, "strike": 130.0, "time": 1.0, "rate": 0.04, "type": "call"}
    price = ok(registry, "price_european_option", {**market, "vol": 0.25})["price"]
    got = ok(registry, "implied_volatility", {**market, "price": price})
    assert got["logMoneyness"] == pytest.approx(math.log(130.0 / FORWARD), rel=1e-12)
    assert got["logMoneyness"] > 0


def test_total_variance_is_the_square_of_volatility_times_time(
    registry: ToolRegistry,
) -> None:
    market = {"spot": 100.0, "strike": 100.0, "time": 2.0, "rate": 0.0, "type": "call"}
    price = ok(registry, "price_european_option", {**market, "vol": 0.3})["price"]
    got = ok(registry, "implied_volatility", {**market, "price": price})
    assert got["totalVariance"] == pytest.approx(0.3**2 * 2.0, rel=1e-9)


# -- slice calibration ----------------------------------------------------


def test_a_fit_recovers_the_slice_its_quotes_came_from(registry: ToolRegistry) -> None:
    got = ok(
        registry,
        "fit_volatility_slice",
        {"time": 1.0, "forward": FORWARD, "quotes": quotes_from(ADMISSIBLE, 1.0, FORWARD, STRIKES)},
    )
    assert got["fit"]["rmse"] < 1e-6
    for k in (-0.4, -0.1, 0.0, 0.2, 0.5):
        assert got["atmVol"] if k == 0.0 else True
    fitted = SVI(**got["svi"])
    for k in (-0.4, -0.1, 0.0, 0.2, 0.5):
        assert fitted.total_variance(k) == pytest.approx(
            ADMISSIBLE.total_variance(k), abs=1e-6
        )


def test_the_fit_reports_each_quote_beside_its_fitted_value(
    registry: ToolRegistry,
) -> None:
    got = ok(
        registry,
        "fit_volatility_slice",
        {"time": 1.0, "forward": FORWARD, "quotes": quotes_from(ADMISSIBLE, 1.0, FORWARD, STRIKES)},
    )
    assert len(got["quotes"]) == len(STRIKES)
    for row in got["quotes"]:
        assert row["fittedVol"] == pytest.approx(row["quotedVol"], abs=1e-4)
        assert row["logMoneyness"] == pytest.approx(
            log_moneyness_on_forward(row["strike"], FORWARD), rel=1e-12
        )


def test_an_admissible_slice_is_reported_arbitrage_free(registry: ToolRegistry) -> None:
    got = ok(
        registry,
        "fit_volatility_slice",
        {"time": 1.0, "forward": FORWARD, "quotes": quotes_from(ADMISSIBLE, 1.0, FORWARD, STRIKES)},
    )
    assert got["arbitrage"]["butterflyFree"] is True
    assert got["arbitrage"]["worstDurrleman"] >= 0.0


def test_a_smile_steep_enough_to_imply_a_negative_density_is_flagged(
    registry: ToolRegistry,
) -> None:
    # A sharp kink at the money implies a negative probability density. The
    # fit succeeds — that is the danger — so the check has to be reported
    # rather than left to the residuals, which look fine.
    steep = [
        {"strike": strike, "vol": 0.20 + 2.5 * abs(math.log(strike / FORWARD))}
        for strike in STRIKES
    ]
    got = ok(registry, "fit_volatility_slice", {"time": 1.0, "forward": FORWARD, "quotes": steep})
    assert got["arbitrage"]["butterflyFree"] is False
    assert got["arbitrage"]["worstDurrleman"] < 0.0


def test_the_wing_slopes_are_checked_against_lees_bound(registry: ToolRegistry) -> None:
    got = ok(
        registry,
        "fit_volatility_slice",
        {"time": 1.0, "forward": FORWARD, "quotes": quotes_from(ADMISSIBLE, 1.0, FORWARD, STRIKES)},
    )
    assert got["wingSlopes"]["withinLeeBound"] is True
    assert max(got["wingSlopes"]["left"], got["wingSlopes"]["right"]) <= 2.0


def test_too_few_quotes_to_pin_five_parameters_is_refused(registry: ToolRegistry) -> None:
    error = refused(
        registry,
        "fit_volatility_slice",
        {"time": 1.0, "forward": FORWARD,
         "quotes": quotes_from(ADMISSIBLE, 1.0, FORWARD, [90.0, 100.0, 110.0])},
    )
    assert error["details"][0]["keyword"] == "minItems"


def test_a_repeated_strike_is_refused(registry: ToolRegistry) -> None:
    quotes = quotes_from(ADMISSIBLE, 1.0, FORWARD, STRIKES)
    quotes[1] = dict(quotes[0])
    error = refused(
        registry, "fit_volatility_slice", {"time": 1.0, "forward": FORWARD, "quotes": quotes}
    )
    assert "repeated" in error["message"]


def test_a_negative_quoted_volatility_is_refused(registry: ToolRegistry) -> None:
    quotes = quotes_from(ADMISSIBLE, 1.0, FORWARD, STRIKES)
    quotes[2] = {"strike": 90.0, "vol": -0.2}
    error = refused(
        registry, "fit_volatility_slice", {"time": 1.0, "forward": FORWARD, "quotes": quotes}
    )
    assert error["kind"] == "invalid_input"


# -- surfaces -------------------------------------------------------------


def _slice_input(time: float, slice_: SVI = ADMISSIBLE) -> dict[str, Any]:
    forward = 100.0 * math.exp(0.04 * time)
    return {
        "time": time,
        "forward": forward,
        "quotes": quotes_from(slice_, time, forward, STRIKES),
    }


def test_a_surface_returns_parameters_rather_than_a_grid(registry: ToolRegistry) -> None:
    # The default output carries no grid at all: five exact numbers per slice
    # beat a matrix that is easy to transpose and easy to truncate.
    got = ok(registry, "fit_volatility_surface", {"slices": [_slice_input(0.5), _slice_input(1.0)]})
    assert "grid" not in got
    assert [s["maturity"] for s in got["slices"]] == [0.5, 1.0]
    for described in got["slices"]:
        assert set(described["svi"]) == {"a", "b", "rho", "m", "s"}


def test_a_grid_is_returned_only_when_asked_for_and_is_labelled(
    registry: ToolRegistry,
) -> None:
    got = ok(
        registry,
        "fit_volatility_surface",
        {"slices": [_slice_input(0.5), _slice_input(1.0)], "strikeRatios": [0.9, 1.0, 1.1]},
    )
    assert len(got["grid"]) == 2
    for row in got["grid"]:
        # Every cell carries its own coordinates, so a row cannot be read as a
        # column and a transposed reading is impossible rather than unlikely.
        assert set(row["volatilityByStrikeOverForward"]) == {"0.9", "1", "1.1"}
        assert row["maturity"] in (0.5, 1.0)


def test_the_grid_agrees_with_the_slice_parameters(registry: ToolRegistry) -> None:
    got = ok(
        registry,
        "fit_volatility_surface",
        {"slices": [_slice_input(1.0)], "strikeRatios": [0.9, 1.0, 1.1]},
    )
    fitted = SVI(**got["slices"][0]["svi"])
    cells = got["grid"][0]["volatilityByStrikeOverForward"]
    for label, value in cells.items():
        assert value == pytest.approx(fitted.volatility(math.log(float(label)), 1.0), rel=1e-9)


def test_a_well_behaved_surface_passes_both_checks(registry: ToolRegistry) -> None:
    got = ok(
        registry,
        "fit_volatility_surface",
        {"slices": [_slice_input(0.25), _slice_input(0.5), _slice_input(1.0)]},
    )
    assert got["arbitrage"]["butterflyFree"] is True
    assert got["arbitrage"]["calendarFree"] is True


def test_a_surface_whose_later_slice_is_cheaper_is_flagged_for_calendar_arbitrage(
    registry: ToolRegistry,
) -> None:
    # Total variance must not fall as maturity rises. This surface is built so
    # it does, which is arbitrage: the later option is worth less than the
    # earlier one at the same moneyness.
    # The near slice is quoted far richer than the far one, so total variance
    # falls with maturity.
    near = _slice_input(0.5, SVI(a=0.35, b=0.02, rho=-0.1, m=0.0, s=0.15))
    far = _slice_input(1.0, SVI(a=0.01, b=0.02, rho=-0.1, m=0.0, s=0.15))
    got = ok(registry, "fit_volatility_surface", {"slices": [near, far]})
    assert got["arbitrage"]["calendarFree"] is False
    worst = got["arbitrage"]["worstCalendar"]
    # Positive: it is the size of the decrease, which is the violation.
    assert worst["worstDecreaseInTotalVariance"] > 0.0
    assert worst["isViolation"] is True
    assert (worst["earlier"], worst["later"]) == (0.5, 1.0)


def test_a_single_slice_reports_the_calendar_check_as_inapplicable(
    registry: ToolRegistry,
) -> None:
    # One slice has no pair to cross. Reported as absent rather than as "free",
    # which would claim a check that never ran.
    got = ok(registry, "fit_volatility_surface", {"slices": [_slice_input(1.0)]})
    assert got["arbitrage"]["calendarFree"] is None
    assert "not applicable" in got["arbitrage"]["calendarNote"]


def test_two_slices_at_the_same_maturity_are_refused(registry: ToolRegistry) -> None:
    error = refused(
        registry, "fit_volatility_surface", {"slices": [_slice_input(1.0), _slice_input(1.0)]}
    )
    assert "distinct maturity" in error["message"]


def test_maturities_come_back_sorted(registry: ToolRegistry) -> None:
    got = ok(
        registry,
        "fit_volatility_surface",
        {"slices": [_slice_input(1.0), _slice_input(0.25), _slice_input(0.5)]},
    )
    assert got["maturities"] == sorted(got["maturities"])


# -- local volatility -----------------------------------------------------


def test_local_volatility_is_returned_where_the_identity_holds(
    registry: ToolRegistry,
) -> None:
    got = ok(
        registry,
        "local_volatility",
        {
            "slices": [_slice_input(0.25), _slice_input(0.5), _slice_input(1.0)],
            "strikeRatios": [0.9, 1.0, 1.1],
            "maturities": [0.4, 0.75],
        },
    )
    assert len(got["points"]) == 6
    for point in got["points"]:
        assert point["admissible"] is True
        assert point["localVol"] > 0
        assert point["localVariance"] == pytest.approx(point["localVol"] ** 2, rel=1e-9)


def test_an_inadmissible_point_says_why_instead_of_returning_a_number(
    registry: ToolRegistry,
) -> None:
    # Where Dupire has no answer the library returns nan. nan is not portable
    # JSON and reads as a number to anything parsing loosely, so the fields are
    # omitted and the two failing quantities are named instead.
    near = _slice_input(0.5, SVI(a=0.35, b=0.02, rho=-0.1, m=0.0, s=0.15))
    far = _slice_input(1.0, SVI(a=0.01, b=0.02, rho=-0.1, m=0.0, s=0.15))
    got = ok(
        registry,
        "local_volatility",
        {"slices": [near, far], "strikeRatios": [1.0], "maturities": [0.75]},
    )
    point = got["points"][0]
    assert point["admissible"] is False
    assert "localVol" not in point
    assert "inadmissible" in point["reason"]


def test_no_local_volatility_point_ever_carries_a_nan(registry: ToolRegistry) -> None:
    near = _slice_input(0.5, SVI(a=0.35, b=0.02, rho=-0.1, m=0.0, s=0.15))
    far = _slice_input(1.0, SVI(a=0.01, b=0.02, rho=-0.1, m=0.0, s=0.15))
    got = ok(
        registry,
        "local_volatility",
        {"slices": [near, far], "strikeRatios": [0.8, 1.0, 1.25], "maturities": [0.6, 0.75, 0.9]},
    )
    for point in got["points"]:
        for key in ("localVol", "localVariance"):
            if key in point:
                assert not math.isnan(point[key])


def test_too_many_grid_points_are_refused(registry: ToolRegistry) -> None:
    error = refused(
        registry,
        "local_volatility",
        {
            "slices": [_slice_input(1.0)],
            "strikeRatios": [1.0 + i / 100 for i in range(50)],
            "maturities": [1.0],
        },
    )
    assert error["details"][0]["keyword"] == "maxItems"
