"""The pricing tools: the numbers they return, and the inputs they refuse."""

from __future__ import annotations

import math
from typing import Any

import moneyness as mny
import pytest

from abacus.analytics import (
    MAX_PLAUSIBLE_RATE,
    MAX_PLAUSIBLE_VOL,
    MAX_PLAUSIBLE_YEARS,
    default_registry,
)
from abacus.tools import ToolRegistry

BASE: dict[str, Any] = {
    "spot": 100.0,
    "strike": 100.0,
    "time": 1.0,
    "rate": 0.05,
    "vol": 0.2,
    "type": "call",
}


@pytest.fixture
def registry() -> ToolRegistry:
    return default_registry()


def payload(registry: ToolRegistry, name: str, **overrides: Any) -> Any:
    result = registry.call(name, {**BASE, **overrides})
    assert result["isError"] is False, result["content"][0]["text"]
    return result["structuredContent"]


def refusal(registry: ToolRegistry, name: str, **overrides: Any) -> dict[str, Any]:
    result = registry.call(name, {**BASE, **overrides})
    assert result["isError"] is True, "expected the call to be refused"
    error = result["structuredContent"]["error"]
    assert isinstance(error, dict)
    return error


# -- agreement with the library ------------------------------------------


@pytest.mark.parametrize("side", ["call", "put"])
def test_price_matches_the_library_exactly(registry: ToolRegistry, side: str) -> None:
    # The tool layer must not perturb the number: same inputs, same float.
    expected = mny.price(
        mny.Inputs(spot=100.0, strike=100.0, time=1.0, rate=0.05, vol=0.2),
        mny.OptionType(side),
    )
    assert payload(registry, "price_european_option", type=side)["price"] == expected


def test_greeks_match_the_library_exactly(registry: ToolRegistry) -> None:
    inputs = mny.Inputs(spot=100.0, strike=100.0, time=1.0, rate=0.05, vol=0.2)
    got = payload(registry, "european_option_greeks")
    assert got["delta"] == mny.delta(inputs, mny.OptionType.CALL)
    assert got["gamma"] == mny.gamma(inputs)
    assert got["vega"] == mny.vega(inputs)
    assert got["theta"] == mny.theta(inputs, mny.OptionType.CALL)
    assert got["rho"] == mny.rho(inputs, mny.OptionType.CALL)


def test_the_combined_tool_agrees_with_the_separate_ones(registry: ToolRegistry) -> None:
    both = payload(registry, "european_option_analytics")
    assert both["price"] == payload(registry, "price_european_option")["price"]
    assert both["greeks"] == payload(registry, "european_option_greeks")


# -- independent checks on the numbers ------------------------------------


def test_delta_agrees_with_a_finite_difference_of_the_price(registry: ToolRegistry) -> None:
    # An independent read on the whole path: the analytic delta the Greeks tool
    # reports against a central difference of the prices the pricing tool
    # reports. They come from different formulae, so agreement is evidence the
    # wiring is right and not merely self-consistent.
    step = 1e-4
    up = payload(registry, "price_european_option", spot=100.0 + step)["price"]
    down = payload(registry, "price_european_option", spot=100.0 - step)["price"]
    numerical = (up - down) / (2 * step)
    assert payload(registry, "european_option_greeks")["delta"] == pytest.approx(
        numerical, abs=1e-6
    )


def test_vega_agrees_with_a_finite_difference_of_the_price(registry: ToolRegistry) -> None:
    step = 1e-5
    up = payload(registry, "price_european_option", vol=0.2 + step)["price"]
    down = payload(registry, "price_european_option", vol=0.2 - step)["price"]
    assert payload(registry, "european_option_greeks")["vega"] == pytest.approx(
        (up - down) / (2 * step), abs=1e-5
    )


def test_gamma_agrees_with_a_second_difference_of_the_price(registry: ToolRegistry) -> None:
    step = 1e-2
    up = payload(registry, "price_european_option", spot=100.0 + step)["price"]
    mid = payload(registry, "price_european_option")["price"]
    down = payload(registry, "price_european_option", spot=100.0 - step)["price"]
    assert payload(registry, "european_option_greeks")["gamma"] == pytest.approx(
        (up - 2 * mid + down) / step**2, abs=1e-6
    )


def test_a_deep_in_the_money_call_is_worth_the_discounted_forward_less_strike(
    registry: ToolRegistry,
) -> None:
    got = payload(registry, "price_european_option", spot=1000.0, vol=0.01)
    assert got["price"] == pytest.approx(1000.0 - 100.0 * math.exp(-0.05), rel=1e-9)
    assert got["timeValue"] == pytest.approx(0.0, abs=1e-6)


def test_a_far_out_of_the_money_call_is_worth_almost_nothing(registry: ToolRegistry) -> None:
    assert payload(registry, "price_european_option", strike=10_000.0)["price"] < 1e-9


def test_price_is_never_below_intrinsic(registry: ToolRegistry) -> None:
    for spot in (50.0, 80.0, 100.0, 130.0, 200.0):
        for side in ("call", "put"):
            got = payload(registry, "price_european_option", spot=spot, type=side)
            assert got["price"] >= got["intrinsic"] - 1e-12
            assert got["timeValue"] >= -1e-12


def test_a_call_is_worth_more_as_volatility_rises(registry: ToolRegistry) -> None:
    prices = [
        payload(registry, "price_european_option", vol=vol)["price"]
        for vol in (0.1, 0.2, 0.3, 0.4)
    ]
    assert prices == sorted(prices)


def test_gamma_and_vega_are_positive_for_both_sides(registry: ToolRegistry) -> None:
    for side in ("call", "put"):
        got = payload(registry, "european_option_greeks", type=side)
        assert got["gamma"] > 0
        assert got["vega"] > 0


def test_call_and_put_delta_differ_by_the_carry_discount(registry: ToolRegistry) -> None:
    call_delta = payload(registry, "european_option_greeks", type="call")["delta"]
    put_delta = payload(registry, "european_option_greeks", type="put")["delta"]
    # With carry defaulting to the rate, the two deltas differ by exactly one.
    assert call_delta - put_delta == pytest.approx(1.0, abs=1e-12)


# -- parity and bounds ----------------------------------------------------


def test_put_call_parity_holds(registry: ToolRegistry) -> None:
    got = registry.call("put_call_parity", {k: v for k, v in BASE.items() if k != "type"})
    body = got["structuredContent"]
    assert body["consistent"] is True
    assert body["parityGap"] == pytest.approx(0.0, abs=1e-9)


def test_bounds_contain_a_price_that_came_from_the_model(registry: ToolRegistry) -> None:
    fair = payload(registry, "price_european_option")["price"]
    args = {k: v for k, v in BASE.items() if k != "vol"}
    got = registry.call("option_price_bounds", {**args, "price": fair})["structuredContent"]
    assert got["attainable"] is True
    assert got["lower"] <= fair <= got["upper"]


def test_a_price_above_the_upper_bound_is_reported_as_unattainable(
    registry: ToolRegistry,
) -> None:
    # No non-negative volatility produces it, so asking for an implied
    # volatility would be futile — which is the point of the tool.
    args = {k: v for k, v in BASE.items() if k != "vol"}
    got = registry.call("option_price_bounds", {**args, "price": 500.0})["structuredContent"]
    assert got["attainable"] is False


# -- domain refusals ------------------------------------------------------


def test_a_volatility_given_as_a_percentage_is_refused_with_the_conversion(
    registry: ToolRegistry,
) -> None:
    # The mistake this whole layer exists for: 20 rather than 0.2 prices without
    # complaint and returns a number that is arithmetically correct and useless.
    error = refusal(registry, "price_european_option", vol=20.0)
    assert error["kind"] == "domain"
    assert "0.2" in error["message"]
    assert error["details"][0]["path"] == "/vol"


def test_a_rate_given_as_a_percentage_is_refused(registry: ToolRegistry) -> None:
    error = refusal(registry, "price_european_option", rate=5.0)
    assert "0.05" in error["message"]
    assert error["details"][0]["path"] == "/rate"


def test_a_carry_given_as_a_percentage_is_refused(registry: ToolRegistry) -> None:
    assert refusal(registry, "price_european_option", carry=4.0)["details"][0]["path"] == "/carry"


def test_an_expiry_given_in_days_is_refused(registry: ToolRegistry) -> None:
    error = refusal(registry, "price_european_option", time=90.0)
    assert "year fraction" in error["message"]
    assert error["details"][0]["path"] == "/time"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vol", MAX_PLAUSIBLE_VOL),
        ("rate", MAX_PLAUSIBLE_RATE),
        ("rate", -MAX_PLAUSIBLE_RATE),
        ("time", MAX_PLAUSIBLE_YEARS),
    ],
)
def test_the_limits_themselves_are_accepted(
    registry: ToolRegistry, field: str, value: float
) -> None:
    # The bounds are inclusive, so a value sitting exactly on one is priced
    # rather than refused.
    result = registry.call("price_european_option", {**BASE, field: value})
    assert result["isError"] is False


def test_a_negative_volatility_is_caught_by_the_schema(registry: ToolRegistry) -> None:
    error = refusal(registry, "price_european_option", vol=-0.2)
    assert error["kind"] == "invalid_input"
    assert error["details"][0]["keyword"] == "exclusiveMinimum"


@pytest.mark.parametrize("field", ["spot", "strike"])
def test_a_zero_price_or_strike_is_refused(registry: ToolRegistry, field: str) -> None:
    assert refusal(registry, "price_european_option", **{field: 0})["kind"] == "invalid_input"


def test_a_negative_time_is_refused(registry: ToolRegistry) -> None:
    # An option cannot expire in the past.
    assert refusal(registry, "price_european_option", time=-1.0)["kind"] == "invalid_input"


def test_an_unknown_option_type_lists_the_two_that_exist(registry: ToolRegistry) -> None:
    error = refusal(registry, "price_european_option", type="straddle")
    assert "'call', 'put'" in error["details"][0]["message"]


# -- expiry and carry -----------------------------------------------------


def test_an_option_at_expiry_is_worth_its_intrinsic_value(registry: ToolRegistry) -> None:
    got = payload(registry, "price_european_option", time=0.0, spot=120.0)
    assert got["price"] == pytest.approx(20.0, abs=1e-12)
    assert got["timeValue"] == pytest.approx(0.0, abs=1e-12)


def test_carry_defaults_to_the_rate(registry: ToolRegistry) -> None:
    # The non-dividend-paying share case, so passing the rate explicitly must
    # change nothing.
    assert payload(registry, "price_european_option")["price"] == payload(
        registry, "price_european_option", carry=0.05
    )["price"]


def test_a_dividend_yield_lowers_a_call(registry: ToolRegistry) -> None:
    with_dividend = payload(registry, "price_european_option", carry=0.05 - 0.03)["price"]
    assert with_dividend < payload(registry, "price_european_option")["price"]


def test_zero_carry_prices_an_option_on_a_future(registry: ToolRegistry) -> None:
    got = payload(registry, "price_european_option", carry=0.0)
    assert got["forward"] == pytest.approx(100.0, rel=1e-12)


# -- degenerate inputs ----------------------------------------------------
#
# At expiry, or with zero volatility, spot or strike, some quantities have a
# limit and others do not exist at all. Each of these went through the library
# as an unhandled ValueError before it was covered, which reached the caller as
# an internal error rather than as anything it could act on.


def test_a_price_at_expiry_omits_the_undefined_standardised_arguments(
    registry: ToolRegistry,
) -> None:
    # The price is the limit and is returned; d1 and d2 do not exist, so they
    # are left out rather than filled with an infinity that would read as real.
    got = payload(registry, "price_european_option", time=0.0, spot=120.0)
    assert got["price"] == pytest.approx(20.0, abs=1e-12)
    assert "d1" not in got
    assert "d2" not in got


def test_a_price_away_from_expiry_reports_them(registry: ToolRegistry) -> None:
    got = payload(registry, "price_european_option")
    assert got["d1"] == pytest.approx(0.35, abs=1e-12)
    assert got["d2"] == pytest.approx(0.15, abs=1e-12)


def test_greeks_at_expiry_are_refused_with_the_reason(registry: ToolRegistry) -> None:
    # Unlike the price, these genuinely do not exist: the payoff is kinked.
    error = refusal(registry, "european_option_greeks", time=0.0)
    assert error["kind"] == "domain"
    assert "kinked" in error["message"]
    # The refusal names what can still be had, so the caller has somewhere to go.
    assert "price_european_option" in error["message"]


def test_the_combined_tool_refuses_at_expiry_too(registry: ToolRegistry) -> None:
    assert refusal(registry, "european_option_analytics", time=0.0)["kind"] == "domain"


def test_bounds_at_expiry_are_refused_with_the_library_explanation(
    registry: ToolRegistry,
) -> None:
    # An expired option has no implied volatility, because every volatility
    # reproduces its price. The library says so; the guard passes that through
    # instead of letting it become an internal error.
    args = {k: v for k, v in BASE.items() if k != "vol"}
    result = registry.call("option_price_bounds", {**args, "time": 0.0, "price": 20.0})
    assert result["isError"] is True
    assert "implied volatility" in result["structuredContent"]["error"]["message"]


def test_no_tool_raises_on_a_degenerate_market(registry: ToolRegistry) -> None:
    # The guard's real job: whatever the library refuses to compute, the caller
    # gets a result it can read rather than an exception crossing the boundary.
    degenerate = [{"time": 0.0}, {"vol": 1e-300}, {"spot": 1e-300}, {"strike": 1e-300}]
    for tool in registry:
        for overrides in degenerate:
            args = {**BASE, **overrides}
            if tool.name == "put_call_parity":
                args.pop("type")
            if tool.name == "option_price_bounds":
                args.pop("vol", None)
                args["price"] = 20.0
            result = registry.call(tool.name, args)
            assert isinstance(result["isError"], bool)
