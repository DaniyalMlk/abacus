"""The curve, bond and spread tools.

Fixed income is where identities are cheap and plentiful, so almost nothing here
is checked against a recorded number.

*A curve must reprice its own instruments.* That is not a property of the code, it
is the definition of a bootstrap, and it is the one check that says the curve means
anything at all.

*Key rate durations partition the effective duration.* The bump weights are tents
that sum to one at every maturity, so the durations sum to the parallel-shift
duration. A table that does not add up is measuring something other than the curve
it claims to.

*The lattice and the curve must agree where they overlap.* A tree calibrated to a
curve reprices it, and the spread it implies with no option modelled is the curve's
own Z-spread up to the tree's discretisation. Two independent routes to one number.

*An option's cost has a sign, and it is not arbitrary.* An issuer's call is worth
something to the issuer, so modelling it leaves a narrower spread explaining the
same price; a holder's put reverses that. And the cost falls as the bond moves away
from the strike, which no self-consistency check would ever notice.
"""

from __future__ import annotations

import math
import random
from itertools import pairwise
from typing import Any

import pytest

from abacus import book, risk
from abacus.curves import (
    DEFAULT_ROLLING,
    MAX_BUCKETS,
    MAX_INSTRUMENTS,
    CurveTools,
    _default_rolling,
)
from abacus.handles import Minter
from abacus.tools import ToolRegistry

REFERENCE = "2021-01-05"

QUOTES: dict[str, Any] = {
    "reference": REFERENCE,
    "basis": "ACT_365F",
    "deposits": [
        {"maturity": "2021-07-05", "rate": 0.0020, "basis": "ACT_360", "label": "6m depo"}
    ],
    "swaps": [
        {
            "maturity": "2023-01-05",
            "rate": 0.0060,
            "frequency": "SEMI_ANNUAL",
            "basis": "THIRTY_360_BOND",
            "label": "2y",
        },
        {
            "maturity": "2026-01-05",
            "rate": 0.0150,
            "frequency": "SEMI_ANNUAL",
            "basis": "THIRTY_360_BOND",
            "label": "5y",
        },
        {
            "maturity": "2031-01-05",
            "rate": 0.0220,
            "frequency": "SEMI_ANNUAL",
            "basis": "THIRTY_360_BOND",
            "label": "10y",
        },
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
def registry() -> ToolRegistry:
    """Curve tools, plus the other two handle-minting groups under the same key.

    Sharing the minter is what makes the wrong-kind-of-handle tests real: a book or
    a moments handle presented here passes its integrity check, so the refusal has
    to come from reading the payload's kind rather than from a failed MAC.
    """
    minter = Minter()
    built = CurveTools(minter).register(ToolRegistry())
    built = book.register(built, minter=minter)
    return risk.register(built, minter=minter)


def call(registry: ToolRegistry, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = registry.call(name, arguments)
    assert result["isError"] is False, result["structuredContent"]
    payload = result["structuredContent"]
    assert isinstance(payload, dict)
    return payload


def refuse(registry: ToolRegistry, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = registry.call(name, arguments)
    assert result["isError"] is True, result["structuredContent"]
    content = result["structuredContent"]
    assert isinstance(content, dict)
    error = content["error"]
    assert isinstance(error, dict)
    return error


def curve_handle(registry: ToolRegistry) -> str:
    handle = call(registry, "bootstrap_discount_curve", QUOTES)["handle"]
    assert isinstance(handle, str)
    return handle


# -- bootstrapping -----------------------------------------------------------


def test_the_curve_reprices_the_instruments_it_was_built_from(
    registry: ToolRegistry,
) -> None:
    """The definition of a bootstrap, and the only check that says it worked.

    A curve that fails this is not slightly wrong; the pillar values look entirely
    ordinary and every number read off it is meaningless.
    """
    payload = call(registry, "bootstrap_discount_curve", QUOTES)
    assert payload["repricesItsInstruments"] is True
    assert payload["sweeps"] >= 1

    solutions = payload["solutions"]
    assert isinstance(solutions, list)
    assert len(solutions) == 4
    for solution in solutions:
        assert solution["converged"] is True
        assert abs(float(solution["residual"])) < 1e-12


def test_one_pillar_per_instrument_plus_the_reference(registry: ToolRegistry) -> None:
    payload = call(registry, "bootstrap_discount_curve", QUOTES)
    pillars = payload["pillars"]
    assert isinstance(pillars, list)
    assert len(pillars) == 5
    assert pillars[0]["date"] == REFERENCE
    assert pillars[0]["discount"] == pytest.approx(1.0, abs=1e-15)
    # Note the last one: the 2031-01-05 swap matures on a Sunday, and the default
    # rolling rule moves its final payment to the Monday, so the pillar is at
    # 2031-01-06. That is the convention doing visible work rather than an error —
    # a bootstrap places its pillar where the cashflow actually lands.
    assert [pillar["date"] for pillar in pillars[1:]] == [
        "2021-07-05",
        "2023-01-05",
        "2026-01-05",
        "2031-01-06",
    ]
    assert payload["horizon"] == "2031-01-06"


def test_the_zero_rate_at_the_reference_is_null_not_zero(registry: ToolRegistry) -> None:
    """Zero would be a real rate at the front of the curve that nothing supports.

    The discount factor at the reference date is one whatever the rate is, so no
    rate is implied by it, and a zero there would be read as a rate rather than as
    an absence of one.
    """
    payload = call(registry, "bootstrap_discount_curve", QUOTES)
    pillars = payload["pillars"]
    assert isinstance(pillars, list)
    assert pillars[0]["zeroRate"] is None
    assert all(pillar["zeroRate"] is not None for pillar in pillars[1:])


def test_discount_factors_fall_along_a_positive_curve(registry: ToolRegistry) -> None:
    payload = call(registry, "bootstrap_discount_curve", QUOTES)
    pillars = payload["pillars"]
    assert isinstance(pillars, list)
    factors = [float(pillar["discount"]) for pillar in pillars]
    assert factors == sorted(factors, reverse=True)
    assert all(0.0 < factor <= 1.0 for factor in factors)


def test_the_curve_handle_carries_the_curve_and_is_small(registry: ToolRegistry) -> None:
    """The whole justification for this handle being different from the returns one.

    A curve is two numbers per instrument, so the handle can hold it outright.
    Asserting the size keeps that claim honest as the payload grows.
    """
    handle = curve_handle(registry)
    assert len(handle) < 1024


# -- reading a curve ---------------------------------------------------------


def test_reading_a_curve_back_reproduces_its_pillars(registry: ToolRegistry) -> None:
    """Queried at the pillar dates, the curve returns the pillar values exactly.

    Which is the round trip that matters: the handle is rebuilt into a curve by
    interpolation, and the interpolant must pass through the points it was given.
    """
    built = call(registry, "bootstrap_discount_curve", QUOTES)
    pillars = built["pillars"]
    assert isinstance(pillars, list)

    read = call(
        registry,
        "discount_curve_rates",
        {"handle": built["handle"], "dates": [pillar["date"] for pillar in pillars]},
    )
    points = read["points"]
    assert isinstance(points, list)
    for pillar, point in zip(pillars, points, strict=True):
        assert point["discount"] == pytest.approx(float(pillar["discount"]), rel=1e-14)
        if pillar["zeroRate"] is not None:
            assert point["zeroRate"] == pytest.approx(float(pillar["zeroRate"]), rel=1e-12)


def test_a_forward_rate_compounds_the_two_discount_factors(registry: ToolRegistry) -> None:
    """``(D1 / D2 - 1) / t`` on a simple basis, recomputed from the output.

    The forward is the one quantity here that relates two points rather than
    describing one, so it is the one that can be checked against the others.
    """
    read = call(
        registry,
        "discount_curve_rates",
        {
            "handle": curve_handle(registry),
            "dates": ["2026-01-05", "2031-01-05"],
            "forwardCompounding": "SIMPLE",
        },
    )
    points = read["points"]
    assert isinstance(points, list)
    near, far = points
    span = float(far["years"]) - float(near["years"])
    expected = (float(near["discount"]) / float(far["discount"]) - 1.0) / span
    assert far["forwardFromPrevious"] == pytest.approx(expected, rel=1e-10)

    # The first point's forward is measured from the curve's reference date, which
    # makes it a spot-starting rate rather than nothing. Checking it the same way
    # confirms that, rather than leaving the field's meaning to the note.
    spot = (1.0 / float(near["discount"]) - 1.0) / float(near["years"])
    assert near["forwardFromPrevious"] == pytest.approx(spot, rel=1e-10)


def test_a_date_past_the_horizon_is_refused_not_extrapolated(
    registry: ToolRegistry,
) -> None:
    error = refuse(
        registry,
        "discount_curve_rates",
        {"handle": curve_handle(registry), "dates": ["2041-01-05"]},
    )
    assert error["kind"] == "domain"
    message = str(error["message"])
    assert "horizon" in message
    assert "2031-01-06" in message


# -- bond analytics ----------------------------------------------------------


def test_a_bond_at_par_prices_to_its_coupon(registry: ToolRegistry) -> None:
    """The identity every bond calculator is checked against first.

    A bond yielding its own coupon rate trades at par, exactly, for any maturity and
    any frequency. It catches a discount factor applied to the wrong period and a
    coupon divided by the wrong frequency.
    """
    payload = call(
        registry,
        "bond_analytics",
        {"bond": BOND, "yieldToMaturity": 0.05, "reference": REFERENCE},
    )
    from_yield = payload["fromYield"]
    assert isinstance(from_yield, dict)
    assert from_yield["cleanPrice"] == pytest.approx(100.0, abs=1e-10)


def test_price_and_yield_invert_each_other(registry: ToolRegistry) -> None:
    priced = call(
        registry,
        "bond_analytics",
        {"bond": BOND, "yieldToMaturity": 0.0327, "reference": REFERENCE},
    )
    from_yield = priced["fromYield"]
    assert isinstance(from_yield, dict)

    solved = call(
        registry,
        "bond_analytics",
        {"bond": BOND, "price": from_yield["cleanPrice"], "reference": REFERENCE},
    )
    back = solved["fromYield"]
    assert isinstance(back, dict)
    assert back["yieldToMaturity"] == pytest.approx(0.0327, rel=1e-10)
    solve = solved["yieldSolve"]
    assert isinstance(solve, dict)
    assert solve["converged"] is True


def test_modified_duration_is_macaulay_over_one_plus_the_periodic_yield(
    registry: ToolRegistry,
) -> None:
    """The exact relation between the two, which is why both are reported.

    Calling one "duration" and leaving the other out is how a price sensitivity gets
    used as a time to maturity, and the two differ by a few percent at any realistic
    yield.
    """
    rate = 0.0327
    payload = call(
        registry,
        "bond_analytics",
        {"bond": BOND, "yieldToMaturity": rate, "reference": REFERENCE},
    )
    from_yield = payload["fromYield"]
    assert isinstance(from_yield, dict)
    macaulay = float(from_yield["macaulayDuration"])
    modified = float(from_yield["modifiedDuration"])
    assert modified == pytest.approx(macaulay / (1.0 + rate / 2.0), rel=1e-10)


def test_pv01_is_a_basis_point_of_duration(registry: ToolRegistry) -> None:
    """``pv01 ~= price * modified duration * 0.0001``, to the accuracy of a secant.

    Not exact, because pv01 is measured by shifting the yield and duration is the
    derivative, and the gap is the convexity. Asserting it to four figures is what
    says both were computed from the same bond.
    """
    rate = 0.0327
    payload = call(
        registry,
        "bond_analytics",
        {"bond": BOND, "yieldToMaturity": rate, "reference": REFERENCE},
    )
    from_yield = payload["fromYield"]
    assert isinstance(from_yield, dict)
    approximate = (
        float(from_yield["dirtyPrice"]) * float(from_yield["modifiedDuration"]) * 0.0001
    )
    assert float(from_yield["pv01"]) == pytest.approx(approximate, rel=1e-3)


def test_dirty_price_is_clean_plus_accrued(registry: ToolRegistry) -> None:
    payload = call(
        registry,
        "bond_analytics",
        {
            "bond": BOND,
            "yieldToMaturity": 0.03,
            "reference": REFERENCE,
            "settlement": "2021-04-05",
        },
    )
    from_yield = payload["fromYield"]
    assert isinstance(from_yield, dict)
    assert float(payload["accrued"]) > 0.0
    assert float(from_yield["dirtyPrice"]) == pytest.approx(
        float(from_yield["cleanPrice"]) + float(payload["accrued"]), rel=1e-12
    )


def test_the_curve_price_discounts_every_cashflow_at_its_own_rate(
    registry: ToolRegistry,
) -> None:
    """Recomputed from the curve read back through a different tool.

    The cashflows come from ``bond_analytics`` and the discount factors from
    ``discount_curve_rates``, so the sum is assembled from two tools' outputs and
    compared against a third number. Nothing shared but the curve.
    """
    handle = curve_handle(registry)
    payload = call(registry, "bond_analytics", {"handle": handle, "bond": BOND})
    flows = payload["cashflows"]
    assert isinstance(flows, list)

    read = call(
        registry,
        "discount_curve_rates",
        {"handle": handle, "dates": [flow["date"] for flow in flows]},
    )
    points = read["points"]
    assert isinstance(points, list)
    present = math.fsum(
        float(flow["amount"]) * float(point["discount"])
        for flow, point in zip(flows, points, strict=True)
    )

    from_curve = payload["fromCurve"]
    assert isinstance(from_curve, dict)
    assert float(from_curve["price"]) == pytest.approx(present, rel=1e-12)


def test_the_yield_and_the_curve_are_reported_as_different_answers(
    registry: ToolRegistry,
) -> None:
    """Both, with the gap named, rather than one quietly chosen.

    The gap is the shape of the curve and is zero only for a flat one. This curve
    rises steeply, so the two durations must actually differ — a test that passed on
    a flat curve would prove nothing.
    """
    payload = call(
        registry,
        "bond_analytics",
        {"handle": curve_handle(registry), "bond": BOND, "price": 120.0},
    )
    from_yield = payload["fromYield"]
    from_curve = payload["fromCurve"]
    gap = payload["yieldAgainstCurve"]
    assert isinstance(from_yield, dict)
    assert isinstance(from_curve, dict)
    assert isinstance(gap, dict)

    assert gap["priceDifference"] == pytest.approx(
        float(from_curve["price"]) - float(from_yield["cleanPrice"]), rel=1e-12
    )
    assert gap["durationDifference"] == pytest.approx(
        float(from_curve["effectiveDuration"]) - float(from_yield["modifiedDuration"]),
        rel=1e-12,
    )
    assert abs(float(gap["durationDifference"])) > 0.01


def test_dv01_is_the_effective_duration_in_price_terms(registry: ToolRegistry) -> None:
    payload = call(registry, "bond_analytics", {"handle": curve_handle(registry), "bond": BOND})
    from_curve = payload["fromCurve"]
    assert isinstance(from_curve, dict)
    expected = float(from_curve["price"]) * float(from_curve["effectiveDuration"]) * 0.0001
    assert float(from_curve["dv01"]) == pytest.approx(expected, rel=1e-3)


def test_the_rolling_rule_is_the_librarys_default_and_is_reported(
    registry: ToolRegistry,
) -> None:
    """Omitted means the library's convention, not a value invented here.

    An earlier draft substituted ``NONE``, which moved every unadjusted schedule off
    the market convention and changed prices by a coupon's worth of accrual with
    nothing in the output to show it. The default is read out of the library's own
    dataclass field, so the two cannot disagree, and the rule used is on the result.
    """
    from tenor import Bond, Swap

    assert DEFAULT_ROLLING.name == "MODIFIED_FOLLOWING"
    assert _default_rolling(Swap) is DEFAULT_ROLLING
    assert _default_rolling(Bond) is DEFAULT_ROLLING

    handle = curve_handle(registry)
    payload = call(
        registry, "bond_analytics", {"handle": handle, "bond": BOND, "yieldToMaturity": 0.03}
    )
    assert payload["rolling"] == DEFAULT_ROLLING.name

    named = call(
        registry,
        "bond_analytics",
        {"handle": handle, "bond": {**BOND, "rolling": "NONE"}, "yieldToMaturity": 0.03},
    )
    assert named["rolling"] == "NONE"

    # It is not cosmetic, and *where* it shows up is the interesting part. The
    # schedule moves: three of this bond's payment dates differ.
    rolled = [flow["date"] for flow in payload["cashflows"]]
    unrolled = [flow["date"] for flow in named["cashflows"]]
    assert rolled != unrolled
    assert rolled[-1] == "2031-01-06"
    assert unrolled[-1] == "2031-01-05"

    # The *curve* price differs, because a curve discounts by actual date.
    assert float(named["fromCurve"]["price"]) != float(payload["fromCurve"]["price"])

    # The *yield* price does not, because a yield discounts by period count and this
    # bond is on a 30/360 basis, where a day's shift leaves every period length
    # unchanged. Two prices for one bond, one of which moves and one of which does
    # not, which is exactly why the rule is reported rather than assumed.
    assert float(named["fromYield"]["cleanPrice"]) == pytest.approx(
        float(payload["fromYield"]["cleanPrice"]), rel=1e-12
    )


# -- curve risk --------------------------------------------------------------


def test_key_rate_durations_sum_to_the_effective_duration(registry: ToolRegistry) -> None:
    """The identity that makes a key rate table mean anything.

    The bump weights are tents that sum to one at every maturity, so the durations
    partition the parallel-shift duration. The residual is a finite-difference
    artefact, not a modelling choice, so the bound is tight.
    """
    payload = call(registry, "bond_curve_risk", {"handle": curve_handle(registry), "bond": BOND})
    rates = payload["keyRates"]
    assert isinstance(rates, list)
    assert math.fsum(float(rate["duration"]) for rate in rates) == pytest.approx(
        float(payload["keyRateDurationSum"]), rel=1e-12
    )
    assert float(payload["keyRateDurationSum"]) == pytest.approx(
        float(payload["effectiveDuration"]), rel=1e-6
    )
    assert abs(float(payload["sumAgainstEffectiveDuration"])) < 1e-6


def test_the_level_shape_duration_is_the_effective_duration(registry: ToolRegistry) -> None:
    """A level shift *is* a parallel shift, so these are the same measurement.

    Which is worth pinning: the three shape durations share one machinery, and if
    the level weighting were wrong the other two would be wrong in a way nothing
    else here would reveal.
    """
    payload = call(registry, "bond_curve_risk", {"handle": curve_handle(registry), "bond": BOND})
    shapes = payload["shapeDurations"]
    assert isinstance(shapes, dict)
    assert float(shapes["level"]) == pytest.approx(float(payload["effectiveDuration"]), rel=1e-12)
    assert 0.0 < float(shapes["slope"]) < float(shapes["level"])


def test_the_risk_sits_where_the_bond_matures(registry: ToolRegistry) -> None:
    """Almost all of a ten-year bond's duration belongs to the ten-year bucket.

    An obvious property, and the one that fails loudly if the buckets and the
    durations are paired up in the wrong order.
    """
    payload = call(registry, "bond_curve_risk", {"handle": curve_handle(registry), "bond": BOND})
    rates = payload["keyRates"]
    assert isinstance(rates, list)
    largest = max(rates, key=lambda rate: float(rate["duration"]))
    assert float(largest["years"]) > 9.0
    assert float(largest["duration"]) > 0.8 * float(payload["effectiveDuration"])


def test_the_buckets_default_to_the_curves_own_pillars(registry: ToolRegistry) -> None:
    """Because a bucket without a pillar shifts nothing and reports zero risk.

    That zero reads as an absence of risk rather than an absence of curve, and the
    risk it should have carried is absorbed by its neighbours instead. A tidy
    0.5/1/2/5/10/30 ladder would do exactly that on this curve, which has no
    one-year pillar.
    """
    built = call(registry, "bootstrap_discount_curve", QUOTES)
    pillars = built["pillars"]
    assert isinstance(pillars, list)
    payload = call(registry, "bond_curve_risk", {"handle": built["handle"], "bond": BOND})
    rates = payload["keyRates"]
    assert isinstance(rates, list)
    assert [float(rate["years"]) for rate in rates] == [
        pytest.approx(float(pillar["years"])) for pillar in pillars[1:]
    ]
    assert [rate["bucket"] for rate in rates] == ["6m", "2y", "5y", "10.01y"]


def test_a_bucket_off_the_curve_is_refused_by_the_library(registry: ToolRegistry) -> None:
    error = refuse(
        registry,
        "bond_curve_risk",
        # 1y sits between the 0.4959 and 2.0 pillars with a bucket on each side, so
        # its tent covers no pillar at all: the shift would move nothing and the
        # duration would come back as zero.
        {"handle": curve_handle(registry), "bond": BOND, "buckets": [0.5, 1.0, 2.0]},
    )
    assert error["kind"] == "domain"
    assert "covers no pillar" in str(error["message"])
    assert "1y" in str(error["message"])


def test_buckets_out_of_order_are_refused(registry: ToolRegistry) -> None:
    error = refuse(
        registry,
        "bond_curve_risk",
        {"handle": curve_handle(registry), "bond": BOND, "buckets": [5.0, 2.0]},
    )
    assert "increasing order" in str(error["message"])


def test_instrument_risk_names_the_tradeable_hedge(registry: ToolRegistry) -> None:
    """Risk in the things that can be traded, which key rate durations are not.

    The ten-year swap must carry most of it for a ten-year bond, and the total is
    close to the bond's dv01 — the same risk expressed two ways, differing only
    because rebuilding the curve from bumped quotes is not the same operation as
    bumping the curve.
    """
    handle = curve_handle(registry)
    payload = call(registry, "bond_curve_risk", {"handle": handle, "bond": BOND})
    hedge = payload["instrumentRisk"]
    assert isinstance(hedge, dict)
    instruments = hedge["instruments"]
    assert isinstance(instruments, list)
    assert [item["instrument"]["label"] for item in instruments] == [
        "6m depo",
        "2y",
        "5y",
        "10y",
    ]
    largest = max(instruments, key=lambda item: float(item["value"]))
    assert largest["instrument"]["label"] == "10y"

    analytics = call(registry, "bond_analytics", {"handle": handle, "bond": BOND})
    from_curve = analytics["fromCurve"]
    assert isinstance(from_curve, dict)
    assert float(hedge["total"]) == pytest.approx(float(from_curve["dv01"]), rel=0.05)


# -- spreads -----------------------------------------------------------------


def test_a_bond_priced_off_the_curve_has_no_spread(registry: ToolRegistry) -> None:
    """Zero, to solver tolerance, and it is the sanity check for the whole tool.

    Price the bond with the curve, then ask what spread that price implies. Anything
    but zero means the two functions disagree about the bond.
    """
    handle = curve_handle(registry)
    analytics = call(registry, "bond_analytics", {"handle": handle, "bond": BOND})
    from_curve = analytics["fromCurve"]
    assert isinstance(from_curve, dict)

    payload = call(
        registry,
        "bond_spreads",
        {"handle": handle, "bond": BOND, "price": from_curve["price"]},
    )
    assert float(payload["zSpread"]) == pytest.approx(0.0, abs=1e-10)
    assert payload["zSpreadConverged"] is True


def test_a_cheaper_bond_implies_a_wider_spread(registry: ToolRegistry) -> None:
    handle = curve_handle(registry)
    spreads = [
        float(
            call(registry, "bond_spreads", {"handle": handle, "bond": BOND, "price": price})[
                "zSpread"
            ]
        )
        for price in (130.0, 120.0, 110.0, 100.0)
    ]
    assert spreads == sorted(spreads)
    assert all(later > earlier for earlier, later in pairwise(spreads))


def test_the_lattice_reprices_the_curve_it_was_calibrated_to(
    registry: ToolRegistry,
) -> None:
    """Otherwise it is not a model of that curve and every spread off it is wrong."""
    payload = call(
        registry,
        "bond_spreads",
        {
            "handle": curve_handle(registry),
            "bond": BOND,
            "price": 110.0,
            "volatility": 0.15,
            "firstCall": "2026-01-05",
        },
    )
    option = payload["embeddedOption"]
    assert isinstance(option, dict)
    assert option["latticeRepricesCurve"] is True
    assert option["steps"] == 20
    assert option["stepYears"] == pytest.approx(0.5)
    assert option["exerciseOpportunities"] == 10


def test_the_lattice_spread_agrees_with_the_curve_spread(registry: ToolRegistry) -> None:
    """Two independent routes to one number, meeting inside the tree's resolution.

    The curve Z-spread discounts cashflows continuously; the lattice spread is added
    to twenty discrete short rates. They are the same quantity computed by machinery
    that shares nothing, so agreement is evidence and a mismatch would be the first
    sign that the tree is calibrated to something else.
    """
    payload = call(
        registry,
        "bond_spreads",
        {
            "handle": curve_handle(registry),
            "bond": BOND,
            "price": 110.0,
            "volatility": 0.15,
            "firstCall": "2026-01-05",
        },
    )
    option = payload["embeddedOption"]
    assert isinstance(option, dict)
    assert float(option["zeroVolatilitySpread"]) == pytest.approx(
        float(payload["zSpread"]), abs=1e-4
    )


def test_an_issuers_call_costs_the_holder_spread(registry: ToolRegistry) -> None:
    """The option-adjusted spread is narrower than the zero-volatility one.

    The issuer's right to redeem early is worth something to the issuer, so once it
    is modelled a narrower spread explains the same price. The sign of the option
    cost is the whole content of the measure and it is not arbitrary.
    """
    payload = call(
        registry,
        "bond_spreads",
        {
            "handle": curve_handle(registry),
            "bond": BOND,
            "price": 110.0,
            "volatility": 0.15,
            "firstCall": "2026-01-05",
        },
    )
    option = payload["embeddedOption"]
    assert isinstance(option, dict)
    assert option["heldBy"] == "issuer"
    assert float(option["optionAdjustedSpread"]) < float(option["zeroVolatilitySpread"])
    assert float(option["optionCost"]) > 0.0
    assert float(option["optionCost"]) == pytest.approx(
        float(option["zeroVolatilitySpread"]) - float(option["optionAdjustedSpread"]),
        rel=1e-12,
    )
    # The model's own view: a callable bond is worth less than the same bullet.
    assert float(option["modelPriceWithOption"]) < float(option["modelBulletPrice"])
    assert float(option["modelOptionValue"]) > 0.0


def test_the_option_costs_less_the_further_the_bond_is_from_the_call(
    registry: ToolRegistry,
) -> None:
    """A property no self-consistency check could notice, and the real test.

    At a high price the call is deep in the money and expensive; at par it is barely
    worth anything. A wiring error that returned a plausible constant would pass
    every other assertion here and fail this one.
    """
    handle = curve_handle(registry)
    costs = []
    for price in (120.0, 110.0, 100.0):
        payload = call(
            registry,
            "bond_spreads",
            {
                "handle": handle,
                "bond": BOND,
                "price": price,
                "volatility": 0.15,
                "firstCall": "2026-01-05",
            },
        )
        option = payload["embeddedOption"]
        assert isinstance(option, dict)
        costs.append(float(option["optionCost"]))
    assert costs == sorted(costs, reverse=True)
    assert costs[0] > 5.0 * costs[-1]


def test_a_holders_put_reverses_the_sign(registry: ToolRegistry) -> None:
    """The option belongs to whoever holds it, and the cost follows.

    A put held by the holder is worth something to the holder, so it takes a *wider*
    spread to explain the same price once it is modelled.
    """
    payload = call(
        registry,
        "bond_spreads",
        {
            "handle": curve_handle(registry),
            "bond": BOND,
            "price": 110.0,
            "volatility": 0.15,
            "firstCall": "2026-01-05",
            "holderOption": True,
        },
    )
    option = payload["embeddedOption"]
    assert isinstance(option, dict)
    assert option["heldBy"] == "holder"
    assert float(option["optionCost"]) < 0.0
    assert float(option["modelOptionValue"]) < 0.0


def test_the_lattice_only_accepts_a_settlement_on_the_step_grid(
    registry: ToolRegistry,
) -> None:
    """And that is why the clean-to-full conversion is always a no-op.

    The step length is one coupon period, and a maturity that is not a whole number
    of steps from settlement is refused — so every settlement the lattice accepts
    falls on a coupon date, where the accrued interest is zero. The conversion stays
    in the code because it is what makes the two price conventions commensurable, and
    the constraint that makes it redundant belongs to the library rather than here.

    Both halves are asserted: a mid-period settlement is refused with the gap named,
    and on an accepted settlement the accrual really is zero and the two price
    conventions give the same spread.
    """
    handle = curve_handle(registry)
    arguments: dict[str, Any] = {
        "handle": handle,
        "bond": BOND,
        "price": 110.0,
        "volatility": 0.15,
        "firstCall": "2026-01-05",
    }

    error = refuse(registry, "bond_spreads", {**arguments, "settlement": "2021-04-05"})
    assert error["kind"] == "domain"
    message = str(error["message"])
    assert "whole number" in message
    assert "9.7589" in message

    as_clean = call(registry, "bond_spreads", {**arguments, "clean": True})
    as_full = call(registry, "bond_spreads", {**arguments, "clean": False})
    clean_option = as_clean["embeddedOption"]
    full_option = as_full["embeddedOption"]
    assert isinstance(clean_option, dict)
    assert isinstance(full_option, dict)
    assert float(clean_option["accrued"]) == pytest.approx(0.0, abs=1e-15)
    assert float(clean_option["fullPriceUsed"]) == pytest.approx(110.0, rel=1e-12)
    assert float(clean_option["optionAdjustedSpread"]) == pytest.approx(
        float(full_option["optionAdjustedSpread"]), rel=1e-12
    )


# -- refusals and handles ----------------------------------------------------


def test_a_moments_handle_is_refused_as_the_wrong_kind(registry: ToolRegistry) -> None:
    """Minted under the same key, so the MAC passes and the kind tag is what refuses.

    Without the tag the payload would be decoded as a curve and the KeyError would
    escape as an internal error rather than as something a caller can fix.
    """
    rng = random.Random(3)
    returns = [[rng.gauss(0.0, 0.01) for _ in range(3)] for _ in range(30)]
    moments = call(
        registry,
        "estimate_return_moments",
        {"returns": returns, "periodsPerYear": 252},
    )
    error = refuse(
        registry,
        "discount_curve_rates",
        {"handle": moments["handle"], "dates": ["2026-01-05"]},
    )
    assert error["kind"] == "handle_wrong_kind"
    assert "bootstrap_discount_curve" in str(error["message"])


def test_a_book_handle_is_refused_as_the_wrong_kind(registry: ToolRegistry) -> None:
    opened = call(
        registry,
        "open_position_book",
        {
            "spot": 100.0,
            "rate": 0.04,
            "legs": [
                {
                    "instrument": "call",
                    "strike": 100.0,
                    "time": 0.5,
                    "vol": 0.2,
                    "quantity": 1.0,
                }
            ],
        },
    )
    error = refuse(
        registry,
        "discount_curve_rates",
        {"handle": opened["handle"], "dates": ["2026-01-05"]},
    )
    assert error["kind"] == "handle_wrong_kind"


def test_a_handle_and_a_quote_set_together_are_refused(registry: ToolRegistry) -> None:
    error = refuse(
        registry,
        "discount_curve_rates",
        {"handle": curve_handle(registry), **QUOTES, "dates": ["2026-01-05"]},
    )
    assert error["kind"] == "domain"
    assert "exactly one" in str(error["message"])


def test_neither_a_handle_nor_a_quote_set_is_refused(registry: ToolRegistry) -> None:
    error = refuse(registry, "discount_curve_rates", {"dates": ["2026-01-05"]})
    assert "exactly one" in str(error["message"])


def test_a_curve_with_no_instruments_is_refused(registry: ToolRegistry) -> None:
    error = refuse(
        registry,
        "bootstrap_discount_curve",
        {"reference": REFERENCE, "basis": "ACT_365F"},
    )
    assert "at least one instrument" in str(error["message"])


def test_a_price_and_a_yield_together_are_refused(registry: ToolRegistry) -> None:
    error = refuse(
        registry,
        "bond_analytics",
        {"bond": BOND, "price": 120.0, "yieldToMaturity": 0.03, "reference": REFERENCE},
    )
    assert "not both" in str(error["message"])


def test_a_missing_basis_is_refused_by_the_schema(registry: ToolRegistry) -> None:
    """No default anywhere, which is the point of the module.

    A basis the caller did not choose is a different question answered, and the
    answer looks entirely normal.
    """
    error = refuse(
        registry,
        "bootstrap_discount_curve",
        {"reference": REFERENCE, "swaps": QUOTES["swaps"]},
    )
    assert error["kind"] == "invalid_input"
    assert "basis" in str(error["details"])


def test_an_unknown_basis_is_refused_with_the_permitted_set(
    registry: ToolRegistry,
) -> None:
    error = refuse(
        registry,
        "bootstrap_discount_curve",
        {"reference": REFERENCE, "basis": "ACT_ACT_ICMA", "swaps": QUOTES["swaps"]},
    )
    assert error["kind"] == "invalid_input"
    assert "ACT_365F" in str(error["details"])


def test_a_malformed_date_is_refused_by_the_schema(registry: ToolRegistry) -> None:
    error = refuse(
        registry,
        "bootstrap_discount_curve",
        {"reference": "05/01/2021", "basis": "ACT_365F", "swaps": QUOTES["swaps"]},
    )
    assert error["kind"] == "invalid_input"


def test_a_call_after_the_last_step_is_refused(registry: ToolRegistry) -> None:
    """An option exercisable only at redemption is not an option.

    Left alone it produces an empty exercise schedule and a callable bond priced
    exactly as a bullet, which is a wrong answer rather than an error.
    """
    error = refuse(
        registry,
        "bond_spreads",
        {
            "handle": curve_handle(registry),
            "bond": BOND,
            "price": 110.0,
            "volatility": 0.15,
            "firstCall": "2031-01-05",
        },
    )
    assert "not an option" in str(error["message"])


def test_too_many_instruments_is_refused_in_instruments(registry: ToolRegistry) -> None:
    swaps = [
        {
            "maturity": f"20{22 + index}-01-05",
            "rate": 0.01,
            "frequency": "SEMI_ANNUAL",
            "basis": "THIRTY_360_BOND",
        }
        for index in range(MAX_INSTRUMENTS + 5)
    ]
    error = refuse(
        registry,
        "bootstrap_discount_curve",
        {"reference": REFERENCE, "basis": "ACT_365F", "swaps": swaps},
    )
    assert error["kind"] == "invalid_input"
    assert "maxItems" in str(error["details"])


def test_too_many_buckets_is_refused(registry: ToolRegistry) -> None:
    error = refuse(
        registry,
        "bond_curve_risk",
        {
            "handle": curve_handle(registry),
            "bond": BOND,
            "buckets": [0.1 * index for index in range(1, MAX_BUCKETS + 5)],
        },
    )
    assert error["kind"] == "invalid_input"
