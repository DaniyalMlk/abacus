"""The execution tools, checked against identities rather than against themselves.

A test that records what a function returned today and asserts it again tomorrow
proves only that nothing changed. The checks below are of the other kind: the
components of a decomposition have to sum to its total whatever the inputs, a
risk-neutral schedule has to be a straight line, a fixed cost has to move the
cost by exactly itself and not move the schedule at all, and the frontier has to
be monotone in both coordinates or it is not a frontier.

Where a number is asserted to a tolerance, the tolerance is there because the
arithmetic is floating point, not because the identity is approximate.
"""

from __future__ import annotations

import json
import math
from itertools import pairwise
from typing import Any

import pytest

from abacus.analytics import default_registry
from abacus.execution import MAX_FRONTIER_POINTS, ExecutionTools
from abacus.tools import DomainError

TOOLS = ExecutionTools()

#: The worked order used throughout: 900 of a 1000-share buy, filled twice.
ORDER: dict[str, Any] = {
    "side": "buy",
    "quantity": 1000.0,
    "fills": [
        {"quantity": 600.0, "price": 100.4, "commission": 6.0},
        {"quantity": 300.0, "price": 100.9, "commission": 3.0},
    ],
    "decisionPrice": 99.8,
    "arrivalPrice": 100.0,
    "finalPrice": 101.5,
    "fees": 0.5,
}

#: A million shares over five periods against a linear impact model.
PROBLEM: dict[str, Any] = {
    "quantity": 1_000_000.0,
    "horizon": 5.0,
    "periods": 5,
    "volatility": 0.95,
    "impact": {"gamma": 2.5e-7, "eta": 2.5e-6, "epsilon": 0.0625},
}


def problem(**changes: Any) -> dict[str, Any]:
    """A copy of ``PROBLEM`` with the impact model's fields overridable."""
    impact = dict(PROBLEM["impact"])
    for key in ("gamma", "eta", "epsilon"):
        if key in changes:
            impact[key] = changes.pop(key)
    return {**PROBLEM, **changes, "impact": impact}


def schedule(risk_aversion: float, **changes: Any) -> dict[str, Any]:
    return TOOLS.schedule_payload(
        {"problem": problem(**changes), "riskAversion": risk_aversion}
    )


# -- the decomposition -------------------------------------------------------


def test_the_components_sum_to_the_total() -> None:
    """Every component, and nothing else, adds up to the reported total.

    This is the property that makes the decomposition a decomposition. If a cost
    could sit in the total without appearing in a component, the four numbers a
    caller is meant to act on would not account for what they were charged.
    """
    result = TOOLS.shortfall_payload(ORDER)
    assert math.isclose(
        sum(result["components"].values()), result["total"], rel_tol=1e-9, abs_tol=1e-6
    )
    assert math.isclose(
        result["explicit"] + result["implicit"], result["total"], rel_tol=1e-9, abs_tol=1e-6
    )


def test_explicit_cost_is_exactly_the_contracted_part() -> None:
    """Explicit is commission plus fees, and nothing market-driven leaks into it."""
    result = TOOLS.shortfall_payload(ORDER)
    components = result["components"]
    assert math.isclose(
        result["explicit"], components["commission"] + components["fees"], abs_tol=1e-9
    )


def test_basis_points_are_the_currency_figures_over_the_paper_notional() -> None:
    """The two views of every component are the same number, scaled once."""
    result = TOOLS.shortfall_payload(ORDER)
    notional = result["paperNotional"]
    assert notional == pytest.approx(ORDER["quantity"] * ORDER["decisionPrice"])
    for name, value in result["components"].items():
        assert result["componentsBps"][name] == pytest.approx(1e4 * value / notional, abs=1e-6)


def test_the_delay_basis_reattributes_cost_without_changing_the_total() -> None:
    """Charging delay on the ordered or the executed quantity is a reattribution.

    Measured on this order the order basis gives a delay of 200 and an
    opportunity of 150, and the executed basis 180 and 170. The total is 869 both
    ways. This is worth pinning precisely because it survives a sanity check on
    the total: two desks using different conventions agree on the headline and
    disagree on which desk to blame.
    """
    on_order = TOOLS.shortfall_payload({**ORDER, "delayBasis": "order", "fees": 0.0})
    on_executed = TOOLS.shortfall_payload({**ORDER, "delayBasis": "executed", "fees": 0.0})

    assert on_order["components"]["delay"] == pytest.approx(200.0)
    assert on_order["components"]["opportunity"] == pytest.approx(150.0)
    assert on_executed["components"]["delay"] == pytest.approx(180.0)
    assert on_executed["components"]["opportunity"] == pytest.approx(170.0)

    assert on_order["total"] == pytest.approx(on_executed["total"])
    assert on_order["total"] == pytest.approx(869.0)
    assert on_order["delayBasis"] == "order"
    assert on_executed["delayBasis"] == "executed"


def test_the_order_the_fills_are_given_in_does_not_matter() -> None:
    """The decomposition reads quantities and prices, and the average is weighted.

    Fills arrive out of order often enough — several venues, one book — that a
    decomposition sensitive to their sequence would be quietly wrong rather than
    obviously so.
    """
    forward = TOOLS.shortfall_payload(ORDER)
    reversed_fills = TOOLS.shortfall_payload({**ORDER, "fills": list(reversed(ORDER["fills"]))})
    assert forward["components"] == reversed_fills["components"]
    assert forward["averagePrice"] == pytest.approx(reversed_fills["averagePrice"])


def test_a_sell_into_a_rising_market_gains_on_delay() -> None:
    """Every component changes sign with the side.

    A price that rises between the decision and the arrival costs a buyer and
    pays a seller, so the same numbers with the side flipped must give the delay
    component the opposite sign. Getting this backwards would produce a report
    that reads plausibly and points at the wrong half of the desk.
    """
    buy = TOOLS.shortfall_payload(ORDER)
    sell = TOOLS.shortfall_payload({**ORDER, "side": "sell"})
    assert buy["components"]["delay"] > 0.0
    assert sell["components"]["delay"] == pytest.approx(-buy["components"]["delay"])


def test_a_fully_filled_order_has_no_opportunity_cost() -> None:
    """Opportunity cost is the unfilled remainder marked to the close."""
    full = TOOLS.shortfall_payload(
        {**ORDER, "fills": [{"quantity": 1000.0, "price": 100.4, "commission": 10.0}]}
    )
    assert full["unfilledQuantity"] == pytest.approx(0.0)
    assert full["fillRate"] == pytest.approx(1.0)
    assert full["components"]["opportunity"] == pytest.approx(0.0)


def test_a_decision_price_equal_to_the_arrival_has_no_delay() -> None:
    """The default is a statement, not an absence: the order reached the market at once."""
    without = {key: value for key, value in ORDER.items() if key != "decisionPrice"}
    result = TOOLS.shortfall_payload(without)
    assert result["decisionPrice"] == ORDER["arrivalPrice"]
    assert result["components"]["delay"] == pytest.approx(0.0)


def test_overfilling_the_order_is_refused_with_both_numbers() -> None:
    """Fills exceeding the order are a data error, and a silent one otherwise.

    Taken at face value they would produce a negative unfilled quantity and an
    opportunity cost with the wrong sign — a plausible-looking report built on a
    book that was stitched together wrongly.

    The check belongs to `slippage`, which refuses it by name; `guard` turns
    that into the same recoverable result as every other refusal, so the test
    goes through the registered handler rather than calling the payload method
    directly. Keeping a second copy of the check here would be a second opinion
    on a question already answered, and the two would eventually disagree.
    """
    handler = default_registry().get("decompose_implementation_shortfall")
    assert handler is not None
    with pytest.raises(DomainError) as raised:
        handler.handler({**ORDER, "fills": [{"quantity": 1500.0, "price": 100.4}]})
    assert "1500" in str(raised.value)
    assert "1000" in str(raised.value)


# -- the schedule ------------------------------------------------------------


def test_risk_neutrality_is_exactly_a_straight_line() -> None:
    """At zero risk aversion the optimum is TWAP, to the last digit.

    This is the limiting case the closed form cannot evaluate — sinh(0)/sinh(0) —
    so it is taken by a separate branch, and a branch that is nearly right would
    be hard to notice against a schedule that is nearly straight anyway.
    """
    result = schedule(0.0)
    trades = result["trajectory"]["trades"]
    assert trades == [pytest.approx(200_000.0)] * 5
    assert result["halfLifeSensitivity"] is None
    assert "infinite" in result["note"]

    # The half-life here is genuinely infinite — an equal slice each period never
    # decays — and `json.dumps` writes a bare `Infinity` for that, which is not
    # JSON. A strict parser rejects the whole message, so one unbounded quantity
    # would destroy every number beside it.
    assert result["trajectory"]["halfLife"] is None
    json.dumps(result, allow_nan=False)


def test_the_trades_sum_to_the_quantity_and_the_holdings_run_to_zero() -> None:
    """A schedule that does not finish the order is not a schedule."""
    for aversion in (0.0, 1e-7, 1e-6, 1e-5):
        trajectory = schedule(aversion)["trajectory"]
        assert math.fsum(trajectory["trades"]) == pytest.approx(PROBLEM["quantity"], rel=1e-9)
        assert trajectory["holdings"][0] == pytest.approx(PROBLEM["quantity"])
        assert trajectory["holdings"][-1] == pytest.approx(0.0, abs=1e-6)
        holdings = trajectory["holdings"]
        assert all(later <= earlier + 1e-6 for earlier, later in pairwise(holdings))


def test_the_reported_cost_is_the_cost_of_the_reported_trades() -> None:
    """The schedule and its cost are recomputed together rather than trusted apart.

    Both figures come back, and they have to agree: a result that showed one
    schedule and priced another would be undetectable from the outside.
    """
    for aversion in (0.0, 1e-6, 1e-5):
        trajectory = schedule(aversion)["trajectory"]
        assert trajectory["expectedCost"] == pytest.approx(
            trajectory["expectedCostFromTrades"], rel=1e-9
        )
        assert trajectory["costVariance"] == pytest.approx(
            trajectory["costVarianceFromTrades"], rel=1e-9
        )
        assert trajectory["costStandardDeviation"] == pytest.approx(
            math.sqrt(trajectory["costVariance"]), rel=1e-9
        )


def test_raising_risk_aversion_front_loads_the_schedule() -> None:
    """More aversion buys less exposure with more impact, and shortens the half-life."""
    previous: dict[str, float] | None = None
    for aversion in (1e-7, 1e-6, 1e-5):
        trajectory = schedule(aversion)["trajectory"]
        current = {
            "half_life": trajectory["halfLife"],
            "cost": trajectory["expectedCost"],
            "std": trajectory["costStandardDeviation"],
            "first": trajectory["trades"][0],
        }
        if previous is not None:
            assert current["half_life"] < previous["half_life"]
            assert current["cost"] > previous["cost"]
            assert current["std"] < previous["std"]
            assert current["first"] > previous["first"]
        previous = current


def test_a_fixed_cost_moves_the_cost_by_itself_and_leaves_the_schedule_alone() -> None:
    """Epsilon is paid per unit traded whatever the order it is traded in.

    Measured: adding an epsilon of 0.0625 to a million-share order raises the
    expected cost by exactly 62,500 and does not move a single trade.
    """
    without = schedule(1e-6, epsilon=0.0)["trajectory"]
    with_fixed = schedule(1e-6, epsilon=0.0625)["trajectory"]
    assert with_fixed["trades"] == without["trades"]
    assert with_fixed["expectedCost"] - without["expectedCost"] == pytest.approx(
        0.0625 * PROBLEM["quantity"], rel=1e-9
    )


def test_permanent_impact_leaves_the_schedule_alone_only_in_the_limit() -> None:
    """Gamma enters through eta - gamma*tau/2, so its effect on the shape is O(tau).

    The textbook statement is that permanent impact drops out of the optimal
    schedule, and in continuous time it does. In discrete time it does not quite,
    and the size of the discrepancy is worth measuring rather than dismissing:
    on this problem it shortens the half-life by 2.5% at a period length of 1 and
    by 0.025% at a period length of 0.01 — a factor of ten in tau for a factor of
    ten in the error, which is the O(tau) the algebra predicts.
    """
    measured = []
    for periods in (5, 50, 500):
        flat = schedule(1e-6, periods=periods, gamma=0.0)["trajectory"]["halfLife"]
        charged = schedule(1e-6, periods=periods, gamma=2.5e-7)["trajectory"]["halfLife"]
        measured.append(abs(charged - flat) / flat)

    assert measured[0] == pytest.approx(0.0246, abs=5e-4)
    assert measured[1] == pytest.approx(0.0025, abs=5e-5)
    assert measured[2] == pytest.approx(0.00025, abs=5e-6)
    # Each tenfold cut in the period length cuts the discrepancy about tenfold.
    assert measured[0] / measured[1] == pytest.approx(10.0, rel=0.05)
    assert measured[1] / measured[2] == pytest.approx(10.0, rel=0.05)


def test_a_horizon_that_vanishes_over_its_periods_is_refused() -> None:
    """A period length that underflows to zero is a division, caught before it happens.

    A horizon of one subnormal over 500 periods rounds to a period length of
    exactly zero, and the closed form divides by it. The refusal names both
    numbers, because neither on its own looks wrong.
    """
    with pytest.raises(DomainError) as raised:
        TOOLS.schedule_payload(
            {"problem": problem(horizon=5e-324, periods=500), "riskAversion": 1e-6}
        )
    assert raised.value.field == "problem"


def test_an_impact_model_that_makes_instant_trading_cheapest_is_refused() -> None:
    """Permanent impact outweighing temporary impact has no interior optimum.

    Once `eta - gamma * tau / 2` goes negative, spreading the order out costs
    more than doing it at once and the schedule has no solution worth returning.
    The library refuses it by name and `guard` turns that into a recoverable
    result, so the check is that the refusal reaches a caller in the shape every
    other refusal has rather than as a traceback.
    """
    handler = default_registry().get("optimal_execution_schedule")
    assert handler is not None
    with pytest.raises(DomainError) as raised:
        handler.handler(
            {"problem": problem(eta=1e-9, gamma=1e-3), "riskAversion": 1e-6}
        )
    assert "permanent impact" in str(raised.value)


# -- the frontier ------------------------------------------------------------


def test_the_frontier_is_monotone_in_both_coordinates() -> None:
    """Cost rises and risk falls along it, or it is not a trade-off."""
    result = TOOLS.frontier_payload(
        {"problem": PROBLEM, "riskAversions": [0.0, 1e-8, 1e-7, 1e-6, 1e-5]}
    )
    costs = [point["expectedCost"] for point in result["points"]]
    risks = [point["costStandardDeviation"] for point in result["points"]]
    assert costs == sorted(costs)
    assert risks == sorted(risks, reverse=True)


def test_each_frontier_point_is_optimal_for_its_own_aversion() -> None:
    """The point at lambda beats every other point when scored at lambda.

    This is what makes the list a frontier rather than five schedules. Scoring
    each schedule under each aversion and insisting the diagonal wins is the
    only check here that would catch a frontier solved at the wrong lambdas.
    """
    aversions = [1e-8, 1e-7, 1e-6, 1e-5]
    result = TOOLS.frontier_payload({"problem": PROBLEM, "riskAversions": aversions})
    points = result["points"]

    for index, aversion in enumerate(aversions):
        own = points[index]
        variance = own["costStandardDeviation"] ** 2
        own_objective = own["expectedCost"] + aversion * variance
        assert own["objective"] == pytest.approx(own_objective, rel=1e-9)
        for other in points:
            other_objective = (
                other["expectedCost"] + aversion * other["costStandardDeviation"] ** 2
            )
            assert own_objective <= other_objective * (1 + 1e-9)


def test_the_frontier_point_at_zero_is_the_risk_neutral_schedule() -> None:
    """The frontier and the single-point solver agree where they overlap."""
    frontier = TOOLS.frontier_payload({"problem": PROBLEM, "riskAversions": [0.0, 1e-6]})
    alone = schedule(0.0)["trajectory"]
    assert frontier["points"][0]["trades"] == alone["trades"]


def test_unsorted_risk_aversions_are_refused() -> None:
    """A frontier read along an unsorted list appears to double back on itself."""
    with pytest.raises(DomainError) as raised:
        TOOLS.frontier_payload({"problem": PROBLEM, "riskAversions": [1e-5, 1e-7]})
    assert raised.value.field == "riskAversions"


def test_too_many_frontier_points_are_refused_with_the_count() -> None:
    with pytest.raises(DomainError) as raised:
        TOOLS.frontier_payload(
            {
                "problem": PROBLEM,
                "riskAversions": [1e-9 * (index + 1) for index in range(MAX_FRONTIER_POINTS + 1)],
            }
        )
    assert str(MAX_FRONTIER_POINTS) in str(raised.value)


# -- registration ------------------------------------------------------------


def test_the_three_tools_are_registered_with_schemas() -> None:
    """A tool the registry does not carry is unreachable however correct it is."""
    registry = default_registry()
    names = {tool.name for tool in registry}
    expected = {
        "decompose_implementation_shortfall",
        "optimal_execution_schedule",
        "execution_cost_frontier",
    }
    assert expected <= names
    for name in expected:
        tool = registry.get(name)
        assert tool is not None
        assert tool.input_schema["type"] == "object"
        assert tool.input_schema["additionalProperties"] is False
        assert tool.annotations["readOnlyHint"] is True
