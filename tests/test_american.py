"""American exercise: lattices, approximations and the early-exercise boundary."""

from __future__ import annotations

from typing import Any

import pytest

from abacus.american import CONVERGENCE_STEPS, MAX_STEPS
from abacus.analytics import default_registry
from abacus.tools import ToolRegistry

PUT: dict[str, Any] = {
    "spot": 100.0,
    "strike": 100.0,
    "time": 1.0,
    "rate": 0.05,
    "vol": 0.2,
    "type": "put",
}


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


# -- the properties an American price must have ---------------------------


def test_an_american_put_is_worth_more_than_the_european_one(
    registry: ToolRegistry,
) -> None:
    got = ok(registry, "price_american_lattice", PUT)
    assert got["price"] > got["europeanPrice"]
    assert got["earlyExercisePremium"] > 0
    # The premium is the lattice-internal difference, so it reconciles against
    # the lattice European value and not the analytic one.
    assert got["earlyExercisePremium"] == pytest.approx(
        got["price"] - got["europeanPriceOnLattice"], abs=1e-9
    )


def test_an_american_call_on_a_non_dividend_share_equals_the_european_one(
    registry: ToolRegistry,
) -> None:
    # Merton's result: with carry equal to the rate there is never a reason to
    # exercise a call early, so the right to do so is worth nothing. A lattice
    # that produced a premium here would be wrong.
    got = ok(registry, "price_american_lattice", {**PUT, "type": "call"})
    # Exactly zero: on the lattice the two valuations coincide node for node.
    assert got["earlyExercisePremium"] == pytest.approx(0.0, abs=1e-12)
    # Against the analytic price the agreement is only as good as the lattice,
    # which is the distinction the two European fields exist to make.
    assert got["price"] == pytest.approx(got["europeanPrice"], rel=1e-3)


def test_a_dividend_paying_call_does_earn_an_early_exercise_premium(
    registry: ToolRegistry,
) -> None:
    got = ok(registry, "price_american_lattice", {**PUT, "type": "call", "carry": -0.04})
    assert got["earlyExercisePremium"] > 0


def test_an_american_price_is_never_below_intrinsic(registry: ToolRegistry) -> None:
    for spot in (60.0, 80.0, 100.0, 120.0, 150.0):
        got = ok(registry, "price_american_lattice", {**PUT, "spot": spot})
        assert got["price"] >= max(0.0, 100.0 - spot) - 1e-9


def test_an_american_price_is_never_below_the_european_one(
    registry: ToolRegistry,
) -> None:
    # The extra right cannot have negative value.
    for side in ("call", "put"):
        for carry in (0.05, 0.0, -0.03):
            got = ok(
                registry, "price_american_lattice", {**PUT, "type": side, "carry": carry}
            )
            assert got["price"] >= got["europeanPrice"] - 1e-9


# -- reporting the method -------------------------------------------------


def test_the_method_travels_with_the_price(registry: ToolRegistry) -> None:
    # The whole point: a bare float would invite a caller to treat a
    # discretisation artefact as a market fact.
    method = ok(registry, "price_american_lattice", PUT)["method"]
    assert method["steps"] == 512
    assert method["lattice"] == "crr"
    assert method["exercise"] == "american"
    assert method["exact"] is False


def test_the_closed_form_says_it_is_an_approximation(registry: ToolRegistry) -> None:
    method = ok(registry, "price_american_closed_form", PUT)["method"]
    assert method["exact"] is False
    assert "approximation" in method["note"]
    assert "price_american_lattice" in method["note"]


def test_the_closed_form_is_never_below_the_european_price(
    registry: ToolRegistry,
) -> None:
    # The 2002 formula is not uniformly sharper than the 1993 one and can fall
    # below the European price on its own; the library returns the best of the
    # three, so this must hold across a range of markets.
    for spot in (70.0, 85.0, 100.0, 115.0, 140.0):
        for vol in (0.1, 0.3, 0.6):
            got = ok(
                registry, "price_american_closed_form", {**PUT, "spot": spot, "vol": vol}
            )
            assert got["price"] >= got["europeanPrice"] - 1e-12


def test_the_lattice_and_the_closed_form_broadly_agree(registry: ToolRegistry) -> None:
    lattice = ok(registry, "price_american_lattice", {**PUT, "steps": 2048})["price"]
    closed = ok(registry, "price_american_closed_form", PUT)["price"]
    # The approximation is a lower bound and a close one, within about 2%.
    assert closed <= lattice + 1e-9
    assert closed == pytest.approx(lattice, rel=0.02)


# -- convergence ----------------------------------------------------------


def test_convergence_is_reported_only_when_asked_for(registry: ToolRegistry) -> None:
    assert "convergence" not in ok(registry, "price_american_lattice", PUT)
    assert "convergence" in ok(
        registry, "price_american_lattice", {**PUT, "convergence": True}
    )


def test_the_ladder_walks_the_requested_resolutions(registry: ToolRegistry) -> None:
    ladder = ok(registry, "price_american_lattice", {**PUT, "convergence": True})[
        "convergence"
    ]["ladder"]
    assert [row["steps"] for row in ladder] == list(CONVERGENCE_STEPS)


def test_the_answer_moves_less_as_the_lattice_refines(registry: ToolRegistry) -> None:
    # What the report is for: the successive changes shrinking is the evidence
    # that the price is settling rather than wandering.
    ladder = ok(registry, "price_american_lattice", {**PUT, "convergence": True})[
        "convergence"
    ]["ladder"]
    changes = [abs(row["changeFromPrevious"]) for row in ladder if "changeFromPrevious" in row]
    assert changes[-1] < changes[0]


def test_the_error_estimate_is_the_last_change_and_says_it_is_an_estimate(
    registry: ToolRegistry,
) -> None:
    report = ok(registry, "price_american_lattice", {**PUT, "convergence": True})[
        "convergence"
    ]
    assert report["estimatedError"] == pytest.approx(
        abs(report["ladder"][-1]["changeFromPrevious"]), abs=1e-15
    )
    # Not dressed up as a bound, because nothing here proves one.
    assert "not a proven bound" in report["note"]


def test_a_finer_lattice_lands_nearer_the_refined_answer(registry: ToolRegistry) -> None:
    reference = ok(registry, "price_american_lattice", {**PUT, "steps": 4096})["price"]
    coarse = ok(registry, "price_american_lattice", {**PUT, "steps": 64})["price"]
    fine = ok(registry, "price_american_lattice", {**PUT, "steps": 1024})["price"]
    assert abs(fine - reference) < abs(coarse - reference)


# -- lattices -------------------------------------------------------------


@pytest.mark.parametrize("lattice", ["crr", "jarrow-rudd", "trinomial"])
def test_every_lattice_agrees_on_the_price(registry: ToolRegistry, lattice: str) -> None:
    # Different discretisations of the same model must converge to the same
    # number; disagreement between them would mean one is wrong.
    got = ok(registry, "price_american_lattice", {**PUT, "lattice": lattice, "steps": 1024})
    assert got["price"] == pytest.approx(6.09, abs=0.01)
    assert got["method"]["lattice"] == lattice


def test_an_unknown_lattice_lists_the_ones_that_exist(registry: ToolRegistry) -> None:
    error = refused(registry, "price_american_lattice", {**PUT, "lattice": "quadnomial"})
    assert "crr" in error["details"][0]["message"]


def test_too_few_steps_for_the_market_is_refused_with_the_minimum(
    registry: ToolRegistry,
) -> None:
    # Below the floor the branch probabilities leave [0, 1] and the tree stops
    # being a probability model. It would still produce a number, which is why
    # this is checked rather than left to the caller.
    error = refused(
        registry,
        "price_american_lattice",
        {**PUT, "time": 30.0, "vol": 0.05, "rate": 0.4, "steps": 1},
    )
    assert error["kind"] == "domain"
    assert "probability model" in error["message"]


def test_an_unbounded_step_count_is_refused(registry: ToolRegistry) -> None:
    # A lattice is quadratic in its layer count, so this is a way to make the
    # server compute for a very long time on request.
    error = refused(registry, "price_american_lattice", {**PUT, "steps": MAX_STEPS + 1})
    assert error["details"][0]["keyword"] == "maximum"


def test_a_fractional_step_count_is_refused(registry: ToolRegistry) -> None:
    assert refused(registry, "price_american_lattice", {**PUT, "steps": 10.5})[
        "details"
    ][0]["keyword"] == "type"


# -- the exercise boundary ------------------------------------------------


def test_the_boundary_is_returned_as_labelled_points(registry: ToolRegistry) -> None:
    got = ok(registry, "american_exercise_boundary", {**PUT, "points": 10})
    assert len(got["boundary"]) <= 10
    for point in got["boundary"]:
        assert set(point) == {"time", "spot"}
        assert 0.0 <= point["time"] <= 1.0


def test_a_put_boundary_lies_below_the_strike(registry: ToolRegistry) -> None:
    got = ok(registry, "american_exercise_boundary", {**PUT, "points": 20})
    assert all(point["spot"] < 100.0 for point in got["boundary"])


def test_the_boundary_runs_forward_from_now_to_expiry(registry: ToolRegistry) -> None:
    got = ok(registry, "american_exercise_boundary", {**PUT, "points": 20})
    times = [point["time"] for point in got["boundary"]]
    assert times == sorted(times)


def test_a_put_boundary_rises_towards_the_strike_as_expiry_nears(
    registry: ToolRegistry,
) -> None:
    # Far from expiry it pays to hold; with little time left the holder
    # exercises at a spot much closer to the strike. Compared end to end rather
    # than pointwise, because of the parity artefact below.
    got = ok(registry, "american_exercise_boundary", {**PUT, "points": 20})
    assert got["boundary"][-1]["spot"] > got["boundary"][0]["spot"]


def test_the_binomial_read_out_warns_that_it_is_not_monotone(
    registry: ToolRegistry,
) -> None:
    # This looks like a bug and is not: consecutive CRR layers sample two
    # interleaved node grids of opposite parity, so the read-out alternates
    # even though the true boundary is monotone. Saying so is better than
    # letting a reader conclude the model is broken.
    method = ok(registry, "american_exercise_boundary", PUT)["method"]
    assert method["monotoneReadout"] is False
    assert "interleaved" in method["note"]
    assert "trinomial" in method["note"]


def test_the_trinomial_read_out_is_monotone_and_says_so(registry: ToolRegistry) -> None:
    got = ok(
        registry,
        "american_exercise_boundary",
        {**PUT, "lattice": "trinomial", "points": 30},
    )
    assert got["method"]["monotoneReadout"] is True
    spots = [point["spot"] for point in got["boundary"]]
    assert spots == sorted(spots)


def test_a_call_that_is_never_exercised_early_returns_only_expiry(
    registry: ToolRegistry,
) -> None:
    # With carry equal to the rate the exercise region is empty until expiry.
    # One point is the correct answer here, not a truncated one.
    got = ok(registry, "american_exercise_boundary", {**PUT, "type": "call"})
    assert len(got["boundary"]) == 1


def test_the_trigger_price_accompanies_the_boundary(registry: ToolRegistry) -> None:
    got = ok(registry, "american_exercise_boundary", PUT)
    assert got["immediateTrigger"] > 0
    assert got["immediateTrigger"] < 100.0
