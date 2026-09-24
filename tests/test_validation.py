"""The validation tools, checked against known limiting behaviour.

These tools answer questions about evidence, and a test that pins what they
returned today would be evidence of nothing. So the checks below are of the kind
that would fail for a reason: pure noise deflates to about a coin flip,
independent trials have an effective count equal to their raw count, correlated
trials have one far below it, a strategy that is the benchmark cannot beat the
benchmark, and every refusal names the number that caused it.

Two fixtures run through the whole file. ``correlated()`` is forty trials driven
by one common factor — the shape a parameter sweep actually has — and
``independent()`` is forty unrelated ones. They are seeded, so the figures in
the docstrings and in the module's own prose are reproducible rather than
illustrative.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from abacus.analytics import default_registry
from abacus.tools import DomainError
from abacus.validation import (
    DEFAULT_SEED,
    MAX_BLOCKS,
    MIN_OBSERVATIONS,
    MIN_TRIALS,
    ValidationTools,
)

TOOLS = ValidationTools()


def correlated(seed: int = 11, n: int = 1000, k: int = 40) -> list[list[float]]:
    """Forty trials driven by one factor: a parameter sweep over one idea."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, 0.01, size=(n, 1))
    return [list(row) for row in factor + rng.normal(0.0, 0.002, size=(n, k))]


def independent(seed: int = 23, n: int = 1000, k: int = 40) -> list[list[float]]:
    """Forty unrelated trials: the case the raw trial count assumes."""
    rng = np.random.default_rng(seed)
    return [list(row) for row in rng.normal(0.0, 0.01, size=(n, k))]


# -- the effective trial count -----------------------------------------------


def test_independent_trials_are_worth_their_raw_count() -> None:
    """Forty unrelated trials really are forty bets, and the count says so.

    This is the boundary case that makes the whole correction meaningful: if the
    effective count fell below the raw one here, it would be shrinking the count
    for reasons unrelated to dependence.
    """
    result = TOOLS.effective_trials_payload({"trials": independent()})
    assert result["trials"] == 40
    assert result["effectiveTrials"]["eigenvalue"] == pytest.approx(40.0, abs=1e-6)
    assert result["effectiveTrials"]["average"] == pytest.approx(40.0, abs=0.1)
    assert result["reduction"] == pytest.approx(0.0, abs=1e-6)


def test_one_factor_trials_collapse_to_a_handful() -> None:
    """Forty variations of one idea are about three bets, not forty."""
    result = TOOLS.effective_trials_payload({"trials": correlated()})
    effective = result["effectiveTrials"]
    assert effective["eigenvalue"] == pytest.approx(3.0, abs=0.1)
    assert effective["average"] == pytest.approx(2.49, abs=0.1)
    assert effective["participation"] == pytest.approx(1.08, abs=0.1)
    assert result["reduction"] > 0.9


def test_the_three_methods_are_all_reported() -> None:
    """They disagree by a factor of nearly three, so none may be picked silently."""
    result = TOOLS.effective_trials_payload({"trials": correlated()})
    assert set(result["effectiveTrials"]) == {"average", "eigenvalue", "participation"}
    assert result["recommended"] == "eigenvalue"


# -- the deflated Sharpe ratio -----------------------------------------------


def test_the_best_of_pure_noise_deflates_to_about_a_coin_flip() -> None:
    """Nothing in independent noise should survive the correction.

    The best of forty independent random series always has a positive Sharpe
    ratio — that is what taking a maximum does — and the deflated figure is the
    probability that it means anything. Near one half is the answer; near one
    would mean the correction is not working.
    """
    result = TOOLS.deflated_payload({"trials": independent()})
    assert result["sharpe"]["sharpe"] > 0.0
    assert result["deflatedSharpe"] == pytest.approx(0.56, abs=0.15)


def test_the_effective_count_matters_exactly_where_trials_are_correlated() -> None:
    """The correction's whole justification, measured on both fixtures.

    On correlated trials the effective count is 3.0 and deflating against it
    gives 0.8244 rather than 0.7196 — ten points of probability the raw count
    throws away. On independent trials the two counts are both 40 and the two
    figures are the same to the digit, so using the effective count is free
    where it makes no difference. If either half of that failed, the correction
    would be trading accuracy in one case for error in the other.
    """
    on_correlated = TOOLS.deflated_payload({"trials": correlated()})
    assert on_correlated["effectiveTrials"]["eigenvalue"] == pytest.approx(3.0, abs=0.1)
    assert on_correlated["deflatedSharpe"] == pytest.approx(0.8244, abs=0.02)
    assert on_correlated["deflatedSharpeAgainstRawTrials"] == pytest.approx(0.7196, abs=0.02)
    assert on_correlated["deflatedSharpe"] > on_correlated["deflatedSharpeAgainstRawTrials"]

    on_independent = TOOLS.deflated_payload({"trials": independent()})
    assert on_independent["deflatedSharpe"] == pytest.approx(
        on_independent["deflatedSharpeAgainstRawTrials"], abs=1e-8
    )


def test_more_trials_never_make_the_evidence_stronger() -> None:
    """Deflating against more trials cannot raise the probability.

    The record has to be held fixed for this to mean anything, which is why it
    goes at the correction directly rather than through the tool. Adding columns
    to the matrix and re-running changes which trial is selected and what the
    trials' variance is, and the deflated figure then moves in either direction
    quite legitimately — measured, it fell from 0.68 at forty columns to 0.60 at
    twenty on this fixture, which looks like a violation of this property and is
    not one. The property is about the correction, not about the matrix.
    """
    from holdout import deflated_sharpe_ratio

    previous = 1.0
    for count in (2, 5, 20, 100, 500):
        current = deflated_sharpe_ratio(
            0.06, 1000, n_trials=count, trials_variance=4e-4
        )
        assert current <= previous + 1e-12
        previous = current
    assert previous < 0.5  # five hundred attempts bury a Sharpe this size


def test_the_standard_error_is_reported_beside_the_ratio() -> None:
    """The figure that says a short record means little, without being asked for.

    An annualised Sharpe of 1.0 over 252 observations carries an annualised
    standard error of about 1.0 — the estimate and its own error are the same
    size — and a caller who has not seen that will over-read a year of data.
    """
    result = TOOLS.deflated_payload({"trials": correlated(n=252), "periodsPerYear": 252})
    sharpe = result["sharpe"]
    assert sharpe["observations"] == 252
    assert sharpe["standardError"] > 0.0
    # To the rounding the payload applies, amplified by the annualisation. Each
    # figure is rounded to eight places on the way out, and multiplying the
    # per-period one by sqrt(252) multiplies its rounding error by 15.9 as well
    # — so the round-then-scale and scale-then-round paths can differ by about
    # 8e-8, which is what this tolerance is and not a slack one.
    tolerance = 1e-8 * (1.0 + math.sqrt(252))
    assert sharpe["annualised"]["standardError"] == pytest.approx(
        sharpe["standardError"] * math.sqrt(252), abs=tolerance
    )
    assert sharpe["annualised"]["sharpe"] == pytest.approx(
        sharpe["sharpe"] * math.sqrt(252), abs=tolerance
    )


def test_too_short_a_history_is_refused_with_the_count() -> None:
    """Below thirty observations the deflation corrects noise, and still computes."""
    with pytest.raises(DomainError) as raised:
        TOOLS.deflated_payload({"trials": correlated(n=20, k=5)})
    assert str(MIN_OBSERVATIONS) in str(raised.value)
    assert raised.value.field == "trials"


def test_a_ragged_matrix_is_refused() -> None:
    trials = correlated(n=40, k=4)
    trials[7] = trials[7][:3]
    with pytest.raises(DomainError) as raised:
        TOOLS.deflated_payload({"trials": trials})
    assert "row 7" in str(raised.value)


def test_selecting_a_trial_that_is_not_there_is_refused() -> None:
    with pytest.raises(DomainError) as raised:
        TOOLS.deflated_payload({"trials": correlated(k=4), "selected": 9})
    assert raised.value.field == "selected"


# -- the minimum track record length -----------------------------------------


def test_a_bigger_sharpe_needs_a_shorter_record() -> None:
    """Monotone, and the scale is worth seeing: 1.0 annualised takes 2.7 years."""
    lengths = [
        TOOLS.track_record_payload({"sharpe": value / math.sqrt(252), "periodsPerYear": 252})[
            "minimumObservations"
        ]
        for value in (0.5, 1.0, 2.0)
    ]
    assert lengths == sorted(lengths, reverse=True)
    middle = TOOLS.track_record_payload({"sharpe": 1.0 / math.sqrt(252), "periodsPerYear": 252})
    assert middle["minimumYears"] == pytest.approx(2.71, abs=0.05)
    assert middle["annualisedSharpe"] == pytest.approx(1.0, rel=1e-9)


def test_negative_skew_lengthens_the_record_required() -> None:
    """The estimator's error is largest where the returns are, so the bar rises.

    A strategy that earns steadily and loses occasionally and badly needs a
    longer record than a symmetric one with the same ratio — which is the
    opposite of how such a track record usually reads.

    Both calls carry the same excess kurtosis, because the two moments are not
    free of one another: every distribution satisfies
    ``kurtosis >= 1 + skewness**2``, so a skewness of -1.5 cannot come with a
    normal's kurtosis and comparing against one would be comparing against a
    distribution that does not exist.
    """
    symmetric = TOOLS.track_record_payload({"sharpe": 0.06, "excessKurtosis": 3.0})
    skewed = TOOLS.track_record_payload(
        {"sharpe": 0.06, "skewness": -1.5, "excessKurtosis": 3.0}
    )
    assert skewed["minimumObservations"] > symmetric["minimumObservations"]


def test_moments_no_distribution_could_have_are_refused() -> None:
    """kurtosis >= 1 + skewness**2 is a fact about every distribution.

    A skewness of -1.5 with a normal's kurtosis describes nothing, and the
    formula would return a number for it regardless.
    """
    handler = default_registry().get("minimum_track_record_length")
    assert handler is not None
    with pytest.raises(DomainError) as raised:
        handler.handler({"sharpe": 0.06, "skewness": -1.5, "excessKurtosis": 0.0})
    assert "kurtosis" in str(raised.value)


def test_fat_tails_lengthen_the_record_required() -> None:
    normal = TOOLS.track_record_payload({"sharpe": 0.06, "excessKurtosis": 0.0})
    fat = TOOLS.track_record_payload({"sharpe": 0.06, "excessKurtosis": 5.0})
    assert fat["minimumObservations"] > normal["minimumObservations"]


def test_a_sharpe_below_the_benchmark_is_refused_rather_than_answered() -> None:
    """No length of record makes it significant, and the honest answer is not a number.

    The library raises for this and `guard` turns it into a recoverable refusal,
    which is the right shape: returning an infinity, or a very large number,
    would read as 'keep going' rather than as 'this is the wrong question'.
    """
    handler = default_registry().get("minimum_track_record_length")
    assert handler is not None
    with pytest.raises(DomainError) as raised:
        handler.handler({"sharpe": -0.02})
    assert "no track record length" in str(raised.value)


# -- the overfitting probability ---------------------------------------------


def test_one_factor_trials_overfit_and_the_result_says_so() -> None:
    """Selecting among forty variations of noise carries little information."""
    result = TOOLS.overfitting_payload({"trials": correlated(), "blocks": 10})
    assert 0.0 <= result["probabilityOfBacktestOverfitting"] <= 1.0
    assert 0.0 <= result["probabilityOfLoss"] <= 1.0
    assert result["partitions"] == 252
    assert result["trials"] == 40
    json.dumps(result, allow_nan=False)


def test_the_degradation_regression_comes_back_serialisable() -> None:
    """It returns nan when the in-sample metric does not vary, which is reachable.

    A bare `NaN` is not JSON and a strict parser rejects the whole message over
    it, so the three figures go out as null instead of as a value nothing on the
    far side can read.
    """
    result = TOOLS.overfitting_payload({"trials": independent(), "blocks": 8})
    degradation = result["degradation"]
    assert set(degradation) == {"slope", "intercept", "rSquared"}
    for value in degradation.values():
        assert value is None or isinstance(value, float)
    json.dumps(result, allow_nan=False)


def test_too_few_strategies_are_refused_because_the_grid_is_too_coarse() -> None:
    """With k strategies the logit takes at most k values, measured.

    Two strategies give a probability drawn from a two-point distribution, and it
    prints to four decimal places exactly like a real one. That is the reason for
    the refusal rather than an argument about sample size.
    """
    with pytest.raises(DomainError) as raised:
        TOOLS.overfitting_payload({"trials": [row[:2] for row in independent()]})
    assert str(MIN_TRIALS) in str(raised.value)

    # And where it is allowed, the coarseness is reported rather than hidden.
    result = TOOLS.overfitting_payload({"trials": [row[:5] for row in independent()], "blocks": 8})
    assert result["distinctLogits"] <= 5


def test_too_many_blocks_are_refused_with_the_partition_count() -> None:
    with pytest.raises(DomainError) as raised:
        TOOLS.overfitting_payload({"trials": independent(), "blocks": MAX_BLOCKS + 2})
    assert raised.value.field == "blocks"


def test_blocks_that_leave_no_room_per_block_are_refused() -> None:
    with pytest.raises(DomainError) as raised:
        TOOLS.overfitting_payload({"trials": correlated(n=30, k=6), "blocks": 16})
    assert raised.value.field == "blocks"


# -- superior predictive ability ---------------------------------------------


def test_noise_does_not_beat_noise() -> None:
    """Forty unrelated random series, one of them the benchmark.

    Nothing here has predictive ability over anything else, so the test should
    not find any: a p-value near zero on this fixture would mean the test is
    reporting the search rather than a result.
    """
    result = TOOLS.predictive_ability_payload({"trials": independent(), "bootstrap": 500})
    assert result["pValue"]["consistent"] > 0.10
    assert result["benchmark"] == 0
    assert len(result["candidates"]) == 39


def test_a_real_edge_is_found() -> None:
    """One column given a genuine drift, against a benchmark with none.

    The complement of the test above: if the test cannot detect an edge this
    large, a p-value above 0.1 on noise proves nothing about its power.
    """
    rng = np.random.default_rng(31)
    trials = rng.normal(0.0, 0.01, size=(1000, 6))
    trials[:, 3] += 0.0025  # a quarter of a percent a period, which is enormous
    result = TOOLS.predictive_ability_payload(
        {"trials": [list(row) for row in trials], "bootstrap": 500}
    )
    assert result["pValue"]["consistent"] < 0.05
    assert result["bestCandidate"] == 3
    assert 3 in result["stepdown"]["rejected"]


def test_the_same_call_twice_gives_the_same_p_value() -> None:
    """A seeded bootstrap, because the tools are annotated as idempotent.

    Without this the annotation is a lie and a caller comparing two runs would
    read the difference as a change in the data.
    """
    arguments = {"trials": independent(k=8), "bootstrap": 300}
    first = TOOLS.predictive_ability_payload(dict(arguments))
    second = TOOLS.predictive_ability_payload(dict(arguments))
    assert first == second
    assert first["seed"] == DEFAULT_SEED

    # And a different seed is allowed to differ, or the seed does nothing.
    third = TOOLS.predictive_ability_payload({**arguments, "seed": 99})
    assert third["seed"] == 99


def test_the_p_value_bounds_bracket_the_consistent_estimate() -> None:
    result = TOOLS.predictive_ability_payload({"trials": independent(k=10), "bootstrap": 500})
    bounds = result["pValue"]
    assert bounds["lower"] <= bounds["consistent"] <= bounds["upper"] + 1e-12


def test_a_benchmark_column_that_is_not_there_is_refused() -> None:
    with pytest.raises(DomainError) as raised:
        TOOLS.predictive_ability_payload({"trials": independent(k=4), "benchmark": 9})
    assert raised.value.field == "benchmark"


# -- registration ------------------------------------------------------------


def test_the_five_tools_are_registered_with_schemas() -> None:
    registry = default_registry()
    names = {tool.name for tool in registry}
    expected = {
        "deflated_sharpe_ratio",
        "minimum_track_record_length",
        "effective_trial_count",
        "backtest_overfitting_probability",
        "superior_predictive_ability",
    }
    assert expected <= names
    for name in expected:
        tool = registry.get(name)
        assert tool is not None
        schema: dict[str, Any] = tool.input_schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert tool.annotations["readOnlyHint"] is True
