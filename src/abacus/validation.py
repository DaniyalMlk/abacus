"""Backtest validation: whether a track record is evidence of anything.

Every other tool here answers a question about a price or a position. These
answer a question about a *claim* — someone says this strategy earns a Sharpe of
1.5, and the honest reply depends entirely on facts about the search that found
it, none of which are in the number. The arithmetic comes from ``holdout``.

Four decisions shape it.

**The trial count that matters is the effective one, not the raw one.** Two
hundred variations of one moving-average rule are not two hundred independent
bets, and correcting as though they were over-penalises the result. Measured on
the forty one-factor trials in ``tests/test_validation.py``, the eigenvalue
method puts the effective count at 3.0 against a raw 40, and the deflated Sharpe
goes from 0.7196 to 0.8244 — ten points of probability the raw count throws
away. On forty independent trials the effective count is 40.0 and the deflated
figure is 0.5591 either way, identical to the digit, so using the effective
count costs nothing where it is not needed. Every tool here that deflates uses it, reports all three
estimates of it, and names the one it used.

**A Sharpe ratio is per period, and the standard error is reported beside it.**
The number people quote is annualised and the number these formulas take is not,
which is the single most likely way to get a wrong answer out of this group
quietly. Beyond stating it, the standard error is always in the result, because
it is usually the answer: an annualised Sharpe of 1.0 measured over 252
observations has an annualised standard error of 1.00, and over 30 observations
of 2.90. A track record of one year says almost nothing, and the figure that
says so should not have to be asked for.

**Bootstrap tools take a seed and report it.** The tests for superior predictive
ability resample, so an unseeded call returns a different p-value every time. A
tool annotated as idempotent that is not is worse than one that admits it, and a
caller comparing two runs would read the difference as a change in the data. The
seed defaults to a fixed value and comes back in the result, so a number can be
reproduced from the result alone.

**Refusals, where the arithmetic would otherwise produce a figure.** This group
needs them more than any other, because everything here will compute. A deflated
Sharpe from four observations is a number. An overfitting probability from two
strategies is a number — and it is drawn from a two-point distribution, since
with ``k`` strategies the logit takes at most ``k`` distinct values, measured: 2
distinct values at ``k = 2``, 3 at ``k = 3``, 10 at ``k = 10``. A probability
estimated on a two-point grid is not a probability anybody should act on, and
the fact that it prints to four decimal places is exactly the problem.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from holdout import (
    deflate_trials,
    effective_number_of_trials,
    estimate_sharpe,
    minimum_track_record_length,
    probability_of_backtest_overfitting,
    romano_wolf,
    sharpe_standard_error,
    superior_predictive_ability,
    trial_correlation,
)
from numpy.typing import NDArray

from .analytics import guard
from .tools import DomainError, ToolRegistry

if TYPE_CHECKING:  # pragma: no cover - the literal is only needed by the checker
    from holdout.trials import EffectiveMethod

#: Fewest observations any of these will work from. Below this the standard
#: error of a Sharpe ratio exceeds the estimate by a factor approaching three —
#: an annualised 1.0 from 30 observations carries an annualised standard error
#: of 2.90 — so the deflation arithmetic is being applied to noise. The figure is
#: reported at every length, because it is still 1.00 at 252 observations and a
#: caller who has not seen that will over-read a year of data.
MIN_OBSERVATIONS = 30

#: Fewest strategies the overfitting probability will accept. With ``k``
#: strategies the rank of the in-sample winner takes at most ``k`` values, so the
#: logit does too and the probability is estimated on a grid of that coarseness.
#: Measured: 2 distinct logits at two strategies, 3 at three, 10 at ten. A
#: probability printed to four places off a two-point grid reads as far more
#: precise than it is.
MIN_TRIALS = 5

#: Most trials any matrix here may carry. The correlation matrix is ``k^2`` and
#: the eigen-decomposition behind the effective count is ``k^3``.
MAX_TRIALS = 200

#: Most cells in a trial matrix, periods times trials. The transport caps the
#: body well above this; the point of the limit is that the combinatorial work
#: below is not something a caller should be able to start by pasting.
MAX_CELLS = 200_000

#: Most blocks the overfitting probability may be cut into. The number of
#: partitions is ``C(n, n/2)``: 12,870 at sixteen blocks and 184,756 at twenty,
#: a fourteenfold jump for a result that does not move fourteen times as much.
MAX_BLOCKS = 16

#: Most bootstrap replications. The library's own floor is 100.
MAX_BOOTSTRAP = 20_000

#: Seed used when the caller does not give one. Fixed rather than random so that
#: two identical calls give one answer; see the module docstring.
DEFAULT_SEED = 0

_EFFECTIVE_METHODS = ("average", "eigenvalue", "participation")

#: Which effective-trial estimate the deflation uses when the caller does not
#: choose. The eigenvalue method counts the dimensions the trials actually span,
#: which is the quantity the deflation wants; the participation ratio is more
#: aggressive and the average-correlation method assumes one common correlation.
DEFAULT_EFFECTIVE_METHOD = "eigenvalue"


# -- schema fragments --------------------------------------------------------

_TRIALS = {
    "type": "array",
    "minItems": MIN_OBSERVATIONS,
    "maxItems": 20_000,
    "items": {
        "type": "array",
        "minItems": 1,
        "maxItems": MAX_TRIALS,
        "items": {"type": "number"},
    },
    "description": (
        "Returns of every trial, one row per period and one column per trial. "
        "Decimal fractions: a 1% period is 0.01. These are per-period returns, "
        "not annualised ones, and every Sharpe ratio derived from them is per "
        "period too."
    ),
}

_PERIODS_PER_YEAR = {
    "type": "number",
    "exclusiveMinimum": 0,
    "description": (
        "Periods in a year, used only to annualise the figures that are reported "
        "both ways. 252 for daily equity data, 52 for weekly, 12 for monthly. "
        "Annualising weekly data with 252 overstates a Sharpe ratio by a factor "
        "of about 2.2, which does not look wrong enough to notice."
    ),
}

_SEED = {
    "type": "integer",
    "minimum": 0,
    "description": (
        f"Seed for the bootstrap. Defaults to {DEFAULT_SEED} rather than to "
        "randomness, so that the same call twice gives the same p-value. The "
        "seed used is in the result, so a figure can be reproduced from the "
        "result alone."
    ),
}


# -- input parsing -----------------------------------------------------------


def _matrix(args: dict[str, Any], key: str = "trials") -> NDArray[np.float64]:
    """Read a trial matrix, refusing the shapes that would still compute."""
    rows = args[key]
    width = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != width:
            raise DomainError(
                f"row 0 has {width} columns and row {index} has {len(row)}. Every "
                "row is one period across every trial, so the rows all have the "
                "same length; a ragged matrix usually means a trial with a "
                "different history, which has to be aligned before it can be "
                "ranked against the others.",
                field=key,
            )

    if len(rows) < MIN_OBSERVATIONS:
        raise DomainError(
            f"{len(rows)} observations is too few. Below {MIN_OBSERVATIONS} the "
            "standard error of a Sharpe ratio is larger than the ratio itself — "
            "an annualised 1.0 from 30 daily observations carries an annualised "
            "standard error of 2.90 — so a deflation computed from it is a "
            "correction applied to noise.",
            field=key,
        )
    if width > MAX_TRIALS:
        raise DomainError(
            f"{width} trials is over the {MAX_TRIALS} limit.",
            field=key,
        )
    cells = len(rows) * width
    if cells > MAX_CELLS:
        raise DomainError(
            f"{len(rows)} periods by {width} trials is {cells} values, over the "
            f"{MAX_CELLS} limit.",
            field=key,
        )

    matrix: NDArray[np.float64] = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise DomainError(
            "the matrix holds a value that is not a finite number. A missing "
            "period has to be filled or dropped across every trial at once, "
            "because the trials are ranked against each other period by period.",
            field=key,
        )
    return matrix


def _effective(matrix: NDArray[np.float64]) -> dict[str, float]:
    """Every estimate of the effective trial count, so none is picked silently."""
    if matrix.shape[1] == 1:
        return dict.fromkeys(_EFFECTIVE_METHODS, 1.0)
    correlation = trial_correlation(matrix)
    # The library types `method` as a Literal and this iterates a tuple of the
    # same strings, which the checker cannot connect. The cast asserts what
    # `_EFFECTIVE_METHODS` already is rather than widening the library's type.
    return {
        method: float(
            effective_number_of_trials(correlation, method=cast("EffectiveMethod", method))
        )
        for method in _EFFECTIVE_METHODS
    }


def _finite(value: float | None, digits: int = 8) -> float | None:
    """Round, or report an absence where there is no number to report.

    ``json.dumps`` writes a bare ``NaN`` or ``Infinity``, neither of which is
    JSON, and a strict parser rejects the whole message over one of them. The
    degradation regression here returns ``nan`` outright when the in-sample
    metric does not vary across partitions — a real and reachable case — so this
    is not defensive.
    """
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _sharpe_payload(
    matrix: NDArray[np.float64], column: int, periods_per_year: float | None
) -> dict[str, Any]:
    """A Sharpe ratio with the evidence for it, per period and annualised."""
    estimate = estimate_sharpe(matrix[:, column])
    error = sharpe_standard_error(
        estimate.value, estimate.n, skewness=estimate.skewness, kurtosis=estimate.kurtosis
    )
    payload: dict[str, Any] = {
        "sharpe": _finite(estimate.value),
        "standardError": _finite(error),
        "observations": estimate.n,
        "skewness": _finite(estimate.skewness),
        "excessKurtosis": _finite(estimate.kurtosis - 3.0),
    }
    if periods_per_year is not None:
        scale = math.sqrt(periods_per_year)
        payload["annualised"] = {
            "periodsPerYear": periods_per_year,
            "sharpe": _finite(estimate.value * scale),
            "standardError": _finite(error * scale),
        }
    return payload


class ValidationTools:
    """The backtest validation tool group."""

    def deflated_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        matrix = _matrix(args)
        periods_per_year = args.get("periodsPerYear")
        method = args.get("effectiveMethod", DEFAULT_EFFECTIVE_METHOD)
        effective = _effective(matrix)

        selected = args.get("selected")
        if selected is not None and not 0 <= selected < matrix.shape[1]:
            raise DomainError(
                f"selected is {selected}, and there are {matrix.shape[1]} trials "
                "numbered from zero. Leave it out to deflate the best of them, "
                "which is the usual question.",
                field="selected",
            )

        result = deflate_trials(matrix, selected=selected, n_trials=effective[method])
        raw = deflate_trials(matrix, selected=selected)
        return {
            "selected": result.selected,
            "sharpe": _sharpe_payload(matrix, result.selected, periods_per_year),
            "trials": matrix.shape[1],
            "effectiveTrials": {name: _finite(value, 4) for name, value in effective.items()},
            "effectiveMethod": method,
            "trialsVariance": _finite(result.trials_variance),
            "expectedMaximumSharpe": _finite(result.expected_maximum),
            "probabilisticSharpe": _finite(result.probabilistic),
            "deflatedSharpe": _finite(result.deflated),
            # The same figure computed against the raw count, so the size of the
            # correction the effective count buys is visible rather than implied.
            "deflatedSharpeAgainstRawTrials": _finite(raw.deflated),
        }

    def track_record_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        sharpe = float(args["sharpe"])
        benchmark = float(args.get("benchmark", 0.0))
        confidence = float(args.get("confidence", 0.95))
        skewness = float(args.get("skewness", 0.0))
        excess = float(args.get("excessKurtosis", 0.0))

        periods = minimum_track_record_length(
            sharpe,
            benchmark=benchmark,
            confidence=confidence,
            skewness=skewness,
            kurtosis=excess + 3.0,
        )
        payload: dict[str, Any] = {
            "sharpe": sharpe,
            "benchmark": benchmark,
            "confidence": confidence,
            "skewness": skewness,
            "excessKurtosis": excess,
            "minimumObservations": _finite(periods, 4),
        }
        periods_per_year = args.get("periodsPerYear")
        if periods_per_year is not None:
            payload["minimumYears"] = _finite(periods / float(periods_per_year), 4)
            payload["annualisedSharpe"] = _finite(sharpe * math.sqrt(float(periods_per_year)))
        return payload

    def effective_trials_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        matrix = _matrix(args)
        effective = _effective(matrix)
        return {
            "trials": matrix.shape[1],
            "observations": matrix.shape[0],
            "effectiveTrials": {name: _finite(value, 4) for name, value in effective.items()},
            "recommended": DEFAULT_EFFECTIVE_METHOD,
            "reduction": _finite(
                1.0 - effective[DEFAULT_EFFECTIVE_METHOD] / matrix.shape[1], 6
            ),
        }

    def overfitting_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        matrix = _matrix(args)
        blocks = int(args.get("blocks", 16))
        if matrix.shape[1] < MIN_TRIALS:
            raise DomainError(
                f"{matrix.shape[1]} strategies is too few. The rank of the "
                f"in-sample winner takes at most {matrix.shape[1]} values, so the "
                "probability is estimated on a grid that coarse and reads as far "
                f"more precise than it is. At least {MIN_TRIALS} are needed.",
                field="trials",
            )
        if blocks > MAX_BLOCKS:
            raise DomainError(
                f"{blocks} blocks gives C({blocks}, {blocks // 2}) partitions, over "
                f"the {MAX_BLOCKS}-block limit — sixteen blocks is already 12,870 "
                "partitions and twenty is 184,756, which is fourteen times the "
                "work for an estimate that does not move fourteen times as much.",
                field="blocks",
            )
        if matrix.shape[0] < blocks * 2:
            raise DomainError(
                f"{matrix.shape[0]} observations cut into {blocks} blocks leaves "
                f"{matrix.shape[0] // blocks} per block. Each half of a partition "
                "has to support a metric on its own, so use fewer blocks or a "
                "longer history.",
                field="blocks",
            )

        result = probability_of_backtest_overfitting(matrix, n_blocks=blocks)
        slope, intercept, r_squared = result.degradation
        return {
            "trials": result.n_strategies,
            "blocks": result.n_blocks,
            "partitions": result.n_partitions,
            "probabilityOfBacktestOverfitting": _finite(result.pbo),
            "probabilityOfLoss": _finite(result.probability_of_loss),
            "degradation": {
                "slope": _finite(slope),
                "intercept": _finite(intercept),
                "rSquared": _finite(r_squared),
            },
            "distinctLogits": len({round(float(value), 10) for value in result.logits}),
        }

    def predictive_ability_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        matrix = _matrix(args)
        benchmark_index = int(args.get("benchmark", 0))
        if not 0 <= benchmark_index < matrix.shape[1]:
            raise DomainError(
                f"benchmark is column {benchmark_index}, and there are "
                f"{matrix.shape[1]} columns numbered from zero.",
                field="benchmark",
            )
        if matrix.shape[1] < 2:
            raise DomainError(
                "one column is the benchmark and there is nothing left to test "
                "against it. Give the benchmark's returns and at least one "
                "candidate's in the same matrix.",
                field="trials",
            )

        bootstrap = int(args.get("bootstrap", 1000))
        if bootstrap > MAX_BOOTSTRAP:
            raise DomainError(
                f"{bootstrap} replications is over the {MAX_BOOTSTRAP} limit.",
                field="bootstrap",
            )
        seed = int(args.get("seed", DEFAULT_SEED))

        candidates = [index for index in range(matrix.shape[1]) if index != benchmark_index]
        differentials = matrix[:, candidates] - matrix[:, [benchmark_index]]

        spa = superior_predictive_ability(differentials, n_bootstrap=bootstrap, seed=seed)
        stepdown = romano_wolf(differentials, n_bootstrap=bootstrap, seed=seed)

        return {
            "benchmark": benchmark_index,
            "candidates": candidates,
            "seed": seed,
            "bootstrap": spa.n_bootstrap,
            "blockLength": _finite(spa.block_length, 4),
            "statistic": _finite(spa.statistic),
            # Three p-values, not one. The consistent estimate is the one to
            # read; the lower and upper are the bounds the test is between, and a
            # wide gap says the answer depends on which poorly performing models
            # are counted as contenders.
            "pValue": {
                "lower": _finite(spa.lower),
                "consistent": _finite(spa.consistent),
                "upper": _finite(spa.upper),
            },
            "bestCandidate": candidates[spa.best],
            "stepdown": {
                "alpha": stepdown.alpha,
                "adjustedPValues": [_finite(value) for value in stepdown.adjusted_pvalues],
                # `rejected` is already a list of column indices, strongest
                # first — not a mask. Enumerating it and treating the entries as
                # flags looks right and silently returns the wrong strategies.
                "rejected": [candidates[int(index)] for index in stepdown.rejected],
            },
        }

    def register(self, registry: ToolRegistry) -> ToolRegistry:
        read_only = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}
        effective_method = {
            "type": "string",
            "enum": list(_EFFECTIVE_METHODS),
            "description": (
                "Which estimate of the effective trial count to deflate against. "
                f"Defaults to '{DEFAULT_EFFECTIVE_METHOD}', which counts the "
                "dimensions the trials actually span. All three are reported "
                "whichever is chosen, because they disagree — on forty trials "
                "driven by one factor they give 2.49, 3.00 and 1.08 — and a caller "
                "should see that rather than be handed one of them."
            ),
        }

        registry.register(
            "deflated_sharpe_ratio",
            title="A Sharpe ratio corrected for how many things were tried",
            description=(
                "The probability that the best of a set of trials has a true "
                "Sharpe ratio above zero, given how many trials there were and "
                "how much they varied. A Sharpe of 1.5 picked out of two hundred "
                "attempts is a different object from one committed to in advance, "
                "and nothing in the number says which it is. The correction uses "
                "the *effective* number of trials rather than the raw count, "
                "because variations of one idea are not independent bets: on "
                "forty trials driven by a common factor the effective count is "
                "3.0 rather than 40, and the deflated figure 0.824 rather than "
                "0.720, while on forty independent trials the two agree to the "
                "digit. Both are reported. Returns are per period, and so is every "
                "Sharpe ratio here."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "trials": _TRIALS,
                    "selected": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Which trial to deflate, numbered from zero. Leave it "
                            "out to take the best, which is the usual question — "
                            "naming one that was chosen in advance asks something "
                            "different and much weaker."
                        ),
                    },
                    "effectiveMethod": effective_method,
                    "periodsPerYear": _PERIODS_PER_YEAR,
                },
                "required": ["trials"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.deflated_payload))

        registry.register(
            "minimum_track_record_length",
            title="How long a record must be to mean anything",
            description=(
                "The number of observations at which a Sharpe ratio of the given "
                "size, with the given higher moments, becomes distinguishable "
                "from the benchmark at the stated confidence. Skew and fat tails "
                "both lengthen it: negative skew in particular, because the "
                "estimator's error is larger exactly where the returns are. The "
                "Sharpe ratio here is per period — pass periodsPerYear to have "
                "the answer in years and the ratio echoed back annualised, so a "
                "mixed-up convention is visible in the result."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sharpe": {
                        "type": "number",
                        "description": (
                            "Observed Sharpe ratio, per period. An annualised 1.0 "
                            "on daily data is about 0.063 here. It has to exceed "
                            "the benchmark: below it no length of record makes it "
                            "significant, and the call is refused rather than "
                            "answered with an infinity."
                        ),
                    },
                    "benchmark": {
                        "type": "number",
                        "description": "Sharpe ratio to beat, per period. Defaults to zero.",
                    },
                    "confidence": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "exclusiveMaximum": 1,
                        "description": "One-sided confidence level. Defaults to 0.95.",
                    },
                    "skewness": {
                        "type": "number",
                        "description": "Skewness of the returns. Defaults to zero.",
                    },
                    "excessKurtosis": {
                        "type": "number",
                        "minimum": -2,
                        "description": (
                            "Excess kurtosis: zero for a normal distribution. "
                            "Given as excess rather than raw, because the two "
                            "differ by three and a raw 3.0 entered here as excess "
                            "quietly lengthens the answer. It is not independent "
                            "of the skewness: every distribution satisfies "
                            "kurtosis >= 1 + skewness^2, so an excess kurtosis "
                            "below skewness^2 - 2 describes nothing and is "
                            "refused rather than used. A skewness of -1.5 "
                            "therefore needs an excess kurtosis of at least 0.25."
                        ),
                    },
                    "periodsPerYear": _PERIODS_PER_YEAR,
                },
                "required": ["sharpe"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.track_record_payload))

        registry.register(
            "effective_trial_count",
            title="How many independent bets a set of trials really is",
            description=(
                "Three estimates of how many independent trials a correlated set "
                "amounts to, and the reduction against the raw count. Two hundred "
                "variations of one rule are not two hundred bets, and any "
                "multiple-testing correction applied as though they were is too "
                "harsh. The methods disagree and all three are returned: the "
                "average-correlation method assumes a single common correlation, "
                "the eigenvalue method counts the dimensions the trials span, and "
                "the participation ratio weights by how concentrated those "
                "dimensions are and is the most aggressive of the three."
            ),
            input_schema={
                "type": "object",
                "properties": {"trials": _TRIALS},
                "required": ["trials"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.effective_trials_payload))

        registry.register(
            "backtest_overfitting_probability",
            title="How often the in-sample winner loses out of sample",
            description=(
                "Split the history into blocks, take every way of halving them, "
                "pick the best strategy on one half and see where it ranks on the "
                "other. The share of splits where it lands in the bottom half is "
                "the probability of backtest overfitting: near one half means the "
                "selection carries no information, and above it means it is "
                "actively misleading. The degradation regression of out-of-sample "
                "on in-sample performance comes with it — a slope at or below "
                "zero says a better backtest predicts a worse future. Needs at "
                "least five strategies, because with k of them the estimate lives "
                "on a k-point grid."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "trials": _TRIALS,
                    "blocks": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": MAX_BLOCKS,
                        "description": (
                            "Blocks the history is cut into; must be even. "
                            "Defaults to 16, which is 12,870 partitions. The count "
                            "of partitions is C(n, n/2) and grows fast: twenty "
                            "blocks is 184,756."
                        ),
                    },
                },
                "required": ["trials"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.overfitting_payload))

        registry.register(
            "superior_predictive_ability",
            title="Whether any strategy really beats the benchmark",
            description=(
                "Hansen's test: given a benchmark and a set of candidates, "
                "whether the best of them beats it by more than the search itself "
                "would produce. This is the question people mean when they say a "
                "strategy beat the index, and comparing the winner to the "
                "benchmark directly answers a different and much easier one. "
                "Three p-values come back rather than one — a wide gap between "
                "the lower and upper bounds means the answer depends on which "
                "poor models are counted as contenders — and a Romano-Wolf "
                "stepdown names which individual candidates survive. The "
                "bootstrap is seeded and the seed is in the result, so a p-value "
                "can be reproduced from the result alone."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "trials": _TRIALS,
                    "benchmark": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Which column is the benchmark, numbered from zero. "
                            "Defaults to the first. Every other column is tested "
                            "against it."
                        ),
                    },
                    "bootstrap": {
                        "type": "integer",
                        "minimum": 100,
                        "maximum": MAX_BOOTSTRAP,
                        "description": (
                            "Stationary bootstrap replications. Defaults to 1000. "
                            "The block length is chosen from the data's own serial "
                            "correlation and reported."
                        ),
                    },
                    "seed": _SEED,
                },
                "required": ["trials"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.predictive_ability_payload))

        return registry


def register(registry: ToolRegistry) -> ToolRegistry:
    """Add the backtest validation tools to ``registry``."""
    return ValidationTools().register(registry)


__all__ = [
    "DEFAULT_EFFECTIVE_METHOD",
    "DEFAULT_SEED",
    "MAX_BLOCKS",
    "MAX_BOOTSTRAP",
    "MAX_CELLS",
    "MAX_TRIALS",
    "MIN_OBSERVATIONS",
    "MIN_TRIALS",
    "ValidationTools",
    "register",
]
