"""Portfolio risk: covariance estimation, tail risk, contributions and drawdown.

Everything in :mod:`abacus.analytics`, :mod:`abacus.vol` and :mod:`abacus.book`
answers a question about an option. This module answers questions about a
portfolio of anything, from a matrix of returns, and the numbers come from
``shortfall`` rather than from here.

Four decisions shape it.

**The method is named in every result, because the number does not identify it.**
A one-day 99% value at risk of 2.3% under a normal assumption and 3.1% from the
sample are the same quantity estimated two ways, and nothing about either figure
says which it is. So every result carries the method, the confidence, the
observation count it was estimated from, and — where the method has them — its
own diagnostics: the degrees of freedom, the moments the correction used, whether
that correction was a valid quantile function at all.

**A returns handle carries the second moments, not the matrix.** This is the
uncomfortable consequence of a stateless protocol and it is better stated than
discovered. A handle has to fit inside a message a model carries through its
context; :mod:`abacus.handles` caps it at 8192 characters for that reason. A year
of daily returns on four assets encodes to roughly 6000 characters and five years
on ten assets to about 69,000, so a handle that carried the matrix would refuse
almost every portfolio worth asking about. A mean vector and a covariance matrix
are ``n + n^2`` numbers regardless of how long the history is, which fits
comfortably up to a couple of dozen assets.

What that buys: the parametric estimators — which need nothing but the first two
moments and the weights — run from a handle, so a matrix is sent once and then
reweighted, decomposed and rebalanced across as many calls as the caller likes.

What it costs: the historical estimators and every drawdown statistic read the
*path*, and a second-moment summary has thrown the path away. Those tools require
the matrix, and say so when handed a handle instead of guessing. The Cornish-Fisher
correction is in the same position for a subtler reason — it needs the skewness and
excess kurtosis of the *portfolio*, which depend on the weights, so they cannot be
precomputed into a handle that does not know them.

**Weights are not normalised silently.** Weights summing to 0.98 are either a
2% cash position or a typo, and the two want opposite treatment. The sum is
reported on every result that takes weights, and a sum far from one is refused
with the total named. Normalising quietly would turn a data error into a
plausible answer.

**Annualised figures are reported alongside per-period ones, never instead.**
A value at risk is a loss over one period of whatever frequency the data has, and
the ``252`` that appears in most code is the number of US equity trading days.
Annualising weekly data with it overstates volatility by a factor of seven, which
does not look wrong enough to notice, so ``periodsPerYear`` is a required argument
and both figures are returned with the frequency stated between them.
"""

from __future__ import annotations

import math
from typing import Any

from shortfall import (
    Allocation,
    Convention,
    Distribution,
    Drawdown,
    Panel,
    QuantileMethod,
    Risk,
    concentration,
    correlation,
    diagnose,
    diversification_ratio,
    drawdowns,
    effective_bets,
    filtered_historical_risk,
    historical_risk,
    ledoit_wolf,
    longest_underwater,
    maximum_drawdown,
    risk_contributions,
    risk_parity,
    sample_covariance,
    ulcer_index,
    volatility_contributions,
)
from shortfall import (
    calmar as calmar_ratio,
)
from shortfall import (
    downside_deviation as downside_deviation_of,
)
from shortfall import (
    portfolio_risk as parametric_portfolio_risk,
)
from shortfall import (
    sortino as sortino_ratio,
)
from shortfall import validate as validate_risk_model
from shortfall.parametric import is_monotone
from shortfall.volatility import MIN_OBSERVATIONS as MIN_GARCH_OBSERVATIONS
from shortfall.volatility import Innovation, fat_tail_test, fit_garch

from .analytics import guard
from .handles import HandleError, HandleTooLarge, Minter
from .tools import DomainError, ToolExecutionError, ToolRegistry

#: Tag inside a handle's payload saying what kind of state it holds. The position
#: book mints handles too, and a book presented to a risk tool has to be refused
#: as the wrong kind rather than misread as an empty one.
MOMENTS_KIND = "return-moments"

#: Most assets a returns matrix may hold. The binding constraint is the handle:
#: a covariance matrix is ``n^2`` numbers, and measured against this server's
#: 8192-character handle limit the payload is about 1100 characters at ten assets,
#: 5400 at twenty-five and 13,000 at forty. Twenty-four leaves room to spare, and
#: a caller over it is told in assets rather than in characters.
MAX_ASSETS = 24

#: Most cells a returns matrix may hold, counting periods times assets. Twenty
#: years of daily data on four assets, or eight years on ten. The transport
#: refuses a body over four megabytes, which this sits well inside; the point of
#: the cap is that a model should not be able to spend the server's time by
#: pasting a spreadsheet.
MAX_RETURN_CELLS = 40_000

#: Fewest observations any estimator here will work from. Below this a covariance
#: is not an estimate of anything: at ten observations on four assets there are
#: ten covariance parameters and forty numbers to fit them from.
MIN_OBSERVATIONS = 12

#: How far the weights may sum from one before the call is refused. Two percent
#: is wide enough for a genuine cash position expressed as a residual and narrow
#: enough that a dropped or duplicated leg does not slip through.
WEIGHT_SUM_TOLERANCE = 0.02

#: Widest confidence level accepted. Above this the tail holds so few
#: observations that a historical estimate is reading one or two of them, and a
#: parametric one is extrapolating a shape nobody fitted out there.
MAX_CONFIDENCE = 0.9999

#: Most one-step-ahead forecasts a validation call may carry. A forecast series
#: is one number per period, so this is twenty years of daily data — far more
#: than the asymptotic tests need and well inside the transport's body limit.
MAX_FORECASTS = 5_000

#: Most replications the expected-shortfall null may be simulated with. The
#: simulation is O(replications * observations) in pure Python, and the p-value
#: it produces has a simulation error of its own around
#: sqrt(p(1-p)/replications) — at 5000 a true 5% is resolved to about three
#: tenths of a percentage point, and more replications buy a digit nobody
#: should be reading.
MAX_REPLICATIONS = 5_000

#: Most one-step-ahead volatilities returned in a payload. The series is the
#: large part of it — one float per period — and a caller wanting the
#: parameters from a longer history can ask for them without it.
MAX_FORECAST_SERIES = 5_000

_METHODS = ("normal", "student-t", "cornish-fisher", "historical", "filtered-historical")

#: The methods that read the return path rather than a summary of it, and so
#: cannot run from a handle. Named here so the refusal and the tool descriptions
#: cannot drift apart.
_PATH_METHODS = ("cornish-fisher", "historical", "filtered-historical")

_DISTRIBUTIONS = {
    "normal": Distribution.NORMAL,
    "student-t": Distribution.STUDENT_T,
    "cornish-fisher": Distribution.CORNISH_FISHER,
}


# -- input schemas -----------------------------------------------------------

_RETURNS = {
    "type": "array",
    "minItems": MIN_OBSERVATIONS,
    "items": {
        "type": "array",
        "minItems": 1,
        "maxItems": MAX_ASSETS,
        "items": {"type": "number", "exclusiveMinimum": -1, "maximum": 10},
    },
    "description": (
        "Periodic returns as decimal fractions, one row per period and one column "
        "per asset, in the same column order as `assets`. A 1% gain is 0.01, not 1. "
        "Rows must all be the same length."
    ),
}

_ASSETS = {
    "type": "array",
    "minItems": 1,
    "maxItems": MAX_ASSETS,
    "items": {"type": "string", "minLength": 1, "maxLength": 64},
    "uniqueItems": True,
    "description": (
        "Column names, in the order the columns appear in `returns`. Defaults to "
        "asset_1, asset_2 and so on. Names are carried through to every per-asset "
        "figure, so supplying them is what makes a contribution table readable."
    ),
}

_WEIGHTS = {
    "type": "array",
    "minItems": 1,
    "maxItems": MAX_ASSETS,
    "items": {"type": "number", "minimum": -10, "maximum": 10},
    "description": (
        "Portfolio weights, one per asset, in the same order as the columns. They "
        "are not normalised: the sum is reported back, and a sum more than 2% away "
        "from one is refused, because weights summing to 0.98 are either a cash "
        "position or a typo and the two want opposite treatment. Negative weights "
        "are short positions."
    ),
}

_PERIODS_PER_YEAR = {
    "type": "number",
    "exclusiveMinimum": 0,
    "maximum": 366,
    "description": (
        "Periods in a year for the data's frequency: 252 for daily trading days, "
        "52 weekly, 12 monthly. Used only to annualise, never to reinterpret the "
        "returns. There is no default because the usual 252 is wrong for every "
        "frequency but one, and annualising weekly data with it overstates "
        "volatility sevenfold."
    ),
}

_CONFIDENCE = {
    "type": "number",
    "exclusiveMinimum": 0.5,
    "maximum": MAX_CONFIDENCE,
    "description": (
        "Confidence level: 0.99 means the 1% tail. The complement convention is "
        "the common mistake, and it is wrong by an amount that grows as the tail "
        "thins, so the result states the tail probability as well."
    ),
}

_CONVENTION = {
    "type": "string",
    "enum": ["simple", "log"],
    "description": (
        "Whether the returns are simple (P/P-1 - 1) or logarithmic (ln(P/P)). "
        "They differ by about half the variance and aggregate in opposite "
        "directions: log returns add across time, simple returns add across a "
        "portfolio. Defaults to simple."
    ),
}

_HANDLE = {
    "type": "string",
    "minLength": 8,
    "description": (
        "A handle from estimate_return_moments, in place of `returns`. It carries "
        "the mean vector and the covariance matrix rather than the returns "
        "themselves, so it serves the parametric methods and cannot serve the ones "
        "that read the return path. It expires; estimate again if it does."
    ),
}


def _optional(schema: dict[str, Any], description: str) -> dict[str, Any]:
    """A schema fragment with its description replaced."""
    return {**schema, "description": description}


# -- input parsing -----------------------------------------------------------


class Moments:
    """First and second moments of a return panel, and where they came from.

    This is what a returns handle carries and what the parametric estimators
    consume. It deliberately does not hold the returns: see the module docstring
    for why, and :meth:`require_path` for what that makes impossible.
    """

    def __init__(
        self,
        *,
        assets: list[str],
        mean: list[float],
        covariance: list[list[float]],
        observations: int,
        periods_per_year: float,
        estimator: str,
        convention: str,
        intensity: float | None = None,
    ) -> None:
        self.assets = assets
        self.mean = mean
        self.covariance = covariance
        self.observations = observations
        self.periods_per_year = periods_per_year
        self.estimator = estimator
        self.convention = convention
        self.intensity = intensity

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": MOMENTS_KIND,
            "assets": self.assets,
            "mean": self.mean,
            "covariance": self.covariance,
            "observations": self.observations,
            "periodsPerYear": self.periods_per_year,
            "estimator": self.estimator,
            "convention": self.convention,
            "intensity": self.intensity,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Moments:
        if payload.get("kind") != MOMENTS_KIND:
            raise ToolExecutionError(
                "that handle does not hold return moments. A handle from "
                "open_position_book holds option positions and cannot be read as a "
                "covariance estimate; call estimate_return_moments to get one of "
                "these.",
                kind="handle_wrong_kind",
                details=[
                    {
                        "path": "/handle",
                        "keyword": "handle_wrong_kind",
                        "message": f"handle holds {payload.get('kind', 'something else')!r}",
                    }
                ],
            )
        return cls(
            assets=list(payload["assets"]),
            mean=[float(value) for value in payload["mean"]],
            covariance=[[float(value) for value in row] for row in payload["covariance"]],
            observations=int(payload["observations"]),
            periods_per_year=float(payload["periodsPerYear"]),
            estimator=str(payload["estimator"]),
            convention=str(payload["convention"]),
            intensity=None if payload.get("intensity") is None else float(payload["intensity"]),
        )

    @classmethod
    def estimate(cls, panel: Panel, *, periods_per_year: float, estimator: str) -> Moments:
        if estimator == "sample":
            matrix = sample_covariance(panel)
            intensity = None
        else:
            shrunk = ledoit_wolf(panel)
            matrix = shrunk.matrix
            intensity = shrunk.intensity
        return cls(
            assets=list(panel.names),
            mean=list(panel.means()),
            covariance=[list(row) for row in matrix],
            observations=panel.observations,
            periods_per_year=periods_per_year,
            estimator=estimator,
            convention=panel.convention.value,
            intensity=intensity,
        )

    def require_path(self, method: str) -> None:
        """Refuse a method that needs the return path, naming what to send instead."""
        raise ToolExecutionError(
            f"the {method!r} method reads the return path, and a handle carries only "
            "the mean vector and the covariance matrix — the path is not recoverable "
            "from them. Send the `returns` matrix in this call instead of `handle`. "
            "(A handle cannot carry a returns matrix: a year of daily data on four "
            "assets encodes to roughly 6000 characters and five years on ten assets "
            "to about 69,000, against a handle limit of 8192.) The methods that do "
            "run from a handle are 'normal' and 'student-t'.",
            kind="handle_lacks_path",
            details=[
                {
                    "path": "/method",
                    "keyword": "handle_lacks_path",
                    "message": f"{method!r} needs `returns`, not `handle`",
                }
            ],
        )


def _panel(args: dict[str, Any]) -> Panel:
    """Build a panel from a returns matrix, checking the shape the schema cannot."""
    rows = args["returns"]
    width = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != width:
            raise DomainError(
                f"row 0 has {width} columns and row {index} has {len(row)}. Every row "
                "is one period across every asset, so the rows all have the same "
                "length; a ragged matrix usually means a missing observation that "
                "should be filled or dropped across all columns at once.",
                field="returns",
            )

    cells = len(rows) * width
    if cells > MAX_RETURN_CELLS:
        raise DomainError(
            f"{len(rows)} periods by {width} assets is {cells} values, over the "
            f"{MAX_RETURN_CELLS} limit. Shorten the window or reduce the assets.",
            field="returns",
        )
    if len(rows) < MIN_OBSERVATIONS:
        raise DomainError(
            f"{len(rows)} observations is too few to estimate from; at least "
            f"{MIN_OBSERVATIONS} are needed.",
            field="returns",
        )

    names = args.get("assets")
    if names is None:
        names = [f"asset_{index + 1}" for index in range(width)]
    elif len(names) != width:
        raise DomainError(
            f"{len(names)} asset names for {width} columns of returns. The names are "
            "positional: name i labels column i.",
            field="assets",
        )

    convention = Convention(args.get("convention", "simple"))
    columns = {
        name: [float(row[index]) for row in rows] for index, name in enumerate(names)
    }
    return Panel.from_columns(columns, convention=convention)


def _observations_per_parameter(assets: int, observations: int) -> float:
    return observations / (assets * (assets + 1) / 2)


def _annualise_return(value: float, periods_per_year: float) -> float:
    return value * periods_per_year


def _annualise_volatility(value: float, periods_per_year: float) -> float:
    return value * math.sqrt(periods_per_year)


def _weights(args: dict[str, Any], moments: Moments) -> list[float]:
    raw = args.get("weights")
    if raw is None:
        count = len(moments.assets)
        return [1.0 / count] * count

    weights = [float(value) for value in raw]
    if len(weights) != len(moments.assets):
        raise DomainError(
            f"{len(weights)} weights for {len(moments.assets)} assets "
            f"({', '.join(moments.assets)}). Weights are positional, one per column.",
            field="weights",
        )
    total = math.fsum(weights)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise DomainError(
            f"the weights sum to {total:.6g}, which is more than "
            f"{WEIGHT_SUM_TOLERANCE:g} away from one. They are not normalised here, "
            "because a sum of 0.98 is either a two percent cash position or a typo "
            "and scaling it quietly would turn the second into a plausible answer. "
            "Send weights that sum to one, or include the cash position as its own "
            "column of zero returns.",
            field="weights",
        )
    return weights


def _diversification(weights: list[float], matrix: list[list[float]]) -> dict[str, Any]:
    """The diversification ratio, or null and the reason it does not apply.

    The ratio is the weighted sum of individual volatilities over the portfolio
    volatility, and it is only a measure of diversification when every weight is
    positive. With a short position the numerator adds a volatility the portfolio
    subtracts, and the quotient stops meaning anything — the library refuses it, and
    it is right to.

    Reported as null rather than omitted, and never as a reason to refuse the whole
    call: the contributions are what the caller asked for and a short position does
    not make them ill-defined.
    """
    if any(weight < 0.0 for weight in weights):
        return {
            "diversificationRatio": None,
            "diversificationNote": (
                "Not defined for a portfolio with a short position: the ratio's "
                "numerator adds up individual volatilities, and a short leg's "
                "volatility is one the portfolio subtracts rather than adds. The risk "
                "contributions above are unaffected."
            ),
        }
    return {"diversificationRatio": diversification_ratio(weights, matrix)}


def _allocation_payload(allocation: Allocation) -> dict[str, Any]:
    """Render a risk allocation, per asset and in aggregate."""
    percentage = allocation.percentage
    return {
        "measure": allocation.measure,
        "total": allocation.total,
        "assets": [
            {
                "name": name,
                "weight": weight,
                "marginal": marginal,
                "component": component,
                "percentage": share,
            }
            for name, weight, marginal, component, share in zip(
                allocation.names,
                allocation.weights,
                allocation.marginal,
                allocation.component,
                percentage,
                strict=True,
            )
        ],
        # The Euler decomposition is exact: the components sum to the total. The
        # residual is reported rather than asserted because a covariance matrix
        # that had to be repaired to be positive semidefinite will not satisfy it
        # exactly, and that is worth seeing rather than hiding behind a tolerance.
        "identityError": allocation.identity_error,
        "hasNegativeContribution": allocation.has_negative_contribution,
        # Both of these read the risk shares as a probability distribution, which
        # a negative share is not. The library refuses to compute them in that
        # case, and it is right to: an entropy over a negative weight is not a
        # smaller number, it is not a number. Reporting null with the reason keeps
        # a hedged portfolio answerable — the contributions themselves are
        # perfectly well defined, and they are what the caller asked for.
        **(
            {
                "concentration": None,
                "effectiveBets": None,
                "concentrationNote": (
                    "Not defined for this portfolio: at least one position has a "
                    "negative risk contribution, and concentration and effective bets "
                    "read the shares as a distribution. A negative share means that "
                    "position removes risk, which is worth knowing and is not a "
                    "quantity an entropy is defined over. The contributions above are "
                    "unaffected."
                ),
            }
            if allocation.has_negative_contribution
            else {
                "concentration": concentration(allocation),
                "effectiveBets": effective_bets(allocation),
            }
        ),
    }


def _risk_payload(risk: Risk, *, periods_per_year: float) -> dict[str, Any]:
    return {
        "valueAtRisk": risk.value_at_risk,
        "expectedShortfall": risk.expected_shortfall,
        "quantile": risk.quantile,
        "confidence": risk.confidence,
        "tailProbability": risk.tail_probability,
        "mean": risk.mean,
        "volatility": risk.volatility,
        "annualisedVolatility": _annualise_volatility(risk.volatility, periods_per_year),
    }


def _drawdown_payload(episode: Drawdown) -> dict[str, Any]:
    """Render one drawdown episode.

    ``recoveryPeriods`` is null rather than zero when the drawdown never
    recovered, because zero would read as instant recovery — the most favourable
    possible reading of the least favourable possible outcome.

    ``recoveryReturn`` is the gain needed from the trough to get back to the peak,
    which is not the depth: a 50% fall needs a 100% gain. Reporting both is the
    point, because the asymmetry is the part people get wrong.
    """
    return {
        "depth": episode.depth,
        "peakPosition": episode.peak_position,
        "troughPosition": episode.trough_position,
        "recoveryPosition": episode.recovery_position,
        "peakLabel": episode.peak_label,
        "troughLabel": episode.trough_label,
        "recoveryLabel": episode.recovery_label,
        "peakValue": episode.peak_value,
        "troughValue": episode.trough_value,
        "declinePeriods": episode.decline_periods,
        "underwaterPeriods": episode.underwater_periods,
        "recoveryPeriods": episode.recovery_periods,
        "recoveryReturn": episode.recovery_return,
        "recovered": episode.recovered,
    }


class RiskTools:
    """The portfolio risk tools, sharing one handle minter."""

    def __init__(self, minter: Minter | None = None) -> None:
        self.minter = minter if minter is not None else Minter()

    # -- handles ----------------------------------------------------------

    def _mint(self, moments: Moments) -> str:
        try:
            return self.minter.mint(moments.to_payload())
        except HandleTooLarge as exc:
            raise DomainError(
                f"{len(moments.assets)} assets encode to a {exc.size}-character handle, "
                f"over the {exc.limit}-character limit. A handle carries the covariance "
                "matrix itself, which is one number per pair, so the size grows with the "
                "square of the asset count. Group the assets or drop the smallest "
                "positions.",
                field="returns",
            ) from exc

    def _moments(self, args: dict[str, Any]) -> Moments:
        """Resolve the moments from either a handle or an inline returns matrix."""
        handle = args.get("handle")
        rows = args.get("returns")
        if (handle is None) == (rows is None):
            raise DomainError(
                "supply exactly one of `returns` or `handle`: the returns matrix to "
                "estimate from now, or a handle from an earlier estimate_return_moments "
                "call to reuse. Sending both leaves it ambiguous which the answer came "
                "from."
            )

        if handle is not None:
            try:
                payload = self.minter.redeem(handle)
            except HandleError as exc:
                raise ToolExecutionError(
                    exc.message,
                    kind=exc.kind,
                    details=[{"path": "/handle", "keyword": exc.kind, "message": exc.message}],
                ) from exc
            return Moments.from_payload(payload)

        panel = _panel(args)
        periods = args.get("periodsPerYear")
        if periods is None:
            raise DomainError(
                "periodsPerYear is required when returns are supplied inline: it is "
                "what the annualised figures are computed with, and there is no safe "
                "default. 252 for daily trading days, 52 weekly, 12 monthly.",
                field="periodsPerYear",
            )
        return Moments.estimate(
            panel,
            periods_per_year=float(periods),
            estimator=str(args.get("estimator", "ledoit-wolf")).replace("-", "_"),
        )

    # -- tool bodies ------------------------------------------------------

    def moments_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        panel = _panel(args)
        periods_per_year = float(args["periodsPerYear"])
        estimator = str(args.get("estimator", "ledoit-wolf"))
        moments = Moments.estimate(
            panel,
            periods_per_year=periods_per_year,
            estimator="sample" if estimator == "sample" else "ledoit_wolf",
        )

        matrix = moments.covariance
        volatilities = [math.sqrt(matrix[i][i]) for i in range(len(matrix))]
        checks = diagnose(matrix, observations=moments.observations)

        payload: dict[str, Any] = {
            "handle": self._mint(moments),
            "expiresInSeconds": self.minter.ttl_s,
            "assets": moments.assets,
            "observations": moments.observations,
            "periodsPerYear": periods_per_year,
            "convention": moments.convention,
            "estimator": estimator,
            "perAsset": [
                {
                    "name": name,
                    "mean": mean,
                    "annualisedMean": _annualise_return(mean, periods_per_year),
                    "volatility": volatility,
                    "annualisedVolatility": _annualise_volatility(volatility, periods_per_year),
                }
                for name, mean, volatility in zip(
                    moments.assets, moments.mean, volatilities, strict=True
                )
            ],
            # The correlation matrix rather than the covariance. Both are the same
            # information, and a correlation is the one a reader can sanity-check
            # at a glance: every diagonal is 1 and every entry is in [-1, 1], so a
            # transposition or a scaling error is visible. The covariance itself
            # travels in the handle, where nobody has to read it.
            "correlation": [list(row) for row in correlation(matrix)],
            "diagnostics": {
                "assets": checks.assets,
                "conditionNumber": checks.condition_number,
                "smallestEigenvalue": checks.smallest_eigenvalue,
                "largestEigenvalue": checks.largest_eigenvalue,
                "positiveSemidefinite": checks.positive_semidefinite,
                "numericallySingular": checks.numerically_singular,
                "observationsPerParameter": checks.observations_per_parameter,
            },
            "note": (
                "The covariance matrix is carried in the handle rather than printed "
                "here; the correlation matrix and the per-asset volatilities are the "
                "same information in a form that can be read. Pass the handle to "
                "portfolio_tail_risk, portfolio_risk_contributions or "
                "risk_parity_weights to reuse this estimate without resending the "
                "returns. Those tools' historical and Cornish-Fisher methods read the "
                "return path, which a handle does not carry, and will ask for the "
                "matrix again."
            ),
        }

        if moments.intensity is not None:
            shrunk = ledoit_wolf(panel)
            payload["shrinkage"] = {
                "intensity": shrunk.intensity,
                "unclampedIntensity": shrunk.unclamped_intensity,
                "clamped": shrunk.clamped,
                "averageCorrelation": shrunk.average_correlation,
                "target": "constant-correlation",
                "note": (
                    "Intensity is how far the sample estimate was pulled towards the "
                    "target, from 0 (the sample, untouched) to 1 (the target alone). A "
                    "high intensity is not a fault; it is the estimator reporting that "
                    f"{moments.observations} observations on {len(moments.assets)} "
                    "assets carry little information about the pairwise structure. "
                    "The variances are left untouched either way."
                ),
            }
        return payload

    def tail_risk_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        method = str(args.get("method", "normal"))
        moments = self._moments(args)
        weights = _weights(args, moments)
        confidence = float(args.get("confidence", 0.99))
        periods_per_year = moments.periods_per_year

        if method in _PATH_METHODS and args.get("handle") is not None:
            moments.require_path(method)

        shared: dict[str, Any] = {
            "method": method,
            "assets": moments.assets,
            "weights": weights,
            "weightSum": math.fsum(weights),
            "observations": moments.observations,
            "periodsPerYear": periods_per_year,
            "convention": moments.convention,
            "covarianceEstimator": moments.estimator,
        }

        if method in ("normal", "student-t"):
            degrees = float(args.get("degrees", 5.0))
            risk = parametric_portfolio_risk(
                weights,
                moments.covariance,
                means=moments.mean,
                confidence=confidence,
                distribution=_DISTRIBUTIONS[method],
                degrees=degrees,
            )
            shared |= _risk_payload(risk, periods_per_year=periods_per_year)
            if method == "student-t":
                shared["degreesOfFreedom"] = degrees
                shared["note"] = (
                    "The t is scaled so its variance is the estimated one. A raw t has "
                    f"variance v/(v-2), which at {degrees:g} degrees of freedom is "
                    f"{degrees / (degrees - 2.0):.4g} rather than 1, so an unscaled one "
                    "would be wider than asked for by more than the heavy tail is "
                    "worth."
                )
            return shared

        # Everything below reads the path, so the returns matrix is present: the
        # handle case was refused above.
        panel = _panel(args)
        series = panel.portfolio(weights)

        if method == "cornish-fisher":
            skewness = series.skewness()
            excess = series.excess_kurtosis()
            if not is_monotone(skewness, excess):
                raise DomainError(
                    f"this portfolio's skewness ({skewness:.4f}) and excess kurtosis "
                    f"({excess:.4f}) put the Cornish-Fisher correction outside the "
                    "region where its corrected quantile increases with the "
                    "probability. Outside it the mapping is not a quantile function and "
                    "nothing read off it is a quantile of anything, so no number is "
                    "returned. Use 'student-t' for heavy tails, or 'historical' to take "
                    "the tail from the sample instead of correcting an assumed shape.",
                    field="method",
                )
            risk = parametric_portfolio_risk(
                weights,
                moments.covariance,
                means=moments.mean,
                confidence=confidence,
                distribution=Distribution.CORNISH_FISHER,
                skewness=skewness,
                excess_kurtosis=excess,
            )
            shared |= _risk_payload(risk, periods_per_year=periods_per_year)
            shared["skewness"] = skewness
            shared["excessKurtosis"] = excess
            shared["note"] = (
                "The volatility comes from the covariance estimate and the shape from "
                "the portfolio's own return series, whose skewness and excess kurtosis "
                "are reported above. Excess kurtosis: a normal series has zero, not "
                "three. Both are the bias-corrected sample estimators."
            )
            return shared

        quantile_method = str(args.get("quantileMethod", "linear"))
        if method == "historical":
            estimate = historical_risk(
                series, confidence=confidence, method=QuantileMethod(quantile_method)
            )
            shared |= {
                "valueAtRisk": estimate.value_at_risk,
                "expectedShortfall": estimate.expected_shortfall,
                "quantile": estimate.quantile,
                "confidence": estimate.confidence,
                "tailProbability": estimate.tail_probability,
                "mean": series.mean,
                "volatility": series.stdev(),
                "annualisedVolatility": _annualise_volatility(
                    series.stdev(), periods_per_year
                ),
                "tailObservations": estimate.tail_observations,
                "effectiveSample": estimate.effective_sample,
                "quantileMethod": quantile_method,
                "note": (
                    f"The effective sample is {estimate.effective_sample:.4g}, not "
                    f"{estimate.observations}: a {estimate.tail_probability:.2%} tail of "
                    f"{estimate.observations} observations is what the estimate is "
                    "actually reading, and the rest of the history only decides which "
                    "observations those are. Treat the figure as having that much "
                    "precision behind it."
                ),
            }
            return shared

        filtered = filtered_historical_risk(
            series,
            confidence=confidence,
            decay=float(args.get("decay", 0.94)),
            method=QuantileMethod(quantile_method),
        )
        shared |= {
            "valueAtRisk": filtered.risk.value_at_risk,
            "expectedShortfall": filtered.risk.expected_shortfall,
            "quantile": filtered.risk.quantile,
            "confidence": filtered.risk.confidence,
            "tailProbability": filtered.risk.tail_probability,
            "mean": series.mean,
            "volatility": filtered.current_volatility,
            "annualisedVolatility": _annualise_volatility(
                filtered.current_volatility, periods_per_year
            ),
            "averageVolatility": filtered.average_volatility,
            "volatilityScaling": filtered.scaling,
            "tailObservations": filtered.risk.tail_observations,
            "effectiveSample": filtered.risk.effective_sample,
            "quantileMethod": quantile_method,
            "decay": float(args.get("decay", 0.94)),
            "note": (
                "The historical tail is taken from returns standardised by an "
                f"exponentially weighted volatility and then rescaled to today's, which "
                f"is {filtered.scaling:.4g} times the window average. A scaling above "
                "one means the estimate is larger than the plain historical figure "
                "because the market is currently more volatile than the history it is "
                "drawn from, which is the point of the method."
            ),
        }
        return shared

    def contributions_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        moments = self._moments(args)
        weights = _weights(args, moments)
        measure = str(args.get("measure", "volatility"))
        confidence = float(args.get("confidence", 0.99))

        if measure == "volatility":
            allocation = volatility_contributions(
                weights, moments.covariance, names=moments.assets
            )
        else:
            allocation = risk_contributions(
                weights,
                moments.covariance,
                means=moments.mean,
                confidence=confidence,
                distribution=Distribution.NORMAL,
                of_expected_shortfall=measure == "expectedShortfall",
                names=moments.assets,
            )

        payload = _allocation_payload(allocation)
        payload |= {
            "assetsCount": len(moments.assets),
            "weightSum": math.fsum(weights),
            "observations": moments.observations,
            "covarianceEstimator": moments.estimator,
            **_diversification(weights, moments.covariance),
            "note": (
                "Components are Euler contributions and sum to the total exactly, so a "
                "share is a share of something rather than a normalised guess. A "
                "negative component belongs to a position that reduces portfolio risk; "
                "its percentage is negative too, and the positive shares therefore sum "
                "past one. Effective bets, when it is defined, is the reciprocal of "
                "the concentration: it says how many independent positions the "
                f"portfolio carries risk like, against the {len(moments.assets)} it "
                "holds."
            ),
        }
        if measure != "volatility":
            payload["confidence"] = confidence
            payload["distribution"] = "normal"
        return payload

    def parity_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        moments = self._moments(args)
        budgets = args.get("budgets")
        if budgets is not None and len(budgets) != len(moments.assets):
            raise DomainError(
                f"{len(budgets)} budgets for {len(moments.assets)} assets. Budgets are "
                "positional and are normalised to sum to one.",
                field="budgets",
            )
        solution = risk_parity(
            moments.covariance,
            budgets=None if budgets is None else [float(value) for value in budgets],
            names=moments.assets,
        )
        payload = _allocation_payload(solution.allocation)
        payload |= {
            "weights": list(solution.weights),
            "budgets": list(solution.budgets),
            "converged": solution.converged,
            "sweeps": solution.sweeps,
            "finalChange": solution.final_change,
            "budgetError": solution.budget_error,
            "equalRisk": solution.equal_risk,
            "observations": moments.observations,
            "covarianceEstimator": moments.estimator,
            **_diversification(list(solution.weights), moments.covariance),
            "note": (
                "The weights are long-only and sum to one. budgetError is the largest "
                "gap between an asset's achieved risk share and its target, reported as "
                "evidence rather than promised: a covariance matrix that had to be "
                "repaired to be positive semidefinite may not admit an exact solution. "
                f"This one converged in {solution.sweeps} sweeps."
                if solution.converged
                else "The solver did not converge; the weights are its last iterate and "
                "budgetError says how far off the risk shares are. That usually means a "
                "near-singular covariance matrix — check the diagnostics from "
                "estimate_return_moments."
            ),
        }
        return payload

    def drawdown_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        panel = _panel(args)
        weights = args.get("weights")
        if weights is None:
            weights = [1.0 / len(panel.names)] * len(panel.names)
        elif len(weights) != len(panel.names):
            raise DomainError(
                f"{len(weights)} weights for {len(panel.names)} assets "
                f"({', '.join(panel.names)}). Weights are positional.",
                field="weights",
            )
        else:
            total = math.fsum(float(value) for value in weights)
            if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
                raise DomainError(
                    f"the weights sum to {total:.6g}, which is more than "
                    f"{WEIGHT_SUM_TOLERANCE:g} away from one.",
                    field="weights",
                )
        weights = [float(value) for value in weights]

        labels = args.get("index")
        if labels is not None and len(labels) != panel.observations + 1:
            raise DomainError(
                f"{len(labels)} index labels for {panel.observations} periods of "
                f"returns, which need {panel.observations + 1}. The labels name points "
                "on the wealth curve, not rows of `returns`: n returns move a portfolio "
                "between n + 1 valuations, and a drawdown runs from one valuation to "
                "another. The first label is the starting value, before any return has "
                "been applied.",
                field="index",
            )

        periods_per_year = float(args["periodsPerYear"])
        series = panel.portfolio(weights)
        worst = maximum_drawdown(series, index=labels)
        longest = longest_underwater(series, index=labels)
        minimum_depth = float(args.get("minimumDepth", 0.0))
        top = int(args.get("worstCount", 5))
        episodes = drawdowns(series, index=labels, minimum_depth=minimum_depth)
        episodes.sort(key=lambda episode: episode.depth, reverse=True)

        return {
            "assets": list(panel.names),
            "weights": weights,
            "weightSum": math.fsum(weights),
            "observations": panel.observations,
            "periodsPerYear": periods_per_year,
            "convention": panel.convention.value,
            "cumulativeReturn": series.cumulative(),
            "annualisedReturn": series.annualised_return(periods_per_year),
            "annualisedVolatility": series.annualised_volatility(periods_per_year),
            "maximumDrawdown": _drawdown_payload(worst),
            "longestUnderwater": _drawdown_payload(longest),
            "ulcerIndex": ulcer_index(series),
            "calmar": calmar_ratio(series, periods_per_year),
            "sortino": sortino_ratio(series, periods_per_year, full_sample=True),
            "downsideDeviation": downside_deviation_of(series, full_sample=True),
            "episodeCount": len(episodes),
            "worstEpisodes": [_drawdown_payload(episode) for episode in episodes[:top]],
            "minimumDepth": minimum_depth,
            "note": (
                "Depths are fractions of the peak: 0.2 is a twenty percent fall. "
                "recoveryPeriods is null for a drawdown that never recovered, which is "
                "not the same as zero. Positions index the wealth curve rather than the "
                "returns, so 0 is the starting valuation and a peak at position 1 is the "
                "value after the first return; there are n + 1 of them for n returns. "
                "The ulcer index reads the whole underwater path "
                "rather than its two extreme points, so two portfolios with the same "
                "maximum drawdown separate on it. Sortino's denominator is the "
                "full-sample downside deviation, which divides by every observation "
                "rather than only the ones below target — stated because the other "
                "convention is also in use and gives a different number."
            ),
        }

    def conditional_volatility_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        observed = [float(value) for value in args["returns"]]
        if len(observed) < MIN_GARCH_OBSERVATIONS:
            raise DomainError(
                f"{len(observed)} observations, below the {MIN_GARCH_OBSERVATIONS} a "
                "GARCH fit needs. Below that the likelihood is nearly flat along the "
                "persistence direction and the optimiser reports whatever it started "
                "near, which would look like a fit and not be one.",
                field="returns",
            )
        horizon = int(args.get("horizon", 10))
        targeting = bool(args.get("varianceTargeting", False))
        confidence = float(args.get("confidence", 0.99))
        choice = str(args.get("innovation", "auto"))
        verdict = None
        try:
            if choice == "auto":
                # Tested rather than assumed in either direction. Always fitting
                # the heavier tail costs a thin-tailed series a slightly wider
                # quantile for no reason the caller asked for; always assuming
                # the normal is the error this parameter exists to remove.
                verdict = fat_tail_test(observed, variance_targeting=targeting)
                innovation = Innovation.STUDENT_T if verdict.fat else Innovation.NORMAL
            else:
                innovation = Innovation(choice)
            fitted = fit_garch(
                observed,
                variance_targeting=targeting,
                strict=False,
                innovation=innovation,
            )
        except ValueError as bad:
            raise DomainError(str(bad), field="returns") from bad
        conditional = fitted.risk(confidence=confidence, last_return=observed[-1])
        # The multiplier the forecast series is scaled by to become a value at
        # risk. Handing it back is the point: under Student-t innovations it is
        # the *standardised* quantile — the raw one times sqrt((v - 2) / v) — and
        # a caller who reaches for the raw quantile widens every forecast by 41%
        # at four degrees of freedom.
        #
        # Backed out of the fitted risk rather than recomputed, so the two cannot
        # drift apart, and with the mean removed: `quantile` is
        # `mean + sigma * z`, so `(mean - quantile) / sigma` is exactly `-z` and
        # nothing else. Leaving the mean in would give a number that is not a
        # quantile of anything, and multiplying a whole forecast series by it
        # would scale a constant drift by each period's volatility.
        next_volatility = math.sqrt(fitted.next_variance(observed[-1]))
        multiplier = (fitted.mean - conditional.quantile) / next_volatility
        kurtosis = fitted.implied_excess_kurtosis

        include = bool(args.get("includeForecasts", True))
        if include and len(observed) > MAX_FORECAST_SERIES:
            raise DomainError(
                f"{len(observed)} observations would return that many forecasts, above "
                f"the limit of {MAX_FORECAST_SERIES}. Pass includeForecasts false for "
                "the parameters alone, or fit a shorter window.",
                field="returns",
            )

        payload: dict[str, Any] = {
            "observations": fitted.observations,
            "omega": fitted.omega,
            "alpha": fitted.alpha,
            "beta": fitted.beta,
            "persistence": fitted.persistence,
            "halfLife": fitted.half_life,
            "longRunVolatility": fitted.long_run_volatility,
            "currentVolatility": fitted.volatilities[-1],
            "nextVolatility": next_volatility,
            "logLikelihood": fitted.log_likelihood,
            "iterations": fitted.iterations,
            "converged": fitted.converged,
            "varianceTargeted": fitted.variance_targeted,
            "horizon": horizon,
            "horizonVolatility": math.sqrt(
                fitted.horizon_variance(horizon, last_return=observed[-1])
            ),
            "squareRootOfTimeRatio": fitted.scaling_against_square_root_of_time(
                horizon, last_return=observed[-1]
            ),
            "innovation": fitted.innovation.value,
            # Withheld rather than reported when the likelihood could not pin it
            # down. Above 200 the standardised-t density is within a percent of
            # the normal everywhere that matters, so the number the optimiser
            # stopped at is noise and "our fitted tail index is 640" is a claim
            # about a series that is simply Gaussian.
            "degreesOfFreedom": fitted.degrees_of_freedom if fitted.degrees_identified else None,
            # None below four degrees of freedom, where 6 / (v - 4) is genuinely
            # infinite. `json.dumps` writes bare Infinity for it, which is not
            # JSON, and a strict parser rejects the whole message rather than the
            # field.
            "impliedExcessKurtosis": (
                kurtosis if fitted.degrees_identified and math.isfinite(kurtosis) else None
            ),
            "confidence": confidence,
            "valueAtRisk": conditional.value_at_risk,
            "expectedShortfall": conditional.expected_shortfall,
            "quantileMultiplier": multiplier,
        }
        if verdict is not None:
            payload["fatTail"] = {
                "statistic": verdict.statistic,
                "pValue": verdict.p_value,
                "degreesOfFreedom": verdict.degrees_of_freedom if verdict.identified else None,
                "identified": verdict.identified,
                "fat": verdict.fat,
            }
        if include:
            payload["volatilityForecasts"] = list(fitted.volatilities)
        payload["note"] = (
            "`volatilityForecasts` is one number per observation, ALREADY ALIGNED: "
            "element i is the forecast made from returns strictly before return i. "
            "Multiply it by the quantile of whatever distribution you are assuming "
            "and pass it to validate_risk_model as `valueAtRisk` beside the same "
            "`returns`, with no offsetting. That alignment is the easiest thing in "
            "this workflow to get wrong, so it is done here rather than "
            "documented.\n\n"
            "`persistence` is alpha + beta and must be below one for a long-run "
            "variance to exist. `halfLife` is how many periods half of a shock "
            "survives; at a persistence of 0.99 that is 69 periods, so a market "
            "disturbed today is still half disturbed three months later.\n\n"
            "`squareRootOfTimeRatio` is the horizon volatility over what scaling "
            "today's volatility by the square root of the horizon would give. It is "
            "BELOW one after a shock and ABOVE one in a calm market, and it is not a "
            "small correction: from four times the long-run variance at a "
            "persistence of 0.975, a one-year horizon is 39% below the scaled "
            "figure. The calm case is the expensive one, because understating risk "
            "arrives while positions are going on rather than coming off.\n\n"
            "Check `converged` before using the parameters. A false there means the "
            "optimiser ran out of iterations and the numbers are the best point it "
            "reached, not a fit; try varianceTargeting, which removes the "
            "worst-determined parameter from the search.\n\n"
            "Scale `volatilityForecasts` by `quantileMultiplier`, not by a normal "
            "quantile. Under Student-t innovations the right multiplier is the "
            "STANDARDISED quantile — the raw t quantile times sqrt((v-2)/v) — and "
            "using the raw one widens every forecast by 41% at four degrees of "
            "freedom, which undershoots the breach count and looks conservative "
            "rather than wrong. The multiplier is the innovation quantile alone "
            "and carries no mean; subtract `mean` from the product if the drift "
            "matters, which on a daily series it does not at any usual "
            "confidence.\n\n"
            "`innovation` defaults to testing for a fat tail by likelihood ratio "
            "and using one only if the data shows one; `fatTail` carries that "
            "verdict. Its p-value is conservative by about a factor of two, "
            "because the null sits on the boundary of the parameter space — "
            "measured over 200 Gaussian samples, a nominal 5% test rejected 5 "
            "times. Read a rejection as meaning what it says and a near miss as "
            "weaker evidence against a fat tail than it looks.\n\n"
            "What estimating the tail buys, measured on regime-switching series: "
            "the 99% breach count over 2000 observations falls from 28.2 to 22.45 "
            "against a nominal 20, so about 70% of the excess a Gaussian GARCH "
            "leaves. Not all of it — a single tail index for a series whose "
            "volatility jumps between regimes is closer than the normal's and "
            "still an approximation. On a series that never had a fat tail the two "
            "agree to within half a breach in twenty. Validate rather than assume."
        )
        return payload

    def validation_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        observed = [float(value) for value in args["returns"]]
        forecasts = [float(value) for value in args["valueAtRisk"]]
        confidence = float(args["confidence"])
        if len(observed) != len(forecasts):
            raise DomainError(
                f"{len(observed)} returns against {len(forecasts)} forecasts. A "
                "forecast series is one-step-ahead, so there is one forecast per "
                "return and neither is offset — the first forecast is the one that "
                "was made for the first return.",
                field="valueAtRisk",
            )
        if len(observed) > MAX_FORECASTS:
            raise DomainError(
                f"{len(observed)} observations, above the limit of {MAX_FORECASTS}.",
                field="returns",
            )
        shortfalls = args.get("expectedShortfall")
        if shortfalls is not None:
            shortfalls = [float(value) for value in shortfalls]
            if len(shortfalls) != len(observed):
                raise DomainError(
                    f"{len(shortfalls)} expected-shortfall forecasts against "
                    f"{len(observed)} returns.",
                    field="expectedShortfall",
                )
        replications = int(args.get("replications", 0))
        if replications > MAX_REPLICATIONS:
            raise DomainError(
                f"{replications} replications, above the limit of {MAX_REPLICATIONS}. "
                "The p-value carries a simulation error around "
                "sqrt(p(1-p)/replications), so past this the extra digit is noise.",
                field="replications",
            )
        if replications and shortfalls is None:
            raise DomainError(
                "replications were asked for without `expectedShortfall`. The "
                "simulation is of the expected-shortfall null; the coverage tests "
                "have closed-form p-values and nothing to simulate.",
                field="replications",
            )
        distribution = _DISTRIBUTIONS[args.get("distribution", "normal")]
        if distribution is Distribution.CORNISH_FISHER:
            raise DomainError(
                "the simulated null draws from a distribution, and the "
                "Cornish-Fisher correction is a quantile mapping rather than one. "
                "Use normal or student-t.",
                field="distribution",
            )

        try:
            result = validate_risk_model(
                observed,
                forecasts,
                confidence=confidence,
                expected_shortfall=shortfalls,
                distribution=distribution,
                degrees=float(args.get("degrees", 5.0)),
                replications=replications,
                seed=int(args.get("seed", 0)),
            )
        except ValueError as bad:
            raise DomainError(str(bad), field="valueAtRisk") from bad

        breaches = result.exceedances
        light = result.traffic_light
        payload: dict[str, Any] = {
            "observations": breaches.observations,
            "confidence": confidence,
            "tailProbability": 1.0 - confidence,
            "breaches": breaches.count,
            "expectedBreaches": breaches.expected,
            "breachRate": breaches.rate,
            "tests": [
                {
                    "name": test.name,
                    "statistic": test.statistic,
                    "degreesOfFreedom": test.degrees_of_freedom,
                    "pValue": test.p_value,
                    "rejectsAt5Percent": test.rejects_at(0.05),
                    "rejectsAt1Percent": test.rejects_at(0.01),
                    "interpretation": test.interpretation,
                    # True when there are too few observations for the
                    # chi-square limit these statistics are scored against.
                    "advisory": test.advisory,
                }
                for test in (result.unconditional, result.independence, result.conditional)
            ],
            "trafficLight": {
                "zone": light.zone.value,
                "cumulativeProbability": light.cumulative_probability,
                "plusFactor": light.plus_factor,
                "supervisorySetup": light.plus_factor is not None,
            },
            "rejectedAt5Percent": list(result.rejected_at(0.05)),
            "warnings": list(result.warnings),
        }
        if result.expected_shortfall is not None:
            found = result.expected_shortfall
            payload["expectedShortfallTest"] = {
                "conditional": found.conditional,
                "unconditional": found.unconditional,
                "realisedOverForecast": found.realised_ratio,
                "breaches": found.breaches,
                "conditionalPValue": found.conditional_p_value,
                "unconditionalPValue": found.unconditional_p_value,
                "replications": found.replications,
                "direction": found.direction,
            }
        payload["note"] = (
            "Forecasts are POSITIVE losses, the same sign convention every risk "
            "tool here returns, and returns are signed — so a breach is "
            "`return < -forecast`. A forecast series of negative numbers is "
            "refused rather than scored.\n\n"
            "The two coverage tests fail for different reasons and need different "
            "fixes. Unconditional coverage is about the NUMBER of breaches and a "
            "rejection means the model is scaled wrong. Independence is about "
            "whether they CLUSTER: a constant-volatility model can produce exactly "
            "the right number of breaches over a year and put them all in one "
            "fortnight, which a count cannot see and which no rescaling fixes — "
            "that one needs a volatility process. Conditional coverage is the two "
            "together and does not say which half failed, so read its parts.\n\n"
            "The traffic-light zone is derived from the binomial, so it adapts to "
            "the sample length: three breaches is green over 250 observations and "
            "yellow over 125. plusFactor is the supervisory capital add-on and is "
            "null outside the 250-observation, 99% setup it is published for, "
            "because those values are tabulated rather than computed and "
            "extrapolating them would invent a number.\n\n"
            "Expected shortfall is not elicitable, so there is no breach-count "
            "equivalent for it. The two Acerbi-Szekely statistics are zero under a "
            "correct model and NEGATIVE when the tail is understated. They are not "
            "on the same scale as the error they detect: test 1 reads about -0.245 "
            "when the true volatility is double the forecast, because it is a ratio "
            "of tail means. realisedOverForecast is the same information in units "
            "that can be read directly. An OVERSTATED tail is nearly untestable — "
            "a forecast twice too wide produces no breaches at all — so a small "
            "positive reading is not evidence of caution.\n\n"
            "`advisory` on a test means too few observations for the chi-square "
            "limit it is scored against; the statistic is still reported and should "
            "be read as descriptive."
        )
        return payload

    # -- registration -----------------------------------------------------

    def register(self, registry: ToolRegistry) -> ToolRegistry:
        read_only = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}
        source = {
            "returns": _RETURNS,
            "assets": _ASSETS,
            "convention": _CONVENTION,
            "handle": _HANDLE,
            "estimator": {
                "type": "string",
                "enum": ["sample", "ledoit-wolf"],
                "description": (
                    "Covariance estimator, used only when returns are supplied inline. "
                    "Defaults to ledoit-wolf, whose shrinkage intensity is estimated "
                    "rather than tuned. Ignored when a handle is passed, since the "
                    "estimate it carries was already made."
                ),
            },
            "periodsPerYear": _PERIODS_PER_YEAR,
        }

        registry.register(
            "estimate_return_moments",
            title="Estimate a covariance matrix from returns",
            description=(
                "Estimate the mean vector and covariance matrix of a returns matrix, "
                "report the per-asset volatilities, the correlation matrix and the "
                "conditioning diagnostics, and return a handle so the estimate can be "
                "reused without resending the returns. Shrinkage is the default and the "
                "intensity it chose is reported, because an intensity near one is the "
                "estimator saying the sample carries little information about the "
                "pairwise structure — which is worth knowing before acting on the "
                "matrix. Returns are decimal fractions: a 1% gain is 0.01."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "returns": _RETURNS,
                    "assets": _ASSETS,
                    "convention": _CONVENTION,
                    "estimator": source["estimator"],
                    "periodsPerYear": _PERIODS_PER_YEAR,
                },
                "required": ["returns", "periodsPerYear"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.moments_payload))

        registry.register(
            "portfolio_tail_risk",
            title="Value at risk and expected shortfall",
            description=(
                "Value at risk and expected shortfall for a weighted portfolio, by one "
                "of five methods, with the method named in the result because the number "
                "does not identify it. 'normal' and 'student-t' need only the first two "
                "moments and run from a handle. 'cornish-fisher' corrects the normal "
                "quantile using the portfolio's own skewness and excess kurtosis, and is "
                "refused rather than approximated when those moments put the correction "
                "outside the region where it is a quantile function at all. "
                "'historical' takes the tail from the sample; 'filtered-historical' does "
                "the same after standardising by an exponentially weighted volatility "
                "and rescaling to today's. The last three read the return path, so they "
                "need the returns matrix rather than a handle. Losses are positive: a "
                "value at risk of 0.023 is a 2.3% loss."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **source,
                    "weights": _WEIGHTS,
                    "confidence": _CONFIDENCE,
                    "method": {
                        "type": "string",
                        "enum": list(_METHODS),
                        "description": (
                            "Which estimator. Defaults to 'normal', which understates the "
                            "tail of essentially every real return series — it is the "
                            "default because it is the reference, not because it is the "
                            "right answer."
                        ),
                    },
                    "degrees": {
                        "type": "number",
                        "exclusiveMinimum": 2,
                        "maximum": 200,
                        "description": (
                            "Degrees of freedom for 'student-t'. Above two, because the "
                            "variance of a t is v/(v-2) and does not exist at or below "
                            "it. Defaults to 5. The distribution is scaled so its "
                            "variance is the estimated one."
                        ),
                    },
                    "quantileMethod": {
                        "type": "string",
                        "enum": ["lower", "higher", "linear", "weibull"],
                        "description": (
                            "How a sample quantile is interpolated, for the historical "
                            "methods. Defaults to linear. It matters far out in a tail, "
                            "where the neighbouring order statistics are far apart."
                        ),
                    },
                    "decay": {
                        "type": "number",
                        "exclusiveMinimum": 0.5,
                        "exclusiveMaximum": 1,
                        "description": (
                            "Exponential weight on yesterday's variance, for "
                            "'filtered-historical'. Defaults to 0.94, the RiskMetrics "
                            "figure for daily data."
                        ),
                    },
                },
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.tail_risk_payload))

        registry.register(
            "portfolio_risk_contributions",
            title="Where a portfolio's risk comes from",
            description=(
                "Decompose portfolio risk across its positions: each asset's marginal "
                "risk, its Euler component, and its share. The components sum to the "
                "total exactly, so a share is a share of something rather than a "
                "normalised guess, and a position that hedges shows a negative "
                "contribution rather than a small positive one. Also reports "
                "concentration, effective bets — how many independent positions the "
                "portfolio carries risk like — and the diversification ratio. Works from "
                "a handle or from a returns matrix."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **source,
                    "weights": _WEIGHTS,
                    "confidence": _CONFIDENCE,
                    "measure": {
                        "type": "string",
                        "enum": ["volatility", "valueAtRisk", "expectedShortfall"],
                        "description": (
                            "Which risk measure to decompose. Defaults to volatility, "
                            "which needs no distributional assumption. The other two are "
                            "decomposed under a normal assumption and report it."
                        ),
                    },
                },
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.contributions_payload))

        registry.register(
            "risk_parity_weights",
            title="Weights that equalise risk contributions",
            description=(
                "Solve for long-only weights whose risk contributions match a budget, "
                "equal by default. Reports whether the solver converged, how far the "
                "achieved risk shares are from the targets, and the resulting "
                "contribution table — as evidence rather than as a promise, because a "
                "near-singular covariance matrix may not admit an exact solution. Works "
                "from a handle or from a returns matrix."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **source,
                    "budgets": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_ASSETS,
                        "items": {"type": "number", "exclusiveMinimum": 0},
                        "description": (
                            "Target risk share per asset, positional, normalised to sum "
                            "to one. Defaults to equal shares. Strictly positive: an "
                            "asset with a zero risk budget is one to leave out."
                        ),
                    },
                },
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.parity_payload))

        registry.register(
            "portfolio_drawdown",
            title="Drawdown and path statistics",
            description=(
                "What the portfolio's path did, which no covariance can see: the deepest "
                "drawdown with the periods it ran between, the longest time underwater, "
                "the ulcer index over the whole underwater path rather than its two "
                "extreme points, and the Calmar and Sortino ratios with their "
                "denominators stated. Depths are fractions of the peak and the recovery "
                "gain is reported alongside — a 50% fall needs a 100% gain, and that "
                "asymmetry is the part that gets misread. Needs the returns matrix: a "
                "handle carries second moments, and the path is not recoverable from "
                "them."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "returns": _RETURNS,
                    "assets": _ASSETS,
                    "convention": _CONVENTION,
                    "weights": _WEIGHTS,
                    "periodsPerYear": _PERIODS_PER_YEAR,
                    "index": {
                        "type": "array",
                        "minItems": MIN_OBSERVATIONS + 1,
                        "items": {"type": "string", "minLength": 1, "maxLength": 32},
                        "description": (
                            "Labels for the points on the wealth curve — dates, usually. "
                            "One MORE than there are rows of `returns`: n returns move a "
                            "portfolio between n + 1 valuations, and the first label is "
                            "the starting value before any return is applied. Used only "
                            "to say when a drawdown began, bottomed and recovered. "
                            "Without them the report uses curve positions, on the same "
                            "0-to-n numbering."
                        ),
                    },
                    "minimumDepth": {
                        "type": "number",
                        "minimum": 0,
                        "exclusiveMaximum": 1,
                        "description": (
                            "Ignore drawdowns shallower than this when counting and "
                            "listing episodes, as a fraction: 0.05 keeps falls of five "
                            "percent or more. Defaults to 0, which counts every dip."
                        ),
                    },
                    "worstCount": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "How many of the deepest episodes to list. Defaults to 5.",
                    },
                },
                "required": ["returns", "periodsPerYear"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.drawdown_payload))

        registry.register(
            "conditional_volatility",
            title="Fit a volatility process and forecast from it",
            description=(
                "A GARCH(1,1) fitted by maximum likelihood over a return series, "
                "which is what to reach for when validate_risk_model rejects on "
                "independence: clustered breaches mean the model has no notion of "
                "volatility changing, and no rescaling fixes that. Returns the "
                "parameters, the persistence, the half-life of a shock, and a "
                "one-step-ahead volatility for every period — already aligned, so it "
                "goes straight to validate_risk_model as a forecast series without "
                "the caller offsetting anything. Also reports the horizon volatility "
                "against what square-root-of-time would give, which differs by tens "
                "of percent over a year and in both directions. The innovation tail "
                "is estimated rather than assumed normal, and the multiplier to turn "
                "the forecast series into a value at risk comes back with it — the "
                "standardised quantile, which is not the raw one."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "returns": {
                        "type": "array",
                        "minItems": MIN_GARCH_OBSERVATIONS,
                        "maxItems": MAX_FORECASTS,
                        "items": {"type": "number", "exclusiveMinimum": -1, "maximum": 10},
                        "description": (
                            "One return per period, in order, oldest first. At least "
                            f"{MIN_GARCH_OBSERVATIONS}: below that the likelihood is "
                            "nearly flat along the persistence direction and the fit "
                            "reports whatever it started near."
                        ),
                    },
                    "horizon": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 2_000,
                        "description": (
                            "Periods to aggregate the variance over, for the horizon "
                            "figure and its comparison with square-root-of-time. "
                            "Defaults to 10."
                        ),
                    },
                    "varianceTargeting": {
                        "type": "boolean",
                        "description": (
                            "Fix the long-run variance to the sample variance and "
                            "estimate only the two dynamic parameters. More robust on "
                            "a short sample, because omega is the product of the "
                            "long-run level and one minus the persistence, and on a "
                            "persistent series that second factor is small and badly "
                            "determined. Defaults to false."
                        ),
                    },
                    "includeForecasts": {
                        "type": "boolean",
                        "description": (
                            "Return the per-period volatility series. Defaults to "
                            "true. Set false for the parameters alone, which is what "
                            "a long history needs since the series is one number per "
                            "observation."
                        ),
                    },
                    "innovation": {
                        "type": "string",
                        "enum": ["auto", "normal", "student-t"],
                        "description": (
                            "Shape assumed for the standardised residuals. 'auto' "
                            "tests for a fat tail by likelihood ratio and fits one "
                            "only if the data shows one; that verdict comes back in "
                            "`fatTail`. Defaults to auto. The variance process says "
                            "how the scale moves and says nothing about the shape "
                            "drawn at that scale, which is why this is a separate "
                            "choice."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "exclusiveMinimum": 0.5,
                        "exclusiveMaximum": 1,
                        "description": (
                            "Confidence for the conditional value at risk, expected "
                            "shortfall and quantile multiplier. Defaults to 0.99."
                        ),
                    },
                },
                "required": ["returns"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.conditional_volatility_payload))

        registry.register(
            "validate_risk_model",
            title="Score a value-at-risk forecast against what happened",
            description=(
                "Whether a risk model worked. Takes a series of realised returns and "
                "the one-step-ahead forecasts that were made for them, and runs "
                "Kupiec's unconditional coverage test on the breach count, "
                "Christoffersen's test on whether the breaches cluster, the joint "
                "conditional coverage test, and the supervisory traffic-light zone. "
                "The clustering test is the one a count cannot replace: a "
                "constant-volatility model can breach exactly the right number of "
                "times over a year and put every breach in the same fortnight. Supply "
                "expected-shortfall forecasts as well and it adds the Acerbi-Szekely "
                "statistics, which are the only way to score a tail mean because "
                "expected shortfall is not elicitable and has no breach-count "
                "equivalent. Forecasts are positive losses, the sign convention every "
                "other risk tool here returns."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "returns": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": MAX_FORECASTS,
                        "items": {"type": "number", "exclusiveMinimum": -1, "maximum": 10},
                        "description": (
                            "Realised returns, signed, one per period in order. A loss "
                            "is negative."
                        ),
                    },
                    "valueAtRisk": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": MAX_FORECASTS,
                        "items": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
                        "description": (
                            "The forecast made FOR each return, as a POSITIVE loss: "
                            "0.023 is a forecast 2.3% loss. One per return and not "
                            "offset — element i is the forecast that was made before "
                            "return i was observed. Getting this alignment wrong is the "
                            "easiest way to make a broken model look fine."
                        ),
                    },
                    "expectedShortfall": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": MAX_FORECASTS,
                        "items": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
                        "description": (
                            "Forecast tail means, positive losses, aligned the same "
                            "way. Each must be at least its own value at risk, since a "
                            "tail mean averages losses no smaller than the quantile. "
                            "Omit to skip the expected-shortfall statistics."
                        ),
                    },
                    "confidence": _CONFIDENCE,
                    "replications": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": MAX_REPLICATIONS,
                        "description": (
                            "Simulate the expected-shortfall null this many times to "
                            "attach p-values to those two statistics. Needs "
                            "`expectedShortfall`. Zero, the default, reports the "
                            "statistics without p-values. The coverage tests have "
                            "closed-form p-values and are unaffected."
                        ),
                    },
                    "distribution": {
                        "type": "string",
                        "enum": ["normal", "student-t"],
                        "description": (
                            "The predictive distribution the forecasts were built "
                            "under, which is what the simulated null draws from. Only "
                            "used when `replications` is above zero. Defaults to "
                            "normal."
                        ),
                    },
                    "degrees": {
                        "type": "number",
                        "exclusiveMinimum": 2,
                        "maximum": 200,
                        "description": (
                            "Degrees of freedom for a student-t null, standardised to "
                            "unit variance so it is comparable with the forecasts. "
                            "Above two, because the variance to standardise by is "
                            "infinite at or below it. Defaults to 5."
                        ),
                    },
                    "seed": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2**31 - 1,
                        "description": (
                            "Seed for the simulation, so a reported p-value can be "
                            "reproduced. Defaults to 0."
                        ),
                    },
                },
                "required": ["returns", "valueAtRisk", "confidence"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.validation_payload))

        return registry


def register(registry: ToolRegistry, *, minter: Minter | None = None) -> ToolRegistry:
    """Add the portfolio risk tools to ``registry``."""
    return RiskTools(minter).register(registry)


__all__ = [
    "MAX_ASSETS",
    "MAX_CONFIDENCE",
    "MAX_RETURN_CELLS",
    "MIN_OBSERVATIONS",
    "MOMENTS_KIND",
    "WEIGHT_SUM_TOLERANCE",
    "Moments",
    "RiskTools",
    "register",
]
