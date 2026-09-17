"""Implied volatility, slice calibration and the volatility surface.

Everything here works in *total implied variance* against *log-moneyness on the
forward*::

    k = log(K / F)        w(k) = sigma(k)^2 T

Note the direction. ``moneyness.log_moneyness`` computes ``log(S / K)``, which
is the negative of this; using it here would silently mirror every smile and
turn a skew into its opposite. The convention is stated in the tool
descriptions for the same reason it is stated here.

**How a surface is returned, and why it is not a grid of numbers.**

The obvious representation of a surface is a dense matrix of volatilities. It is
also the one a language model reads worst: an unlabelled nested array gives no
clue which axis is maturity and which is strike, and transposing it produces
something that still looks plausible. Reading it wrong is easy and silent.

So a surface comes back three ways at once, in descending order of how much
trust each deserves:

1. **The parameters.** Five numbers per slice, exact, sufficient to rebuild the
   slice, and small enough to carry through a conversation without truncation.
2. **Named scalars.** The at-the-money volatility, the wing slopes, the fit
   residuals — the handful of quantities a reader actually reasons about.
3. **A labelled grid, only when asked for.** Every row carries its maturity and
   every column its strike as an explicit key, so a row cannot be read as a
   column.

Arbitrage is reported as flags with the offending location attached, rather than
left to be inferred from the numbers. Whether a surface admits arbitrage is a
yes-or-no question, and answering it in the output is more useful than shipping
a grid that a careful reader could in principle check for themselves.
"""

from __future__ import annotations

import math
from typing import Any

from moneyness import (
    SVI,
    Method,
    OptionType,
    Quote,
    Surface,
    bounds,
    calibrate,
    solve,
)

from .analytics import _CARRY, _RATE, _SPOT, _STRIKE, _TIME, _TYPE, build_quote, guard
from .tools import DomainError, ToolRegistry

#: Half-width in log-moneyness for the arbitrage scans. Roughly five standard
#: deviations at a typical one-year volatility, which is wider than any strike
#: that trades and wide enough for a wing violation to show up.
SCAN_WING = 5.0


def forward_of(spot: float, time: float, carry: float | None, rate: float) -> float:
    """``S e^{bT}``. Carry defaults to the rate, the non-dividend-paying case."""
    return spot * math.exp((rate if carry is None else carry) * time)


def log_moneyness_on_forward(strike: float, forward: float) -> float:
    """``log(K / F)``.

    Deliberately not ``moneyness.log_moneyness``, which is ``log(S / K)`` and
    carries the opposite sign.
    """
    return math.log(strike / forward)


# -- implied volatility ---------------------------------------------------


def implied_payload(args: dict[str, Any]) -> dict[str, Any]:
    quote: Quote = build_quote(args)
    option = OptionType(args["type"])

    # Check attainability first rather than letting the solver discover it. A
    # price outside the range no volatility can produce has no implied
    # volatility at all, and saying so — with the range, and which side it fell
    # on — is more use than a number from a solver that gave up.
    limits = bounds(quote, option)
    if not (limits.lower <= quote.price <= limits.upper):
        side = "below" if quote.price < limits.lower else "above"
        raise DomainError(
            f"price={quote.price:g} is {side} the attainable range "
            f"[{limits.lower:g}, {limits.upper:g}], so it has no implied volatility: "
            "no non-negative volatility reproduces it. The quote is inconsistent with "
            "the spot, strike, time and rate given.",
            field="price",
        )

    solution = solve(quote, option)
    return {
        "impliedVol": solution.vol,
        "totalVol": solution.total_vol,
        "totalVariance": solution.vol**2 * quote.time,
        "logMoneyness": log_moneyness_on_forward(
            quote.strike, forward_of(quote.spot, quote.time, quote.carry, quote.rate)
        ),
        "diagnostics": {
            "method": solution.method.value,
            "iterations": solution.iterations,
            # The price error at the recovered volatility. A residual that is
            # not tiny means the answer is not converged, whatever the solver
            # reported, so it travels with the number rather than being dropped.
            "residual": solution.residual,
            "attainableRange": {"lower": limits.lower, "upper": limits.upper},
        },
    }


# -- slices ---------------------------------------------------------------


def _slice_from_quotes(
    strikes: list[float], vols: list[float], forward: float, time: float
) -> tuple[SVI, Any, list[float]]:
    ks = [log_moneyness_on_forward(strike, forward) for strike in strikes]
    variances = [vol * vol * time for vol in vols]
    fit = calibrate(ks, variances)
    return fit.slice_, fit, ks


def _describe_slice(slice_: SVI, time: float, fit: Any | None = None) -> dict[str, Any]:
    left, right = slice_.wing_slopes
    described: dict[str, Any] = {
        "maturity": time,
        "svi": {
            "a": slice_.a,
            "b": slice_.b,
            "rho": slice_.rho,
            "m": slice_.m,
            "s": slice_.s,
        },
        "atmVol": slice_.volatility(0.0, time),
        "wingSlopes": {
            "left": left,
            "right": right,
            # Lee's moment formula caps each slope at 2. Above it, the fit is
            # claiming the underlying has no moment of the corresponding order,
            # which is a strong thing to assert by accident while fitting five
            # parameters to a handful of quotes.
            "withinLeeBound": max(left, right) <= 2.0,
        },
    }
    if fit is not None:
        described["fit"] = {
            "rmse": fit.rmse,
            "maxError": fit.max_error,
            "iterations": fit.iterations,
            "converged": fit.converged,
        }
    return described


def _butterfly_report(slice_: SVI, time: float) -> dict[str, Any]:
    surface = Surface([(time, slice_)])
    worst = surface.butterfly(wing=SCAN_WING, maturities=[time])[0]
    _, value, at = worst
    return {
        "butterflyFree": value >= 0.0,
        "worstDurrleman": value,
        "atLogMoneyness": at,
    }


def fit_slice_payload(args: dict[str, Any]) -> dict[str, Any]:
    quotes = args["quotes"]
    time = float(args["time"])
    forward = float(args["forward"])
    strikes = [float(q["strike"]) for q in quotes]
    vols = [float(q["vol"]) for q in quotes]

    if any(strike <= 0 for strike in strikes):
        raise DomainError("every strike must be positive", field="quotes")
    if any(vol <= 0 for vol in vols):
        raise DomainError("every quoted volatility must be positive", field="quotes")
    if len(set(strikes)) != len(strikes):
        raise DomainError("strikes must be distinct; one of them is repeated", field="quotes")

    slice_, fit, ks = _slice_from_quotes(strikes, vols, forward, time)
    described = _describe_slice(slice_, time, fit)
    described["forward"] = forward
    described["arbitrage"] = _butterfly_report(slice_, time)
    described["quotes"] = [
        {
            "strike": strike,
            "logMoneyness": k,
            "quotedVol": vol,
            "fittedVol": slice_.volatility(k, time),
        }
        for strike, k, vol in zip(strikes, ks, vols, strict=True)
    ]
    return described


# -- surfaces -------------------------------------------------------------


def _build_surface(args: dict[str, Any]) -> tuple[Surface, list[dict[str, Any]]]:
    described: list[dict[str, Any]] = []
    pairs: list[tuple[float, SVI]] = []
    for entry in args["slices"]:
        time = float(entry["time"])
        forward = float(entry["forward"])
        strikes = [float(q["strike"]) for q in entry["quotes"]]
        vols = [float(q["vol"]) for q in entry["quotes"]]
        slice_, fit, _ = _slice_from_quotes(strikes, vols, forward, time)
        pairs.append((time, slice_))
        entry_described = _describe_slice(slice_, time, fit)
        entry_described["forward"] = forward
        described.append(entry_described)
    return Surface(pairs), described


def _arbitrage_report(surface: Surface) -> dict[str, Any]:
    # The default scan checks the quoted maturities *and* the midpoint of every
    # gap between them. The midpoints are the point: quoted slices are usually
    # fitted to be admissible, and it is the interpolation between them where
    # the condition quietly stops holding.
    butterflies = surface.butterfly(wing=SCAN_WING)
    worst_time, worst_value, worst_at = min(butterflies, key=lambda row: row[1])

    calendars = surface.calendar(wing=SCAN_WING)
    report: dict[str, Any] = {
        "butterflyFree": worst_value >= 0.0,
        "worstButterfly": {
            "durrleman": worst_value,
            "atMaturity": worst_time,
            "atLogMoneyness": worst_at,
        },
    }
    if calendars:
        # `worst` is the largest amount by which total variance *decreases* with
        # maturity, so a positive number is the violation and the worst pair is
        # the maximum, not the minimum. The opposite sign convention to the
        # butterfly scan above, where a negative Durrleman value is the defect,
        # which is exactly why the output field names the direction.
        worst_calendar = max(calendars, key=lambda c: c.worst)
        report["calendarFree"] = all(c.free for c in calendars)
        report["worstCalendar"] = {
            "worstDecreaseInTotalVariance": worst_calendar.worst,
            "isViolation": not worst_calendar.free,
            "atLogMoneyness": worst_calendar.at,
            "earlier": worst_calendar.earlier,
            "later": worst_calendar.later,
        }
    else:
        # Calendar arbitrage is a statement about a pair of maturities, so a
        # single-slice surface cannot exhibit it. Reported as absent rather than
        # as "free", which would imply a check that was never run.
        report["calendarFree"] = None
        report["calendarNote"] = "not applicable: a single slice has no pair to cross"
    return report


def _labelled_grid(
    surface: Surface, slices: list[dict[str, Any]], ratios: list[float]
) -> list[dict[str, Any]]:
    """Sample the surface onto a grid where every value carries its coordinates.

    Rows are keyed by maturity and cells by strike-over-forward, as strings. The
    cost is verbosity; the benefit is that a row cannot be mistaken for a column
    and a transposed reading is impossible rather than merely unlikely.
    """
    grid: list[dict[str, Any]] = []
    for described in slices:
        time = described["maturity"]
        by_ratio = {
            f"{ratio:g}": surface.volatility(math.log(ratio), time) for ratio in ratios
        }
        grid.append(
            {
                "maturity": time,
                "forward": described["forward"],
                "volatilityByStrikeOverForward": by_ratio,
            }
        )
    return grid


def surface_payload(args: dict[str, Any]) -> dict[str, Any]:
    entries = args["slices"]
    times = [float(entry["time"]) for entry in entries]
    if len(set(times)) != len(times):
        raise DomainError("each slice must have a distinct maturity", field="slices")

    surface, described = _build_surface(args)
    payload: dict[str, Any] = {
        "slices": described,
        "maturities": list(surface.maturities),
        "arbitrage": _arbitrage_report(surface),
    }
    ratios = args.get("strikeRatios")
    if ratios:
        payload["grid"] = _labelled_grid(surface, described, [float(r) for r in ratios])
    return payload


def local_vol_payload(args: dict[str, Any]) -> dict[str, Any]:
    surface, described = _build_surface(args)
    points: list[dict[str, Any]] = []
    for ratio in args["strikeRatios"]:
        k = math.log(float(ratio))
        for time in args["maturities"]:
            local = surface.local_vol(k, float(time))
            point: dict[str, Any] = {
                "strikeOverForward": float(ratio),
                "maturity": float(time),
                "admissible": local.admissible,
            }
            if local.admissible:
                point["localVol"] = local.volatility
                point["localVariance"] = local.variance
            else:
                # Where the Dupire identity has no answer the library returns
                # nan. Emitting that as JSON is not portable and reads as a
                # number to anything that parses it loosely, so the fields are
                # omitted and the reason is given instead.
                point["reason"] = (
                    "the Dupire identity is inadmissible here: "
                    f"dw/dT={local.dw_dt:g} and Durrleman g={local.durrleman:g}, "
                    "and a local variance requires both to be positive"
                )
            points.append(point)
    return {"slices": described, "points": points}


# -- schemas --------------------------------------------------------------

_QUOTE_ITEM = {
    "type": "object",
    "properties": {
        "strike": {"type": "number", "exclusiveMinimum": 0, "description": "Strike."},
        "vol": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Quoted implied volatility, as a decimal fraction: 20% is 0.2.",
        },
    },
    "required": ["strike", "vol"],
    "additionalProperties": False,
}

_SLICE_ITEM = {
    "type": "object",
    "properties": {
        "time": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Year fraction to this maturity.",
        },
        "forward": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Forward price at this maturity, which log-moneyness is measured on.",
        },
        "quotes": {
            "type": "array",
            "items": _QUOTE_ITEM,
            "minItems": 5,
            "description": "At least five strikes, since a raw SVI slice has five parameters.",
        },
    },
    "required": ["time", "forward", "quotes"],
    "additionalProperties": False,
}

_STRIKE_RATIOS = {
    "type": "array",
    "items": {"type": "number", "exclusiveMinimum": 0},
    "minItems": 1,
    "maxItems": 40,
    "description": (
        "Strikes to report, as multiples of the forward: 1.0 is at the money, "
        "0.9 is ten percent below it."
    ),
}


def register(registry: ToolRegistry) -> ToolRegistry:
    """Add the volatility tools to ``registry``."""

    quote_properties = {
        "spot": _SPOT,
        "strike": _STRIKE,
        "time": _TIME,
        "rate": _RATE,
        "carry": _CARRY,
        "type": _TYPE,
        "price": {
            "type": "number",
            "minimum": 0,
            "description": "Observed market price of the option.",
        },
    }
    registry.register(
        "implied_volatility",
        title="Implied volatility",
        description=(
            "Recover the Black-Scholes volatility implied by an observed option "
            "price, with the evidence that the answer is right: how many "
            "iterations it took, which method produced it, and the price "
            "residual at the recovered volatility. A price outside the "
            "attainable range is refused with the range, since no volatility "
            "reproduces it."
        ),
        input_schema={
            "type": "object",
            "properties": quote_properties,
            "required": ["spot", "strike", "time", "rate", "price", "type"],
            "additionalProperties": False,
        },
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )(guard(implied_payload))

    registry.register(
        "fit_volatility_slice",
        title="Fit a volatility smile",
        description=(
            "Fit a raw SVI slice to quoted volatilities at one maturity. Returns "
            "the five parameters, the fit residuals, the asymptotic wing slopes "
            "against Lee's bound, and whether the fitted slice implies a "
            "non-negative probability density. Log-moneyness is measured on the "
            "forward as log(strike / forward)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "time": _SLICE_ITEM["properties"]["time"],  # type: ignore[index]
                "forward": _SLICE_ITEM["properties"]["forward"],  # type: ignore[index]
                "quotes": _SLICE_ITEM["properties"]["quotes"],  # type: ignore[index]
            },
            "required": ["time", "forward", "quotes"],
            "additionalProperties": False,
        },
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )(guard(fit_slice_payload))

    registry.register(
        "fit_volatility_surface",
        title="Fit a volatility surface",
        description=(
            "Fit a volatility surface through quoted slices at several "
            "maturities. Returns the SVI parameters for each slice rather than a "
            "grid of numbers, since the parameters are exact and a grid is easy "
            "to misread, plus both no-arbitrage checks: butterfly, which is "
            "whether the implied density stays non-negative, and calendar, which "
            "is whether two maturities cross. Both are scanned between the "
            "quoted maturities as well as at them, because interpolation is "
            "where a condition quietly stops holding. Ask for strikeRatios to "
            "also receive a labelled grid of volatilities."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "slices": {"type": "array", "items": _SLICE_ITEM, "minItems": 1},
                "strikeRatios": _STRIKE_RATIOS,
            },
            "required": ["slices"],
            "additionalProperties": False,
        },
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )(guard(surface_payload))

    registry.register(
        "local_volatility",
        title="Dupire local volatility",
        description=(
            "Extract Dupire local volatilities from a fitted surface at chosen "
            "strikes and maturities. Points where the identity has no answer are "
            "returned marked inadmissible, with the two quantities that failed, "
            "rather than as a number that would read as a real volatility."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "slices": {"type": "array", "items": _SLICE_ITEM, "minItems": 1},
                "strikeRatios": _STRIKE_RATIOS,
                "maturities": {
                    "type": "array",
                    "items": {"type": "number", "exclusiveMinimum": 0},
                    "minItems": 1,
                    "maxItems": 40,
                    "description": "Year fractions at which to evaluate the local volatility.",
                },
            },
            "required": ["slices", "strikeRatios", "maturities"],
            "additionalProperties": False,
        },
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )(guard(local_vol_payload))

    return registry


__all__ = [
    "SCAN_WING",
    "Method",
    "forward_of",
    "log_moneyness_on_forward",
    "register",
]
