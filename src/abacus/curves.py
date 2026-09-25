"""Discount curves, bond analytics, curve risk and spreads.

The numbers come from ``tenor``; this module is the wiring and the conventions.

**Every convention is an argument, never a default, and that is deliberate.**
Fixed income is the one corner of this server where the arithmetic is easy and the
conventions decide the answer. A bond priced on the wrong day count basis is wrong
by a few basis points, which is exactly the size of the spread anyone is trying to
measure, so it is not approximately right — it is answering a different question,
and the answer looks entirely normal. ``tenor`` refuses to guess a basis, a
frequency or a rolling rule, and the schemas here do not undo that by supplying
one.

**A curve handle carries the curve, not a summary of it.** This is the opposite of
the returns handle in :mod:`abacus.risk`, and the difference has a reason worth
stating so nobody harmonises the two by mistake. A returns matrix grows with the
length of the history — five years of daily data on ten assets encodes to about
69,000 characters against an 8192-character handle limit — so that handle has to
carry second moments and give up the path. A bootstrapped curve *is* its pillars:
two values per instrument, whatever the history behind the quotes. Measured, a
five-pillar curve encodes to 264 characters and a sixty-pillar curve to 764. So
the curve handle carries the pillars and the quotes they were solved from, and
nothing is lost by reusing one — key rate risk, instrument risk, spreads and bond
valuation all work from it exactly as they would from a fresh bootstrap.

**A bootstrap reports whether it worked.** A curve that fails to reprice the
instruments it was built from is not a slightly wrong curve, it is a curve that
means nothing, and the failure is invisible in the pillar values. So the result
carries the sweep count, whether every pillar reprices its instrument, and the
residual on each solve.

**A yield and a curve are different questions.** The yield to maturity is one rate
that reprices the bond; the curve prices each cashflow at its own rate. Duration
computed from the first is not duration computed from the second, and neither is
wrong. Both are reported when both are available, labelled, rather than one being
quietly chosen.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date
from typing import Any

from tenor import (
    BadHorizon,
    Basis,
    Bond,
    Compounding,
    Deposit,
    DiscountCurve,
    Exercise,
    Frequency,
    Future,
    Instrument,
    Interpolation,
    Lattice,
    Rolling,
    Swap,
    bootstrap,
    buckets_from,
    curvature,
    horizon_return,
    i_spread,
    instrument_risk,
    key_rates,
    lattice_price,
    level,
    option_adjusted_spread,
    option_cost,
    shape_duration,
    slope,
    steps_between,
    z_spread,
)

from .analytics import guard
from .handles import HandleError, HandleTooLarge, Minter
from .tools import DomainError, ToolExecutionError, ToolRegistry

#: Tag inside a handle's payload saying what kind of state it holds. Three tool
#: groups mint handles under the same key, so the kind has to be checked rather
#: than assumed.
CURVE_KIND = "discount-curve"

#: Most instruments a bootstrap may take. One pillar each, and a screen longer
#: than this is a data feed rather than something a person assembled.
MAX_INSTRUMENTS = 60

#: Most dates a single curve query may ask about.
MAX_QUERY_POINTS = 60

#: Most key rate buckets. More buckets than pillars does not add information —
#: the durations in between are interpolations of the ones at the pillars.
MAX_BUCKETS = 24

#: Most steps a short-rate lattice may carry. A semi-annual tree to thirty years
#: is sixty, and the cost of a calibration is linear in this.
MAX_LATTICE_STEPS = 120

#: The shift used for every bumped-curve sensitivity unless the caller says
#: otherwise: one basis point, which is what a "01" in dv01 or pv01 means.
DEFAULT_SHIFT = 0.0001

def _default_rolling(cls: type) -> Rolling:
    """The rolling rule a ``tenor`` class uses when none is given.

    Read out of the library's own dataclass default rather than chosen here. The
    alternative — writing ``Rolling.NONE`` when the caller omits the field — is what
    the first draft of this module did, and it silently moved every unadjusted
    schedule off the market convention, changing prices by a coupon's worth of
    accrual with nothing in the output to show it.
    """
    for field in dataclasses.fields(cls):
        if field.name == "rolling":
            assert isinstance(field.default, Rolling)
            return field.default
    raise AssertionError(f"{cls.__name__} has no rolling field")


#: What an omitted ``rolling`` means. Asserted equal across the classes that take
#: one, so a divergence in the library shows up here rather than in a price.
DEFAULT_ROLLING = _default_rolling(Swap)

_BASES = tuple(basis.name for basis in Basis)
_COMPOUNDINGS = tuple(item.name for item in Compounding)
_FREQUENCIES = tuple(item.name for item in Frequency)
_INTERPOLATIONS = tuple(item.name for item in Interpolation)
_ROLLINGS = tuple(item.name for item in Rolling)


# -- schema fragments --------------------------------------------------------

_DATE = {
    "type": "string",
    "pattern": r"^\d{4}-\d{2}-\d{2}$",
    "description": "A calendar date in ISO form, as 2026-01-05.",
}

_BASIS = {
    "type": "string",
    "enum": list(_BASES),
    "description": (
        "Day count basis. Required, with no default anywhere: a bond on the wrong "
        "basis is wrong by a few basis points, which is the size of the spread being "
        "measured, so the answer looks normal and is to a different question. "
        "ACT_360 for money-market deposits, ACT_365F or ACT_ACT_ISDA for curves, "
        "THIRTY_360_BOND for a US corporate coupon."
    ),
}

_FREQUENCY = {
    "type": "string",
    "enum": list(_FREQUENCIES),
    "description": "Coupon or fixed-leg payments per year.",
}

_COMPOUNDING = {
    "type": "string",
    "enum": list(_COMPOUNDINGS),
    "description": (
        "How a zero rate is compounded. CONTINUOUS is the curve's own convention "
        "and the default for anything derived from it; the others are for quoting."
    ),
}

_ROLLING = {
    "type": "string",
    "enum": list(_ROLLINGS),
    "description": (
        f"How a payment date landing on a non-business day is moved. Defaults to "
        f"{DEFAULT_ROLLING.name}, which is the market convention: forward "
        "to the next business day unless that crosses into the next month, in which "
        "case backward. NONE leaves the schedule unadjusted, which is what an "
        "unadjusted analytic convention wants. It changes the cashflow dates and "
        "therefore the price, so it is reported back on every result."
    ),
}

_DEPOSIT = {
    "type": "object",
    "properties": {
        "start": _DATE,
        "maturity": _DATE,
        "rate": {"type": "number", "minimum": -0.1, "maximum": 1.0},
        "basis": _BASIS,
        "label": {"type": "string", "maxLength": 64},
    },
    "required": ["maturity", "rate", "basis"],
    "additionalProperties": False,
    "description": (
        "A simple-interest deposit. `start` defaults to the curve's reference date. "
        "The rate is a decimal fraction: twenty basis points is 0.002."
    ),
}

_FUTURE = {
    "type": "object",
    "properties": {
        "start": _DATE,
        "end": _DATE,
        "price": {"type": "number", "minimum": 0, "maximum": 200},
        "basis": _BASIS,
        "convexity": {
            "type": "number",
            "minimum": -0.05,
            "maximum": 0.05,
            "description": (
                "Convexity adjustment in rate terms, subtracted from the implied "
                "forward. Defaults to zero, which is the unadjusted futures rate. "
                "A futures contract settles on a rate and a forward does not, and "
                "the gap is not negligible at long maturities."
            ),
        },
        "label": {"type": "string", "maxLength": 64},
    },
    "required": ["start", "end", "price", "basis"],
    "additionalProperties": False,
    "description": "A rate future, quoted as 100 minus the rate.",
}

_SWAP = {
    "type": "object",
    "properties": {
        "effective": _DATE,
        "maturity": _DATE,
        "rate": {"type": "number", "minimum": -0.1, "maximum": 1.0},
        "frequency": _FREQUENCY,
        "basis": _BASIS,
        "rolling": _ROLLING,
        "label": {"type": "string", "maxLength": 64},
    },
    "required": ["maturity", "rate", "frequency", "basis"],
    "additionalProperties": False,
    "description": (
        "A par interest rate swap: the fixed rate that makes it worth nothing today. "
        "`effective` defaults to the curve's reference date."
    ),
}

_BOND = {
    "type": "object",
    "properties": {
        "effective": _DATE,
        "maturity": _DATE,
        "coupon": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": (
                "Annual coupon rate as a decimal fraction: a 5% coupon is 0.05. It is "
                "divided by the frequency to give the periodic payment."
            ),
        },
        "frequency": _FREQUENCY,
        "basis": _BASIS,
        "redemption": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": 1000,
            "description": "Redemption amount per 100 of face. Defaults to 100.",
        },
        "rolling": _ROLLING,
        "label": {"type": "string", "maxLength": 64},
    },
    "required": ["effective", "maturity", "coupon", "frequency", "basis"],
    "additionalProperties": False,
    "description": (
        "A fixed-coupon bond. Prices are per 100 of face throughout, so a bond "
        "trading at a 20% premium is 120."
    ),
}

_CURVE_HANDLE = {
    "type": "string",
    "minLength": 8,
    "description": (
        "A handle from bootstrap_discount_curve, in place of the instruments. Unlike "
        "the returns handle used by the portfolio tools, this one carries the curve "
        "itself — a curve is its pillars, which is two numbers per instrument — so "
        "nothing is lost by reusing it and every curve tool accepts it. It expires; "
        "bootstrap again if it does."
    ),
}

_SETTLEMENT = {
    **_DATE,
    "description": (
        "Settlement date for the valuation. Defaults to the curve's reference date. "
        "Accrued interest is measured from the last coupon to this date."
    ),
}

_SHIFT = {
    "type": "number",
    "exclusiveMinimum": 0,
    "maximum": 0.01,
    "description": (
        f"Rate shift for the bumped-curve sensitivities, as a decimal fraction. "
        f"Defaults to {DEFAULT_SHIFT}, one basis point, which is what the '01' in "
        "dv01 means. A larger shift measures a secant rather than a tangent, which "
        "is sometimes what is wanted and is never the same number."
    ),
}


def _day(raw: object, field: str) -> date:
    """Parse an ISO date, blaming the field rather than the parser."""
    if not isinstance(raw, str):
        raise DomainError(f"{field} must be an ISO date string", field=field)
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise DomainError(f"{field} is not a valid ISO date: {raw!r}", field=field) from exc


def _instruments(args: dict[str, Any], reference: date) -> list[Instrument]:
    """Build the quote set, in the order the caller gave it.

    Order is not sorted here. ``tenor`` sorts by maturity internally and refuses a
    duplicate, and reordering the caller's list would make the instrument risk
    report disagree with the list the caller sent.
    """
    built: list[Instrument] = []
    for index, raw in enumerate(args.get("deposits") or []):
        built.append(
            Deposit(
                start=(
                    _day(raw["start"], f"deposits/{index}/start")
                    if "start" in raw
                    else reference
                ),
                maturity=_day(raw["maturity"], f"deposits/{index}/maturity"),
                rate=float(raw["rate"]),
                basis=Basis[raw["basis"]],
                label=str(raw.get("label", "")),
            )
        )
    for index, raw in enumerate(args.get("futures") or []):
        built.append(
            Future(
                start=_day(raw["start"], f"futures/{index}/start"),
                end=_day(raw["end"], f"futures/{index}/end"),
                price=float(raw["price"]),
                basis=Basis[raw["basis"]],
                convexity=float(raw.get("convexity", 0.0)),
                label=str(raw.get("label", "")),
            )
        )
    for index, raw in enumerate(args.get("swaps") or []):
        built.append(
            Swap(
                effective=(
                    _day(raw["effective"], f"swaps/{index}/effective")
                    if "effective" in raw
                    else reference
                ),
                maturity=_day(raw["maturity"], f"swaps/{index}/maturity"),
                rate=float(raw["rate"]),
                frequency=Frequency[raw["frequency"]],
                basis=Basis[raw["basis"]],
                # Omitted means the library's own default, not a value chosen here.
                # Substituting Rolling.NONE would silently change the schedule — and
                # so the answer — for every caller who did not name a rule.
                rolling=Rolling[raw["rolling"]] if "rolling" in raw else DEFAULT_ROLLING,
                label=str(raw.get("label", "")),
            )
        )

    if not built:
        raise DomainError(
            "a curve needs at least one instrument: supply `deposits`, `futures` or "
            "`swaps`. Each one fixes one pillar, and the curve is only defined out to "
            "the longest maturity among them.",
            field="swaps",
        )
    if len(built) > MAX_INSTRUMENTS:
        raise DomainError(
            f"{len(built)} instruments is over the {MAX_INSTRUMENTS} limit.",
            field="swaps",
        )
    return built


def _bond(raw: dict[str, Any], reference: date) -> Bond:
    return Bond(
        effective=_day(raw["effective"], "bond/effective"),
        maturity=_day(raw["maturity"], "bond/maturity"),
        coupon=float(raw["coupon"]),
        frequency=Frequency[raw["frequency"]],
        basis=Basis[raw["basis"]],
        redemption=float(raw.get("redemption", 100.0)),
        # Omitted means the library's own default rather than a value invented here.
        rolling=Rolling[raw["rolling"]] if "rolling" in raw else DEFAULT_ROLLING,
        label=str(raw.get("label", "")),
    )


def _instrument_payload(instrument: Instrument) -> dict[str, Any]:
    """Describe an instrument by kind and terms, for echoing back in a risk report."""
    common = {"label": instrument.label or None}
    if isinstance(instrument, Deposit):
        return {
            **common,
            "kind": "deposit",
            "start": instrument.start.isoformat(),
            "maturity": instrument.maturity.isoformat(),
            "rate": instrument.rate,
            "basis": instrument.basis.name,
        }
    if isinstance(instrument, Future):
        return {
            **common,
            "kind": "future",
            "start": instrument.start.isoformat(),
            "end": instrument.end.isoformat(),
            "price": instrument.price,
            "convexity": instrument.convexity,
            "basis": instrument.basis.name,
        }
    return {
        **common,
        "kind": "swap",
        "effective": instrument.effective.isoformat(),
        "maturity": instrument.maturity.isoformat(),
        "rate": instrument.rate,
        "frequency": instrument.frequency.name,
        "basis": instrument.basis.name,
    }


def _bucket_label(years: float) -> str:
    """A readable name for a key rate bucket.

    ``buckets_from`` leaves the label empty, and an unlabelled bucket in a table of
    durations is exactly the sort of thing that gets read against the wrong row.
    """
    if years < 1.0:
        return f"{round(years * 12)}m"
    # Two decimals, trailing zeros stripped. Pillar times are day counts rather than
    # round years, so 5.0027 reads as "5y" and 10.0082 as "10.01y" — close enough to
    # find the row, precise enough not to claim two pillars are the same one.
    return f"{years:.2f}".rstrip("0").rstrip(".") + "y"


class CurveTools:
    """The curve, bond and spread tools, sharing one handle minter."""

    def __init__(self, minter: Minter | None = None) -> None:
        self.minter = minter if minter is not None else Minter()

    # -- handles ----------------------------------------------------------

    def _mint(self, curve: DiscountCurve, instruments: list[Instrument]) -> str:
        payload = {
            "kind": CURVE_KIND,
            "reference": curve.reference.isoformat(),
            "basis": curve.basis.name,
            "interpolation": curve.interpolation.name,
            "pillars": [[pillar.day.isoformat(), pillar.discount] for pillar in curve.pillars],
            "quotes": [_instrument_payload(instrument) for instrument in instruments],
        }
        try:
            return self.minter.mint(payload)
        except HandleTooLarge as exc:
            raise DomainError(
                f"this curve encodes to a {exc.size}-character handle, over the "
                f"{exc.limit}-character limit. A handle carries the curve's pillars and "
                "the quotes behind them, so the size grows with the instrument count. "
                "Drop the instruments whose maturities the question does not reach.",
                field="swaps",
            ) from exc

    def _redeem(self, handle: object) -> tuple[DiscountCurve, list[Instrument]]:
        try:
            payload = self.minter.redeem(handle)
        except HandleError as exc:
            raise ToolExecutionError(
                exc.message,
                kind=exc.kind,
                details=[{"path": "/handle", "keyword": exc.kind, "message": exc.message}],
            ) from exc
        if payload.get("kind") != CURVE_KIND:
            raise ToolExecutionError(
                "that handle does not hold a discount curve. A handle from "
                "open_position_book holds option positions and one from "
                "estimate_return_moments holds a covariance estimate; call "
                "bootstrap_discount_curve to get one of these.",
                kind="handle_wrong_kind",
                details=[
                    {
                        "path": "/handle",
                        "keyword": "handle_wrong_kind",
                        "message": f"handle holds {payload.get('kind', 'something else')!r}",
                    }
                ],
            )
        curve = DiscountCurve.from_discounts(
            date.fromisoformat(str(payload["reference"])),
            [(date.fromisoformat(day), float(factor)) for day, factor in payload["pillars"]],
            basis=Basis[str(payload["basis"])],
            interpolation=Interpolation[str(payload["interpolation"])],
        )
        quotes = self._quotes_from(payload.get("quotes") or [], curve.reference)
        return curve, quotes

    @staticmethod
    def _quotes_from(raw: list[Any], reference: date) -> list[Instrument]:
        """Rebuild the instruments a handle carries.

        They are carried because instrument risk — the profit and loss of each
        hedging instrument against a curve move — is a question about the quotes,
        not about the pillars, and re-deriving quotes from pillars is not possible.
        """
        rebuilt: list[Instrument] = []
        for entry in raw:
            kind = entry.get("kind")
            if kind == "deposit":
                rebuilt.append(
                    Deposit(
                        start=date.fromisoformat(entry["start"]),
                        maturity=date.fromisoformat(entry["maturity"]),
                        rate=float(entry["rate"]),
                        basis=Basis[entry["basis"]],
                        label=entry.get("label") or "",
                    )
                )
            elif kind == "future":
                rebuilt.append(
                    Future(
                        start=date.fromisoformat(entry["start"]),
                        end=date.fromisoformat(entry["end"]),
                        price=float(entry["price"]),
                        basis=Basis[entry["basis"]],
                        convexity=float(entry.get("convexity", 0.0)),
                        label=entry.get("label") or "",
                    )
                )
            elif kind == "swap":
                rebuilt.append(
                    Swap(
                        effective=date.fromisoformat(entry["effective"]),
                        maturity=date.fromisoformat(entry["maturity"]),
                        rate=float(entry["rate"]),
                        frequency=Frequency[entry["frequency"]],
                        basis=Basis[entry["basis"]],
                        label=entry.get("label") or "",
                    )
                )
        _ = reference
        return rebuilt

    def _curve(self, args: dict[str, Any]) -> tuple[DiscountCurve, list[Instrument]]:
        """Resolve a curve from a handle or from an inline quote set."""
        handle = args.get("handle")
        inline = any(args.get(key) for key in ("deposits", "futures", "swaps"))
        if (handle is None) == (not inline):
            raise DomainError(
                "supply exactly one of a `handle` or a quote set (`deposits`, "
                "`futures`, `swaps`): the handle to reuse a curve already bootstrapped, "
                "or the quotes to build one now. Sending both leaves it ambiguous which "
                "curve the answer came from."
            )
        if handle is not None:
            return self._redeem(handle)

        reference = _day(args["reference"], "reference")
        instruments = _instruments(args, reference)
        built = bootstrap(
            reference,
            instruments,
            basis=Basis[args["basis"]],
            interpolation=Interpolation[args.get("interpolation", "LOG_LINEAR_DISCOUNT")],
        )
        return built.curve, list(built.instruments)

    # -- tool bodies ------------------------------------------------------

    def bootstrap_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        reference = _day(args["reference"], "reference")
        instruments = _instruments(args, reference)
        basis = Basis[args["basis"]]
        interpolation = Interpolation[args.get("interpolation", "LOG_LINEAR_DISCOUNT")]
        built = bootstrap(reference, instruments, basis=basis, interpolation=interpolation)
        curve = built.curve
        compounding = Compounding[args.get("compounding", "CONTINUOUS")]

        return {
            "handle": self._mint(curve, list(built.instruments)),
            "expiresInSeconds": self.minter.ttl_s,
            "reference": reference.isoformat(),
            "basis": basis.name,
            "interpolation": interpolation.name,
            "horizon": curve.horizon.isoformat(),
            "sweeps": built.sweeps,
            # The check that matters. A curve that does not reprice its own inputs is
            # not a slightly wrong curve; it is a curve that means nothing, and the
            # pillar values look entirely ordinary either way.
            "repricesItsInstruments": curve.reprices_pillars(),
            "pillars": [
                {
                    "date": pillar.day.isoformat(),
                    "years": pillar.time,
                    "discount": pillar.discount,
                    # Null at the reference date, and not zero. The discount factor
                    # there is one whatever the rate is, so no rate is implied — and
                    # reporting zero would put a real number on the curve's front end
                    # that nothing in the quotes supports.
                    "zeroRate": (
                        None
                        if pillar.day == curve.reference
                        else curve.zero_rate(pillar.day, compounding)
                    ),
                }
                for pillar in curve.pillars
            ],
            "zeroRateCompounding": compounding.name,
            # One solve per pillar, with its residual. The residual is the thing
            # worth seeing: it is how far the instrument is from repricing, in the
            # instrument's own units, so a bootstrap that converged loosely is
            # visible rather than merely reported as converged.
            "solutions": [
                {
                    "instrument": _instrument_payload(instrument),
                    "discount": root.value,
                    "residual": root.residual,
                    "iterations": root.iterations,
                    "converged": root.converged,
                }
                for instrument, root in zip(built.instruments, built.solutions, strict=True)
            ],
            "note": (
                "Pass the handle to any other curve or bond tool instead of resending "
                "the quotes. It carries the pillars and the quotes behind them, so "
                "nothing is lost by reusing it — including instrument risk, which is a "
                "question about the quotes rather than about the pillars. The curve is "
                f"only defined out to {curve.horizon.isoformat()}; a cashflow beyond "
                "that has no discount factor and is refused rather than extrapolated."
            ),
        }

    def query_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        curve, _ = self._curve(args)
        compounding = Compounding[args.get("compounding", "CONTINUOUS")]
        forward_compounding = Compounding[args.get("forwardCompounding", "SIMPLE")]
        days = [_day(raw, f"dates/{index}") for index, raw in enumerate(args["dates"])]

        rows: list[dict[str, Any]] = []
        previous = curve.reference
        for day in days:
            if day > curve.horizon:
                raise DomainError(
                    f"{day.isoformat()} is past the curve's horizon of "
                    f"{curve.horizon.isoformat()}. The curve is only defined out to its "
                    "longest instrument, and extrapolating past it would return a "
                    "number with nothing behind it. Add a longer instrument.",
                    field="dates",
                )
            at_reference = day == curve.reference
            rows.append(
                {
                    "date": day.isoformat(),
                    "years": curve.time_to(day),
                    "discount": curve.discount(day),
                    # Null at the reference date rather than zero: the discount factor
                    # is one there whatever the rate is, so no rate is implied by it.
                    "zeroRate": None if at_reference else curve.zero_rate(day, compounding),
                    "forwardFromPrevious": (
                        curve.forward_rate(previous, day, forward_compounding)
                        if day > previous
                        else None
                    ),
                    "instantaneousForward": curve.instantaneous_forward(curve.time_to(day)),
                }
            )
            previous = day

        return {
            "reference": curve.reference.isoformat(),
            "basis": curve.basis.name,
            "interpolation": curve.interpolation.name,
            "horizon": curve.horizon.isoformat(),
            "zeroRateCompounding": compounding.name,
            "forwardCompounding": forward_compounding.name,
            "points": rows,
            "note": (
                "forwardFromPrevious is the rate between each date and the one before "
                "it in the list, so the list order is part of the question. For the "
                "first date the previous point is the curve's reference, which makes it "
                "a spot-starting rate; it is null only when a date repeats or the "
                "reference itself is asked for. Zero rates and forwards are quoted under the "
                "compoundings named above; the curve itself holds discount factors, "
                "and every rate here is derived from them. A zero rate at the reference "
                "date is null rather than zero: the discount factor there is one "
                "whatever the rate is, so none is implied."
            ),
        }

    def bond_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        raw = args["bond"]
        has_curve = args.get("handle") is not None or any(
            args.get(key) for key in ("deposits", "futures", "swaps")
        )
        quoted_price = args.get("price")
        quoted_yield = args.get("yieldToMaturity")

        if quoted_price is not None and quoted_yield is not None:
            raise DomainError(
                "supply a `price` or a `yieldToMaturity`, not both: each determines the "
                "other, and sending two that disagree has no meaning.",
                field="price",
            )

        reference = (
            _day(args["reference"], "reference")
            if "reference" in args
            else _day(raw["effective"], "bond/effective")
        )
        curve: DiscountCurve | None = None
        if has_curve:
            curve, _ = self._curve(args)
            reference = curve.reference
        bond = _bond(raw, reference)
        settlement = _day(args["settlement"], "settlement") if "settlement" in args else reference
        shift = float(args.get("shift", DEFAULT_SHIFT))

        payload: dict[str, Any] = {
            "bond": bond.name,
            "settlement": settlement.isoformat(),
            "basis": bond.basis.name,
            "frequency": bond.frequency.name,
            "rolling": bond.rolling.name,
            "accrual": bond.accrual.name,
            "periodicCoupon": bond.periodic_coupon,
            "redemption": bond.redemption,
            "accrued": bond.accrued(settlement),
            "cashflows": [
                {
                    "date": flow.day.isoformat(),
                    "amount": flow.amount,
                    "years": flow.years,
                    "periods": flow.periods,
                    "isRedemption": flow.redemption,
                }
                for flow in bond.cashflows(settlement)
            ],
        }

        if quoted_price is not None or quoted_yield is not None:
            if quoted_yield is None:
                assert quoted_price is not None  # the branch condition, for the checker
                root = bond.yield_from_clean(float(quoted_price), settlement)
                rate = root.value
                payload["yieldSolve"] = {
                    "iterations": root.iterations,
                    "residual": root.residual,
                    "converged": root.converged,
                }
            else:
                rate = float(quoted_yield)

            clean = bond.clean_price(rate, settlement)
            payload["fromYield"] = {
                "yieldToMaturity": rate,
                "cleanPrice": clean,
                "dirtyPrice": bond.dirty_price(rate, settlement),
                "macaulayDuration": bond.macaulay_duration(rate, settlement),
                "modifiedDuration": bond.modified_duration(rate, settlement),
                "convexity": bond.convexity(rate, settlement),
                "pv01": bond.pv01(rate, settlement, shift=shift),
                "note": (
                    "A yield to maturity is the single rate that reprices the bond, so "
                    "every sensitivity here is with respect to that one rate. Macaulay "
                    "duration is in years; modified duration is the price sensitivity, "
                    "smaller by a factor of one plus the periodic yield."
                ),
            }

        if curve is not None:
            curve_root = bond.curve_yield(curve, settlement)
            payload["fromCurve"] = {
                "reference": curve.reference.isoformat(),
                "price": bond.price_from_curve(curve, settlement),
                "curveYield": curve_root.value,
                "curveYieldConverged": curve_root.converged,
                "effectiveDuration": bond.effective_duration(curve, settlement, shift=shift),
                "effectiveConvexity": bond.effective_convexity(curve, settlement, shift=shift),
                "dv01": bond.dv01(curve, settlement, shift=shift),
                "shift": shift,
                "note": (
                    "The curve prices each cashflow at its own rate, so these are not "
                    "the yield-based figures under another name. Effective duration and "
                    "convexity are measured by shifting the whole curve, which is why "
                    "the shift is reported: a larger shift measures a secant rather "
                    "than a tangent."
                ),
            }
            if "fromYield" in payload:
                payload["yieldAgainstCurve"] = {
                    "priceDifference": payload["fromCurve"]["price"]
                    - payload["fromYield"]["cleanPrice"],
                    "durationDifference": payload["fromCurve"]["effectiveDuration"]
                    - payload["fromYield"]["modifiedDuration"],
                    "note": (
                        "The gap between the two is the shape of the curve. It is zero "
                        "only for a flat curve, and reporting it is the point: neither "
                        "figure is wrong and they answer different questions."
                    ),
                }
        return payload

    def horizon_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        curve, _ = self._curve(args)
        reference = curve.reference
        bond = _bond(args["bond"], reference)
        settlement = _day(args["settlement"], "settlement") if "settlement" in args else reference
        horizon = _day(args["horizon"], "horizon")
        try:
            found = horizon_return(bond, curve, settlement, horizon)
        except BadHorizon as bad:
            raise DomainError(str(bad), field="horizon") from bad

        return {
            "bond": bond.name,
            "settlement": settlement.isoformat(),
            "horizon": horizon.isoformat(),
            "periodYears": found.period,
            "startPrice": found.start_price,
            "forwardPrice": found.forward_price,
            "rolledPrice": found.rolled_price,
            "coupons": [
                {
                    "date": one.day.isoformat(),
                    "amount": one.amount,
                    "valueAtHorizon": one.value_at_horizon,
                }
                for one in found.coupons
            ],
            "couponIncome": found.coupon_income,
            "forwardPriceChange": found.forward_price_change,
            "financingRate": found.financing_rate,
            "financingCost": found.financing_cost,
            "carry": found.carry,
            "incomeLessFinancing": found.income_less_financing,
            "rollDown": found.roll_down,
            "totalReturn": found.total_return,
            "totalReturnBasisPoints": found.total_return_bps,
            "excessOverFinancing": found.excess_over_financing,
            # The identity this whole decomposition rests on. A curve that
            # fails it is not self-consistent, and every figure above is then
            # a number about that inconsistency rather than about the bond.
            "arbitrageFree": found.is_arbitrage_free(),
            "note": (
                "Every amount is per 100 of face and every price is DIRTY. Clean "
                "prices are the wrong unit here: accrued interest is part of what a "
                "holder earns over the period, and netting it out of the price "
                "without adding it back to income loses it.\n\n"
                "`carry` is coupon income plus the forward price change, and on an "
                "arbitrage-free curve it is IDENTICALLY the financing cost — "
                "`arbitrageFree` asserts that to within a hundredth of a basis point "
                "of price. So carry is not a source of return. A bond held to a "
                "horizon on a curve that evolves to its own forwards earns its "
                "funding cost and nothing else.\n\n"
                "The whole of the expected excess return is `rollDown`: the curve "
                "failing to evolve to its forwards, so the bond ages into a "
                "different point on an unchanged spot curve. It is positive on an "
                "upward-sloping curve, negative on an inverted one, and exactly zero "
                "on a flat one. `excessOverFinancing` equals it, which is the "
                "finding rather than a coincidence of the arithmetic.\n\n"
                "`incomeLessFinancing` is the market's OTHER definition of carry and "
                "is reported because the word is used both ways. Do not rank "
                "positions on it: a 9% bond can show +4.83 against a zero-coupon "
                "bond's -2.00 and go on to earn 27 basis points LESS over the year, "
                "because the high coupon is paid for by a price falling towards par "
                "by the same amount.\n\n"
                "Coupons inside the window are reinvested at the curve's own forward "
                "rates. That is the only assumption under which the identity holds; "
                "reinvesting at a chosen rate would make carry differ from the "
                "financing cost by the size of the view.\n\n"
                "The horizon must fall strictly before maturity. A redeemed bond has "
                "no price to roll to, so there is no price change to decompose — "
                "that is refused rather than reported as zero."
            ),
        }

    def curve_risk_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        curve, quotes = self._curve(args)
        bond = _bond(args["bond"], curve.reference)
        settlement = (
            _day(args["settlement"], "settlement") if "settlement" in args else curve.reference
        )
        shift = float(args.get("shift", DEFAULT_SHIFT))
        years = [float(year) for year in (args.get("buckets") or [])]
        if not years:
            # The curve's own pillars, which is the only default that is always valid.
            # A standard ladder of 0.5, 1, 2, 5, 10 and 30 years looks tidier and is
            # wrong: a bucket at a maturity where the curve has no pillar shifts
            # nothing, so its duration comes back as zero — which reads as an absence
            # of risk rather than an absence of curve, while the risk it should have
            # carried is quietly absorbed by its neighbours. `tenor` refuses such a
            # bucket, and the pillars are where the information is.
            years = [time for time in curve.times if time > 0.0]
        if len(years) > MAX_BUCKETS:
            raise DomainError(
                f"{len(years)} buckets is over the {MAX_BUCKETS} limit.", field="buckets"
            )
        if sorted(years) != years:
            raise DomainError(
                "buckets must be in increasing order of maturity, so the table reads "
                "along the curve.",
                field="buckets",
            )

        def value(shifted: DiscountCurve) -> float:
            return bond.price_from_curve(shifted, settlement)

        buckets = buckets_from(curve, years)
        rates = key_rates(value, curve, buckets, shift=shift)
        total = math.fsum(rate.duration for rate in rates)
        effective = bond.effective_duration(curve, settlement, shift=shift)
        horizon = curve.times[-1]

        payload: dict[str, Any] = {
            "bond": bond.name,
            "settlement": settlement.isoformat(),
            "price": value(curve),
            "shift": shift,
            "keyRates": [
                {
                    "bucket": _bucket_label(years[index]),
                    "years": years[index],
                    "duration": rate.duration,
                    "value": rate.value,
                }
                for index, rate in enumerate(rates)
            ],
            "keyRateDurationSum": total,
            "effectiveDuration": effective,
            # The identity that makes a key rate table mean anything: the durations
            # partition the effective duration, because the tent weights at the
            # buckets sum to one everywhere. A table that does not add up is measuring
            # something other than the curve it claims to.
            "sumAgainstEffectiveDuration": total - effective,
            "shapeDurations": {
                "level": shape_duration(value, curve, level(), shift=shift),
                "slope": shape_duration(value, curve, slope(horizon), shift=shift),
                "curvature": shape_duration(value, curve, curvature(horizon), shift=shift),
                "horizonYears": horizon,
                "note": (
                    "Level is a parallel shift and equals the effective duration. Slope "
                    "pivots the curve about its midpoint and curvature bends it, both "
                    "normalised over the curve's own horizon, so these three are a "
                    "coordinate system on curve moves rather than three separate "
                    "sensitivities."
                ),
            },
            "note": (
                "Each key rate duration is the sensitivity to a triangular bump centred "
                "on its bucket and falling to zero at the neighbouring ones. They sum "
                "to the effective duration because those tents add to one at every "
                "maturity, and the residual above says how nearly they do."
            ),
        }

        if quotes:
            built = bootstrap(
                curve.reference,
                quotes,
                basis=curve.basis,
                interpolation=curve.interpolation,
            )
            risks = instrument_risk(value, built, shift=shift)
            payload["instrumentRisk"] = {
                "instruments": [
                    {"instrument": _instrument_payload(item.instrument), "value": item.value}
                    for item in risks
                ],
                "total": math.fsum(item.value for item in risks),
                "note": (
                    "The profit and loss from bumping each quoted instrument by the "
                    "shift and rebuilding the curve from it. Unlike a key rate "
                    "duration, this is expressed in the things that can actually be "
                    "traded, so it is the hedge rather than a description of the risk."
                ),
            }
        return payload

    def spreads_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        curve, _ = self._curve(args)
        bond = _bond(args["bond"], curve.reference)
        settlement = (
            _day(args["settlement"], "settlement") if "settlement" in args else curve.reference
        )
        price = float(args["price"])
        clean = bool(args.get("clean", True))

        zero = z_spread(bond, price, curve, settlement, clean=clean)
        payload: dict[str, Any] = {
            "bond": bond.name,
            "settlement": settlement.isoformat(),
            "price": price,
            "priceIsClean": clean,
            "curvePrice": bond.price_from_curve(curve, settlement),
            "zSpread": zero.value,
            "zSpreadConverged": zero.converged,
            "zSpreadResidual": zero.residual,
            "iSpread": i_spread(bond, price, curve, settlement, clean=clean),
            "note": (
                "The Z-spread is the parallel shift to every zero rate that makes the "
                "curve reprice the bond. The I-spread is the simpler quantity: the gap "
                "between the bond's yield and the curve's par rate at its maturity. "
                "They differ by the shape of the curve, and the Z-spread is the one "
                "that discounts each cashflow consistently."
            ),
        }

        first_call = args.get("firstCall")
        if first_call is None:
            return payload

        volatility = float(args["volatility"])
        step = 1.0 / bond.frequency.value
        steps = steps_between(curve, settlement, bond.maturity, step)
        if steps > MAX_LATTICE_STEPS:
            raise DomainError(
                f"a tree to {bond.maturity.isoformat()} at {bond.frequency.name} steps is "
                f"{steps} steps, over the {MAX_LATTICE_STEPS} limit.",
                field="bond",
            )
        tree = Lattice.calibrated(
            curve, settlement, steps=steps, step=step, volatility=volatility
        )
        first = steps_between(curve, settlement, _day(first_call, "firstCall"), step)
        call_price = float(args.get("callPrice", 100.0))
        issuer = not bool(args.get("holderOption", False))
        schedule = tuple((index, call_price) for index in range(first, steps))
        if not schedule:
            raise DomainError(
                f"the first exercise date {first_call} leaves no exercise opportunity "
                f"before the {bond.maturity.isoformat()} redemption. An option that can "
                "only be exercised at maturity is not an option.",
                field="firstCall",
            )
        exercise = Exercise(schedule, issuer=issuer)

        coupon = bond.periodic_coupon
        # A lattice values a bond as a full price at its settlement node, so an
        # observed clean price has to have its accrued interest added back before the
        # two can be compared.
        #
        # In practice this is always a no-op, and the reason is worth writing down. The
        # step length is one coupon period and `steps_between` refuses a maturity that
        # is not a whole number of steps from settlement, so every settlement the
        # lattice accepts falls on a coupon date, where the accrual is zero. The
        # conversion stays because it is what makes the two price conventions
        # commensurable, and because the alignment constraint is the library's rather
        # than something this code should assume will hold forever. A mid-period
        # settlement is refused, with the library naming the gap.
        accrued = bond.accrued(settlement)
        full_price = price + accrued if clean else price

        bullet = lattice_price(tree, coupon=coupon, redemption=bond.redemption)
        with_option = lattice_price(
            tree, coupon=coupon, redemption=bond.redemption, exercise=exercise
        )
        # Both spreads are solved against the *observed* price. Solving either against
        # the model's own price would be a self-consistency check dressed up as a
        # spread: it returns zero by construction and says nothing about the bond.
        ignoring = option_adjusted_spread(
            tree, full_price, coupon=coupon, redemption=bond.redemption
        )
        modelling = option_adjusted_spread(
            tree, full_price, coupon=coupon, redemption=bond.redemption, exercise=exercise
        )

        payload["embeddedOption"] = {
            "volatility": volatility,
            "steps": steps,
            "stepYears": step,
            "firstExercise": _day(first_call, "firstCall").isoformat(),
            "exerciseOpportunities": len(schedule),
            "callPrice": call_price,
            "heldBy": "issuer" if issuer else "holder",
            # A tree that does not reprice the curve it was calibrated to is not a
            # model of that curve, and every spread read off it would be wrong by an
            # amount nothing else reports.
            "latticeRepricesCurve": tree.reprices(curve, settlement),
            "accrued": accrued,
            "fullPriceUsed": full_price,
            "modelBulletPrice": bullet,
            "modelPriceWithOption": with_option,
            "modelOptionValue": bullet - with_option,
            "zeroVolatilitySpread": ignoring.value,
            "zeroVolatilitySpreadConverged": ignoring.converged,
            "optionAdjustedSpread": modelling.value,
            "optionAdjustedSpreadConverged": modelling.converged,
            "optionCost": option_cost(ignoring.value, modelling.value),
            "note": (
                "Both spreads are solved against the observed price. The "
                "option-adjusted spread reprices it with the option modelled; the "
                "zero-volatility spread reprices it with the option ignored. The "
                "difference is the option cost, and for an issuer's call it is positive: "
                "the issuer's right to redeem early is worth something, so once it is "
                "modelled a narrower spread is enough to explain the same price. The "
                "model prices above are at zero spread and are reported for context — "
                "the option value is the model's, not the market's."
            ),
        }
        return payload

    # -- registration -----------------------------------------------------

    def register(self, registry: ToolRegistry) -> ToolRegistry:
        read_only = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}
        quotes = {
            "deposits": {
                "type": "array",
                "items": _DEPOSIT,
                "maxItems": MAX_INSTRUMENTS,
                "description": "Money-market deposits, one pillar each.",
            },
            "futures": {
                "type": "array",
                "items": _FUTURE,
                "maxItems": MAX_INSTRUMENTS,
                "description": "Rate futures, one pillar each.",
            },
            "swaps": {
                "type": "array",
                "items": _SWAP,
                "maxItems": MAX_INSTRUMENTS,
                "description": "Par interest rate swaps, one pillar each.",
            },
            "reference": {
                **_DATE,
                "description": (
                    "The curve's reference date: where it starts and where every year "
                    "fraction on it is measured from. Required when quotes are supplied."
                ),
            },
            "basis": {
                **_BASIS,
                "description": (
                    "The day count basis of the curve itself, which is what its year "
                    "fractions are measured on. Independent of each instrument's own "
                    "basis, which stays with the instrument. " + str(_BASIS["description"])
                ),
            },
            "interpolation": {
                "type": "string",
                "enum": list(_INTERPOLATIONS),
                "description": (
                    "How the curve is interpolated between pillars. "
                    "LOG_LINEAR_DISCOUNT is the default and gives piecewise-constant "
                    "forwards. LINEAR_ZERO is smoother in zero rates and rougher in "
                    "forwards. MONOTONE_CONVEX keeps forwards positive and continuous, "
                    "which matters when the curve is read for forwards rather than for "
                    "discount factors."
                ),
            },
            "handle": _CURVE_HANDLE,
        }
        # Everything but `compounding`, which only some of the tools take. The
        # quote keys and the handle travel together because every tool below
        # accepts either source, and repeating the fragments per tool is how the
        # descriptions would drift apart.
        reusable = {
            key: quotes[key]
            for key in (
                "deposits",
                "futures",
                "swaps",
                "reference",
                "basis",
                "interpolation",
                "handle",
            )
        }

        registry.register(
            "bootstrap_discount_curve",
            title="Bootstrap a discount curve",
            description=(
                "Build a discount curve from deposits, rate futures and par swaps, and "
                "return a handle so it can be reused without resending the quotes. "
                "Reports the pillars it solved, the zero rate at each, and — the check "
                "that matters — whether the curve reprices the instruments it was built "
                "from, because a curve that does not is not slightly wrong, it means "
                "nothing, and the pillar values look ordinary either way. Every day "
                "count basis is required, with no default anywhere: a curve on the wrong "
                "basis answers a different question and looks normal doing it."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **{k: v for k, v in quotes.items() if k != "handle"},
                    "compounding": _COMPOUNDING,
                },
                "required": ["reference", "basis"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.bootstrap_payload))

        registry.register(
            "discount_curve_rates",
            title="Read discount factors and rates off a curve",
            description=(
                "Discount factors, zero rates, instantaneous forwards and the forward "
                "rate between consecutive dates, at whatever dates you ask for. Works "
                "from a curve handle or from a quote set. A date beyond the curve's "
                "horizon is refused rather than extrapolated: past the longest "
                "instrument there is nothing behind the number."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **reusable,
                    "dates": {
                        "type": "array",
                        "items": _DATE,
                        "minItems": 1,
                        "maxItems": MAX_QUERY_POINTS,
                        "description": (
                            "The dates to read the curve at, in the order you want them. "
                            "The order matters: each point's forward rate is measured "
                            "from the date before it."
                        ),
                    },
                    "compounding": _COMPOUNDING,
                    "forwardCompounding": {
                        **_COMPOUNDING,
                        "description": (
                            "Compounding for the forward rates between dates. Defaults "
                            "to SIMPLE, which is how a forward is normally quoted."
                        ),
                    },
                },
                "required": ["dates"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.query_payload))

        registry.register(
            "bond_analytics",
            title="Price, yield, duration and convexity of a bond",
            description=(
                "Bond analytics from a yield, from a curve, or from both. Given a price "
                "it solves the yield; given a yield it prices. With a curve it also "
                "prices every cashflow at its own rate and reports the effective "
                "duration and convexity from shifting the curve. Both sets are returned "
                "when both are available, labelled and with the difference between them, "
                "because a yield-based duration and a curve-based one are answers to "
                "different questions and neither is wrong. Prices are per 100 of face."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **reusable,
                    "bond": _BOND,
                    "price": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1000,
                        "description": (
                            "Clean price per 100 of face. The yield is solved from it. "
                            "Send this or a yieldToMaturity, not both."
                        ),
                    },
                    "yieldToMaturity": {
                        "type": "number",
                        "minimum": -0.5,
                        "maximum": 2,
                        "description": (
                            "Yield as a decimal fraction: 3% is 0.03. The price is "
                            "computed from it."
                        ),
                    },
                    "settlement": _SETTLEMENT,
                    "shift": _SHIFT,
                },
                "required": ["bond"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.bond_payload))

        registry.register(
            "bond_curve_risk",
            title="Key rate durations and curve shape risk",
            description=(
                "Where a bond's interest rate risk sits along the curve: a key rate "
                "duration per bucket, from a triangular bump centred on it; the level, "
                "slope and curvature durations, which are a coordinate system on curve "
                "moves rather than three unrelated numbers; and — when the curve came "
                "from quotes — the profit and loss of each quoted instrument, which is "
                "the hedge rather than a description of the risk. The key rate durations "
                "sum to the effective duration, and the residual is reported so a table "
                "that does not add up cannot be mistaken for one that does."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **reusable,
                    "bond": _BOND,
                    "settlement": _SETTLEMENT,
                    "shift": _SHIFT,
                    "buckets": {
                        "type": "array",
                        "items": {"type": "number", "exclusiveMinimum": 0, "maximum": 100},
                        "maxItems": MAX_BUCKETS,
                        "description": (
                            "Bucket maturities in years, increasing. Defaults to the "
                            "standard 0.5, 1, 2, 5, 10 and 30 year points that fall "
                            "inside the curve. More buckets than pillars adds no "
                            "information: the durations between pillars are "
                            "interpolations of the ones at them."
                        ),
                    },
                },
                "required": ["bond"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.curve_risk_payload))

        registry.register(
            "bond_carry_rolldown",
            title="Holding-period return: carry and roll-down",
            description=(
                "What a bond position earns between settlement and a horizon if "
                "nothing happens — and the split that matters, because 'nothing "
                "happens' means two different things. If the curve evolves to its own "
                "forwards the bond earns its funding cost and nothing else; that is an "
                "identity, not an approximation, and the result asserts it. Everything "
                "above the funding cost is roll-down: the bond ages, its remaining "
                "maturity shortens, and on an unchanged spot curve it is repriced off "
                "a lower point. On a curve running from 20bp to 220bp a ten-year bond "
                "held for a year earns 279 basis points, of which 50 is financing and "
                "249 is roll-down. Both market meanings of 'carry' are reported "
                "because the word is used for both, and the conventional one can rank "
                "two positions backwards."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **reusable,
                    "bond": _BOND,
                    "settlement": _SETTLEMENT,
                    "horizon": {
                        **_DATE,
                        "description": (
                            "End of the holding period. Must fall after settlement and "
                            "strictly before the bond's maturity: a redeemed bond has "
                            "no price at the horizon, so there is no price change to "
                            "decompose and the call is refused rather than answered "
                            "with zero."
                        ),
                    },
                },
                "required": ["bond", "horizon"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.horizon_payload))

        registry.register(
            "bond_spreads",
            title="Z-spread, I-spread and option-adjusted spread",
            description=(
                "The spread a bond's price implies over a curve. The Z-spread is the "
                "parallel shift to every zero rate that reprices it; the I-spread is the "
                "gap to the curve's par rate at its maturity, and the two differ by the "
                "shape of the curve. Give a first exercise date and a volatility and it "
                "also calibrates a short-rate lattice to the curve, values the embedded "
                "option, and reports the option-adjusted spread and the option cost — "
                "with whether the lattice reprices the curve it was calibrated to, "
                "because a tree that does not is not a model of that curve and every "
                "spread read off it would be wrong."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **reusable,
                    "bond": _BOND,
                    "price": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1000,
                        "description": "Observed price per 100 of face.",
                    },
                    "clean": {
                        "type": "boolean",
                        "description": (
                            "Whether `price` is clean, that is excludes accrued "
                            "interest. Defaults to true, which is how bonds are quoted."
                        ),
                    },
                    "settlement": _SETTLEMENT,
                    "volatility": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 2,
                        "description": (
                            "Short rate volatility for the lattice, as a decimal "
                            "fraction. Required with firstCall."
                        ),
                    },
                    "firstCall": {
                        **_DATE,
                        "description": (
                            "First date the option may be exercised. Supplying it turns "
                            "on the lattice and the option-adjusted spread; leaving it "
                            "out reports the two curve spreads only."
                        ),
                    },
                    "callPrice": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1000,
                        "description": "Exercise price per 100 of face. Defaults to 100.",
                    },
                    "holderOption": {
                        "type": "boolean",
                        "description": (
                            "True for a put held by the holder rather than a call held "
                            "by the issuer. Defaults to false. The sign of the option "
                            "cost follows from this, so it is not cosmetic."
                        ),
                    },
                },
                "required": ["bond", "price"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.spreads_payload))

        return registry


def register(registry: ToolRegistry, *, minter: Minter | None = None) -> ToolRegistry:
    """Add the curve and bond tools to ``registry``."""
    return CurveTools(minter).register(registry)


__all__ = [
    "CURVE_KIND",
    "DEFAULT_ROLLING",
    "DEFAULT_SHIFT",
    "MAX_BUCKETS",
    "MAX_INSTRUMENTS",
    "MAX_LATTICE_STEPS",
    "MAX_QUERY_POINTS",
    "CurveTools",
    "register",
]
