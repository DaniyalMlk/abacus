"""Positions, books, aggregate risk and scenario grids.

Everything else in this server answers a question about one option. A book is
the first thing here that is more than the sum of its calls, and it exists
because the sum is precisely what a language model should not be asked to
compute. Fifteen Greeks across eight legs is a hundred and twenty signed
multiplications and a hundred and twelve additions, every one of which is an
opportunity for a plausible wrong number, and none of which anybody downstream
can check.

Three decisions shape this module.

**A book is identified by a handle that contains it.** See :mod:`abacus.handles`
for why the alternative — a dictionary on the server — is not available in a
stateless protocol. The consequence here is that a book is immutable: amending
one mints a new handle and leaves the old one valid until it expires, which is
the only honest thing to do when the server has no record of either.

**Undefined sensitivities are reported as undefined, never as zero.** A leg that
has expired, or that carries no volatility, has no derivative at the kink in its
payoff. Contributing a zero for it would make a book look hedged in exactly the
situation where it is not, and nothing downstream could tell. So such legs are
priced — the price is still a limit and still exists — but excluded from the
aggregate, and the result says which legs were excluded and why, with
``complete`` set to ``false`` so a reader cannot miss it.

**A scenario grid is labelled, not a bare matrix.** The same argument
:mod:`abacus.vol` makes about volatility surfaces applies with more force here,
because a spot-versus-volatility grid is square often enough that a transposed
one still looks plausible. Every row states its volatility shift and every cell
states its spot, so the two axes cannot be confused for one another.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from moneyness import (
    Inputs,
    OptionType,
    charm,
    colour,
    delta,
    dual_delta,
    dual_gamma,
    gamma,
    intrinsic,
    price,
    rho,
    rho_carry,
    speed,
    theta,
    vanna,
    vega,
    veta,
    volga,
    zomma,
)

from .analytics import (
    _CARRY,
    _RATE,
    _SPOT,
    _STRIKE,
    _TIME,
    MAX_PLAUSIBLE_RATE,
    MAX_PLAUSIBLE_VOL,
    MAX_PLAUSIBLE_YEARS,
    _percentage_hint,
    guard,
)
from .handles import HandleError, HandleTooLarge, Minter
from .tools import DomainError, ToolExecutionError, ToolRegistry

#: Most legs a book may hold. The binding constraint is the handle size, which
#: refuses a book that will not encode; this is the friendlier limit in front of
#: it, so a caller building a book too large is told in terms of positions
#: rather than characters.
MAX_LEGS = 64

#: Most cells a scenario grid may contain. Twenty-one spot steps against eleven
#: volatility steps is a finer grid than anybody reads, and the cap exists so a
#: model cannot ask for a hundred thousand prices in one call.
MAX_GRID_CELLS = 231

#: Grid axes used when the caller does not supply them. Relative spot moves and
#: absolute volatility shifts, both chosen to span a normal day's risk report
#: rather than a stress test.
DEFAULT_SPOT_SHIFTS = (-0.10, -0.05, 0.0, 0.05, 0.10)
DEFAULT_VOL_SHIFTS = (-0.05, 0.0, 0.05)

#: The sensitivities an option leg contributes. Order is fixed so that the
#: aggregate, the per-leg breakdown and the tests all agree on it without
#: anybody having to keep three lists in step.
GREEK_NAMES = (
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
    "charm",
    "speed",
    "zomma",
    "colour",
    "veta",
    "dualDelta",
    "dualGamma",
    "rhoCarry",
)

_OPTION_GREEKS: dict[str, Callable[[Inputs, OptionType], float]] = {
    "delta": delta,
    "gamma": lambda i, _t: gamma(i),
    "vega": lambda i, _t: vega(i),
    "theta": theta,
    "rho": rho,
    "vanna": lambda i, _t: vanna(i),
    "volga": lambda i, _t: volga(i),
    "charm": charm,
    "speed": lambda i, _t: speed(i),
    "zomma": lambda i, _t: zomma(i),
    "colour": lambda i, _t: colour(i),
    "veta": lambda i, _t: veta(i),
    "dualDelta": dual_delta,
    "dualGamma": lambda i, _t: dual_gamma(i),
    "rhoCarry": rho_carry,
}

#: An underlying position. Delta is one per unit by definition and every other
#: sensitivity is exactly zero — not undefined, genuinely zero — so these legs
#: always count towards the aggregate.
_UNDERLYING_GREEKS = {name: (1.0 if name == "delta" else 0.0) for name in GREEK_NAMES}


class Leg:
    """One position in a book: an option on the book's underlying, or the underlying.

    Quantity is signed and is in contracts for an option and in units for the
    underlying; no contract multiplier is applied anywhere, because guessing one
    is worse than making the caller state its positions in the units it means.
    """

    __slots__ = ("carry", "instrument", "label", "quantity", "strike", "time", "vol")

    def __init__(
        self,
        *,
        instrument: str,
        quantity: float,
        strike: float | None = None,
        time: float | None = None,
        vol: float | None = None,
        carry: float | None = None,
        label: str | None = None,
    ) -> None:
        self.instrument = instrument
        self.quantity = quantity
        self.strike = strike
        self.time = time
        self.vol = vol
        self.carry = carry
        self.label = label

    @property
    def is_option(self) -> bool:
        return self.instrument in ("call", "put")

    @property
    def option_type(self) -> OptionType:
        return OptionType(self.instrument)

    def inputs(self, spot: float, rate: float, *, vol_shift: float = 0.0) -> Inputs:
        """Build the pricing inputs for this leg under a possibly shifted market.

        A volatility shift is floored at zero rather than rejected. A scenario
        that takes a 15-vol option down 20 points is a legitimate thing to ask
        for; the answer at zero volatility is the discounted intrinsic, which is
        a real number and the right one.
        """
        assert self.strike is not None and self.time is not None and self.vol is not None
        return Inputs(
            spot=spot,
            strike=self.strike,
            time=self.time,
            rate=rate,
            vol=max(0.0, self.vol + vol_shift),
            carry=self.carry,
        )

    def to_dict(self) -> dict[str, Any]:
        """Canonical form, as stored inside a handle and echoed back to the caller."""
        out: dict[str, Any] = {"instrument": self.instrument, "quantity": self.quantity}
        for name in ("strike", "time", "vol", "carry", "label"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out

    @classmethod
    def from_dict(cls, raw: Any, *, where: str) -> Leg:
        if not isinstance(raw, dict):
            raise DomainError(f"{where} must be an object")
        instrument = raw.get("instrument")
        if instrument not in ("call", "put", "underlying"):
            raise DomainError(
                f'{where}.instrument must be "call", "put" or "underlying"', field="legs"
            )
        quantity = raw.get("quantity")
        if not isinstance(quantity, int | float) or isinstance(quantity, bool):
            raise DomainError(f"{where}.quantity must be a number", field="legs")
        if quantity != quantity or math.isinf(quantity):
            raise DomainError(f"{where}.quantity must be finite", field="legs")

        leg = cls(
            instrument=instrument,
            quantity=float(quantity),
            strike=_optional_number(raw, "strike", where),
            time=_optional_number(raw, "time", where),
            vol=_optional_number(raw, "vol", where),
            carry=_optional_number(raw, "carry", where),
            label=raw.get("label") if isinstance(raw.get("label"), str) else None,
        )
        leg.check(where)
        return leg

    def check(self, where: str) -> None:
        """Reject a leg that parses but could not describe a position."""
        if self.is_option:
            for name in ("strike", "time", "vol"):
                if getattr(self, name) is None:
                    raise DomainError(
                        f"{where} is a {self.instrument} and needs {name}", field="legs"
                    )
            assert self.strike is not None and self.time is not None and self.vol is not None
            if self.strike <= 0.0:
                raise DomainError(f"{where}.strike must be positive", field="legs")
            if self.time < 0.0:
                raise DomainError(f"{where}.time must not be negative", field="legs")
            if self.vol < 0.0:
                raise DomainError(f"{where}.vol must not be negative", field="legs")
            if self.vol > MAX_PLAUSIBLE_VOL:
                raise DomainError(
                    _percentage_hint(f"{where}.vol", self.vol, MAX_PLAUSIBLE_VOL), field="legs"
                )
            if self.time > MAX_PLAUSIBLE_YEARS:
                raise DomainError(
                    f"{where}.time={self.time:g} is a year fraction, not a count of days",
                    field="legs",
                )
            if self.carry is not None and abs(self.carry) > MAX_PLAUSIBLE_RATE:
                raise DomainError(
                    _percentage_hint(f"{where}.carry", self.carry, MAX_PLAUSIBLE_RATE),
                    field="legs",
                )
        else:
            for name in ("strike", "time", "vol"):
                if getattr(self, name) is not None:
                    raise DomainError(
                        f"{where} is an underlying position and must not carry {name}; "
                        "it has no strike, no expiry and no volatility of its own",
                        field="legs",
                    )


def _optional_number(raw: dict[str, Any], name: str, where: str) -> float | None:
    value = raw.get(name)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise DomainError(f"{where}.{name} must be a number", field="legs")
    if value != value or math.isinf(value):
        raise DomainError(f"{where}.{name} must be finite", field="legs")
    return float(value)


class Book:
    """A set of legs over one underlying and one discount rate."""

    __slots__ = ("label", "legs", "rate", "spot")

    def __init__(
        self, *, spot: float, rate: float, legs: list[Leg], label: str | None = None
    ) -> None:
        self.spot = spot
        self.rate = rate
        self.legs = legs
        self.label = label

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "spot": self.spot,
            "rate": self.rate,
            "legs": [leg.to_dict() for leg in self.legs],
        }
        if self.label is not None:
            out["label"] = self.label
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Book:
        spot = raw.get("spot")
        rate = raw.get("rate")
        if not isinstance(spot, int | float) or isinstance(spot, bool) or spot <= 0:
            raise DomainError("spot must be a positive number", field="spot")
        if not isinstance(rate, int | float) or isinstance(rate, bool):
            raise DomainError("rate must be a number", field="rate")
        if abs(rate) > MAX_PLAUSIBLE_RATE:
            raise DomainError(
                _percentage_hint("rate", float(rate), MAX_PLAUSIBLE_RATE), field="rate"
            )

        raw_legs = raw.get("legs")
        if not isinstance(raw_legs, list) or not raw_legs:
            raise DomainError("legs must be a non-empty array", field="legs")
        if len(raw_legs) > MAX_LEGS:
            raise DomainError(
                f"a book holds at most {MAX_LEGS} legs, this one has {len(raw_legs)}. "
                "Split the positions across two books and add the results.",
                field="legs",
            )
        legs = [Leg.from_dict(item, where=f"legs[{i}]") for i, item in enumerate(raw_legs)]
        label = raw.get("label")
        return cls(
            spot=float(spot),
            rate=float(rate),
            legs=legs,
            label=label if isinstance(label, str) else None,
        )


def _leg_report(leg: Leg, index: int, spot: float, rate: float) -> dict[str, Any]:
    """Value and sensitivities for one leg, scaled by its quantity."""
    report: dict[str, Any] = {
        "index": index,
        "instrument": leg.instrument,
        "quantity": leg.quantity,
    }
    if leg.label is not None:
        report["label"] = leg.label

    if not leg.is_option:
        unit = spot
        report["unitValue"] = unit
        report["value"] = leg.quantity * unit
        report["greeks"] = {
            name: leg.quantity * value for name, value in _UNDERLYING_GREEKS.items()
        }
        report["greeksDefined"] = True
        return report

    inputs = leg.inputs(spot, rate)
    option = leg.option_type
    report["strike"] = leg.strike
    report["time"] = leg.time
    report["vol"] = leg.vol
    unit = price(inputs, option)
    report["unitValue"] = unit
    report["value"] = leg.quantity * unit
    report["intrinsic"] = leg.quantity * intrinsic(inputs, option)

    if inputs.is_degenerate:
        # Priced, but not differentiated. The payoff is a kink and the
        # derivative at a kink does not exist; reporting zero here is the
        # mistake this module is built to avoid.
        report["greeks"] = None
        report["greeksDefined"] = False
        report["undefinedBecause"] = (
            f"time={inputs.time:g} and vol={inputs.vol:g}: with no time left or no "
            "volatility the payoff is kinked at the strike and no derivative exists "
            "there. The value above is the limit and is exact."
        )
        return report

    report["greeks"] = {
        name: leg.quantity * fn(inputs, option) for name, fn in _OPTION_GREEKS.items()
    }
    report["greeksDefined"] = True
    return report


def value_and_greeks(book: Book, *, spot: float, vol_shift: float = 0.0) -> dict[str, Any]:
    """Total value and aggregate sensitivities of ``book`` under a shifted market.

    ``spot`` is passed rather than read off the book so scenarios reuse this
    without building a second book per cell.
    """
    total = 0.0
    aggregate = dict.fromkeys(GREEK_NAMES, 0.0)
    excluded: list[int] = []

    for index, leg in enumerate(book.legs):
        if not leg.is_option:
            total += leg.quantity * spot
            aggregate["delta"] += leg.quantity
            continue
        inputs = leg.inputs(spot, book.rate, vol_shift=vol_shift)
        total += leg.quantity * price(inputs, leg.option_type)
        if inputs.is_degenerate:
            excluded.append(index)
            continue
        for name, fn in _OPTION_GREEKS.items():
            aggregate[name] += leg.quantity * fn(inputs, leg.option_type)

    return {
        "value": total,
        "greeks": aggregate,
        "complete": not excluded,
        "excludedLegs": excluded,
    }


def summarise(book: Book) -> dict[str, Any]:
    """The headline numbers reported whenever a book is opened or amended."""
    options = [leg for leg in book.legs if leg.is_option]
    expiries = sorted({leg.time for leg in options if leg.time is not None})
    strikes = sorted({leg.strike for leg in options if leg.strike is not None})
    gross = sum(abs(leg.quantity) for leg in book.legs)
    net = sum(leg.quantity for leg in book.legs)
    return {
        "legCount": len(book.legs),
        "optionLegs": len(options),
        "underlyingLegs": len(book.legs) - len(options),
        "grossQuantity": gross,
        "netQuantity": net,
        "strikes": strikes,
        "expiries": expiries,
    }


# -- schemas --------------------------------------------------------------

_QUANTITY = {
    "type": "number",
    "description": (
        "Signed size. Positive is long, negative is short. Contracts for an "
        "option and units for the underlying; no contract multiplier is applied, "
        "so state the size in the units you mean."
    ),
}

#: A leg's volatility, which unlike everywhere else on this surface may be
#: zero. A zero-volatility leg is a deterministic claim rather than a mistake,
#: the scenario grid produces them by flooring a downward shift, and
#: :func:`value_and_greeks` handles them explicitly. Rejecting at the boundary
#: what the engine behind it supports would be an inconsistency, not a check.
_LEG_VOL = {
    "type": "number",
    "minimum": 0,
    "description": (
        "Annualised lognormal volatility for this leg, as a decimal fraction: "
        "20% is 0.2, not 20. Zero is accepted and makes the leg a deterministic "
        "claim with no sensitivities."
    ),
}

#: A leg is one closed object covering both forms, rather than a `oneOf` over
#: one schema per form. `oneOf` expresses the two shapes more precisely and
#: reports them far worse: a leg missing its strike fails every branch, and all
#: the validator can then say is that the leg matched no accepted form, which
#: names neither the field nor the fix. Since the point of this whole surface is
#: refusals a caller can act on, the variant rules are enforced in
#: :meth:`Leg.check` instead, where the message can name the missing field and
#: the instrument that wanted it. The object stays closed, so a misspelling is
#: still reported rather than dropped.
_LEG = {
    "type": "object",
    "properties": {
        "instrument": {
            "type": "string",
            "enum": ["call", "put", "underlying"],
            "description": (
                "`call` or `put` for an option on the book's underlying, which "
                "then needs its own strike, time and vol; `underlying` for a "
                "position in the underlying itself, which takes none of those."
            ),
        },
        "quantity": _QUANTITY,
        "strike": _STRIKE,
        "time": _TIME,
        "vol": _LEG_VOL,
        "carry": _CARRY,
        "label": {"type": "string", "description": "Your name for this leg, echoed back."},
    },
    "required": ["instrument", "quantity"],
    "additionalProperties": False,
    "description": (
        "Either an option on the book's underlying, with its own strike, expiry "
        "and volatility, or a position in the underlying itself."
    ),
}

_LEGS = {
    "type": "array",
    "items": _LEG,
    "minItems": 1,
    "maxItems": MAX_LEGS,
    "description": f"The positions in the book, at most {MAX_LEGS} of them.",
}

_HANDLE = {
    "type": "string",
    "minLength": 8,
    "description": (
        "The handle returned by open_position_book. It carries the book itself "
        "rather than pointing at stored state, so pass it back unchanged. It "
        "expires; if it does, open the book again."
    ),
}

_OPEN_SCHEMA = {
    "type": "object",
    "properties": {
        "spot": _SPOT,
        "rate": _RATE,
        "legs": _LEGS,
        "label": {"type": "string", "description": "Your name for the book, echoed back."},
    },
    "required": ["spot", "rate", "legs"],
    "additionalProperties": False,
}


def _shift_axis(
    raw: Any, *, name: str, default: tuple[float, ...], limit: float
) -> list[float]:
    """Read one axis of a scenario grid, or fall back to the default."""
    if raw is None:
        return list(default)
    if not isinstance(raw, list) or not raw:
        raise DomainError(f"{name} must be a non-empty array of numbers", field=name)
    values: list[float] = []
    for item in raw:
        if not isinstance(item, int | float) or isinstance(item, bool):
            raise DomainError(f"{name} must contain only numbers", field=name)
        if item != item or math.isinf(item):
            raise DomainError(f"{name} must contain only finite numbers", field=name)
        if abs(item) > limit:
            raise DomainError(
                f"{name} contains {item:g}, beyond the ±{limit:g} this tool accepts. "
                "Shifts are decimal fractions: ten percent is 0.1, not 10.",
                field=name,
            )
        values.append(float(item))
    return values


class BookTools:
    """The book tools, bound to one :class:`~abacus.handles.Minter`.

    A class rather than free functions because every one of these tools needs
    the same minter, and threading it through as an argument would put it in the
    tool schemas where a caller would see it and reasonably try to set it.
    """

    def __init__(self, minter: Minter | None = None) -> None:
        self.minter = minter if minter is not None else Minter()

    # -- handle plumbing --------------------------------------------------

    def _mint(self, book: Book) -> str:
        try:
            return self.minter.mint(book.to_dict())
        except HandleTooLarge as exc:
            raise ToolExecutionError(
                exc.message, kind="book_too_large", details=[{"size": exc.size, "limit": exc.limit}]
            ) from exc

    def _redeem(self, args: dict[str, Any]) -> Book:
        handle = args.get("handle")
        try:
            payload = self.minter.redeem(handle)
        except HandleError as exc:
            # A recoverable result rather than a JSON-RPC error: the request was
            # well formed and the caller can fix it by opening the book again,
            # which is exactly the distinction `isError` exists to draw.
            raise ToolExecutionError(
                exc.message,
                kind=exc.kind,
                details=[{"path": "/handle", "keyword": exc.kind, "message": exc.message}],
            ) from exc
        return Book.from_dict(payload)

    # -- tool bodies ------------------------------------------------------

    def open_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        book = Book.from_dict(args)
        handle = self._mint(book)
        valued = value_and_greeks(book, spot=book.spot)
        return {
            "handle": handle,
            "expiresInSeconds": self.minter.ttl_s,
            "book": book.to_dict(),
            "summary": summarise(book),
            "value": valued["value"],
        }

    def describe_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        book = self._redeem(args)
        return {
            "book": book.to_dict(),
            "summary": summarise(book),
            "value": value_and_greeks(book, spot=book.spot)["value"],
        }

    def amend_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        """Add legs to a book, drop legs from it, or move the market under it.

        Amending mints a new handle and leaves the old one redeemable until it
        expires. There is no way to revoke it — the server holds no record of
        either — and pretending otherwise would be the more dangerous lie, so
        the result says which handle it replaces and that the old one still
        works.
        """
        book = self._redeem(args)
        legs = list(book.legs)

        drop = args.get("removeLegs")
        if drop is not None:
            if not isinstance(drop, list):
                raise DomainError("removeLegs must be an array of leg indices", field="removeLegs")
            for item in drop:
                if not isinstance(item, int) or isinstance(item, bool):
                    raise DomainError(
                        "removeLegs must contain leg indices as integers", field="removeLegs"
                    )
            wanted = sorted({int(i) for i in drop})
            if len(wanted) != len(drop):
                raise DomainError("removeLegs must contain distinct indices", field="removeLegs")
            for index in wanted:
                if not 0 <= index < len(legs):
                    raise DomainError(
                        f"removeLegs names leg {index}, but this book has legs 0..{len(legs) - 1}",
                        field="removeLegs",
                    )
            legs = [leg for i, leg in enumerate(legs) if i not in set(wanted)]

        add = args.get("addLegs")
        if add is not None:
            if not isinstance(add, list):
                raise DomainError("addLegs must be an array of legs", field="addLegs")
            legs.extend(
                Leg.from_dict(item, where=f"addLegs[{i}]") for i, item in enumerate(add)
            )

        if not legs:
            raise DomainError(
                "that would empty the book. A book holds at least one leg; close it by "
                "dropping the handle instead.",
                field="removeLegs",
            )

        amended = Book.from_dict(
            {
                "spot": args.get("spot", book.spot),
                "rate": args.get("rate", book.rate),
                "legs": [leg.to_dict() for leg in legs],
                **({"label": book.label} if book.label is not None else {}),
            }
        )
        handle = self._mint(amended)
        return {
            "handle": handle,
            "expiresInSeconds": self.minter.ttl_s,
            "replaces": args["handle"],
            "priorHandleStillValid": True,
            "book": amended.to_dict(),
            "summary": summarise(amended),
            "value": value_and_greeks(amended, spot=amended.spot)["value"],
        }

    def greeks_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        book = self._redeem(args)
        aggregate = value_and_greeks(book, spot=book.spot)
        legs = [_leg_report(leg, i, book.spot, book.rate) for i, leg in enumerate(book.legs)]
        out: dict[str, Any] = {
            "spot": book.spot,
            "rate": book.rate,
            "value": aggregate["value"],
            "greeks": aggregate["greeks"],
            "complete": aggregate["complete"],
            "deltaEquivalentUnits": aggregate["greeks"]["delta"],
            "legs": legs,
        }
        if not aggregate["complete"]:
            out["excludedLegs"] = aggregate["excludedLegs"]
            out["incompleteBecause"] = (
                "legs "
                + ", ".join(str(i) for i in aggregate["excludedLegs"])
                + " have no defined sensitivities — they have expired or carry no "
                "volatility — so they are priced into the value above but contribute "
                "nothing to the aggregate Greeks. Treat the aggregate as covering the "
                "remaining legs only."
            )
        return out

    def scenarios_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        book = self._redeem(args)
        spot_shifts = _shift_axis(
            args.get("spotShifts"), name="spotShifts", default=DEFAULT_SPOT_SHIFTS, limit=0.99
        )
        vol_shifts = _shift_axis(
            args.get("volShifts"), name="volShifts", default=DEFAULT_VOL_SHIFTS, limit=5.0
        )
        cells = len(spot_shifts) * len(vol_shifts)
        if cells > MAX_GRID_CELLS:
            raise DomainError(
                f"{len(spot_shifts)} spot steps by {len(vol_shifts)} volatility steps is "
                f"{cells} cells, over the {MAX_GRID_CELLS}-cell limit. Coarsen an axis.",
                field="spotShifts",
            )

        base = value_and_greeks(book, spot=book.spot)
        rows: list[dict[str, Any]] = []
        for vol_shift in vol_shifts:
            row_cells: list[dict[str, Any]] = []
            for spot_shift in spot_shifts:
                spot = book.spot * (1.0 + spot_shift)
                point = value_and_greeks(book, spot=spot, vol_shift=vol_shift)
                cell: dict[str, Any] = {
                    "spot": spot,
                    "spotShift": spot_shift,
                    "value": point["value"],
                    "pnl": point["value"] - base["value"],
                    "delta": point["greeks"]["delta"] if point["complete"] else None,
                    "gamma": point["greeks"]["gamma"] if point["complete"] else None,
                }
                row_cells.append(cell)
            rows.append({"volShift": vol_shift, "cells": row_cells})

        values = [cell["value"] for row in rows for cell in row["cells"]]
        worst = min(values)
        best = max(values)
        return {
            "base": {"spot": book.spot, "value": base["value"], "complete": base["complete"]},
            "spotShifts": spot_shifts,
            "volShifts": vol_shifts,
            "rows": rows,
            "worstValue": worst,
            "worstPnl": worst - base["value"],
            "bestValue": best,
            "bestPnl": best - base["value"],
            "note": (
                "Each row is one volatility shift, in volatility points added to every "
                "option leg and floored at zero. Each cell within a row states the spot "
                "it was priced at. Spot shifts are relative: -0.1 is a ten percent fall."
            ),
        }

    # -- registration -----------------------------------------------------

    def register(self, registry: ToolRegistry) -> ToolRegistry:
        read_only = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}

        registry.register(
            "open_position_book",
            title="Open a position book",
            description=(
                "Assemble option and underlying positions over one underlying into a "
                "book, and return a handle for asking further questions of it. The "
                "handle carries the book itself rather than pointing at stored state, "
                "so it works across processes and expires on its own; pass it back "
                "unchanged. Volatilities and rates are decimal fractions and time is a "
                "year fraction."
            ),
            input_schema=_OPEN_SCHEMA,
            annotations=read_only,
        )(guard(self.open_payload))

        registry.register(
            "describe_position_book",
            title="Read a position book back",
            description=(
                "Return the positions a handle carries, with the book's total value. "
                "Use this to confirm a handle is still live and holds what you think "
                "it holds before acting on a number derived from it."
            ),
            input_schema={
                "type": "object",
                "properties": {"handle": _HANDLE},
                "required": ["handle"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.describe_payload))

        registry.register(
            "amend_position_book",
            title="Amend a position book",
            description=(
                "Add legs, remove legs by index, or move the spot and rate under an "
                "existing book, returning a new handle. The original handle keeps "
                "working until it expires, because the server holds no record of it "
                "and so cannot revoke it."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "handle": _HANDLE,
                    "addLegs": {"type": "array", "items": _LEG, "maxItems": MAX_LEGS},
                    "removeLegs": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "uniqueItems": True,
                        "description": (
                            "Indices of legs to drop, as reported by "
                            "describe_position_book. Applied before addLegs."
                        ),
                    },
                    "spot": _SPOT,
                    "rate": _RATE,
                },
                "required": ["handle"],
                "additionalProperties": False,
            },
            annotations={"readOnlyHint": True, "idempotentHint": False, "openWorldHint": False},
        )(guard(self.amend_payload))

        registry.register(
            "position_book_greeks",
            title="Aggregate risk of a position book",
            description=(
                "Total value and net sensitivities across a book, with the per-leg "
                "breakdown they were summed from. Legs with no defined derivative — "
                "expired, or carrying no volatility — are priced into the value but "
                "left out of the aggregate and named explicitly, so a book is never "
                "reported as flat when part of it simply has no delta."
            ),
            input_schema={
                "type": "object",
                "properties": {"handle": _HANDLE},
                "required": ["handle"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.greeks_payload))

        registry.register(
            "position_book_scenarios",
            title="Scenario grid over spot and volatility",
            description=(
                "Reprice a book across a grid of relative spot moves and absolute "
                "volatility shifts, reporting value, profit and loss against the "
                "current market, and net delta and gamma in each cell. Rows are "
                "labelled with their volatility shift and cells with their spot, so "
                "the two axes cannot be read for one another."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "handle": _HANDLE,
                    "spotShifts": {
                        "type": "array",
                        "items": {"type": "number", "minimum": -0.99, "maximum": 0.99},
                        "minItems": 1,
                        "description": (
                            "Relative moves in the underlying, as decimal fractions: "
                            "-0.1 is a ten percent fall. Defaults to ±10% and ±5%."
                        ),
                    },
                    "volShifts": {
                        "type": "array",
                        "items": {"type": "number", "minimum": -5, "maximum": 5},
                        "minItems": 1,
                        "description": (
                            "Volatility points added to every option leg, floored at "
                            "zero: 0.05 is five volatility points. Defaults to ±5."
                        ),
                    },
                },
                "required": ["handle"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.scenarios_payload))

        return registry


def register(registry: ToolRegistry, *, minter: Minter | None = None) -> ToolRegistry:
    """Add the position book tools to ``registry``."""
    return BookTools(minter).register(registry)


__all__ = [
    "DEFAULT_SPOT_SHIFTS",
    "DEFAULT_VOL_SHIFTS",
    "GREEK_NAMES",
    "MAX_GRID_CELLS",
    "MAX_LEGS",
    "Book",
    "BookTools",
    "Leg",
    "register",
    "summarise",
    "value_and_greeks",
]
