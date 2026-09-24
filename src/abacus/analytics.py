"""Turning validated arguments into option analytics.

Two layers sit between a tool call and the numbers. The schema decides whether
the arguments are the right shape; the checks here decide whether they describe
a market that could exist. A schema can say ``vol`` is a number and not
negative. It cannot say that ``vol: 20`` is almost certainly twenty *percent*
written the wrong way, and pricing it as two thousand percent produces a number
that is arithmetically correct and completely wrong.

That distinction is the point of this module. A model that hands over a
percentage gets told so and can fix it; without the check it would get a price,
and nothing downstream would ever know.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from moneyness import (
    Inputs,
    OptionType,
    Quote,
    bounds,
    charm,
    colour,
    d1_d2,
    delta,
    dual_delta,
    dual_gamma,
    forward,
    gamma,
    intrinsic,
    parity_gap,
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

from .tools import DomainError, ToolRegistry

#: Above this, a volatility is far more likely to be a percentage than a real
#: quote. Five is 500% annualised; the highest single-name implied volatilities
#: seen in a crisis sit well beneath it, so the rule costs nothing real and
#: catches the commonest unit mistake there is.
MAX_PLAUSIBLE_VOL = 5.0

#: Same reasoning for rates. 1.0 is 100% continuously compounded.
MAX_PLAUSIBLE_RATE = 1.0

#: A year fraction beyond this is almost certainly a count of days or months.
#: The longest listed options run about ten years.
MAX_PLAUSIBLE_YEARS = 50.0


def _percentage_hint(name: str, value: float, limit: float) -> str:
    return (
        f"{name}={value:g} is out of range; it looks like a percentage. "
        f"This field is a decimal fraction, so {value:g}% is {value / 100:g}. "
        f"Values above {limit:g} are rejected as implausible."
    )


def check_market(args: dict[str, Any]) -> None:
    """Reject arguments that parse but could not describe a real market."""
    vol = args.get("vol")
    if isinstance(vol, int | float) and vol > MAX_PLAUSIBLE_VOL:
        raise DomainError(_percentage_hint("vol", float(vol), MAX_PLAUSIBLE_VOL), field="vol")

    rate = args.get("rate")
    if isinstance(rate, int | float) and abs(rate) > MAX_PLAUSIBLE_RATE:
        raise DomainError(_percentage_hint("rate", float(rate), MAX_PLAUSIBLE_RATE), field="rate")

    carry = args.get("carry")
    if isinstance(carry, int | float) and abs(carry) > MAX_PLAUSIBLE_RATE:
        raise DomainError(
            _percentage_hint("carry", float(carry), MAX_PLAUSIBLE_RATE), field="carry"
        )

    time = args.get("time")
    if isinstance(time, int | float) and time > MAX_PLAUSIBLE_YEARS:
        raise DomainError(
            f"time={time:g} is out of range; it is a year fraction, not a count of days. "
            f"Thirty days is roughly 0.082. Values above {MAX_PLAUSIBLE_YEARS:g} are rejected.",
            field="time",
        )


def option_type(args: dict[str, Any]) -> OptionType:
    """Read the option type. The schema has already limited it to the two values."""
    return OptionType(args["type"])


def build_inputs(args: dict[str, Any]) -> Inputs:
    """Build a :class:`moneyness.Inputs` from checked arguments."""
    check_market(args)
    return Inputs(
        spot=float(args["spot"]),
        strike=float(args["strike"]),
        time=float(args["time"]),
        rate=float(args["rate"]),
        vol=float(args["vol"]),
        carry=None if args.get("carry") is None else float(args["carry"]),
    )


def build_quote(args: dict[str, Any]) -> Quote:
    """Build a :class:`moneyness.Quote`, which carries a price instead of a volatility."""
    check_market(args)
    return Quote(
        spot=float(args["spot"]),
        strike=float(args["strike"]),
        time=float(args["time"]),
        rate=float(args["rate"]),
        price=float(args["price"]),
        carry=None if args.get("carry") is None else float(args["carry"]),
    )


# -- schemas --------------------------------------------------------------
#
# Descriptions carry the units. A model reads these and nothing else before
# composing a call, so "annualised, as a decimal fraction" in the description is
# what prevents the percentage mistake the domain checks exist to catch.

_SPOT = {
    "type": "number",
    "exclusiveMinimum": 0,
    "description": "Price of the underlying now. Must be positive.",
}
_STRIKE = {
    "type": "number",
    "exclusiveMinimum": 0,
    "description": "Exercise price of the option. Must be positive.",
}
_TIME = {
    "type": "number",
    "minimum": 0,
    "description": (
        "Year fraction remaining to expiry, not a count of days. "
        "Thirty days is roughly 0.082; one year is 1.0."
    ),
}
_RATE = {
    "type": "number",
    "description": (
        "Continuously compounded risk-free rate, annualised, as a decimal "
        "fraction: 4.5% is 0.045."
    ),
}
_VOL = {
    "type": "number",
    "exclusiveMinimum": 0,
    "description": (
        "Annualised lognormal volatility, as a decimal fraction: 20% is 0.2, "
        "not 20."
    ),
}
_CARRY = {
    "type": "number",
    "description": (
        "Cost of carry, annualised, as a decimal fraction. Defaults to the "
        "rate, which is the non-dividend-paying share case. Use rate minus the "
        "dividend yield for a paying share, and zero for a future."
    ),
}
_TYPE = {
    "type": "string",
    "enum": ["call", "put"],
    "description": "Which side of the strike the holder receives.",
}


def _option_schema(*, with_type: bool = True) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "spot": _SPOT,
        "strike": _STRIKE,
        "time": _TIME,
        "rate": _RATE,
        "vol": _VOL,
        "carry": _CARRY,
    }
    required = ["spot", "strike", "time", "rate", "vol"]
    if with_type:
        properties["type"] = _TYPE
        required.append("type")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        # Closed rather than open, so a misspelled field is reported instead of
        # being silently dropped and priced with a default.
        "additionalProperties": False,
    }


_PRICE_OUTPUT = {
    "type": "object",
    "properties": {
        "price": {"type": "number", "description": "The option's present value."},
        "forward": {"type": "number", "description": "The forward the option is written on."},
        "intrinsic": {"type": "number", "description": "Discounted intrinsic value."},
        "timeValue": {"type": "number", "description": "Price less intrinsic value."},
        "d1": {"type": "number"},
        "d2": {"type": "number"},
        "model": {"type": "string"},
    },
    "required": ["price", "forward", "intrinsic", "timeValue", "model"],
}

_GREEKS_OUTPUT = {
    "type": "object",
    "properties": {
        "delta": {"type": "number", "description": "Sensitivity to spot."},
        "gamma": {"type": "number", "description": "Sensitivity of delta to spot."},
        "vega": {"type": "number", "description": "Sensitivity to volatility, per 1.00 of vol."},
        "theta": {"type": "number", "description": "Sensitivity to time, per year."},
        "rho": {"type": "number", "description": "Sensitivity to the rate."},
        "vanna": {"type": "number"},
        "volga": {"type": "number"},
        "charm": {"type": "number"},
        "speed": {"type": "number"},
        "zomma": {"type": "number"},
        "colour": {"type": "number"},
        "veta": {"type": "number"},
        "dualDelta": {"type": "number", "description": "Sensitivity to the strike."},
        "dualGamma": {"type": "number"},
        "rhoCarry": {"type": "number", "description": "Sensitivity to the cost of carry."},
    },
    "required": ["delta", "gamma", "vega", "theta", "rho"],
}


def price_payload(args: dict[str, Any]) -> dict[str, Any]:
    inputs = build_inputs(args)
    option = option_type(args)
    value = price(inputs, option)
    floor = intrinsic(inputs, option)
    payload = {
        "price": value,
        "forward": forward(inputs),
        "intrinsic": floor,
        "timeValue": value - floor,
        "model": "generalised Black-Scholes-Merton",
    }
    # At expiry, or with zero volatility, spot or strike, the price is still
    # well defined — it is the limit, and the library returns it — but the two
    # standardised log-moneyness arguments are not. They are omitted rather than
    # filled with an infinity or a zero, both of which would read as a real
    # number to whatever consumed them. The output schema does not require them
    # for exactly this reason.
    if not inputs.is_degenerate:
        first, second = d1_d2(inputs)
        payload["d1"] = first
        payload["d2"] = second
    return payload


def greeks_payload(args: dict[str, Any]) -> dict[str, Any]:
    inputs = build_inputs(args)
    option = option_type(args)
    if inputs.is_degenerate:
        # Unlike the price, which has a limit at expiry, the sensitivities do
        # not exist here: the payoff is kinked at the strike and the derivative
        # is undefined at the kink. Refusing and saying what is still available
        # beats returning an infinity, a zero, or a stack trace — each of which
        # would be read downstream as a real sensitivity.
        raise DomainError(
            "Greeks are undefined when time, volatility, spot or strike is zero: "
            "the payoff is kinked at the strike and the derivative does not exist "
            "there. The price itself is still defined as a limit and is available "
            "from price_european_option.",
            field="time" if inputs.time == 0 else "vol",
        )
    return {
        "delta": delta(inputs, option),
        "gamma": gamma(inputs),
        "vega": vega(inputs),
        "theta": theta(inputs, option),
        "rho": rho(inputs, option),
        "vanna": vanna(inputs),
        "volga": volga(inputs),
        "charm": charm(inputs, option),
        "speed": speed(inputs),
        "zomma": zomma(inputs),
        "colour": colour(inputs),
        "veta": veta(inputs),
        "dualDelta": dual_delta(inputs, option),
        "dualGamma": dual_gamma(inputs),
        "rhoCarry": rho_carry(inputs, option),
    }


def analytics_payload(args: dict[str, Any]) -> dict[str, Any]:
    """Price and Greeks together.

    Offered as one tool as well as two because a model asking for a price
    usually wants the sensitivities in the same breath, and a round trip it does
    not have to make is a round trip it cannot get wrong.
    """
    return {**price_payload(args), "greeks": greeks_payload(args)}


def parity_payload(args: dict[str, Any]) -> dict[str, Any]:
    inputs = build_inputs(args)
    call_price = price(inputs, OptionType.CALL)
    put_price = price(inputs, OptionType.PUT)
    gap = parity_gap(inputs)
    return {
        "call": call_price,
        "put": put_price,
        "forward": forward(inputs),
        "parityGap": gap,
        "consistent": abs(gap) < 1e-9,
    }


def bounds_payload(args: dict[str, Any]) -> dict[str, Any]:
    quote = build_quote(args)
    option = option_type(args)
    limits = bounds(quote, option)
    observed = float(args["price"])
    return {
        "lower": limits.lower,
        "upper": limits.upper,
        "price": observed,
        "attainable": limits.lower <= observed <= limits.upper,
    }


def guard(handler: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
    """Turn a library domain complaint into a recoverable tool error.

    ``moneyness`` raises :class:`ValueError` exactly when inputs fall outside
    the domain of what it is being asked to compute — an expired option has no
    implied volatility, a kinked payoff has no derivative — and its messages are
    written to explain which. Left alone those would surface as ``-32603``,
    which tells the caller only that something broke inside the server.

    Converting them here means a caller gets the library's own explanation in
    the same shape as every other refusal. It also means the checks stay in one
    place: the tools are not obliged to re-derive the domain of each function
    they call, and a tool added later cannot forget to.
    """

    def wrapped(args: dict[str, Any]) -> Any:
        try:
            return handler(args)
        except ValueError as exc:
            raise DomainError(str(exc)) from exc
        except ArithmeticError as exc:
            # ZeroDivisionError, OverflowError and friends. A numerical routine
            # divides by a time step or a volatility that a degenerate market
            # has made zero. The exception text ("float division by zero") says
            # nothing a caller could act on, so it is replaced rather than
            # passed through — but it is still a statement about the inputs, and
            # it must not escape as an internal error.
            raise DomainError(
                "this market is degenerate for the method requested: a quantity the "
                "calculation divides by is zero. That usually means time, volatility, "
                "spot or strike is zero, or so close to it that it underflowed. "
                f"({type(exc).__name__}: {exc})"
            ) from exc

    return wrapped


def register(registry: ToolRegistry) -> ToolRegistry:
    """Add the pricing and Greek tools to ``registry``."""

    registry.register(
        "price_european_option",
        title="European option price",
        description=(
            "Price a European option under the generalised Black-Scholes-Merton "
            "model, and report the forward, the discounted intrinsic value and "
            "the time value alongside it. Volatility and rates are decimal "
            "fractions and time is a year fraction."
        ),
        input_schema=_option_schema(),
        output_schema=_PRICE_OUTPUT,
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )(guard(price_payload))

    registry.register(
        "european_option_greeks",
        title="European option Greeks",
        description=(
            "Analytic sensitivities of a European option: delta, gamma, vega, "
            "theta and rho, together with the second- and third-order Greeks "
            "(vanna, volga, charm, speed, zomma, colour, veta) and the "
            "sensitivities to strike and carry. Vega is per 1.00 of volatility "
            "and theta is per year."
        ),
        input_schema=_option_schema(),
        output_schema=_GREEKS_OUTPUT,
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )(guard(greeks_payload))

    registry.register(
        "european_option_analytics",
        title="European option price and Greeks",
        description=(
            "Price and the full Greek set in one call, for when both are wanted "
            "and a second round trip is not."
        ),
        input_schema=_option_schema(),
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )(guard(analytics_payload))

    registry.register(
        "put_call_parity",
        title="Put-call parity check",
        description=(
            "Price both sides of a strike and report the put-call parity "
            "residual. The residual is zero for a consistent pair, so a "
            "non-zero value is a direct read on accumulated rounding error."
        ),
        input_schema=_option_schema(with_type=False),
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )(guard(parity_payload))

    quote_schema = _option_schema()
    quote_properties = dict(quote_schema["properties"])
    del quote_properties["vol"]
    quote_properties["price"] = {
        "type": "number",
        "minimum": 0,
        "description": "Observed market price of the option.",
    }
    registry.register(
        "option_price_bounds",
        title="Attainable price range",
        description=(
            "The range of prices attainable by some non-negative volatility, "
            "and whether an observed price falls inside it. A price outside the "
            "range has no implied volatility, so this is worth checking before "
            "asking for one."
        ),
        input_schema={
            "type": "object",
            "properties": quote_properties,
            "required": ["spot", "strike", "time", "rate", "price", "type"],
            "additionalProperties": False,
        },
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )(guard(bounds_payload))

    return registry


def default_registry() -> ToolRegistry:
    """A registry holding every tool this server exposes."""
    from . import american, book, curves, execution, risk, validation, vol

    built = register(ToolRegistry())
    for module in (vol, american, book, risk, curves, execution, validation):
        built = module.register(built)
    return built
