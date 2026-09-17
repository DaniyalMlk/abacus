"""American exercise: lattices, closed-form approximations, and the boundary.

An American price is the output of a numerical method rather than a formula, so
it carries an error the caller cannot see from the number itself. A lattice
price at 64 layers and the same price at 4096 layers are different numbers, and
neither is wrong exactly — one is simply further from the limit. A tool that
returns a bare float invites a caller to treat a discretisation artefact as a
market fact.

So nothing here returns a price alone. Every American valuation carries the
method that produced it, the resolution it ran at, the early-exercise premium
over the European price, and — when asked — how much the answer is still moving
as the grid refines. A caller can then tell a converged number from one that
needs more steps, which is not something it could work out from the number.

The approximations are labelled as approximations for the same reason. They are
fast and close, and they are not the limit of anything.
"""

from __future__ import annotations

from typing import Any

from moneyness import (
    Exercise,
    Lattice,
    OptionType,
    bjerksund_stensland,
    bjerksund_stensland_2002,
    boundary,
    min_steps,
    price,
    price_lattice,
    trigger_price,
)

from .analytics import _CARRY, _RATE, _SPOT, _STRIKE, _TIME, _TYPE, _VOL, build_inputs, guard
from .tools import DomainError, ToolRegistry

#: Layer counts used for the convergence report. Doubling makes the successive
#: differences directly comparable, which is what turns them into an error
#: estimate rather than a list of numbers.
CONVERGENCE_STEPS = (64, 128, 256, 512, 1024)

#: Cap on the requested resolution. A lattice is quadratic in its layer count,
#: so an unbounded step count is a way to make the server compute for a very
#: long time on request.
MAX_STEPS = 8192


def _lattice_of(args: dict[str, Any]) -> Lattice:
    return Lattice(args.get("lattice", "crr"))


def lattice_payload(args: dict[str, Any]) -> dict[str, Any]:
    inputs = build_inputs(args)
    option = OptionType(args["type"])
    lattice = _lattice_of(args)
    steps = int(args.get("steps", 512))

    floor = min_steps(inputs, lattice)
    if steps < floor:
        # Below this the branch probabilities leave [0, 1] and the lattice stops
        # being a probability model at all. It would still produce a number.
        raise DomainError(
            f"steps={steps} is too few for this market on a {lattice.value} lattice: "
            f"below {floor} layers the branch probabilities leave [0, 1] and the "
            "tree is no longer a probability model. Use at least that many.",
            field="steps",
        )

    valued = price_lattice(
        inputs, option, steps=steps, lattice=lattice, exercise=Exercise.AMERICAN
    )
    payload: dict[str, Any] = {
        "price": valued.value,
        # Two European prices, because they are not the same number and saying
        # which is which matters. The analytic one is exact. The lattice one
        # carries the same discretisation error as the American price above it.
        #
        # The premium is the *lattice-internal* difference, and deliberately so:
        # the discretisation error is common to both legs and largely cancels
        # when they are subtracted, making it a better estimate of the premium
        # than differencing against the analytic price would give. The cost is
        # that price - europeanPrice does not reproduce it, so both are reported
        # rather than leaving a reader to find the discrepancy and distrust all
        # three.
        "europeanPrice": price(inputs, option),
        "europeanPriceOnLattice": valued.value - valued.early_exercise_premium,
        "earlyExercisePremium": valued.early_exercise_premium,
        "premiumNote": (
            "The premium is the difference between the American and European "
            "values on the same lattice, where the discretisation error largely "
            "cancels. It will not equal price minus europeanPrice, which "
            "differences a lattice value against an analytic one and so retains "
            "that error."
        ),
        "method": {
            "name": "binomial lattice" if lattice is not Lattice.TRINOMIAL else "trinomial lattice",
            "lattice": valued.lattice.value,
            "steps": valued.steps,
            "minimumSteps": floor,
            "exercise": valued.exercise.value,
            "exact": False,
        },
    }
    if args.get("convergence"):
        payload["convergence"] = _convergence(inputs, option, lattice, floor)
    return payload


def _convergence(
    inputs: Any, option: OptionType, lattice: Lattice, floor: int
) -> dict[str, Any]:
    """Price at a doubling ladder of resolutions and report how much it moves.

    The last successive difference is the honest thing to quote as an error
    estimate. It is not a bound — nothing here proves one — and it is described
    as an estimate rather than dressed up as an accuracy claim.
    """
    ladder: list[dict[str, Any]] = []
    previous: float | None = None
    for steps in CONVERGENCE_STEPS:
        if steps < floor:
            continue
        value = price_lattice(
            inputs, option, steps=steps, lattice=lattice, exercise=Exercise.AMERICAN
        ).value
        row: dict[str, Any] = {"steps": steps, "price": value}
        if previous is not None:
            row["changeFromPrevious"] = value - previous
        ladder.append(row)
        previous = value

    report: dict[str, Any] = {"ladder": ladder}
    if len(ladder) >= 2:
        last = ladder[-1].get("changeFromPrevious")
        if last is not None:
            report["estimatedError"] = abs(last)
            report["note"] = (
                "The estimate is the size of the last change as the layer count "
                "doubled. It indicates how much the answer is still moving; it is "
                "not a proven bound on the distance to the limit."
            )
    return report


def approximation_payload(args: dict[str, Any]) -> dict[str, Any]:
    inputs = build_inputs(args)
    option = OptionType(args["type"])
    european = price(inputs, option)
    value = bjerksund_stensland_2002(inputs, option)
    return {
        "price": value,
        "europeanPrice": european,
        "earlyExercisePremium": value - european,
        "triggerPrice": trigger_price(inputs, option),
        "bjerksundStensland1993": bjerksund_stensland(inputs, option),
        "method": {
            "name": "Bjerksund-Stensland 2002",
            "exact": False,
            "note": (
                "A closed-form approximation and a lower bound, not the limit of a "
                "refinement. The reported price is the best of the European price, "
                "the 1993 formula and the 2002 formula, because the 2002 formula is "
                "not uniformly sharper than the 1993 one and can fall below the "
                "European price. For a converging answer use price_american_lattice."
            ),
        },
    }


def boundary_payload(args: dict[str, Any]) -> dict[str, Any]:
    inputs = build_inputs(args)
    option = OptionType(args["type"])
    lattice = _lattice_of(args)
    steps = int(args.get("steps", 512))
    floor = min_steps(inputs, lattice)
    if steps < floor:
        raise DomainError(
            f"steps={steps} is below the {floor} this market needs on a "
            f"{lattice.value} lattice.",
            field="steps",
        )

    series = boundary(inputs, option, steps=steps, lattice=lattice)
    points = int(args.get("points", 25))
    # Thin a 512-layer boundary down to something readable. Reporting every
    # layer would bury the shape in numbers, and the shape is the point.
    if len(series) > points:
        stride = len(series) / points
        sampled = [series[min(len(series) - 1, int(i * stride))] for i in range(points)]
    else:
        sampled = list(series)

    method: dict[str, Any] = {"lattice": lattice.value, "steps": steps, "exact": False}
    if lattice is not Lattice.TRINOMIAL:
        # Worth stating, because it looks like a bug and is not. A
        # Cox-Ross-Rubinstein layer n holds the nodes S*u^(2j-n), so the parity
        # of the exponent flips with every layer and consecutive layers sample
        # two interleaved grids. Each parity subsequence is monotone; the
        # interleaving of them is not. The true boundary *is* monotone in time.
        method["monotoneReadout"] = False
        method["note"] = (
            "On a binomial lattice the boundary read-out alternates between two "
            "values one node apart, because consecutive layers sample two "
            "interleaved node grids of opposite parity. The true boundary is "
            "monotone in time; this artefact is the discretisation, not the "
            "option. Use lattice='trinomial' when the boundary itself is the "
            "object of interest, since every layer there holds every node and "
            "the read-out comes out monotone."
        )
    else:
        method["monotoneReadout"] = True

    return {
        "boundary": [{"time": t, "spot": s} for t, s in sampled],
        "immediateTrigger": trigger_price(inputs, option),
        "method": method,
        "note": (
            "The early-exercise boundary: the spot at which exercising now is worth "
            "at least as much as holding on. Times run forward from now and end at "
            "expiry, so the last point is at expiry. A put is exercised below its "
            "boundary and a call above it. Resolution is the node spacing, so this "
            "is a picture of where the exercise region lies rather than a "
            "high-precision curve. An American call that is never worth exercising "
            "early returns only the expiry point, which is the correct answer "
            "rather than a truncated one."
        ),
    }


# -- schemas --------------------------------------------------------------

_LATTICE = {
    "type": "string",
    "enum": ["crr", "jarrow-rudd", "trinomial"],
    "description": (
        "Which discretisation to build. Cox-Ross-Rubinstein by default; the "
        "trinomial tree converges more smoothly at a higher cost per layer."
    ),
}

_STEPS = {
    "type": "integer",
    "minimum": 1,
    "maximum": MAX_STEPS,
    "description": (
        "Layers in the lattice. More is closer to the limit and costs "
        "quadratically more. 512 by default."
    ),
}


def _american_schema(**extra: Any) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "spot": _SPOT,
        "strike": _STRIKE,
        "time": _TIME,
        "rate": _RATE,
        "vol": _VOL,
        "carry": _CARRY,
        "type": _TYPE,
        **extra,
    }
    return {
        "type": "object",
        "properties": properties,
        "required": ["spot", "strike", "time", "rate", "vol", "type"],
        "additionalProperties": False,
    }


def register(registry: ToolRegistry) -> ToolRegistry:
    """Add the American exercise tools to ``registry``."""

    registry.register(
        "price_american_lattice",
        title="American option price on a lattice",
        description=(
            "Price an American option on a binomial or trinomial lattice. Returns "
            "the price with the method and resolution that produced it, the "
            "European price, and the early-exercise premium between them. Set "
            "convergence to also price at a doubling ladder of resolutions and "
            "see how much the answer is still moving — an American price is the "
            "output of a numerical method, and how converged it is cannot be read "
            "off the number."
        ),
        input_schema=_american_schema(
            steps=_STEPS,
            lattice=_LATTICE,
            convergence={
                "type": "boolean",
                "description": "Also report prices at 64 to 1024 layers and the error estimate.",
            },
        ),
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )(guard(lattice_payload))

    registry.register(
        "price_american_closed_form",
        title="American option price, closed form",
        description=(
            "Price an American option by the Bjerksund-Stensland 2002 "
            "approximation: fast, close, and explicitly not the limit of a "
            "refinement. Reports the trigger price it exercises at and the 1993 "
            "formula beside it. Use price_american_lattice when the error matters."
        ),
        input_schema=_american_schema(),
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )(guard(approximation_payload))

    registry.register(
        "american_exercise_boundary",
        title="Early-exercise boundary",
        description=(
            "The early-exercise boundary of an American option, read off the "
            "lattice as it unwinds: the spot at which exercising now is worth at "
            "least as much as holding on, at each time to expiry. This is what "
            "makes an American option different from a European one, and it is "
            "often more useful than the price."
        ),
        input_schema=_american_schema(
            steps=_STEPS,
            lattice=_LATTICE,
            points={
                "type": "integer",
                "minimum": 2,
                "maximum": 200,
                "description": "How many points to report. The series is thinned to this.",
            },
        ),
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )(guard(boundary_payload))

    return registry


__all__ = ["CONVERGENCE_STEPS", "MAX_STEPS", "register"]
