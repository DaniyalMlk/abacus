"""Execution: what a trade cost, and how it should have been spread out.

Everything else this server exposes prices a position or sizes its risk, and
each of those answers is about a price nobody traded at. This module is about
the gap between that price and the one that happened — after the fact, by
decomposing an order's shortfall, and before it, by laying out a schedule. The
numbers come from ``slippage`` rather than from here.

Four decisions shape it.

**The total shortfall is the least useful number in the result.** A trade that
cost 87 basis points tells you to feel bad; the same trade split into 20 of
delay, 51 of trading, 15 of opportunity and 1 of commission tells you which
thing to change, and the four have nothing to do with each other. Delay is the
price moving between the decision and the order reaching the market, and is
fixed by shortening that gap. Trading is the cost of the order's own footprint,
and is fixed by spreading it out. Opportunity is the part that never got done.
Commission is a contract. So every decomposition reports the components, and the
total is reported alongside them rather than instead of them.

**Currency and basis points are both reported, always.** A cost in currency
cannot be compared across two orders of different size, and a cost in basis
points cannot be added up across a book. Callers need both, and the conversion
needs a notional they can see, so the paper notional the basis points are taken
against is in the result too.

**The delay basis is an argument with no safe default, so it is reported.**
Whether the delay component is charged on the quantity that was *ordered* or the
quantity that was *executed* is a convention, and shops differ. Measured on a
900-of-1000 buy filled at an average of 100.57, with a decision price of 99.80,
an arrival of 100.00 and a close of 101.50: the order basis gives a delay of 200
and an opportunity of 150; the executed basis gives 180 and 170. The *total* is
869 either way. The basis moves cost between two components and never changes
the sum, which is exactly the kind of difference that survives a sanity check on
the total and then makes two desks' reports disagree by 20 basis points. The
result names the basis it used.

**Fill timestamps are not required, because the decomposition does not read
them.** ``slippage.Order`` carries them because the rest of that library — the
volume curves, the participation-rate work — needs them. The shortfall
decomposition reads only the quantities, prices and commissions: feeding the same
fills at one-minute and at six-hour spacings gives an identical breakdown to
every digit. Requiring a model to invent ISO timestamps to satisfy a field
nothing consumes would be asking for fabricated data, so this tool takes fills
without them and says here why it can.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

from slippage import (
    DelayBasis as SlippageDelayBasis,
)
from slippage import (
    ExecutionProblem,
    Fill,
    LinearImpact,
    Order,
    Side,
    Trajectory,
    efficient_frontier,
    half_life_sensitivity,
    implementation_shortfall,
    linear_trajectory,
    schedule_moments,
)

from .analytics import guard
from .tools import DomainError, ToolExecutionError, ToolRegistry

#: Most fills one order may carry. A decomposition reads each fill once, so the
#: cost is linear and the cap is not about time: it is that a caller pasting a
#: full day's tape into one argument has almost certainly meant to aggregate it
#: first, and a refusal naming the count is more useful than a correct answer to
#: a question nobody asked.
MAX_FILLS = 2_000

#: Most periods an execution schedule may be cut into. The closed form is
#: evaluated once per period, so this is a bound on the size of the result rather
#: than on the work: a thousand-row trajectory is past the point where a model
#: can read it, and the shape of the curve is visible at twenty.
MAX_PERIODS = 500

#: Most points a frontier may hold. Each one is a full trajectory solve.
MAX_FRONTIER_POINTS = 50

#: A stand-in origin for fills that arrive without timestamps. Nothing in the
#: decomposition reads these; see the module docstring. They are spaced by a
#: minute so that the order they were given in survives, which keeps the
#: ``slippage`` invariant that fills are chronological.
_EPOCH = datetime(2000, 1, 1, 0, 0, 0)


# -- schema fragments --------------------------------------------------------

_SIDE = {
    "type": "string",
    "enum": ["buy", "sell"],
    "description": (
        "Direction of the order. This is not cosmetic: a price rise between the "
        "decision and the fill is a cost to a buyer and a gain to a seller, and "
        "every component below changes sign with it."
    ),
}

_FILL = {
    "type": "object",
    "properties": {
        "quantity": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Shares or contracts in this fill, unsigned. The side carries the sign.",
        },
        "price": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Price this fill traded at.",
        },
        "commission": {
            "type": "number",
            "minimum": 0,
            "description": (
                "Commission charged on this fill, in currency. Defaults to zero. "
                "This is an explicit cost and is reported separately from the "
                "implicit ones, because it is a contract rather than a market "
                "outcome."
            ),
        },
    },
    "required": ["quantity", "price"],
    "additionalProperties": False,
}

_FILLS = {
    "type": "array",
    "minItems": 1,
    "maxItems": MAX_FILLS,
    "items": _FILL,
    "description": (
        "The fills, in the order they happened. Timestamps are not needed: the "
        "decomposition reads only the quantities, prices and commissions, and "
        "gives the same answer whatever the spacing."
    ),
}

_IMPACT = {
    "gamma": {
        "type": "number",
        "minimum": 0,
        "description": (
            "Permanent impact per unit traded, in price per share. The part of "
            "the move the trade leaves behind. It does not enter the optimal "
            "schedule at all — permanent impact costs the same however the "
            "order is spread — which is why the trajectory below is insensitive "
            "to it and the total cost is not."
        ),
    },
    "eta": {
        "type": "number",
        "exclusiveMinimum": 0,
        "description": (
            "Temporary impact per unit of trading *rate*, in price per share per "
            "unit time. This is the term the schedule trades against risk. It "
            "must be positive: at zero the cost of trading instantly is zero and "
            "the problem has no interior solution."
        ),
    },
    "epsilon": {
        "type": "number",
        "minimum": 0,
        "description": (
            "Fixed cost per unit traded — half the spread, plus fees quoted per "
            "share. Defaults to zero. It is paid on the whole quantity whatever "
            "the schedule, so like gamma it shifts the cost level without moving "
            "the optimum."
        ),
    },
}

_PROBLEM = {
    "quantity": {
        "type": "number",
        "exclusiveMinimum": 0,
        "description": "Total quantity to execute, unsigned.",
    },
    "horizon": {
        "type": "number",
        "exclusiveMinimum": 0,
        "description": (
            "Length of the trading window, in the same time unit as the "
            "volatility. If volatility is annualised, a horizon of 5 is five "
            "years, not five days — pass the horizon in years too, or quote the "
            "volatility per day."
        ),
    },
    "periods": {
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_PERIODS,
        "description": "Number of equal slices the window is cut into.",
    },
    "volatility": {
        "type": "number",
        "exclusiveMinimum": 0,
        "description": (
            "Price volatility in currency per unit of the square root of the same "
            "time unit as the horizon. Not a percentage: this is the absolute "
            "price move, because impact is quoted in price per share and the two "
            "have to be in one currency to be traded off."
        ),
    },
    "impact": {
        "type": "object",
        "properties": _IMPACT,
        "required": ["eta"],
        "additionalProperties": False,
        "description": "The linear impact model the schedule is optimised against.",
    },
}

_RISK_AVERSION = {
    "type": "number",
    "minimum": 0,
    "description": (
        "Aversion to the variance of the cost, in inverse currency. Zero is risk "
        "neutrality and gives a straight line — an equal slice each period, which "
        "is TWAP. Raising it front-loads the schedule, which pays more impact to "
        "spend less time exposed to the price moving."
    ),
}


# -- input parsing -----------------------------------------------------------


def _side(value: str) -> Side:
    return Side.BUY if value == "buy" else Side.SELL


def _order(args: dict[str, Any]) -> Order:
    """Build an order from the arguments, checking what the schema cannot."""
    quantity = float(args["quantity"])
    fills = args["fills"]

    filled = math.fsum(float(fill["quantity"]) for fill in fills)
    if filled > quantity * (1 + 1e-9):
        raise DomainError(
            f"the fills total {filled:g} against an order for {quantity:g}. An order "
            "cannot be overfilled; if these fills belong to more than one order, "
            "decompose them one order at a time, because the opportunity cost is "
            "measured against a single order's unfilled remainder.",
            field="fills",
        )

    built = tuple(
        Fill(
            timestamp=_EPOCH + timedelta(minutes=index),
            quantity=float(fill["quantity"]),
            price=float(fill["price"]),
            commission=float(fill.get("commission", 0.0)),
        )
        for index, fill in enumerate(fills)
    )
    return Order(
        symbol=str(args.get("symbol", "order")),
        side=_side(args["side"]),
        quantity=quantity,
        decision_time=_EPOCH,
        arrival_time=_EPOCH,
        fills=built,
        decision_price=float(args.get("decisionPrice", args["arrivalPrice"])),
    )


def _problem(args: dict[str, Any]) -> ExecutionProblem:
    """Build an execution problem, refusing the degenerate ones by name."""
    spec = args["problem"]
    impact = spec["impact"]
    eta = float(impact["eta"])
    periods = int(spec["periods"])
    horizon = float(spec["horizon"])

    # The closed form divides by the period length. A horizon that underflows to
    # nothing once cut into periods produces an infinity rather than an error, so
    # it is caught here where the numbers are still readable.
    if horizon / periods <= 0.0:
        raise DomainError(
            f"a horizon of {horizon:g} over {periods} periods gives a period length "
            "of zero. Lengthen the horizon or cut it into fewer periods.",
            field="problem",
        )

    return ExecutionProblem(
        quantity=float(spec["quantity"]),
        horizon=horizon,
        periods=periods,
        volatility=float(spec["volatility"]),
        impact=LinearImpact(
            gamma=float(impact.get("gamma", 0.0)),
            eta=eta,
            epsilon=float(impact.get("epsilon", 0.0)),
        ),
    )


# -- result rendering --------------------------------------------------------


def _bps(value: float, notional: float) -> float | None:
    """A cost in basis points of the paper notional, or nothing if there is none.

    A notional of zero happens when the decision price is zero, which is a
    degenerate input rather than an impossible one — so the currency figures stay
    and the ratio is reported as absent rather than as an infinity JSON cannot
    carry.
    """
    if notional == 0.0:
        return None
    return round(1e4 * value / notional, 6)


def _trajectory_payload(trajectory: Trajectory, problem: ExecutionProblem) -> dict[str, Any]:
    """Render a trajectory, with the cost figures it is chosen for."""
    expected, variance = schedule_moments(problem, trajectory.trades)
    return {
        "times": [round(t, 10) for t in trajectory.times],
        "holdings": [round(h, 6) for h in trajectory.holdings],
        "trades": [round(x, 6) for x in trajectory.trades],
        "expectedCost": round(trajectory.expected_cost, 6),
        "costVariance": round(trajectory.variance, 6),
        "costStandardDeviation": round(trajectory.std, 6),
        # Recomputed from the trades rather than read off the trajectory, so a
        # schedule and its cost cannot drift apart in the result.
        "expectedCostFromTrades": round(expected, 6),
        "costVarianceFromTrades": round(variance, 6),
        "halfLife": None if trajectory.half_life is None else round(trajectory.half_life, 6),
        "kappa": None if trajectory.kappa is None else round(trajectory.kappa, 10),
        "riskAversion": trajectory.risk_aversion,
    }


class ExecutionTools:
    """The execution tool group."""

    def shortfall_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        order = _order(args)
        arrival = float(args["arrivalPrice"])
        final = float(args["finalPrice"])
        basis = SlippageDelayBasis(args.get("delayBasis", "order"))

        breakdown = implementation_shortfall(
            order,
            arrival_price=arrival,
            final_price=final,
            decision_price=order.decision_price,
            fees=float(args.get("fees", 0.0)),
            half_spread=args.get("halfSpread"),
            delay_basis=basis,
        )

        notional = breakdown.paper_notional
        components = {
            "delay": breakdown.delay,
            "trading": breakdown.trading,
            "opportunity": breakdown.opportunity,
            "commission": breakdown.commission,
            "fees": breakdown.fees,
        }
        return {
            "side": breakdown.side.value,
            "delayBasis": basis.value,
            "targetQuantity": breakdown.target_quantity,
            "filledQuantity": breakdown.filled_quantity,
            "unfilledQuantity": round(breakdown.target_quantity - breakdown.filled_quantity, 10),
            "fillRate": round(order.fill_rate, 10),
            "decisionPrice": breakdown.decision_price,
            "arrivalPrice": breakdown.arrival_price,
            "finalPrice": breakdown.final_price,
            "averagePrice": breakdown.average_price,
            "paperNotional": round(notional, 6),
            "components": {name: round(value, 6) for name, value in components.items()},
            "componentsBps": {name: _bps(value, notional) for name, value in components.items()},
            "explicit": round(breakdown.explicit, 6),
            "implicit": round(breakdown.implicit, 6),
            "total": round(breakdown.total, 6),
            "totalBps": _bps(breakdown.total, notional),
        }

    def schedule_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        problem = _problem(args)
        aversion = float(args["riskAversion"])
        # A risk aversion of zero is the straight line, and asking the closed form
        # for it means evaluating sinh(0)/sinh(0). The library has a separate
        # entry point for that case; taking it here is exact rather than nearly
        # so, and cheaper than approaching the limit.
        solved = linear_trajectory(problem) if aversion == 0.0 else _solve(problem, aversion)
        trajectory = _trajectory_payload(solved, problem)
        sensitivity = half_life_sensitivity(problem, aversion) if aversion > 0.0 else None
        payload: dict[str, Any] = {
            "quantity": problem.quantity,
            "horizon": problem.horizon,
            "periods": problem.periods,
            "periodLength": round(problem.tau, 10),
            "riskAversion": aversion,
            "trajectory": trajectory,
        }
        if sensitivity is None:
            payload["halfLifeSensitivity"] = None
            payload["note"] = (
                "At zero risk aversion the schedule is a straight line, so its "
                "half-life is infinite and has no elasticities: an equal slice each "
                "period never decays. Raise riskAversion above zero for a finite "
                "one."
            )
        else:
            payload["halfLifeSensitivity"] = {
                "halfLife": round(sensitivity.half_life, 6),
                "elasticityToRiskAversion": round(sensitivity.risk_aversion, 6),
                "elasticityToVolatility": round(sensitivity.volatility, 6),
                "elasticityToTemporaryImpact": round(sensitivity.eta, 6),
                "elasticityToPermanentImpact": round(sensitivity.gamma, 6),
            }
        return payload

    def frontier_payload(self, args: dict[str, Any]) -> dict[str, Any]:
        problem = _problem(args)
        aversions = [float(value) for value in args["riskAversions"]]
        if len(aversions) > MAX_FRONTIER_POINTS:
            raise DomainError(
                f"{len(aversions)} risk aversions is over the {MAX_FRONTIER_POINTS} "
                "limit. The frontier is smooth; a dozen points spread over several "
                "orders of magnitude show its shape better than fifty bunched "
                "together.",
                field="riskAversions",
            )
        if sorted(aversions) != aversions:
            raise DomainError(
                "the risk aversions are not in increasing order. The frontier is "
                "read along it, and an unsorted list produces a result that looks "
                "like it doubles back on itself.",
                field="riskAversions",
            )

        trajectories = efficient_frontier(problem, aversions)
        points = []
        for aversion, trajectory in zip(aversions, trajectories, strict=True):
            expected, variance = schedule_moments(problem, trajectory.trades)
            points.append(
                {
                    "riskAversion": aversion,
                    "expectedCost": round(expected, 6),
                    "costStandardDeviation": round(math.sqrt(variance), 6),
                    "objective": round(expected + aversion * variance, 6),
                    "halfLife": (
                        None if trajectory.half_life is None else round(trajectory.half_life, 6)
                    ),
                    "trades": [round(x, 6) for x in trajectory.trades],
                }
            )
        return {
            "quantity": problem.quantity,
            "horizon": problem.horizon,
            "periods": problem.periods,
            "points": points,
        }

    def register(self, registry: ToolRegistry) -> ToolRegistry:
        read_only = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}

        registry.register(
            "decompose_implementation_shortfall",
            title="Split an order's cost into delay, trading and opportunity",
            description=(
                "Decompose what an order actually cost against the price at the "
                "moment it was decided on. The total is the least useful number in "
                "the result: delay is the price moving before the order reached the "
                "market, trading is the order's own footprint, opportunity is the "
                "part that never got done, and commission and fees are explicit. "
                "Each has a different remedy, so each is reported in currency and in "
                "basis points of the paper notional. Fill timestamps are not needed "
                "— the decomposition reads only quantities, prices and commissions. "
                "Whether delay is charged on the ordered or the executed quantity is "
                "a convention with no safe default: it moves cost between delay and "
                "opportunity without changing the total, and the basis used is named "
                "in the result."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "maxLength": 64,
                        "description": (
                            "Label for the instrument. Carried through, not used in "
                            "the arithmetic."
                        ),
                    },
                    "side": _SIDE,
                    "quantity": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": (
                            "Quantity the order was for, unsigned. Anything the fills "
                            "do not cover is the unfilled part, and is charged as "
                            "opportunity cost against the final price."
                        ),
                    },
                    "fills": _FILLS,
                    "decisionPrice": {
                        "type": "number",
                        "minimum": 0,
                        "description": (
                            "Price when the decision was taken. Defaults to the "
                            "arrival price, which sets the delay component to zero "
                            "— that is a statement that the order reached the market "
                            "instantly, not an absence of data."
                        ),
                    },
                    "arrivalPrice": {
                        "type": "number",
                        "minimum": 0,
                        "description": (
                            "Price when the order reached the market. This is the "
                            "boundary between the delay component and the trading "
                            "one."
                        ),
                    },
                    "finalPrice": {
                        "type": "number",
                        "minimum": 0,
                        "description": (
                            "Price at the end of the window. What the unfilled "
                            "quantity is marked against."
                        ),
                    },
                    "fees": {
                        "type": "number",
                        "minimum": 0,
                        "description": (
                            "Fees in currency beyond the per-fill commissions. An "
                            "explicit cost, reported apart from the implicit ones."
                        ),
                    },
                    "halfSpread": {
                        "type": "number",
                        "minimum": 0,
                        "description": (
                            "Half the quoted spread, if known. Carried into the "
                            "result as a reference to read the trading component "
                            "against."
                        ),
                    },
                    "delayBasis": {
                        "type": "string",
                        "enum": [basis.value for basis in SlippageDelayBasis],
                        "description": (
                            "Charge delay on the quantity ordered ('order', the "
                            "default) or the quantity executed ('executed'). The two "
                            "give the same total and split it differently."
                        ),
                    },
                },
                "required": ["side", "quantity", "fills", "arrivalPrice", "finalPrice"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.shortfall_payload))

        registry.register(
            "optimal_execution_schedule",
            title="Schedule an order against impact and price risk",
            description=(
                "The Almgren-Chriss trajectory for a linear impact model: how much "
                "to trade in each period, what it is expected to cost, and the "
                "variance around that. At zero risk aversion the answer is a "
                "straight line — an equal slice each period, which is TWAP — and "
                "raising the aversion front-loads the schedule, paying more impact "
                "to spend less time exposed. The half-life says how front-loaded, "
                "and its elasticities say which input would move it. Permanent "
                "impact and fixed costs are charged on the whole quantity whatever "
                "the schedule, so they change the cost and not the shape. "
                "Volatility is an absolute price move per unit of root time, not a "
                "percentage, because it is traded off against an impact quoted in "
                "price per share."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "object",
                        "properties": _PROBLEM,
                        "required": ["quantity", "horizon", "periods", "volatility", "impact"],
                        "additionalProperties": False,
                    },
                    "riskAversion": _RISK_AVERSION,
                },
                "required": ["problem", "riskAversion"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.schedule_payload))

        registry.register(
            "execution_cost_frontier",
            title="The cost-risk trade-off across risk aversions",
            description=(
                "Expected cost against the standard deviation of that cost, one "
                "point per risk aversion, each with the schedule that achieves it. "
                "A single optimal schedule answers a question the caller has "
                "already had to answer — how much cost is a unit of certainty worth "
                "— and this shows the trade-off instead of assuming it. Cost falls "
                "and risk rises as the aversion goes down; the aversions must be "
                "given in increasing order so the frontier reads along it."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "object",
                        "properties": _PROBLEM,
                        "required": ["quantity", "horizon", "periods", "volatility", "impact"],
                        "additionalProperties": False,
                    },
                    "riskAversions": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": MAX_FRONTIER_POINTS,
                        "items": {"type": "number", "minimum": 0},
                        "description": (
                            "Risk aversions to solve at, increasing. Spread them "
                            "over orders of magnitude rather than linearly: the "
                            "schedule changes with the square root of this, so a "
                            "linear sweep spends most of its points in one corner."
                        ),
                    },
                },
                "required": ["problem", "riskAversions"],
                "additionalProperties": False,
            },
            annotations=read_only,
        )(guard(self.frontier_payload))

        return registry


def _solve(problem: ExecutionProblem, aversion: float) -> Trajectory:
    """Solve, turning an unbounded problem into a refusal rather than an overflow.

    The closed form is a ratio of hyperbolic sines in ``kappa * T``. A large
    enough aversion, or a small enough temporary impact, sends that argument past
    where a double can hold ``sinh`` and the result comes back as a nan rather
    than as a failure. A nan reaching the caller is worse than a refusal, because
    it looks like an answer.
    """
    from slippage import optimal_trajectory

    trajectory = optimal_trajectory(problem, aversion)
    if not all(math.isfinite(value) for value in trajectory.holdings) or not math.isfinite(
        trajectory.expected_cost
    ):
        raise ToolExecutionError(
            f"a risk aversion of {aversion:g} against a temporary impact of "
            f"{problem.impact.eta:g} makes the schedule numerically unbounded — the "
            "optimum is to trade everything immediately, and the closed form "
            "overflows before it says so. Lower the risk aversion or raise eta.",
            kind="domain",
            details=[
                {
                    "path": "/riskAversion",
                    "keyword": "domain",
                    "message": "the trajectory overflowed at this aversion",
                }
            ],
        )
    return trajectory


def register(registry: ToolRegistry) -> ToolRegistry:
    """Add the execution tools to ``registry``."""
    return ExecutionTools().register(registry)


__all__ = [
    "MAX_FILLS",
    "MAX_FRONTIER_POINTS",
    "MAX_PERIODS",
    "ExecutionTools",
    "register",
]
