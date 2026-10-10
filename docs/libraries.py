"""What each library underneath this server contains, and a way to notice drift.

The libraries page is the only place a reader who arrived at the server is told
that the libraries exist and what is in them. It is the distribution, and it had
already gone stale twice before this table existed: a library gained a whole
model and the page went on describing the one before it, because prose is not
checked against anything.

So the page's descriptions are assembled here from a per-module table, and
``tests/test_docs.py`` asserts three things about it:

1. every module in every installed library is accounted for, either by a
   :class:`Capability` or by being named as plumbing;
2. every capability's phrase actually appears on the rendered page; and
3. every module named here still exists in the library.

A library that grows a module then fails a build until somebody writes a phrase
for it or declares it internal. That is the whole mechanism, and it is
deliberately crude: the alternative designs all end up parsing prose.

The phrases are written to describe what the library is *for*, not to enumerate
its modules. Several modules share one phrase, which is correct — a reader does
not need to know that the barrier correction lives in its own file.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LIBRARIES", "Capability", "Library"]


@dataclass(frozen=True)
class Capability:
    """One thing a library does, and the modules that do it.

    Attributes:
        phrase: Text that has to appear on the libraries page. Written as it
            reads in the sentence, so the check is on the published words.
        modules: Modules providing it. Each must exist in the library.
    """

    phrase: str
    modules: tuple[str, ...]


@dataclass(frozen=True)
class Library:
    """A library, as the page presents it.

    Attributes:
        distribution: The name it installs under.
        importable: The name it imports as.
        capabilities: What it does, in the order the page lists them.
        plumbing: Modules that exist to serve the others and are not described
            on the page. Named explicitly so that a new one is a decision
            rather than an omission.
    """

    distribution: str
    importable: str
    capabilities: tuple[Capability, ...]
    plumbing: tuple[str, ...]

    @property
    def sentence(self) -> str:
        """The description the page prints."""
        phrases = [capability.phrase for capability in self.capabilities]
        if len(phrases) == 1:
            return phrases[0] + "."
        return ", ".join(phrases[:-1]) + ", and " + phrases[-1] + "."

    @property
    def described_modules(self) -> frozenset[str]:
        return frozenset(
            module for capability in self.capabilities for module in capability.modules
        )


LIBRARIES: tuple[Library, ...] = (
    Library(
        distribution="moneyness",
        importable="moneyness",
        capabilities=(
            Capability(
                "option pricing under Black-Scholes-Merton with the full Greek set",
                ("bsm", "greeks", "normal", "bivariate"),
            ),
            Capability("implied volatility", ("implied",)),
            Capability("SVI surfaces and Dupire local volatility", ("svi", "surface")),
            Capability(
                "pricing on a finite-difference grid under that local volatility, "
                "European, American and knocked out at a barrier",
                ("pde",),
            ),
            Capability(
                "single-barrier options in closed form, agreeing with the grid and "
                "the simulation to the accuracy each of those supports",
                ("barrier",),
            ),
            Capability(
                "American exercise on lattices and in closed form",
                ("lattice", "american"),
            ),
            Capability(
                "Heston stochastic volatility priced through its characteristic "
                "function by two independent routes",
                ("heston", "quadrature"),
            ),
            Capability(
                "Monte Carlo under both models, including Andersen's "
                "quadratic-exponential scheme for the variance process",
                ("monte_carlo", "heston_mc", "optimise"),
            ),
            Capability(
                "SABR smiles in both volatility conventions, with a Bachelier "
                "model and the density the smile implies",
                ("sabr",),
            ),
            Capability(
                "variance and volatility swap strikes replicated out of the "
                "quoted smile, with no model in the replication at all",
                ("variance",),
            ),
            Capability(
                "average-price options, whose price can only be bracketed between "
                "bounds that hold as inequalities",
                ("asian",),
            ),
            Capability(
                "options on the difference of two correlated assets, with an upper "
                "bound that carries no correlation at all because it is the price "
                "under the worst one",
                ("spread",),
            ),
            Capability(
                "lookback options on where the path got to, all four of them from "
                "one integral of the running extreme's own law",
                ("lookback",),
            ),
            Capability(
                "choosers and compound options, where the decision is taken at a "
                "date before the payoff is known and the price is checked against "
                "a decomposition that puts nothing of its own on the other side",
                ("deferred",),
            ),
        ),
        plumbing=("cli",),
    ),
    Library(
        distribution="shortfall",
        importable="shortfall",
        capabilities=(
            Capability(
                "shrinkage covariance and factor risk attribution",
                ("covariance", "factors", "linalg"),
            ),
            Capability(
                "value at risk and expected shortfall by five methods",
                ("parametric", "historical", "distributions"),
            ),
            Capability(
                "spectral risk measures, where coherence is a property of the "
                "weight function rather than of the construction",
                ("spectral",),
            ),
            Capability(
                "the loss tail of a portfolio of obligors from its cumulant "
                "generating function, expanded where the question is rather than "
                "at the centre of the distribution",
                ("saddlepoint",),
            ),
            Capability(
                "risk contributions, risk parity and drawdown statistics",
                ("contributions", "drawdown"),
            ),
            Capability(
                "conditional volatility, simulated horizons and coverage backtests",
                ("volatility", "horizon", "backtest"),
            ),
            Capability(
                "generalised Pareto tails and Gaussian or Student-t copulas",
                ("extreme", "copula", "ranks"),
            ),
            Capability(
                "volatility from the whole bar by five range estimators",
                ("realised",),
            ),
            Capability(
                "strictly consistent scores for ranking two adequate risk models",
                ("scoring",),
            ),
            Capability(
                "expectiles, the only risk measure that is coherent and elicitable "
                "at once",
                ("expectile",),
            ),
            Capability(
                "a view imposed on a scenario set by reweighting it rather than by "
                "filtering it, so the dependence that made the sample worth using "
                "survives the stress",
                ("entropy",),
            ),
            Capability(
                "the range value at risk can occupy when the marginals are known "
                "and the dependence is not, bracketed between two inequalities and "
                "two couplings that are constructed rather than assumed",
                ("bounds",),
            ),
        ),
        plumbing=("cli", "series"),
    ),
    Library(
        distribution="tenor",
        importable="tenor",
        capabilities=(
            Capability(
                "day counts, business-day conventions and schedules",
                ("daycount", "calendar", "schedule"),
            ),
            Capability(
                "curve bootstrapping with monotone interpolation",
                ("bootstrap", "curve", "monotone", "rates", "instruments", "solve"),
            ),
            Capability("bond analytics and key rate durations", ("bond", "risk")),
            Capability(
                "option-adjusted spreads on a short-rate lattice",
                ("lattice", "spread"),
            ),
            Capability("floating rate notes and index-linked bonds", ("floating", "inflation")),
            Capability("forward curves, carry and roll-down", ("horizon",)),
            Capability(
                "survival curves, credit default swaps and risky bonds",
                ("credit",),
            ),
            Capability(
                "deliverable bond futures, conversion factors and the basis",
                ("futures",),
            ),
            Capability(
                "forecasting and discounting on separate curves, with tenor basis",
                ("multicurve",),
            ),
            Capability(
                "swaptions on the annuity measure, caps and floors",
                ("options",),
            ),
            Capability(
                "caplet volatilities bootstrapped out of flat cap quotes, with how "
                "sharply each one is pinned reported beside it",
                ("stripping",),
            ),
            Capability(
                "constant maturity swaps, replicated out of those same swaptions "
                "because the rate is paid outside the measure it is a martingale "
                "under",
                ("cms",),
            ),
            Capability(
                "a Gaussian short rate model fitted to the curve, with bond and "
                "swaption prices in closed form",
                ("hullwhite",),
            ),
            Capability(
                "a second factor, so that two rates are no longer forced to move "
                "together",
                ("g2",),
            ),
            Capability(
                "Bermudan swaptions, where several exercise dates leave no formula "
                "for any of them",
                ("bermudan",),
            ),
            Capability(
                "covered interest parity across two currencies, and the "
                "cross-currency basis that is the residual of it",
                ("fx",),
            ),
            Capability(
                "overnight rates compounded over an accrual, under each of the "
                "conventions on where the observation window sits, with the "
                "replication identity reported where one exists",
                ("overnight",),
            ),
        ),
        plumbing=("cli",),
    ),
    Library(
        distribution="slippage-tca",
        importable="slippage",
        capabilities=(
            Capability(
                "implementation shortfall decomposed against its benchmarks",
                ("shortfall", "benchmarks", "costs", "report"),
            ),
            Capability(
                "market impact fitting and reversion",
                ("impact", "calibration", "reversion"),
            ),
            Capability(
                "Almgren-Chriss schedules, constrained and for a whole basket",
                ("scheduling", "execution", "basket"),
            ),
            Capability(
                "transient impact under a decay kernel, and the schedule it implies",
                ("transient",),
            ),
            Capability("volume curves and participation", ("volume",)),
            Capability("a simulator to score a schedule against", ("simulate", "synthetic")),
            Capability(
                "schedules that react to the liquidity regime they find",
                ("adaptive",),
            ),
            Capability(
                "resting a limit order against crossing the spread: the fill "
                "probability, the adverse selection and the cost of each",
                ("placement",),
            ),
            Capability(
                "tracking a volume-weighted benchmark, which moves with the market "
                "and so inverts what risk aversion does",
                ("tracking",),
            ),
            Capability(
                "a schedule that can hold a view, where the smoothing a price "
                "forecast receives is the urgency the problem already had",
                ("alpha",),
            ),
        ),
        plumbing=("cli", "exceptions", "io", "series", "types"),
    ),
    Library(
        distribution="holdout-backtest",
        importable="holdout",
        capabilities=(
            Capability(
                "deflated Sharpe ratios and effective trial counts",
                ("deflated", "sharpe", "trials", "moments"),
            ),
            Capability("the probability of backtest overfitting", ("pbo",)),
            Capability("purged and combinatorial cross-validation", ("splits",)),
            Capability(
                "tests for superior predictive ability and a model confidence set",
                ("spa", "mcs", "bootstrap"),
            ),
            Capability(
                "multiplicity haircuts and a robust Sharpe-difference test",
                ("multiple", "pairwise"),
            ),
            Capability(
                "a test for a break in the Sharpe ratio at a date the data chose",
                ("stability",),
            ),
            Capability(
                "sample weights for overlapping labels, and the effective sample "
                "size they imply",
                ("uniqueness",),
            ),
        ),
        plumbing=("cli", "exceptions", "io", "series", "synthetic"),
    ),
)


@dataclass(frozen=True)
class Unexposed:
    """Something a library does that the server does not offer a tool for.

    Attributes:
        library: The import name.
        phrase: What it does, as the page prints it.
        modules: Where it lives. Checked to exist, so this table cannot
            describe something that has been removed.
    """

    library: str
    phrase: str
    modules: tuple[str, ...]


#: The tool surface has a size budget, and it is nearly spent. Everything here
#: is built, tested and reachable by importing the library; none of it has a
#: tool. Listing it is more useful than leaving a reader to discover the gap by
#: asking the server for something it does not have -- and it is the honest
#: description of the boundary, which is a budget rather than a judgement about
#: what matters.
UNEXPOSED: tuple[Unexposed, ...] = (
    Unexposed(
        "moneyness",
        "Heston stochastic volatility: the transform, the smile it generates, "
        "and simulation of the variance process",
        ("heston", "heston_mc"),
    ),
    Unexposed(
        "moneyness",
        "Asian and barrier payoffs under stochastic volatility",
        ("heston_mc",),
    ),
    Unexposed(
        "shortfall",
        "volatility from open, high, low and close, by five range estimators",
        ("realised",),
    ),
    Unexposed("tenor", "floating rate notes and discount margins", ("floating",)),
    Unexposed(
        "tenor",
        "index-linked bonds, real duration and breakeven inflation",
        ("inflation",),
    ),
    Unexposed(
        "tenor",
        "survival curves bootstrapped from par credit default swap spreads, "
        "risky bond pricing and the credit triangle's measured error",
        ("credit",),
    ),
    Unexposed(
        "tenor",
        "deliverable bond futures: conversion factors, the basis and the "
        "cheapest bond to deliver",
        ("futures",),
    ),
    Unexposed("slippage", "post-trade mark-outs and reversion decay", ("reversion",)),
    Unexposed(
        "slippage",
        "impact that decays at a rate rather than instantly, and the "
        "block-rate-block schedule that minimises its cost",
        ("transient",),
    ),
    Unexposed(
        "holdout",
        "a bootstrap test for a break in the Sharpe ratio at a date chosen by "
        "the data, and how little power it has",
        ("stability",),
    ),
    Unexposed(
        "slippage",
        "basket liquidation under a matrix of impact and a matrix of risk",
        ("basket",),
    ),
    Unexposed(
        "tenor",
        "a projection curve solved against a discount curve, tenor basis swaps, "
        "and how little the discount curve moves a par swap rate",
        ("multicurve",),
    ),
    Unexposed(
        "slippage",
        "the optimal schedule when liquidity switches regime, and what reacting "
        "to it is worth against the best schedule fixed in advance",
        ("adaptive",),
    ),
    Unexposed(
        "moneyness",
        "SABR in both volatility conventions, a Bachelier model, and where a "
        "long-dated smile stops being a distribution",
        ("sabr",),
    ),
    Unexposed(
        "tenor",
        "swaptions on the annuity measure, caps and floors as strips, and the "
        "hundredfold gap between what the discount curve is worth to an option "
        "and to the rate underneath it",
        ("options",),
    ),
    Unexposed(
        "moneyness",
        "variance and volatility swaps replicated from the quoted smile, and how "
        "much further out the strikes have to reach than a Gaussian rule of thumb "
        "would suggest",
        ("variance",),
    ),
    Unexposed(
        "tenor",
        "Hull-White's one factor, with Jamshidian's decomposition for swaptions "
        "and the measured fact that one quote cannot separate mean reversion from "
        "volatility",
        ("hullwhite",),
    ),
    Unexposed(
        "slippage",
        "whether to rest a limit order or cross the spread, and why the placement "
        "distance turns out not to be a choice",
        ("placement",),
    ),
    Unexposed(
        "moneyness",
        "a finite-difference solver that prices under the Dupire local volatility "
        "rather than only computing it, and the coordinate mistake no derivative "
        "test can catch",
        ("pde",),
    ),
    Unexposed(
        "shortfall",
        "scoring functions that rank two adequate models, and the measured failure "
        "of the one anybody builds first",
        ("scoring",),
    ),
    Unexposed(
        "shortfall",
        "expectiles, coherent and elicitable at once, and the measured reason a "
        "single level cannot stand in for a confidence level",
        ("expectile",),
    ),
    Unexposed(
        "holdout",
        "sample weights for overlapping labels, and the arithmetic cap that leaves "
        "a sequential bootstrap nothing to recover at full size",
        ("uniqueness",),
    ),
    Unexposed(
        "tenor",
        "a two-factor Gaussian model, the decorrelation one factor cannot produce, "
        "and the measured fact that a cap carries no information about it",
        ("g2",),
    ),
    Unexposed(
        "moneyness",
        "single-barrier options in closed form, and the measured fact that a "
        "knock-out is the one price in the library that falls as volatility rises",
        ("barrier",),
    ),
    Unexposed(
        "tenor",
        "constant maturity swaps by static replication, and the measured sign "
        "change the payment date produces in the convexity adjustment",
        ("cms",),
    ),
    Unexposed(
        "shortfall",
        "spectral risk measures, and the measured factor of two between two "
        "spectra that agree on the headline charge",
        ("spectral",),
    ),
    Unexposed(
        "tenor",
        "Bermudan swaptions by backward induction on a curve-fitted tree, with "
        "the switch value over the best single date the extra rights actually buy",
        ("bermudan",),
    ),
    Unexposed(
        "moneyness",
        "arithmetic-average options bracketed rather than priced, where the "
        "moment-matched price every system quotes falls below a rigorous lower "
        "bound out of the money",
        ("asian",),
    ),
    Unexposed(
        "slippage",
        "tracking error against a volume-weighted benchmark, which moves with the "
        "market, and the floor volume uncertainty sets underneath it",
        ("tracking",),
    ),
    Unexposed(
        "moneyness",
        "options on the difference of two correlated assets, and the measured fact "
        "that the approximation desks quote leaves a rigorous no-arbitrage bracket "
        "at eighteen of eighty-four strikes and correlations",
        ("spread",),
    ),
    Unexposed(
        "tenor",
        "caplet volatilities bootstrapped out of flat cap quotes, and the strike "
        "below which a quote identifies no volatility at all because the premium "
        "and the intrinsic value are the same double",
        ("stripping",),
    ),
    Unexposed(
        "shortfall",
        "a portfolio loss tail from its cumulant generating function, where the "
        "lattice correction most treatments call a refinement is worth a factor of "
        "seven hundred",
        ("saddlepoint",),
    ),
    Unexposed(
        "slippage",
        "an execution schedule that holds a price forecast, and the measured fact "
        "that a tuned proportional tilt recovers 99% of what solving it exactly is "
        "worth on anything but a sharp signal",
        ("alpha",),
    ),
    Unexposed(
        "moneyness",
        "lookback options on the running extreme, where the closed form everybody "
        "transcribes divides by zero on an option on a future and a monthly fix is "
        "worth eighteen per cent less than the continuous contract",
        ("lookback",),
    ),
    Unexposed(
        "shortfall",
        "a view imposed on a scenario set by minimum relative entropy, where the "
        "move it induces in a series the view never mentioned is that series' own "
        "regression coefficient and its tail moves by under half of what a "
        "parallel shift predicts",
        ("entropy",),
    ),
    Unexposed(
        "tenor",
        "covered interest parity between two currencies and the basis left over, "
        "where the two-day settlement lag nobody prices is nearly five per cent of "
        "a three-month forward's points",
        ("fx",),
    ),
    Unexposed(
        "moneyness",
        "choosers and compound options, where the chooser's closed form is checked "
        "against the vanilla pair put-call parity decomposes it into and each "
        "compound pair against a parity that constrains both of its members at once",
        ("deferred",),
    ),
    Unexposed(
        "shortfall",
        "value at risk bounded with no dependence named at all, where the worst "
        "coupling runs to twice the comonotonic one on heavy tails and most of the "
        "gap between the attained bound and the proved one is quadrature rather "
        "than dependence",
        ("bounds",),
    ),
    Unexposed(
        "tenor",
        "compounded overnight legs and the four conventions on the observation "
        "window, where a five-day lookback is worth nothing on a smoothly "
        "interpolated curve and nearly four basis points across a policy step",
        ("overnight",),
    ),
)
