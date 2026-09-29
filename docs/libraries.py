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
            Capability("volume curves and participation", ("volume",)),
            Capability("a simulator to score a schedule against", ("simulate", "synthetic")),
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
    Unexposed("slippage", "post-trade mark-outs and reversion decay", ("reversion",)),
    Unexposed(
        "slippage",
        "basket liquidation under a matrix of impact and a matrix of risk",
        ("basket",),
    ),
)
