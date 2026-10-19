"""The front page's figures, measured rather than remembered.

`docs/claims.py` holds the figures the documentation *site* prints and the test
that checks each one. This module is the same idea for the README, and the
figures are different in kind because the question is different.

A library's front page should say what its mathematics was checked against. A
*server's* front page should say what its wiring was checked against, because a
boundary's characteristic failure is not a wrong formula — the formulae live in
five libraries that each validate their own — but a tool that returns a
plausible number computed from the wrong argument, or from the right argument
in the wrong units. Nothing in a plausible number says which.

So the rows here are about the boundary:

* **Greeks against finite differences of this server's own prices.** The
  analytic delta one tool reports, against a central difference of the prices
  another tool reports. Those come from different formulae and travel through
  different schema paths, so agreement is evidence about the wiring rather than
  a restatement of it. This is the strongest check in the repository and it was
  a sentence in the Development section.
* **The listing budget.** What a client actually receives when it lists the
  tools, against the ceiling. It is the constraint on what can be exposed next,
  and a reader deciding whether to depend on this should be able to see it.
* **The capability accounting.** How much of the five libraries is *not*
  exposed, held to a table that fails in both directions.
* **The conformance suite**, which has a count, and which the server passes.

Run it directly; it exits non-zero if a figure moves outside the band the
README states, so it is a gate and not a printout:

    python docs/evidence.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from libraries import LIBRARIES, UNEXPOSED

from abacus.analytics import default_registry
from abacus.cli import build_server
from abacus.client import Client, InProcessTransport
from abacus.conformance import CHECKS
from abacus.conformance import run as run_conformance
from abacus.protocol import SUPPORTED_VERSIONS

#: The ceiling on the serialised tool listing. Written here as well as in
#: `docs/build.py` and `tests/test_skill.py`; a test holds the three in step,
#: because they are one budget written in three places and nothing else would
#: notice them drifting.
LISTING_BUDGET = 140_000

#: Steps for the finite differences. Small enough that the truncation error is
#: below the tolerance and large enough that cancellation is not: a central
#: difference's total error is minimised somewhere in the middle and both ends
#: of that trade-off look like disagreement.
#:
#: The tolerances were chosen by measuring what they catch rather than by
#: picking a round number, and the pair of results is the reason to trust them.
#: Widening ``DELTA_STEP`` by a factor of a thousand, to 1e-1, moves the delta
#: residue only to 6.5e-07 and stays inside the 1e-06 bound — so the figure is
#: not an artefact of the step. Reading the Greeks at a volatility 1% away from
#: the one the prices were taken at moves it to 3.9e-06 and fails. That is the
#: wiring error this check exists to catch: a tolerance that survives a
#: thousandfold change of step and refuses a one-per-cent change of argument is
#: measuring the thing it claims to.
DELTA_STEP = 1e-4
VEGA_STEP = 1e-5
GAMMA_STEP = 1e-2


@dataclass(frozen=True)
class Figures:
    """Everything the front page quotes, measured."""

    tools: int
    listing_characters: int
    listing_budget: int
    listing_headroom: int
    libraries: int
    unexposed: int
    conformance_checks: int
    conformance_passed: int
    #: Checks a single transport cannot exercise. Reported rather than folded
    #: into the pass count, since "29 of 29" would otherwise be a lie told by
    #: whichever transport the figure was measured on.
    conformance_skipped: int
    revision: str
    delta_error: float
    vega_error: float
    gamma_error: float
    worst_greek_error: float


def _payload(name: str, **arguments: object) -> dict[str, object]:
    """Call one tool through the registry and return its structured content.

    Through the registry rather than by calling the library behind it, because
    the schema coercion and the argument names are exactly what this is
    checking: a number that arrived through the right path and a number that
    arrived through the wrong one look the same once it is a float.
    """
    result = default_registry().call(name, dict(arguments))
    if result["isError"]:
        raise SystemExit(f"{name} refused the call: {result['content'][0]['text']}")
    content = result["structuredContent"]
    if not isinstance(content, dict):
        raise SystemExit(f"{name} returned no structured content")
    return content


#: The option the finite differences are taken around. At the money with a
#: year to run, which is where every Greek is away from a boundary and none of
#: them is dominated by a term the others are not.
OPTION: dict[str, object] = {
    "spot": 100.0,
    "strike": 100.0,
    "vol": 0.2,
    "rate": 0.02,
    "time": 1.0,
    "type": "call",
}


def _greeks_against_finite_differences() -> tuple[float, float, float]:
    """The analytic Greeks against differences of the server's own prices."""

    def price(**overrides: object) -> float:
        value = _payload("price_european_option", **{**OPTION, **overrides})["price"]
        if not isinstance(value, (int, float)):
            raise SystemExit(f"price came back as {value!r}")
        return float(value)

    greeks = _payload("european_option_greeks", **OPTION)

    def reported(name: str) -> float:
        value = greeks[name]
        if not isinstance(value, (int, float)):
            raise SystemExit(f"{name} came back as {value!r}")
        return float(value)

    spot = float(OPTION["spot"])  # type: ignore[arg-type]
    vol = float(OPTION["vol"])  # type: ignore[arg-type]

    numerical_delta = (price(spot=spot + DELTA_STEP) - price(spot=spot - DELTA_STEP)) / (
        2 * DELTA_STEP
    )
    numerical_vega = (price(vol=vol + VEGA_STEP) - price(vol=vol - VEGA_STEP)) / (2 * VEGA_STEP)
    numerical_gamma = (
        price(spot=spot + GAMMA_STEP) - 2 * price() + price(spot=spot - GAMMA_STEP)
    ) / GAMMA_STEP**2

    return (
        abs(reported("delta") - numerical_delta),
        abs(reported("vega") - numerical_vega),
        abs(reported("gamma") - numerical_gamma),
    )


def measure() -> Figures:
    registry = default_registry()
    tools = list(registry)
    listing = json.dumps([tool.describe() for tool in tools])

    # In process rather than over a socket: the suite drives the same server
    # object either transport would, and CI runs it over both separately.
    report = run_conformance(Client(InProcessTransport(build_server())))
    if report.failures:
        raise SystemExit(
            "conformance failed: " + ", ".join(o.name for o in report.failures)
        )
    passed = len(report.passed)
    skipped = len(report.skipped)

    delta, vega, gamma = _greeks_against_finite_differences()

    return Figures(
        tools=len(tools),
        listing_characters=len(listing),
        listing_budget=LISTING_BUDGET,
        listing_headroom=LISTING_BUDGET - len(listing),
        libraries=len(LIBRARIES),
        unexposed=len(UNEXPOSED),
        conformance_checks=len(CHECKS),
        conformance_passed=passed,
        conformance_skipped=skipped,
        revision=SUPPORTED_VERSIONS[0],
        delta_error=delta,
        vega_error=vega,
        gamma_error=gamma,
        worst_greek_error=max(delta, vega, gamma),
    )


#: The bands the README promises, as bounds rather than digits. Where an
#: analytic derivative and a difference of two prices stop agreeing depends on
#: the step and on the floating-point unit, so a figure quoted to ten places
#: would be a number that happens to be true where it was measured.
BOUNDS: tuple[tuple[str, str], ...] = (
    ("delta against a central difference", "1e-06"),
    ("vega against a central difference", "1e-05"),
    ("gamma against a second difference", "1e-06"),
)


def check(figures: Figures) -> None:
    """Fail on anything the README would then be wrong about."""
    if figures.delta_error > 1e-6:
        raise SystemExit(f"delta is {figures.delta_error:.3e} from the difference")
    if figures.vega_error > 1e-5:
        raise SystemExit(f"vega is {figures.vega_error:.3e} from the difference")
    if figures.gamma_error > 1e-6:
        raise SystemExit(f"gamma is {figures.gamma_error:.3e} from the difference")

    if figures.listing_characters > figures.listing_budget:
        raise SystemExit(
            f"the listing is {figures.listing_characters:,} characters against a ceiling "
            f"of {figures.listing_budget:,}"
        )
    if figures.conformance_passed + figures.conformance_skipped != figures.conformance_checks:
        raise SystemExit(
            f"{figures.conformance_passed} passed and {figures.conformance_skipped} "
            f"skipped of {figures.conformance_checks} conformance checks, which "
            "leaves some unaccounted for"
        )
    if figures.unexposed == 0 or figures.libraries == 0:
        raise SystemExit("the capability accounting is empty, which it cannot be")


def rows(figures: Figures) -> tuple[tuple[str, str], ...]:
    """The README's table, as it is rendered there."""
    return (
        (
            "The Greeks against finite differences of this server's own prices",
            f"delta, vega and gamma all within {figures.worst_greek_error:.0e}, "
            f"from different formulae",
        ),
        (
            "The conformance suite, against this server in process",
            f"{figures.conformance_passed} of {figures.conformance_checks} pass, the "
            f"other {figures.conformance_skipped} needing a real HTTP endpoint; "
            f"revision {figures.revision}",
        ),
        (
            "What a client receives when it lists the tools, against the ceiling",
            f"{figures.listing_characters:,} of {figures.listing_budget:,} characters "
            f"across {figures.tools} tools, {figures.listing_headroom:,} left",
        ),
        (
            "What is deliberately not exposed, held to a table that fails both ways",
            f"{figures.unexposed} capabilities across {figures.libraries} libraries",
        ),
    )


def report(figures: Figures | None = None) -> str:
    """The table, plus the figures spelled out underneath it."""
    measured = measure() if figures is None else figures
    check(measured)

    lines = ["", "1. What is checked, and against what", "", "| Check | Result |", "| --- | --- |"]
    for label, result in rows(measured):
        lines.append(f"| {label} | {result} |")

    lines.extend(["", "2. The same figures, spelled out", ""])
    lines.append(f"   delta error: {measured.delta_error:.3e}")
    lines.append(f"   vega error: {measured.vega_error:.3e}")
    lines.append(f"   gamma error: {measured.gamma_error:.3e}")
    lines.append(
        f"   listing: {measured.listing_characters} of {measured.listing_budget} "
        f"characters, {measured.tools} tools"
    )
    lines.append(
        f"   headroom: {measured.listing_headroom} characters "
        f"({measured.listing_headroom / measured.listing_budget:.1%})"
    )
    lines.append(f"   unexposed: {measured.unexposed} across {measured.libraries} libraries")
    lines.append(
        f"   conformance: {measured.conformance_passed} passed, "
        f"{measured.conformance_skipped} skipped, of {measured.conformance_checks} "
        f"on revision {measured.revision}"
    )
    for what, bound in BOUNDS:
        lines.append(f"   bound: {what} to {bound}")
    lines.extend(["", "every figure above is inside the band the README states"])
    return "\n".join(lines)


def main() -> int:
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
