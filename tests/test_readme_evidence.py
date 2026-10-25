"""The figures on the front page are the figures the server produces.

`tests/test_docs.py` already holds the documentation *site* to account, through
`docs/claims.py`. The README was outside that: it made its claims in prose and
quoted a suite size that had gone stale by two hundred tests, which is the whole
failure mode in one line — a number copied into a document once stops being true
quietly, because nothing in the suite reads the document.

So the front page is pinned the same way, in both directions. Every figure it
quotes has to appear in the output of `docs/evidence.py`, which is itself a gate
— it exits non-zero if a figure moves outside the band the README states. And
every row label has to still be in the document, which is the other way a claim
disappears: somebody deletes the row.

This test is also what makes "runs in CI" true. CI runs `pytest`, so the gate
runs here as well as in a step of its own, and a figure that moves fails the
suite on the commit that moved it.

One direction is deliberately not attempted. Scanning the README for a number
and comparing it to the measurement cannot tell which number it found: this page
carries 138,605 and 140,000 within a few characters of each other, and a check
looking for "some number near here" can pass against the wrong one. The forward
direction plus the row labels catches a figure that moved and a row that
vanished, which is what matters.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
GATE = ROOT / "docs" / "evidence.py"


def _load_gate() -> Any:
    """Import `docs/evidence.py`, which is a script beside a package."""
    sys.path.insert(0, str(GATE.parent))
    spec = importlib.util.spec_from_file_location("abacus_readme_evidence", GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: Figures the README quotes, exactly as the gate prints them.
#:
#: Written out rather than parsed out of the README on purpose: deleting a row
#: from the table should not silently delete the check that defends it.
QUOTED = (
    "delta, vega and gamma all within 6e-10, from different formulae",
    "19 of 29 pass, the other 10 needing a real HTTP endpoint; revision 2026-07-28",
    "138,605 of 140,000 characters across 39 tools, 1,395 left",
    "56 capabilities across 5 libraries",
)

#: Row labels from the evidence table. Presence is what is checked, not the
#: value beside it — see the module docstring for why that direction is the
#: wrong one to scan for.
ROWS = (
    "The Greeks against finite differences of this server's own prices",
    "The conformance suite, against this server in process",
    "What a client receives when it lists the tools, against the ceiling",
    "What is deliberately not exposed, held to a table that fails both ways",
)


@pytest.fixture(scope="module")
def gate() -> Any:
    return _load_gate()


@pytest.fixture(scope="module")
def gate_output(gate: Any) -> str:
    """Run the gate once and share its output. It runs the conformance suite."""
    return str(gate.report())


@pytest.mark.parametrize("figure", QUOTED)
def test_the_readme_quotes_a_figure_the_gate_prints(figure: str, gate_output: str) -> None:
    assert figure in gate_output, "the gate no longer produces this figure"
    assert figure in README, "the README no longer quotes this figure"


@pytest.mark.parametrize("row", ROWS)
def test_the_evidence_table_still_has_the_row(row: str) -> None:
    assert row in README


def test_the_gate_renders_the_table_the_readme_carries(gate_output: str) -> None:
    """Every row the gate renders is in the README, verbatim and whole.

    This catches a row added to the gate and never put on the front page, and a
    row whose label and value drifted apart: the two halves are checked together
    rather than separately, so a figure moved into the wrong row fails too.
    """
    rendered = [
        line
        for line in gate_output.splitlines()
        if line.startswith("| ")
        and not line.startswith("| ---")
        and line != "| Check | Result |"
    ]
    assert len(rendered) == len(QUOTED)
    for line in rendered:
        assert line in README


def test_the_listing_budget_is_one_number_written_in_three_places(gate: Any) -> None:
    """`docs/evidence.py`, `docs/build.py` and `tests/test_skill.py` agree.

    Three files now state the ceiling, and nothing but this would notice them
    drifting. A gate measuring against a stale ceiling would pass while the
    listing was over the one the site advertises.
    """
    from build import LISTING_BUDGET as BUILD_BUDGET

    skill = (ROOT / "tests" / "test_skill.py").read_text(encoding="utf-8")
    match = re.search(r"^LISTING_BUDGET = ([\d_]+)", skill, re.MULTILINE)
    assert match is not None, "tests/test_skill.py no longer declares LISTING_BUDGET"

    declared = int(match.group(1).replace("_", ""))
    assert gate.LISTING_BUDGET == BUILD_BUDGET == declared


def test_the_headroom_is_the_difference_the_readme_states(gate: Any) -> None:
    """1,395 characters, and it is a subtraction rather than a second figure.

    The README treats the headroom as the reason nothing substantial can be
    exposed next, so it has to be the ceiling less the listing and not a number
    somebody typed.
    """
    figures = gate.measure()
    assert figures.listing_headroom == figures.listing_budget - figures.listing_characters
    assert f"{figures.listing_headroom:,} left" in README
    assert "1.0% of" in README


def test_the_tolerance_catches_what_the_readme_says_it_catches(gate: Any) -> None:
    """The claim that a 1% argument slip fails, measured rather than asserted.

    This is the sentence the first row's credibility rests on: a bound that
    survives a thousandfold change of step and refuses a one-per-cent change of
    argument is measuring the wiring. Both halves are checked here, because a
    bound nothing can violate is not evidence of anything.
    """
    option = dict(gate.OPTION)
    price = gate._payload

    def priced(**overrides: object) -> float:
        return float(price("price_european_option", **{**option, **overrides})["price"])

    step = gate.DELTA_STEP
    spot = float(option["spot"])
    numerical = (priced(spot=spot + step) - priced(spot=spot - step)) / (2 * step)

    # The honest reading agrees.
    honest = float(price("european_option_greeks", **option)["delta"])
    assert abs(honest - numerical) < 1e-6

    # A volatility 1% away from the one the prices were taken at does not, and
    # 1e-6 is the bound that refuses it.
    slipped = float(
        price("european_option_greeks", **{**option, "vol": float(option["vol"]) * 1.01})["delta"]
    )
    assert abs(slipped - numerical) > 1e-6

    # And a thousandfold wider step still agrees, so the figure is about the
    # wiring and not about the step.
    wide = 1e-1
    coarse = (priced(spot=spot + wide) - priced(spot=spot - wide)) / (2 * wide)
    assert abs(honest - coarse) < 1e-6


def test_the_conformance_figure_accounts_for_every_check(gate: Any) -> None:
    """19 passing plus 10 skipped is 29, and none is unaccounted for.

    Reported as two numbers rather than one on purpose: ten of the checks need
    a real HTTP endpoint, so "29 of 29" would be a claim about whichever
    transport the figure was measured on.
    """
    figures = gate.measure()
    assert figures.conformance_passed + figures.conformance_skipped == figures.conformance_checks
    assert f"{figures.conformance_passed} of {figures.conformance_checks} pass" in README
    assert "needing a real HTTP endpoint" in README


def test_the_suite_size_the_readme_claims_is_about_right() -> None:
    """A loose bound, on purpose.

    The line this replaces said 629 against a suite of 866, which is how an
    exact count goes wrong: it fails on every commit that adds a test, so people
    update the number without reading it until one day they stop. A quarter
    either way catches a figure that has gone properly stale and tolerates
    ordinary growth.
    """
    import subprocess

    match = re.search(r"pytest\s+# ([\d,]+) tests", README)
    assert match is not None, "the README no longer says how large the suite is"
    claimed = int(match.group(1).replace(",", ""))

    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "--collect-only",
            "tests",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    collected = re.search(r"(\d+) tests? collected", finished.stdout)
    if collected is None:  # pragma: no cover - no checkout to collect from
        pytest.skip(f"could not collect: {finished.stdout[-400:]}")

    actual = int(collected.group(1))
    message = f"the README claims {claimed}, the suite is {actual}"
    assert claimed >= actual * 0.75, message
    assert claimed <= actual * 1.25, message
