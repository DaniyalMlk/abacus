"""The documentation site builds, and the numbers on it are held to account.

Two different jobs here.

The first is that the site builds and covers the surface. Every tool has to
appear on the reference page and in its navigation, because the generator's
whole justification is that it cannot describe a server that does not exist —
and a tool the generator quietly drops is exactly the case that would break
that.

The second is the validation table. The documentation on this project prints a
lot of specific figures, and each was measured rather than guessed. A measured
claim with no check behind it drifts, and the drift is silent: the number goes
on reading as authoritative while the code has moved. So `docs/claims.py` names,
for each figure, the test that measures it, and the tests here assert that every
one of those tests exists and that the figure is present in its source.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs"))

from claims import CLAIMS  # noqa: E402

from abacus.analytics import default_registry  # noqa: E402


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Build the site by running the script, and return every page's HTML.

    Through the script rather than by importing and calling `build()`, because
    the documented way to produce the site is `python docs/build.py` and an
    import-time failure in that path — a missing module, a bad relative import —
    would not show up any other way.
    """
    completed = subprocess.run(
        [sys.executable, str(ROOT / "docs" / "build.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    out = ROOT / "site"
    assert out.is_dir()
    return {path.name: path.read_text() for path in out.iterdir() if path.is_file()}


# -- the site covers the server ----------------------------------------------


def test_the_expected_pages_are_written(site: dict[str, str]) -> None:
    for name in (
        "index.html",
        "conventions.html",
        "tools.html",
        "validation.html",
        "libraries.html",
        "style.css",
        ".nojekyll",
    ):
        assert name in site, f"{name} is missing from the built site"


def test_every_tool_appears_on_the_reference_page(site: dict[str, str]) -> None:
    """The generator's justification, tested.

    If a tool can be registered and not reach the page, the site is no longer a
    rendering of the registry and the claim that it cannot go stale is false.
    """
    page = site["tools.html"]
    for tool in default_registry():
        assert f'id="{tool.name}"' in page, f"{tool.name} has no entry on the tool page"
        assert f'href="#{tool.name}"' in page, f"{tool.name} is not in the tool index"


def test_every_tool_is_placed_in_exactly_one_group(site: dict[str, str]) -> None:
    """A tool listed twice would appear twice in the index and once in the page."""
    page = site["tools.html"]
    for tool in default_registry():
        assert page.count(f'id="{tool.name}"') == 1, f"{tool.name} is in more than one group"


def test_required_arguments_are_marked(site: dict[str, str]) -> None:
    """A reader scanning for what they must supply needs it on the page."""
    page = site["tools.html"]
    assert page.count("<span class=\"req\">required</span>") >= 50
    # And `price_european_option` names its six required arguments.
    entry = page.split('id="price_european_option"', 1)[1].split("</article>", 1)[0]
    for name in ("spot", "strike", "time", "rate", "vol", "type"):
        assert f"<td>{name}<span" in entry, f"{name} is not marked required"


def test_the_pages_do_not_carry_unescaped_markup(site: dict[str, str]) -> None:
    """Descriptions are written prose and will eventually contain a `<` or an `&`.

    Everything from the registry goes through escaping on the way in, and an
    unescaped angle bracket would break the document silently rather than
    loudly.
    """
    for name, text in site.items():
        if not name.endswith(".html"):
            continue
        # Between the tags, not after the opening one: everything after
        # `<main>` includes the shell div that the template closes, and a test
        # that counted it would fail on a document that is perfectly balanced.
        body = text.split("<main>", 1)[1].split("</main>", 1)[0]
        # No stray closing tag the generator did not open.
        assert body.count("<article") == body.count("</article>"), name
        assert body.count("<table") == body.count("</table>"), name
        assert body.count("<div") == body.count("</div>"), name


def test_every_page_links_to_every_other(site: dict[str, str]) -> None:
    """The rail is the only navigation, so a page missing from it is unreachable."""
    pages = [name for name in site if name.endswith(".html")]
    for name in pages:
        for other in pages:
            assert f'href="{other}"' in site[name], f"{name} does not link to {other}"


def test_the_current_page_is_marked_in_the_rail(site: dict[str, str]) -> None:
    for name, text in site.items():
        if name.endswith(".html"):
            assert text.count('aria-current="page"') == 1, name


# -- the validated numbers ---------------------------------------------------


def test_every_claim_names_a_test_that_exists() -> None:
    """A row pointing at a test that was renamed is worse than no row.

    It reads as though the figure is checked, and the thing that was checking it
    is gone.
    """
    for claim in CLAIMS:
        module = ROOT / claim.module
        assert module.is_file(), f"{claim.figure}: {claim.module} does not exist"
        source = module.read_text()
        assert f"def {claim.test}(" in source, (
            f"{claim.figure}: {claim.module} has no test named {claim.test}"
        )


def test_every_claim_appears_in_the_test_that_checks_it() -> None:
    """The figure as documented occurs in the source of its checking test.

    This is the join that keeps the two honest. A number that moves in the code
    has to move in its test, and a test whose figure no longer matches the one
    on the page fails here rather than leaving a reader with a stale number that
    still looks authoritative.
    """
    for claim in CLAIMS:
        source = (ROOT / claim.module).read_text()
        body = source.split(f"def {claim.test}(", 1)[1].split("\ndef ", 1)[0]
        # Or in a module-level constant the test uses. Two of the budget figures
        # are written once as `LISTING_BUDGET` and `TOOL_BUDGET` and referred to
        # by name, which is better code than repeating the literal — so the join
        # accepts a constant in the same file rather than forcing the number to
        # be inlined for the sake of this check.
        constants = re.findall(r"^[A-Z][A-Z_]* *(?::[^=]+)?= *(.+)$", source, re.MULTILINE)
        assert claim.figure in body or any(claim.figure in value for value in constants), (
            f"{claim.test} does not mention {claim.figure}, which the documentation "
            "says it checks, and neither does any constant in that file"
        )


def test_the_validation_page_shows_every_claim(site: dict[str, str]) -> None:
    page = site["validation.html"]
    for claim in CLAIMS:
        assert claim.figure in page
        assert claim.test in page


def test_the_claims_are_distinct() -> None:
    """Two rows with the same figure and the same test is a copy-paste, not a claim."""
    seen = {(claim.figure, claim.test, claim.about) for claim in CLAIMS}
    assert len(seen) == len(CLAIMS)


def test_the_claimed_tests_pass() -> None:
    """Named, collected and run — not merely present in a file.

    Asserting the functions exist proves the table points somewhere. Running
    them proves the figures on the page are ones that currently hold, which is
    what the page says.
    """
    selection = sorted({f"{claim.module}::{claim.test}" for claim in CLAIMS})
    completed = subprocess.run(
        # `-o addopts=` because the project's own addopts is `-q`, and a second
        # -q suppresses the summary line this test then reads.
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            *selection,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-4000:]
    # And every one of them was actually collected, rather than silently skipped
    # by a name that no longer matches anything.
    match = re.search(r"(\d+) passed", completed.stdout)
    assert match is not None, completed.stdout[-2000:]
    assert int(match.group(1)) == len(selection)
