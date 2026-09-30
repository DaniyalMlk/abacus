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

import html
import importlib
import json
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs"))

from build import LISTING_BUDGET as BUILD_BUDGET  # noqa: E402
from claims import CLAIMS  # noqa: E402
from libraries import LIBRARIES, UNEXPOSED  # noqa: E402

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


# -- the libraries page describes the libraries that exist -------------------


def test_every_library_module_is_described_or_declared_internal() -> None:
    """A library that grows a capability fails here until the page gains one.

    This is the check that did not exist while the page went stale twice. It
    compares the modules actually installed against the ones `docs/libraries.py`
    accounts for, and a module in neither column is a capability nobody has
    written a sentence about.
    """
    for library in LIBRARIES:
        module = importlib.import_module(library.importable)
        installed = {
            name
            for _, name, _ in pkgutil.iter_modules(module.__path__)
            if not name.startswith("_")
        }
        accounted = library.described_modules | set(library.plumbing)
        undescribed = sorted(installed - accounted)
        assert not undescribed, (
            f"{library.importable} has module(s) {undescribed} that the libraries "
            "page neither describes nor declares internal; add a Capability for "
            "them in docs/libraries.py, or list them as plumbing"
        )


def test_no_described_module_has_gone_away() -> None:
    """The other direction: a phrase about a module that no longer exists."""
    for library in LIBRARIES:
        module = importlib.import_module(library.importable)
        installed = {
            name
            for _, name, _ in pkgutil.iter_modules(module.__path__)
            if not name.startswith("_")
        }
        accounted = library.described_modules | set(library.plumbing)
        vanished = sorted(accounted - installed)
        assert not vanished, (
            f"docs/libraries.py names module(s) {vanished} in {library.importable} "
            "that are not installed"
        )


def test_no_module_is_both_described_and_declared_internal() -> None:
    """A module belongs to one column or the other, never both.

    The two checks above compare the union of the columns against what is
    installed, so a module listed twice passes both of them while the table it
    is read out of says two contradictory things about it. It matters in one
    direction in particular: adding a module to the plumbing tuple to get a
    failing build green, when a capability already names it, leaves the page
    describing something the table also calls internal, and no other assertion
    here would notice.
    """
    for library in LIBRARIES:
        both = sorted(library.described_modules & frozenset(library.plumbing))
        assert not both, (
            f"{library.importable}: module(s) {both} are named by a capability and "
            "also declared plumbing; a module is one or the other"
        )


def test_no_module_is_described_twice() -> None:
    """Two phrases about one module means the page says it twice."""
    for library in LIBRARIES:
        seen: dict[str, str] = {}
        for capability in library.capabilities:
            for module in capability.modules:
                assert module not in seen, (
                    f"{library.importable}.{module} is named by both "
                    f"{seen[module]!r} and {capability.phrase!r}"
                )
                seen[module] = capability.phrase
        assert len(library.plumbing) == len(set(library.plumbing))


def test_every_phrase_reaches_the_page(site: dict[str, str]) -> None:
    page = html.unescape(site["libraries.html"])
    for library in LIBRARIES:
        for capability in library.capabilities:
            assert capability.phrase in page, (
                f"{library.importable}: the page does not carry the phrase "
                f"{capability.phrase!r}"
            )


def test_both_names_reach_the_page(site: dict[str, str]) -> None:
    page = site["libraries.html"]
    for library in LIBRARIES:
        assert f">{library.distribution}<" in page
        assert f">{library.importable}<" in page


def test_the_install_names_match_the_packaging() -> None:
    """The page's install names are the requirements the server declares."""
    text = (ROOT / "pyproject.toml").read_text()
    for library in LIBRARIES:
        assert f'"{library.distribution}' in text, (
            f"{library.distribution} is on the libraries page but is not a "
            "declared dependency of the server"
        )


def test_a_capability_names_at_least_one_module() -> None:
    for library in LIBRARIES:
        assert library.capabilities
        for capability in library.capabilities:
            assert capability.modules
            assert capability.phrase.strip() == capability.phrase
            assert not capability.phrase.endswith(".")


def test_the_sentence_reads_as_a_sentence() -> None:
    for library in LIBRARIES:
        sentence = library.sentence
        assert sentence.endswith(".")
        assert ", and " in sentence
        assert "  " not in sentence


def test_every_unexposed_entry_names_modules_that_exist() -> None:
    """The gap table cannot describe something that has been removed."""
    for entry in UNEXPOSED:
        module = importlib.import_module(entry.library)
        installed = {
            name
            for _, name, _ in pkgutil.iter_modules(module.__path__)
            if not name.startswith("_")
        }
        missing = sorted(set(entry.modules) - installed)
        assert not missing, f"{entry.library} has no module(s) {missing}"


def test_nothing_unexposed_is_actually_exposed() -> None:
    """The other half: a gap that has since been filled by a tool.

    Matched on the module the entry names rather than on its prose. Every tool
    in the registry is asked which library modules it reaches, by importing its
    handler's module and reading what it imported -- crude, and enough to catch
    the case this is for, which is somebody adding a tool and forgetting the
    page still calls it missing.
    """
    reached: set[tuple[str, str]] = set()
    for tool in default_registry():
        handler = sys.modules.get(type(tool).__module__)
        if handler is None:
            continue
        for value in vars(handler).values():
            name = getattr(value, "__module__", None) or getattr(value, "__name__", None)
            if not isinstance(name, str) or "." not in name:
                continue
            library, _, submodule = name.partition(".")
            reached.add((library, submodule))
    for entry in UNEXPOSED:
        for submodule in entry.modules:
            assert (entry.library, submodule) not in reached, (
                f"{entry.library}.{submodule} is listed as unexposed but a tool "
                "imports it; remove the entry from docs/libraries.py"
            )


def test_the_unexposed_section_reaches_the_page(site: dict[str, str]) -> None:
    page = html.unescape(site["libraries.html"])
    assert "In the libraries, not on the server" in page
    for entry in UNEXPOSED:
        assert entry.phrase in page


def test_the_page_states_the_budget_the_suite_enforces(site: dict[str, str]) -> None:
    """One budget, written in two files, asserted equal.

    ``tests/test_skill.py`` holds the number the suite enforces and
    ``docs/build.py`` holds the one the page prints. The first is read out of
    the source rather than imported: the tests directory is not a package, so
    importing a sibling test module by name works only when pytest happens to
    have put it on the path.
    """
    source = (ROOT / "tests" / "test_skill.py").read_text()
    match = re.search(r"^LISTING_BUDGET = ([\d_]+)", source, re.MULTILINE)
    assert match is not None, "tests/test_skill.py no longer declares LISTING_BUDGET"
    enforced = int(match.group(1).replace("_", ""))

    assert enforced == BUILD_BUDGET
    listing = json.dumps([tool.describe() for tool in default_registry()])
    page = html.unescape(site["libraries.html"])
    assert f"{len(listing):,} characters" in page
    assert f"ceiling of {enforced:,}" in page
    assert f"{len(list(default_registry()))} tools" in page
