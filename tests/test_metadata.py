"""Checks on the distribution rather than on the server.

Nothing here exercises the protocol. These are the failures that only appear
when someone tries to install the project, which is the worst moment to discover
them: a dependency an index will not accept, a version written in two files that
have drifted apart, a typing marker that never reaches the wheel, a licence
claimed in metadata and absent from the artefact.

The direct-reference check is the one that matters most. The server was
unpublishable for its whole life because of a single requirement line, and
nothing in the suite noticed.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import metadata, version
from pathlib import Path

import pytest

import abacus

DISTRIBUTION = "abacus-mcp"
ROOT = Path(__file__).resolve().parent.parent

# A requirement is a direct reference when it names its source with `@` followed
# by a URL, as in `moneyness @ git+https://...`. That form resolves locally and
# is refused on upload, so it is legal Python packaging and still fatal here.
_DIRECT_REFERENCE = re.compile(r"@\s*[a-z+]+://", re.IGNORECASE)


def _requirements() -> list[str]:
    return metadata(DISTRIBUTION).get_all("Requires-Dist", [])


def test_the_import_name_is_not_the_distribution_name() -> None:
    """Stated rather than assumed, because the mismatch is deliberate.

    `abacus` is taken on the index by an unrelated project. Anything looking the
    version up has to ask for `abacus-mcp`, and a future rename of one without
    the other should fail here rather than at install time.
    """
    assert abacus.__name__ == "abacus"
    assert version(DISTRIBUTION) == abacus.__version__


def test_version_is_a_release_number() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", abacus.__version__)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib is 3.11+")
def test_version_matches_the_source_of_truth() -> None:
    """The attribute agrees with pyproject.toml in the checkout.

    Comparing against the *installed* metadata, as the test above does, compares
    against a snapshot taken at install time. In an editable install that
    snapshot can be older than the file, so it would pass on a tree whose
    declared version has already moved.
    """
    import tomllib

    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():  # installed with no source tree alongside
        pytest.skip("no checkout to compare against")

    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert abacus.__version__ == declared


def test_no_dependency_is_a_direct_reference() -> None:
    """No requirement points at a repository, an archive or a path.

    This is the check that would have caught the fault that made the project
    unpublishable. `pip install -e .` is perfectly happy with a `git+https`
    requirement; an index rejects any distribution whose metadata contains one,
    so the build succeeds, the tests pass, and the upload is the first thing that
    fails.
    """
    offenders = [line for line in _requirements() if _DIRECT_REFERENCE.search(line)]
    assert offenders == [], f"direct references cannot be published: {offenders}"


def test_every_runtime_dependency_is_bounded_at_both_ends() -> None:
    """The numerical libraries, each bounded above as well as below.

    Unbounded below, a resolver may pick a release predating the functions this
    server calls. Unbounded above, the next breaking release of a library breaks
    the server for everyone who installs it after that, with no change here. The
    bounds are the only thing that makes a published version mean something a
    year later.
    """
    runtime = {
        line.split(maxsplit=1)[0].split(">")[0].split("<")[0].split("=")[0]: line.replace(" ", "")
        for line in _requirements()
        if "extra ==" not in line
    }
    # Named for the distribution, not the module: `slippage` on the index is an
    # unrelated project, so the library that `import slippage` reaches publishes
    # as `slippage-tca`, and a requirement is a distribution name.
    expected = {"moneyness", "shortfall", "tenor", "slippage-tca", "holdout-backtest"}
    assert set(runtime) == expected, runtime
    for name, requirement in runtime.items():
        assert ">=0.1" in requirement, name
        assert "<0.2" in requirement, name


def test_the_typing_marker_ships() -> None:
    """`Typing :: Typed` is only true if py.typed is inside the package.

    Without the marker file, a type checker treats the installed package as
    untyped and every annotation in it is ignored, while the classifier goes on
    promising otherwise.
    """
    assert (Path(abacus.__file__).parent / "py.typed").is_file()
    assert "Typing :: Typed" in metadata(DISTRIBUTION).get_all("Classifier", [])


def test_the_licence_reaches_the_distribution() -> None:
    fields = metadata(DISTRIBUTION)
    assert fields["License-Expression"] == "MIT"
    assert "LICENSE" in fields.get_all("License-File", [])


def test_every_supported_interpreter_is_advertised() -> None:
    """The classifiers and `requires-python` agree on what is supported.

    They are two independent statements of the same fact and are read by
    different audiences — resolvers use one, people browsing an index use the
    other — so they drift silently when a version is added to only one.
    """
    fields = metadata(DISTRIBUTION)
    assert fields["Requires-Python"] == ">=3.10"
    advertised = {
        line.rsplit(" :: ", 1)[1]
        for line in fields.get_all("Classifier", [])
        if line.startswith("Programming Language :: Python :: 3.")
    }
    assert advertised == {"3.10", "3.11", "3.12", "3.13"}


def test_both_console_scripts_reach_the_entry_point() -> None:
    """`abacus` and `abacus-mcp` are both installed, and are the same command.

    The alias exists so that `uvx abacus-mcp stdio` works: `uvx` runs the console
    script whose name matches the distribution, and without the alias a client
    configuration would have to carry `uvx --from abacus-mcp abacus stdio`
    instead. Losing the alias would break every configuration written against it,
    silently, at launch time rather than at install time.
    """
    from importlib.metadata import entry_points

    scripts = {
        entry.name: entry.value
        for entry in entry_points(group="console_scripts")
        if entry.name in {"abacus", "abacus-mcp"}
    }
    assert scripts == {"abacus": "abacus.cli:main", "abacus-mcp": "abacus.cli:main"}
