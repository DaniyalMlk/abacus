"""The README's known-gaps list defends itself.

A list of shortcomings is the first thing in a document to go stale, because
closing a gap is satisfying and deleting the line that described it is not. So
each line carries a predicate, and the predicate holds only while the gap is
real: implement the feature and its line fails, forcing the line out.

Two rules about what belongs on the list:

**Absences, not refusals.** Most of what this server declines it declines on
purpose -- a volatility of 20 is refused as a percentage written the wrong way,
the Greeks at expiry are refused because the derivative does not exist, the
older stateful protocol is refused because it was removed -- and those are
documented as behaviour where they belong.

**Call something.** A predicate that searches for a name cannot see the feature
arriving under a different one. Each predicate below talks to a server, over a
transport where the gap is about a transport, or resolves what the server
actually imports to the module that defines it.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import io
import json
import pkgutil
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from abacus.cli import build_server
from abacus.client import Client, InProcessTransport
from abacus.errors import METHOD_NOT_FOUND
from abacus.transports import StreamableHttp, serve_stdio

from .conftest import request as build_request

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
SOURCE = ROOT / "src" / "abacus"

sys.path.insert(0, str(ROOT / "docs"))
from libraries import UNEXPOSED  # noqa: E402

#: The heading the list lives under, and the heading that ends it.
SECTION = "## Known gaps"
NEXT_SECTION = "## Development"

LIBRARIES = frozenset({"moneyness", "shortfall", "tenor", "slippage", "holdout"})

PRICE = {"spot": 100.0, "strike": 100.0, "time": 1.0, "rate": 0.05, "vol": 0.2, "type": "call"}


def call_payload(*, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_request(
        "tools/call", {"name": "price_european_option", "arguments": PRICE}, extra=extra
    )


def http_headers(payload: dict[str, Any], **more: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": payload["params"]["_meta"][
            "io.modelcontextprotocol/protocolVersion"
        ],
        "Mcp-Method": payload["method"],
        "Mcp-Name": payload["params"]["name"],
        **more,
    }


# --- one predicate per line of the list ----------------------------------- #


def tools_and_nothing_else() -> None:
    """The capability map has one entry, and every other family is unknown."""
    client = Client(InProcessTransport(build_server()))
    discovered = client.discover().result
    assert discovered is not None
    assert set(discovered["capabilities"]) == {"tools"}
    for method in (
        "resources/list",
        "resources/read",
        "prompts/list",
        "prompts/get",
        "completion/complete",
    ):
        error = client.request(method, {}).error
        assert error is not None and error["code"] == METHOD_NOT_FOUND, (method, error)
    assert client.request("tools/list", {}).ok, "the probe cannot tell a known method"


def no_progress_is_reported() -> None:
    """A call carrying a progress token gets its result and nothing else."""
    payload = call_payload(extra={"progressToken": "known-gaps"})

    out = io.StringIO()
    serve_stdio(build_server(), stdin=io.StringIO(json.dumps(payload) + "\n"), stdout=out)
    lines = out.getvalue().splitlines()
    assert len(lines) == 1, lines
    (only,) = (json.loads(line) for line in lines)
    assert only["id"] == payload["id"] and "result" in only, only

    response = StreamableHttp(build_server()).handle(
        "POST", "/mcp", http_headers(payload), json.dumps(payload).encode()
    )
    assert response.status == 200
    assert isinstance(response.body, dict) and "result" in response.body, response.body


def http_has_no_authorisation() -> None:
    """Served alike with no credential and with a credential that means nothing."""
    http = StreamableHttp(build_server())
    payload = call_payload()
    for extra in ({}, {"Authorization": "Bearer not-a-credential-of-any-kind"}):
        response = http.handle(
            "POST", "/mcp", http_headers(payload, **extra), json.dumps(payload).encode()
        )
        assert response.status == 200, (extra, response.status)
        assert response.body is not None
        assert response.body["result"]["isError"] is False


def defining_modules() -> set[str]:
    """Every library module that defines something the server imports.

    An import from a submodule is owned by that submodule, or by the module a
    class or function says defined it. Any other import from a library's root
    -- a bare number, a tuple, a type alias, none of which records where it was
    made -- is owned by the submodule whose source binds the name at top level.
    Resolving the object rather than matching its name is what keeps ``price``,
    declared by several moneyness modules, from counting against the wrong one.

    The source is read rather than each module's ``__all__``, because the
    published releases predate the libraries' declared-surface gates: in
    ``tenor-curves`` 0.1.0 the root exports ``DEFAULT_DIGITS`` and
    ``tenor.futures``, which defines it, does not declare it.
    """
    owners: set[str] = set()
    for path in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in LIBRARIES:
                        owners.add(alias.name)
            if not isinstance(node, ast.ImportFrom) or node.level or node.module is None:
                continue
            library = node.module.split(".")[0]
            if library not in LIBRARIES:
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                obj = getattr(module, alias.name)
                if node.module != library:
                    owners.add(node.module)
                if callable(obj) and isinstance(getattr(obj, "__module__", None), str):
                    owners.add(obj.__module__)
                elif node.module == library:
                    owners.update(modules_defining(library, alias.name))
    return owners


def top_level_bindings(source: str) -> set[str]:
    """Names a module's own source binds at top level: definitions and assignments.

    Imports are excluded, so a module that merely re-uses a name -- a command
    line importing a default to print it -- is not taken for its owner.
    """
    bound: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                bound.update(assigned_names(target))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bound.update(assigned_names(node.target))
    return bound


def assigned_names(target: ast.expr) -> set[str]:
    """The names an assignment target binds: through tuples, never through ``a.b``."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Tuple | ast.List):
        return {name for element in target.elts for name in assigned_names(element)}
    if isinstance(target, ast.Starred):
        return assigned_names(target.value)
    return set()


@functools.cache
def bindings_by_module(library: str) -> dict[str, frozenset[str]]:
    """Each submodule of an installed library, with what its source binds."""
    package = importlib.import_module(library)
    out: dict[str, frozenset[str]] = {}
    for info in pkgutil.iter_modules(package.__path__):
        spec = importlib.util.find_spec(f"{library}.{info.name}")
        assert spec is not None and spec.origin is not None, info.name
        source = Path(spec.origin).read_text(encoding="utf-8")
        out[f"{library}.{info.name}"] = frozenset(top_level_bindings(source))
    return out


def modules_defining(library: str, name: str) -> set[str]:
    return {module for module, bound in bindings_by_module(library).items() if name in bound}


def fifty_six_capabilities_have_no_tool() -> None:
    assert len(UNEXPOSED) == 56
    owners = defining_modules()
    assert len(owners) >= 20, f"the scan found only {sorted(owners)}, so it is not looking"
    held = {f"{entry.library}.{module}" for entry in UNEXPOSED for module in entry.modules}
    reached = sorted(held & owners)
    assert not reached, f"the server imports from modules listed as having no tool: {reached}"


GAPS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("The server offers tools, and no resources or prompts.", tools_and_nothing_else),
    ("A tool call cannot report progress.", no_progress_is_reported),
    ("The HTTP transport has no authorisation.", http_has_no_authorisation),
    ("Fifty-six things the libraries do have no tool.", fifty_six_capabilities_have_no_tool),
)


def gaps_section() -> str:
    """The text from the list's heading up to the heading that follows it."""
    text = README.read_text(encoding="utf-8")
    start = text.index(SECTION)
    end = text.index(NEXT_SECTION, start + len(SECTION))
    return text[start:end]


def rendered_bullets() -> list[str]:
    """The bold lead of each bullet under the heading, in page order."""
    return re.findall(r"^- \*\*(.+?)\*\*", gaps_section(), re.MULTILINE)


@pytest.mark.parametrize("line, predicate", GAPS, ids=[line for line, _ in GAPS])
def test_the_gap_is_still_real(line: str, predicate: Callable[[], None]) -> None:
    predicate()


@pytest.mark.parametrize("line", [line for line, _ in GAPS])
def test_the_line_is_still_on_the_page(line: str) -> None:
    assert line in rendered_bullets(), f"not a bullet under {SECTION!r}: {line!r}"


def test_the_page_lists_nothing_that_has_no_predicate() -> None:
    """Same lines, same order: a bullet added to the page needs a predicate too."""
    assert rendered_bullets() == [line for line, _ in GAPS]


def test_every_predicate_is_distinct() -> None:
    predicates = [predicate for _, predicate in GAPS]
    assert len({p.__name__ for p in predicates}) == len(predicates)


def test_a_bold_lead_does_not_wrap() -> None:
    """A lead broken across two lines is invisible to the bullet reader."""
    for line in gaps_section().splitlines():
        if line.startswith("- **"):
            assert line.count("**") >= 2, f"the bold lead wraps: {line!r}"


def test_the_section_ends_at_the_heading_that_follows_it() -> None:
    headings = re.findall(r"^## .+$", README.read_text(encoding="utf-8"), re.MULTILINE)
    assert SECTION in headings
    following = headings[headings.index(SECTION) + 1]
    assert following == NEXT_SECTION, (
        f"the list is bounded by {NEXT_SECTION!r} and is followed by {following!r}"
    )


# --- the ownership probe, against cases built for it ---------------------- #


def test_every_root_name_the_server_could_import_has_a_defining_module() -> None:
    """Whichever release is installed, every root export is bound somewhere."""
    for library in sorted(LIBRARIES):
        for name in importlib.import_module(library).__all__:
            if name == "__version__":
                continue
            assert modules_defining(library, name), f"{library}.{name} is defined nowhere"


def test_a_bare_number_is_owned_by_the_module_that_binds_it() -> None:
    """``DEFAULT_DIGITS`` is bound in ``tenor.futures`` and only imported by the CLI."""
    assert modules_defining("tenor", "DEFAULT_DIGITS") == {"tenor.futures"}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("X = 1", {"X"}),
        ("X: int = 1", {"X"}),
        ("A, (B, C) = 1, (2, 3)", {"A", "B", "C"}),
        ("A = B = 0", {"A", "B"}),
        ("def f(): pass\nasync def g(): pass\nclass K: pass", {"f", "g", "K"}),
        ("from .futures import DEFAULT_DIGITS\nimport math", set()),
        ("X: int", set()),
        ("first, *rest = range(3)", {"first", "rest"}),
        ("settings.value = 1\ntable[0] = 2", set()),
        ("if True:\n    HIDDEN = 1", set()),
        ("def f():\n    LOCAL = 1", {"f"}),
    ],
)
def test_the_binding_probe_against_source_written_for_it(
    source: str, expected: set[str]
) -> None:
    assert top_level_bindings(source) == expected


def test_the_probe_attributes_a_shared_function_name_correctly() -> None:
    """``moneyness.price`` is the Black-Scholes one, not Heston's or the quanto's."""
    owners = defining_modules()
    assert "moneyness.heston" not in owners
    assert "moneyness.quanto" not in owners
    assert importlib.import_module("moneyness").price.__module__ in owners
