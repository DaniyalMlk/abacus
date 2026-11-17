"""What each module declares is the authority on what is public.

The package root is curated on purpose: it carries the protocol surface a server
author needs — the error codes and exceptions, the message types, ``Server`` —
and none of the tool modules, which are reached through the tool registry and
over the wire rather than by import. So the rule the sibling libraries hold
(the root equals the union of every module's declaration) is the wrong rule
here, and is not asserted.

What the curation needs instead is for each module to say what it makes
public, and that was missing where it mattered most. ``errors`` and
``protocol`` carried no ``__all__``, and the root takes 28 of its 31 names from
them: the root was the only statement anywhere that those names were public,
and nothing could check it. Six more modules declared nothing, and four that
did left out names they define — ``conformance.expect`` and its three siblings
among them, which a caller writing their own ``Check`` for ``run(checks=...)``
needs.

So the checks here are: every module declares; each declaration holds every
public name its source defines, and names nothing it does not define except a
pinned re-export; every root name is declared by exactly one module that
defines it; every public name imported across a module boundary — in ``src``,
``tests`` and ``docs`` — is declared by its owner; and a type in the signature
of anything reachable from the root is itself reachable from the root.

Definitions and imports are read with :mod:`ast`, not from the imported
modules. An attribute sweep cannot attribute a bare ``int``, ``str`` or
``frozenset`` to the module that defines it, and most of what was undeclared
here — ``DISCOVER_TTL_MS``, ``MAC_BYTES``, the ``META_*`` keys — is exactly that.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import typing
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

import abacus
from abacus.protocol import Notification, Request
from abacus.server import Server

REPOSITORY = Path(__file__).resolve().parent.parent

#: Not part of the importable surface: ``python -m abacus`` runs the server.
NOT_SURFACE = frozenset({"abacus.__main__"})

#: Names a module declares and does not define: re-exports, pinned so that a
#: new one is a decision rather than drift. ``server`` hands on the two
#: protocol types its own signatures use; ``transports`` is a package whose
#: ``__init__`` re-exports a curated part of its two modules.
PINNED_REEXPORTS: dict[str, frozenset[str]] = {
    "abacus": frozenset(),  # the root is checked separately, below
    "abacus.server": frozenset({"Capabilities", "InvalidRequest"}),
    "abacus.transports": frozenset(
        {"MAX_BODY_BYTES", "HttpResponse", "StreamableHttp", "serve_http", "serve_stdio"}
    ),
}

#: Where cross-module imports are looked for.
CONSUMERS = ("src", "tests", "docs")

#: Source written to exercise :func:`public_names_in`, not to be a module.
PROBE_SOURCE = '''
"""A docstring, which defines nothing."""

from __future__ import annotations

import math                      # an import is not a definition
from collections.abc import Callable

TTL_MS = 3_600_000               # a bare int: no __module__ to sweep by
Handler = Callable[[int], None]  # an alias: likewise
ANNOTATED: int = 2               # an annotated assignment
PAIR_ONE = PAIR_TWO = 3          # one statement, two names
_PRIVATE = 4                     # underscore: not public
__all__ = ["TTL_MS"]             # the declaration is never a definition


def public_function() -> None:
    """A function."""
    local_name = 5               # a name in a body is not top level


async def public_coroutine() -> None:
    """An async function."""


class PublicClass:
    """A class."""

    attribute = 6                # a class attribute is not a module name
'''

#: Source written to exercise :func:`package_imports_in`, as if it were
#: ``abacus/transports/http.py``.
IMPORT_PROBE = """
from __future__ import annotations

import json
from abacus.errors import ProtocolError
from abacus import Server
from ..protocol import META_PROTOCOL_VERSION, error_response as respond
from .stdio import serve_stdio
from ..schema import _private_helper
from . import stdio
"""


def public_names_in(source: str) -> frozenset[str]:
    """The public top-level names a piece of module source defines."""
    found: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        found.update(name for name in names if not name.startswith("_"))
    return frozenset(found)


def package_imports_in(source: str, package: str | None) -> list[tuple[str, str]]:
    """Every public ``(module, name)`` a piece of source imports from abacus.

    ``package`` is the dotted package the source lives in, or ``None`` outside
    the package, where a relative import cannot refer to it. Private names are
    skipped — a test reaching for ``_parser`` is testing an implementation and
    makes no claim about the surface — and so is a bare ``from . import``,
    which names a module rather than something a module declares.
    """
    found: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.level == 0:
            if not node.module.startswith("abacus."):
                continue
            module = node.module
        elif package is None:
            continue
        else:
            parts = package.split(".")
            base = parts[: len(parts) - (node.level - 1)]
            module = ".".join([*base, node.module])
        found.extend(
            (module, alias.name) for alias in node.names if not alias.name.startswith("_")
        )
    return found


def module_source(name: str) -> Path:
    """The file a dotted module name is defined in."""
    module = importlib.import_module(name)
    assert module.__file__ is not None
    return Path(module.__file__)


def modules() -> dict[str, ModuleType]:
    """Every module of the package, the subpackage's included, bar ``__main__``."""
    found = {"abacus": abacus}
    for info in pkgutil.walk_packages(abacus.__path__, "abacus."):
        if info.name not in NOT_SURFACE:
            found[info.name] = importlib.import_module(info.name)
    return found


def declared() -> dict[str, frozenset[str]]:
    """Each module's own declaration of what it makes public."""
    return {
        name: frozenset(module.__all__)
        for name, module in modules().items()
        if getattr(module, "__all__", None) is not None
    }


def definitions(name: str) -> frozenset[str]:
    """What a module's own source defines publicly."""
    return public_names_in(module_source(name).read_text(encoding="utf-8"))


def package_types_in(hint: object) -> set[type]:
    """Every abacus class mentioned anywhere inside a type annotation."""
    found: set[type] = set()
    if isinstance(hint, type) and hint.__module__.startswith("abacus."):
        found.add(hint)
    origin = typing.get_origin(hint)
    if isinstance(origin, type) and origin.__module__.startswith("abacus."):
        found.add(origin)
    for argument in typing.get_args(hint):
        found |= package_types_in(argument)
    return found


def root_callables(root: frozenset[str]) -> Iterator[tuple[str, Callable[..., object]]]:
    """Each root function, and each public method or property of each root class."""
    for name in sorted(root):
        obj = getattr(abacus, name)
        if inspect.isfunction(obj):
            yield name, obj
        elif inspect.isclass(obj):
            for attribute, value in vars(obj).items():
                if attribute.startswith("_") and attribute != "__init__":
                    continue
                if isinstance(value, (classmethod, staticmethod)):
                    value = value.__func__
                if isinstance(value, property):
                    value = value.fget
                if inspect.isfunction(value):
                    yield f"{name}.{attribute}", value


def public_hints(function: Callable[..., object]) -> list[object]:
    """The annotations of a callable's public parameters and its return.

    A dataclass field named with an underscore still appears in the generated
    ``__init__``; ``Server._methods`` is one. It is private by name, so the
    types it is written in say nothing about the public surface.
    """
    hints = typing.get_type_hints(function)
    return [hint for name, hint in hints.items() if not name.startswith("_")]


@pytest.fixture(scope="module")
def root() -> frozenset[str]:
    return frozenset(abacus.__all__)


def test_every_module_declares_something() -> None:
    """No module is exempt from the checks below by declaring nothing.

    Eight failed this, ``errors`` and ``protocol`` among them.
    """
    silent = sorted(
        name for name, module in modules().items() if getattr(module, "__all__", None) is None
    )
    assert not silent, f"modules carrying no __all__: {silent}"


def test_a_module_declares_every_public_name_it_defines() -> None:
    """``__all__`` is not a curated subset of what the module defines.

    The root is the exception and is curated on purpose; it is checked from the
    other side below. Nineteen names failed this, twelve of them tool bodies
    that are now private rather than declared.
    """
    undeclared: dict[str, list[str]] = {}
    for module, names in declared().items():
        if module == "abacus":
            continue
        missing = sorted(definitions(module) - names)
        if missing:
            undeclared[module] = missing
    assert not undeclared, f"defined and absent from the module's __all__: {undeclared}"


def test_a_declaration_names_only_what_the_module_defines_or_a_pinned_reexport() -> None:
    """A declared name that the module does not define is a decision, pinned above.

    Its first run found ``vol`` declaring ``Method``: imported from
    ``moneyness`` and used for nothing else, so the import was dropped rather
    than pinned.
    """
    reexports = {
        module: frozenset(names - definitions(module))
        for module, names in declared().items()
        if module != "abacus"
    }
    reexports = {module: names for module, names in reexports.items() if names}
    pinned = {module: names for module, names in PINNED_REEXPORTS.items() if names}
    assert reexports == pinned


def test_every_declared_name_exists() -> None:
    """A declaration lists only names the module has."""
    found = modules()
    phantom = {
        module: sorted(name for name in names if not hasattr(found[module], name))
        for module, names in declared().items()
    }
    phantom = {module: names for module, names in phantom.items() if names}
    assert not phantom, f"declared but not attributes of the module: {phantom}"


def test_every_root_name_is_declared_by_exactly_one_module_that_defines_it(
    root: frozenset[str],
) -> None:
    """The curated root draws only on what the modules say is public.

    Before, 26 of the root's 31 names were declared by no module. Exactly one
    defining owner, because a name two modules define means the root's import
    line chose between them, and nothing records which.
    """
    owners = Counter(
        name
        for module, names in declared().items()
        if module != "abacus"
        for name in names & definitions(module)
    )
    problems = {name: owners[name] for name in root - {"__version__"} if owners[name] != 1}
    assert not problems, f"root names by number of defining modules declaring them: {problems}"


def test_the_root_exports_each_name_once() -> None:
    """``__all__`` does not repeat itself."""
    repeated = {name: count for name, count in Counter(abacus.__all__).items() if count > 1}
    assert not repeated, f"repeated in abacus.__all__: {repeated}"


def test_every_cross_module_import_names_a_declared_name() -> None:
    """The check made at the import that relies on a declaration.

    Every public name imported from another abacus module, anywhere in the
    repository's code, is one that module declares. On main, 68 distinct names
    were not, imported 240 times between them.
    """
    surfaces = declared()
    undeclared: dict[str, list[str]] = {}
    for path in consumer_files():
        package = owning_package(path)
        for module, name in package_imports_in(path.read_text(encoding="utf-8"), package):
            if name not in surfaces.get(module, frozenset()):
                where = str(path.relative_to(REPOSITORY))
                undeclared.setdefault(f"{module}.{name}", []).append(where)
    assert not undeclared, f"imported from a module that does not declare it: {undeclared}"


def test_a_type_in_a_reachable_signature_is_itself_reachable(root: frozenset[str]) -> None:
    """A caller who can call something can name what it takes and returns."""
    unreachable: dict[str, list[str]] = {}
    for label, function in root_callables(root):
        mentioned = set().union(*(package_types_in(hint) for hint in public_hints(function)))
        missing = sorted({kind.__name__ for kind in mentioned} - root)
        if missing:
            unreachable[label] = missing
    assert not unreachable, (
        f"reachable from abacus, written in types it does not export: {unreachable}"
    )


def consumer_files() -> list[Path]:
    """Every Python file in the repository that could import from abacus."""
    return sorted(
        path
        for directory in CONSUMERS
        for path in (REPOSITORY / directory).rglob("*.py")
        if ".venv" not in path.parts
    )


def owning_package(path: Path) -> str | None:
    """The dotted package a file belongs to, or ``None`` outside ``src/abacus``."""
    source = REPOSITORY / "src"
    if source / "abacus" not in path.parents:
        return None
    return ".".join(path.relative_to(source).parent.parts)


def test_the_probes_read_a_surface_worth_checking(root: frozenset[str]) -> None:
    """The checks above are not passing because they found nothing.

    Loose bounds that say a surface was read, and nothing about which modules
    declare — that is the checks' job, and a probe failing for their reason
    would report one defect twice.
    """
    assert len(modules()) >= 18, f"only {len(modules())} modules were found"
    assert "abacus.transports.http" in modules(), "the subpackage was not walked"
    assert len(root) >= 25, f"the root came back as {len(root)} names"
    imports = [
        pair
        for path in consumer_files()
        for pair in package_imports_in(path.read_text(encoding="utf-8"), owning_package(path))
    ]
    assert len(imports) >= 250, f"only {len(imports)} cross-module imports were read"
    assert ("abacus.protocol", "META_PROTOCOL_VERSION") in imports, "a ../ import was missed"
    assert len(list(root_callables(root))) >= 30, "too few signatures were read"


def test_the_definition_probe_reads_each_kind_and_skips_each_non_definition() -> None:
    """The definition probe, against source written for it."""
    found = public_names_in(PROBE_SOURCE)
    assert found == {
        "TTL_MS",
        "Handler",
        "ANNOTATED",
        "PAIR_ONE",
        "PAIR_TWO",
        "public_function",
        "public_coroutine",
        "PublicClass",
    }, f"the probe read {sorted(found)}"
    for absent in ("math", "Callable", "_PRIVATE", "__all__", "local_name", "attribute"):
        assert absent not in found, f"the probe counted {absent!r} as a definition"


def test_the_import_probe_resolves_each_form_and_skips_each_non_claim() -> None:
    """The import probe, against source written for it.

    Two dots from ``abacus.transports`` is ``abacus``, one dot is the
    subpackage itself. An aliased import is checked under the exported name.
    Skipped: a private name, a bare ``from . import``, the root (checked on its
    own), and — outside the package — every relative import.
    """
    assert package_imports_in(IMPORT_PROBE, "abacus.transports") == [
        ("abacus.errors", "ProtocolError"),
        ("abacus.protocol", "META_PROTOCOL_VERSION"),
        ("abacus.protocol", "error_response"),
        ("abacus.transports.stdio", "serve_stdio"),
    ]
    assert package_imports_in(IMPORT_PROBE, None) == [("abacus.errors", "ProtocolError")]


def test_the_signature_walk_skips_a_private_dataclass_field() -> None:
    """``Server.__init__`` takes ``_methods``, a private field, typed in ``Method``.

    Before the walk skipped private parameters it reported ``Method`` as
    missing from the root on that account alone. The public registrations —
    ``register``, ``method`` and ``on_notification`` — take handlers written as
    callables over root types.
    """
    every = typing.get_type_hints(Server.__init__)
    assert "_methods" in every, "the case this guards no longer exists"
    kinds = set().union(*(package_types_in(hint) for hint in public_hints(Server.__init__)))
    assert {kind.__name__ for kind in kinds} <= frozenset(abacus.__all__)
    nested = typing.Optional[tuple[list[Request], dict[str, Notification]]]  # noqa: UP045
    assert package_types_in(nested) == {Request, Notification}
    assert package_types_in(int | str | list[float]) == set()
