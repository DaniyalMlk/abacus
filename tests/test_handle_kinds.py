"""A handle says what it holds, and a tool refuses one that holds something else.

Three families of tool mint handles -- position books, return moments and
discount curves -- and a handle is only authenticated, not typed, by its MAC. So
each family stamps a kind into its payload and checks it before reading
anything else. The checks matter as soon as the families share a key, and the
test therefore builds all three under one: with a key apiece a foreign handle
fails its MAC, and the kind check is never reached.

The book family had no kind until this test existed. A moments handle passed to
``describe_position_book`` was parsed as a book and refused for the first book
field it lacked -- "spot must be a positive number" at ``/spot``, to a caller
who had sent no spot.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import pytest

from abacus import book, curves, risk
from abacus.book import BOOK_KIND
from abacus.curves import CURVE_KIND
from abacus.handles import Minter
from abacus.risk import MOMENTS_KIND
from abacus.tools import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    """Every handle-minting family, under one key."""
    shared = Minter(key=b"one key for every family of handle")
    built = ToolRegistry()
    for family in (book, risk, curves):
        built = family.register(built, minter=shared)
    return built


def mint_book(registry: ToolRegistry) -> str:
    leg = {"instrument": "call", "quantity": 1.0, "strike": 100.0, "time": 0.5, "vol": 0.2}
    return handle_of(registry, "open_position_book", {"spot": 100.0, "rate": 0.04, "legs": [leg]})


def mint_moments(registry: ToolRegistry) -> str:
    draws = random.Random(7)
    returns = [[draws.gauss(0.0, 0.01) for _ in range(3)] for _ in range(60)]
    return handle_of(
        registry, "estimate_return_moments", {"returns": returns, "periodsPerYear": 252}
    )


def mint_curve(registry: ToolRegistry) -> str:
    quotes = {
        "reference": "2021-01-05",
        "basis": "ACT_365F",
        "deposits": [{"maturity": "2021-07-05", "rate": 0.002, "basis": "ACT_360"}],
        "swaps": [
            {
                "maturity": "2023-01-05",
                "rate": 0.006,
                "frequency": "SEMI_ANNUAL",
                "basis": "THIRTY_360_BOND",
            }
        ],
    }
    return handle_of(registry, "bootstrap_discount_curve", quotes)


def handle_of(registry: ToolRegistry, name: str, args: dict[str, Any]) -> str:
    result = registry.call(name, args)
    assert result["isError"] is False, result["structuredContent"]
    handle: str = result["structuredContent"]["handle"]
    return handle


#: A tool from each family that reads a handle, with whatever else its schema
#: requires, so that the refusal can only be about the handle.
READERS: dict[str, tuple[str, dict[str, Any]]] = {
    BOOK_KIND: ("describe_position_book", {}),
    MOMENTS_KIND: ("risk_parity_weights", {}),
    CURVE_KIND: ("discount_curve_rates", {"dates": ["2022-01-05"]}),
}

MINTERS: dict[str, Callable[[ToolRegistry], str]] = {
    BOOK_KIND: mint_book,
    MOMENTS_KIND: mint_moments,
    CURVE_KIND: mint_curve,
}

#: Every ordered pair of distinct kinds: six crossings.
CROSSINGS = [(held, wanted) for held in MINTERS for wanted in READERS if held != wanted]


def test_the_kinds_are_distinct() -> None:
    assert len({BOOK_KIND, MOMENTS_KIND, CURVE_KIND}) == 3


@pytest.mark.parametrize("kind", list(READERS))
def test_each_family_reads_its_own_handle(registry: ToolRegistry, kind: str) -> None:
    name, extra = READERS[kind]
    result = registry.call(name, {"handle": MINTERS[kind](registry), **extra})
    assert result["isError"] is False, result["structuredContent"]


@pytest.mark.parametrize(("held", "wanted"), CROSSINGS, ids=[f"{h}-to-{w}" for h, w in CROSSINGS])
def test_a_foreign_handle_is_refused_by_name(
    registry: ToolRegistry, held: str, wanted: str
) -> None:
    name, extra = READERS[wanted]
    result = registry.call(name, {"handle": MINTERS[held](registry), **extra})
    assert result["isError"] is True
    error = result["structuredContent"]["error"]
    assert error["kind"] == "handle_wrong_kind", error
    (detail,) = error["details"]
    assert detail["path"] == "/handle"
    assert detail["message"] == f"handle holds {held!r}"


def test_a_book_reads_back_without_the_kind(registry: ToolRegistry) -> None:
    """The kind is the handle's: the book echoed back is the book that was opened."""
    result = registry.call("describe_position_book", {"handle": mint_book(registry)})
    assert "kind" not in result["structuredContent"]["book"]


def test_a_book_handle_minted_without_a_kind_is_refused() -> None:
    """The shape every book handle had before the kind: authenticated, untyped."""
    shared = Minter(key=b"one key for every family of handle")
    registry = book.register(ToolRegistry(), minter=shared)
    shares = {"instrument": "underlying", "quantity": 1.0}
    untyped = shared.mint({"spot": 100.0, "rate": 0.04, "legs": [shares]})
    result = registry.call("describe_position_book", {"handle": untyped})
    assert result["structuredContent"]["error"]["kind"] == "handle_wrong_kind"
