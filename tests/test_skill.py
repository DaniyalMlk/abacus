"""The skill is checked against the server it describes, and executed.

Documentation that names a tool which has since been renamed is worse than no
documentation, because it is confidently wrong and a model has no way to tell.
So nothing in ``skill/SKILL.md`` is taken on trust here:

* every tool it names exists, and every tool that exists is named by it;
* every worked transcript is run against a live server, in order, with the
  handles threaded between calls exactly as written;
* the tool surface is measured against a budget, because the listing is loaded
  into a context window before any work happens and it grows with every group.

The transcripts are the part that matters most. A prose example goes stale
silently — the argument names drift, a field is renamed, a required key is
added — and the only way to find out is for somebody to copy it and fail. These
run, so the file fails instead.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from abacus.analytics import default_registry
from abacus.cli import build_server
from abacus.client import Client, InProcessTransport

SKILL = Path(__file__).resolve().parent.parent / "skill" / "SKILL.md"

#: Ceiling on the whole tool listing, serialised as JSON, in characters.
#: Measured at 115,162 when this was written, so the headroom is about a fifth.
#: The number is a budget rather than a limit of the protocol: `tools/list`
#: paginates, and nothing breaks above it. What it buys is that adding a group
#: which pushes the surface past it is a decision somebody makes rather than a
#: drift nobody notices.
LISTING_BUDGET = 140_000

#: Ceiling on any single tool. The largest today is `bond_spreads` at 10,593,
#: and the realistic failure is one tool growing without bound rather than the
#: total creeping, so this is the tighter of the two constraints.
TOOL_BUDGET = 12_000


@pytest.fixture(scope="module")
def skill_text() -> str:
    assert SKILL.is_file(), f"the skill is missing from {SKILL}"
    return SKILL.read_text()


@pytest.fixture
def client() -> Iterator[Client]:
    with Client(InProcessTransport(build_server())) as connected:
        yield connected


# -- the skill and the server agree ------------------------------------------


def _tool_names() -> set[str]:
    return {tool.name for tool in default_registry()}


def _named_in(text: str) -> set[str]:
    """Every registered tool name the text mentions, in prose or in a transcript."""
    known = _tool_names()
    return {name for name in known if re.search(rf"\b{re.escape(name)}\b", text)}


def test_every_tool_the_skill_names_exists(skill_text: str) -> None:
    """Anything in backticks that looks like a tool name has to be one.

    The failure this catches is a rename: the tool moves, the skill does not,
    and a model is told to call something that will come back as an unknown
    tool. Matching the shape rather than a list means a newly invented name is
    caught too.
    """
    known = _tool_names()
    mentioned = set(re.findall(r"`([a-z][a-z0-9_]{6,})`", skill_text))
    # Words in backticks that are arguments or values rather than tool names.
    not_tools = {
        name
        for name in mentioned
        if name in {"periodsPerYear", "spotShifts", "volShifts", "isError", "transcript"}
    }
    for name in mentioned - not_tools:
        if "_" in name:  # tool names are snake_case; arguments here are not
            assert name in known, f"the skill names {name}, which the server does not register"


def test_every_tool_the_server_registers_is_named_by_the_skill(skill_text: str) -> None:
    """A tool nobody is told about might as well not be registered.

    This is the direction that fails as the server grows rather than as it
    changes: a phase lands, five tools appear, and the skill still describes the
    surface as it was.
    """
    missing = _tool_names() - _named_in(skill_text)
    assert not missing, f"the skill does not mention {sorted(missing)}"


def test_the_skill_carries_the_frontmatter_a_loader_needs(skill_text: str) -> None:
    assert skill_text.startswith("---\n")
    frontmatter = skill_text.split("---", 2)[1]
    assert re.search(r"^name:\s*\S+", frontmatter, re.MULTILINE)
    assert re.search(r"^description:", frontmatter, re.MULTILINE)


# -- the budget --------------------------------------------------------------


def test_the_tool_listing_fits_its_budget() -> None:
    """The whole surface, serialised as a client receives it."""
    listing = json.dumps([tool.describe() for tool in default_registry()])
    assert len(listing) <= LISTING_BUDGET, (
        f"the tool listing is {len(listing)} characters against a budget of "
        f"{LISTING_BUDGET}. Either trim the schemas or raise the budget "
        "deliberately — this is meant to be a decision, not a failure."
    )


def test_no_single_tool_takes_more_than_its_share() -> None:
    oversized = {
        tool.name: len(json.dumps(tool.describe()))
        for tool in default_registry()
        if len(json.dumps(tool.describe())) > TOOL_BUDGET
    }
    assert not oversized, f"over the per-tool budget of {TOOL_BUDGET}: {oversized}"


def test_the_skill_states_the_budget_it_is_checked_against(skill_text: str) -> None:
    """The figures in the skill are the ones enforced here, or they are fiction."""
    assert f"{LISTING_BUDGET:,}" in skill_text
    assert f"{TOOL_BUDGET:,}" in skill_text


def test_the_skill_is_much_smaller_than_the_surface_it_describes(skill_text: str) -> None:
    """An overview the size of the thing it summarises is not an overview."""
    listing = json.dumps([tool.describe() for tool in default_registry()])
    assert len(skill_text) < len(listing) / 5


# -- the transcripts run -----------------------------------------------------


def _transcripts(text: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """Every ```transcript block, grouped under the heading above it.

    Grouping matters because the handles are threaded within a transcript and
    not across them: two sequences that happened to share a capture name would
    otherwise leak state between them and pass for the wrong reason.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    heading = "(none)"
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            heading = line[3:].strip()
        grouped.setdefault(heading, [])

    heading = "(none)"
    # `.` must not match newlines in the heading branch or the alternation
    # swallows the whole document in one match; the body branch needs it to.
    pattern = re.compile(r"^## ([^\n]+)$|^```transcript\n((?s:.*?))^```$", re.MULTILINE)
    for match in pattern.finditer(text):
        if match.group(1) is not None:
            heading = match.group(1).strip()
        else:
            grouped.setdefault(heading, []).append(json.loads(match.group(2)))
    return [(name, steps) for name, steps in grouped.items() if steps]


def _returns(periods: int = 260, assets: int = 4, *, seed: int = 5) -> list[list[float]]:
    """A one-factor return panel, for the transcript that asks for `$returns`."""
    rng = random.Random(seed)
    factor = [rng.gauss(0.0003, 0.008) for _ in range(periods)]
    return [
        [factor[t] * (0.7 + 0.15 * a) + rng.gauss(0.0, 0.004) for a in range(assets)]
        for t in range(periods)
    ]


def _trials(periods: int = 500, count: int = 20, *, seed: int = 9) -> list[list[float]]:
    """A parameter sweep over one idea, for the transcript that asks for `$trials`."""
    rng = random.Random(seed)
    factor = [rng.gauss(0.0002, 0.01) for _ in range(periods)]
    return [
        [factor[t] + rng.gauss(0.0, 0.002) for _ in range(count)] for t in range(periods)
    ]


def _substitute(value: Any, bindings: dict[str, Any]) -> Any:
    """Replace `$name` placeholders, whole-string only.

    Whole-string rather than interpolated because every placeholder here stands
    for a handle or a matrix — a structure, not a fragment of text — and a
    substitution that spliced one into a longer string would produce something
    that is not JSON-shaped and fail confusingly.
    """
    if isinstance(value, str) and value.startswith("$"):
        name = value[1:]
        assert name in bindings, f"the transcript uses ${name} before anything captured it"
        return bindings[name]
    if isinstance(value, dict):
        return {key: _substitute(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, bindings) for item in value]
    return value


def test_the_file_holds_the_transcripts_it_claims_to(skill_text: str) -> None:
    """Three of them, so a heading renamed into nothing does not pass silently."""
    transcripts = _transcripts(skill_text)
    assert len(transcripts) == 3, [name for name, _ in transcripts]
    assert all(len(steps) >= 3 for _, steps in transcripts)


def test_every_transcript_runs_against_the_server(client: Client, skill_text: str) -> None:
    """Each sequence, in order, with the handles threaded as written.

    This is the test that keeps the skill honest. An argument renamed in a
    schema, a required field added, a handle that no longer round-trips: all of
    them show up here as a refusal on a call the documentation says works.
    """
    transcripts = _transcripts(skill_text)
    assert transcripts, "no transcripts found — the fence language may have changed"

    for name, steps in transcripts:
        bindings: dict[str, Any] = {
            "returns": _returns(),
            "trials": _trials(),
        }
        for index, step in enumerate(steps):
            arguments = _substitute(step["arguments"], bindings)
            exchange = client.call_tool(step["tool"], arguments)
            assert exchange.error is None, f"{name} step {index}: {exchange.error}"
            result = exchange.result
            assert result is not None
            assert result["isError"] is False, (
                f"{name} step {index} ({step['tool']}) was refused: "
                f"{result['structuredContent']}"
            )
            structured = result["structuredContent"]
            # A strict reader would reject NaN or Infinity, and the documented
            # path is exactly where an unserialisable number would be worst.
            json.dumps(structured, allow_nan=False)

            for binding, field in step.get("capture", {}).items():
                assert field in structured, (
                    f"{name} step {index} captures {field}, which is not in the result"
                )
                bindings[binding] = structured[field]


def test_the_book_transcript_threads_one_handle_through(client: Client, skill_text: str) -> None:
    """The handle captured in the first step is the one the later steps use.

    Running the sequence proves the calls work; this proves they are the same
    book, which is the thing the transcript is actually demonstrating.
    """
    transcripts = dict(_transcripts(skill_text))
    steps = transcripts["Worked transcript: a book and its risk"]
    assert steps[0]["capture"] == {"book": "handle"}
    assert all(step["arguments"].get("handle") == "$book" for step in steps[1:])

    opened = client.call_tool(steps[0]["tool"], steps[0]["arguments"]).result
    assert opened is not None
    handle = opened["structuredContent"]["handle"]

    greeks = client.call_tool("position_book_greeks", {"handle": handle}).result
    assert greeks is not None
    assert greeks["isError"] is False
    assert len(greeks["structuredContent"]["legs"]) == 3
