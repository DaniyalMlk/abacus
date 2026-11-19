"""The roadmap agrees with itself, and with the release that happened.

Phase 8's "A first release on the index, which waits on the publisher being
registered there for this project and for the library it depends on" stayed
unchecked after `abacus-mcp` 0.1.0 was on the index, and nothing read the
roadmap to notice. So this file reads it:

* the release item is pinned as checked, since a published version is
  permanent, and no other open item may mention a release however worded;
* the phases are numbered with no duplicate, no gap and no descent;
* every cross-reference names a phase that exists, and points backwards unless
  it is a note saying where a deferred item was settled.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

ROADMAP = Path(__file__).resolve().parent.parent / "ROADMAP.md"

HEADING = re.compile(r"^## Phase (\d+) — (.+)$", re.MULTILINE)
REFERENCE = re.compile(r"\b[Pp]hases? (\d+)(?: (and|to) (\d+))?")
RELEASE_ITEM = re.compile(r"^- \[( |x)\] A first release on the index", re.MULTILINE)
OPEN_RELEASE_ITEM = re.compile(r"^- \[ \] .*\brelease\b", re.MULTILINE | re.IGNORECASE)

#: The words that make a forward reference a pointer to where a deferred item
#: was closed, immediately before ``phase N``.
RESOLUTION = re.compile(r"\b(?:settled|cleared|closed|resolved|done|landed) (?:in|by) $")


def phase_numbers(roadmap: str) -> list[int]:
    return [int(number) for number, _ in HEADING.findall(roadmap)]


def numbering_faults(numbers: list[int]) -> list[str]:
    faults: list[str] = []
    repeated = sorted(n for n, count in Counter(numbers).items() if count > 1)
    if repeated:
        faults.append(f"repeated {repeated}")
    expected = set(range(1, len(numbers) + 1))
    if set(numbers) != expected:
        faults.append(
            f"missing {sorted(expected - set(numbers))}, "
            f"unexpected {sorted(set(numbers) - expected)}"
        )
    descents = [
        (numbers[i], numbers[i + 1])
        for i in range(len(numbers) - 1)
        if numbers[i + 1] <= numbers[i]
    ]
    if descents:
        faults.append(f"out of order at {descents}")
    return faults


def cross_references(roadmap: str) -> list[tuple[int, int, bool]]:
    """Every prose reference as (written in, named, is a resolution note).

    Only a heading's own number is skipped, so a phase its title names is still
    a reference. A note is a resolution note when the words before ``phase``
    on the same line say the item was settled there.
    """
    bounds = [(match.start(), int(match.group(1))) for match in HEADING.finditer(roadmap)]
    own_numbers = {start + len("## ") for start, _ in bounds}
    found: list[tuple[int, int, bool]] = []
    for match in REFERENCE.finditer(roadmap):
        if match.start() in own_numbers:
            continue
        written_in = 0
        for start, number in bounds:
            if start > match.start():
                break
            written_in = number
        line_start = roadmap.rfind("\n", 0, match.start()) + 1
        resolution = RESOLUTION.search(roadmap[line_start : match.start()]) is not None
        first, joiner, last = int(match.group(1)), match.group(2), match.group(3)
        if joiner == "to":
            named = list(range(first, int(last) + 1))
        elif joiner == "and":
            named = [first, int(last)]
        else:
            named = [first]
        found.extend((written_in, n, resolution) for n in named)
    return found


def misplaced_references(roadmap: str) -> list[tuple[int, int]]:
    """References naming the phase they are in, or a later one, outside a note."""
    return [
        (at, named)
        for at, named, resolution in cross_references(roadmap)
        if named >= at and not (resolution and named > at)
    ]


def release_recorded(roadmap: str) -> bool:
    found: list[str] = RELEASE_ITEM.findall(roadmap)
    assert len(found) == 1, f"expected one release item in the roadmap, found {len(found)}"
    return found[0] == "x"


@pytest.fixture(scope="module")
def roadmap() -> str:
    return ROADMAP.read_text(encoding="utf-8")


def test_the_release_stays_recorded(roadmap: str) -> None:
    assert release_recorded(roadmap), "the first release is on the index"


def test_no_open_item_waits_on_a_release(roadmap: str) -> None:
    assert OPEN_RELEASE_ITEM.findall(roadmap) == []


def test_the_numbering_is_sound(roadmap: str) -> None:
    assert numbering_faults(phase_numbers(roadmap)) == []


def test_every_reference_names_a_phase_that_exists(roadmap: str) -> None:
    numbers = set(phase_numbers(roadmap))
    assert sorted({named for _, named, _ in cross_references(roadmap)} - numbers) == []


def test_every_reference_points_backwards_or_is_a_resolution_note(roadmap: str) -> None:
    assert misplaced_references(roadmap) == []


def test_the_probe_reads_the_references_that_are_there(roadmap: str) -> None:
    """Both reference rules pass vacuously if the pattern stops matching."""
    assert cross_references(roadmap) == [(20, 19, False), (23, 22, False)]


# --- the rules against text built for them -------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("## Phase 1 — A\n\nsettled in phase 2.\n\n## Phase 2 — B\n", []),
        ("## Phase 1 — A\n\nSee phase 2.\n\n## Phase 2 — B\n", [(1, 2)]),
        ("## Phase 1 — A\n\n## Phase 2 — B\n\nsettled in phase 2.\n", [(2, 2)]),
        ("## Phase 1 — A\n\n## Phase 2 — B\n\nAs phase 1 did.\n", []),
        ("## Phase 1 — A\n\n## Phase 2 — Before phase 3\n\n## Phase 3 — C\n", [(2, 3)]),
        ("Read phase 1 first.\n\n## Phase 1 — A\n", [(0, 1)]),
    ],
)
def test_misplaced_references_on_built_text(text: str, expected: list[tuple[int, int]]) -> None:
    assert misplaced_references(text) == expected


def test_a_resolution_word_on_an_earlier_line_does_not_count() -> None:
    text = "## Phase 1 — A\n\nIt was settled in\nphase 2.\n\n## Phase 2 — B\n"
    assert misplaced_references(text) == [(1, 2)]


@pytest.mark.parametrize(
    ("numbers", "fragment"),
    [([1, 2, 2, 4], "repeated [2]"), ([1, 2, 4], "missing [3]"), ([1, 3, 2], "out of order")],
)
def test_numbering_faults_fires_on_each_fault(numbers: list[int], fragment: str) -> None:
    assert any(fragment in fault for fault in numbering_faults(numbers))


def test_numbering_faults_is_silent_on_a_clean_list() -> None:
    assert numbering_faults([1, 2, 3]) == []


def test_an_open_release_item_is_seen_however_it_is_worded() -> None:
    assert len(OPEN_RELEASE_ITEM.findall("- [ ] A first release on the package index\n")) == 1
    assert OPEN_RELEASE_ITEM.findall("- [x] A first release on the index: 0.1.0\n") == []


@pytest.mark.parametrize(("box", "expected"), [("x", True), (" ", False)])
def test_release_recorded_reads_the_box(box: str, expected: bool) -> None:
    assert release_recorded(f"- [{box}] A first release on the index: 0.1.0\n") is expected
