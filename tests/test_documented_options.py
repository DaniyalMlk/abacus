"""The options nothing had ever chosen, each exercised by a test that can fail.

``tests/test_options.py`` asks mechanically whether every documented option is
supplied somewhere. These are the answers for the six it found unsupplied.

Each one checks what the option *governs*, not that passing it is allowed: a
call that supplies an argument and asserts nothing would satisfy the gate and
establish nothing, which is the failure mode a gate like that invites. The two
that reach a caller — the scenario grid's axes — are driven over the wire as
well, because a message a model has to act on is only as good as what arrives
at the other end of the transport.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from abacus.book import (
    _AXIS_CONVENTION,
    DEFAULT_SPOT_SHIFTS,
    DEFAULT_VOL_SHIFTS,
    Book,
    Leg,
    _shift_axis,
    value_and_greeks,
)
from abacus.cli import build_server
from abacus.client import Client, InProcessTransport
from abacus.conformance import CHECKS, run
from abacus.server import capabilities
from abacus.tools import DomainError

SPOT = 100.0
RATE = 0.02


def option_leg(**overrides: Any) -> Leg:
    fields: dict[str, Any] = {
        "instrument": "call",
        "quantity": 10.0,
        "strike": 100.0,
        "time": 0.5,
        "vol": 0.20,
    }
    fields.update(overrides)
    return Leg(**fields)


def one_option_book() -> Book:
    return Book(spot=SPOT, rate=RATE, legs=[option_leg()])


class TestTheVolatilityShift:
    """``Leg.inputs(vol_shift=)`` and ``value_and_greeks(vol_shift=)``."""

    def test_the_shift_is_added_to_the_leg_and_is_not_relative_to_it(self) -> None:
        # The convention the whole of this branch turns on. A shift of 0.05 on a
        # 20-vol leg gives 25 vol, not 21 -- which is what a relative shift of
        # five percent would give, and the two differ by enough to tell apart.
        leg = option_leg(vol=0.20)
        assert leg.inputs(SPOT, RATE, vol_shift=0.05).vol == pytest.approx(0.25, abs=1e-15)
        assert leg.inputs(SPOT, RATE, vol_shift=-0.05).vol == pytest.approx(0.15, abs=1e-15)
        assert leg.inputs(SPOT, RATE).vol == pytest.approx(0.20, abs=1e-15)
        # And it is the same number of points whatever the leg's own volatility,
        # which a relative shift would not be.
        for own in (0.05, 0.20, 0.80):
            shifted = option_leg(vol=own).inputs(SPOT, RATE, vol_shift=0.05).vol
            assert shifted - own == pytest.approx(0.05, abs=1e-15)

    def test_a_shift_past_zero_is_floored_rather_than_refused(self) -> None:
        # Documented behaviour: taking a 15-vol option down 20 points is a
        # legitimate scenario and the answer at zero volatility is the
        # discounted intrinsic.
        leg = option_leg(vol=0.15)
        assert leg.inputs(SPOT, RATE, vol_shift=-0.20).vol == 0.0
        assert leg.inputs(SPOT, RATE, vol_shift=-1e9).vol == 0.0

    def test_the_shift_moves_the_value_by_the_vega_the_same_call_reports(self) -> None:
        # The check that ties the option to the arithmetic rather than to
        # itself: vega is the derivative with respect to volatility, so a small
        # shift has to move the value by vega times the shift. Vega here is per
        # unit of volatility, so no scaling by a hundred enters -- and that is
        # the mistake this would catch.
        book = one_option_book()
        base = value_and_greeks(book, spot=SPOT)
        vega = base["greeks"]["vega"]
        for shift in (1e-4, -1e-4, 1e-3):
            moved = value_and_greeks(book, spot=SPOT, vol_shift=shift)
            predicted = base["value"] + vega * shift
            assert moved["value"] == pytest.approx(predicted, rel=2e-3), shift
        # A spread, so that a shift which did nothing would not pass: ten points
        # moves this book by more than a currency unit.
        assert value_and_greeks(book, spot=SPOT, vol_shift=0.10)["value"] - base["value"] > 1.0

    def test_the_underlying_leg_does_not_care(self) -> None:
        # The shift reaches option legs only, so a book of stock is invariant
        # under it -- which is also why a grid of volatility rows on such a book
        # is flat rather than wrong.
        stock = Book(spot=SPOT, rate=RATE, legs=[Leg(instrument="underlying", quantity=100.0)])
        flat = value_and_greeks(stock, spot=SPOT)
        for shift in (-0.5, 0.0, 0.5, 5.0):
            assert value_and_greeks(stock, spot=SPOT, vol_shift=shift) == flat


class TestTheAxisConventions:
    """What a caller is told when a scenario axis cannot take their number."""

    def test_each_axis_names_its_own_convention(self) -> None:
        spot = _AXIS_CONVENTION["spotShifts"]
        vol = _AXIS_CONVENTION["volShifts"]
        assert spot != vol, "one shared sentence for two conventions is what went wrong"
        assert "relative to the current spot" in spot
        assert "0.1" in spot
        assert "volatility points" in vol
        assert "0.05" in vol
        # The sentence that was wrong on the volatility axis, and the only thing
        # it could have meant there.
        assert "percent" not in vol
        assert "not relative to the leg" in vol.lower()

    @pytest.mark.parametrize(
        ("name", "limit", "expected"),
        [("spotShifts", 0.99, "spotShifts"), ("volShifts", 5.0, "volShifts")],
    )
    def test_the_refusal_carries_that_axis_sentence(
        self, name: str, limit: float, expected: str
    ) -> None:
        with pytest.raises(DomainError) as raised:
            _shift_axis([10.0], name=name, default=(0.0,), limit=limit)
        message = str(raised.value)
        assert _AXIS_CONVENTION[expected] in message
        other = "volShifts" if expected == "spotShifts" else "spotShifts"
        assert _AXIS_CONVENTION[other] not in message
        assert f"±{limit:g}" in message

    def test_the_other_refusals_are_unchanged(self) -> None:
        for bad, fragment in (
            ("not-a-list", "non-empty array"),
            ([], "non-empty array"),
            (["x"], "only numbers"),
            ([True], "only numbers"),
            ([float("nan")], "only finite"),
            ([float("inf")], "only finite"),
        ):
            with pytest.raises(DomainError, match=fragment):
                _shift_axis(bad, name="volShifts", default=(0.0,), limit=5.0)

    def test_an_omitted_axis_falls_back_to_its_default(self) -> None:
        spot = _shift_axis(None, name="spotShifts", default=DEFAULT_SPOT_SHIFTS, limit=0.99)
        vol = _shift_axis(None, name="volShifts", default=DEFAULT_VOL_SHIFTS, limit=5.0)
        assert spot == list(DEFAULT_SPOT_SHIFTS)
        assert vol == list(DEFAULT_VOL_SHIFTS)
        # The volatility default is five points either way, in the units the
        # sentence above describes.
        assert DEFAULT_VOL_SHIFTS == (-0.05, 0.0, 0.05)


class TestTheServerCapabilities:
    """``capabilities(prompts=)`` and ``capabilities(resources=)``."""

    def test_an_absent_argument_is_absent_from_the_map(self) -> None:
        # Declaring a capability is a promise to answer the corresponding
        # requests, so the distinction between omitted and empty is the whole
        # point of the function.
        assert capabilities() == {}
        assert capabilities(tools={}) == {"tools": {}}
        assert "resources" not in capabilities(tools={})
        assert "prompts" not in capabilities(tools={})

    def test_each_capability_appears_only_when_it_is_given(self) -> None:
        offered = capabilities(
            tools={"listChanged": False},
            resources={"subscribe": False},
            prompts={"listChanged": False},
            extensions={"x": 1},
        )
        assert offered == {
            "tools": {"listChanged": False},
            "resources": {"subscribe": False},
            "prompts": {"listChanged": False},
            "extensions": {"x": 1},
        }
        # An empty map is still a declaration, which is why `None` and `{}` are
        # not the same argument.
        assert capabilities(prompts={}) == {"prompts": {}}
        assert capabilities(prompts=None) == {}

    def test_this_server_promises_only_what_it_answers(self) -> None:
        # The reason the two flags exist, read off the handshake rather than off
        # the call that builds it: this server serves tools and neither
        # resources nor prompts, so the map must carry one key and not the other
        # two. A promise it cannot keep would send a client to a method that
        # answers "method not found".
        with Client(InProcessTransport(build_server())) as client:
            exchange = client.discover()
        assert exchange.error is None
        assert exchange.result is not None
        advertised = exchange.result["capabilities"]
        assert advertised == {"tools": {"listChanged": False}}
        assert "resources" not in advertised, "advertising resources this server cannot serve"
        assert "prompts" not in advertised, "advertising prompts this server cannot serve"


class TestTheConformanceSubset:
    """``run(checks=)``: which checks to run."""

    def test_a_subset_runs_exactly_that_subset(self) -> None:
        chosen = list(CHECKS[:3])
        with Client(InProcessTransport(build_server())) as client:
            report = run(client, checks=chosen)
        assert [outcome.name for outcome in report.outcomes] == [check.name for check in chosen]

    def test_an_empty_subset_runs_nothing_and_reports_nothing(self) -> None:
        # Worth pinning rather than asserting a refusal, because the truthful
        # reading of an empty run is the one that surprises: `ok` is "nothing
        # failed", and nothing ran, so it is True. The rendering is what keeps
        # that honest -- it states the check count, which is zero -- and the
        # command line never passes a subset, so an empty one is reachable only
        # from Python.
        with Client(InProcessTransport(build_server())) as client:
            report = run(client, checks=[])
        assert report.outcomes == []
        assert report.passed == [] and report.failures == [] and report.skipped == []
        assert report.ok is True
        assert "0 passed, 0 failed, 0 skipped, 0 checks" in report.render()

    def test_the_default_is_every_check(self) -> None:
        with Client(InProcessTransport(build_server())) as client:
            report = run(client)
        assert len(report.outcomes) == len(CHECKS)
        assert len(CHECKS) > 10, "a default of nothing much would make the subset meaningless"


class TestTheClientInfoBlock:
    """``Client.meta(include_client_info=)``."""

    def test_the_block_carries_the_client_info_unless_it_is_asked_not_to(self) -> None:
        with Client(InProcessTransport(build_server())) as client:
            with_info = client.meta()
            without = client.meta(include_client_info=False)
        extra = set(with_info) - set(without)
        assert len(extra) == 1, (with_info, without)
        key = extra.pop()
        assert "clientInfo" in key or "client_info" in key
        assert with_info[key]
        # Everything else is the same block, so the flag removes one key and
        # changes nothing else.
        assert {k: v for k, v in with_info.items() if k != key} == without

    def test_the_other_meta_arguments_still_apply_without_it(self) -> None:
        with Client(InProcessTransport(build_server())) as client:
            block = client.meta(
                version="2025-06-18",
                capabilities={"roots": {}},
                log_level="debug",
                include_client_info=False,
            )
        values = list(block.values())
        assert "2025-06-18" in values
        assert {"roots": {}} in values
        assert "debug" in values
        assert all("clientInfo" not in key for key in block)


class TestOverTheWire:
    """The axis messages as a caller actually receives them."""

    @pytest.fixture
    def client(self) -> Iterator[Client]:
        with Client(InProcessTransport(build_server())) as connected:
            yield connected

    @staticmethod
    def handle(client: Client) -> str:
        exchange = client.call_tool(
            "open_position_book",
            {
                "spot": SPOT,
                "rate": RATE,
                "legs": [
                    {
                        "instrument": "call",
                        "quantity": 10.0,
                        "strike": 100.0,
                        "time": 0.5,
                        "vol": 0.2,
                    }
                ],
            },
        )
        assert exchange.result is not None and exchange.result["isError"] is False
        handle = exchange.result["structuredContent"]["handle"]
        assert isinstance(handle, str)
        return handle

    def test_a_good_grid_comes_back_with_both_axes_labelled(self, client: Client) -> None:
        exchange = client.call_tool(
            "position_book_scenarios",
            {"handle": self.handle(client), "spotShifts": [-0.05, 0.0], "volShifts": [0.0, 0.05]},
        )
        assert exchange.error is None
        result = exchange.result
        assert result is not None and result["isError"] is False
        payload = result["structuredContent"]
        assert json.loads(result["content"][0]["text"]) == payload
        assert payload["volShifts"] == [0.0, 0.05]
        assert [row["volShift"] for row in payload["rows"]] == [0.0, 0.05]
        # Five volatility points is worth something on this book, so the rows
        # differ -- a grid whose volatility axis did nothing would be two equal
        # rows and the labels would be the only difference.
        first, second = (row["cells"][0]["value"] for row in payload["rows"])
        assert second > first

    @pytest.mark.parametrize("axis", ["spotShifts", "volShifts"])
    def test_a_bad_axis_is_refused_as_a_result_the_caller_can_repair(
        self, client: Client, axis: str
    ) -> None:
        # A result with isError set rather than a protocol error, because that is
        # what lets a model fix the arguments and try again, and it has to name
        # the offending element and the bound.
        arguments: dict[str, Any] = {"handle": self.handle(client), axis: [10.0]}
        exchange = client.call_tool("position_book_scenarios", arguments)
        assert exchange.error is None, (
            f"a fixable mistake became a protocol error: {exchange.error}"
        )
        result = exchange.result
        assert result is not None
        assert result["isError"] is True, result["structuredContent"]
        message = result["structuredContent"]["error"]["message"]
        assert f"/{axis}/0" in message
        assert "10" in message
        assert result["content"][0]["text"] == message

    @pytest.mark.parametrize("axis", ["spotShifts", "volShifts"])
    def test_the_handler_sentence_is_not_what_a_protocol_caller_sees(
        self, client: Client, axis: str
    ) -> None:
        """Where the axis conventions reach a caller, and where they do not.

        Found while writing the test above, and pinned because it is the sort of
        thing that is discovered twice otherwise. ``_shift_axis``'s refusals are
        unreachable over the protocol: the schema's own bounds are the same
        numbers as the handler's limits, its ``minItems`` covers the empty array,
        its ``type`` covers a non-number, and JSON cannot carry a NaN -- so
        every branch of that validation is pre-empted by schema validation and
        only a Python caller ever reads its message.

        That makes the schema *description* the only place a convention reaches a
        model, which is where both axes already state theirs. What the
        out-of-range message cannot currently carry is the unit it was measured
        in, and giving it one means either spending listing budget on the
        description or a per-field hint channel the registry does not have.
        """
        arguments: dict[str, Any] = {"handle": self.handle(client), axis: [10.0]}
        exchange = client.call_tool("position_book_scenarios", arguments)
        result = exchange.result
        assert result is not None and result["isError"] is True
        message = result["structuredContent"]["error"]["message"]
        for sentence in _AXIS_CONVENTION.values():
            assert sentence not in message
        # The convention does reach the caller, in the listing, and each axis
        # states its own there.
        listing = client.list_tools().result
        assert listing is not None
        listed = {tool["name"]: tool for tool in listing["tools"]}
        schema = listed["position_book_scenarios"]["inputSchema"]["properties"]
        assert "decimal fractions" in schema["spotShifts"]["description"]
        assert "volatility points" in schema["volShifts"]["description"]
        assert "percent" not in schema["volShifts"]["description"]
