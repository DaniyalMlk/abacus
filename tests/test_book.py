"""Position books: parsing, aggregation, scenarios and handle recovery.

The aggregation tests lean on identities rather than on recorded numbers
wherever one exists. A book of a long call and a short put at the same strike
has a delta of exactly one and a gamma and vega of exactly zero, whatever the
market is, so a test written against those catches a sign error that a
regression fixture would happily preserve.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from abacus.book import MAX_GRID_CELLS, MAX_LEGS, register
from abacus.handles import Minter
from abacus.tools import ToolRegistry

SPOT = 100.0
RATE = 0.04
TIME = 0.5
VOL = 0.2

CALL = {"instrument": "call", "quantity": 1.0, "strike": 100.0, "time": TIME, "vol": VOL}
PUT = {"instrument": "put", "quantity": -1.0, "strike": 100.0, "time": TIME, "vol": VOL}
SHARES = {"instrument": "underlying", "quantity": -1.0}


@pytest.fixture
def registry() -> ToolRegistry:
    """A registry holding only the book tools, minted under a fixed key."""
    return register(ToolRegistry(), minter=Minter(key=b"fixed key for the book tests"))


def payload(registry: ToolRegistry, name: str, **args: Any) -> dict[str, Any]:
    """Call a tool and return its payload, failing loudly on a refusal."""
    result = registry.call(name, args)
    assert result["isError"] is False, result["structuredContent"]
    content: dict[str, Any] = result["structuredContent"]
    return content


def refusal(registry: ToolRegistry, name: str, **args: Any) -> dict[str, Any]:
    """Call a tool expecting a recoverable refusal, and return the error object."""
    result = registry.call(name, args)
    assert result["isError"] is True, result["structuredContent"]
    error: dict[str, Any] = result["structuredContent"]["error"]
    return error


def open_book(registry: ToolRegistry, *legs: dict[str, Any], **market: Any) -> str:
    opened = payload(
        registry,
        "open_position_book",
        spot=market.get("spot", SPOT),
        rate=market.get("rate", RATE),
        legs=list(legs),
    )
    handle: str = opened["handle"]
    return handle


# -- opening and reading back ---------------------------------------------


def test_a_book_reads_back_exactly_as_it_was_opened(registry: ToolRegistry) -> None:
    handle = open_book(registry, CALL, PUT, SHARES)
    read = payload(registry, "describe_position_book", handle=handle)
    assert read["book"]["legs"] == [CALL, PUT, SHARES]
    assert read["book"]["spot"] == SPOT
    assert read["book"]["rate"] == RATE


def test_opening_reports_the_shape_of_the_book(registry: ToolRegistry) -> None:
    opened = payload(
        registry,
        "open_position_book",
        spot=SPOT,
        rate=RATE,
        legs=[CALL, PUT, SHARES],
        label="conversion",
    )
    assert opened["summary"] == {
        "legCount": 3,
        "optionLegs": 2,
        "underlyingLegs": 1,
        "grossQuantity": 3.0,
        "netQuantity": -1.0,
        "strikes": [100.0],
        "expiries": [TIME],
    }
    assert opened["book"]["label"] == "conversion"
    assert opened["expiresInSeconds"] > 0


def test_a_label_on_a_leg_is_carried_through(registry: ToolRegistry) -> None:
    labelled = {**CALL, "label": "upside"}
    handle = open_book(registry, labelled)
    risk = payload(registry, "position_book_greeks", handle=handle)
    assert risk["legs"][0]["label"] == "upside"


# -- identities the aggregate must satisfy --------------------------------


def test_a_conversion_is_worth_minus_the_discounted_strike(registry: ToolRegistry) -> None:
    # Long call, short put, short the underlying, all at one strike. Put-call
    # parity gives C - P = e^{-rT}(F - K) and with carry equal to the rate
    # F = S e^{rT}, so the whole book is worth exactly -K e^{-rT} whatever the
    # volatility is. A closed form, so it is asserted to machine precision.
    handle = open_book(registry, CALL, PUT, SHARES)
    risk = payload(registry, "position_book_greeks", handle=handle)
    assert risk["value"] == pytest.approx(-100.0 * math.exp(-RATE * TIME), abs=1e-12)


def test_a_call_less_a_put_is_exactly_one_delta(registry: ToolRegistry) -> None:
    handle = open_book(registry, CALL, PUT)
    risk = payload(registry, "position_book_greeks", handle=handle)
    assert risk["greeks"]["delta"] == pytest.approx(1.0, abs=1e-12)


def test_a_conversion_has_no_delta_gamma_or_vega(registry: ToolRegistry) -> None:
    # The hedge removes the one delta, and a call and a put on the same strike
    # have identical gamma and vega, so both net to zero. Any sign error in the
    # aggregation shows up here as a number that is not zero.
    handle = open_book(registry, CALL, PUT, SHARES)
    risk = payload(registry, "position_book_greeks", handle=handle)
    assert risk["greeks"]["delta"] == pytest.approx(0.0, abs=1e-12)
    assert risk["greeks"]["gamma"] == pytest.approx(0.0, abs=1e-12)
    assert risk["greeks"]["vega"] == pytest.approx(0.0, abs=1e-12)
    assert risk["complete"] is True


def test_the_aggregate_is_the_sum_of_the_per_leg_breakdown(registry: ToolRegistry) -> None:
    # The aggregate and the breakdown are computed by separate code paths, and a
    # divergence between them would be invisible in either one alone.
    handle = open_book(
        registry,
        {"instrument": "call", "quantity": 7.0, "strike": 110.0, "time": 0.75, "vol": 0.25},
        {"instrument": "put", "quantity": -3.0, "strike": 90.0, "time": 0.25, "vol": 0.31},
        {"instrument": "underlying", "quantity": 12.5},
    )
    risk = payload(registry, "position_book_greeks", handle=handle)
    assert risk["value"] == pytest.approx(sum(leg["value"] for leg in risk["legs"]), abs=1e-9)
    for name, total in risk["greeks"].items():
        summed = sum(leg["greeks"][name] for leg in risk["legs"])
        assert total == pytest.approx(summed, abs=1e-9), name


def test_doubling_every_quantity_doubles_every_number(registry: ToolRegistry) -> None:
    single = payload(
        registry, "position_book_greeks", handle=open_book(registry, CALL, PUT, SHARES)
    )
    doubled = payload(
        registry,
        "position_book_greeks",
        handle=open_book(
            registry,
            {**CALL, "quantity": 2.0},
            {**PUT, "quantity": -2.0},
            {**SHARES, "quantity": -2.0},
        ),
    )
    assert doubled["value"] == pytest.approx(2.0 * single["value"], abs=1e-12)
    for name, value in single["greeks"].items():
        assert doubled["greeks"][name] == pytest.approx(2.0 * value, abs=1e-12), name


def test_an_underlying_leg_is_one_delta_per_unit_and_nothing_else(
    registry: ToolRegistry,
) -> None:
    handle = open_book(registry, {"instrument": "underlying", "quantity": 40.0})
    risk = payload(registry, "position_book_greeks", handle=handle)
    assert risk["value"] == pytest.approx(40.0 * SPOT, abs=1e-12)
    assert risk["greeks"]["delta"] == pytest.approx(40.0, abs=1e-12)
    assert all(value == 0.0 for name, value in risk["greeks"].items() if name != "delta")
    assert risk["legs"][0]["greeksDefined"] is True


def test_a_short_leg_reverses_the_sign_of_its_risk(registry: ToolRegistry) -> None:
    long_call = payload(registry, "position_book_greeks", handle=open_book(registry, CALL))
    short_call = payload(
        registry,
        "position_book_greeks",
        handle=open_book(registry, {**CALL, "quantity": -1.0}),
    )
    for name, value in long_call["greeks"].items():
        assert short_call["greeks"][name] == pytest.approx(-value, abs=1e-12), name


# -- legs whose derivative does not exist ----------------------------------


def test_an_expired_leg_is_priced_but_left_out_of_the_aggregate(
    registry: ToolRegistry,
) -> None:
    expired = {"instrument": "call", "quantity": 2.0, "strike": 90.0, "time": 0.0, "vol": VOL}
    handle = open_book(registry, CALL, expired)
    risk = payload(registry, "position_book_greeks", handle=handle)

    # Its intrinsic value is in the total — that number exists and is exact.
    assert risk["value"] == pytest.approx(risk["legs"][0]["value"] + 2.0 * 10.0, abs=1e-9)
    # Its sensitivities are not, and are not silently zero either.
    assert risk["complete"] is False
    assert risk["excludedLegs"] == [1]
    assert risk["legs"][1]["greeks"] is None
    assert risk["legs"][1]["greeksDefined"] is False
    assert "kinked" in risk["legs"][1]["undefinedBecause"]
    assert "aggregate" in risk["incompleteBecause"]


def test_the_aggregate_of_a_book_with_an_expired_leg_covers_the_rest(
    registry: ToolRegistry,
) -> None:
    expired = {"instrument": "put", "quantity": 5.0, "strike": 120.0, "time": 0.0, "vol": VOL}
    alone = payload(registry, "position_book_greeks", handle=open_book(registry, CALL))
    together = payload(
        registry, "position_book_greeks", handle=open_book(registry, CALL, expired)
    )
    assert together["greeks"] == alone["greeks"]


def test_a_zero_volatility_leg_is_excluded_too(registry: ToolRegistry) -> None:
    flat = {"instrument": "call", "quantity": 1.0, "strike": 100.0, "time": TIME, "vol": 0.0}
    risk = payload(registry, "position_book_greeks", handle=open_book(registry, flat))
    assert risk["complete"] is False
    assert risk["excludedLegs"] == [0]


# -- what a leg may and may not say ---------------------------------------


@pytest.mark.parametrize("missing", ["strike", "time", "vol"])
def test_an_option_leg_without_its_terms_is_refused(
    registry: ToolRegistry, missing: str
) -> None:
    leg = {k: v for k, v in CALL.items() if k != missing}
    error = refusal(registry, "open_position_book", spot=SPOT, rate=RATE, legs=[leg])
    assert missing in error["message"]


@pytest.mark.parametrize("extra", ["strike", "time", "vol"])
def test_an_underlying_leg_carrying_option_terms_is_refused(
    registry: ToolRegistry, extra: str
) -> None:
    # A strike on an underlying position means the caller has muddled two legs.
    # Dropping the field would price the muddle, so it is refused — and the
    # refusal names the field, which is why this rule lives in the leg check
    # rather than in a `oneOf` that could only say "no accepted form".
    leg = {**SHARES, extra: 100.0}
    error = refusal(registry, "open_position_book", spot=SPOT, rate=RATE, legs=[leg])
    assert error["kind"] == "domain"
    assert extra in error["message"]


def test_a_volatility_given_as_a_percentage_is_caught_on_a_leg(
    registry: ToolRegistry,
) -> None:
    error = refusal(
        registry, "open_position_book", spot=SPOT, rate=RATE, legs=[{**CALL, "vol": 20.0}]
    )
    assert "percentage" in error["message"]


def test_an_empty_book_is_refused(registry: ToolRegistry) -> None:
    assert refusal(registry, "open_position_book", spot=SPOT, rate=RATE, legs=[])


def test_a_book_beyond_the_leg_limit_is_refused(registry: ToolRegistry) -> None:
    legs = [{**CALL, "strike": 50.0 + i} for i in range(MAX_LEGS + 1)]
    error = refusal(registry, "open_position_book", spot=SPOT, rate=RATE, legs=legs)
    assert str(MAX_LEGS) in error["message"]


def test_a_book_at_the_leg_limit_is_accepted(registry: ToolRegistry) -> None:
    legs = [{**CALL, "strike": 50.0 + i} for i in range(MAX_LEGS)]
    opened = payload(registry, "open_position_book", spot=SPOT, rate=RATE, legs=legs)
    assert opened["summary"]["legCount"] == MAX_LEGS


@pytest.mark.parametrize("spot", [0.0, -1.0])
def test_a_non_positive_spot_is_refused(registry: ToolRegistry, spot: float) -> None:
    assert refusal(registry, "open_position_book", spot=spot, rate=RATE, legs=[CALL])


# -- handles, as the caller experiences them ------------------------------


def test_an_altered_handle_is_a_recoverable_refusal_not_a_protocol_error(
    registry: ToolRegistry,
) -> None:
    handle = open_book(registry, CALL)
    error = refusal(registry, "position_book_greeks", handle=handle[:-3] + "AAA")
    assert error["kind"] == "handle_unrecognised"
    assert "Open the book again" in error["message"]


def test_an_expired_handle_says_so_specifically() -> None:
    minter = Minter(key=b"k", ttl_s=1)
    registry = register(ToolRegistry(), minter=minter)
    handle = open_book(registry, CALL)
    # Minted against the epoch, so no scheduling accident can make this flaky.
    stale = minter.mint({"spot": SPOT, "rate": RATE, "legs": [CALL]}, now=0.0)
    error = refusal(registry, "describe_position_book", handle=stale)
    assert error["kind"] == "handle_expired"
    assert registry.call("describe_position_book", {"handle": handle})["isError"] is False


def test_a_handle_from_a_different_server_is_not_recognised(registry: ToolRegistry) -> None:
    foreign = Minter(key=b"some other server entirely").mint(
        {"spot": SPOT, "rate": RATE, "legs": [CALL]}
    )
    assert refusal(registry, "describe_position_book", handle=foreign)[
        "kind"
    ] == "handle_unrecognised"


def test_a_book_too_large_to_encode_is_refused_when_it_is_opened() -> None:
    registry = register(ToolRegistry(), minter=Minter(key=b"k", max_chars=64))
    error = refusal(registry, "open_position_book", spot=SPOT, rate=RATE, legs=[CALL])
    assert error["kind"] == "book_too_large"


# -- amendment -------------------------------------------------------------


def test_amending_adds_and_removes_legs_in_that_order(registry: ToolRegistry) -> None:
    handle = open_book(registry, CALL, PUT, SHARES)
    amended = payload(
        registry,
        "amend_position_book",
        handle=handle,
        removeLegs=[1],
        addLegs=[{"instrument": "underlying", "quantity": 4.0}],
    )
    assert [leg["instrument"] for leg in amended["book"]["legs"]] == [
        "call",
        "underlying",
        "underlying",
    ]


def test_amending_leaves_the_original_handle_working(registry: ToolRegistry) -> None:
    # It has to: the server keeps no record of it and so cannot revoke it. The
    # result says as much rather than implying an invalidation that never happens.
    handle = open_book(registry, CALL, PUT)
    amended = payload(registry, "amend_position_book", handle=handle, removeLegs=[1])
    assert amended["replaces"] == handle
    assert amended["priorHandleStillValid"] is True
    assert amended["handle"] != handle
    before = payload(registry, "describe_position_book", handle=handle)
    assert len(before["book"]["legs"]) == 2


def test_amending_the_market_reprices_without_touching_the_legs(
    registry: ToolRegistry,
) -> None:
    handle = open_book(registry, CALL)
    moved = payload(registry, "amend_position_book", handle=handle, spot=120.0)
    assert moved["book"]["legs"] == [CALL]
    assert moved["book"]["spot"] == 120.0
    assert moved["value"] > payload(registry, "describe_position_book", handle=handle)["value"]


def test_amending_a_book_to_nothing_is_refused(registry: ToolRegistry) -> None:
    handle = open_book(registry, CALL)
    error = refusal(registry, "amend_position_book", handle=handle, removeLegs=[0])
    assert "empty the book" in error["message"]


def test_removing_a_leg_that_is_not_there_is_refused(registry: ToolRegistry) -> None:
    handle = open_book(registry, CALL, PUT)
    error = refusal(registry, "amend_position_book", handle=handle, removeLegs=[5])
    assert "legs 0..1" in error["message"]


def test_removing_the_same_leg_twice_is_refused(registry: ToolRegistry) -> None:
    handle = open_book(registry, CALL, PUT)
    assert refusal(registry, "amend_position_book", handle=handle, removeLegs=[0, 0])


# -- scenario grids --------------------------------------------------------


def test_the_grid_has_the_shape_its_axes_describe(registry: ToolRegistry) -> None:
    handle = open_book(registry, CALL, PUT, SHARES)
    grid = payload(
        registry,
        "position_book_scenarios",
        handle=handle,
        spotShifts=[-0.1, 0.0, 0.1],
        volShifts=[-0.02, 0.0],
    )
    assert [row["volShift"] for row in grid["rows"]] == [-0.02, 0.0]
    assert len(grid["rows"][0]["cells"]) == 3
    assert [cell["spotShift"] for cell in grid["rows"][0]["cells"]] == [-0.1, 0.0, 0.1]


def test_every_cell_states_the_spot_it_was_priced_at(registry: ToolRegistry) -> None:
    # The labelling is the safeguard against a transposed read, so it is tested
    # rather than left to inspection.
    grid = payload(
        registry,
        "position_book_scenarios",
        handle=open_book(registry, CALL),
        spotShifts=[-0.25, 0.5],
        volShifts=[0.0],
    )
    assert [cell["spot"] for cell in grid["rows"][0]["cells"]] == [75.0, 150.0]


def test_the_unshifted_cell_is_the_current_book(registry: ToolRegistry) -> None:
    handle = open_book(registry, CALL, PUT, SHARES)
    grid = payload(
        registry, "position_book_scenarios", handle=handle, spotShifts=[0.0], volShifts=[0.0]
    )
    cell = grid["rows"][0]["cells"][0]
    assert cell["pnl"] == pytest.approx(0.0, abs=1e-12)
    assert cell["value"] == pytest.approx(grid["base"]["value"], abs=1e-12)


def test_a_long_call_gains_as_spot_rises(registry: ToolRegistry) -> None:
    grid = payload(
        registry,
        "position_book_scenarios",
        handle=open_book(registry, CALL),
        spotShifts=[-0.2, -0.1, 0.0, 0.1, 0.2],
        volShifts=[0.0],
    )
    values = [cell["value"] for cell in grid["rows"][0]["cells"]]
    assert values == sorted(values)
    assert grid["worstValue"] == pytest.approx(values[0], abs=1e-12)
    assert grid["bestValue"] == pytest.approx(values[-1], abs=1e-12)


def test_a_long_option_gains_as_volatility_rises(registry: ToolRegistry) -> None:
    grid = payload(
        registry,
        "position_book_scenarios",
        handle=open_book(registry, CALL),
        spotShifts=[0.0],
        volShifts=[-0.05, 0.0, 0.05],
    )
    values = [row["cells"][0]["value"] for row in grid["rows"]]
    assert values == sorted(values)


def test_a_volatility_shift_past_zero_floors_there(registry: ToolRegistry) -> None:
    # Taking a 20-vol book down 50 points is a legitimate ask; the answer is the
    # zero-volatility value, not a refusal and not a negative volatility.
    handle = open_book(registry, CALL)
    grid = payload(
        registry, "position_book_scenarios", handle=handle, spotShifts=[0.0], volShifts=[-0.5]
    )
    floored = grid["rows"][0]["cells"][0]
    flat = payload(
        registry,
        "position_book_greeks",
        handle=open_book(registry, {**CALL, "vol": 0.0}),
    )
    assert floored["value"] == pytest.approx(flat["value"], abs=1e-12)
    # And with no volatility the cell has no delta to report, so it says so.
    assert floored["delta"] is None


def test_a_grid_beyond_the_cell_limit_is_refused(registry: ToolRegistry) -> None:
    handle = open_book(registry, CALL)
    error = refusal(
        registry,
        "position_book_scenarios",
        handle=handle,
        spotShifts=[i / 100.0 for i in range(40)],
        volShifts=[i / 100.0 for i in range(40)],
    )
    assert str(MAX_GRID_CELLS) in error["message"]


def test_a_shift_given_as_a_percentage_is_caught(registry: ToolRegistry) -> None:
    handle = open_book(registry, CALL)
    error = refusal(registry, "position_book_scenarios", handle=handle, spotShifts=[10.0])
    assert error["kind"] == "invalid_input"


def test_the_grid_falls_back_to_a_readable_default(registry: ToolRegistry) -> None:
    grid = payload(registry, "position_book_scenarios", handle=open_book(registry, CALL))
    assert len(grid["rows"]) == 3
    assert len(grid["rows"][0]["cells"]) == 5
    assert len(grid["spotShifts"]) * len(grid["volShifts"]) <= MAX_GRID_CELLS


def test_no_book_tool_raises_on_a_degenerate_book(registry: ToolRegistry) -> None:
    # The counterpart of the sweep over the single-option tools: whatever the
    # library declines to compute for a book, the caller gets a readable result
    # rather than an exception crossing the tool boundary. It runs over every
    # registered tool so a tool added later cannot quietly opt out.
    degenerate = [
        {"time": 0.0},
        {"vol": 1e-300},
        {"strike": 1e-300},
        {"time": 1e-320},
    ]
    for overrides in degenerate:
        handle = open_book(registry, {**CALL, **overrides}, SHARES)
        for tool in registry:
            args = {"spot": SPOT, "rate": RATE, "legs": [CALL]}
            if tool.name != "open_position_book":
                args = {"handle": handle}
            result = registry.call(tool.name, args)
            assert isinstance(result["isError"], bool)
