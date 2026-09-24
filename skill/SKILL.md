---
name: abacus
description: >-
  Option pricing, implied volatility, portfolio risk, interest rate curves,
  execution cost and backtest validation over MCP. Use when a question involves
  pricing a derivative, sizing a portfolio's risk, discounting a cash flow,
  estimating what a trade cost, or judging whether a backtest is evidence of
  anything.
---

# abacus

Thirty-five tools in six groups. Each one's schema says what it takes; this says
which to reach for, and the conventions all of them share.

## Conventions, which apply everywhere

These are the mistakes that produce a plausible number rather than an error.
Every one of them has been made.

**Rates and volatilities are decimal fractions.** 20% is `0.2`, not `20`. So is
a 20% return. A volatility of `20` is not refused — it describes a market that
moves 2000% a year, which is a market, so the server prices it.

**Time is a year fraction, not a count of days.** Thirty days is about `0.082`.
One year is `1.0`.

**Losses are positive.** A value at risk of `0.023` is a 2.3% loss. An expected
shortfall is always at least as large as the value at risk at the same
confidence; if it comes back smaller, the arguments are not what you think.

**A Sharpe ratio is per period.** The figure people quote is annualised. An
annualised 1.0 on daily data is about `0.063` per period. Everything in the
validation group takes the per-period one, and `periodsPerYear` is what gets you
both back.

**`periodsPerYear` is required rather than assumed** wherever an annualised
figure is reported. 252 for daily equity data, 52 for weekly, 12 for monthly.
Annualising weekly data with 252 overstates volatility by a factor of about 2.2,
which does not look wrong enough to notice.

**Weights are not normalised for you.** Weights summing to 0.98 are either a 2%
cash position or a typo, and the two want opposite treatment. The sum is
reported; a sum far from one is refused with the total named.

**Handles are typed, and the types are not interchangeable.** Three tools mint
them and the payloads carry a `kind` tag, so presenting the wrong one is refused
rather than misread:

| Minted by | Carries | Refuses |
| --- | --- | --- |
| `open_position_book` | option and underlying legs, the spot, the rate | being read as a covariance estimate |
| `estimate_return_moments` | a mean vector and a covariance matrix — **not** the returns | the historical and drawdown tools, which need the path |
| `bootstrap_discount_curve` | the pillars and their quotes | being read as either of the above |

A handle caps at 8192 characters, which is why the moments handle carries second
moments rather than the matrix: a year of daily returns on four assets encodes
to about 6000 characters and five years on ten assets to about 69,000. A curve
does fit — five pillars is 264 characters — so the curve handle carries its
pillars and quotes.

**Refusals are results, not protocol errors.** A tool that declines an argument
returns `isError: true` with a message naming the field and what would be
accepted. That is a repairable failure; fix the argument and call again. A
JSON-RPC error means something else went wrong.

## Which tool

**Pricing one option.** `price_european_option` for the price,
`european_option_greeks` for the sensitivities, `european_option_analytics` for
both in one call. `put_call_parity` when the question is whether two quotes are
consistent. For American exercise, `price_american_lattice` when convergence
evidence matters and `price_american_closed_form` when speed does;
`american_exercise_boundary` for where exercise becomes optimal.

**Going from a price to a volatility.** `implied_volatility`. Before that, if
the price came from a screen rather than from a model, `option_price_bounds` —
a price outside the attainable range has no implied volatility at all, and
knowing that is more useful than a failed solve.

**A surface.** `fit_volatility_slice` for one expiry, `fit_volatility_surface`
for several with both no-arbitrage conditions checked, `local_volatility` for
Dupire local volatilities including where the identity has no answer.

**More than one position.** `open_position_book` and then
`position_book_greeks`; `position_book_scenarios` to reprice across a grid of
spot and volatility shifts; `amend_position_book` to change a leg or move the
market without resending the book; `describe_position_book` to read a book back
out of a handle when the legs are no longer in reach.

**Portfolio risk from returns.** `estimate_return_moments` first — it returns a
handle, and the parametric tools run from it without resending the matrix.
`portfolio_tail_risk` for value at risk and expected shortfall,
`portfolio_risk_contributions` for where the risk sits, `risk_parity_weights`
for weights that equalise it, `portfolio_drawdown` for the path statistics. The
last of those, and the historical tail-risk methods, need the matrix rather than
the handle, because a second-moment summary has thrown the path away.

**Discounting and bonds.** `bootstrap_discount_curve` from deposits, futures and
par swaps; `discount_curve_rates` for factors, zero rates and forwards;
`bond_analytics` for price, yield, duration and convexity; `bond_curve_risk` for
key rate durations and the hedge; `bond_spreads` for Z-spread, I-spread and
option-adjusted spread.

**What a trade cost, or should cost.**
`decompose_implementation_shortfall` after the fact,
`optimal_execution_schedule` before it, `execution_cost_frontier` when the
trade-off between cost and certainty has not been decided.

**Whether a backtest means anything.** `deflated_sharpe_ratio` when a result was
selected out of several attempts — which is nearly always;
`effective_trial_count` to see how many independent bets those attempts really
were; `minimum_track_record_length` for how long a record would have to be;
`backtest_overfitting_probability` for whether selection carries information at
all; `superior_predictive_ability` for whether anything beats the benchmark.

## Reading the results

**The method is named because the number does not identify it.** A one-day 99%
value at risk of 2.3% under a normal assumption and 3.1% from the sample are the
same quantity estimated two ways. Every risk result carries the method, the
confidence and the observation count.

**Diagnostics are data, not decoration.** A shrinkage intensity near one says
the sample carries little information about the pairwise structure. A calibrated
lattice reports whether it reprices the curve it was built from, and one that
does not is not a model of that curve. A fitted surface reports both
no-arbitrage conditions. Read them before acting on the number beside them.

**An absent number comes back as `null`, and means something.** A zero rate at
the curve's reference date is `null` rather than zero, because the discount
factor there is one whatever the rate is. A risk-neutral execution schedule's
half-life is `null` because it is infinite. `Infinity` and `NaN` are not JSON, so
neither is ever sent.

## Worked transcript: a book and its risk

Open a book of three legs, read its aggregate sensitivities, then shock it.

```transcript
{
  "tool": "open_position_book",
  "arguments": {
    "spot": 100.0,
    "rate": 0.04,
    "label": "overwrite",
    "legs": [
      {"instrument": "underlying", "quantity": 1000, "label": "stock"},
      {"instrument": "call", "quantity": -10, "strike": 105.0, "time": 0.25, "vol": 0.22, "label": "short call"},
      {"instrument": "put", "quantity": -5, "strike": 90.0, "time": 0.25, "vol": 0.28, "label": "short put"}
    ]
  },
  "capture": {"book": "handle"}
}
```

```transcript
{
  "tool": "position_book_greeks",
  "arguments": {"handle": "$book"}
}
```

```transcript
{
  "tool": "position_book_scenarios",
  "arguments": {
    "handle": "$book",
    "spotShifts": [-0.1, -0.05, 0.0, 0.05, 0.1],
    "volShifts": [-0.05, 0.0, 0.05]
  }
}
```

The handle goes in every call after the first. It is opaque and carries its own
integrity check, so it survives being passed through a context window, and a
truncated one is refused rather than misread.

## Worked transcript: sizing a portfolio's risk

Estimate once, then ask several questions of the estimate.

```transcript
{
  "tool": "estimate_return_moments",
  "arguments": {
    "returns": "$returns",
    "assets": ["EQ", "CR", "GV", "CM"],
    "periodsPerYear": 252
  },
  "capture": {"moments": "handle"}
}
```

```transcript
{
  "tool": "portfolio_tail_risk",
  "arguments": {
    "handle": "$moments",
    "weights": [0.4, 0.25, 0.15, 0.2],
    "confidence": 0.99,
    "method": "student-t"
  }
}
```

```transcript
{
  "tool": "portfolio_risk_contributions",
  "arguments": {
    "handle": "$moments",
    "weights": [0.4, 0.25, 0.15, 0.2]
  }
}
```

The drawdown statistics cannot run from that handle, because they read the
return path and the handle holds only second moments. Send the matrix:

```transcript
{
  "tool": "portfolio_drawdown",
  "arguments": {
    "returns": "$returns",
    "weights": [0.4, 0.25, 0.15, 0.2],
    "periodsPerYear": 252
  }
}
```

## Worked transcript: judging a backtest

A parameter sweep produced a best Sharpe ratio. Three questions decide whether
it means anything, and they are not the same question.

How many independent bets were those trials, really?

```transcript
{
  "tool": "effective_trial_count",
  "arguments": {"trials": "$trials"}
}
```

Does the best of them survive being corrected for the search?

```transcript
{
  "tool": "deflated_sharpe_ratio",
  "arguments": {"trials": "$trials", "periodsPerYear": 252}
}
```

Does selecting on this backtest carry any information about what follows?

```transcript
{
  "tool": "backtest_overfitting_probability",
  "arguments": {"trials": "$trials", "blocks": 10}
}
```

A probability of backtest overfitting near one half means the selection says
nothing; above it means it is actively misleading, and the degradation slope
will usually be negative when that happens. Read the deflated figure and this
one together: a strategy can survive the first and fail the second, which says
the record is long enough to be real and the selection is still worthless.

## The size of this surface

The tool listing is loaded before any work happens, and it grows with every
group added. It is currently 35 tools and about 115,000 characters of JSON, of
which roughly 15,000 are descriptions and 91,000 are schemas. The budget is
140,000 characters in total and 12,000 for any single tool, enforced by a test,
so crossing it is a decision rather than a drift. `tools/list` paginates, and a
caller that needs one group can page rather than load all of it.
