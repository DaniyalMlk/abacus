# abacus

An MCP server that exposes option and portfolio analytics as tools.

Language models are unreliable at arithmetic, so the useful thing a tool
boundary can do is move the numbers somewhere trustworthy. `abacus` puts them in
libraries whose numerical cores are validated against closed forms,
high-precision references and published results, and exposes those cores over the
Model Context Protocol. Nothing here wraps a third-party pricing API: the numbers
are computed by these libraries and tested where they live.

It targets MCP revision **2026-07-28** and has three runtime dependencies, all pure
Python with none of their own:
[`moneyness`](https://github.com/DaniyalMlk/moneyness) for the option mathematics,
[`shortfall`](https://github.com/DaniyalMlk/shortfall) for the portfolio risk
estimators and [`tenor`](https://github.com/DaniyalMlk/tenor) for curves and bonds.

## Running it

```bash
pip install abacus-mcp

abacus stdio          # what an MCP client launches
abacus http           # Streamable HTTP on 127.0.0.1:8000/mcp
abacus tools          # print the tool surface and exit
abacus conform        # run the conformance suite against this server
```

The distribution is `abacus-mcp`; the package you import is `abacus`. The short
name was already taken on the index by an unrelated project, and renaming the
package to match would have changed every import for the sake of a registry
collision.

## Registering it with a client

Nothing has to be installed first if you have [`uv`](https://docs.astral.sh/uv/):
`uvx` fetches the server, runs it, and caches it for next time.

```json
{
  "mcpServers": {
    "abacus": {
      "command": "uvx",
      "args": ["abacus-mcp", "stdio"]
    }
  }
}
```

Where that file lives depends on the client:

| Client | Configuration |
| --- | --- |
| Claude Desktop | `claude_desktop_config.json`, under `mcpServers` |
| Claude Code | `claude mcp add abacus -- uvx abacus-mcp stdio` |
| Cursor | `.cursor/mcp.json` in the project, or `~/.cursor/mcp.json` |
| VS Code | `.vscode/mcp.json`, under `servers` |
| Zed | `context_servers` in the settings |

With the package installed into an environment rather than run through `uvx`,
point the client at the installed script instead. An absolute path is worth the
noise: a client launched from a desktop session rarely has the same `PATH` as
your shell, and a bare `abacus` that works in a terminal and not in the client is
the most common way this goes wrong.

```json
{
  "mcpServers": {
    "abacus": {
      "command": "/path/to/venv/bin/abacus",
      "args": ["stdio"]
    }
  }
}
```

To check the server is healthy before wiring a client to it, run the conformance
suite against the launch command you are about to configure:

```bash
abacus conform --stdio "uvx abacus-mcp stdio"
```

It exits non-zero on a failure, so it works as a gate rather than a report.

## The tools

| Tool | What it does |
| --- | --- |
| `price_european_option` | Price under generalised Black-Scholes-Merton, with forward, intrinsic and time value |
| `european_option_greeks` | Delta, gamma, vega, theta, rho, the higher-order Greeks, and sensitivities to strike and carry |
| `european_option_analytics` | Both of the above in one call |
| `put_call_parity` | Both sides of a strike and the parity residual |
| `option_price_bounds` | The price range attainable by some non-negative volatility |
| `implied_volatility` | Recover volatility from a price, with convergence evidence |
| `fit_volatility_slice` | Fit a raw SVI slice, with fit quality and a butterfly check |
| `fit_volatility_surface` | Fit a surface, with both no-arbitrage conditions checked |
| `local_volatility` | Dupire local volatilities, including where the identity has no answer |
| `price_american_lattice` | American price on a lattice, with convergence reporting |
| `price_american_closed_form` | Bjerksund-Stensland 2002, labelled as an approximation |
| `american_exercise_boundary` | The early-exercise boundary as a series |
| `open_position_book` | Assemble option and underlying positions into a book, returning a handle |
| `describe_position_book` | Read a book back from its handle |
| `amend_position_book` | Add or remove legs, or move the market, returning a new handle |
| `position_book_greeks` | Total value and net sensitivities, with the per-leg breakdown |
| `position_book_scenarios` | Reprice a book across a grid of spot and volatility shifts |
| `estimate_return_moments` | Covariance from a returns matrix, with shrinkage, diagnostics and a reusable handle |
| `portfolio_tail_risk` | Value at risk and expected shortfall by five methods, each naming itself |
| `portfolio_risk_contributions` | Euler risk contributions, concentration, effective bets |
| `risk_parity_weights` | Weights that equalise risk contributions, with the convergence evidence |
| `portfolio_drawdown` | Deepest drawdown, time underwater, ulcer index, Calmar and Sortino |
| `bootstrap_discount_curve` | A curve from deposits, futures and par swaps, with a reusable handle |
| `discount_curve_rates` | Discount factors, zero rates and forwards at whatever dates you ask for |
| `bond_analytics` | Price, yield, duration and convexity, from a yield and from a curve |
| `bond_curve_risk` | Key rate durations, curve shape risk, and the tradeable hedge |
| `bond_spreads` | Z-spread, I-spread, and option-adjusted spread off a calibrated lattice |
| `decompose_implementation_shortfall` | What an order cost, split into delay, trading, opportunity and explicit |
| `optimal_execution_schedule` | The Almgren-Chriss trajectory, its cost, and the half-life's elasticities |
| `execution_cost_frontier` | Expected cost against cost risk, one schedule per risk aversion |
| `deflated_sharpe_ratio` | A Sharpe ratio corrected for how many things were tried, on the effective count |
| `minimum_track_record_length` | How long a record must be before a ratio that size means anything |
| `effective_trial_count` | How many independent bets a correlated set of trials really is |
| `backtest_overfitting_probability` | How often the in-sample winner lands in the bottom half out of sample |
| `superior_predictive_ability` | Whether the best candidate beats the benchmark by more than the search would |

Conventions, which are also stated in the server's instructions and in every
schema description: volatilities and rates are decimal fractions, so 20% is
`0.2` — and so is a 20% return; time is a year fraction, so thirty days is about
`0.082`; log-moneyness is measured on the forward as `log(strike / forward)`; a
value at risk is a positive loss over one period of whatever frequency the
returns have, so `periodsPerYear` is required rather than assumed.

## Portfolio risk

```bash
# estimate once, then ask several questions of the estimate
abacus tools | grep -A2 estimate_return_moments
```

Four decisions in this part of the surface are worth knowing before using it.

**Every result names its method.** A one-day 99% value at risk of 2.2% under a
normal assumption and 2.0% from the sample are the same quantity estimated two
ways, and nothing about either number says which. So the method is on the result,
along with the observation count and whatever diagnostics that method has: the
degrees of freedom, the moments a correction used, the effective sample behind a
historical tail.

**A returns handle carries the second moments, not the returns.** A handle has to
fit in a message a model carries through its context, and is capped at 8192
characters for that reason. Measured: a year of daily returns on four assets
encodes to about 6000 characters and five years on ten assets to about 69,000, so
a handle carrying the matrix would refuse almost every portfolio worth asking
about. A mean vector and a covariance matrix are `n + n²` numbers whatever the
history length.

What that buys is real — send a matrix once, then reweight, decompose and
rebalance across as many calls as you like. What it costs is that the historical
estimators and every drawdown statistic read the *path*, which a second-moment
summary has discarded. Those need the matrix again, and say so rather than
answering from what they have. The Cornish-Fisher correction is in the same
position for a subtler reason: it needs the skewness and excess kurtosis of the
*portfolio*, which depend on the weights.

**Some questions have no answer for some portfolios, and get none.** Cornish-Fisher
is refused outright when the estimated moments put its corrected quantile outside
the region where it increases with the probability — outside it the mapping is not
a quantile function and nothing read off it is a quantile of anything. Concentration
and effective bets come back null for a portfolio with a negative risk
contribution, because they read the shares as a distribution and a negative share
is not one; the contributions themselves are unaffected and still returned.

**Weights are not normalised silently.** A sum of 0.98 is either a two percent
cash position or a typo, and scaling it quietly turns the second into a plausible
answer. The sum is reported on every result and a sum far from one is refused with
the total named.

## Curves and bonds

Three things to know before using this part of the surface.

**No convention has a default, and that is the point.** A bond priced on the wrong
day count basis is wrong by a few basis points — exactly the size of the spread
anyone is trying to measure — so it is not approximately right, it answers a
different question and looks entirely normal doing it. Every basis, frequency and
rolling rule is an argument. The one exception is the rolling rule, which has a
market convention; it is read out of the underlying library's own default rather
than chosen here, and every result reports the rule it used, because it changes
the cashflow dates and therefore the price.

**The curve handle carries the curve, unlike the returns handle.** The difference
is measured rather than stylistic. A returns matrix grows with the length of the
history — five years of daily data on ten assets encodes to about 69,000
characters, against a handle limit of 8192 — so that handle carries second moments
and gives up the path. A bootstrapped curve *is* its pillars: two numbers per
instrument, 264 characters at five pillars and 764 at sixty. So this handle carries
the pillars and the quotes behind them, and every curve tool works from it exactly
as from a fresh bootstrap, including instrument risk, which is a question about the
quotes.

**A bootstrap says whether it worked.** A curve that fails to reprice the
instruments it was built from is not slightly wrong; it means nothing, and its
pillar values look ordinary either way. The result carries the sweep count, the
per-instrument solve residuals, and the repricing check. The same applies to the
short-rate lattice: it reports whether it reprices the curve it was calibrated to,
because a tree that does not is not a model of that curve and every spread read off
it is wrong.

A zero rate at the curve's reference date comes back as null rather than zero. The
discount factor there is one whatever the rate is, so no rate is implied, and a zero
would read as a rate rather than as the absence of one.

## Execution

Two questions sit either side of a trade, and both are about the gap between the
price a model priced and the price that happened.

```bash
abacus tools | grep -A2 decompose_implementation_shortfall
```

**After the fact.** `decompose_implementation_shortfall` takes an order, its
fills and three prices, and splits what it cost four ways. The total is the least
useful number in the result: a trade that cost 87 basis points tells you to feel
bad, while the same trade split into 20 of delay, 51 of trading, 15 of
opportunity and 1 of commission tells you which thing to change. Delay is the
price moving before the order reached the market, and is fixed by shortening that
gap. Trading is the order's own footprint, and is fixed by spreading it out.
Opportunity is the part that never got done. Commission is a contract.

Fill timestamps are not required. `slippage.Order` carries them for its volume
work, but the decomposition reads only quantities, prices and commissions and
gives an identical breakdown at one-minute and six-hour fill spacings — so asking
for them would be asking for data to be invented.

Whether delay is charged on the quantity *ordered* or the quantity *executed* is
a house convention, and it matters more than it looks. On the worked order above
the order basis gives a delay of 200 and an opportunity of 150; the executed
basis gives 180 and 170. The total is 869 either way. It is a reattribution that
survives a check on the headline, which is exactly how two desks end up agreeing
on the cost and disagreeing on the cause, so every result names the basis it
used.

**Before the fact.** `optimal_execution_schedule` lays the order out against
impact and price risk. At zero risk aversion the answer is a straight line — an
equal slice each period, which is TWAP — and raising the aversion front-loads the
schedule, paying more impact to spend less time exposed. `execution_cost_frontier`
does the same across a range of aversions, because a single optimal schedule
answers a question the caller has already had to answer.

Two notes on the model. A fixed cost per share is paid whatever the order is
traded in, so it moves the expected cost by exactly itself and does not move a
single trade. Permanent impact is the one the textbook says drops out of the
schedule, and in continuous time it does; in discrete time it enters through
`eta - gamma*tau/2` and the leftover is of the order of the period length —
measured here, it shortens the half-life by 2.5% at `tau = 1`, 0.25% at
`tau = 0.1` and 0.025% at `tau = 0.01`.

A risk-neutral schedule's half-life is infinite, and it comes back as `null`
rather than as a number. `Infinity` is not JSON: a strict parser rejects the
whole message over it, so one unbounded quantity would take every number beside
it down.

## Backtest validation

Every other tool here answers a question about a price. These answer a question
about a claim: somebody says this strategy earns a Sharpe of 1.5, and the honest
reply depends on facts about the search that found it, none of which are in the
number.

```bash
abacus tools | grep -A2 deflated_sharpe_ratio
```

**The trial count that matters is the effective one.** Two hundred variations of
one moving-average rule are not two hundred independent bets, and deflating as
though they were over-penalises the result. On the forty one-factor trials in
`tests/test_validation.py` the eigenvalue method puts the effective count at
3.0 against a raw 40, and the deflated Sharpe at 0.8244 against 0.7196 — ten
points of probability the raw count throws away. On forty independent trials
both counts are 40.0 and both figures are 0.5591, identical to the digit. The
effective count is free where it is not needed and substantial where it is, so
it is what the deflation uses; both figures come back, and so do all three
estimates of the count, because they disagree — 2.49, 3.00 and 1.08 on the same
data.

**A Sharpe ratio here is per period, and its standard error is always beside
it.** The number people quote is annualised and these formulas take the
unannualised one, which is the quietest way to get a wrong answer out of this
group. The standard error is usually the answer anyway: an annualised 1.0 over
252 observations carries an annualised standard error of 1.00, and over 30
observations of 2.90.

**The bootstrap tools are seeded and say so.** `superior_predictive_ability`
resamples, so an unseeded call would return a different p-value each time, which
would make the idempotent annotation a lie and make two runs look like a change
in the data. The seed defaults to a fixed value and comes back in the result, so
a figure can be reproduced from the result alone.

**Refusals, because everything in this group will compute.** A deflated Sharpe
from four observations is a number. An overfitting probability from two
strategies is a number, drawn from a two-point distribution — with `k`
strategies the logit takes at most `k` distinct values, measured at 2 for two
strategies and 3 for three — and it prints to four decimal places exactly like a
real one. Skewness and kurtosis are refused when no distribution could have
them: every distribution satisfies `kurtosis >= 1 + skewness^2`, so a skewness
of -1.5 needs an excess kurtosis of at least 0.25.

## The skill

[`skill/SKILL.md`](skill/SKILL.md) is the overview the individual schemas cannot
be: which tool to reach for, the conventions the whole surface shares, and three
worked transcripts — a book and its Greeks, a portfolio's risk, and whether a
backtest is evidence of anything.

The transcripts are not prose about calls. They are fenced `transcript` blocks
holding the arguments verbatim, with `$name` placeholders for the handles
threaded between steps, and `tests/test_skill.py` parses them out and runs every
one against a live server. An argument renamed in a schema, a required field
added, a handle that stopped round-tripping: each shows up as a failing test
rather than as an example somebody copies and cannot make work.

The same file checks that every tool the skill names is registered and every
registered tool is named — a rename breaks the first, a new phase breaks the
second — and enforces a budget on the tool listing, which is 115,162 characters
today against a ceiling of 140,000, with no single tool over 12,000. The listing
is loaded before any work happens and grows with every group added, so the
ceiling exists to make crossing it a decision rather than a drift.

## Checking a server against the specification

The conformance suite drives a server through the wire format and reports on
requirements of the specification — not of this code — so it is meaningful
pointed somewhere else:

```bash
abacus conform                                   # this server, in process
abacus conform --stdio "abacus stdio"            # a launched command
abacus conform --http http://127.0.0.1:8000/mcp  # a running endpoint
abacus conform --http ... --json                 # machine-readable
```

It exits non-zero on a failure, so it works as a gate rather than only as a
report, and CI runs it over both transports and against the built wheel.

Checks cover discovery, `resultType` and server identity on every result, cache
hints on list results, version negotiation, the error-code allocation rules, the
distinction between a protocol error and a tool execution error, and — on
Streamable HTTP — header validation, the Base64 sentinel, and the verbs and
headers this revision retired. A requirement that cannot be tested against a
given server reports `skip` with the reason; counting an untested requirement as
satisfied would make the whole suite worthless.

The suite is itself tested by injecting one defect at a time into this server —
the handshake still implemented, `resultType` missing, a retired error code, an
execution failure escalated to a JSON-RPC error — and requiring that the check
written for that requirement is the one that turns red. Passing a healthy server
proves very little; failing a broken one on the right check is the evidence.

## Design

### The protocol core is written against the specification

Revision `2026-07-28` removed the `initialize` handshake and made MCP stateless.
There is no session: every request carries its own protocol version, client
identity and capabilities in `_meta`, and a server may not infer any of them
from the connection a request arrived on, because two requests on one stdio pipe
may belong to unrelated conversations.

Building the core directly against that model keeps it explicit, keeps the
runtime dependency list at a single entry, and makes the specification's own
requirements — the error-code allocation policy, the `$ref` rules, the header
validation — things the code enforces rather than things it assumes someone else
did.

### Failures are sorted by who can act on them

The specification draws a line between two kinds of failure, and the line is
about audience.

A **protocol error** is a JSON-RPC error. It says the request was not actionable
at all: an unknown tool, a malformed body, an unsupported protocol version. A
model cannot usually recover from one, because the fault is in the plumbing.

A **tool execution error** comes back as an ordinary result with `isError` set.
It says the call arrived intact and the server declined to answer it, and it is
addressed to the model: which field, what was wrong with it, what would be
accepted instead.

So calling `price_european_option` with a misspelled field gets back a result
rather than an error, naming both the mistake and the properties that are
accepted:

```
price_european_option was called with invalid arguments:
  /time: is required but was not given
  /vol: is required but was not given
  /volatility: is not a recognised property; accepted properties are:
    carry, rate, spot, strike, time, type, vol
```

Every violation is reported at once, so a caller with three problems learns
about three problems instead of fixing them one round trip at a time.

### Nonsense is refused before it is priced

A schema can say `vol` is a positive number. It cannot say that `vol: 20` is
twenty percent written the wrong way — and pricing it as two thousand percent
returns a number that is arithmetically correct and completely wrong, with
nothing downstream any the wiser. Inputs beyond plausible bounds are refused
with the conversion spelled out:

```
vol=20 is out of range; it looks like a percentage. This field is a
decimal fraction, so 20% is 0.2. Values above 5 are rejected as implausible.
```

The same applies to a rate given as `5` and an expiry given as a count of days.

Quantities that genuinely do not exist are not invented either. At expiry the
price is still well defined as a limit and is returned, but `d1` and `d2` are
omitted rather than filled with an infinity that would read downstream as a real
number, and the Greeks are refused outright because the payoff is kinked at the
strike and the derivative does not exist there. A local volatility at a point
where the Dupire identity has no answer comes back marked inadmissible, naming
the two quantities that failed, rather than as a `nan` that is not portable JSON
and reads as a number to anything parsing it loosely.

### A surface is not returned as a grid of numbers

The obvious representation of a volatility surface is a dense matrix, and it is
the one a language model reads worst: an unlabelled nested array gives no clue
which axis is maturity and which is strike, and a transposed reading still looks
plausible. A surface therefore comes back three ways at once, in descending
order of how much trust each deserves:

1. **The five SVI parameters per slice** — exact, sufficient to rebuild the
   slice, and small enough to survive a conversation without truncation.
2. **Named scalars** — the at-the-money volatility, the wing slopes against
   Lee's bound, the fit residuals.
3. **A labelled grid, only when asked for**, where every row carries its
   maturity and every cell its strike as an explicit key, so a row cannot be
   read as a column.

Both no-arbitrage conditions are reported as flags with the offending location
attached, rather than left to be inferred. They are scanned *between* the quoted
maturities as well as at them, because quoted slices are usually fitted to be
admissible and interpolation is where the condition quietly stops holding.

### A numerical answer carries how it was produced

An American price is the output of a method, not a formula, so its error is
invisible in the number. A lattice price at 64 layers and the same price at 4096
layers are different numbers, and a tool returning a bare float invites a caller
to treat a discretisation artefact as a market fact. So every American valuation
carries the method, the resolution, the early-exercise premium, and on request a
ladder of prices as the grid doubles, with the last change quoted as an error
*estimate* — described as such, because nothing here proves a bound.

Two European prices are reported beside it, analytic and on-lattice, because the
premium is the lattice-internal difference: the discretisation error is common
to both legs and largely cancels, which makes it the better estimate. The cost
is that `price - europeanPrice` does not reproduce it, so both are given rather
than leaving a reader to find the discrepancy and distrust all three.

The boundary read-out carries a similar caveat. On a binomial lattice it
alternates between two values one node apart, because consecutive layers sample
interleaved node grids of opposite parity — the true boundary is monotone, and
the wobble is the discretisation rather than the option. The tool says so and
points at the trinomial lattice, which has no such parity.

### A book is carried by its handle, not stored behind one

A position book is state, and this revision has nowhere to keep it. The obvious
implementation — a dictionary on the server, keyed by a random string handed
back to the caller — fails three ways that have nothing to do with taste. It
reintroduces the session the revision removed, and with no session scope the
book is reachable by whoever presents the key. It grows without bound, because
nothing in the protocol tells a server that a caller has finished. And it breaks
across processes, since a handle minted by one worker is unknown to the next.

So the handle *contains* the book: canonical JSON, compressed, and authenticated
with a truncated HMAC-SHA256 that is checked in constant time before anything is
decompressed. The server stores nothing and verifies everything, which disposes
of all three problems at once.

The trade is stated rather than buried. The payload is signed, not encrypted, so
the caller can read it — acceptable because it is the caller's own book, and a
reason nothing else may be put in there. It cannot be altered, because an edited
handle fails its MAC. It is bounded in size, so a book too large to encode is
refused when it is opened rather than minting something that fails on use. And
the key lives for the life of the process, which is why a handle that fails its
check is reported as *unrecognised* rather than expired: a caller seeing that
after a working call has learned something true about the server.

Books are therefore immutable. Amending one mints a new handle and leaves the
old one working until it expires, and the result says so — the server holds no
record of either and could not revoke the old one if it claimed to.

### An aggregate says which legs it covers

A leg that has expired, or that carries no volatility, has no derivative: the
payoff is kinked and there is nothing to differentiate. Contributing a zero for
it would report a book as flat in precisely the case where part of it has no
delta at all, and nothing downstream could tell.

Such legs are priced into the total — the price is a limit and exists — but
excluded from the aggregate Greeks, which then come back with `complete: false`,
the indices of the excluded legs, and a sentence saying the aggregate covers the
remaining legs only.

Scenario grids are labelled for the reason surfaces are. A spot-versus-vol grid
is square often enough that a transposed reading still looks plausible, so every
row states its volatility shift and every cell states the spot it was priced at.

### The reference client implements this revision and nothing else

There is no `initialize`, no session header, no fallback to an older shape, and
no accommodation for a server that answers the way servers used to. That is the
point: MCP moved, most published guidance still describes the stateful form, and
a client that quietly tolerated it would make a non-conforming server look fine.

It is also deliberately thin on judgement. It checks only the JSON-RPC framing
that must hold for a response to be matched to its request, and returns
everything else untouched — because a client that raised on a missing
`resultType` could not report on one.

### Header validation is a security control, not a formality

Streamable HTTP mirrors selected body fields into headers so intermediaries can
route without parsing the body. That creates two sources of truth for one fact.
If a load balancer routes on `Mcp-Name: read_file` while the server executes a
`params.name` of `delete_everything`, every control in front of the server was
applied to a request that is not the one that ran.

So a disagreement between header and body is refused with `HeaderMismatch` and
`400` — and the comparison is made on *decoded* values, since a server that
skipped the check for Base64-encoded headers would have handed an attacker the
way around it.

The transport also declines the older shape of itself: `405` for the GET and
DELETE that used to open a stream and end a session, `Mcp-Session-Id` ignored
rather than echoed, and `404` *with a JSON-RPC body* for an unknown method, which
is what lets a client tell a modern server that lacks the method from a legacy
endpoint that was never there.

## Development

```bash
pip install -e ".[dev]"
pytest          # 629 tests
mypy --strict
ruff check .
```

Continuous integration runs the suite on Python 3.10 through 3.13, type-checks
and lints, runs the conformance suite over both transports, and installs the
wheel *and* the sdist into separate clean environments to confirm each one's entry
point answers a real request and passes conformance — a distribution that imports
but cannot serve is not a working server, and the two artefacts are built by
different code paths.

The declared dependencies are ordinary bounded version ranges. One of them used to
be a `git+https` direct reference, which resolves perfectly well locally
and is refused outright when a distribution carrying it is uploaded to an index —
so the server was unpublishable while every check was green. `tests/test_metadata.py`
now fails if such a requirement reappears. Until the library has a release on the
index, CI builds it from its repository into a local wheelhouse and lets pip
resolve the declared range against that; the resolution path is the one an index
install takes, and only the source of the file differs.

### Releasing

The version lives in `pyproject.toml`, is mirrored by `abacus.__version__`, and a
test asserts they agree. Pushing `v<version>` builds both artefacts, checks the
metadata the way the index will, installs each into a clean environment and makes
it pass conformance, and then publishes — using the index's trusted-publishing
flow, so there is no upload token in this repository or in its secrets.

Prices and Greeks are checked against the library exactly and, independently,
against finite differences of the prices the pricing tool itself returns. Those
come from different formulae, so agreement is evidence the wiring is right
rather than merely self-consistent.

## Licence

MIT.
