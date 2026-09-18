# abacus

An MCP server that exposes option and portfolio analytics as tools.

Language models are unreliable at arithmetic, so the useful thing a tool
boundary can do is move the numbers somewhere trustworthy. `abacus` puts them in
[`moneyness`](https://github.com/DaniyalMlk/moneyness), an options library whose
numerical core is validated against closed-form results and high-precision
references, and exposes that core over the Model Context Protocol.

It targets MCP revision **2026-07-28** and has one runtime dependency.

## Running it

```bash
pip install git+https://github.com/DaniyalMlk/abacus.git

abacus stdio          # what an MCP client launches
abacus http           # Streamable HTTP on 127.0.0.1:8000/mcp
abacus tools          # print the tool surface and exit
abacus conform        # run the conformance suite against this server
```

To register it with a client that launches servers over stdio:

```json
{
  "mcpServers": {
    "abacus": { "command": "abacus", "args": ["stdio"] }
  }
}
```

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

Conventions, which are also stated in the server's instructions and in every
schema description: volatilities and rates are decimal fractions, so 20% is
`0.2`; time is a year fraction, so thirty days is about `0.082`; log-moneyness
is measured on the forward as `log(strike / forward)`.

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
pytest          # 507 tests
mypy --strict
ruff check .
```

Continuous integration runs the suite on Python 3.10 through 3.13, type-checks
and lints, runs the conformance suite over both transports, and installs the
built wheel into a clean environment to confirm the entry point answers a real
request and passes conformance — a wheel that imports but cannot serve is not a
working server.

Prices and Greeks are checked against the library exactly and, independently,
against finite differences of the prices the pricing tool itself returns. Those
come from different formulae, so agreement is evidence the wiring is right
rather than merely self-consistent.

## Licence

MIT.
