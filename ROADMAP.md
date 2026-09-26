# Roadmap

Phases are ordered but not dated. An item is checked when the code is written,
exercised end to end, and covered by tests that run.

## Phase 1 — Protocol core

- [x] JSON-RPC 2.0 message parsing and serialisation with strict shape checks
- [x] Per-request `_meta` handling: protocol version, client info, client capabilities
- [x] `resultType` on every result; `complete` and `input_required` variants
- [x] MCP error codes, including the `-32020`..`-32099` reserved range
- [x] `server/discover` with supported versions, capabilities and server identity
- [x] Protocol version negotiation with `UnsupportedProtocolVersionError`
- [x] Capability gating with `MissingRequiredClientCapabilityError`
- [x] Method dispatch and a registry decoupled from transport

## Phase 2 — Tool surface over pricing and Greeks

- [x] JSON Schema 2020-12 validator covering the keywords the tool schemas use
- [x] Tool registry with input and output schemas, titles and annotations
- [x] `tools/list` with deterministic ordering, pagination and cache hints
- [x] `tools/call` returning both `content` and `structuredContent`
- [x] Structured, recoverable execution errors distinct from protocol errors
- [x] Domain validation that rejects economically meaningless inputs
- [x] European pricing and the full Greek set exposed as tools

## Phase 3 — Implied volatility and the volatility surface

- [x] Implied volatility tool with bracketing diagnostics and no-solution reporting
- [x] Surface calibration tool over a quote set
- [x] Surface representation that stays legible when read as text
- [x] Arbitrage diagnostics reported as data, not prose
- [x] Local volatility extraction exposed as a tool

## Phase 4 — American exercise

- [x] Lattice pricing tool with step control and convergence reporting
- [x] Closed-form American approximation tool
- [x] Early exercise boundary exposed as a queryable series
- [x] Method selection reported alongside every American price

## Phase 5 — Position book

- [x] Position and book model with an explicit server-minted handle
- [x] Aggregate Greeks across a book
- [x] Scenario grids over spot and volatility
- [x] Handle lifetime, expiry and recovery errors

## Phase 6 — Transport and authorization hardening

- [x] stdio transport with framing and backward-compatibility probing
- [x] Streamable HTTP transport with a single POST endpoint
- [x] Standard request header validation and `HeaderMismatch` rejection
- [x] Origin validation and localhost binding by default
- [x] Request size limits, timeouts and schema resource bounds

## Phase 7 — Conformance, documentation and continuous integration

- [x] Conformance suite driving the server through the wire format
- [x] Reference client used by the conformance suite
- [x] Command line entry point for both transports
- [x] README covering the tool surface and the design decisions
- [x] Continuous integration across supported Python versions with types and lint

## Phase 8 — Installable in one line

- [x] Release metadata, classifiers and keywords on the server and on each library it depends on
- [x] Tag-driven release workflow using trusted publishing, so no credential is stored
- [x] Depend on released versions rather than repository URLs, which package indexes reject
- [x] A test that fails if a direct-reference requirement returns, since the build,
      the suite and the type checker all stayed green while one was present
- [x] `uvx abacus-mcp stdio` and `pip install abacus-mcp` verified from a clean
      environment, resolving the dependency by name against a wheelhouse standing
      in for the index, and passing conformance from both the wheel and the sdist
- [x] Registration snippets for the clients people actually use
- [ ] A first release on the index, which waits on the publisher being registered
      there for this project and for the library it depends on

## Phase 9 — Portfolio risk tools

- [x] Value at risk and expected shortfall, parametric and historical, with the method named in the result
- [x] Shrinkage covariance from a returns matrix, reporting the shrinkage intensity it chose
- [x] Risk contributions and concentration, so the answer says where the risk is
- [x] Risk parity weights, with the convergence evidence rather than a promise
- [x] Drawdown and tail statistics
- [x] Returns handles, so a matrix is estimated once and reused across calls — carrying
      the mean vector and the covariance matrix rather than the returns, because a
      handle has to fit in a message and a returns matrix does not, and refusing the
      methods that read the path rather than answering them from a summary
- [x] Every tool called over the wire with a good input and a bad one, including
      through a launched process, with each refusal arriving as a recoverable result

## Phase 10 — Curve and bond tools

- [x] Discount curve bootstrapping from deposits, futures and swaps, reporting the
      per-instrument solve residuals and whether the curve reprices its own inputs
- [x] Discount factors, zero rates and forwards read off a curve, refusing a date past
      its horizon rather than extrapolating past the longest instrument
- [x] Bond analytics: price, yield, duration, convexity — from a yield *and* from a
      curve, both reported with the gap between them, because they answer different
      questions and neither is wrong
- [x] Key rate risk, with the durations summing to the effective duration and the
      residual stated, plus level, slope and curvature, plus instrument risk in the
      things that can actually be traded
- [x] Option-adjusted spread off a lattice calibrated to the curve, solved against the
      observed price rather than the model's own, with the lattice's repricing check
      and the option cost
- [x] Curve handles, carrying the pillars and the quotes behind them — a curve is two
      numbers per instrument, so unlike the returns handle this one loses nothing
- [x] Every tool called over the wire with a good input and a bad one, including
      through a launched process

## Phase 11 — Execution and backtest-validation tools

- [x] Implementation shortfall decomposition over an order, its fills and the market
- [x] Optimal execution trajectory and the cost-risk frontier
- [x] Deflated Sharpe ratio, minimum track record length and effective trial count
- [x] Probability of backtest overfitting and the test for superior predictive ability
- [x] Refusals rather than numbers when the inputs cannot support one

## Phase 12 — The skill

- [x] A packaged skill covering when to reach for which tool and the conventions they share
- [x] Worked transcripts: pricing a book, sizing its risk, judging a backtest
- [x] A budget on the tool surface, so loading it stays cheap as it grows
- [x] The skill checked against the server it describes, in continuous integration

## Phase 13 — Documentation

- [x] A published documentation site: tools, conventions, worked examples
- [x] README rewritten around what a user has working in the first minute
- [x] A validation table: every number the documentation claims, and where it is checked
- [x] One page explaining what each library underneath does and when to use it directly

## Phase 14 — Gaps the libraries had, found by exposing them

Two analytics that the flagship had a clear use for and the libraries
underneath could not provide. Both were built there and surfaced here.

- [x] `validate_risk_model`: the server could compute a value at risk five ways
      and could not say whether any of them worked
- [x] Kupiec, Christoffersen, conditional coverage and the supervisory traffic
      light, with the zone derived from the binomial so it adapts to the sample
      length rather than only answering for 250 observations
- [x] The Acerbi-Szekely statistics where expected-shortfall forecasts are
      supplied, with a simulated null and a bound on the replications
- [x] `bond_carry_rolldown`: `bond_analytics` prices a bond and `bond_curve_risk`
      says what moves it; neither answered what a position earns if nothing
      happens
- [x] Both callable over stdio and HTTP with a good input and a bad one, the bad
      one arriving as a result a model can repair
- [x] The curve handle reused by the horizon tool, round-tripped through a
      launched subprocess
- [x] Documented on the site, named in the packaged skill, inside the tool budget

The clustering test is the one worth having. A constant-volatility model can
breach exactly the right number of times over a year and put every breach in the
same fortnight; a count cannot see that and no rescaling fixes it. Driven end to
end through the tool over a regime-switching process, a breach is followed by
another 10.00% of the time against 1.84% after a calm day.

The bond tool reports both market meanings of "carry" because the conventional
one ranks positions backwards: a 9% bond shows +4.83 of income less financing
against a zero-coupon bond's -2.00, and earns 27 basis points less over the year.

Adding the tools broke the docs-site group table and the packaged-skill test
until both named them, which is what those tests are for. The tool listing went
from 115,162 characters to 128,716 against a budget of 140,000.

## Phase 15 — The validator's input, produced here

`validate_risk_model` scored a forecast series and made the caller bring one,
which left it the single tool on this surface presupposing work done somewhere
else. A model holding returns and nothing else could not use it.

- [x] `conditional_volatility`: a GARCH(1,1) fitted by maximum likelihood, with
      the persistence, the half-life of a shock, and the horizon volatility
      against what square-root-of-time would give
- [x] A one-step-ahead forecast per period, already aligned, so it goes to the
      validator with nothing offset by the caller
- [x] The forecast series capped, and withholdable, since it is one number per
      observation and the large part of the payload
- [x] Convergence reported honestly rather than a best-effort point returned as
      though it were a fit
- [x] The two composing, over the wire, with the clustering verdict changing

Over ten independent regime-switching samples driven through both tools, a
constant forecast has its breaches rejected as clustered nine times and the
fitted forecast once. The breach count goes from about 51 to about 27 against a
nominal 20.

Halved rather than fixed. A Gaussian GARCH still understates the tail of a
series whose standardised residuals are fat, and the note says so rather than
claiming the model solves the problem — a tool whose description oversells it
is worse than one that does less.

Alignment is the thing this makes impossible rather than documents. A forecast
series offset by one period against its returns scores a different model
entirely and looks completely normal doing it.

The tool listing went from 128,716 to 130,795 characters against 140,000.

## Phase 16 — The innovation tail, and a field with no incumbent

Two gaps found by using the surface rather than by reading it.

- [x] `conditional_volatility` estimates the innovation tail instead of assuming
      a normal one, testing for it by likelihood ratio rather than assuming
      either answer
- [x] The quantile multiplier returned, so the standardised quantile is not
      reconstructed by hand
- [x] Conditional value at risk and expected shortfall in the same payload
- [x] The note's claim about a Gaussian fit replaced by what the numbers now are
- [x] `model_confidence_set`, for a set of candidates with no benchmark among them
- [x] Both driven over the wire with a good input and a bad one
- [x] The listing still inside its budget, with the figure to prove it

The multiplier is the part that matters most and looks least interesting. The
standardised Student-t quantile is the raw one times `sqrt((v-2)/v)`, and the raw
one is 41% larger at four degrees of freedom. A caller who reached for the raw
quantile widened every forecast and undershot the breach count, which reads as
conservatism rather than as an error — so the number is computed here instead of
being described. It is backed out of the fitted risk rather than recomputed, so
the two cannot drift, and the mean is removed: multiplying a whole forecast
series by a number with a drift folded into it scales that drift by each period's
volatility.

Measured on regime-switching series: the 99% breach count over 2000 observations
falls from 28.2 under normal innovations to 22.45 against a nominal 20, about 70%
of what the variance model left behind. Not all of it, and the note says why. On
a series that never had a fat tail the two agree to within half a breach in
twenty, which is what stops this from being a forecast widened indiscriminately.

The confidence set answers a question the surface could not ask. Nominating the
sample-best as a benchmark and calling `superior_predictive_ability` chooses the
benchmark with the data the test runs on. Its answer is uncomfortable and the
note leads with it: on the thirty-rule sweep with a genuine drift, 29 of 30 rules
survive at 10% and the surviving set spans 9.3% of annualised mean return.

Two things about it read backwards without being told, so the note says both. The
size of the set is the result rather than a shortcoming of it, and a smaller alpha
gives a larger set.

The tool listing went from 131,635 to 134,053 characters against 140,000. That is
the last tool that fits without raising the budget deliberately.

## Phase 17 — A horizon figure the caller does not have to build

`conditional_volatility` reported an aggregate horizon volatility and invited
exactly the step this library refuses one period out: multiplying it by a
quantile. There is no quantile to multiply by — the sum of the horizon's
innovations is not a member of the family they were drawn from — and the error the
substitution makes does not have a fixed sign.

- [x] Optional horizon simulation, off unless `paths` is given
- [x] The innovation draw selectable between the fitted family and a resample of
      the model's own standardised residuals
- [x] The Monte Carlo standard error in the payload, described as the lower bound
      it measured as
- [x] Both square-root-of-time ratios reported, on the volatility and the quantile
- [x] A short history refused for the bootstrap, with the parametric route named
- [x] Over the wire with a good input and two bad ones

Measured, and the shape of the measurement is the interesting part. The gap
between the simulated figure and the substitution came to 2.5, 11.0, 8.2 and 0.6
standard errors over four samples at ten steps and 40,000 paths. Real on average
and not decisive on every series, because how far a horizon quantile departs from
a scaled one depends on where the fit sits relative to its long-run level.

The direction is sample-dependent too, which is why the tool reports both ratios
and the tests assert neither sign on a single series. Over five fat-tailed samples
the quantile ratio ranged 0.87 to 1.11 with a mean of 0.97, against a mean of 1.08
under normal innovations. The two effects are a stochastic variance path making
the total leptokurtic, and aggregation pulling a fat innovation's total towards
normality.

The listing went from 134,053 to 134,992 characters against 140,000. Extending an
existing tool rather than adding one is what kept that affordable; the next
addition needs the budget raised as a decision or a schema trimmed.
