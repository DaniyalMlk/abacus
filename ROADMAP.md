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

- [ ] Value at risk and expected shortfall, parametric and historical, with the method named in the result
- [ ] Shrinkage covariance from a returns matrix, reporting the shrinkage intensity it chose
- [ ] Risk contributions and concentration, so the answer says where the risk is
- [ ] Drawdown and tail statistics
- [ ] Returns handles, so a matrix is uploaded once and reused across calls

## Phase 10 — Curve and bond tools

- [ ] Discount curve bootstrapping from deposits, futures and swaps
- [ ] Bond analytics: price, yield, duration, convexity
- [ ] Key rate risk and option-adjusted spread
- [ ] Curve handles, so a bootstrapped curve is reused rather than rebuilt per call

## Phase 11 — Execution and backtest-validation tools

- [ ] Implementation shortfall decomposition over an order, its fills and the market
- [ ] Optimal execution trajectory and the cost-risk frontier
- [ ] Deflated Sharpe ratio, minimum track record length and effective trial count
- [ ] Probability of backtest overfitting and the test for superior predictive ability
- [ ] Refusals rather than numbers when the inputs cannot support one

## Phase 12 — The skill

- [ ] A packaged skill covering when to reach for which tool and the conventions they share
- [ ] Worked transcripts: pricing a book, sizing its risk, judging a backtest
- [ ] A budget on the tool surface, so loading it stays cheap as it grows
- [ ] The skill checked against the server it describes, in continuous integration

## Phase 13 — Documentation

- [ ] A published documentation site: tools, conventions, worked examples
- [ ] README rewritten around what a user has working in the first minute
- [ ] A validation table: every number the documentation claims, and where it is checked
- [ ] One page explaining what each library underneath does and when to use it directly
