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

- [ ] Position and book model with an explicit server-minted handle
- [ ] Aggregate Greeks across a book
- [ ] Scenario grids over spot and volatility
- [ ] Handle lifetime, expiry and recovery errors

## Phase 6 — Transport and authorization hardening

- [x] stdio transport with framing and backward-compatibility probing
- [x] Streamable HTTP transport with a single POST endpoint
- [x] Standard request header validation and `HeaderMismatch` rejection
- [x] Origin validation and localhost binding by default
- [x] Request size limits, timeouts and schema resource bounds

## Phase 7 — Conformance, documentation and continuous integration

- [ ] Conformance suite driving the server through the wire format
- [ ] Reference client used by the conformance suite
- [x] Command line entry point for both transports
- [x] README covering the tool surface and the design decisions
- [x] Continuous integration across supported Python versions with types and lint
