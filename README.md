# abacus

An MCP server that exposes option and portfolio analytics as tools.

Language models are unreliable at arithmetic, so the useful thing a tool
boundary can do is move the numbers somewhere trustworthy. `abacus` puts them
in [`moneyness`](https://github.com/DaniyalMlk/moneyness), an options library
whose numerical core is validated against closed-form results and
high-precision references, and exposes that core over the Model Context
Protocol.

## Status

Under construction. See [ROADMAP.md](ROADMAP.md) for what is built and what is
not.

## Design

**The protocol core is implemented directly against the specification rather
than taken from a framework.** The server targets MCP revision `2026-07-28`,
which removed the `initialize` handshake and made the protocol stateless: every
request carries its own protocol version, client identity and capabilities in
`_meta`, and no state is inferred from the connection it arrived on. Writing
the core against the specification keeps that model explicit and keeps the
runtime dependency list at one entry.

**Validation happens before arithmetic, and failures come back as data.** A
malformed tool call returns a structured execution error the caller can act
on — which field, what was wrong with it, what would be acceptable — rather
than a stack trace. Inputs that parse but are economically meaningless are
rejected in the same way, so a number is never computed from nonsense and
returned as if it meant something.

## Licence

MIT.
