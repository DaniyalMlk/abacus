"""Command line entry point.

``abacus stdio`` is what an MCP client launches. ``abacus http`` runs the same
server over Streamable HTTP. ``abacus tools`` prints the tool surface, which is
the quickest way to see what the server offers without speaking the protocol at
it. ``abacus conform`` runs the conformance suite — against this server in
process by default, and against a launched command or a running endpoint when
told to, since the suite tests the specification rather than this code.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence

from . import __version__
from .analytics import default_registry
from .client import Client, HttpTransport, InProcessTransport, StdioTransport, Transport
from .conformance import run as run_conformance
from .protocol import PROTOCOL_VERSION
from .server import Server, capabilities
from .tools import ToolRegistry, install
from .transports import serve_http, serve_stdio

INSTRUCTIONS = (
    "Option and portfolio analytics. Prices European and American options and "
    "reports their Greeks, recovers implied volatility from a price, fits SVI "
    "volatility slices and surfaces with both no-arbitrage checks, extracts Dupire "
    "local volatilities, aggregates option positions into a book, and estimates "
    "portfolio risk from a matrix of returns: covariance with shrinkage, value at "
    "risk and expected shortfall by five methods, Euler risk contributions, risk "
    "parity weights and drawdown statistics. "
    "Volatilities and rates are decimal fractions, not percentages: 20% is 0.2, "
    "and so is a 20% return. Time is a year fraction, not a count of days: thirty "
    "days is roughly 0.082. Log-moneyness is measured on the forward, as "
    "log(strike / forward). A value at risk is a positive loss over one period of "
    "whatever frequency the returns have, so periodsPerYear must be given rather "
    "than assumed. Every risk result names the method that produced it, because "
    "the number does not."
)


def build_server(registry: ToolRegistry | None = None) -> Server:
    """Assemble the server with its tools installed."""
    server = Server(
        name="abacus",
        version=__version__,
        title="Abacus analytics",
        instructions=INSTRUCTIONS,
        # `listChanged` is false and honest: the tool set is fixed at startup,
        # so promising change notifications would mean promising a stream that
        # would never carry anything.
        capabilities=capabilities(tools={"listChanged": False}),
    )
    install(server, registry if registry is not None else default_registry())
    return server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abacus",
        description="An MCP server exposing option and portfolio analytics.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"abacus {__version__} (MCP {PROTOCOL_VERSION})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stdio", help="serve over stdio, which is how a client launches it")

    http = sub.add_parser("http", help="serve over Streamable HTTP")
    http.add_argument("--host", default="127.0.0.1", help="default: loopback only")
    http.add_argument("--port", type=int, default=8000)
    http.add_argument("--endpoint", default="/mcp")
    http.add_argument(
        "--allow-origin",
        action="append",
        default=None,
        metavar="ORIGIN",
        help="permit this Origin; repeatable. Omitted, no Origin is permitted.",
    )

    tools = sub.add_parser("tools", help="print the tool surface and exit")
    tools.add_argument("--schemas", action="store_true", help="include the full JSON schemas")

    conform = sub.add_parser(
        "conform",
        help="run the conformance suite; in process unless a target is named",
    )
    target = conform.add_mutually_exclusive_group()
    target.add_argument(
        "--stdio",
        metavar="COMMAND",
        help='launch a server and drive it over stdio, e.g. --stdio "abacus stdio"',
    )
    target.add_argument(
        "--http",
        metavar="URL",
        help="drive a running Streamable HTTP endpoint, e.g. --http http://127.0.0.1:8000/mcp",
    )
    conform.add_argument("--json", action="store_true", help="emit the report as JSON")

    return parser


def _conform(args: argparse.Namespace) -> int:
    """Run the conformance suite against whichever target was named.

    The HTTP checks are skipped rather than failed when the target is not an
    endpoint, because a server reached over stdio is not required to have one.
    """
    http: HttpTransport | None = None
    transport: Transport
    if args.http:
        http = HttpTransport(args.http)
        transport = http
    elif args.stdio:
        transport = StdioTransport(shlex.split(args.stdio))
    else:
        transport = InProcessTransport(build_server())

    client = Client(transport)
    try:
        report = run_conformance(client, http=http)
    finally:
        client.close()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    server = build_server()

    if args.command == "stdio":
        serve_stdio(server)
        return 0

    if args.command == "http":
        origins = frozenset(args.allow_origin) if args.allow_origin else None
        httpd = serve_http(
            server,
            host=args.host,
            port=args.port,
            endpoint=args.endpoint,
            allowed_origins=origins,
        )
        print(
            f"abacus {__version__} serving MCP {PROTOCOL_VERSION} "
            f"on http://{args.host}:{args.port}{args.endpoint}",
            file=sys.stderr,
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
        return 0

    if args.command == "conform":
        return _conform(args)

    if args.command == "tools":
        registry = default_registry()
        for tool in registry:
            print(f"{tool.name}\n    {tool.description}")
            if args.schemas:
                print(json.dumps(tool.input_schema, indent=2))
            print()
        return 0

    return 1  # pragma: no cover - argparse rejects anything else first


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
