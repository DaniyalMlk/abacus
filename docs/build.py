"""Build the documentation site from the live registry.

The reason this is a generator rather than a folder of hand-written pages is
that a hand-written page describes the server somebody had in mind when they
wrote it. Every tool entry here is rendered from the registry the server
actually serves — the same objects a client receives from ``tools/list`` — so a
renamed argument or a retired tool changes the site on the next build and there
is no version of it that is confidently wrong.

What is *not* generated is the prose: the conventions, the worked examples, the
page on the libraries underneath. Those are judgements and they are written out
below. The seam between the two is deliberate and narrow: generated content is
everything that restates the schemas, and written content is everything that
says what to do about them.

Run it with ``python docs/build.py`` and it writes ``site/``. No dependency
beyond the standard library and the server itself, because a documentation build
that needs a toolchain is a documentation build that stops working.
"""

from __future__ import annotations

import html
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claims import CLAIMS

from abacus import __version__
from abacus.analytics import default_registry
from abacus.protocol import SUPPORTED_VERSIONS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site"
SOURCE = "https://github.com/DaniyalMlk/abacus"

#: The from-source install, for as long as nothing is on the package index. The
#: shell line-continuations are joined here rather than written as a literal
#: block: this module is Python source, and a backslash before a newline inside
#: a string literal is a line continuation that silently removes both.
_LIBRARIES = ("moneyness", "shortfall", "tenor", "slippage", "holdout")
FROM_SOURCE = " \\\n  ".join(
    ["pip install"] + [f'"git+{SOURCE.rsplit("/", 1)[0]}/{name}.git"' for name in _LIBRARIES]
) + f'\npip install "git+{SOURCE}.git"'


@dataclass(frozen=True)
class Group:
    """One family of tools, and the sentence that says when to reach for it."""

    slug: str
    name: str
    blurb: str
    tools: tuple[str, ...]


GROUPS: tuple[Group, ...] = (
    Group(
        "options",
        "Options",
        "Pricing, Greeks and the checks worth running before either. "
        "Generalised Black-Scholes-Merton, so one parameterisation covers shares, "
        "futures and currencies by the cost of carry you pass.",
        (
            "price_european_option",
            "european_option_greeks",
            "european_option_analytics",
            "put_call_parity",
            "option_price_bounds",
        ),
    ),
    Group(
        "volatility",
        "Volatility",
        "Recovering a volatility from a price, and fitting a surface to a set of "
        "them with both no-arbitrage conditions reported rather than assumed.",
        (
            "implied_volatility",
            "fit_volatility_slice",
            "fit_volatility_surface",
            "local_volatility",
        ),
    ),
    Group(
        "american",
        "American exercise",
        "Two methods and the boundary between exercising and holding. The method "
        "is named in every result, because a lattice price and a closed-form "
        "approximation are not the same claim.",
        (
            "price_american_lattice",
            "price_american_closed_form",
            "american_exercise_boundary",
        ),
    ),
    Group(
        "book",
        "Position books",
        "More than one position at once, carried by a handle so the book is sent "
        "once and asked several questions.",
        (
            "open_position_book",
            "describe_position_book",
            "amend_position_book",
            "position_book_greeks",
            "position_book_scenarios",
        ),
    ),
    Group(
        "risk",
        "Portfolio risk",
        "Covariance, tail risk, where the risk sits, what the path did, and "
        "whether the forecast was any good once the period has passed. Estimate "
        "once and reuse the estimate; the tools that read the return path say "
        "so rather than guessing.",
        (
            "estimate_return_moments",
            "portfolio_tail_risk",
            "portfolio_risk_contributions",
            "risk_parity_weights",
            "portfolio_drawdown",
            "validate_risk_model",
        ),
    ),
    Group(
        "curves",
        "Curves and bonds",
        "A discount curve from the instruments that trade, and the bond analytics "
        "that hang off it — pricing, risk, spreads, and what a position earns "
        "over a holding period. Every calibration reports whether it reprices "
        "what it was built from.",
        (
            "bootstrap_discount_curve",
            "discount_curve_rates",
            "bond_analytics",
            "bond_curve_risk",
            "bond_spreads",
            "bond_carry_rolldown",
        ),
    ),
    Group(
        "execution",
        "Execution",
        "What a trade cost, and how the next one should be spread out. The total "
        "is the least useful number; the split into delay, trading and "
        "opportunity is the one with a remedy attached.",
        (
            "decompose_implementation_shortfall",
            "optimal_execution_schedule",
            "execution_cost_frontier",
        ),
    ),
    Group(
        "validation",
        "Backtest validation",
        "Whether a track record is evidence of anything, given how many things "
        "were tried to find it. Everything here will compute a number; several of "
        "these tools refuse instead, which is the point of them.",
        (
            "deflated_sharpe_ratio",
            "effective_trial_count",
            "minimum_track_record_length",
            "backtest_overfitting_probability",
            "superior_predictive_ability",
        ),
    ),
)

PAGES = (
    ("index.html", "Overview"),
    ("conventions.html", "Conventions"),
    ("tools.html", "Tool reference"),
    ("validation.html", "Validated numbers"),
    ("libraries.html", "The libraries underneath"),
)


# -- rendering helpers -------------------------------------------------------


def esc(text: str) -> str:
    return html.escape(str(text), quote=False)


def code(text: str) -> str:
    return f"<code>{esc(text)}</code>"


def type_of(schema: dict[str, Any]) -> str:
    """The argument's type as a reader needs it, not as JSON Schema spells it."""
    kind = schema.get("type", "any")
    if kind == "array":
        items = schema.get("items", {})
        inner = items.get("type", "any")
        if inner == "array":
            return "number[][]" if items.get("items", {}).get("type") == "number" else "array[]"
        if inner == "object":
            return "object[]"
        return f"{inner}[]"
    if "enum" in schema:
        return " | ".join(str(value) for value in schema["enum"])
    return str(kind)


def arguments_table(schema: dict[str, Any]) -> str:
    properties: dict[str, Any] = schema.get("properties", {})
    if not properties:
        return ""
    required = set(schema.get("required", []))
    rows = []
    # Required arguments first: a reader scanning for what they must supply
    # should not have to read the whole table to find out.
    order = sorted(properties, key=lambda name: (name not in required, name))
    for name in order:
        field = properties[name]
        flag = '<span class="req">required</span>' if name in required else ""
        rows.append(
            "<tr>"
            f"<td>{esc(name)}{flag}</td>"
            f"<td>{esc(type_of(field))}</td>"
            f"<td>{esc(field.get('description', ''))}</td>"
            "</tr>"
        )
    return (
        '<div class="scroller"><table class="args">'
        "<thead><tr><th>Argument</th><th>Type</th><th>Meaning</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def rail(current: str, tool_index: bool) -> str:
    links = []
    for href, label in PAGES:
        mark = ' aria-current="page"' if href == current else ""
        links.append(f'<li><a href="{href}"{mark}>{esc(label)}</a></li>')

    sections = [f"<h2>Documentation</h2><ul>{''.join(links)}</ul>"]
    if tool_index:
        for group in GROUPS:
            entries = "".join(
                f'<li><a class="tool-link" href="#{esc(name)}">{esc(name)}</a></li>'
                for name in group.tools
            )
            sections.append(f"<h2>{esc(group.name)}</h2><ul>{entries}</ul>")

    return (
        '<aside class="rail">'
        f'<a class="wordmark" href="index.html">abacus<span>MCP analytics server</span></a>'
        f"<nav>{''.join(sections)}</nav>"
        "</aside>"
    )


def page(*, filename: str, title: str, body: str, tool_index: bool = False) -> str:
    description = (
        "abacus is an MCP server exposing option pricing, portfolio risk, "
        "interest rate curves, execution cost and backtest validation as tools."
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} &middot; abacus</title>
<meta name="description" content="{esc(description)}">
<meta name="color-scheme" content="light dark">
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="shell">
{rail(filename, tool_index)}
<main>
{body}
<footer>
<p>abacus {esc(__version__)}, speaking MCP {esc(", ".join(SUPPORTED_VERSIONS))}.
Built from the registry this server serves, so nothing here describes a tool that
does not exist. <a href="{SOURCE}">Source</a>.</p>
</footer>
</main>
</div>
</body>
</html>
"""


# -- the pages ---------------------------------------------------------------


def overview() -> str:
    total = len(list(default_registry()))
    return f"""
<div class="lede">
<h1><span class="eyebrow">MCP server</span>Analytics a model can call, with the
reasoning left in.</h1>
<p class="standfirst">{total} tools over option pricing, portfolio risk,
interest rate curves, execution cost and backtest validation &mdash; each one
reporting how the number was produced, and refusing rather than guessing when it
cannot be.</p>
</div>

<h2>The first minute</h2>
<p>The server and its five libraries are not on the package index yet, so today
it installs from source. Both halves of that are one command, and nothing has to
be resolved by name once the libraries are in place:</p>
<pre><code>{FROM_SOURCE}

abacus tools          # what it can do
abacus stdio          # serve over stdio
</code></pre>

<div class="note">
<p><strong>After the first release</strong> the whole thing becomes
<code>pip install abacus-mcp</code>, or <code>uvx abacus-mcp stdio</code> with no
install at all. This page will say so when that is true rather than before
&mdash; a documented command that does not work is worse than an undocumented
one, because it fails in the reader's terminal rather than on the page.</p>
</div>

<p>Registering it with a client means one entry naming that command. The
<a href="{SOURCE}#registering-it-with-a-client">README</a> has the exact shape
for the common ones.</p>

<h2>What makes it different from a calculator</h2>
<p>The numbers come from five libraries underneath rather than from a wrapper
around somebody else's API, which means the server can say <em>how</em> it got
an answer, and can decline to give one.</p>

<div class="note">
<p><strong>The method is named, because the number does not identify it.</strong>
A one-day 99% value at risk of 2.3% under a normal assumption and 3.1% from the
sample are the same quantity estimated two ways, and nothing about either figure
says which it is.</p>
</div>

<div class="note">
<p><strong>Nonsense is refused before it is priced.</strong> A price outside the
range attainable by any non-negative volatility has no implied volatility, an
option cannot expire in the past, and weights summing to 1.4 are a typo rather
than a portfolio. Each comes back as a result a model can repair, naming the
field and what would be accepted, rather than as a stack trace.</p>
</div>

<div class="note">
<p><strong>Claims are measured.</strong> Every specific figure in this
documentation is checked by a named test, and the
<a href="validation.html">validated numbers</a> page lists which.</p>
</div>

<h2>Where to go next</h2>
<p><a href="conventions.html">Conventions</a> first if you are about to call
something &mdash; they are the mistakes that produce a plausible number rather
than an error. <a href="tools.html">The tool reference</a> when you know what
you want. <a href="libraries.html">The libraries underneath</a> if you would
rather import than serve.</p>
"""


def conventions() -> str:
    return """
<div class="lede">
<h1><span class="eyebrow">Read this first</span>Conventions</h1>
<p class="standfirst">These are the mistakes that produce a plausible number
rather than an error. Every one of them has been made.</p>
</div>

<h2>Units</h2>
<h3>Rates and volatilities are decimal fractions</h3>
<p>20% is <code>0.2</code>, not <code>20</code>. So is a 20% return. A volatility
of <code>20</code> is not refused, because it describes a market that moves 2000%
a year &mdash; which is a market, so the server prices it.</p>

<h3>Time is a year fraction</h3>
<p>Thirty days is about <code>0.082</code>. One year is <code>1.0</code>. Nothing
here takes a count of days or a date for an option's life.</p>

<h3>Losses are positive</h3>
<p>A value at risk of <code>0.023</code> is a 2.3% loss. An expected shortfall is
always at least as large as the value at risk at the same confidence; if it comes
back smaller, the arguments are not what you think they are.</p>

<h3>A Sharpe ratio is per period</h3>
<p>The figure people quote is annualised. An annualised 1.0 on daily data is
about <code>0.063</code> per period, and everything in the validation group takes
the per-period one. Pass <code>periodsPerYear</code> to get both back.</p>

<h3><code>periodsPerYear</code> is required, not assumed</h3>
<p>252 for daily equity data, 52 for weekly, 12 for monthly. Annualising weekly
data with 252 overstates volatility by a factor of about 2.2, which does not look
wrong enough to notice.</p>

<h2>Weights</h2>
<p>Weights are not normalised for you. Weights summing to 0.98 are either a 2%
cash position or a typo, and the two want opposite treatment; normalising quietly
would turn the second into a plausible answer. The sum is reported on every
result that takes weights, and a sum far from one is refused with the total
named.</p>

<h2>Handles</h2>
<p>Three tools mint handles, and the three kinds are not interchangeable. Each
payload carries a <code>kind</code> tag, so presenting the wrong one is refused
rather than misread as an empty one of the right kind.</p>

<div class="scroller"><table class="wide">
<thead><tr><th>Minted by</th><th>Carries</th><th>Cannot be used for</th></tr></thead>
<tbody>
<tr>
<td class="figure">open_position_book</td>
<td>option and underlying legs, the spot, the rate</td>
<td>anything expecting a covariance estimate</td>
</tr>
<tr>
<td class="figure">estimate_return_moments</td>
<td>a mean vector and a covariance matrix &mdash; not the returns</td>
<td>the historical and drawdown tools, which read the path</td>
</tr>
<tr>
<td class="figure">bootstrap_discount_curve</td>
<td>the pillar times and their quotes</td>
<td>either of the above</td>
</tr>
</tbody>
</table></div>

<div class="note">
<p><strong>A handle caps at 8192 characters, which is why the moments handle
carries second moments rather than the matrix.</strong> A year of daily returns
on four assets encodes to roughly 6000 characters and five years on ten assets to
about 69,000, so a handle carrying the matrix would refuse almost every portfolio
worth asking about. A mean vector and a covariance matrix are <code>n + n²</code>
numbers however long the history is. A curve does fit &mdash; five pillars is 264
characters &mdash; so the curve handle carries its pillars and quotes.</p>
</div>

<p>What that buys: the parametric estimators run from a handle, so a matrix is
sent once and then reweighted and decomposed across as many calls as you like.
What it costs: the historical estimators and every drawdown statistic read the
return path, and a second-moment summary has thrown it away. Those tools require
the matrix and say so when handed a handle.</p>

<h2>Failures</h2>
<p>The specification draws a line between two kinds of failure and this server
draws it in the same place, because the two have different audiences.</p>
<p><strong>Protocol errors</strong> are JSON-RPC errors. They say the request was
not a thing the server could act on &mdash; an unknown tool, a malformed body
&mdash; and a model is unlikely to recover, because the fault is in the plumbing
rather than in the arguments.</p>
<p><strong>Execution errors</strong> come back as ordinary results with
<code>isError</code> set. They say the call arrived intact and the server
declined it, and they are addressed to the model: which field, what was wrong,
what would be accepted instead. A schema violation is one of these, not a
protocol error &mdash; a volatility given as <code>20</code> instead of
<code>0.2</code> is a fixable mistake, and saying so is more useful than refusing
the request.</p>

<h2>Absent numbers</h2>
<p>An absent number comes back as <code>null</code>, and means something
specific.</p>
<ul>
<li>A zero rate at a curve's own reference date, because the discount factor there
is one whatever the rate is, so no rate is implied.</li>
<li>The half-life of a risk-neutral execution schedule, because an equal slice
each period never decays and the half-life is infinite.</li>
<li>A degradation slope where the in-sample metric did not vary across
partitions, so there is no regression to report.</li>
</ul>
<p><code>Infinity</code> and <code>NaN</code> are never sent. Neither is JSON, and
a strict parser rejects the whole message over one of them &mdash; so a single
unbounded quantity would destroy every number beside it.</p>
"""


def tools_page() -> str:
    registry = default_registry()
    placed = {name for group in GROUPS for name in group.tools}
    every = {tool.name for tool in registry}
    missing = every - placed
    if missing:  # pragma: no cover - the test for this fails before the build does
        raise SystemExit(f"tools missing from a documented group: {sorted(missing)}")

    sections = []
    for group in GROUPS:
        entries = []
        for name in group.tools:
            tool = registry.get(name)
            assert tool is not None
            entries.append(
                f'<article class="tool" id="{esc(name)}">'
                f'<h3><a href="#{esc(name)}">{esc(name)}</a></h3>'
                f'<p class="title">{esc(tool.title or "")}</p>'
                f'<p class="description">{esc(tool.description)}</p>'
                f"{arguments_table(tool.input_schema)}"
                "</article>"
            )
        sections.append(
            f'<section class="group" id="{esc(group.slug)}">'
            f"<h2>{esc(group.name)}</h2>"
            f"<p>{esc(group.blurb)}</p>"
            f"{''.join(entries)}"
            "</section>"
        )

    return f"""
<div class="lede">
<h1><span class="eyebrow">{len(every)} tools</span>Tool reference</h1>
<p class="standfirst">Every entry below is rendered from the registry this
server serves, so a renamed argument changes this page on the next build. The
descriptions are the ones a client receives in <code>tools/list</code>.</p>
<p class="count">Read <a href="conventions.html">the conventions</a> first if you
are about to call one of these.</p>
</div>
{''.join(sections)}
"""


def validation() -> str:
    rows = "".join(
        "<tr>"
        f'<td class="figure">{esc(claim.figure)}</td>'
        f"<td>{esc(claim.about)}</td>"
        f'<td class="where">{esc(claim.module)}<br>{esc(claim.test)}</td>'
        "</tr>"
        for claim in CLAIMS
    )
    return f"""
<div class="lede">
<h1><span class="eyebrow">{len(CLAIMS)} claims</span>Validated numbers</h1>
<p class="standfirst">Every specific figure this documentation prints was
measured rather than guessed. This is where each one is held to account.</p>
</div>

<p>A claim with no check behind it drifts from the code eventually, and the
drift is silent: the number goes on reading as authoritative while the software
has moved. So each row names the test that measures its figure, and a test in
this repository asserts that every one of those tests exists and that the figure
appears in its source. A number that changes in the code and not on the page
fails a build.</p>

<div class="note">
<p><strong>What this does not catch</strong> is a new claim added to the
documentation and never added to the table. That is why the table renders here
rather than living only in the test suite: a figure that is not in this list is
visibly not in this list.</p>
</div>

<div class="scroller"><table class="wide">
<thead><tr><th>Figure</th><th>What it measures</th><th>Checked by</th></tr></thead>
<tbody>{rows}</tbody>
</table></div>

<h2>Claims that were wrong when measured</h2>
<p>The table above exists because several of these were. Each of the following
was written down as plausible, measured afterwards, and corrected.</p>
<ul>
<li>Permanent impact was said not to enter an optimal execution schedule. In
continuous time it does not; in discrete time it acts through
<code>eta - gamma·tau/2</code>, shortening the half-life by 2.5% at a period
length of 1.</li>
<li>The bias of uncorrected excess kurtosis on normal data is
<code>-6/(n+1)</code>, not the commonly quoted <code>-6/n</code> &mdash; four
standard errors apart at twenty observations.</li>
<li>A futures convexity adjustment was quoted at 5 basis points and measured at
51.</li>
<li>&ldquo;More trials never make the evidence stronger&rdquo; is true of the
deflation and false of the tool that applies it: adding columns changes which
trial is selected and what the trials' variance is, so the deflated figure
legitimately moves in either direction.</li>
</ul>
"""


def libraries() -> str:
    return """
<div class="lede">
<h1><span class="eyebrow">Underneath</span>The libraries</h1>
<p class="standfirst">The server is a protocol surface over five libraries, each
of which stands on its own. If you are writing Python rather than driving a
model, import them directly &mdash; the server adds validation, handles and a
wire format, and none of that is useful inside a process.</p>
</div>

<div class="scroller"><table class="wide">
<thead><tr><th>Install</th><th>Import</th><th>What it does</th></tr></thead>
<tbody>
<tr>
<td class="figure">moneyness</td>
<td class="figure">moneyness</td>
<td>Option pricing, the full Greek set, implied volatility, SVI surfaces, local volatility and
American exercise.</td>
</tr>
<tr>
<td class="figure">shortfall</td>
<td class="figure">shortfall</td>
<td>Shrinkage covariance, value at risk and expected shortfall by five methods, risk
contributions, risk parity and drawdown statistics.</td>
</tr>
<tr>
<td class="figure">tenor</td>
<td class="figure">tenor</td>
<td>Day counts, business-day conventions, curve bootstrapping, bond analytics, key rate
durations and option-adjusted spreads.</td>
</tr>
<tr>
<td class="figure">slippage-tca</td>
<td class="figure">slippage</td>
<td>Implementation shortfall, market impact fitting, Almgren-Chriss schedules, constrained
scheduling and volume curves.</td>
</tr>
<tr>
<td class="figure">holdout-backtest</td>
<td class="figure">holdout</td>
<td>Deflated Sharpe ratios, effective trial counts, backtest overfitting probability, purged
cross-validation and tests for superior predictive ability.</td>
</tr>
</tbody>
</table></div>

<div class="note">
<p><strong>Two of those install under a name that is not the name you
import.</strong> <code>slippage</code> and <code>holdout</code> on the package
index belong to unrelated projects that were there first, so these publish as
<code>slippage-tca</code> and <code>holdout-backtest</code>. Renaming the Python
packages to match would have broken every existing import to settle a registry
collision. The failure mode is quiet &mdash; <code>pip install slippage</code>
succeeds and hands you somebody else's library &mdash; which is why it is stated
here rather than left to be discovered.</p>
</div>

<h2>When to use the server instead</h2>
<p>The server earns its place when a language model is the caller. It adds
argument validation with messages addressed at a caller that can retry, handles
so state survives a stateless protocol, refusals that distinguish a fixable
mistake from a broken request, and a tool surface with a budget on its size. In
a Python process every one of those is overhead: you have exceptions, variables
and a type checker already.</p>

<h2>When to use the libraries instead</h2>
<p>Any time the calling code is yours. They have no dependency on this server,
no dependency on each other, and between them no dependency beyond NumPy &mdash;
<code>moneyness</code>, <code>shortfall</code> and <code>tenor</code> have none
at all. Each ships type information, a command line, and its own test suite
against published reference figures.</p>
"""


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    bodies = {
        "index.html": (overview(), "Overview", False),
        "conventions.html": (conventions(), "Conventions", False),
        "tools.html": (tools_page(), "Tool reference", True),
        "validation.html": (validation(), "Validated numbers", False),
        "libraries.html": (libraries(), "The libraries underneath", False),
    }
    for filename, (body, title, index) in bodies.items():
        (OUT / filename).write_text(
            page(filename=filename, title=title, body=body, tool_index=index)
        )

    shutil.copy(Path(__file__).parent / "style.css", OUT / "style.css")
    # Pages otherwise runs the output through Jekyll, which ignores anything
    # beginning with an underscore and rewrites what it does not ignore.
    (OUT / ".nojekyll").write_text("")

    written = sorted(path.name for path in OUT.iterdir())
    print(f"wrote {len(written)} files to {OUT}: {', '.join(written)}")


if __name__ == "__main__":
    build()
