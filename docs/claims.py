"""Every measured number the documentation claims, and the test that checks it.

The documentation on this project makes a lot of specific claims — a delay
reattribution of exactly 20 currency units, a half-life shift of 2.5% at one
period length, an effective trial count of 3.0 against a raw 40. Each of them was
measured rather than guessed, which is the whole reason they are worth printing,
and each of them will drift from the code eventually unless something holds the
two together.

That is what this table is. It is not documentation of the tests; it is the
index the documentation is checked against. ``tests/test_docs.py`` asserts that
every test named here exists and every claim's figure appears in the source of
the test that is said to check it, so a number that moves in the code and not on
the page fails a build rather than quietly misleading a reader.

Adding a claim to the site without adding it here is the failure mode this
cannot catch by itself, so the site's own build reads this table and renders it:
a claim that is not in it is not on the validation page either, which is at
least visible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    """A number the documentation prints, and where it is held to account."""

    #: The figure as the documentation writes it. This exact string has to occur
    #: in the checking test's source, which is what stops the two drifting.
    figure: str
    #: What it is a measurement of, in one line.
    about: str
    #: Test file, relative to the repository root.
    module: str
    #: Test function inside that file.
    test: str


CLAIMS: tuple[Claim, ...] = (
    Claim(
        figure="869",
        about=(
            "The total implementation shortfall of the worked order, which is the "
            "same under either delay basis."
        ),
        module="tests/test_execution.py",
        test="test_the_delay_basis_reattributes_cost_without_changing_the_total",
    ),
    Claim(
        figure="200.0",
        about="Delay cost of that order charged on the quantity ordered.",
        module="tests/test_execution.py",
        test="test_the_delay_basis_reattributes_cost_without_changing_the_total",
    ),
    Claim(
        figure="180.0",
        about=(
            "Delay cost of the same order charged on the quantity executed — the "
            "20 units that move into opportunity cost."
        ),
        module="tests/test_execution.py",
        test="test_the_delay_basis_reattributes_cost_without_changing_the_total",
    ),
    Claim(
        figure="0.0246",
        about=(
            "Fraction by which permanent impact shortens the execution half-life "
            "at a period length of 1."
        ),
        module="tests/test_execution.py",
        test="test_permanent_impact_leaves_the_schedule_alone_only_in_the_limit",
    ),
    Claim(
        figure="0.00025",
        about=(
            "The same fraction at a period length of 0.01, which is the O(tau) "
            "behaviour the algebra predicts."
        ),
        module="tests/test_execution.py",
        test="test_permanent_impact_leaves_the_schedule_alone_only_in_the_limit",
    ),
    Claim(
        figure="0.0625",
        about=(
            "Fixed cost per share that raises the expected cost by exactly itself "
            "times the quantity, leaving the schedule unmoved."
        ),
        module="tests/test_execution.py",
        test="test_a_fixed_cost_moves_the_cost_by_itself_and_leaves_the_schedule_alone",
    ),
    Claim(
        figure="200_000.0",
        about=(
            "Each slice of the risk-neutral schedule for a million shares over "
            "five periods: TWAP, exactly."
        ),
        module="tests/test_execution.py",
        test="test_risk_neutrality_is_exactly_a_straight_line",
    ),
    Claim(
        figure="3.0",
        about=(
            "Effective number of trials among forty driven by one common factor, "
            "by the eigenvalue method."
        ),
        module="tests/test_validation.py",
        test="test_one_factor_trials_collapse_to_a_handful",
    ),
    Claim(
        figure="40.0",
        about=(
            "Effective number of trials among forty independent ones: the raw "
            "count, which is what makes the correction meaningful."
        ),
        module="tests/test_validation.py",
        test="test_independent_trials_are_worth_their_raw_count",
    ),
    Claim(
        figure="0.8244",
        about="Deflated Sharpe ratio of those correlated trials on the effective count.",
        module="tests/test_validation.py",
        test="test_the_effective_count_matters_exactly_where_trials_are_correlated",
    ),
    Claim(
        figure="0.7196",
        about=(
            "The same figure deflated against the raw count of forty — ten points "
            "of probability the raw count throws away."
        ),
        module="tests/test_validation.py",
        test="test_the_effective_count_matters_exactly_where_trials_are_correlated",
    ),
    Claim(
        figure="2.71",
        about=(
            "Years of daily data before an annualised Sharpe ratio of 1.0 is "
            "distinguishable from zero at 95% confidence."
        ),
        module="tests/test_validation.py",
        test="test_a_bigger_sharpe_needs_a_shorter_record",
    ),
    Claim(
        figure="87.124248",
        about="Total cost of the worked order in basis points of its paper notional.",
        module="tests/test_execution_over_the_wire.py",
        test="test_the_decomposition_survives_the_wire",
    ),
    Claim(
        figure="1.64724",
        about=(
            "Half-life of the worked execution schedule at a risk aversion of "
            "1e-6, in periods."
        ),
        module="tests/test_execution_over_the_wire.py",
        test="test_the_schedule_survives_the_wire",
    ),
    Claim(
        figure="140_000",
        about="Budget on the whole tool listing, in characters of serialised JSON.",
        module="tests/test_skill.py",
        test="test_the_tool_listing_fits_its_budget",
    ),
    Claim(
        figure="12_000",
        about="Budget on any single tool, in characters of serialised JSON.",
        module="tests/test_skill.py",
        test="test_no_single_tool_takes_more_than_its_share",
    ),
    Claim(
        figure="279.0",
        about=(
            "Holding-period return of a 3% 2031 bond over one year on the bundled "
            "quote screen, in basis points. Quoted in the bond_carry_rolldown "
            "description, which is asserted in the same test."
        ),
        module="tests/test_model_validation.py",
        test="test_the_figure_the_description_quotes_is_the_one_it_computes",
    ),
    Claim(
        figure="2.49",
        about=(
            "Roll-down inside that 279 basis points, per 100 of face. The "
            "financing cost is the other 0.50, so nearly all of the return is "
            "the curve failing to evolve to its forwards."
        ),
        module="tests/test_model_validation.py",
        test="test_the_figure_the_description_quotes_is_the_one_it_computes",
    ),
    Claim(
        figure="-0.245",
        about=(
            "The Acerbi-Szekely conditional statistic when the true volatility is "
            "double the forecast. Quoted in validate_risk_model's note to warn "
            "that the statistic is not on the scale of the error it detects."
        ),
        module="tests/test_model_validation.py",
        test="test_the_note_gives_the_measured_scale_of_the_shortfall_statistic",
    ),
    Claim(
        figure="10.00% of the time",
        about=(
            "How often a breach follows a breach under a regime-switching "
            "volatility, against 1.84% after a calm day — the clustering a "
            "breach count cannot see."
        ),
        module="tests/test_model_validation.py",
        test="test_clustered_breaches_are_caught_by_independence_not_by_the_count",
    ),
    Claim(
        figure="28.2 to 22.45",
        about=(
            "The 99% breach count over 2000 observations of a regime-switching "
            "series, under normal innovations and under an estimated tail, against "
            "a nominal 20. Quoted in conditional_volatility's note to say what "
            "estimating the tail buys and what it leaves behind."
        ),
        module="tests/test_model_validation.py",
        test="test_the_note_refuses_the_flattering_summary",
    ),
    Claim(
        figure="41% larger at four degrees of freedom",
        about=(
            "How much wider a forecast becomes if the raw Student-t quantile is "
            "used where the standardised one belongs. The reason "
            "conditional_volatility returns the multiplier rather than describing "
            "how to build it."
        ),
        module="tests/test_model_validation.py",
        test="test_the_quantile_multiplier_is_the_standardised_quantile",
    ),
)
