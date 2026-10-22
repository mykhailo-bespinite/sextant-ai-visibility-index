from decimal import Decimal

import pytest

from aeo_index import pricing
from aeo_index.modes import MAX_SEARCH_TOOL_CALLS
from aeo_index.pricing import (
    JUDGE_HIGH_INPUT_TOKENS,
    JUDGE_OUTPUT_TOKENS,
    REQUEST_FEES_PER_1K,
    SEARCH_FEES_PER_1K,
    TYPICAL_AS_OF,
    TYPICAL_CALL_USD,
    RateCard,
    call_cost,
    estimate,
    request_fee_per_1k,
    unpriced,
)
from aeo_index.providers.base import DEFAULT_MODELS, ProviderReply, output_cap
from aeo_index.records import JudgeSpec
from vendor_doubles import cells

JUDGE = JudgeSpec.parse("anthropic:claude-haiku-4-5-20251001")


def reply(**overrides) -> ProviderReply:
    values = dict(
        text="x",
        model_version="m",
        input_tokens=1000,
        output_tokens=1000,
        citations=None,
        search_count=None,
        search_queries=None,
        vendor_cost_usd=None,
    )
    values.update(overrides)
    return ProviderReply(**values)


def test_fees_are_the_published_ones():
    assert SEARCH_FEES_PER_1K == {
        "openai": Decimal("10.00"),
        "anthropic": Decimal("10.00"),
        "gemini": Decimal("14.00"),
        "grok": Decimal("5.00"),
    }
    assert REQUEST_FEES_PER_1K == {
        ("perplexity", "sonar-pro"): Decimal("6.00"),
        ("perplexity", "sonar"): Decimal("5.00"),
    }
    assert (JUDGE_HIGH_INPUT_TOKENS, JUDGE_OUTPUT_TOKENS) == (4000, 500)


def test_default_rates_cover_every_default_model_and_the_judges():
    card = RateCard.default()
    for engine, model in DEFAULT_MODELS.items():
        assert card.rate(engine, model) is not None, engine
    assert card.rate("anthropic", "claude-haiku-4-5") == (Decimal("0.001"), Decimal("0.005"))
    assert card.rate("openai", "gpt-4o-mini") == (Decimal("0.00015"), Decimal("0.0006"))
    assert card.rate("openai", "gpt-5.5") == (Decimal("0.005"), Decimal("0.030"))
    assert card.rate("gemini", "models/gemini-3.5-flash") == (Decimal("0.0015"), Decimal("0.009"))


def test_dated_models_resolve_to_the_longest_base_key():
    card = RateCard.default()
    assert card.rate("openai", "gpt-5.5-2026-04-23") == card.rate("openai", "gpt-5.5")
    assert card.rate("anthropic", "claude-haiku-4-5-20251001") == card.rate("anthropic", "claude-haiku-4-5")
    # "gpt-4o-mini" must never price as a different "gpt-4o" prefix, and a
    # longer name is not a dated snapshot of a shorter one.
    assert card.rate("openai", "gpt-5.55") is None
    assert card.rate("openai", "gpt-9") is None
    assert card.rate("grok", "gpt-5.5") is None


def test_only_a_dated_snapshot_prices_as_its_base_model():
    card = RateCard.default()
    assert card.rate("openai", "gpt-5.5-2026-04-23") == card.rate("openai", "gpt-5.5")
    assert card.rate("anthropic", "claude-sonnet-4-6-20260101") == card.rate("anthropic", "claude-sonnet-4-6")
    # A variant is another model at another price: it needs --price.
    for engine, model in [
        ("openai", "gpt-5.5-pro"),
        ("grok", "grok-4.3-heavy"),
        ("anthropic", "claude-sonnet-4-6-latest"),
        ("perplexity", "sonar-pro-deep-research"),
        ("openai", "gpt-5.5-2026-04"),
        ("openai", "gpt-5.5-2026-13-45x"),
        ("openai", "gpt-5.5-\u0662\u0660\u0662\u0666\u0660\u0661\u0660\u0661"),  # Arabic-Indic digits
    ]:
        assert card.rate(engine, model) is None, model
    assert request_fee_per_1k("perplexity", "sonar-pro-deep-research") == Decimal("0")


def test_with_override_returns_a_new_card():
    card = RateCard.default()
    priced = card.with_override("openai", "gpt-9", Decimal("0.01"), Decimal("0.02"))
    assert priced.rate("openai", "gpt-9") == (Decimal("0.01"), Decimal("0.02"))
    assert card.rate("openai", "gpt-9") is None
    overridden = card.with_override("openai", "gpt-5.5", Decimal("1"), Decimal("2"))
    assert overridden.rate("openai", "gpt-5.5-2026-04-23") == (Decimal("1"), Decimal("2"))
    with pytest.raises(ValueError):
        card.with_override("openai", "gpt-9", Decimal("-1"), Decimal("1"))


def test_call_cost_tokens_and_search_fee():
    card = RateCard.default()
    cost = call_cost("openai", "gpt-5.5", "search", reply(search_count=2), card)
    assert cost == Decimal("0.005") + Decimal("0.030") + Decimal("0.020")
    # The search fee is billed in search mode only.
    assert call_cost("openai", "gpt-5.5", "api", reply(search_count=2), card) == Decimal("0.035000")


def test_call_cost_adds_the_request_fee_and_prefers_the_vendor_cost():
    card = RateCard.default()
    assert call_cost("perplexity", "sonar-pro", "search", reply(), card) == Decimal("0.003") + Decimal(
        "0.015"
    ) + Decimal("0.006")
    assert call_cost("grok", "grok-4.3", "search", reply(vendor_cost_usd=Decimal("0.0312504")), card) == Decimal(
        "0.031250"
    )


def test_call_cost_of_an_unpriced_model_raises():
    with pytest.raises(ValueError):
        call_cost("openai", "gpt-9", "api", reply(), RateCard.default())


def test_typical_figures_cover_every_engine_and_mode():
    expected = {
        (engine, mode)
        for engine in DEFAULT_MODELS
        for mode in (("search",) if engine in ("perplexity", "google_aio") else ("api", "search"))
    }
    assert set(TYPICAL_CALL_USD) == expected
    for median, p95 in TYPICAL_CALL_USD.values():
        assert Decimal(0) < median <= p95
    assert TYPICAL_AS_OF


def test_estimate_of_default_models_uses_the_typical_figures():
    planned = cells(engines=("anthropic", "openai"), questions=10, samples=1, mode="search")
    est = estimate(planned, DEFAULT_MODELS, JUDGE, RateCard.default())
    rows = {(engine, mode): (n, typical, high) for engine, mode, n, typical, high in est.rows}
    median, p95 = TYPICAL_CALL_USD[("openai", "search")]
    assert rows[("openai", "search")] == (10, median * 10, p95 * 10)
    assert [r[:2] for r in est.rows] == [("anthropic", "search"), ("openai", "search")]
    judge_per_cell = Decimal(4000) * Decimal("0.001") / 1000 + Decimal(500) * Decimal("0.005") / 1000
    assert est.judge_high == judge_per_cell * 20
    assert est.typical == sum(r[3] for r in est.rows) + est.judge_high
    assert est.high == sum(r[4] for r in est.rows) + est.judge_high


def test_estimate_of_another_model_uses_its_output_cap_both_ways():
    card = RateCard.default().with_override("openai", "gpt-9", Decimal("0.01"), Decimal("0.02"))
    planned = cells(engines=("openai",), questions=2, samples=1, mode="search")
    models = {**DEFAULT_MODELS, "openai": "gpt-9"}
    est = estimate(planned, models, JUDGE, card)
    (row,) = est.rows
    per_call = (
        Decimal(output_cap("openai", "gpt-9", "search")) * Decimal("0.02") / 1000
        + Decimal(2000) * Decimal("0.01") / 1000
        + MAX_SEARCH_TOOL_CALLS * Decimal("10.00") / 1000
    )
    assert row == ("openai", "search", 2, per_call * 2, per_call * 2)


def test_unpriced_lists_models_and_judge_without_a_rate():
    card = RateCard.default()
    assert unpriced(DEFAULT_MODELS, JUDGE, card) == []
    assert unpriced({"openai": "gpt-9"}, JudgeSpec("gemini", "gemini-9"), card) == [
        "openai:gpt-9",
        "gemini:gemini-9",
    ]


def test_module_has_no_unpinned_floats():
    assert all(isinstance(v, Decimal) for v in pricing.SEARCH_FEES_PER_1K.values())


MEASURED = {
    ("anthropic", "api"),
    ("anthropic", "search"),
    ("gemini", "api"),
    ("gemini", "search"),
    ("grok", "api"),
    ("grok", "search"),
    ("openai", "api"),
    ("openai", "search"),
    ("perplexity", "search"),
}


def _default_run_estimate():
    from aeo_index.dispatch import plan_cells
    from aeo_index.modes import DEFAULT_SAMPLES, ENGINES
    from aeo_index.preflight import DEFAULT_JUDGE
    from aeo_index.questions import default_questions

    planned = plan_cells(default_questions(), ENGINES, "search", {}, DEFAULT_SAMPLES)
    return estimate(planned, DEFAULT_MODELS, JudgeSpec.parse(DEFAULT_JUDGE), RateCard.default())


def test_measured_figures_are_pinned_and_the_default_run_high_is_known():
    assert TYPICAL_AS_OF == "2026-10-21"
    for key in MEASURED:
        median, p95 = TYPICAL_CALL_USD[key]
        assert median <= p95 and median.as_tuple().exponent >= -4 and p95.as_tuple().exponent >= -4
    # The spec's default run (ten questions, every engine, search, three
    # samples): pinned so a change to the figures or the plan shows here.
    est = _default_run_estimate()
    assert len(est.rows) == 6
    assert est.high == Decimal("16.2720")
    assert TYPICAL_CALL_USD[("google_aio", "search")] == (Decimal("0.0012"), Decimal("0.0012"))


def test_the_default_run_fits_the_default_budget():
    from aeo_index.preflight import DEFAULT_BUDGET_USD

    assert _default_run_estimate().high <= DEFAULT_BUDGET_USD


def test_a_non_default_results_page_model_is_priced_at_the_ceiling():
    from aeo_index.providers.google_aio import FALLBACK_COST_USD

    card = RateCard.default().with_override("google_aio", "serp-v2", Decimal("0"), Decimal("0"))
    planned = cells(engines=("google_aio",), questions=1, samples=1, mode="search")
    (row,) = estimate(planned, {"google_aio": "serp-v2"}, JUDGE, card).rows
    assert row[3] == row[4] == FALLBACK_COST_USD
