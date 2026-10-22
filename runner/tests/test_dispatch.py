import json
from decimal import Decimal

import httpx
import pytest

from aeo_index.dispatch import (
    MAX_IN_FLIGHT,
    MAX_PER_ENGINE,
    MAX_RETRY_AFTER,
    RETRY_DELAYS,
    Budget,
    Cell,
    plan_cells,
)
from aeo_index.fixtures import fixture_response, load_vendor_fixture
from aeo_index.modes import CALL_TIMEOUT_SECONDS, MAX_ANSWER_CITATIONS
from aeo_index.questions import default_questions
from vendor_doubles import (
    FAKE_ENV,
    Barrier,
    ConcurrencyCounter,
    Scripted,
    cells,
    recording_sleep,
    run_cells,
)


def test_limits_are_the_methods():
    assert (MAX_IN_FLIGHT, MAX_PER_ENGINE) == (8, 4)
    assert RETRY_DELAYS == (1.0, 4.0) and MAX_RETRY_AFTER == 60.0


def test_plan_cells_orders_question_engine_mode_sample():
    qset = default_questions()
    planned = plan_cells(qset, ["openai", "anthropic", "perplexity"], "both", {"openai": "api"}, 2)
    first = [(c.key, c.engine, c.mode, c.sample) for c in planned[:7]]
    assert first == [
        ("bhag_01", "anthropic", "api", 1),
        ("bhag_01", "anthropic", "api", 2),
        ("bhag_01", "anthropic", "search", 1),
        ("bhag_01", "anthropic", "search", 2),
        ("bhag_01", "openai", "api", 1),
        ("bhag_01", "openai", "api", 2),
        ("bhag_01", "perplexity", "search", 1),
    ]
    assert len(planned) == 10 * (2 + 1 + 1) * 2
    assert planned[0].text == qset.questions[0].text


def test_plan_cells_refuses_bad_samples_and_engines():
    qset = default_questions()
    with pytest.raises(ValueError):
        plan_cells(qset, ["openai"], "search", {}, 0)
    with pytest.raises(ValueError):
        plan_cells(qset, ["openai"], "search", {}, 6)
    with pytest.raises(ValueError):
        plan_cells(qset, ["bing"], "search", {}, 1)


def test_budget_arithmetic():
    budget = Budget(Decimal("5.00"), reserved=Decimal("1.00"))
    budget.charge(Decimal("0.50"))
    assert budget.available == Decimal("3.50") and budget.spent == Decimal("0.50")
    budget.release(Decimal("0.25"))
    assert budget.reserved == Decimal("0.75") and budget.available == Decimal("3.75")
    with pytest.raises(ValueError):
        budget.charge(Decimal("-1"))
    with pytest.raises(ValueError):
        budget.release(Decimal("2"))


async def test_success_answers_come_back_in_cell_order_with_cost():
    planned = cells(engines=("anthropic", "openai"), questions=2, samples=2)
    answers = await run_cells(planned, Scripted({"openai": "openai_api_ok.json", "anthropic": "anthropic_api_ok.json"}))
    assert [(a.key, a.engine, a.mode, a.sample) for a in answers] == [
        (c.key, c.engine, c.mode, c.sample) for c in planned
    ]
    claude = answers[0]
    assert claude.status == "success" and claude.text and claude.cost_usd == Decimal("0.007266")
    assert claude.model_version == "claude-sonnet-4-6" and claude.error is None and claude.error_type is None


async def test_auth_failure_stops_engine_and_names_env_var(capsys):
    transport = Scripted({"openai": ["openai_401.json"], "anthropic": "anthropic_api_ok.json"})
    answers = await run_cells(cells(engines=("anthropic", "openai"), questions=3, samples=3), transport)
    openai = [a for a in answers if a.engine == "openai"]
    assert transport.calls["openai"] == 1
    assert all(a.status == "error" and a.error_type == "auth" for a in openai) and len(openai) == 9
    assert all(a.cost_usd == 0 for a in openai)
    assert all(a.status == "success" for a in answers if a.engine == "anthropic")
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY was rejected (401); ChatGPT is skipped, the other engines continue." in err
    assert err.count("was rejected") == 1 and "k-test-123456" not in err


async def test_engine_outage_rows_are_kept():
    answers = await run_cells(
        cells(engines=("perplexity",), questions=2, samples=3), Scripted({"perplexity": "perplexity_503.json"})
    )
    assert (
        len(answers) == 6 and {a.status for a in answers} == {"error"} and {a.error_type for a in answers} == {"server"}
    )


async def test_5xx_is_retried_twice_with_backoff_then_recorded():
    sleeps: list[float] = []
    transport = Scripted({"perplexity": "perplexity_503.json"})
    answers = await run_cells(
        cells(engines=("perplexity",), questions=1, samples=1), transport, sleep=recording_sleep(sleeps)
    )
    assert transport.calls["perplexity"] == 3 and answers[0].status == "error"
    assert len(sleeps) == 2
    assert 1.0 <= sleeps[0] <= 1.25 and 4.0 <= sleeps[1] <= 5.0


async def test_a_retry_that_succeeds_is_a_success():
    transport = Scripted({"grok": ["grok_503.json", "grok_api_ok.json"]})
    answers = await run_cells(cells(engines=("grok",), questions=1, samples=1), transport)
    assert answers[0].status == "success" and transport.calls["grok"] == 2


async def test_429_retried_with_retry_after_then_rate_limited():
    sleeps: list[float] = []
    answers = await run_cells(
        cells(engines=("grok",), questions=1, samples=1),
        Scripted({"grok": "grok_429_retry_after_90.json"}),
        sleep=recording_sleep(sleeps),
    )
    assert sleeps == [60.0, 60.0] and answers[0].status == "rate_limited"
    assert answers[0].error_type == "rate_limited"


async def test_in_flight_limits_hold():
    t = ConcurrencyCounter(delay=0.01)
    answers = await run_cells(cells(engines=("anthropic", "openai", "gemini"), questions=10, samples=3), t)
    assert t.max_total <= 8 and max(t.max_per_engine.values()) <= 4
    assert all(a.status == "success" for a in answers)
    # The limits are reached, not just respected.
    assert t.max_total == 8


async def test_budget_reached_records_skipped_cells_without_calls():
    # Each Claude answer costs 0.007266; the cap is under one call.
    budget = Budget(Decimal("0.005"))
    transport = Scripted({"anthropic": "anthropic_api_ok.json"})
    answers = await run_cells(cells(engines=("anthropic",), questions=4, samples=3), transport, budget=budget)
    assert transport.calls["anthropic"] == 1
    assert answers[0].status == "success"
    assert [a.status for a in answers[1:]] == ["skipped"] * 11
    assert all(a.cost_usd == 0 and a.error_type == "budget" for a in answers[1:])
    assert budget.skipped == 11 and budget.spent == Decimal("0.007266")


async def test_calls_in_flight_finish_when_the_cap_is_reached():
    # The engine's first answer is a probe and goes alone (0.007266 of a
    # 0.010 cap). The next four leave together and are held until all four
    # are out, so all four finish; the cap then stops the other seven.
    transport = Barrier("anthropic_api_ok.json", hold_until=4)
    budget = Budget(Decimal("0.010"))
    answers = await run_cells(cells(engines=("anthropic",), questions=4, samples=3), transport, budget=budget)
    statuses = [a.status for a in answers]
    assert transport.calls == 5 and statuses.count("success") == 5 and statuses.count("skipped") == 7


async def test_the_judges_reserve_is_not_spent_by_dispatch():
    budget = Budget(Decimal("0.01"), reserved=Decimal("0.009"))
    transport = Scripted({"anthropic": "anthropic_api_ok.json"})
    answers = await run_cells(cells(engines=("anthropic",), questions=1, samples=2), transport, budget=budget)
    assert [a.status for a in answers] == ["success", "skipped"]


async def test_timeout_is_a_timeout_row_with_zero_cost():
    now = [0.0]

    def clock() -> float:
        return now[0]

    def advance() -> None:
        now[0] += CALL_TIMEOUT_SECONDS["api"] + 1

    transport = Scripted({"anthropic": "anthropic_api_ok.json"}, on_request=advance)
    answers = await run_cells(cells(engines=("anthropic",), questions=1, samples=1), transport, clock=clock)
    (answer,) = answers
    assert answer.status == "timeout" and answer.error_type == "timeout" and answer.cost_usd == 0
    assert answer.text is None and answer.latency_ms == CALL_TIMEOUT_SECONDS["api"] * 1000


async def test_a_transport_timeout_is_a_timeout_row():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    answers = await run_cells(cells(engines=("openai",), questions=1, samples=1), httpx.MockTransport(handler))
    assert answers[0].status == "timeout" and answers[0].cost_usd == 0


async def test_empty_answer_is_empty_and_billed():
    answers = await run_cells(
        cells(engines=("openai",), questions=1, samples=1), Scripted({"openai": "openai_api_empty.json"})
    )
    assert answers[0].status == "empty" and answers[0].cost_usd == Decimal("0.245880")


async def test_secrets_never_reach_answers():
    key = FAKE_ENV["OPENAI_API_KEY"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"Incorrect API key provided: {key}. Bearer {key}"}})

    answers = await run_cells(cells(engines=("openai",), questions=1, samples=2), httpx.MockTransport(handler))
    for answer in answers:
        assert answer.error is not None and key not in answer.error
    assert "[redacted]" in (answers[0].error or "")


async def test_citations_bounded_to_ten_in_vendor_order():
    urls = [f"https://example.com/source-{n:02d}" for n in range(14)]
    fixture = load_vendor_fixture("perplexity_search_ok.json")
    fixture["body"]["citations"] = urls
    fixture["body"]["search_results"] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return fixture_response(fixture)

    answers = await run_cells(cells(engines=("perplexity",), questions=1, samples=1), httpx.MockTransport(handler))
    assert answers[0].citations == tuple(urls[:MAX_ANSWER_CITATIONS])


async def test_search_queries_are_bounded():
    fixture = load_vendor_fixture("gemini_search_ok.json")
    queries = ["q" * 600] + [f"query {n}" for n in range(30)]
    fixture["body"]["candidates"][0]["groundingMetadata"]["webSearchQueries"] = queries

    def handler(request: httpx.Request) -> httpx.Response:
        return fixture_response(fixture)

    answers = await run_cells(
        cells(engines=("gemini",), questions=1, samples=1, mode="search"), httpx.MockTransport(handler)
    )
    stored = answers[0].search_queries
    assert stored is not None and len(stored) == 20 and len(stored[0]) == 500
    # The search fee is billed on every query the vendor ran, not the stored ones.
    assert answers[0].search_count == 31


async def test_an_unexpected_exception_is_an_error_row():
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("boom")

    answers = await run_cells(cells(engines=("anthropic",), questions=1, samples=1), httpx.MockTransport(handler))
    assert answers[0].status == "error" and answers[0].error_type == "bad_response"


async def test_progress_reports_each_engine():
    seen = []
    await run_cells(
        cells(engines=("anthropic", "openai"), questions=2, samples=1),
        Scripted({"openai": "openai_api_ok.json", "anthropic": "anthropic_api_ok.json"}),
        on_progress=seen.append,
    )
    last = {p.engine: p for p in seen}
    assert set(last) == {"anthropic", "openai"}
    assert last["openai"].asked == 2 and last["openai"].answered == 2 and last["openai"].failed == 0
    assert last["openai"].spent == Decimal("0.038640")


async def test_a_cell_for_an_engine_without_a_provider_is_an_error_row():
    from aeo_index.dispatch import ask_all
    from aeo_index.pricing import RateCard
    from aeo_index.redact import Redactor

    cell = Cell("bhag_01", "Which tool?", "grok", "api", 1)
    async with httpx.AsyncClient(transport=Scripted({})) as client:
        (answer,) = await ask_all(
            [cell], {}, budget=Budget(Decimal("1")), card=RateCard.default(), redactor=Redactor([]), client=client
        )
    assert answer.status == "error" and answer.error_type == "client"


def test_cell_is_frozen():
    cell = Cell("bhag_01", "x", "openai", "api", 1)
    with pytest.raises(AttributeError):
        cell.sample = 2  # type: ignore[misc]
    assert json.dumps(cell.__dict__)


def _aio_task_failure_then(name: str):
    """A DataForSEO task that fails with 40101 (billed 0.002, retried), then `name`."""
    failed = load_vendor_fixture("google_aio_search_ok.json")
    failed["body"]["tasks"][0]["status_code"] = 40101
    failed["body"]["tasks"][0]["status_message"] = "Internal SE Server Error."
    replies = [failed, load_vendor_fixture(name)]

    def handler(request: httpx.Request) -> httpx.Response:
        return fixture_response(replies.pop(0) if len(replies) > 1 else replies[0])

    return httpx.MockTransport(handler)


async def test_a_retried_calls_billed_attempts_are_in_the_row():
    budget = Budget(Decimal("1"))
    answers = await run_cells(
        cells(engines=("google_aio",), questions=1, samples=1),
        _aio_task_failure_then("google_aio_search_ok.json"),
        budget=budget,
    )
    (answer,) = answers
    assert answer.status == "success" and answer.cost_usd == Decimal("0.004000")
    assert answer.cost_usd == budget.spent


async def test_a_403_also_stops_the_engine(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "This key may not use this model."}})

    transport = httpx.MockTransport(handler)
    answers = await run_cells(cells(engines=("anthropic",), questions=2, samples=2), transport)
    assert {a.error_type for a in answers} == {"auth"}
    assert "ANTHROPIC_API_KEY was rejected (403)" in capsys.readouterr().err


async def test_a_401_after_the_probe_stops_the_engine_once(capsys):
    transport = Scripted({"openai": ["openai_api_ok.json", "openai_401.json"]})
    answers = await run_cells(cells(engines=("openai",), questions=4, samples=3), transport)
    assert answers[0].status == "success"
    assert all(a.status == "error" and a.error_type == "auth" for a in answers[1:])
    # Only calls already in flight when the first 401 came back were made.
    assert transport.calls["openai"] <= 1 + MAX_PER_ENGINE
    assert capsys.readouterr().err.count("OPENAI_API_KEY was rejected (401)") == 1


async def test_a_timeout_row_costs_nothing_but_the_cap_counts_a_ceiling():
    now = [0.0]

    def advance() -> None:
        now[0] += CALL_TIMEOUT_SECONDS["api"] + 1

    budget = Budget(Decimal("1"))
    transport = Scripted({"anthropic": "anthropic_api_ok.json"}, on_request=advance)
    (answer,) = await run_cells(
        cells(engines=("anthropic",), questions=1, samples=1), transport, budget=budget, clock=lambda: now[0]
    )
    assert answer.status == "timeout" and answer.cost_usd == 0
    from aeo_index.pricing import RateCard, ceiling_call_usd

    assert budget.spent == ceiling_call_usd("anthropic", "claude-sonnet-4-6", "api", RateCard.default())


async def test_progress_is_reported_at_start_and_every_interval_while_waiting():
    import asyncio

    from aeo_index.dispatch import PROGRESS_INTERVAL_S

    now = [0.0]
    reports = []

    async def fake_sleep(seconds: float) -> None:
        now[0] += seconds
        await asyncio.sleep(0)

    async def handler(request: httpx.Request) -> httpx.Response:
        while len(reports) < 3:
            await asyncio.sleep(0)
        return fixture_response(load_vendor_fixture("anthropic_api_ok.json"))

    await run_cells(
        cells(engines=("anthropic",), questions=1, samples=1),
        httpx.MockTransport(handler),
        sleep=fake_sleep,
        clock=lambda: now[0],
        on_progress=reports.append,
    )
    count = len(reports)
    await asyncio.sleep(0)
    assert len(reports) == count  # the ticker stopped with the run
    assert [r.elapsed_s for r in reports[:3]] == [0.0, PROGRESS_INTERVAL_S, 2 * PROGRESS_INTERVAL_S]
    assert reports[0].asked == 0 and reports[1].asked == 1 and reports[1].answered == 0
    assert reports[-1].answered == 1 and reports[-1].spent == Decimal("0.007266")


async def test_a_progress_callback_that_raises_does_not_stop_the_run(capsys):
    def explode(progress) -> None:
        raise RuntimeError("display broke")

    answers = await run_cells(
        cells(engines=("anthropic",), questions=1, samples=2),
        Scripted({"anthropic": "anthropic_api_ok.json"}),
        on_progress=explode,
    )
    assert [a.status for a in answers] == ["success", "success"]
    assert capsys.readouterr().err.count("progress") == 1


def test_plan_cells_takes_the_briefs_keyword_names():
    planned = plan_cells(
        questions=default_questions(), engines=["openai"], mode="api", engine_modes={"openai": "search"}, samples=1
    )
    assert {c.mode for c in planned} == {"search"}


async def test_a_ticker_that_fails_keeps_the_answers_and_reports_redacted(capsys):
    import asyncio

    from aeo_index.dispatch import PROGRESS_INTERVAL_S

    secret = FAKE_ENV["ANTHROPIC_API_KEY"]

    async def broken_sleep(seconds: float) -> None:
        if seconds == PROGRESS_INTERVAL_S:
            raise RuntimeError(f"clock broke near {secret}")

    async def handler(request: httpx.Request) -> httpx.Response:
        for _ in range(5):  # let the ticker reach its first sleep and fail
            await asyncio.sleep(0)
        return fixture_response(load_vendor_fixture("anthropic_api_ok.json"))

    answers = await run_cells(
        cells(engines=("anthropic",), questions=1, samples=2),
        httpx.MockTransport(handler),
        sleep=broken_sleep,
        on_progress=lambda progress: None,
    )
    assert [a.status for a in answers] == ["success", "success"]
    err = capsys.readouterr().err
    assert "progress reports stopped" in err and "RuntimeError" in err
    assert secret not in err and "[redacted]" in err


async def test_a_reply_that_cannot_be_read_is_charged_its_ceiling_and_its_row_records_zero():
    from aeo_index.pricing import RateCard, ceiling_call_usd
    from aeo_index.providers.base import DEFAULT_MODELS

    def corrupt(request):
        return httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=b"not gzip")

    budget = Budget(Decimal("100"))
    [row] = await run_cells(
        cells(engines=["anthropic"], questions=1, samples=1), httpx.MockTransport(corrupt), budget=budget
    )
    assert row.status == "error" and row.error_type == "bad_response" and row.cost_usd == Decimal("0")
    assert budget.spent == ceiling_call_usd("anthropic", DEFAULT_MODELS["anthropic"], "api", RateCard.default())


async def test_an_unexpected_error_reading_a_reply_is_charged_its_ceiling():
    from aeo_index.pricing import RateCard, ceiling_call_usd
    from aeo_index.providers.anthropic import AnthropicProvider
    from aeo_index.providers.base import DEFAULT_MODELS

    async def broken(self, *args, **kwargs):
        raise KeyError("text")

    budget = Budget(Decimal("100"))
    original = AnthropicProvider.ask
    AnthropicProvider.ask = broken
    try:
        [row] = await run_cells(
            cells(engines=["anthropic"], questions=1, samples=1), httpx.MockTransport(lambda r: None), budget=budget
        )
    finally:
        AnthropicProvider.ask = original
    assert row.error_type == "bad_response" and row.cost_usd == Decimal("0")
    assert budget.spent == ceiling_call_usd("anthropic", DEFAULT_MODELS["anthropic"], "api", RateCard.default())
