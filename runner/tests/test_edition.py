import copy
import json
from decimal import Decimal

import pytest

from aeo_index.aliases import AliasTable
from aeo_index.edition import (
    BASIS_API_WHERE_ASKED,
    BASIS_SEARCH_ONLY,
    PUBLISHER_ONLY,
    PUBLISHER_ONLY_RUN_FIELDS,
    TOP_LEVEL_BASIS,
    asked_from_facts,
    build_edition,
    domain_rows,
    facts_from_answers,
    is_legacy,
    run_info,
    strip_publisher_fields,
    validate_edition,
)
from aeo_index.modes import TOP_DOMAINS
from aeo_index.questions import make_brand
from aeo_index.records import Judged
from run_doubles import answer, asked_of, build, fact, questions, sample_facts


@pytest.fixture
def facts():
    return sample_facts()


def count(rows, mode):
    return sum(1 for f in rows if f.mode == mode)


def test_constants():
    assert PUBLISHER_ONLY == ("generated_at", "sacs", "engines_under_review_note", "producer")
    assert PUBLISHER_ONLY_RUN_FIELDS == ("goal_met", "streak_days")


def test_edition_reconciles_and_parses(facts):
    doc = build(facts)
    assert validate_edition(doc) == []
    assert sum(r["answers"] for r in doc["modes"]["search"]["engines"]) == count(facts, "search")
    assert sum(r["answers"] for r in doc["modes"]["api"]["engines"]) == count(facts, "api")
    assert [k for k in doc["modes"]] == ["search", "api"]


def test_modes_are_never_summed(facts):
    doc = build(facts)
    assert all(r["mode"] == "search" for r in doc["modes"]["search"]["leaderboard_by_engine"])
    assert all(r["mode"] == "api" for r in doc["modes"]["api"]["leaderboard_by_engine"])
    assert doc["basis"] == TOP_LEVEL_BASIS == "api row per engine; search row for retrieval engines"
    assert [(r["engine"], r["mode"]) for r in doc["engines"]] == [
        ("anthropic", "api"),
        ("openai", "api"),
        ("perplexity", "search"),
    ]
    assert doc["modes"]["search"]["basis"] == "search row only" and doc["modes"]["api"]["basis"] == "api row only"
    search_board = {r["tool"]: r["first_count"] for r in doc["modes"]["search"]["leaderboard"]}
    assert search_board == {"Profound": 11, "Peec AI": 2}
    assert all(r["mode"] == "search" for r in doc["modes"]["search"]["leaderboard"])


def test_top_level_reads_one_row_per_engine(facts):
    doc = build(facts)
    top = {r["tool"]: r["first_count"] for r in doc["leaderboard"]}
    # anthropic api 6 + openai api 2 (Peec AI) + perplexity search 5.
    assert top == {"Profound": 11, "Peec AI": 2}
    assert "mode" not in doc["leaderboard"][0]
    assert sum(r["c"] for r in doc["leaderboard_by_engine"]) == sum(top.values())


def test_search_only_run_top_level_reads_search_rows(facts):
    search_only = [f for f in facts if f.mode == "search"]
    doc = build(search_only)
    assert doc["basis"] == BASIS_SEARCH_ONLY == "search row per engine"
    assert doc["modes"]["api"]["engines"] == [] and validate_edition(doc) == []
    assert doc["goal_mode"] == "search"


def test_mixed_run_says_where_the_api_row_was_asked(facts):
    mixed = [f for f in facts if not (f.engine == "openai" and f.mode == "api")]
    doc = build(mixed)
    assert doc["basis"] == BASIS_API_WHERE_ASKED == "api row where asked; search row otherwise"
    assert {(r["engine"], r["mode"]) for r in doc["engines"]} == {
        ("anthropic", "api"),
        ("openai", "search"),
        ("perplexity", "search"),
    }


def test_retrieval_only_run_reads_search_rows():
    doc = build([fact("q01", "perplexity", "search")])
    assert doc["basis"] == BASIS_SEARCH_ONLY and validate_edition(doc) == []


def test_experimental_question_not_on_leaderboard():
    rows = [fact("q01", "anthropic", "api"), fact("exp_01", "anthropic", "api", tools=("Otterly",))]
    doc = build(rows, questions("exp_01", "q01", experimental=["exp_01"]))
    assert "exp_01" not in {r["key"] for r in doc["leaderboard_by_prompt"]}
    assert "exp_01" in {r["key"] for r in doc["by_prompt"]}
    assert {r["tool"] for r in doc["leaderboard"]} == {"Profound"}
    assert validate_edition(doc) == []


def test_failed_answers_count_in_answers_not_ok(facts):
    doc = build(facts)
    api = {r["engine"]: r for r in doc["modes"]["api"]["engines"]}
    assert (api["openai"]["answers"], api["openai"]["ok"]) == (6, 4)
    search = {r["engine"]: r for r in doc["modes"]["search"]["engines"]}
    assert (search["perplexity"]["answers"], search["perplexity"]["ok"]) == (6, 5)


def test_no_answer_text_in_edition():
    text = "A distinctive answer sentence that must never be published."
    answers = [answer("q01", text=text), answer("q01", sample=2, text=text)]
    judged = [Judged(("Profound",), (), Decimal("0.001")), Judged(None, (), Decimal("0"))]
    facts = facts_from_answers(answers, judged, run_date="2026-10-21", brand=None, table=AliasTable.default())
    doc = build(facts)
    assert text not in json.dumps(doc) and "distinctive" not in json.dumps(doc)


def test_brand_rank_and_domain_flag(facts):
    doc = build(facts)
    cell = next(r for r in doc["by_prompt"] if (r["key"], r["mode"], r["engine"]) == ("q01", "search", "anthropic"))
    assert (cell["sextant_named"], cell["sextant_first"], cell["sextant_ranks"]) == (1, 0, [2])
    domains = {r["domain"]: r for r in doc["modes"]["search"]["top_domains"]}
    assert domains["acmecrm.com"]["sextant"] is True and domains["example.com"]["sextant"] is False
    assert domains["example.com"]["n"] == 17  # the URL and the bare domain count once per answer
    rows = domain_rows(facts, brand_domain="acmecrm.com")
    assert {
        "mode": "search",
        "domain": "acmecrm.com",
        "citations": 17,
        "engines": "anthropic;openai;perplexity",
        "sextant": True,
    } in rows


def test_no_brand_domain_flags_nothing(facts):
    doc = build(facts, brand_domain=None)
    assert not any(r["sextant"] for r in doc["top_domains"] + doc["modes"]["search"]["top_domains"])


def test_top_urls_keep_http_urls_and_join_engines(facts):
    doc = build(facts)
    urls = {r["url"]: r for r in doc["modes"]["search"]["top_urls"]}
    assert set(urls) == {"https://example.com/reviews/aeo-tools", "https://acmecrm.com/blog"}
    assert urls["https://acmecrm.com/blog"]["engines"] == "anthropic,openai,perplexity"
    assert urls["https://acmecrm.com/blog"]["n"] == 17 and urls["https://acmecrm.com/blog"]["mode"] == "search"


def test_top_domains_are_capped_and_domain_rows_are_not():
    rows = [
        fact("q01", "perplexity", "search", sample, citations=[f"https://site{i:03d}.example.com/" for i in range(50)])
        for sample in (1, 2)
    ]
    doc = build(rows)
    assert len(doc["modes"]["search"]["top_domains"]) == TOP_DOMAINS
    assert len([r for r in domain_rows(rows, brand_domain=None) if r["mode"] == "search"]) == 50


def test_by_prompt_carries_the_run_window():
    rows = [
        fact("q01", "anthropic", "api", run_date="2026-10-20"),
        fact("q01", "anthropic", "api", 2, run_date="2026-10-21"),
    ]
    doc = build(rows)
    (cell,) = doc["by_prompt"]
    assert (cell["first_run"], cell["last_run"], cell["days"]) == ("2026-10-20", "2026-10-21", 2)
    assert [r["run_date"] for r in doc["runs"]] == ["2026-10-20", "2026-10-21"]
    assert doc["edition"] == "2026-10-21"


def test_leaderboard_by_engine_is_in_roster_order():
    rows = [fact("q01", "openai", "api"), fact("q01", "anthropic", "api", tools=("Otterly",))]
    doc = build(rows)
    assert [r["engine"] for r in doc["leaderboard_by_engine"]] == ["anthropic", "openai"]


def test_prompts_are_listed_by_key():
    rows = [fact("q02", "anthropic", "api"), fact("q01", "anthropic", "api")]
    doc = build(rows, questions("q02", "q01"))
    assert [p["key"] for p in doc["prompts"]] == ["q01", "q02"]


def test_facts_from_unknown_question_or_unasked_engine_are_refused(facts):
    with pytest.raises(ValueError, match="q09"):
        build(facts + [fact("q09", "anthropic", "api")], questions("q01", "q02"))
    with pytest.raises(ValueError, match="grok"):
        build_edition(
            [fact("q01", "grok", "api")],
            questions("q01"),
            [],
            asked={"api": ["anthropic"], "search": []},
            edition="2026-10-21",
            generated_at="2026-10-21T12:00:00+00:00",
            brand_domain=None,
        )


def test_producer_and_under_review_are_written(facts):
    asked = asked_of(facts)
    doc = build_edition(
        facts,
        questions("q01", "q02"),
        [run_info(facts, asked, run_date="2026-10-21")],
        asked=asked,
        edition="2026-10-21",
        generated_at="2026-10-21T12:00:00+00:00",
        brand_domain=None,
        engines_under_review=["openai"],
        producer={"name": "aeo-index"},
    )
    assert doc["engines_under_review"] == ["openai"] and doc["producer"] == {"name": "aeo-index"}
    assert "sacs" not in doc


# ------------------------------------------------------------------ facts and runs


def test_facts_from_answers_ranks_the_brand():
    brand = make_brand("Acme CRM", ["Acme"], "acmecrm.com")
    table = AliasTable.default().merged({brand.name: brand.spellings()})
    answers = [answer("q01"), answer("q01", sample=2), answer("q01", sample=3, status="error")]
    judged = [
        Judged(("Profound", "Acme CRM"), ("Ghost",), Decimal("0.001")),
        Judged((), (), Decimal("0.001")),
        Judged(None, (), Decimal("0")),
    ]
    facts = facts_from_answers(answers, judged, run_date="2026-10-21", brand=brand, table=table)
    assert [(f.first_named, f.sextant_rank, f.tools_named) for f in facts] == [
        ("Profound", 2, ("Profound", "Acme CRM")),
        (None, None, ()),
        (None, None, None),
    ]
    assert facts[2].status == "error"
    unbranded = facts_from_answers(answers, judged, run_date="2026-10-21", brand=None, table=table)
    assert unbranded[0].sextant_rank is None


def test_facts_from_answers_needs_one_judged_per_answer():
    with pytest.raises(ValueError):
        facts_from_answers([answer()], [], run_date="2026-10-21", brand=None, table=AliasTable.default())


def test_run_info_statuses():
    asked = {"api": ["anthropic", "openai"], "search": []}
    full = [fact("q01", "anthropic", "api"), fact("q01", "openai", "api")]
    info = run_info(full, asked, run_date="2026-10-21")
    assert (info.status, info.engines_answered, info.goal_mode) == ("completed", 2, "api")
    part = [fact("q01", "anthropic", "api"), fact("q01", "openai", "api", status="error")]
    assert run_info(part, asked, run_date="2026-10-21").status == "partial"
    none = [fact("q01", "anthropic", "api", status="error"), fact("q01", "openai", "api", status="timeout")]
    info = run_info(none, asked, run_date="2026-10-21")
    assert (info.status, info.engines_answered) == ("failed", 0)


def test_run_info_needs_a_success_in_every_cell_not_every_sample(facts):
    # Every (question, engine, mode) cell of the sample has at least one success.
    assert run_info(facts, asked_of(facts), run_date="2026-10-21").status == "completed"
    assert run_info(facts, asked_of(facts), run_date="2026-10-21").goal_mode == "search"


def test_asked_from_facts(facts):
    assert asked_from_facts(facts) == {"api": ["anthropic", "openai"], "search": ["anthropic", "openai", "perplexity"]}


# ------------------------------------------------------------------ validation


def test_validate_catches_rows_that_do_not_add_up(facts):
    doc = build(facts)
    broken = copy.deepcopy(doc)
    broken["modes"]["search"]["engines"][0]["answers"] += 1
    assert any("do not add up" in e for e in validate_edition(broken))


def test_validate_catches_missing_and_malformed_fields(facts):
    doc = build(facts)
    for field in ("edition", "prompts", "by_prompt", "engines_under_review", "top_urls"):
        broken = copy.deepcopy(doc)
        del broken[field]
        assert validate_edition(broken), field
    broken = copy.deepcopy(doc)
    broken["by_prompt"][0]["sextant_ranks"] = [0]
    assert validate_edition(broken)
    broken = copy.deepcopy(doc)
    broken["leaderboard_by_prompt"].append({"key": "q01", "mode": "api", "engine": "grok", "tool": "X", "c": 1})
    assert any("by_prompt does not" in e for e in validate_edition(broken))
    assert validate_edition([]) and validate_edition({"modes": {"web": {}}})


def test_validate_does_not_need_sacs(facts):
    doc = build(facts)
    assert "sacs" not in doc and validate_edition(doc) == []
    assert validate_edition({**doc, "sacs": {"evaluations": -1}})


def test_a_legacy_document_is_checked_on_its_legacy_keys(facts):
    doc = build(facts)
    legacy = {
        k: doc[k]
        for k in (
            "generated_at",
            "runs",
            "engines",
            "prompts",
            "leaderboard",
            "leaderboard_by_engine",
            "top_domains",
            "top_urls",
            "engines_under_review",
        )
    }
    assert is_legacy(legacy) and not is_legacy(doc)
    assert validate_edition(legacy) == []
    del legacy["leaderboard"]
    assert validate_edition(legacy)


def test_strip_publisher_fields(facts):
    doc = build(facts)
    published = {
        **copy.deepcopy(doc),
        "sacs": {"evaluations": 3},
        "engines_under_review_note": "x",
        "producer": {"name": "y"},
    }
    for run in published["runs"]:
        run.update(goal_met=False, streak_days=0)
    stripped = strip_publisher_fields(published)
    assert set(PUBLISHER_ONLY).isdisjoint(stripped)
    assert all(set(PUBLISHER_ONLY_RUN_FIELDS).isdisjoint(run) for run in stripped["runs"])
    assert stripped == strip_publisher_fields(doc)
    assert "goal_met" in published["runs"][0]  # the input is not changed


# ------------------------------------------------------------------ fix round 1: the goal row


def test_an_api_run_with_a_retrieval_engine_reads_its_goal_on_the_api_row():
    rows = [fact("q01", "openai", "api"), fact("q01", "perplexity", "search")]
    info = run_info(rows, asked_of(rows), run_date="2026-10-21")
    assert info.goal_mode == "api"
    assert build(rows)["goal_mode"] == "api"


def test_the_goal_row_is_search_once_a_model_engine_searched():
    rows = [fact("q01", "openai", "api"), fact("q01", "openai", "search"), fact("q01", "perplexity", "search")]
    assert run_info(rows, asked_of(rows), run_date="2026-10-21").goal_mode == "search"
    only_retrieval = [fact("q01", "google_aio", "search")]
    assert run_info(only_retrieval, asked_of(only_retrieval), run_date="2026-10-21").goal_mode == "api"
