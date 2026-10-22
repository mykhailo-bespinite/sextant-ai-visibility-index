import pytest

from aeo_index.table import render_table
from run_doubles import build, fact, sample_facts


@pytest.fixture
def outage_facts():
    rows = []
    for key in ("q01", "q02"):
        for sample in (1, 2, 3):
            rows.append(fact(key, "anthropic", "search", sample))
            rows.append(fact(key, "perplexity", "search", sample, status="error"))
    return rows


def test_table_shows_no_answers_engine_with_status_counts(outage_facts):
    md = render_table(build(outage_facts), outage_facts)
    assert "Perplexity: no answers (error 6)" in md and md.count("| Tool |") == 1  # search only run: one table


def test_one_table_per_mode_search_first():
    facts = sample_facts()
    md = render_table(build(facts), facts)
    assert md.count("| Tool |") == 2
    assert md.index("With web search") < md.index("API only")


def test_columns_are_engines_with_their_answered_counts():
    facts = sample_facts()
    md = render_table(build(facts), facts)
    search = md.split("API only")[0]
    assert "| Tool | Claude (6) | ChatGPT (6) | Perplexity (5) | All engines |" in search
    assert "| Profound | 6 | 0 | 5 | 11 |" in search and "| Peec AI | 0 | 2 | 0 | 2 |" in search
    assert "Perplexity: 5 of 6 answered (timeout 1)" in search
    api = md.split("API only")[1]
    assert "ChatGPT: 4 of 6 answered (error 2)" in api


def test_results_page_engine_says_so():
    facts = [fact("q01", "google_aio", "search")]
    assert "Google AI Overview, results page via DataForSEO (1)" in render_table(build(facts), facts)


def test_top_limits_the_rows():
    facts = [fact("q01", "anthropic", "api", s, tools=(f"Tool {s}",)) for s in range(1, 6)]
    md = render_table(build(facts), facts, top=3)
    assert md.count("| Tool ") == 1 + 3  # the header and three tools
    assert "2 more tools named first" in md


def test_tool_names_cannot_break_the_table():
    facts = [fact("q01", "anthropic", "api", tools=("A | B\nC",))]
    md = render_table(build(facts), facts)
    assert "| A \\| B C | 1 | 1 |" in md


def test_a_mode_where_nothing_was_named_first_says_so():
    facts = [fact("q01", "anthropic", "api", tools=())]
    md = render_table(build(facts), facts)
    assert "No answer named a tool first." in md
