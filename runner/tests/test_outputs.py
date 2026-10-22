import csv
import json

import pytest

from aeo_index.edition import strip_publisher_fields
from aeo_index.outputs import (
    DOMAINS_COLUMNS,
    LEADERBOARD_COLUMNS,
    OUTPUT_FILES,
    PROMPTS_COLUMNS,
    read_facts,
    read_prompts_csv,
    write_run,
)
from aeo_index.questions import default_questions
from aeo_index.table import render_table
from run_doubles import answer, build, fact, questions, sample_facts


def _write(tmp_path, facts, answers=None, *, include_text=True, qset=None):
    doc = build(facts)
    return doc, write_run(
        tmp_path / "out",
        doc=doc,
        facts=facts,
        answers=answers,
        questions=qset or default_questions(),
        table_md=render_table(doc, facts),
        include_text=include_text,
    )


def test_columns():
    assert PROMPTS_COLUMNS == (
        "key",
        "prompt_text",
        "experimental",
        "mode",
        "engine",
        "answers",
        "ok",
        "sextant_named",
        "sextant_first",
    )
    assert LEADERBOARD_COLUMNS == ("mode", "engine", "tool", "first_count")
    assert DOMAINS_COLUMNS == ("mode", "domain", "citations", "engines", "sextant")
    assert OUTPUT_FILES == (
        "index.json",
        "table.md",
        "leaderboard.csv",
        "cited_domains.csv",
        "prompts.csv",
        "answers.jsonl",
    )


def test_write_run_writes_the_six_files(tmp_path):
    facts = sample_facts()
    doc, paths = _write(tmp_path, facts)
    assert [p.name for p in paths] == list(OUTPUT_FILES)
    assert json.loads(paths[0].read_text(encoding="utf-8")) == doc
    board = list(csv.DictReader(paths[2].open(encoding="utf-8")))
    assert {(r["mode"], r["engine"], r["tool"], r["first_count"]) for r in board} >= {
        ("search", "anthropic", "Profound", "6"),
        ("api", "openai", "Peec AI", "2"),
    }
    prompts = list(csv.DictReader(paths[4].open(encoding="utf-8")))
    assert len(prompts) == len(doc["by_prompt"]) and prompts[0]["prompt_text"].startswith("Which tool")
    lines = paths[5].read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(facts)


def test_cited_domains_flag_the_question_sets_brand(tmp_path):
    facts = [fact("q01", "perplexity", "search", citations=["https://sextantaeo.com/research"])]
    _, paths = _write(tmp_path, facts)
    rows = list(csv.DictReader(paths[3].open(encoding="utf-8")))
    assert rows == [
        {"mode": "search", "domain": "sextantaeo.com", "citations": "1", "engines": "perplexity", "sextant": "True"}
    ]


def test_formula_cells_are_quoted(tmp_path):
    facts = [fact("q01", "anthropic", "api", tools=("=HYPERLINK(1)", "+x"))]
    _, paths = _write(tmp_path, facts)
    text = paths[2].read_text(encoding="utf-8")
    assert "'=HYPERLINK(1)" in text and ",=HYPERLINK" not in text


def test_answers_jsonl_holds_the_facts_and_the_text_unless_left_out(tmp_path):
    answers = [answer("q01", text="Profound leads."), answer("q01", sample=2, status="error")]
    facts = [fact("q01"), fact("q01", sample=2, status="error")]
    _, paths = _write(tmp_path, facts, answers)
    rows = [json.loads(line) for line in paths[5].read_text(encoding="utf-8").splitlines()]
    assert rows[0]["text"] == "Profound leads." and rows[0]["tools_named"] == ["Profound"]
    assert rows[1]["text"] is None and rows[1]["error"] == "the vendor answered HTTP 503"
    assert rows[0]["cost_usd"] == "0.01" and rows[0]["model_version"] == "model-x"
    _, paths = _write(tmp_path / "b", facts, answers, include_text=False)
    rows = [json.loads(line) for line in paths[5].read_text(encoding="utf-8").splitlines()]
    assert "text" not in rows[0] and "Profound leads." not in paths[5].read_text(encoding="utf-8")


def test_answers_must_pair_with_facts(tmp_path):
    with pytest.raises(ValueError):
        _write(tmp_path, [fact("q01")], [answer("q02")])


def test_read_facts_round_trips_and_names_the_bad_line(tmp_path):
    facts = sample_facts()
    _, paths = _write(tmp_path, facts)
    assert read_facts(paths[5]) == facts
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(facts[0].to_json()) + "\n\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        read_facts(bad)
    bad.write_text(json.dumps({**facts[0].to_json(), "status": "great"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1.*status"):
        read_facts(bad)


def test_read_prompts_csv_reads_ours_and_the_published_one(tmp_path):
    facts = sample_facts()
    _, paths = _write(tmp_path, facts)
    assert read_prompts_csv(paths[4]) == questions("q01", "q02")
    from pathlib import Path

    published = Path(__file__).resolve().parents[2] / "data" / "2026-10-18" / "prompts.csv"
    assert [q.key for q in read_prompts_csv(published)] == [q.key for q in default_questions().questions]


def test_read_prompts_csv_refuses_a_key_with_two_texts(tmp_path):
    path = tmp_path / "prompts.csv"
    path.write_text("key,prompt_text,experimental\nq01,One text here?,False\nq01,Another text?,False\n")
    with pytest.raises(ValueError, match="q01"):
        read_prompts_csv(path)


def test_rebuild_equals_the_written_edition(tmp_path):
    facts = sample_facts()
    doc, paths = _write(tmp_path, facts)
    again = build(read_facts(paths[5]), read_prompts_csv(paths[4]))
    assert strip_publisher_fields(again) == strip_publisher_fields(doc)
