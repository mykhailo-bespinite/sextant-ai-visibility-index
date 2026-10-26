"""The `agreement` block of an edition: how often the samples of one prompt
agree (proposal 3 part A).

`fixtures/agreement_cases.json` holds hand-built facts, the value each cell
must have, and the whole block they add up to. The Sextant backend keeps a
byte-identical copy and holds its own exporter to it: one method, two
implementations.
"""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from aeo_index.agreement import DEFINITION, agreement_block, cell_agreement
from aeo_index.edition import validate_edition
from aeo_index.modes import ENGINES
from aeo_index.records import AnswerFacts

CASES = json.loads((Path(__file__).parent / "fixtures" / "agreement_cases.json").read_text(encoding="utf-8"))
FACTS = [AnswerFacts.from_json(row) for row in CASES["facts"]]


def _fraction(value):
    return None if value is None else Fraction(value)


def _samples(case):
    where = (case["run_date"], case["key"], case["mode"], case["engine"])
    return [f for f in FACTS if (f.run_date, f.key, f.mode, f.engine) == where]


def _case(prefix):
    return next(c for c in CASES["cells"] if c["label"].startswith(prefix + " "))


@pytest.mark.parametrize("case", CASES["cells"], ids=[c["label"] for c in CASES["cells"]])
def test_each_cell_has_the_hand_derived_agreement(case):
    assert cell_agreement(_samples(case)) == (_fraction(case["first"]), _fraction(case["set"]))


def test_the_fixture_exercises_every_branch_of_the_definitions():
    assert {c["first"] for c in CASES["cells"]} >= {"1", "1/2", "0", None}
    assert {f.status for f in FACTS} >= {"success", "timeout", "empty", "error"}
    assert any(f.status == "success" and f.tools_named is None for f in FACTS)
    assert any(f.status == "success" and f.tools_named == () for f in FACTS)


def test_the_block_is_the_hand_derived_one():
    assert agreement_block(FACTS) == CASES["expected"]


def test_the_block_does_not_depend_on_the_order_of_the_facts():
    assert agreement_block(list(reversed(FACTS))) == CASES["expected"]


def test_three_agreeing_samples_score_one():
    assert cell_agreement(_samples(_case("A"))) == (1, 1)


def test_two_of_three_agreeing_score_one_half_first_named():
    first, _ = cell_agreement(_samples(_case("B")))
    assert first == Fraction(1, 2)


def test_a_failed_sample_leaves_n_two_and_alias_spellings_agree():
    samples = _samples(_case("C"))
    assert [f.status for f in samples].count("success") == 2
    assert {f.first_named for f in samples if f.status == "success"} == {"Semrush", "Semrush AI Toolkit"}
    assert cell_agreement(samples) == (1, 1)


def test_a_cell_where_no_sample_names_a_tool_has_no_first_agreement():
    assert cell_agreement(_samples(_case("D"))) == (None, 1)


def test_the_definitions_are_two_sentences_in_the_block():
    assert set(DEFINITION) == {"first_agreement", "set_agreement"}
    assert all(sentence.endswith(".") for sentence in DEFINITION.values())


def test_no_facts_give_an_empty_block():
    block = agreement_block([])
    assert block["by_engine"] == block["by_prompt"] == block["by_mode"] == []


def test_the_block_lists_engines_in_roster_order_within_each_mode():
    rows = agreement_block(FACTS)["by_engine"]
    for mode in {row["mode"] for row in rows}:
        engines = [row["engine"] for row in rows if row["mode"] == mode]
        assert engines == sorted(engines, key=ENGINES.index)


def test_the_block_never_pools_the_two_modes():
    block = agreement_block(FACTS)
    assert set(block) == {"definition", "by_engine", "by_prompt", "by_mode"}
    assert all("mode" in row for group in ("by_engine", "by_prompt", "by_mode") for row in block[group])


def test_the_definitions_say_a_judged_answer_is_required():
    for sentence in DEFINITION.values():
        assert "not defined with fewer than two such answers" in sentence
    assert "tools the judge listed" in DEFINITION["first_agreement"]


def _parity_edition():
    return json.loads(
        (Path(__file__).parent / "fixtures" / "edition-parity" / "index.json").read_text(encoding="utf-8")
    )


def _edition_with(block):
    return {**_parity_edition(), "agreement": block}


def test_an_edition_without_the_block_still_validates():
    doc = {key: value for key, value in _parity_edition().items() if key != "agreement"}
    assert validate_edition(doc) == []


def test_an_edition_with_the_block_validates():
    assert "agreement" in _parity_edition()
    assert validate_edition(_parity_edition()) == []


def test_a_block_for_other_questions_than_the_editions_is_refused():
    errors = validate_edition(_edition_with(CASES["expected"]))
    assert any("which the edition does not list" in error for error in errors)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda b: b.update(by_mode="x"), 'agreement: "by_mode" is not a list'),
        (
            lambda b: b["by_mode"][0].update(first_agreement=1.5),
            "agreement.by_mode.first_agreement is not a share between 0 and 1",
        ),
        (
            lambda b: b["by_mode"][0].update(set_agreement="0.5"),
            "agreement.by_mode.set_agreement is not a share between 0 and 1",
        ),
        (lambda b: b["by_mode"][0].update(cells=-1), "agreement.by_mode.cells is not a count"),
        (lambda b: b["by_engine"][0].update(mode="bogus"), 'agreement.by_engine: unknown mode "bogus"'),
        (
            lambda b: b["by_mode"][0].update(first_cells=b["by_mode"][0]["cells"] + 1),
            "agreement.by_mode: first_cells",
        ),
        (
            lambda b: b["by_mode"][0].update(first_agreement=None, first_cells=1),
            "agreement.by_mode: first_agreement and first_cells disagree on whether any cell was defined",
        ),
        (
            lambda b: b["by_mode"][0].update(set_agreement=0.5, cells=0, first_cells=0),
            "agreement.by_mode: set_agreement and cells disagree on whether any cell was defined",
        ),
        (lambda b: b["by_mode"].append(dict(b["by_mode"][0])), "agreement.by_mode has "),
        (lambda b: b["by_prompt"][0].update(key="not-a-question"), 'agreement.by_prompt names prompt "not-a-question"'),
        (
            lambda b: b["by_engine"][0].update(engine="nobody"),
            "agreement.by_engine has nobody in ",
        ),
        (lambda b: b["by_prompt"][0].pop("key"), "agreement.by_prompt.key is not a non-empty string"),
        (
            lambda b: b["definition"].update(set_agreement=""),
            "agreement.definition.set_agreement is not a non-empty string",
        ),
    ],
)
def test_a_malformed_block_is_named(mutate, message):
    block = json.loads(json.dumps(_parity_edition()["agreement"]))
    mutate(block)
    assert any(message in error for error in validate_edition(_edition_with(block)))
