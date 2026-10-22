"""`aeo-index rebuild` reproduces an edition the Sextant backend exported.

`fixtures/edition-parity/` holds `answers.jsonl`, `index.json` and
`prompts.csv` exactly as the backend's exporter wrote them from its seeded
test database: two runs of the ten index questions, every engine in each mode
it is asked in, one failed answer, Gemini under review, and the citation
shapes the domain rule reads. The rebuild must give the same `index.json`
except the fields only a publisher writes (`strip_publisher_fields`): one
method, two implementations. The run dates are in the year 2600, so the
fixture can never be read as a real edition.
"""

import json
from pathlib import Path
from urllib.parse import urlsplit

from aeo_index.citations import citation_domains, citation_urls
from aeo_index.cli import main
from aeo_index.edition import build_edition, strip_publisher_fields, validate_edition
from aeo_index.outputs import read_facts, read_prompts_csv
from aeo_index.questions import default_questions
from aeo_index.rebuild import load_set
from aeo_index.records import RunInfo

PARITY = Path(__file__).parent / "fixtures" / "edition-parity"
ANSWERS = PARITY / "answers.jsonl"
PROMPTS = PARITY / "prompts.csv"
FIXTURE_YEAR = "2600"


def _exported() -> dict:
    return json.loads((PARITY / "index.json").read_text(encoding="utf-8"))


def _rebuild_args(exported: dict, out: Path) -> list[str]:
    args = ["rebuild", "--answers", str(ANSWERS), "--questions", str(PROMPTS)]
    args += ["--edition", exported["edition"], "--out", str(out)]
    for engine in exported["engines_under_review"]:
        args += ["--under-review", engine]
    return args


def test_rebuild_equals_the_backend_export(tmp_path):
    exported = _exported()
    out = tmp_path / "rebuilt"

    assert main(_rebuild_args(exported, out)) == 0

    rebuilt = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert strip_publisher_fields(rebuilt) == strip_publisher_fields(exported)


def test_the_exported_edition_validates():
    assert validate_edition(_exported()) == []


def test_the_fixture_exercises_what_parity_must_cover():
    """Not a vacuous fixture: both modes, a failed answer, an engine under
    review, the brand's own domain, and a schemeless host."""
    exported = _exported()
    facts = read_facts(ANSWERS)
    assert {f.mode for f in facts} == {"api", "search"}
    assert any(f.status != "success" for f in facts)
    assert exported["engines_under_review"] == ["gemini"]
    domains = {d["domain"]: d for d in exported["modes"]["search"]["top_domains"]}
    assert domains["sextantaeo.com"]["sextant"] is True
    assert "bare-host.example" in domains


def test_prompts_csv_is_the_index_set_so_the_brand_applies():
    """`rebuild` tracks Sextant AEO only when the prompts.csv holds exactly
    the index's ten questions; the parity above depends on it."""
    assert set(read_prompts_csv(PROMPTS)) == set(default_questions().questions)
    loaded, index = load_set(PROMPTS, brand=None, brand_aliases=(), brand_domain=None), default_questions()
    assert (loaded.id, loaded.brand) == (index.id, index.brand)
    assert loaded.brand is not None and loaded.brand.domain == "sextantaeo.com"


def test_the_package_builder_over_the_facts_equals_the_export():
    """The same parity through `edition.build_edition`, with the runs and
    the asked engines the export states, as the backend's own test makes."""
    exported = _exported()
    qset = default_questions()
    keys = [p["key"] for p in exported["prompts"]]
    by_key = {q.key: q for q in qset.questions}
    runs = [RunInfo(r["run_date"], r["status"], r["engines_answered"], r["goal_mode"]) for r in exported["runs"]]
    asked = {mode: [row["engine"] for row in block["engines"]] for mode, block in exported["modes"].items()}
    rebuilt = build_edition(
        read_facts(ANSWERS),
        [by_key[k] for k in keys],
        runs,
        asked=asked,
        edition=exported["edition"],
        generated_at=exported["generated_at"],
        brand_domain=qset.brand.domain if qset.brand is not None else None,
        engines_under_review=exported["engines_under_review"],
    )
    assert strip_publisher_fields(rebuilt) == strip_publisher_fields(exported)


def test_every_fixture_date_is_in_the_year_2600():
    exported = _exported()
    dates = [exported["edition"], exported["generated_at"][:10]]
    dates += [run["run_date"] for run in exported["runs"]]
    dates += [fact.run_date for fact in read_facts(ANSWERS)]
    assert dates and all(day.startswith(f"{FIXTURE_YEAR}-") for day in dates), sorted(set(dates))


def test_answers_carry_only_web_urls_or_hosts_the_domain_rule_reads():
    citations = [c for fact in read_facts(ANSWERS) for c in fact.citations]
    assert citations
    for citation in citations:
        scheme = urlsplit(citation).scheme.lower()
        if scheme:
            assert scheme in ("http", "https") and citation_urls([citation]) == [citation], citation
        else:
            assert citation_domains([citation]), citation
    raw = ANSWERS.read_text(encoding="utf-8")
    assert "mailto:" not in raw and "javascript:" not in raw and "data:" not in raw


def test_answers_jsonl_holds_facts_only():
    """No answer text, and every row the shape `AnswerFacts` writes."""
    lines = ANSWERS.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    for fact, row in zip(read_facts(ANSWERS), rows, strict=True):
        assert fact.to_json() == row
    assert all("text" not in row and "answer" not in row for row in rows)
