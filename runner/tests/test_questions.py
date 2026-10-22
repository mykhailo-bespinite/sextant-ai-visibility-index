import csv
from pathlib import Path

import pytest

from aeo_index.questions import (
    DEFAULT_EXCLUDE,
    DEFAULT_GENERIC_EXAMPLE,
    DEFAULT_PURPOSE,
    DEFAULT_SET_ID,
    DEFAULT_SUBJECT,
    SEXTANT_BRAND,
    Brand,
    QuestionFileError,
    branded_names,
    default_questions,
    load_questions,
    make_brand,
)
from conftest import write_toml

PUBLISHED_PROMPTS = Path(__file__).parents[2] / "data" / "2026-10-18" / "prompts.csv"
EXAMPLE = Path(__file__).parents[1] / "examples" / "example-category.toml"


def test_default_set_equals_the_published_prompts():
    if not PUBLISHED_PROMPTS.exists():
        pytest.skip("needs the repository's data/ folder")
    with PUBLISHED_PROMPTS.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    qs = default_questions()
    assert qs.id == "sextant-index-v1" == DEFAULT_SET_ID
    assert [(q.key, q.text, q.experimental) for q in qs.questions] == [
        (r["key"], r["prompt"], r["experimental"] == "True") for r in rows
    ]
    assert len(qs.questions) == 10 and qs.brand == SEXTANT_BRAND


def test_default_set_uses_the_judge_defaults():
    qs = default_questions()
    assert (qs.subject, qs.purpose, qs.generic_example, qs.exclude) == (
        DEFAULT_SUBJECT,
        DEFAULT_PURPOSE,
        DEFAULT_GENERIC_EXAMPLE,
        DEFAULT_EXCLUDE,
    )
    assert DEFAULT_SUBJECT == "AEO (Answer Engine Optimization) or AI visibility tools"
    assert DEFAULT_PURPOSE == "AEO or AI visibility"
    assert DEFAULT_GENERIC_EXAMPLE == "an AI visibility tool"
    assert DEFAULT_EXCLUDE == ("ChatGPT", "Claude", "Perplexity", "Gemini", "Grok")
    assert SEXTANT_BRAND == Brand("Sextant AEO", ("Sextant", "sextantaeo.com"), "sextantaeo.com")


def test_branded_question_refused_names_key_and_name(tmp_path):
    f = write_toml(tmp_path, questions=[("q01", "Is Profound better than the rest for agencies?")])
    with pytest.raises(QuestionFileError, match=r"q01.*Profound"):
        load_questions(f)


def test_brand_and_file_aliases_count_as_branded(tmp_path):
    by_brand = write_toml(
        tmp_path,
        "brand.toml",
        brand="Acme CRM",
        brand_aliases=["Acme", "acmecrm.com"],
        brand_domain="acmecrm.com",
        questions=[("q01", "Is the pricing page at acmecrm.com fair for agencies?")],
    )
    with pytest.raises(QuestionFileError, match=r"q01.*acmecrm\.com"):
        load_questions(by_brand)
    by_alias = write_toml(
        tmp_path,
        "alias.toml",
        aliases={"Zed CRM": ["Zed", "zedcrm.io"]},
        questions=[("q07", "Should a small agency move to Zed this year?")],
    )
    with pytest.raises(QuestionFileError, match=r"q07.*Zed"):
        load_questions(by_alias)


def test_branded_check_is_word_bounded_and_case_insensitive():
    assert branded_names("which PEEC plan?", ["Peec", "Peec AI"]) == ["Peec"]
    assert branded_names("Peecock tools", ["Peec"]) == []
    assert branded_names("see otterly.ai today", ["Otterly.AI", "Otterly"]) == ["Otterly.AI", "Otterly"]
    assert branded_names("a+b tools", ["a+b"]) == ["a+b"]


def test_allow_branded_loads_as_experimental(tmp_path):
    qs = load_questions(
        write_toml(tmp_path, questions=[("q01", "Is Profound any good?"), ("q02", "Which CRM fits a small agency?")]),
        allow_branded=True,
    )
    assert [q.experimental for q in qs.questions] == [True, False]


def test_engine_names_are_not_brands(tmp_path):
    assert default_questions().questions[2].text.startswith("What should I use to track my brand in ChatGPT")
    qs = load_questions(write_toml(tmp_path, questions=[("q01", "Which CRM do ChatGPT, Claude and Gemini suggest?")]))
    assert qs.questions[0].experimental is False


def test_file_fields_and_brand_are_read(tmp_path):
    qs = load_questions(
        write_toml(
            tmp_path,
            generic_example="a CRM tool",
            exclude=["ChatGPT"],
            brand="Acme CRM",
            brand_aliases=["Acme"],
            brand_domain="acmecrm.com",
            aliases={"Zed CRM": ["Zed"]},
            questions=[("q01", "Which CRM fits a small agency?", True)],
        )
    )
    assert (qs.generic_example, qs.exclude) == ("a CRM tool", ("ChatGPT",))
    assert qs.brand == Brand("Acme CRM", ("Acme",), "acmecrm.com")
    assert qs.aliases == {"Zed CRM": ("Zed",)}
    assert qs.questions[0].experimental is True


def test_question_set_alias_table_has_file_aliases_and_the_brand(tmp_path):
    qs = load_questions(
        write_toml(
            tmp_path, brand="Acme CRM", brand_aliases=["Acme"], brand_domain="acmecrm.com", aliases={"Zed CRM": ["Zed"]}
        )
    )
    table = qs.alias_table()
    assert table.normalise("zed") == "Zed CRM"
    assert table.normalise("acmecrm.com") == "Acme CRM" and table.normalise("acme") == "Acme CRM"
    assert table.normalise("Peec") == "Peec AI"
    assert default_questions().alias_table().normalise("sextant") == "Sextant AEO"


def test_custom_file_without_brand_has_none(tmp_path):
    assert load_questions(write_toml(tmp_path)).brand is None


MALFORMED = {
    "duplicate key": (
        {"questions": [("q01", "Which CRM fits a small agency?"), ("q01", "Which CRM fits a large agency?")]},
        "used twice",
    ),
    "key with a space": ({"questions": [("q 01", "Which CRM fits a small agency?")]}, "needs a key"),
    "no questions": ({"questions": []}, "has 0 questions"),
    "51 questions": (
        {"questions": [(f"q{i:02d}", f"Which CRM fits agency number {i}?") for i in range(51)]},
        "has 51 questions",
    ),
    "text of 9 chars": ({"questions": [("q01", "Which CRM")]}, "is 9 characters"),
    "text of 501 chars": ({"questions": [("q01", "W" * 501)]}, "is 501 characters"),
    "unknown top-level field": ({"colour": "blue"}, "unknown field 'colour'"),
    "experimental not true or false": (
        {"questions": [("q01", "Which CRM fits a small agency?", "yes")]},
        "experimental must be true or false",
    ),
    "missing subject": ({"omit": ["subject"]}, "'subject' is required"),
    "missing id": ({"omit": ["id"]}, "'id' is required"),
    "id with a slash": ({"id": "../escape"}, "id '../escape'"),
    "exclude not a list": ({"exclude": "ChatGPT"}, "exclude must be a list"),
    "brand alias without brand": ({"brand_aliases": ["Acme"]}, "needs a brand"),
    "bad brand domain": ({"brand": "Acme CRM", "brand_domain": "https://acmecrm.com/"}, "bare domain"),
    "alias clash": ({"aliases": {"Semrush One": ["Semrush"]}}, "claimed by both"),
}


@pytest.mark.parametrize("bad", list(MALFORMED))
def test_malformed_files_are_refused_with_a_reason(tmp_path, bad):
    kwargs, reason = MALFORMED[bad]
    path = write_toml(tmp_path, **kwargs)
    with pytest.raises(QuestionFileError) as err:
        load_questions(path)
    message = str(err.value)
    assert message.startswith(f"{path}: ") and reason in message


def test_cli_brand_override_is_checked_against_questions(tmp_path):
    qs = load_questions(
        write_toml(tmp_path, questions=[("q01", "Is Acme worth it for a small agency?"), ("q02", "Which CRM fits us?")])
    )
    brand = make_brand("Acme CRM", aliases=["Acme"], domain="acmecrm.com")
    with pytest.raises(QuestionFileError, match=r"^--brand: question q01 names 'Acme'"):
        qs.with_brand(brand)
    marked = qs.with_brand(brand, allow_branded=True)
    assert [q.experimental for q in marked.questions] == [True, False]
    assert marked.brand == Brand("Acme CRM", ("Acme",), "acmecrm.com")
    assert marked.alias_table().normalise("acmecrm.com") == "Acme CRM"


def test_brand_override_replaces_the_default_brand():
    qs = default_questions().with_brand(make_brand("Acme CRM", aliases=["Acme"]))
    assert qs.brand == Brand("Acme CRM", ("Acme",), None)
    assert qs.alias_table().normalise("acme") == "Acme CRM"
    assert default_questions().with_brand(None).brand is None


def test_brand_override_clash_is_a_question_file_error():
    with pytest.raises(QuestionFileError, match=r"^--brand: .*claimed by both"):
        default_questions().with_brand(make_brand("Semrush One", aliases=["Semrush"]))


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"name": " "}, "brand must be a non-empty string"),
        ({"name": "Acme", "aliases": ["", "x"]}, "brand_aliases"),
        ({"name": "Acme", "domain": "acme.com/path"}, "bare domain"),
    ],
)
def test_bad_brand_flags_are_refused(kwargs, reason):
    with pytest.raises(QuestionFileError, match=reason):
        make_brand(**kwargs)


def test_question_sets_are_not_hashable():
    with pytest.raises(TypeError):
        hash(default_questions())


def test_unknown_question_field_is_refused(tmp_path):
    path = tmp_path / "q.toml"
    path.write_text(
        'id = "x1"\nsubject = "CRM tools"\npurpose = "CRM"\n\n'
        '[[question]]\nkey = "q01"\ntext = "Which CRM fits a small agency?"\nweight = 2\n',
        encoding="utf-8",
    )
    with pytest.raises(QuestionFileError, match="unknown field 'weight'"):
        load_questions(path)


def test_not_toml_and_missing_file_are_refused(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("id = [unclosed\n", encoding="utf-8")
    with pytest.raises(QuestionFileError, match="not valid TOML"):
        load_questions(bad)
    with pytest.raises(QuestionFileError, match=r"cannot read the file \(No such file or directory\)"):
        load_questions(tmp_path / "missing.toml")


def test_example_category_file_loads():
    qs = load_questions(EXAMPLE)
    assert qs.brand is not None and qs.brand.name == "Acme CRM"
    assert qs.id == "crm-for-agencies-v1" and qs.aliases["Zed CRM"] == ("Zed", "zedcrm.io")
