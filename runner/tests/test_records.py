from decimal import Decimal

import pytest

from aeo_index.records import Answer, AnswerFacts, Judged, JudgeSpec, RunInfo

FACTS = AnswerFacts(
    key="bhag_01",
    engine="openai",
    mode="search",
    sample=1,
    run_date="2026-10-21",
    status="success",
    tools_named=("Profound", "Peec AI"),
    first_named="Profound",
    sextant_rank=None,
    citations=("https://example.com/a", "example.com"),
)


def test_facts_round_trip_through_json():
    obj = FACTS.to_json()
    assert obj["tools_named"] == ["Profound", "Peec AI"] and obj["citations"] == [
        "https://example.com/a",
        "example.com",
    ]
    assert AnswerFacts.from_json(obj) == FACTS


def test_not_judged_round_trips():
    obj = {**FACTS.to_json(), "tools_named": None, "first_named": None, "status": "error"}
    assert AnswerFacts.from_json(obj).tools_named is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("key", 3),
        ("engine", "bing"),
        ("mode", "deep"),
        ("sample", 0),
        ("sample", True),
        ("run_date", "21/10/2026"),
        ("status", "fine"),
        ("tools_named", "Profound"),
        ("tools_named", [1]),
        ("first_named", 2),
        ("sextant_rank", 0),
        ("citations", None),
    ],
)
def test_from_json_names_the_bad_field(field, value):
    obj = {**FACTS.to_json(), field: value}
    with pytest.raises(ValueError, match=field):
        AnswerFacts.from_json(obj)


def test_from_json_names_a_missing_field():
    obj = FACTS.to_json()
    del obj["status"]
    with pytest.raises(ValueError, match="status"):
        AnswerFacts.from_json(obj)


def test_judge_spec_parse():
    assert JudgeSpec.parse("anthropic:claude-haiku-4-5-20251001") == JudgeSpec("anthropic", "claude-haiku-4-5-20251001")
    assert JudgeSpec.parse("gemini:models/gemini-3.5-flash").model == "models/gemini-3.5-flash"
    for bad in (
        "anthropic",
        "perplexity:sonar",
        "anthropic:",
        ":m",
        "openai: gpt",
        "gemini:models/../files",
        "gemini:a/b/c",
    ):
        with pytest.raises(ValueError):
            JudgeSpec.parse(bad)


def test_records_are_frozen():
    answer = Answer(
        key="k",
        engine="openai",
        mode="api",
        sample=1,
        status="success",
        text="t",
        model_version="m",
        latency_ms=1,
        cost_usd=Decimal("0"),
        citations=(),
        search_count=None,
        search_queries=None,
        error_type=None,
        error=None,
    )
    with pytest.raises(AttributeError):
        answer.text = "u"  # type: ignore[misc]
    assert Judged(tools=None, ungrounded=(), cost_usd=Decimal(0)).tools is None
    assert RunInfo("2026-10-21", "completed", 5, "search").extra == {}
