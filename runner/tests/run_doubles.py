"""Builders for the records a run is made of (answers, facts) and a judge
double that answers like the Anthropic Messages API, so the judge, the
edition, the tables and the CLI can be tested without a vendor."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal

import httpx

from aeo_index.fixtures import fixture_response, load_vendor_fixture
from aeo_index.questions import Question
from aeo_index.records import Answer, AnswerFacts
from vendor_doubles import engine_of, ok_fixture_for

JUDGE_MODEL = "claude-haiku-4-5-20251001"


def answer(
    key: str = "q01",
    engine: str = "anthropic",
    mode: str = "api",
    sample: int = 1,
    *,
    status: str = "success",
    text: str | None = "Profound leads. Peec AI is second.",
    citations: Sequence[str] = (),
    cost: str = "0.01",
) -> Answer:
    ok = status == "success"
    return Answer(
        key=key,
        engine=engine,
        mode=mode,
        sample=sample,
        status=status,
        text=text if ok else None,
        model_version="model-x",
        latency_ms=10,
        cost_usd=Decimal(cost),
        citations=tuple(citations),
        search_count=None,
        search_queries=None,
        error_type=None if ok else "server",
        error=None if ok else "the vendor answered HTTP 503",
    )


def fact(
    key: str = "q01",
    engine: str = "anthropic",
    mode: str = "api",
    sample: int = 1,
    *,
    status: str = "success",
    tools: Sequence[str] | None = ("Profound",),
    rank: int | None = None,
    citations: Sequence[str] = (),
    run_date: str = "2026-10-21",
) -> AnswerFacts:
    ok = status == "success"
    named = tuple(tools) if ok and tools is not None else None
    return AnswerFacts(
        key=key,
        engine=engine,
        mode=mode,
        sample=sample,
        run_date=run_date,
        status=status,
        tools_named=named,
        first_named=named[0] if named else None,
        sextant_rank=rank if ok else None,
        citations=tuple(citations),
    )


def questions(*keys: str, experimental: Iterable[str] = ()) -> list[Question]:
    exp = set(experimental)
    return [Question(key, f"Which tool is best for job {key}?", key in exp) for key in keys]


def anthropic_text_body(text: str, *, model: str = JUDGE_MODEL, input_tokens: int = 300, output_tokens: int = 20):
    return {
        "id": "msg_judge",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def judge_reply(tools: Sequence[str]) -> httpx.Response:
    return httpx.Response(200, json=anthropic_text_body(json.dumps({"tools": list(tools)})))


def is_judge_request(request: httpx.Request) -> bool:
    if request.url.host != "api.anthropic.com":
        return False
    return json.loads(request.content).get("model") == JUDGE_MODEL


def answer_in(request: httpx.Request) -> str:
    """The answer text a judge request wraps."""
    prompt = json.loads(request.content)["messages"][0]["content"]
    return prompt.split("<answer>\n", 1)[1].split("\n</answer>", 1)[0]


class RunTransport(httpx.MockTransport):
    """A whole run: each engine answers with its recorded success body (or
    the fixture `script` names for it), and the judge lists the tools
    `judge` returns for the answer text (default: Profound, Peec AI)."""

    def __init__(
        self,
        script: Mapping[str, str] | None = None,
        *,
        judge: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self.script = dict(script or {})
        self.judge = judge or (lambda text: ["Profound", "Peec AI"])
        self.calls: list[str] = []
        self.judge_calls = 0
        super().__init__(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if is_judge_request(request):
            self.judge_calls += 1
            return judge_reply(self.judge(answer_in(request)))
        engine = engine_of(request)
        self.calls.append(engine)
        name = self.script.get(engine) or ok_fixture_for(request)
        return fixture_response(load_vendor_fixture(name))


SAMPLE_QUESTIONS = ("q01", "q02")
SAMPLE_ENGINES = {"api": ("anthropic", "openai"), "search": ("anthropic", "openai", "perplexity")}


def sample_facts() -> list[AnswerFacts]:
    """2 questions x 3 engines x both modes (perplexity searches only) x 3
    samples, mixed statuses: openai's api sample 3 fails, perplexity's q02
    sample 2 times out, anthropic search names Acme CRM second on q01."""
    rows: list[AnswerFacts] = []
    for key in SAMPLE_QUESTIONS:
        for mode in ("api", "search"):
            for engine in SAMPLE_ENGINES[mode]:
                for sample in (1, 2, 3):
                    status = "success"
                    if engine == "openai" and mode == "api" and sample == 3:
                        status = "error"
                    if engine == "perplexity" and key == "q02" and sample == 2:
                        status = "timeout"
                    tools: tuple[str, ...] = ("Profound", "Peec AI")
                    rank = None
                    if engine == "openai":
                        tools = ("Peec AI",) if sample == 1 else ()
                    if engine == "anthropic" and mode == "search" and key == "q01" and sample == 1:
                        tools, rank = ("Profound", "Acme CRM"), 2
                    cites = (
                        ("https://example.com/reviews/aeo-tools", "https://acmecrm.com/blog", "example.com")
                        if mode == "search" and status == "success"
                        else ()
                    )
                    rows.append(fact(key, engine, mode, sample, status=status, tools=tools, rank=rank, citations=cites))
    return rows


def asked_of(facts: Iterable[AnswerFacts]) -> dict[str, list[str]]:
    from aeo_index.modes import ENGINES, MODES

    seen = {(f.mode, f.engine) for f in facts}
    return {mode: [e for e in ENGINES if (mode, e) in seen] for mode in MODES}


def build(
    facts: Sequence[AnswerFacts], qs: Sequence[Question] | None = None, *, brand_domain: str | None = "acmecrm.com"
):
    from aeo_index.edition import build_edition, run_info

    facts = list(facts)
    qs = list(qs) if qs is not None else questions(*sorted({f.key for f in facts}))
    asked = asked_of(facts)
    dates = sorted({f.run_date for f in facts})
    runs = [run_info(facts, asked, run_date=d) for d in dates]
    return build_edition(
        facts,
        qs,
        runs,
        asked=asked,
        edition=dates[-1],
        generated_at="2026-10-21T12:00:00+00:00",
        brand_domain=brand_domain,
    )
