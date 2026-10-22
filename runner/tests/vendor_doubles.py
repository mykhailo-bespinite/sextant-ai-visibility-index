"""Test doubles for the vendor APIs: every response comes from a recorded
fixture in `aeo_index/fixtures/vendors/`, served through
`httpx.MockTransport`, so no test can reach a vendor."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from decimal import Decimal

import httpx

from aeo_index.dispatch import Budget, Cell, ask_all, plan_cells
from aeo_index.fixtures import fixture_response, load_vendor_fixture
from aeo_index.pricing import RateCard
from aeo_index.providers.base import DEFAULT_MODELS, build_providers
from aeo_index.questions import default_questions
from aeo_index.records import Answer
from aeo_index.redact import Redactor

# One fake credential per engine. None is a real key or key shape a vendor
# would accept.
FAKE_ENV: dict[str, str] = {
    "OPENAI_API_KEY": "k-test-123456",
    "ANTHROPIC_API_KEY": "k-test-anthropic-1",
    "GEMINI_API_KEY": "AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE",
    "XAI_API_KEY": "k-test-grok-12345",
    "PERPLEXITY_API_KEY": "k-test-pplx-12345",
    "DATAFORSEO_LOGIN": "login-test-1",
    "DATAFORSEO_PASSWORD": "pw-test-123456",
}

HOST_ENGINES: dict[str, str] = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "generativelanguage.googleapis.com": "gemini",
    "api.x.ai": "grok",
    "api.perplexity.ai": "perplexity",
    "api.dataforseo.com": "google_aio",
}


def engine_of(request: httpx.Request) -> str:
    return HOST_ENGINES[request.url.host]


def mode_of(request: httpx.Request) -> str:
    engine = engine_of(request)
    if engine in ("perplexity", "google_aio"):
        return "search"
    if engine in ("openai", "grok"):
        return "search" if request.url.path.endswith("/responses") else "api"
    return "search" if "tools" in json.loads(request.content) else "api"


def ok_fixture_for(request: httpx.Request) -> str:
    return f"{engine_of(request)}_{mode_of(request)}_ok.json"


class Recorder:
    """Serves one fixture and records what was sent."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.fixture: dict[str, object] = {}

    def client(self, name: str) -> httpx.AsyncClient:
        self.fixture = load_vendor_fixture(name)

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return fixture_response(self.fixture)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    @property
    def last_path(self) -> str:
        return self.last.url.path

    @property
    def last_url(self) -> httpx.URL:
        return self.last.url

    @property
    def last_headers(self) -> httpx.Headers:
        return self.last.headers

    @property
    def last_json(self) -> object:
        return json.loads(self.last.content)

    @property
    def expected(self) -> dict[str, object]:
        expected = self.fixture["expected"]
        assert isinstance(expected, dict)
        return expected

    @property
    def fixture_search_calls(self) -> object:
        return self.expected["search_count"]


class Scripted(httpx.MockTransport):
    """Replies per engine: a fixture name always, or a list served in order
    (the last one repeating). Counts the calls each engine received."""

    def __init__(self, script: Mapping[str, str | Sequence[str]], *, on_request: Callable[[], None] | None = None):
        self.script = {engine: [names] if isinstance(names, str) else list(names) for engine, names in script.items()}
        self.calls: Counter[str] = Counter()
        self.on_request = on_request
        super().__init__(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        engine = engine_of(request)
        self.calls[engine] += 1
        if self.on_request is not None:
            self.on_request()
        names = self.script[engine]
        name = names.pop(0) if len(names) > 1 else names[0]
        return fixture_response(load_vendor_fixture(name))


class ConcurrencyCounter(httpx.MockTransport):
    """Serves each engine's success body after `delay` seconds and records
    the most requests in flight at once, in total and per engine."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.in_flight: Counter[str] = Counter()
        self.max_per_engine: Counter[str] = Counter()
        self.total = 0
        self.max_total = 0
        super().__init__(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        engine = engine_of(request)
        self.total += 1
        self.in_flight[engine] += 1
        self.max_total = max(self.max_total, self.total)
        self.max_per_engine[engine] = max(self.max_per_engine[engine], self.in_flight[engine])
        try:
            await asyncio.sleep(self.delay)
            return fixture_response(load_vendor_fixture(ok_fixture_for(request)))
        finally:
            self.total -= 1
            self.in_flight[engine] -= 1


class Barrier(httpx.MockTransport):
    """Serves `name`. The first request is answered at once; later ones are
    held until `hold_until` of them are waiting, then all are released."""

    def __init__(self, name: str, *, hold_until: int) -> None:
        self.name = name
        self.hold_until = hold_until
        self.calls = 0
        self.gate = asyncio.Event()
        super().__init__(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls > 1:
            if self.calls - 1 >= self.hold_until:
                self.gate.set()
            await self.gate.wait()
        return fixture_response(load_vendor_fixture(self.name))


def cells(*, engines: Sequence[str], questions: int, samples: int, mode: str = "api") -> list[Cell]:
    qset = default_questions()
    qset = replace(qset, questions=qset.questions[:questions])
    return plan_cells(qset, engines, mode, {}, samples)


def recording_sleep(sleeps: list[float]) -> Callable[[float], Awaitable[None]]:
    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return _sleep


async def instant_sleep(seconds: float) -> None:
    return None


async def run_cells(
    cell_list: Sequence[Cell],
    transport: httpx.AsyncBaseTransport,
    *,
    sleep: Callable[[float], Awaitable[None]] = instant_sleep,
    budget: Budget | None = None,
    clock: Callable[[], float] | None = None,
    on_progress=None,
    env: Mapping[str, str] = FAKE_ENV,
) -> list[Answer]:
    engines = sorted({cell.engine for cell in cell_list})
    providers, missing = build_providers(env, engines, DEFAULT_MODELS)
    assert missing == {}
    kwargs = {} if clock is None else {"clock": clock}
    async with httpx.AsyncClient(transport=transport) as client:
        return await ask_all(
            cell_list,
            providers,
            budget=budget if budget is not None else Budget(Decimal("100")),
            card=RateCard.default(),
            redactor=Redactor(env.values()),
            client=client,
            on_progress=on_progress,
            sleep=sleep,
            **kwargs,
        )
