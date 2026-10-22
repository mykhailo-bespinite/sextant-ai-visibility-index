"""What a run costs: the rate card, the price of one call, and the estimate
printed before the first call.

Rates are USD per 1,000 tokens, as the vendors quote them. A dated model
snapshot (`gpt-5.5-2026-04-23`, `claude-haiku-4-5-20251001`: the base name
and a real date) is priced as its base key; any other longer name
(`gpt-5.55`, `gpt-5.5-pro`) is another model and is not priced. A model the card does not know has no price,
and a run that would ask it is refused unless the user gives one
(`--price`), because a budget cannot hold over an unknown price.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from aeo_index.modes import ENGINES, MAX_SEARCH_TOOL_CALLS, MODE_SEARCH, MODES
from aeo_index.providers.base import DEFAULT_MODELS, ProviderReply, output_cap
from aeo_index.providers.google_aio import FALLBACK_COST_USD as AIO_FALLBACK_COST_USD
from aeo_index.records import JudgeSpec

if TYPE_CHECKING:
    from aeo_index.dispatch import Cell

_PER: Final[Decimal] = Decimal(1000)
_QUANTUM: Final[Decimal] = Decimal("0.000001")
_MODEL_PREFIX: Final[str] = "models/"

# (engine, base model) -> (input, output) USD per 1K tokens, read from the
# vendors' price pages. A results page has no tokens: its cost is the
# vendor's own per-task figure.
_DEFAULT_RATES: Final[Mapping[tuple[str, str], tuple[Decimal, Decimal]]] = MappingProxyType(
    {
        ("anthropic", "claude-sonnet-4-6"): (Decimal("0.003"), Decimal("0.015")),
        ("anthropic", "claude-haiku-4-5"): (Decimal("0.001"), Decimal("0.005")),
        ("openai", "gpt-5.5"): (Decimal("0.005"), Decimal("0.030")),
        ("openai", "gpt-4o-mini"): (Decimal("0.00015"), Decimal("0.0006")),
        ("gemini", "gemini-3.5-flash"): (Decimal("0.0015"), Decimal("0.009")),
        ("grok", "grok-4.3"): (Decimal("0.00125"), Decimal("0.0025")),
        ("perplexity", "sonar-pro"): (Decimal("0.003"), Decimal("0.015")),
        ("google_aio", "google-ai-overview"): (Decimal("0"), Decimal("0")),
    }
)

# Web search tool fees, USD per 1,000 searches, billed on top of tokens in
# search mode. Perplexity's search is its request fee below.
SEARCH_FEES_PER_1K: Final[Mapping[str, Decimal]] = MappingProxyType(
    {
        "openai": Decimal("10.00"),
        "anthropic": Decimal("10.00"),
        "gemini": Decimal("14.00"),
        "grok": Decimal("5.00"),
    }
)
# Per-request fees, USD per 1,000 requests, billed on top of tokens on every
# call. sonar-pro's fee is $6 to $14 by search context tier; $6 is the floor
# (the vendor's own reported cost, preferred when present, carries the tier).
REQUEST_FEES_PER_1K: Final[Mapping[tuple[str, str], Decimal]] = MappingProxyType(
    {("perplexity", "sonar-pro"): Decimal("6.00"), ("perplexity", "sonar"): Decimal("5.00")}
)

# The judge's high figure per answer: an answer of up to this many tokens
# in, and the judge's output cap out.
JUDGE_HIGH_INPUT_TOKENS: Final[int] = 4000
JUDGE_OUTPUT_TOKENS: Final[int] = 500
# Input tokens assumed for one call of a model with no typical figure.
UNKNOWN_MODEL_INPUT_TOKENS: Final[int] = 2000


@dataclass(frozen=True)
class RateCard:
    """(engine, model) -> (input, output) USD per 1K tokens. Immutable:
    `with_override` returns a new card."""

    rates: Mapping[tuple[str, str], tuple[Decimal, Decimal]] = field(default_factory=lambda: _DEFAULT_RATES)

    __hash__ = None  # type: ignore[assignment]  # a mapping is not hashable

    @classmethod
    def default(cls) -> RateCard:
        return cls(_DEFAULT_RATES)

    def with_override(self, engine: str, model: str, input_per_1k: Decimal, output_per_1k: Decimal) -> RateCard:
        """This card with `model` priced at the given rates (an exact match
        wins over a base key, so an override of a base key also prices its
        dated snapshots)."""
        for value in (input_per_1k, output_per_1k):
            if not value.is_finite() or value < 0:
                raise ValueError(f"a price must be a number of 0 or more, got {value}")
        return RateCard(MappingProxyType({**self.rates, (engine, _bare(model)): (input_per_1k, output_per_1k)}))

    def rate(self, engine: str, model: str) -> tuple[Decimal, Decimal] | None:
        base = _resolve(engine, _bare(model), self.rates)
        return self.rates[(engine, base)] if base is not None else None


def _bare(model: str) -> str:
    return model[len(_MODEL_PREFIX) :] if model.startswith(_MODEL_PREFIX) else model


# A dated snapshot of a model: the model's name, then "-YYYY-MM-DD" or
# "-YYYYMMDD" (gpt-5.5-2026-04-23, claude-haiku-4-5-20251001). Any other
# suffix names another model (gpt-5.5-pro), which has its own price.
_SNAPSHOT_SUFFIX: Final[re.Pattern[str]] = re.compile(r"-(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{8})")


def _is_date(suffix: str) -> bool:
    digits = suffix.replace("-", "")
    try:
        date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    except ValueError:
        return False
    return True


def _resolve(engine: str, model: str, keys: Mapping[tuple[str, str], object]) -> str | None:
    """The exact key, else the base key `b` of which `model` is a dated
    snapshot (`b` followed by a real date). Nothing else resolves, so a
    variant of a priced model stays unpriced and needs `--price`."""
    if (engine, model) in keys:
        return model
    for e, base in keys:
        suffix = model[len(base) :]
        if e == engine and model.startswith(base) and _SNAPSHOT_SUFFIX.fullmatch(suffix) and _is_date(suffix):
            return base
    return None


def request_fee_per_1k(engine: str, model: str) -> Decimal:
    base = _resolve(engine, _bare(model), REQUEST_FEES_PER_1K)
    return REQUEST_FEES_PER_1K[(engine, base)] if base is not None else Decimal("0")


def call_cost(engine: str, model: str, mode: str, reply: ProviderReply, card: RateCard) -> Decimal:
    """The cost of one call, to six decimals: the vendor's own billed
    figure when it reports one; otherwise the tokens at the card's rate,
    plus the search fee for each search in search mode, plus the request
    fee. Raises ValueError for a model the card cannot price."""
    if reply.vendor_cost_usd is not None:
        return reply.vendor_cost_usd.quantize(_QUANTUM)
    rate = card.rate(engine, model)
    if rate is None:
        raise ValueError(f"no price for {engine}:{model}")
    input_rate, output_rate = rate
    cost = Decimal(reply.input_tokens) * input_rate / _PER + Decimal(reply.output_tokens) * output_rate / _PER
    if mode == MODE_SEARCH and reply.search_count:
        cost += Decimal(reply.search_count) * SEARCH_FEES_PER_1K.get(engine, Decimal("0")) / _PER
    cost += request_fee_per_1k(engine, model) / _PER
    return cost.quantize(_QUANTUM)


def ceiling_call_usd(engine: str, model: str, mode: str, card: RateCard) -> Decimal:
    """The most one call of a model with no typical figure is expected to
    cost: its whole output cap, `UNKNOWN_MODEL_INPUT_TOKENS` in, every
    search the tool cap allows in search mode (Gemini's search has no cap;
    its ceiling assumes the same three searches), and the request fee. A
    results page has no tokens; its ceiling is the per-task ceiling."""
    rate = card.rate(engine, model)
    if rate is None:
        raise ValueError(f"no price for {engine}:{model}")
    if engine == "google_aio":
        return AIO_FALLBACK_COST_USD
    input_rate, output_rate = rate
    cost = Decimal(output_cap(engine, model, mode)) * output_rate / _PER
    cost += Decimal(UNKNOWN_MODEL_INPUT_TOKENS) * input_rate / _PER
    if mode == MODE_SEARCH:
        cost += MAX_SEARCH_TOOL_CALLS * SEARCH_FEES_PER_1K.get(engine, Decimal("0")) / _PER
    return cost + request_fee_per_1k(engine, model) / _PER


# (engine, mode) -> (median, 95th percentile) USD per call of the default
# model: successful index answers over the 7 days to TYPICAL_AS_OF, 60 per
# row, rounded up to four places. The models that answered were the
# defaults (OpenAI's dated snapshot gpt-5.5-2026-04-23 of gpt-5.5); the
# results page row is DataForSEO's billed cost per task.
TYPICAL_CALL_USD: Final[Mapping[tuple[str, str], tuple[Decimal, Decimal]]] = MappingProxyType(
    {
        ("anthropic", "api"): (Decimal("0.0056"), Decimal("0.0119")),
        ("anthropic", "search"): (Decimal("0.0693"), Decimal("0.1288")),
        ("openai", "api"): (Decimal("0.0332"), Decimal("0.0597")),
        ("openai", "search"): (Decimal("0.2003"), Decimal("0.2336")),
        ("perplexity", "search"): (Decimal("0.0040"), Decimal("0.0053")),
        ("gemini", "api"): (Decimal("0.0106"), Decimal("0.0128")),
        ("gemini", "search"): (Decimal("0.0573"), Decimal("0.0819")),
        ("grok", "api"): (Decimal("0.0035"), Decimal("0.0042")),
        ("grok", "search"): (Decimal("0.0447"), Decimal("0.0526")),
        ("google_aio", "search"): (Decimal("0.0012"), Decimal("0.0012")),
    }
)
TYPICAL_AS_OF: Final[str] = "2026-10-21"


@dataclass(frozen=True)
class Estimate:
    """The cost of a planned run. `rows` are (engine, mode, cells,
    typical, high); `typical` and `high` include the judge's high figure."""

    rows: tuple[tuple[str, str, int, Decimal, Decimal], ...]
    judge_high: Decimal
    typical: Decimal
    high: Decimal


def judge_call_high(judge: JudgeSpec, card: RateCard) -> Decimal:
    rate = card.rate(judge.provider, judge.model)
    if rate is None:
        raise ValueError(f"no price for {judge}")
    input_rate, output_rate = rate
    return Decimal(JUDGE_HIGH_INPUT_TOKENS) * input_rate / _PER + Decimal(JUDGE_OUTPUT_TOKENS) * output_rate / _PER


def estimate(cells: Sequence[Cell], models: Mapping[str, str], judge: JudgeSpec, card: RateCard) -> Estimate:
    """Price `cells` before asking. A default model uses its typical
    figures; any other model its ceiling for both figures. The judge's
    high figure covers one judged answer per cell."""
    counts: dict[tuple[str, str], int] = {}
    for cell in cells:
        counts[(cell.engine, cell.mode)] = counts.get((cell.engine, cell.mode), 0) + 1
    rows: list[tuple[str, str, int, Decimal, Decimal]] = []
    for engine in ENGINES:
        model = models.get(engine, DEFAULT_MODELS[engine])
        for mode in MODES:
            n = counts.get((engine, mode), 0)
            if not n:
                continue
            if model == DEFAULT_MODELS[engine] and (engine, mode) in TYPICAL_CALL_USD:
                median, p95 = TYPICAL_CALL_USD[(engine, mode)]
            else:
                median = p95 = ceiling_call_usd(engine, model, mode, card)
            rows.append((engine, mode, n, median * n, p95 * n))
    judge_high = judge_call_high(judge, card) * len(cells)
    return Estimate(
        rows=tuple(rows),
        judge_high=judge_high,
        typical=sum((row[3] for row in rows), Decimal("0")) + judge_high,
        high=sum((row[4] for row in rows), Decimal("0")) + judge_high,
    )


def unpriced(models: Mapping[str, str], judge: JudgeSpec, card: RateCard) -> list[str]:
    """`engine:model` for each model in `models`, and the judge, that the
    card cannot price, in that order."""
    missing = [f"{engine}:{model}" for engine, model in models.items() if card.rate(engine, model) is None]
    if card.rate(judge.provider, judge.model) is None:
        missing.append(str(judge))
    return missing
