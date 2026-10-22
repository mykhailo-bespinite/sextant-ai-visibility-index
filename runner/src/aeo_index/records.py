"""The records a run is made of.

`Answer` is one engine's reply to one cell (question, engine, mode,
sample), success or not. `Judged` is what the judge made of one answer.
`AnswerFacts` is the per-answer fact row an edition is built from, and
what `answers.jsonl` stores (it never holds the answer text). `RunInfo`
describes a run; `JudgeSpec` names the judge model.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final

from aeo_index.modes import ENGINES, MODES, STATUSES, check_model

# The engines that can judge: the ones with a plain chat API.
JUDGE_PROVIDERS: Final[tuple[str, ...]] = ("openai", "anthropic", "gemini", "grok")
_ISO_DATE: Final[re.Pattern[str]] = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class Answer:
    """One engine's reply to one cell. `error` is already redacted."""

    key: str
    engine: str
    mode: str
    sample: int
    status: str
    text: str | None
    model_version: str
    latency_ms: int
    cost_usd: Decimal
    citations: tuple[str, ...]
    search_count: int | None
    search_queries: tuple[str, ...] | None
    error_type: str | None
    error: str | None


@dataclass(frozen=True)
class Judged:
    """The tools the judge listed for one answer, in its order, after the
    grounding check; `tools=None` means the answer was not judged.
    `ungrounded` holds the names the check dropped."""

    tools: tuple[str, ...] | None
    ungrounded: tuple[str, ...]
    cost_usd: Decimal


@dataclass(frozen=True)
class AnswerFacts:
    """The facts of one answer that an edition is built from.

    `citations` are the provider's citation strings as given (a URL, or a
    bare domain), unfiltered; the edition builder filters them."""

    key: str
    engine: str
    mode: str
    sample: int
    run_date: str
    status: str
    tools_named: tuple[str, ...] | None
    first_named: str | None
    sextant_rank: int | None
    citations: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "key": self.key,
            "engine": self.engine,
            "mode": self.mode,
            "sample": self.sample,
            "run_date": self.run_date,
            "status": self.status,
            "tools_named": list(self.tools_named) if self.tools_named is not None else None,
            "first_named": self.first_named,
            "sextant_rank": self.sextant_rank,
            "citations": list(self.citations),
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, object]) -> AnswerFacts:
        """Read one fact row. Raises ValueError naming the field that is
        missing or of the wrong shape."""
        if not isinstance(obj, Mapping):
            raise ValueError("an answer fact row must be a JSON object")
        tools = _optional_names(obj, "tools_named")
        return cls(
            key=_text(obj, "key"),
            engine=_choice(obj, "engine", ENGINES),
            mode=_choice(obj, "mode", MODES),
            sample=_positive_int(obj, "sample"),
            run_date=_date(obj, "run_date"),
            status=_choice(obj, "status", STATUSES),
            tools_named=tools,
            first_named=_optional_text(obj, "first_named"),
            sextant_rank=_optional_positive_int(obj, "sextant_rank"),
            citations=_names(obj, "citations"),
        )


@dataclass(frozen=True)
class RunInfo:
    """A run: its date, its status (`completed`, `partial` or `failed`),
    how many engines answered, the mode its goal is read on, and anything
    else the edition records about it."""

    run_date: str
    status: str
    engines_answered: int
    goal_mode: str
    extra: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeSpec:
    """The judge: a provider that can judge and one of its models."""

    provider: str
    model: str

    @classmethod
    def parse(cls, value: str) -> JudgeSpec:
        """`PROVIDER:MODEL`, e.g. `anthropic:claude-haiku-4-5-20251001`."""
        provider, sep, model = value.partition(":")
        if not sep or provider not in JUDGE_PROVIDERS:
            raise ValueError(
                f"--judge must be PROVIDER:MODEL with PROVIDER one of {', '.join(JUDGE_PROVIDERS)}, got {value!r}"
            )
        try:
            return cls(provider, check_model(model))
        except ValueError as exc:
            raise ValueError(f"--judge: {exc}") from exc

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def _get(obj: Mapping[str, object], name: str) -> object:
    if name not in obj:
        raise ValueError(f"the field {name!r} is missing")
    return obj[name]


def _text(obj: Mapping[str, object], name: str) -> str:
    value = _get(obj, name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"the field {name!r} must be a non-empty string, got {value!r}")
    return value


def _optional_text(obj: Mapping[str, object], name: str) -> str | None:
    value = _get(obj, name)
    if value is None:
        return None
    return _text(obj, name)


def _choice(obj: Mapping[str, object], name: str, allowed: tuple[str, ...]) -> str:
    value = _text(obj, name)
    if value not in allowed:
        raise ValueError(f"the field {name!r} must be one of {', '.join(allowed)}, got {value!r}")
    return value


def _positive_int(obj: Mapping[str, object], name: str) -> int:
    value = _get(obj, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"the field {name!r} must be a whole number of 1 or more, got {value!r}")
    return value


def _optional_positive_int(obj: Mapping[str, object], name: str) -> int | None:
    if _get(obj, name) is None:
        return None
    return _positive_int(obj, name)


def _date(obj: Mapping[str, object], name: str) -> str:
    value = _text(obj, name)
    if not _ISO_DATE.fullmatch(value):
        raise ValueError(f"the field {name!r} must be a date as YYYY-MM-DD, got {value!r}")
    return value


def _names(obj: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = _get(obj, name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"the field {name!r} must be a list of strings, got {value!r}")
    return tuple(value)


def _optional_names(obj: Mapping[str, object], name: str) -> tuple[str, ...] | None:
    if _get(obj, name) is None:
        return None
    return _names(obj, name)
