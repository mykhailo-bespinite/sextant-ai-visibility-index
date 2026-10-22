"""Engines, modes, answer statuses and the method's fixed limits.

These values define the index method, so they match the published edition
exactly: the engine roster and its order, the display labels, the two ways
an engine is asked, the six statuses an answer can have, and the caps that
keep one answer or one vendor from outweighing the others.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

# The engine roster, in the order every table lists it.
ENGINES: Final[tuple[str, ...]] = ("anthropic", "openai", "perplexity", "gemini", "grok", "google_aio")

ENGINE_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "anthropic": "Claude",
        "openai": "ChatGPT",
        "perplexity": "Perplexity",
        "gemini": "Gemini",
        "grok": "Grok",
        "google_aio": "Google AI Overview",
    }
)

# Engines whose answer is a scraped search results page rather than a model
# answering a question. Every label of them says so.
RESULTS_PAGE_ENGINES: Final[tuple[str, ...]] = ("google_aio",)
RESULTS_PAGE_NOTE: Final[str] = "results page via DataForSEO"

# `api` is the vendor API's training knowledge alone; `search` attaches the
# vendor's web search tool, which is what a person sees in the consumer app.
MODE_API: Final[str] = "api"
MODE_SEARCH: Final[str] = "search"
MODES: Final[tuple[str, ...]] = (MODE_API, MODE_SEARCH)
# The order tables list the two rows in: the web search row first.
DISPLAY_MODES: Final[tuple[str, ...]] = (MODE_SEARCH, MODE_API)
# What `--mode` accepts.
MODE_BOTH: Final[str] = "both"
REQUESTABLE_MODES: Final[tuple[str, ...]] = (MODE_API, MODE_SEARCH, MODE_BOTH)

# Engines with a vendor web search tool are asked in either mode. Retrieval
# engines search on every call, so they are asked in `search` only.
SEARCH_ENGINES: Final[tuple[str, ...]] = ("openai", "anthropic", "gemini", "grok")
RETRIEVAL_ENGINES: Final[tuple[str, ...]] = ("perplexity", "google_aio")

STATUS_SUCCESS: Final[str] = "success"
STATUS_EMPTY: Final[str] = "empty"
STATUS_ERROR: Final[str] = "error"
STATUS_TIMEOUT: Final[str] = "timeout"
STATUS_RATE_LIMITED: Final[str] = "rate_limited"
STATUS_SKIPPED: Final[str] = "skipped"
STATUSES: Final[tuple[str, ...]] = (
    STATUS_SUCCESS,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_TIMEOUT,
    STATUS_RATE_LIMITED,
    STATUS_SKIPPED,
)

# Samples per (question, engine, mode) cell.
DEFAULT_SAMPLES: Final[int] = 3
MIN_SAMPLES: Final[int] = 1
MAX_SAMPLES: Final[int] = 5

# The longest one call may take before it is recorded as a `timeout`.
CALL_TIMEOUT_SECONDS: Final[Mapping[str, int]] = MappingProxyType({MODE_API: 90, MODE_SEARCH: 180})
AIO_TIMEOUT_SECONDS: Final[int] = 120

# Web search tool calls one answer may make (each one is billed).
MAX_SEARCH_TOOL_CALLS: Final[int] = 3
# Cited sources kept per answer, in the vendor's order.
MAX_ANSWER_CITATIONS: Final[int] = 10
# Search queries kept per answer, and the longest one kept.
MAX_SEARCH_QUERIES: Final[int] = 20
MAX_SEARCH_QUERY_CHARS: Final[int] = 500

# Rows the edition keeps in its top cited domains and URLs.
TOP_DOMAINS: Final[int] = 40
TOP_URLS: Final[int] = 60


def engine_label(engine: str) -> str:
    """The label every table shows for `engine`. A results-page engine says
    so in its label. Raises KeyError for an engine outside the roster."""
    label = ENGINE_LABELS[engine]
    if engine in RESULTS_PAGE_ENGINES:
        return f"{label}, {RESULTS_PAGE_NOTE}"
    return label


def engine_modes(engine: str, requested: str) -> tuple[str, ...]:
    """The modes `engine` is asked in when the run requested `requested`
    (`api`, `search` or `both`). Retrieval engines are always asked in
    `search` only; `both` gives the modes in `MODES` order."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected one of {', '.join(ENGINES)}")
    if requested not in REQUESTABLE_MODES:
        raise ValueError(f"unknown mode {requested!r}; expected one of {', '.join(REQUESTABLE_MODES)}")
    if engine in RETRIEVAL_ENGINES:
        return (MODE_SEARCH,)
    if requested == MODE_BOTH:
        return MODES
    return (requested,)


# A model id as the runner sends it: in a request body, and for Gemini in
# the URL path, so at most one '/' and never '..'.
MODEL_ID_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?")


def check_model(model: str) -> str:
    """`model` if it is a model id the runner will send; else ValueError."""
    if not MODEL_ID_RE.fullmatch(model) or ".." in model:
        raise ValueError(f"{model!r} is not a model id (letters, digits, '.', '_', '-', at most one '/')")
    return model
