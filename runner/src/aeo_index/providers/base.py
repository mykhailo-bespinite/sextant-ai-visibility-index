"""What every engine provider shares: the reply, the error, the protocol,
the default models, the output caps, and building providers from keys.

A provider speaks one vendor's REST API through the `httpx.AsyncClient`
it is handed, so a test can hand it one over `httpx.MockTransport`. It
never retries and never prices: `dispatch.ask_all` retries, and
`pricing.call_cost` prices from the tokens and searches the reply reports.
A provider raises `ProviderError` for every failure, with one of the
error types below.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Final, Protocol

import httpx

from aeo_index.modes import AIO_TIMEOUT_SECONDS, CALL_TIMEOUT_SECONDS, ENGINES, MODE_SEARCH
from aeo_index.modes import check_model as check_model  # re-exported for the providers

ERROR_TYPES: Final[tuple[str, ...]] = ("auth", "rate_limited", "server", "timeout", "empty", "bad_response", "client")

DEFAULT_MODELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "openai": "gpt-5.5",
        "anthropic": "claude-sonnet-4-6",
        "gemini": "models/gemini-3.5-flash",
        "grok": "grok-4.3",
        "perplexity": "sonar-pro",
        "google_aio": "google-ai-overview",
    }
)

# The variables each engine's key is read from, the first one preferred.
KEY_ENV_VARS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "grok": ("XAI_API_KEY", "GROK_API_KEY"),
        "perplexity": ("PERPLEXITY_API_KEY",),
    }
)
AIO_LOGIN_VAR: Final[str] = "DATAFORSEO_LOGIN"
AIO_PASSWORD_VAR: Final[str] = "DATAFORSEO_PASSWORD"

# The longest vendor error text kept in an error message.
VENDOR_ERROR_MAX_CHARS: Final[int] = 500

# Output caps (the most tokens one answer may generate). Reasoning models
# share the cap between thinking and the answer, so they get more room;
# only generated tokens are billed.
CLASSIC_MAX_TOKENS: Final[int] = 1024
REASONING_MAX_TOKENS: Final[int] = 8192
ANTHROPIC_SEARCH_MAX_TOKENS: Final[int] = 4096
DEFAULT_TEMPERATURE: Final[float] = 0.7
_REASONING_PREFIXES: Final[tuple[str, ...]] = ("gpt-5", "o1", "o3", "o4")


@dataclass(frozen=True)
class ProviderReply:
    """One successful answer as the vendor returned it.

    `citations`, `search_count` and `search_queries` are None when the
    vendor reported none (an answer without web search).
    `vendor_cost_usd` is the vendor's own billed figure when it reports
    one; `pricing.call_cost` prefers it over the rate card."""

    text: str
    model_version: str
    input_tokens: int
    output_tokens: int
    citations: tuple[str, ...] | None
    search_count: int | None
    search_queries: tuple[str, ...] | None
    vendor_cost_usd: Decimal | None


class ProviderError(Exception):
    """A call that gave no usable answer.

    `error_type` is one of `ERROR_TYPES`. `retry_after` is the seconds a
    429 or 5xx asked for. `cost_usd` is what the vendor billed for the
    failed call when it says so; `reply`, when set, holds the tokens and
    searches of a billed answer with no text, for `pricing.call_cost`."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        status_code: int | None = None,
        retry_after: float | None = None,
        cost_usd: Decimal = Decimal("0"),
        reply: ProviderReply | None = None,
    ) -> None:
        if error_type not in ERROR_TYPES:
            raise ValueError(f"unknown error type {error_type!r}")
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.retry_after = retry_after
        self.cost_usd = cost_usd
        self.reply = reply


class Provider(Protocol):
    """One engine. `env_var` names the variable its key came from."""

    engine: str
    model: str
    env_var: str

    async def ask(
        self, prompt: str, *, mode: str, client: httpx.AsyncClient, max_output_tokens: int | None = None
    ) -> ProviderReply: ...


def is_reasoning_model(model: str) -> bool:
    """GPT-5 and the o-series: `max_completion_tokens`, no temperature."""
    return model.lower().startswith(_REASONING_PREFIXES)


def output_cap(engine: str, model: str, mode: str) -> int:
    """The output tokens one answer is allowed, as the provider sends it."""
    if engine == "openai":
        return REASONING_MAX_TOKENS if is_reasoning_model(model) else CLASSIC_MAX_TOKENS
    if engine == "anthropic":
        return ANTHROPIC_SEARCH_MAX_TOKENS if mode == MODE_SEARCH else CLASSIC_MAX_TOKENS
    if engine == "gemini":
        return REASONING_MAX_TOKENS
    if engine == "grok":
        return REASONING_MAX_TOKENS if mode == MODE_SEARCH else CLASSIC_MAX_TOKENS
    if engine == "perplexity":
        return CLASSIC_MAX_TOKENS
    if engine == "google_aio":
        return 0
    raise ValueError(f"unknown engine {engine!r}")


def call_timeout(engine: str, mode: str) -> float:
    """The deadline of one call, in seconds."""
    return float(AIO_TIMEOUT_SECONDS if engine == "google_aio" else CALL_TIMEOUT_SECONDS[mode])


# ---------------------------------------------------------------- parsing helpers


def as_dict(value: Any) -> dict[str, Any]:
    """`value` if it is a dict, else {}: an odd vendor shape loses that
    field, never the answer."""
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    """`value` if it is a list, else []."""
    return value if isinstance(value, list) else []


def as_int(value: Any) -> int:
    """`value` as a non-negative int; 0 when absent or not a number."""
    if isinstance(value, bool):
        return 0
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(number, 0)


def token_sum(usage: Mapping[str, Any], *keys: str) -> int:
    """The sum of a usage object's counts under `keys`."""
    return sum(as_int(usage.get(key)) for key in keys)


def usable_queries(values: Iterable[Any]) -> list[str]:
    """The non-blank string search queries in `values`, in order."""
    return [value for value in values if isinstance(value, str) and value.strip()]


def dedupe_http_urls(urls: Iterable[Any]) -> list[str]:
    """Order-preserving, de-duplicated http(s) URL strings; anything else
    is dropped."""
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        if isinstance(url, str) and url.startswith(("http://", "https://")) and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def positive_decimal(value: Any) -> Decimal | None:
    """`value` as a finite Decimal above zero, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = Decimal(str(value))
    except ArithmeticError:
        return None
    return number if number.is_finite() and number > 0 else None


def vendor_message(body: Any) -> str:
    """The vendor's error text only (never the whole body), truncated."""
    message: Any = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("status") or error.get("type")
        elif isinstance(error, str):
            message = error
        message = message or body.get("status_message") or body.get("message")
    return str(message or "no error message")[:VENDOR_ERROR_MAX_CHARS]


# ---------------------------------------------------------------- HTTP

_HTTP_ERROR_TYPES: Final[Mapping[int, str]] = MappingProxyType({401: "auth", 403: "auth", 429: "rate_limited"})


def retry_after_seconds(response: httpx.Response) -> float | None:
    """A numeric `Retry-After` header, in seconds."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


# A vendor's test for "this 400 means the key was rejected", given the body.
BadKeyTest = Callable[[Any], bool]


def error_for_status(
    response: httpx.Response, label: str, body: Any, bad_key: BadKeyTest | None = None
) -> ProviderError:
    """The ProviderError for a 4xx or 5xx reply. `bad_key` lets a vendor
    that rejects a key with HTTP 400 have it counted as `auth` (the status
    code is kept)."""
    code = response.status_code
    error_type = _HTTP_ERROR_TYPES.get(code) or ("server" if code >= 500 else "client")
    if code == 400 and bad_key is not None and bad_key(body):
        error_type = "auth"
    retry_after = retry_after_seconds(response) if error_type in ("rate_limited", "server") else None
    return ProviderError(
        f"{label} answered HTTP {code}: {vendor_message(body)}",
        error_type=error_type,
        status_code=code,
        retry_after=retry_after,
    )


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    label: str,
    payload: object,
    headers: Mapping[str, str] | None = None,
    auth: httpx.Auth | None = None,
    timeout: float,
    bad_key: BadKeyTest | None = None,
) -> dict[str, Any]:
    """POST `payload` as JSON and return the reply object. Every failure
    raises ProviderError: a transport timeout is `timeout`, another
    transport failure `server`, a 4xx or 5xx by its status, a body that
    cannot be decoded or is not a JSON object `bad_response`. The URL is never in a message."""
    try:
        response = await client.post(url, json=payload, headers=dict(headers or {}), auth=auth, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise ProviderError(f"{label} did not answer in time", error_type="timeout") from exc
    except httpx.TransportError as exc:
        raise ProviderError(f"{label} could not be reached ({exc.__class__.__name__})", error_type="server") from exc
    except httpx.HTTPError as exc:
        # A body the client could not decode, and any other failure of the
        # exchange: the class name only, never the URL or the body.
        raise ProviderError(
            f"{label} sent a reply that could not be read ({exc.__class__.__name__})", error_type="bad_response"
        ) from exc
    body = _json_or_none(response)
    if response.status_code >= 400:
        raise error_for_status(response, label, body, bad_key)
    if not isinstance(body, dict):
        raise ProviderError(f"{label} answered with a body that is not a JSON object", error_type="bad_response")
    return body


def is_unreadable(error: ProviderError) -> bool:
    """A reply arrived but could not be read (decoded, parsed, or turned into
    an answer), and the vendor said nothing of its cost: it may still have
    been billed, so the budget charges its ceiling, as for a timeout."""
    return error.error_type == "bad_response" and error.reply is None and error.cost_usd == 0


def _json_or_none(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None


def empty_error(
    label: str, detail: str, reply: ProviderReply | None = None, cost_usd: Decimal = Decimal("0")
) -> ProviderError:
    """The error for an answer that arrived with no text."""
    return ProviderError(
        f"{label} returned no answer text ({detail})", error_type="empty", reply=reply, cost_usd=cost_usd
    )


# ---------------------------------------------------------------- building


def _key(env: Mapping[str, str], names: Sequence[str]) -> tuple[str, str] | None:
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            return name, value
    return None


def build_providers(
    env: Mapping[str, str], engines: Sequence[str], models: Mapping[str, str]
) -> tuple[dict[str, Provider], dict[str, str]]:
    """A provider for each engine in `engines` that has its key in `env`,
    asking `models[engine]` (default: `DEFAULT_MODELS`). The second value
    says, per engine left out, what to set (`"set XAI_API_KEY"`)."""
    from aeo_index.providers.anthropic import AnthropicProvider
    from aeo_index.providers.gemini import GeminiProvider
    from aeo_index.providers.google_aio import GoogleAioProvider
    from aeo_index.providers.grok import GrokProvider
    from aeo_index.providers.openai import OpenAIProvider
    from aeo_index.providers.perplexity import PerplexityProvider

    keyed = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "grok": GrokProvider,
        "perplexity": PerplexityProvider,
    }
    providers: dict[str, Provider] = {}
    missing: dict[str, str] = {}
    for engine in engines:
        if engine not in ENGINES:
            raise ValueError(f"unknown engine {engine!r}; expected one of {', '.join(ENGINES)}")
        model = models.get(engine, DEFAULT_MODELS[engine])
        if engine == "google_aio":
            login = _key(env, (AIO_LOGIN_VAR,))
            password = _key(env, (AIO_PASSWORD_VAR,))
            if login is None or password is None:
                missing[engine] = f"set {AIO_LOGIN_VAR} and {AIO_PASSWORD_VAR}"
                continue
            providers[engine] = GoogleAioProvider(login[1], password[1], model)
            continue
        found = _key(env, KEY_ENV_VARS[engine])
        if found is None:
            missing[engine] = f"set {KEY_ENV_VARS[engine][0]}"
            continue
        env_var, key = found
        providers[engine] = keyed[engine](key, model, env_var=env_var)
    return providers, missing
