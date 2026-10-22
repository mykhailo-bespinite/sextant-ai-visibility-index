"""Google AI Overviews, read off a live Google results page through
DataForSEO (`POST /v3/serp/google/organic/live/advanced`, Basic auth).

This is a scraped results page, not a model answering: every label says
so. The page is fetched as a US desktop search (location 2840, English),
with the AI Overview loaded. The answer is the AI Overview's markdown (its
elements' text when it has none); the citations are its references, minus
unresolved `google.com/goto` redirects, de-duplicated by URL. A page with
no AI Overview is `empty`, and still billed. The cost is the task's own
`cost` field.

DataForSEO wraps every reply twice: the envelope and the task each carry
a `status_code`, and both must be 20000 with a non-null result.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Final
from urllib.parse import urlsplit

import httpx

from aeo_index.providers.base import (
    ProviderError,
    ProviderReply,
    as_dict,
    as_list,
    call_timeout,
    empty_error,
    positive_decimal,
    post_json,
)

DATAFORSEO_SERP_URL: Final[str] = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
US_LOCATION_CODE: Final[int] = 2840
LANGUAGE_CODE: Final[str] = "en"
SERP_DEPTH: Final[int] = 10
# The live advanced SERP price plus the AI Overview surcharge: used only
# when a reply carries no `cost`, as an upper bound, never an expected value.
FALLBACK_COST_USD: Final[Decimal] = Decimal("0.0026")
_SUCCESS: Final[int] = 20000
_AUTH_CODES: Final[frozenset[int]] = frozenset({40100, 40104})
_RATE_LIMIT_CODES: Final[frozenset[int]] = frozenset({40202, 40209})
_GOTO_HOSTS: Final[frozenset[str]] = frozenset({"google.com", "www.google.com"})
_LABEL: Final[str] = "DataForSEO"


def _task_error_type(code: int) -> str:
    if code in _AUTH_CODES:
        return "auth"
    if code in _RATE_LIMIT_CODES:
        return "rate_limited"
    # 40101 and 40103 are the search engine failing; 5xxxx are DataForSEO's own.
    if code in (40101, 40103) or code >= 50000:
        return "server"
    return "client"


def _status(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _unwrap(envelope: dict[str, Any]) -> tuple[dict[str, Any], Decimal | None]:
    """The first task and its billed cost; ProviderError on any failure."""
    tasks = as_list(envelope.get("tasks"))
    task = as_dict(tasks[0]) if tasks else {}
    cost = positive_decimal(task.get("cost"))
    for level in (envelope, task):
        code = _status(level.get("status_code"))
        if code != _SUCCESS:
            message = str(level.get("status_message") or "no status message")[:200]
            raise ProviderError(
                f"{_LABEL} task failed: {code} {message}",
                error_type=_task_error_type(code),
                status_code=code or None,
                cost_usd=cost or Decimal("0"),
            )
    result = as_list(task.get("result"))
    if not result or not isinstance(result[0], dict):
        raise ProviderError(f"{_LABEL} task has no result", error_type="bad_response", cost_usd=cost or Decimal("0"))
    return result[0], cost


def _overview(page: dict[str, Any]) -> dict[str, Any] | None:
    if "ai_overview" not in as_list(page.get("item_types")):
        return None
    return next(
        (item for item in as_list(page.get("items")) if isinstance(item, dict) and item.get("type") == "ai_overview"),
        None,
    )


def _overview_text(item: dict[str, Any]) -> str:
    markdown = item.get("markdown")
    if isinstance(markdown, str) and markdown.strip():
        return markdown
    return "".join(
        sub["text"] for sub in as_list(item.get("items")) if isinstance(sub, dict) and isinstance(sub.get("text"), str)
    )


def _is_unresolved_goto(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return (parts.hostname or "").lower() in _GOTO_HOSTS and parts.path.rstrip("/") == "/goto"


def _references(item: dict[str, Any]) -> tuple[str, ...]:
    refs = [ref for ref in as_list(item.get("references")) if isinstance(ref, dict)]
    for element in as_list(item.get("items")):
        refs.extend(ref for ref in as_list(as_dict(element).get("references")) if isinstance(ref, dict))
    seen: set[str] = set()
    kept: list[str] = []
    for ref in refs:
        url = ref.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")) or url in seen:
            continue
        if _is_unresolved_goto(url):
            continue
        seen.add(url)
        kept.append(url)
    return tuple(kept)


class GoogleAioProvider:
    engine = "google_aio"
    env_var = "DATAFORSEO_LOGIN/DATAFORSEO_PASSWORD"

    def __init__(self, login: str, password: str, model: str = "google-ai-overview") -> None:
        self._auth = httpx.BasicAuth(login, password)
        self.model = model

    async def ask(
        self, prompt: str, *, mode: str, client: httpx.AsyncClient, max_output_tokens: int | None = None
    ) -> ProviderReply:
        """`mode` and `max_output_tokens` change nothing: a results page is
        always a search, and has no output tokens."""
        task = {
            "keyword": prompt,
            "location_code": US_LOCATION_CODE,
            "language_code": LANGUAGE_CODE,
            "device": "desktop",
            "depth": SERP_DEPTH,
            "load_async_ai_overview": True,
        }
        envelope = await post_json(
            client,
            DATAFORSEO_SERP_URL,
            label=_LABEL,
            payload=[task],
            auth=self._auth,
            timeout=call_timeout(self.engine, mode),
        )
        page, cost = _unwrap(envelope)
        billed = cost if cost is not None else FALLBACK_COST_USD
        overview = _overview(page)
        if overview is None:
            raise empty_error(_LABEL, "Google showed no AI Overview for this search", cost_usd=billed)
        text = _overview_text(overview)
        if not text.strip():
            raise empty_error(_LABEL, "the AI Overview had no text", cost_usd=billed)
        return ProviderReply(
            text=text,
            model_version=self.model,
            input_tokens=0,
            output_tokens=0,
            citations=_references(overview),
            search_count=None,
            search_queries=None,
            vendor_cost_usd=billed,
        )
