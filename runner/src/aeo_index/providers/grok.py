"""Grok through the xAI REST API (OpenAI-compatible).

`api` mode is Chat Completions; reasoning tokens bill as output, so they
are added to the output count. `search` mode is xAI's Responses API with
the `web_search` tool, capped at `MAX_SEARCH_TOOL_CALLS`; its cost is
xAI's own billed figure (`usage.cost_in_usd_ticks`) when the reply has it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Final

import httpx

from aeo_index.modes import MAX_SEARCH_TOOL_CALLS, MODE_SEARCH
from aeo_index.providers.base import (
    DEFAULT_TEMPERATURE,
    ProviderReply,
    as_dict,
    as_list,
    call_timeout,
    check_model,
    dedupe_http_urls,
    output_cap,
    post_json,
)
from aeo_index.providers.openai import chat_completion_reply, checked_responses_reply
from aeo_index.providers.responses import parse_responses_body

XAI_API_BASE: Final[str] = "https://api.x.ai/v1"
WEB_SEARCH_TOOL: Final[dict[str, str]] = {"type": "web_search"}
# usage.cost_in_usd_ticks: 1 USD is 10^10 ticks.
_TICKS_PER_USD: Final[Decimal] = Decimal(10) ** 10
_QUANTUM: Final[Decimal] = Decimal("0.000001")
_LABEL: Final[str] = "xAI"


def _vendor_cost(usage: dict[str, Any]) -> Decimal | None:
    ticks = usage.get("cost_in_usd_ticks")
    if isinstance(ticks, bool) or not isinstance(ticks, (int, float)) or ticks <= 0:
        return None
    return (Decimal(str(ticks)) / _TICKS_PER_USD).quantize(_QUANTUM)


def _search_count(usage: dict[str, Any], fallback: int) -> int:
    reported = usage.get("num_server_side_tools_used")
    if isinstance(reported, int) and not isinstance(reported, bool) and reported >= 0:
        return reported
    return fallback


def _citations(reply: dict[str, Any], annotated: tuple[str, ...]) -> tuple[str, ...]:
    """Inline citations first, then the top-level `citations` list and the
    `sources` entries."""
    sources = (item.get("url") for item in as_list(reply.get("sources")) if isinstance(item, dict))
    return tuple(dedupe_http_urls([*annotated, *as_list(reply.get("citations")), *sources]))


def is_bad_key(body: Any) -> bool:
    """xAI rejects a key with HTTP 400 ("Client specified an invalid
    argument") and an `error` text saying the API key is incorrect or
    invalid."""
    error = as_dict(body).get("error")
    text = (error if isinstance(error, str) else str(as_dict(error).get("message") or "")).lower()
    return "api key" in text and ("incorrect" in text or "invalid" in text)


class GrokProvider:
    engine = "grok"

    def __init__(self, api_key: str, model: str, *, env_var: str = "XAI_API_KEY") -> None:
        self._api_key = api_key
        self.model = check_model(model)
        self.env_var = env_var

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def ask(
        self, prompt: str, *, mode: str, client: httpx.AsyncClient, max_output_tokens: int | None = None
    ) -> ProviderReply:
        cap = max_output_tokens if max_output_tokens is not None else output_cap(self.engine, self.model, mode)
        if mode == MODE_SEARCH:
            return await self._search(prompt, cap, client)
        reply = await post_json(
            client,
            f"{XAI_API_BASE}/chat/completions",
            label=_LABEL,
            payload={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": cap,
                "temperature": DEFAULT_TEMPERATURE,
            },
            headers=self._headers(),
            timeout=call_timeout(self.engine, mode),
            bad_key=is_bad_key,
        )
        return chat_completion_reply(reply, self.model, _LABEL, reasoning_as_output=True)

    async def _search(self, prompt: str, cap: int, client: httpx.AsyncClient) -> ProviderReply:
        reply = await post_json(
            client,
            f"{XAI_API_BASE}/responses",
            label=_LABEL,
            payload={
                "model": self.model,
                "input": prompt,
                "tools": [dict(WEB_SEARCH_TOOL)],
                "max_tool_calls": MAX_SEARCH_TOOL_CALLS,
                "max_output_tokens": cap,
                "temperature": DEFAULT_TEMPERATURE,
            },
            headers=self._headers(),
            timeout=call_timeout(self.engine, MODE_SEARCH),
            bad_key=is_bad_key,
        )
        answer = parse_responses_body(reply)
        result = ProviderReply(
            text=answer.text,
            model_version=answer.model or self.model,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            citations=_citations(reply, answer.citations),
            search_count=_search_count(answer.usage, answer.search_calls),
            search_queries=answer.search_queries,
            vendor_cost_usd=_vendor_cost(answer.usage),
        )
        return checked_responses_reply(result, answer.failure, answer.status, answer.incomplete_reason, _LABEL)
