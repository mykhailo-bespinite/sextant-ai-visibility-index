"""ChatGPT through the OpenAI REST API.

`api` mode is Chat Completions (`POST /v1/chat/completions`), no tools.
`search` mode is the Responses API (`POST /v1/responses`) with the
`web_search` tool, which is what ChatGPT browses with, capped at
`MAX_SEARCH_TOOL_CALLS` tool calls (`max_tool_calls` counts the free
`open_page` follow-ups too, so it bounds billed searches from above).
"""

from __future__ import annotations

from typing import Any, Final

import httpx

from aeo_index.modes import MAX_SEARCH_TOOL_CALLS, MODE_SEARCH
from aeo_index.providers.base import (
    DEFAULT_TEMPERATURE,
    ProviderError,
    ProviderReply,
    as_dict,
    as_int,
    as_list,
    call_timeout,
    check_model,
    empty_error,
    is_reasoning_model,
    output_cap,
    post_json,
)
from aeo_index.providers.responses import parse_responses_body

OPENAI_API_BASE: Final[str] = "https://api.openai.com/v1"
WEB_SEARCH_TOOL: Final[dict[str, str]] = {"type": "web_search"}
_LABEL: Final[str] = "OpenAI"


class OpenAIProvider:
    engine = "openai"

    def __init__(self, api_key: str, model: str, *, env_var: str = "OPENAI_API_KEY") -> None:
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
        return await self._chat(prompt, cap, mode, client)

    async def _chat(self, prompt: str, cap: int, mode: str, client: httpx.AsyncClient) -> ProviderReply:
        body: dict[str, Any] = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        if is_reasoning_model(self.model):
            body["max_completion_tokens"] = cap
        else:
            body["max_tokens"] = cap
            body["temperature"] = DEFAULT_TEMPERATURE
        reply = await post_json(
            client,
            f"{OPENAI_API_BASE}/chat/completions",
            label=_LABEL,
            payload=body,
            headers=self._headers(),
            timeout=call_timeout(self.engine, mode),
        )
        return chat_completion_reply(reply, self.model, _LABEL)

    async def _search(self, prompt: str, cap: int, client: httpx.AsyncClient) -> ProviderReply:
        body: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "tools": [dict(WEB_SEARCH_TOOL)],
            "max_tool_calls": MAX_SEARCH_TOOL_CALLS,
            "max_output_tokens": cap,
        }
        if not is_reasoning_model(self.model):
            body["temperature"] = DEFAULT_TEMPERATURE
        reply = await post_json(
            client,
            f"{OPENAI_API_BASE}/responses",
            label=_LABEL,
            payload=body,
            headers=self._headers(),
            timeout=call_timeout(self.engine, MODE_SEARCH),
        )
        answer = parse_responses_body(reply)
        result = ProviderReply(
            text=answer.text,
            model_version=answer.model or self.model,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            citations=answer.citations,
            search_count=answer.search_calls,
            search_queries=answer.search_queries,
            vendor_cost_usd=None,
        )
        return checked_responses_reply(result, answer.failure, answer.status, answer.incomplete_reason, _LABEL)


def chat_completion_reply(
    reply: dict[str, Any], model: str, label: str, *, reasoning_as_output: bool = False
) -> ProviderReply:
    """The ProviderReply of a Chat Completions body (OpenAI's or xAI's).
    `reasoning_as_output` adds `completion_tokens_details.reasoning_tokens`
    to the output tokens, for a vendor that reports them apart."""
    choices = as_list(reply.get("choices"))
    choice = as_dict(choices[0]) if choices else {}
    content = as_dict(choice.get("message")).get("content")
    text = content if isinstance(content, str) else ""
    usage = as_dict(reply.get("usage"))
    output_tokens = as_int(usage.get("completion_tokens"))
    if reasoning_as_output:
        output_tokens += as_int(as_dict(usage.get("completion_tokens_details")).get("reasoning_tokens"))
    served = reply.get("model")
    result = ProviderReply(
        text=text,
        model_version=served if isinstance(served, str) and served else model,
        input_tokens=as_int(usage.get("prompt_tokens")),
        output_tokens=output_tokens,
        citations=None,
        search_count=None,
        search_queries=None,
        vendor_cost_usd=None,
    )
    if not text.strip():
        raise empty_error(label, f"finish_reason={choice.get('finish_reason')!r}", result)
    return result


def checked_responses_reply(
    result: ProviderReply, failure: str | None, status: str | None, reason: str | None, label: str
) -> ProviderReply:
    """`result`, unless the response failed or has no text: then the error,
    carrying the reply so the billed tokens and searches are priced."""
    if failure:
        raise ProviderError(f"{label} web search failed: {failure}", error_type="bad_response", reply=result)
    if not result.text.strip():
        raise empty_error(label, f"status={status!r} reason={reason!r}", result)
    return result
