"""Claude through the Anthropic Messages API (`POST /v1/messages`).

`search` mode attaches the server-side `web_search_20250305` tool with
`max_uses` = `MAX_SEARCH_TOOL_CALLS`. When a search ran, the answer is the
text after the last search result: the "I'll search for X" preamble and
any narration between searches are not the answer, and must not credit a
tool named only there.
"""

from __future__ import annotations

from typing import Any, Final

import httpx

from aeo_index.modes import MAX_SEARCH_TOOL_CALLS, MODE_SEARCH
from aeo_index.providers.base import (
    ProviderReply,
    as_dict,
    as_list,
    call_timeout,
    check_model,
    dedupe_http_urls,
    empty_error,
    output_cap,
    post_json,
    token_sum,
    usable_queries,
)

ANTHROPIC_MESSAGES_URL: Final[str] = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION: Final[str] = "2023-06-01"
WEB_SEARCH_TOOL: Final[dict[str, object]] = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": MAX_SEARCH_TOOL_CALLS,
}
_TEXT: Final[str] = "text"
_TOOL_USE: Final[str] = "server_tool_use"
_SEARCH_RESULT: Final[str] = "web_search_tool_result"
_SEARCH_ERROR: Final[str] = "web_search_tool_result_error"
_CITATION: Final[str] = "web_search_result_location"
_LABEL: Final[str] = "Anthropic"


def answer_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The blocks after the last search result (after the last tool call
    when no result came back); every block when no search ran."""
    types = [block.get("type") for block in blocks]
    if _TOOL_USE not in types:
        return blocks
    marker = _SEARCH_RESULT if _SEARCH_RESULT in types else _TOOL_USE
    last = len(types) - 1 - types[::-1].index(marker)
    return blocks[last + 1 :]


def answer_text(blocks: list[dict[str, Any]]) -> str:
    """The answer's text. Consecutive text blocks are one run of prose
    split at citation boundaries, so they join with no separator; a
    non-text block between runs ends a paragraph."""
    runs: list[str] = []
    current = ""
    for block in answer_blocks(blocks):
        if block.get("type") == _TEXT and isinstance(block.get("text"), str):
            current += block["text"]
        elif current:
            runs.append(current)
            current = ""
    if current:
        runs.append(current)
    return "\n\n".join(run.strip() for run in runs if run.strip())


def _citations(blocks: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        dedupe_http_urls(
            citation.get("url")
            for block in answer_blocks(blocks)
            if block.get("type") == _TEXT
            for citation in as_list(block.get("citations"))
            if isinstance(citation, dict) and citation.get("type") == _CITATION
        )
    )


def _queries(blocks: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        usable_queries(as_dict(block.get("input")).get("query") for block in blocks if block.get("type") == _TOOL_USE)
    )


def _search_errors(blocks: list[dict[str, Any]]) -> list[str]:
    return [
        str(as_dict(block.get("content")).get("error_code"))
        for block in blocks
        if block.get("type") == _SEARCH_RESULT and as_dict(block.get("content")).get("type") == _SEARCH_ERROR
    ]


def _searches_billed(usage: dict[str, Any], blocks: list[dict[str, Any]]) -> int:
    """`usage.server_tool_use.web_search_requests` (failed searches are not
    billed); absent, the successful result blocks."""
    reported = as_dict(usage.get("server_tool_use")).get("web_search_requests")
    if isinstance(reported, int) and not isinstance(reported, bool) and reported >= 0:
        return reported
    return sum(1 for b in blocks if b.get("type") == _SEARCH_RESULT and isinstance(b.get("content"), list))


class AnthropicProvider:
    engine = "anthropic"

    def __init__(self, api_key: str, model: str, *, env_var: str = "ANTHROPIC_API_KEY") -> None:
        self._api_key = api_key
        self.model = check_model(model)
        self.env_var = env_var

    async def ask(
        self, prompt: str, *, mode: str, client: httpx.AsyncClient, max_output_tokens: int | None = None
    ) -> ProviderReply:
        cap = max_output_tokens if max_output_tokens is not None else output_cap(self.engine, self.model, mode)
        searching = mode == MODE_SEARCH
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": cap,
            "messages": [{"role": "user", "content": prompt}],
        }
        if searching:
            body["tools"] = [dict(WEB_SEARCH_TOOL)]
        reply = await post_json(
            client,
            ANTHROPIC_MESSAGES_URL,
            label=_LABEL,
            payload=body,
            headers={"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION},
            timeout=call_timeout(self.engine, mode),
        )
        blocks = [block for block in as_list(reply.get("content")) if isinstance(block, dict)]
        usage = as_dict(reply.get("usage"))
        served = reply.get("model")
        result = ProviderReply(
            text=answer_text(blocks),
            model_version=served if isinstance(served, str) and served else self.model,
            input_tokens=token_sum(usage, "input_tokens"),
            output_tokens=token_sum(usage, "output_tokens"),
            citations=_citations(blocks) if searching else None,
            search_count=_searches_billed(usage, blocks) if searching else None,
            search_queries=_queries(blocks) if searching else None,
            vendor_cost_usd=None,
        )
        if not result.text:
            detail = f"stop_reason={reply.get('stop_reason')!r} search_errors={_search_errors(blocks)}"
            raise empty_error(_LABEL, detail, result)
        return result
