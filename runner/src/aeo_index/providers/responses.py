"""The Responses API body (`POST /v1/responses`), which OpenAI and xAI
answer web-search requests with in the same shape: an `output` list of
`web_search_call` items and `message` items whose `output_text` parts
carry `url_citation` annotations. One parser, so the two cannot drift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from aeo_index.providers.base import as_dict, as_int, as_list, dedupe_http_urls, usable_queries, vendor_message

# Only `search` actions are billed; `open_page` and `find_in_page` are free.
_BILLED_SEARCH_ACTION: Final[str] = "search"
_INCOMPLETE: Final[str] = "incomplete"
_FAILED: Final[str] = "failed"


@dataclass(frozen=True)
class ResponsesAnswer:
    text: str
    citations: tuple[str, ...]
    search_calls: int
    search_queries: tuple[str, ...]
    status: str | None
    incomplete_reason: str | None
    input_tokens: int
    output_tokens: int
    model: str | None
    usage: dict[str, Any]
    # The vendor's error summary when the response failed.
    failure: str | None

    @property
    def truncated(self) -> bool:
        return self.status == _INCOMPLETE


def _message_parts(output: list[Any]) -> list[dict[str, Any]]:
    return [
        part
        for item in output
        if isinstance(item, dict) and item.get("type") == "message"
        for part in as_list(item.get("content"))
        if isinstance(part, dict) and part.get("type") == "output_text"
    ]


def _search_calls(output: list[Any]) -> list[dict[str, Any]]:
    return [item for item in output if isinstance(item, dict) and item.get("type") == "web_search_call"]


def _billed_searches(output: list[Any]) -> int:
    # A call with no recorded action is counted: an over-count beats a silent zero.
    return sum(
        1
        for item in _search_calls(output)
        if not isinstance(item.get("action"), dict) or item["action"].get("type") == _BILLED_SEARCH_ACTION
    )


def _billed_queries(output: list[Any]) -> list[str]:
    actions = (as_dict(item.get("action")) for item in _search_calls(output))
    return usable_queries(a.get("query") for a in actions if a.get("type") == _BILLED_SEARCH_ACTION)


def parse_responses_body(body: dict[str, Any]) -> ResponsesAnswer:
    """Reduce a Responses API reply. Never raises; a failed response is
    reported in `failure`, so the caller can price it before raising."""
    status = body.get("status") if isinstance(body.get("status"), str) else None
    error = body.get("error")
    failure = None
    if status == _FAILED or error:
        failure = f"status={status!r} error={vendor_message({'error': error})}"
    output = as_list(body.get("output"))
    parts = _message_parts(output)
    # Joined with no separator, as the vendors' own `output_text` is.
    text = "".join(p["text"] for p in parts if isinstance(p.get("text"), str))
    annotations = (
        a.get("url")
        for p in parts
        for a in as_list(p.get("annotations"))
        if isinstance(a, dict) and a.get("type") == "url_citation"
    )
    incomplete = body.get("incomplete_details")
    usage = as_dict(body.get("usage"))
    model = body.get("model")
    return ResponsesAnswer(
        text=text,
        citations=tuple(dedupe_http_urls(annotations)),
        search_calls=_billed_searches(output),
        search_queries=tuple(_billed_queries(output)),
        status=status,
        incomplete_reason=incomplete.get("reason") if isinstance(incomplete, dict) else None,
        input_tokens=as_int(usage.get("input_tokens")),
        output_tokens=as_int(usage.get("output_tokens")),
        model=model if isinstance(model, str) and model else None,
        usage=usage,
        failure=failure,
    )
