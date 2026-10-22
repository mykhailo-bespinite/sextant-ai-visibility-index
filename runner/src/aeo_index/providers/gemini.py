"""Gemini through the Generative Language REST API
(`POST /v1beta/{model}:generateContent`).

The key travels in the `x-goog-api-key` header, never in the URL. `search`
mode grounds the answer with Google Search (`{"google_search": {}}`); the
tool takes no limit on its queries, and each query it runs is billed.
Thought parts are not the answer. A reply cut off (`MAX_TOKENS`) before
any answer text is `empty`, never a success.
"""

from __future__ import annotations

import re
from typing import Any, Final
from urllib.parse import urlsplit

import httpx

from aeo_index.modes import MODE_SEARCH
from aeo_index.providers.base import (
    DEFAULT_TEMPERATURE,
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

GEMINI_API_BASE: Final[str] = "https://generativelanguage.googleapis.com/v1beta"
GOOGLE_SEARCH_TOOL: Final[dict[str, dict[str, object]]] = {"google_search": {}}
_MODEL_PREFIX: Final[str] = "models/"
# A grounding chunk's `uri` is a redirect that hides the page; its title
# (or domain) names the source's host.
_GROUNDING_REDIRECT_HOST: Final[str] = "vertexaisearch.cloud.google.com"
_HOSTNAME: Final[re.Pattern[str]] = re.compile(r"(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}")
_MODEL_VERSION_MAX_CHARS: Final[int] = 128
_LABEL: Final[str] = "Gemini"


def _grounding_source(web: dict[str, Any]) -> str | None:
    uri = web.get("uri")
    if not isinstance(uri, str):
        return None
    if urlsplit(uri).hostname != _GROUNDING_REDIRECT_HOST:
        return uri
    for candidate in (web.get("domain"), web.get("title")):
        if isinstance(candidate, str) and _HOSTNAME.fullmatch(candidate.strip().lower()):
            return f"https://{candidate.strip().lower()}"
    return uri


def _citations(metadata: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        dedupe_http_urls(
            _grounding_source(as_dict(chunk.get("web")))
            for chunk in as_list(metadata.get("groundingChunks"))
            if isinstance(chunk, dict)
        )
    )


def _text(candidate: dict[str, Any]) -> str:
    parts = as_list(as_dict(candidate.get("content")).get("parts"))
    return "\n".join(
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
    )


def is_bad_key(body: Any) -> bool:
    """Google rejects a key with HTTP 400: `INVALID_ARGUMENT` with the
    reason `API_KEY_INVALID` (or its "API key not valid" message)."""
    error = as_dict(as_dict(body).get("error"))
    reasons = {as_dict(detail).get("reason") for detail in as_list(error.get("details"))}
    message = str(error.get("message") or "").lower()
    return "API_KEY_INVALID" in reasons or (
        error.get("status") == "INVALID_ARGUMENT" and "api key not valid" in message
    )


def _responding_model(reply: dict[str, Any], requested: str) -> str:
    """The model that answered (`modelVersion`), with the `models/` prefix
    the requested id carries, else the requested id."""
    value = reply.get("modelVersion")
    if not isinstance(value, str) or not value.strip():
        return requested
    responding = value.strip()
    if requested.startswith(_MODEL_PREFIX) and not responding.startswith(_MODEL_PREFIX):
        responding = _MODEL_PREFIX + responding
    return responding[:_MODEL_VERSION_MAX_CHARS]


class GeminiProvider:
    engine = "gemini"

    def __init__(self, api_key: str, model: str, *, env_var: str = "GEMINI_API_KEY") -> None:
        self._api_key = api_key
        self.model = check_model(model)
        self.env_var = env_var

    def _url(self) -> str:
        path = self.model if self.model.startswith(_MODEL_PREFIX) else _MODEL_PREFIX + self.model
        return f"{GEMINI_API_BASE}/{path}:generateContent"

    async def ask(
        self, prompt: str, *, mode: str, client: httpx.AsyncClient, max_output_tokens: int | None = None
    ) -> ProviderReply:
        cap = max_output_tokens if max_output_tokens is not None else output_cap(self.engine, self.model, mode)
        searching = mode == MODE_SEARCH
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": cap, "temperature": DEFAULT_TEMPERATURE},
        }
        if searching:
            body["tools"] = [dict(GOOGLE_SEARCH_TOOL)]
        reply = await post_json(
            client,
            self._url(),
            label=_LABEL,
            payload=body,
            headers={"x-goog-api-key": self._api_key},
            timeout=call_timeout(self.engine, mode),
            bad_key=is_bad_key,
        )
        candidates = as_list(reply.get("candidates"))
        candidate = as_dict(candidates[0]) if candidates else {}
        metadata = as_dict(candidate.get("groundingMetadata"))
        queries = tuple(usable_queries(as_list(metadata.get("webSearchQueries"))))
        usage = as_dict(reply.get("usageMetadata"))
        result = ProviderReply(
            text=_text(candidate),
            model_version=_responding_model(reply, self.model),
            input_tokens=token_sum(usage, "promptTokenCount", "toolUsePromptTokenCount"),
            output_tokens=token_sum(usage, "candidatesTokenCount", "thoughtsTokenCount"),
            citations=_citations(metadata) if searching else None,
            search_count=len(queries) if searching else None,
            search_queries=queries if searching else None,
            vendor_cost_usd=None,
        )
        if not candidate:
            block = as_dict(reply.get("promptFeedback")).get("blockReason")
            raise empty_error(_LABEL, f"no candidate, blockReason={block!r}", result)
        if not result.text.strip():
            raise empty_error(_LABEL, f"finishReason={candidate.get('finishReason')!r}", result)
        return result
