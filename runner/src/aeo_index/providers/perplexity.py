"""Perplexity through the Sonar API (`POST /chat/completions`).

Sonar searches on every call, so it is asked in `search` mode only; the
mode argument changes nothing. Its sources are the reply's `citations`
list, then the `search_results` entries. The cost is the vendor's own
`usage.cost.total_cost` when present (it reflects the search context tier
the request resolved to); the rate card and request fee otherwise.
"""

from __future__ import annotations

from typing import Final

import httpx

from aeo_index.providers.base import (
    DEFAULT_TEMPERATURE,
    ProviderReply,
    as_dict,
    as_int,
    as_list,
    call_timeout,
    check_model,
    dedupe_http_urls,
    empty_error,
    output_cap,
    positive_decimal,
    post_json,
)

PERPLEXITY_URL: Final[str] = "https://api.perplexity.ai/chat/completions"
_LABEL: Final[str] = "Perplexity"


class PerplexityProvider:
    engine = "perplexity"

    def __init__(self, api_key: str, model: str, *, env_var: str = "PERPLEXITY_API_KEY") -> None:
        self._api_key = api_key
        self.model = check_model(model)
        self.env_var = env_var

    async def ask(
        self, prompt: str, *, mode: str, client: httpx.AsyncClient, max_output_tokens: int | None = None
    ) -> ProviderReply:
        cap = max_output_tokens if max_output_tokens is not None else output_cap(self.engine, self.model, mode)
        reply = await post_json(
            client,
            PERPLEXITY_URL,
            label=_LABEL,
            payload={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": cap,
                "temperature": DEFAULT_TEMPERATURE,
            },
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=call_timeout(self.engine, mode),
        )
        choices = as_list(reply.get("choices"))
        choice = as_dict(choices[0]) if choices else {}
        content = as_dict(choice.get("message")).get("content")
        usage = as_dict(reply.get("usage"))
        results = (item.get("url") for item in as_list(reply.get("search_results")) if isinstance(item, dict))
        served = reply.get("model")
        result = ProviderReply(
            text=content if isinstance(content, str) else "",
            model_version=served if isinstance(served, str) and served else self.model,
            input_tokens=as_int(usage.get("prompt_tokens")),
            output_tokens=as_int(usage.get("completion_tokens")),
            citations=tuple(dedupe_http_urls([*as_list(reply.get("citations")), *results])),
            search_count=None,
            search_queries=None,
            vendor_cost_usd=positive_decimal(as_dict(usage.get("cost")).get("total_cost")),
        )
        if not result.text.strip():
            raise empty_error(_LABEL, f"finish_reason={choice.get('finish_reason')!r}", result)
        return result
