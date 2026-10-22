from decimal import Decimal

import pytest

from aeo_index.pricing import RateCard, call_cost
from aeo_index.providers.base import ProviderError
from aeo_index.providers.perplexity import PerplexityProvider


def provider() -> PerplexityProvider:
    return PerplexityProvider("k-test-pplx-12345", "sonar-pro")


async def test_perplexity_posts_chat_completions_and_reads_citations(recorder):
    reply = await provider().ask("q", mode="search", client=recorder.client("perplexity_search_ok.json"))
    body = recorder.last_json
    assert recorder.last_url == "https://api.perplexity.ai/chat/completions"
    assert body["model"] == "sonar-pro" and body["messages"] == [{"role": "user", "content": "q"}]
    assert recorder.last_headers["authorization"] == "Bearer k-test-pplx-12345"
    assert reply.citations is not None and reply.citations[0] == "https://example.com/reviews/aeo-tools"
    assert reply.vendor_cost_usd is not None and reply.search_count is None


async def test_without_a_vendor_cost_the_rate_card_prices_it(recorder):
    client = recorder.client("perplexity_search_ok.json")
    del recorder.fixture["body"]["usage"]["cost"]
    reply = await provider().ask("q", mode="search", client=client)
    assert reply.vendor_cost_usd is None
    # 12 in at 0.003, 420 out at 0.015, and the $6 per 1K request fee.
    assert call_cost("perplexity", "sonar-pro", "search", reply, RateCard.default()) == Decimal("0.012336")


async def test_truncated_empty_reply_raises_empty(recorder):
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode="search", client=recorder.client("perplexity_search_empty.json"))
    assert e.value.error_type == "empty" and e.value.reply is not None


async def test_a_non_json_body_is_a_bad_response(recorder):
    client = recorder.client("perplexity_search_ok.json")
    recorder.fixture["body"] = None
    recorder.fixture["raw"] = "<html>gateway</html>"
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode="search", client=client)
    assert e.value.error_type == "bad_response"
