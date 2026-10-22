from decimal import Decimal

import pytest

from aeo_index.pricing import RateCard, call_cost
from aeo_index.providers.base import ProviderError
from aeo_index.providers.grok import GrokProvider


def provider() -> GrokProvider:
    return GrokProvider("k-test-grok-12345", "grok-4.3")


async def test_grok_api_mode_is_xai_chat_completions(recorder):
    reply = await provider().ask("q", mode="api", client=recorder.client("grok_api_ok.json"))
    body = recorder.last_json
    assert recorder.last_url == "https://api.x.ai/v1/chat/completions"
    assert "tools" not in body and body["max_tokens"] == 1024 and body["temperature"] == 0.7
    # Reasoning tokens are billed as output.
    assert reply.output_tokens == 700


async def test_grok_search_posts_xai_responses_with_web_search(recorder):
    reply = await provider().ask("q", mode="search", client=recorder.client("grok_search_ok.json"))
    body = recorder.last_json
    assert recorder.last_url == "https://api.x.ai/v1/responses"
    assert body["tools"] == [{"type": "web_search"}] and body["max_tool_calls"] == 3
    assert body["max_output_tokens"] == 8192
    assert reply.search_count == 2 and reply.vendor_cost_usd is not None


async def test_grok_search_without_vendor_cost_falls_back_to_the_rate_card(recorder):
    client = recorder.client("grok_search_ok.json")
    del recorder.fixture["body"]["usage"]["cost_in_usd_ticks"]
    reply = await provider().ask("q", mode="search", client=client)
    assert reply.vendor_cost_usd is None
    # 1000 in at 0.00125, 500 out at 0.0025, two searches at $5 per 1K.
    assert call_cost("grok", "grok-4.3", "search", reply, RateCard.default()) == Decimal("0.012500")


@pytest.mark.parametrize("name", ["grok_api_empty.json", "grok_search_empty.json"])
async def test_truncated_empty_reply_raises_empty(recorder, name):
    mode = "api" if "_api_" in name else "search"
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode=mode, client=recorder.client(name))
    assert e.value.error_type == "empty"


async def test_a_bad_key_400_is_auth_with_its_status_kept(recorder):
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode="api", client=recorder.client("grok_400_bad_key.json"))
    assert e.value.error_type == "auth" and e.value.status_code == 400
    assert "Incorrect API key provided" in str(e.value)


async def test_another_400_is_a_client_error(recorder):
    client = recorder.client("grok_400_bad_key.json")
    recorder.fixture["body"] = {"code": "Client specified an invalid argument", "error": "max_tokens is too large."}
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode="api", client=client)
    assert e.value.error_type == "client"
