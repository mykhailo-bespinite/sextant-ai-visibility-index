import pytest

from aeo_index.providers.anthropic import AnthropicProvider
from aeo_index.providers.base import ProviderError


def provider() -> AnthropicProvider:
    return AnthropicProvider("k-test-anthropic-1", "claude-sonnet-4-6")


async def test_anthropic_search_attaches_web_search_20250305_with_max_uses_3(recorder):
    reply = await provider().ask("q", mode="search", client=recorder.client("anthropic_search_ok.json"))
    body = recorder.last_json
    assert recorder.last_url == "https://api.anthropic.com/v1/messages"
    assert body["tools"] == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    assert body["max_tokens"] == 4096
    # The "I'll search" preamble is not the answer.
    assert "I'll search" not in reply.text and reply.search_count == 1


async def test_anthropic_api_mode_has_no_tools_and_1024_tokens(recorder):
    await provider().ask("q", mode="api", client=recorder.client("anthropic_api_ok.json"))
    body = recorder.last_json
    assert "tools" not in body and body["max_tokens"] == 1024
    assert recorder.last_headers["x-api-key"] == "k-test-anthropic-1"
    assert recorder.last_headers["anthropic-version"] == "2023-06-01"


async def test_a_thinking_block_before_the_text_is_skipped(recorder):
    client = recorder.client("anthropic_api_ok.json")
    recorder.fixture["body"]["content"].insert(0, {"type": "thinking", "thinking": "hmm", "signature": "x"})
    reply = await provider().ask("q", mode="api", client=client)
    assert reply.text.startswith("Profound")


@pytest.mark.parametrize("name", ["anthropic_api_empty.json", "anthropic_search_empty.json"])
async def test_truncated_empty_reply_raises_empty(recorder, name):
    mode = "api" if "_api_" in name else "search"
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode=mode, client=recorder.client(name))
    assert e.value.error_type == "empty"


async def test_503_is_a_server_error(recorder):
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode="api", client=recorder.client("anthropic_503.json"))
    assert e.value.error_type == "server" and e.value.status_code == 503
