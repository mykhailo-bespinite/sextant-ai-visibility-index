import httpx
import pytest

from aeo_index.providers.base import ProviderError
from aeo_index.providers.openai import OpenAIProvider


def provider(model: str = "gpt-5.5") -> OpenAIProvider:
    return OpenAIProvider("k-test-123456", model)


async def test_openai_search_posts_responses_with_web_search_and_tool_cap(recorder):
    reply = await provider().ask("q", mode="search", client=recorder.client("openai_search_ok.json"))
    body = recorder.last_json
    assert recorder.last_path == "/v1/responses" and body["tools"][0]["type"] == "web_search"
    assert body["max_tool_calls"] == 3
    assert reply.text and reply.citations and reply.search_count == recorder.fixture_search_calls


async def test_openai_api_mode_uses_chat_completions_without_tools(recorder):
    reply = await provider().ask("q", mode="api", client=recorder.client("openai_api_ok.json"))
    body = recorder.last_json
    assert recorder.last_url == "https://api.openai.com/v1/chat/completions"
    assert "tools" not in body and "max_tool_calls" not in body
    assert body["messages"] == [{"role": "user", "content": "q"}]
    # A reasoning model takes max_completion_tokens and no temperature.
    assert body["max_completion_tokens"] == 8192 and "temperature" not in body
    assert reply.citations is None and reply.search_count is None


async def test_openai_key_travels_as_bearer(recorder):
    await provider().ask("q", mode="api", client=recorder.client("openai_api_ok.json"))
    assert recorder.last_headers["authorization"] == "Bearer k-test-123456"


async def test_classic_model_takes_max_tokens_and_temperature(recorder):
    await provider("gpt-4o-mini").ask("q", mode="api", client=recorder.client("openai_api_ok.json"))
    body = recorder.last_json
    assert body["max_tokens"] == 1024 and body["temperature"] == 0.7 and "max_completion_tokens" not in body


async def test_max_output_tokens_overrides_the_cap(recorder):
    await provider().ask("q", mode="search", client=recorder.client("openai_search_ok.json"), max_output_tokens=500)
    assert recorder.last_json["max_output_tokens"] == 500


@pytest.mark.parametrize("name", ["openai_api_empty.json", "openai_search_empty.json"])
async def test_truncated_empty_reply_raises_empty(recorder, name):
    mode = "api" if "_api_" in name else "search"
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode=mode, client=recorder.client(name))
    assert e.value.error_type == "empty" and e.value.reply is not None


async def test_401_is_auth_and_429_carries_retry_after(recorder):
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode="api", client=recorder.client("openai_401.json"))
    assert e.value.error_type == "auth" and e.value.status_code == 401
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode="api", client=recorder.client("openai_429_retry_after_90.json"))
    assert e.value.error_type == "rate_limited" and e.value.retry_after == 90.0


async def test_post_json_turns_any_http_failure_into_a_provider_error():
    from aeo_index.providers.base import ProviderError, post_json

    def corrupt(request):
        return httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=b"not gzip")

    async with httpx.AsyncClient(transport=httpx.MockTransport(corrupt)) as client:
        with pytest.raises(ProviderError) as caught:
            await post_json(client, "https://api.openai.com/v1/x", label="ChatGPT", payload={}, timeout=5)
    assert caught.value.error_type == "bad_response" and "DecodingError" in str(caught.value)
    assert "api.openai.com" not in str(caught.value)
