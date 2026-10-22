import pytest

from aeo_index.providers.base import ProviderError
from aeo_index.providers.gemini import GeminiProvider

KEY = "AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"


def provider(model: str = "models/gemini-3.5-flash") -> GeminiProvider:
    return GeminiProvider(KEY, model)


async def test_gemini_key_travels_in_header_not_url(recorder):
    await provider().ask("q", mode="api", client=recorder.client("gemini_api_ok.json"))
    assert "key=" not in str(recorder.last_url) and recorder.last_headers["x-goog-api-key"].startswith("AIza")


async def test_gemini_api_mode_posts_generate_content_without_tools(recorder):
    await provider().ask("q", mode="api", client=recorder.client("gemini_api_ok.json"))
    body = recorder.last_json
    assert recorder.last_url == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"
    )
    assert "tools" not in body and body["generationConfig"]["maxOutputTokens"] == 8192


async def test_gemini_search_mode_grounds_with_google_search(recorder):
    reply = await provider().ask("q", mode="search", client=recorder.client("gemini_search_ok.json"))
    assert recorder.last_json["tools"] == [{"google_search": {}}]
    assert reply.search_count == 2 and reply.search_queries == (
        "best AEO platform for agencies",
        "AEO tools comparison",
    )


async def test_a_bare_model_id_gets_the_models_prefix(recorder):
    await provider("gemini-3.5-flash").ask("q", mode="api", client=recorder.client("gemini_api_ok.json"))
    assert recorder.last_url.path == "/v1beta/models/gemini-3.5-flash:generateContent"


async def test_the_responding_model_is_recorded(recorder):
    client = recorder.client("gemini_search_ok.json")
    recorder.fixture["body"]["modelVersion"] = "gemini-3.5-flash-rerouted"
    reply = await provider().ask("q", mode="search", client=client)
    assert reply.model_version == "models/gemini-3.5-flash-rerouted"


@pytest.mark.parametrize("name", ["gemini_api_empty.json", "gemini_search_empty.json"])
async def test_truncated_empty_reply_raises_empty(recorder, name):
    mode = "api" if "_api_" in name else "search"
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode=mode, client=recorder.client(name))
    assert e.value.error_type == "empty"


async def test_no_candidate_raises_empty(recorder):
    client = recorder.client("gemini_api_ok.json")
    recorder.fixture["body"]["candidates"] = []
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode="api", client=client)
    assert e.value.error_type == "empty"


def test_a_model_with_a_path_in_it_is_refused():
    with pytest.raises(ValueError):
        GeminiProvider(KEY, "models/../v1/files")


async def test_a_bad_key_400_is_auth_with_its_status_kept(recorder):
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode="api", client=recorder.client("gemini_400_bad_key.json"))
    assert e.value.error_type == "auth" and e.value.status_code == 400


async def test_another_400_is_a_client_error(recorder):
    client = recorder.client("gemini_400_bad_key.json")
    recorder.fixture["body"] = {
        "error": {"code": 400, "message": "Invalid JSON payload.", "status": "INVALID_ARGUMENT"}
    }
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode="api", client=client)
    assert e.value.error_type == "client"
