import base64
from decimal import Decimal

import pytest

from aeo_index.providers.base import ProviderError
from aeo_index.providers.google_aio import GoogleAioProvider


def provider() -> GoogleAioProvider:
    return GoogleAioProvider("login-test-1", "pw-test-123456")


async def test_google_aio_is_live_advanced_us_and_reads_vendor_cost(recorder):
    reply = await provider().ask(
        "best AEO platform", mode="search", client=recorder.client("google_aio_search_ok.json")
    )
    assert recorder.last_url == "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
    (task,) = recorder.last_json
    assert task["keyword"] == "best AEO platform" and task["location_code"] == 2840
    assert task["language_code"] == "en" and task["load_async_ai_overview"] is True
    expected = "Basic " + base64.b64encode(b"login-test-1:pw-test-123456").decode()
    assert recorder.last_headers["authorization"] == expected
    assert reply.vendor_cost_usd == Decimal("0.002") and reply.model_version == "google-ai-overview"


async def test_unresolved_google_redirects_are_not_citations(recorder):
    reply = await provider().ask("q", mode="search", client=recorder.client("google_aio_search_ok.json"))
    assert reply.citations is not None and not any("goto" in url for url in reply.citations)


async def test_no_ai_overview_is_empty_and_billed(recorder):
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode="search", client=recorder.client("google_aio_search_empty.json"))
    assert e.value.error_type == "empty" and e.value.cost_usd == Decimal("0.002")


@pytest.mark.parametrize(
    ("code", "error_type"), [(40100, "auth"), (40101, "server"), (40202, "rate_limited"), (40501, "client")]
)
async def test_task_level_failures_map_to_error_types(recorder, code, error_type):
    client = recorder.client("google_aio_search_ok.json")
    recorder.fixture["body"]["tasks"][0]["status_code"] = code
    recorder.fixture["body"]["tasks"][0]["status_message"] = "vendor said no"
    with pytest.raises(ProviderError) as e:
        await provider().ask("q", mode="search", client=client)
    assert e.value.error_type == error_type and e.value.cost_usd == Decimal("0.002")
    assert e.value.status_code == code


async def test_a_missing_task_cost_uses_the_published_ceiling(recorder):
    client = recorder.client("google_aio_search_ok.json")
    del recorder.fixture["body"]["tasks"][0]["cost"]
    reply = await provider().ask("q", mode="search", client=client)
    assert reply.vendor_cost_usd == Decimal("0.0026")
