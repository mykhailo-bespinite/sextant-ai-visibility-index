"""Every recorded vendor body gives what its `expected` block says.

The same files are the dispatch-parity data: another implementation of
these six vendors can parse them and must reach the same text, citations,
search count, search queries, model version and cost to six decimals."""

from decimal import Decimal

import pytest

from aeo_index.fixtures import vendor_fixture_names
from aeo_index.pricing import RateCard, call_cost
from aeo_index.providers.base import DEFAULT_MODELS, ProviderError, build_providers
from vendor_doubles import FAKE_ENV, Recorder

ENGINES = ("openai", "anthropic", "gemini", "grok", "perplexity", "google_aio")


def _engine(name: str) -> str:
    return next(engine for engine in ENGINES if name.startswith(engine + "_"))


def _mode(name: str) -> str:
    return "api" if "_api_" in name else "search"


def test_every_engine_has_its_recorded_bodies():
    names = set(vendor_fixture_names())
    for engine in ENGINES:
        modes = ("search",) if engine in ("perplexity", "google_aio") else ("api", "search")
        for mode in modes:
            assert {f"{engine}_{mode}_ok.json", f"{engine}_{mode}_empty.json"} <= names
        # A rejected key, as the vendor sends it: Google and xAI answer 400.
        bad_key = f"{engine}_400_bad_key.json" if engine in ("gemini", "grok") else f"{engine}_401.json"
        assert {bad_key, f"{engine}_429_retry_after_90.json", f"{engine}_503.json"} <= names


@pytest.mark.parametrize("name", vendor_fixture_names())
async def test_fixture_parses_to_its_expected_result(name):
    engine, mode = _engine(name), _mode(name)
    providers, _ = build_providers(FAKE_ENV, [engine], DEFAULT_MODELS)
    provider = providers[engine]
    recorder = Recorder()
    client = recorder.client(name)
    expected = recorder.expected
    card = RateCard.default()
    if "error_type" in expected:
        with pytest.raises(ProviderError) as caught:
            await provider.ask("Which AEO tool is best?", mode=mode, client=client)
        error = caught.value
        assert error.error_type == expected["error_type"]
        assert error.status_code == expected["status_code"]
        assert error.retry_after == expected["retry_after"]
        billed = call_cost(engine, provider.model, mode, error.reply, card) if error.reply else error.cost_usd
        assert billed == Decimal(expected["cost_usd"])
        return
    reply = await provider.ask("Which AEO tool is best?", mode=mode, client=client)
    assert reply.text == expected["text"]
    assert reply.model_version == expected["model_version"]
    assert (reply.input_tokens, reply.output_tokens) == (expected["input_tokens"], expected["output_tokens"])
    citations = expected["citations"]
    assert reply.citations == (tuple(citations) if citations is not None else None)
    assert reply.search_count == expected["search_count"]
    queries = expected["search_queries"]
    assert reply.search_queries == (tuple(queries) if queries is not None else None)
    vendor = expected["vendor_cost_usd"]
    assert reply.vendor_cost_usd == (Decimal(vendor) if vendor is not None else None)
    assert call_cost(engine, provider.model, mode, reply, card) == Decimal(expected["cost_usd"])
