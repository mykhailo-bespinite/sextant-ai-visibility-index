import pytest

from aeo_index import modes
from aeo_index.modes import engine_label, engine_modes


def test_roster_order_and_labels_are_the_published_ones():
    assert modes.ENGINES == ("anthropic", "openai", "perplexity", "gemini", "grok", "google_aio")
    assert modes.ENGINE_LABELS == {
        "anthropic": "Claude",
        "openai": "ChatGPT",
        "perplexity": "Perplexity",
        "gemini": "Gemini",
        "grok": "Grok",
        "google_aio": "Google AI Overview",
    }
    assert list(modes.ENGINE_LABELS) == list(modes.ENGINES)


def test_modes_statuses_and_engine_split():
    assert modes.MODES == ("api", "search") and modes.DISPLAY_MODES == ("search", "api")
    assert (modes.MODE_API, modes.MODE_SEARCH) == modes.MODES
    assert modes.STATUSES == ("success", "empty", "error", "timeout", "rate_limited", "skipped")
    assert modes.STATUSES == (
        modes.STATUS_SUCCESS,
        modes.STATUS_EMPTY,
        modes.STATUS_ERROR,
        modes.STATUS_TIMEOUT,
        modes.STATUS_RATE_LIMITED,
        modes.STATUS_SKIPPED,
    )
    assert modes.SEARCH_ENGINES == ("openai", "anthropic", "gemini", "grok")
    assert modes.RETRIEVAL_ENGINES == ("perplexity", "google_aio")
    assert sorted(modes.SEARCH_ENGINES + modes.RETRIEVAL_ENGINES) == sorted(modes.ENGINES)
    assert modes.RESULTS_PAGE_ENGINES == ("google_aio",)


def test_method_constants():
    assert (modes.DEFAULT_SAMPLES, modes.MIN_SAMPLES, modes.MAX_SAMPLES) == (3, 1, 5)
    assert modes.CALL_TIMEOUT_SECONDS == {"api": 90, "search": 180}
    assert modes.AIO_TIMEOUT_SECONDS == 120
    assert (modes.MAX_SEARCH_TOOL_CALLS, modes.MAX_ANSWER_CITATIONS) == (3, 10)
    assert (modes.MAX_SEARCH_QUERIES, modes.MAX_SEARCH_QUERY_CHARS) == (20, 500)
    assert (modes.TOP_DOMAINS, modes.TOP_URLS) == (40, 60)


def test_constants_cannot_be_changed_in_place():
    with pytest.raises(TypeError):
        modes.CALL_TIMEOUT_SECONDS["api"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        modes.ENGINE_LABELS["openai"] = "x"  # type: ignore[index]


def test_retrieval_engines_only_search():
    assert engine_modes("perplexity", "api") == ("search",)
    assert engine_modes("google_aio", "both") == ("search",)
    assert engine_modes("openai", "both") == ("api", "search")
    assert engine_modes("anthropic", "api") == ("api",)
    assert engine_modes("grok", "search") == ("search",)


@pytest.mark.parametrize(("engine", "requested"), [("openai", "web"), ("bing", "api"), ("", "search")])
def test_unknown_engine_or_mode_is_refused(engine, requested):
    with pytest.raises(ValueError):
        engine_modes(engine, requested)


def test_engine_label_marks_the_results_page():
    assert engine_label("google_aio") == "Google AI Overview, results page via DataForSEO"
    assert engine_label("openai") == "ChatGPT"
    with pytest.raises(KeyError):
        engine_label("bing")
