import json
from pathlib import Path

import httpx
import pytest

from aeo_index.compare import BRANCH, REPO, CompareError, render_compare, resolve_edition
from run_doubles import build, fact, sample_facts

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = REPO_ROOT / "data" / "2026-10-18" / "index.json"


def test_repo_and_branch():
    assert REPO == "mykhailo-bespinite/sextant-ai-visibility-index" and BRANCH == "master"


def test_compare_against_a_legacy_edition_states_its_basis():
    ours = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    out = render_compare(build(sample_facts()), ours, ours_label="Sextant edition 2026-10-18")
    assert "Sextant edition 2026-10-18" in out and "api row per engine" in out and "Profound" in out
    assert "With web search" not in out  # a legacy edition has one row per engine, not a per-mode block


def test_shares_are_first_named_over_each_sides_own_answers():
    facts = sample_facts()
    mine = build(facts)
    out = render_compare(mine, mine, ours_label="Sextant edition 2026-10-21")
    # Search row: Profound named first in 11 of 17 answers.
    assert "| Profound | 11 | 64.7% | 11 | 64.7% |" in out
    assert "With web search" in out and "API only" in out
    assert out.index("With web search") < out.index("API only")


def test_different_question_sets_show_shares_only():
    custom = [fact("q01", "anthropic", "api"), fact("q02", "anthropic", "api", tools=("Peec AI",))]
    ours = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    out = render_compare(build(custom), ours, ours_label="Sextant edition 2026-10-18")
    assert "answer different questions" in out
    assert "| Tool | Your run, share | Sextant edition 2026-10-18, share |" in out
    assert "| Profound | 50.0% |" in out


def test_a_search_only_run_against_a_legacy_edition_says_the_rows_differ():
    facts = [f for f in sample_facts() if f.mode == "search"]
    ours = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    out = render_compare(build(facts), ours, ours_label="Sextant edition 2026-10-18")
    assert "count different rows" in out and "search row per engine" in out


def test_a_mode_the_edition_lacks_is_named():
    api_only = [f for f in sample_facts() if f.mode == "api"]
    search_only = [f for f in sample_facts() if f.mode == "search"]
    out = render_compare(build(api_only), build(search_only), ours_label="Sextant edition X")
    assert "Sextant edition X has no api row, so there is nothing to set beside yours." in out


def test_top_limits_each_side():
    facts = [fact("q01", "anthropic", "api", s, tools=(f"Tool {s}",)) for s in range(1, 6)]
    doc = build(facts)
    out = render_compare(doc, doc, ours_label="E", top=2)
    assert out.count("| Tool ") == 1 + 2


# ------------------------------------------------------------------ resolving an edition


def _listing_transport(seen, *, listing=None, doc=None, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "api.github.com":
            names = listing if listing is not None else ["2026-10-18", "2026-10-25", "README.md"]
            return httpx.Response(
                status,
                json=[{"name": n, "type": "file" if n.endswith(".md") else "dir"} for n in names],
            )
        assert request.url.host == "raw.githubusercontent.com"
        return httpx.Response(status, json=doc if doc is not None else build(sample_facts()))

    return httpx.MockTransport(handler)


@pytest.fixture
def github_listing_transport():
    seen: list[httpx.Request] = []
    return seen, _listing_transport(seen)


async def test_latest_resolves_newest_date_directory(github_listing_transport):
    seen, transport = github_listing_transport
    async with httpx.AsyncClient(transport=transport) as client:
        label, doc = await resolve_edition("latest", client=client)
    assert label == "Sextant edition 2026-10-25" and "modes" in doc
    assert str(seen[0].url).startswith(f"https://api.github.com/repos/{REPO}/contents/data")
    assert str(seen[1].url) == f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data/2026-10-25/index.json"


async def test_a_date_reads_that_edition():
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_listing_transport(seen)) as client:
        label, _ = await resolve_edition("2026-10-18", client=client)
    assert label == "Sextant edition 2026-10-18" and len(seen) == 1 and "2026-10-18/index.json" in str(seen[0].url)


async def test_a_local_path_reads_the_file_or_folder(tmp_path):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("no request"))) as client:
        label, doc = await resolve_edition(str(PUBLISHED), client=client)
        assert "runs" in doc and label == "Edition 2026-10-18"  # the legacy file has no edition key: its folder
        label, _ = await resolve_edition(str(PUBLISHED.parent), client=client)
        assert label == "Edition 2026-10-18"
        mine = tmp_path / "run"
        mine.mkdir()
        (mine / "index.json").write_text(json.dumps(build(sample_facts())), encoding="utf-8")
        label, _ = await resolve_edition(str(mine), client=client)
        assert label == "Edition 2026-10-21"


async def test_failures_are_compare_errors(tmp_path):
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_listing_transport(seen, status=404)) as client:
        with pytest.raises(CompareError, match="404"):
            await resolve_edition("2026-10-19", client=client)
    async with httpx.AsyncClient(transport=_listing_transport(seen, listing=["README.md"])) as client:
        with pytest.raises(CompareError, match="no edition"):
            await resolve_edition("latest", client=client)
    async with httpx.AsyncClient(transport=_listing_transport(seen, doc={"runs": "nope"})) as client:
        with pytest.raises(CompareError, match="not a valid edition"):
            await resolve_edition("2026-10-18", client=client)
    async with httpx.AsyncClient(transport=_listing_transport(seen)) as client:
        with pytest.raises(CompareError, match="no file"):
            await resolve_edition(str(tmp_path / "missing.json"), client=client)
        with pytest.raises(CompareError, match="not a date"):
            await resolve_edition("2026-13-45", client=client)


async def test_unreachable_github_is_a_compare_error():
    def handler(request):
        raise httpx.ConnectError("no route")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CompareError, match="could not be reached"):
            await resolve_edition("latest", client=client)


# ------------------------------------------------------------------ untrusted strings


EVIL_TOOL = "Evil\x1b[31mTool​\x07"
EVIL_LABEL = "Sextant\x1b]0;owned\x07 edition‮"


def _evil_edition():
    doc = build([fact("q01", "anthropic", "api", tools=(EVIL_TOOL,))])
    doc["modes"]["api"]["engines"][0]["engine"] = "anthropic"
    return doc


def _has_controls(text: str) -> bool:
    import unicodedata

    return any(unicodedata.category(ch).startswith("C") and ch != "\n" for ch in text)


def test_control_and_format_characters_never_reach_the_comparison():
    mine = build([fact("q01", "anthropic", "api")])
    ours = _evil_edition()
    ours["modes"]["api"]["engines"].append(
        {"engine": "bad\x1b[2Jengine", "mode": "api", "answers": 1, "ok": 1, "sextant_named": 0, "sextant_first": 0}
    )
    out = render_compare(mine, ours, ours_label=EVIL_LABEL)
    assert "\x1b" not in out and "​" not in out and "‮" not in out and "\x07" not in out
    assert not _has_controls(out)
    assert "| EvilTool |" in out.replace("[31m", "")  # the name stays readable without its escape bytes
    assert "bad[2Jengine" in out  # an unknown engine label is cleaned, not dropped


def test_control_characters_never_reach_the_table():
    from aeo_index.table import render_table

    facts = [fact("q01", "anthropic", "api", tools=(EVIL_TOOL,))]
    md = render_table(build(facts), facts)
    assert "\x1b" not in md and "​" not in md and not _has_controls(md)


async def test_a_bad_edition_error_carries_no_control_characters(tmp_path):
    bad = build(sample_facts())
    bad["modes"]["web\x1b[31m"] = {}
    path = tmp_path / "index.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("no request"))) as client:
        with pytest.raises(CompareError) as caught:
            await resolve_edition(str(path), client=client)
    assert "\x1b" not in str(caught.value)


def test_an_engine_with_no_answers_is_named_in_the_comparison():
    rows = [fact("q01", "anthropic", "search", s) for s in (1, 2)]
    rows += [fact("q01", "perplexity", "search", s, status="error") for s in (1, 2, 3)]
    doc = build(rows)
    out = render_compare(doc, doc, ours_label="E")
    assert "Your run: 2 answers from Claude; no answers: Perplexity (3 asked)." in out


async def test_a_leaderboard_count_that_is_not_a_count_is_refused(tmp_path):
    crafted = build(sample_facts())
    crafted["modes"]["search"]["leaderboard"][0]["first_count"] = "5"
    path = tmp_path / "index.json"
    path.write_text(json.dumps(crafted), encoding="utf-8")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("no request"))) as client:
        with pytest.raises(CompareError, match="first_count is not a count"):
            await resolve_edition(str(path), client=client)


async def test_a_mode_block_without_a_leaderboard_list_is_refused(tmp_path):
    crafted = build(sample_facts())
    crafted["modes"]["api"]["leaderboard"] = {"tool": "X"}
    path = tmp_path / "index.json"
    path.write_text(json.dumps(crafted), encoding="utf-8")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("no request"))) as client:
        with pytest.raises(CompareError, match="leaderboard"):
            await resolve_edition(str(path), client=client)


def test_a_missing_mode_block_is_read_as_empty():
    mine = build(sample_facts())
    ours = build(sample_facts())
    del ours["modes"]["api"]
    del mine["modes"]["search"]
    out = render_compare(mine, ours, ours_label="E")
    assert "E has no api row" in out


async def test_an_oversized_edition_is_refused_while_it_downloads(monkeypatch):
    import aeo_index.compare as compare_module

    monkeypatch.setattr(compare_module, "MAX_EDITION_BYTES", 1000)

    def handler(request):
        return httpx.Response(200, content=b"[" + b" " * 5000 + b"]")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CompareError, match="larger than 1000 bytes"):
            await resolve_edition("2026-10-18", client=client)


async def test_a_corrupt_compressed_body_is_a_compare_error():
    def handler(request):
        return httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=b"\x1f\x8b broken")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CompareError, match="the edition 2026-10-18 sent a body that could not be decoded"):
            await resolve_edition("2026-10-18", client=client)


def test_the_header_names_engines_under_review_and_different_extraction():
    ours = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    mine = build(sample_facts())
    mine["producer"] = {"name": "aeo-index", "extract_version": 2}
    out = render_compare(mine, ours, ours_label="Sextant edition 2026-10-18")
    assert (
        "Sextant edition 2026-10-18 lists Gemini as under review; its answers are counted in that side's shares." in out
    )
    assert (
        "The two sides read the names differently: Your run with extraction version 2, which keeps a name only "
        "when the answer contains it; Sextant edition 2026-10-18 with version 1, which did not check." in out
    )


def test_no_review_or_extraction_line_when_both_sides_agree():
    doc = build(sample_facts())
    out = render_compare(doc, doc, ours_label="E")
    assert "under review" not in out and "read the names differently" not in out
