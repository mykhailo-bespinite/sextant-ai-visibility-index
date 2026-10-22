"""Live smoke: real vendors, real keys, real money (a few cents).

Skipped unless AEO_INDEX_LIVE=1. Run it with the keys in the environment:

    AEO_INDEX_LIVE=1 pytest tests/test_live_smoke.py -q -s --force-enable-socket

The first test asks one question of every configured engine in api mode
(the retrieval engines search, as always), one sample, under a $0.25
budget. The web search row costs more than that (about $0.40 typical for
the five model engines), so its test also needs AEO_INDEX_LIVE_SEARCH=1
and runs under a $0.75 budget.
"""

import json
import os
from pathlib import Path

import pytest

from aeo_index.cli import main
from aeo_index.edition import validate_edition

ONE_QUESTION_FILE = Path(__file__).parent / "fixtures" / "one-question.toml"

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("AEO_INDEX_LIVE") != "1", reason="live smoke: set AEO_INDEX_LIVE=1 and real keys"
    ),
    pytest.mark.enable_socket,
]


def _run(out: Path, *extra: str) -> int:
    return main(["run", "--yes", "--samples", "1", "--questions", str(ONE_QUESTION_FILE), "--out", str(out), *extra])


def test_live_one_question_every_configured_engine(tmp_path):
    out = tmp_path / "live"
    code = _run(out, "--mode", "api", "--budget", "0.25")
    assert code in (0, 3)
    assert validate_edition(json.loads((out / "index.json").read_text(encoding="utf-8"))) == []


@pytest.mark.skipif(
    os.environ.get("AEO_INDEX_LIVE_SEARCH") != "1", reason="the search row: set AEO_INDEX_LIVE_SEARCH=1"
)
def test_live_one_question_with_web_search(tmp_path):
    out = tmp_path / "live-search"
    code = _run(out, "--mode", "search", "--budget", "0.75")
    assert code in (0, 3)
    assert validate_edition(json.loads((out / "index.json").read_text(encoding="utf-8"))) == []
