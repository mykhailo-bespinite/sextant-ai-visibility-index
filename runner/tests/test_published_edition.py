"""The editions published in `data/` hold to the checks the research page
makes, and, from the first one that carries its per-answer facts, rebuild
from them."""

import json
from pathlib import Path

import pytest

from aeo_index.cli import main
from aeo_index.edition import strip_publisher_fields, validate_edition

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_there_is_a_published_edition():
    assert sorted((REPO_ROOT / "data").glob("*/index.json"))


def test_published_editions_validate():
    for path in sorted((REPO_ROOT / "data").glob("*/index.json")):
        assert validate_edition(json.loads(path.read_text(encoding="utf-8"))) == [], path


def test_rebuild_reproduces_published_editions_with_facts(tmp_path):
    found = sorted((REPO_ROOT / "data").glob("*/answers.jsonl"))
    if not found:
        pytest.skip("no published edition carries answers.jsonl yet (first one: the edition after Task 5 ships)")
    for answers in found:
        folder = answers.parent
        published = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        out = tmp_path / folder.name
        args = ["rebuild", "--answers", str(answers), "--questions", str(folder / "prompts.csv")]
        args += ["--edition", published["edition"], "--out", str(out)]
        for engine in published.get("engines_under_review", []):
            args += ["--under-review", engine]
        assert main(args) == 0, folder
        rebuilt = json.loads((out / "index.json").read_text(encoding="utf-8"))
        assert strip_publisher_fields(rebuilt) == strip_publisher_fields(published), folder
