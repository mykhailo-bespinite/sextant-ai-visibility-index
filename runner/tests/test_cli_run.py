"""The whole run through the command line, over recorded vendor bodies
(the judge included), with sockets disabled."""

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from aeo_index import pricing
from aeo_index.cli import main
from aeo_index.edition import strip_publisher_fields, validate_edition
from aeo_index.outputs import OUTPUT_FILES
from run_doubles import RunTransport
from vendor_doubles import FAKE_ENV

REPO_ROOT = Path(__file__).resolve().parents[2]
SIX_FILES = set(OUTPUT_FILES)
TWO_ENGINES = ["--engines", "openai,anthropic", "--samples", "1"]


@pytest.fixture
def use_transport(monkeypatch):
    """Route the CLI's HTTP client through `transport`."""

    def _use(transport: httpx.MockTransport) -> httpx.MockTransport:
        monkeypatch.setattr("aeo_index.cli.http_client", lambda: httpx.AsyncClient(transport=transport))
        return transport

    return _use


@pytest.fixture
def fixtures_transport(use_transport):
    return use_transport(RunTransport())


@pytest.fixture
def openai_401_transport(use_transport):
    return use_transport(RunTransport({"openai": "openai_401.json"}))


def _files(out: Path) -> set[str]:
    return {p.name for p in out.iterdir()}


def _doc(out: Path) -> dict:
    return json.loads((out / "index.json").read_text(encoding="utf-8"))


def _rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()]


def test_run_writes_every_file_and_exits_0(tmp_path, env_with_two_keys, fixtures_transport, capsys):
    out = tmp_path / "r"
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(out)]) == 0
    assert _files(out) == SIX_FILES
    doc = _doc(out)
    assert validate_edition(doc) == []
    assert [r["status"] for r in doc["runs"]] == ["completed"] and doc["goal_mode"] == "search"
    producer = doc["producer"]
    assert producer["name"] == "aeo-index" and producer["version"] == "0.1.0"
    assert producer["judge"] == "anthropic:claude-haiku-4-5-20251001"
    assert producer["models"] == {"anthropic": "claude-sonnet-4-6", "openai": "gpt-5.5"}
    assert producer["brand"] == "Sextant AEO" and producer["question_set"] == "sextant-index-v1"
    # Claude's recorded search answer names Profound only; the judge's "Peec AI" is dropped on its 10 answers.
    assert producer["judge_ungrounded"] == 10 and producer["extract_version"] == 2
    assert fixtures_transport.judge_calls == 20 and len(fixtures_transport.calls) == 20
    captured = capsys.readouterr()
    assert "With web search" in captured.out and "| Profound |" in captured.out
    assert "Run completed" in captured.out and "Actual cost $" in captured.out
    assert "judge_ungrounded 10" in captured.out and "ChatGPT, search: success 10" in captured.out
    assert "Claude: asked" in captured.err  # progress lines


def test_run_with_rejected_key_exits_3_and_writes_outputs(tmp_path, env_with_two_keys, openai_401_transport, capsys):
    out = tmp_path / "r"
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(out)]) == 3
    assert _files(out) == SIX_FILES
    everything = "".join(p.read_text(encoding="utf-8") for p in out.iterdir())
    assert FAKE_ENV["OPENAI_API_KEY"] not in everything
    captured = capsys.readouterr()
    assert FAKE_ENV["OPENAI_API_KEY"] not in captured.out + captured.err
    assert "OPENAI_API_KEY was rejected (401)" in captured.err
    doc = _doc(out)
    assert doc["runs"][0]["status"] == "partial"
    assert openai_401_transport.calls.count("openai") == 1  # one probe, nothing after the rejection
    assert "ChatGPT, search: error 10" in captured.out


def test_budget_stop_exits_4_with_skipped_rows(tmp_path, env_with_two_keys, use_transport, monkeypatch, capsys):
    # Typical figures far below the recorded bodies' real cost let a run fit
    # a budget its first answer then uses up.
    tiny = {key: (Decimal("0.0001"), Decimal("0.0001")) for key in pricing.TYPICAL_CALL_USD}
    monkeypatch.setattr("aeo_index.pricing.TYPICAL_CALL_USD", tiny)
    use_transport(RunTransport())
    out = tmp_path / "r"
    code = main(
        [
            "run",
            "--yes",
            "--engines",
            "anthropic",
            "--mode",
            "api",
            "--samples",
            "1",
            "--budget",
            "0.07",
            "--out",
            str(out),
        ]
    )
    assert code == 4
    rows = _rows(out)
    assert [r["status"] for r in rows].count("skipped") == 9 and rows[0]["status"] == "success"
    assert rows[1]["error_type"] == "budget"
    assert _files(out) == SIX_FILES
    captured = capsys.readouterr()
    assert "stopped at the budget" in captured.out


def test_actual_cost_is_the_budgets_spend(tmp_path, env_with_two_keys, fixtures_transport, capsys):
    from aeo_index.preflight import money

    out = tmp_path / "r"
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(out)]) == 0
    rows_cost = sum(Decimal(r["cost_usd"]) for r in _rows(out))
    judge_cost = 20 * Decimal("0.000400")  # 300 tokens in and 20 out at Haiku's rates, per judged answer
    printed = capsys.readouterr().out
    expected = (
        f"Actual cost {money(rows_cost + judge_cost)} of the $20.00 budget "
        f"(answers {money(rows_cost)}, judge {money(judge_cost)})."
    )
    assert expected in printed


def test_existing_output_dir_refused_without_force(tmp_path, env_with_two_keys, fixtures_transport, capsys):
    out = tmp_path / "r"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(out)]) == 2
    assert fixtures_transport.calls == [] and fixtures_transport.judge_calls == 0
    assert "in use" in capsys.readouterr().err
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(out), "--force"]) == 0
    assert _files(out) == SIX_FILES | {"keep.txt"}


def test_no_text_omits_answer_text(tmp_path, env_with_two_keys, fixtures_transport):
    out = tmp_path / "r"
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(out), "--no-text"]) == 0
    rows = _rows(out)
    assert rows and all("text" not in row for row in rows)
    everything = "".join(p.read_text(encoding="utf-8") for p in out.iterdir())
    assert "Based on the results" not in everything and "Peec AI is second" not in everything
    with_text = tmp_path / "t"
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(with_text)]) == 0
    assert any(row.get("text") for row in _rows(with_text))
    assert "Based on the results" not in (with_text / "index.json").read_text(encoding="utf-8")


def test_rebuild_from_answers_jsonl_equals_the_run_edition(tmp_path, env_with_two_keys, fixtures_transport, capsys):
    out = tmp_path / "r"
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(out)]) == 0
    rebuilt = tmp_path / "again"
    assert main(["rebuild", "--answers", str(out / "answers.jsonl"), "--out", str(rebuilt)]) == 0
    assert strip_publisher_fields(_doc(rebuilt)) == strip_publisher_fields(_doc(out))
    assert _files(rebuilt) == SIX_FILES
    from_csv = tmp_path / "csv"
    args = ["rebuild", "--answers", str(out / "answers.jsonl"), "--questions", str(out / "prompts.csv")]
    assert main([*args, "--out", str(from_csv)]) == 0
    assert strip_publisher_fields(_doc(from_csv)) == strip_publisher_fields(_doc(out))
    assert (rebuilt / "table.md").read_text(encoding="utf-8") == (out / "table.md").read_text(encoding="utf-8")


def test_rebuild_refuses_bad_input(tmp_path, capsys):
    bad = tmp_path / "answers.jsonl"
    bad.write_text("{}\n", encoding="utf-8")
    assert main(["rebuild", "--answers", str(bad), "--out", str(tmp_path / "o")]) == 2
    assert "line 1" in capsys.readouterr().err
    assert main(["rebuild", "--answers", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path / "o")]) == 2


def test_run_with_compare_prints_the_side_by_side(tmp_path, env_with_two_keys, fixtures_transport, capsys):
    out = tmp_path / "r"
    published = REPO_ROOT / "data" / "2026-10-18" / "index.json"
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(out), "--compare", str(published)]) == 0
    captured = capsys.readouterr().out
    assert "beside Edition 2026-10-18" in captured and "api row per engine" in captured
    assert "| Profound |" in captured.split("beside Edition 2026-10-18")[1]


def test_compare_subcommand(tmp_path, env_with_two_keys, fixtures_transport, capsys):
    out = tmp_path / "r"
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["compare", str(out), "--against", str(out / "index.json")]) == 0
    assert "| Profound |" in capsys.readouterr().out
    assert main(["compare", str(tmp_path / "nothing")]) == 2


def test_a_failed_compare_keeps_the_run(tmp_path, env_with_two_keys, fixtures_transport, capsys):
    out = tmp_path / "r"
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(out), "--compare", str(tmp_path / "none.json")]) == 0
    assert _files(out) == SIX_FILES and "could not compare" in capsys.readouterr().err.lower()


def test_judge_progress_and_summary_name_the_judge(tmp_path, env_with_two_keys, fixtures_transport, capsys):
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(tmp_path / "r")]) == 0
    err = capsys.readouterr().err
    assert "Judging 20 answers with anthropic:claude-haiku-4-5-20251001" in err


def test_rebuild_lists_only_the_questions_answered(tmp_path, capsys):
    from aeo_index.outputs import ANSWERS_FILE
    from run_doubles import fact

    facts = tmp_path / ANSWERS_FILE
    facts.write_text(json.dumps(fact("bhag_03", "perplexity", "search").to_json()) + "\n", encoding="utf-8")
    assert main(["rebuild", "--answers", str(facts), "--out", str(tmp_path / "o")]) == 0
    doc = _doc(tmp_path / "o")
    assert [p["key"] for p in doc["prompts"]] == ["bhag_03"] and validate_edition(doc) == []
    assert doc["producer"]["brand"] == "Sextant AEO"
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text(json.dumps(fact("zz_01", "perplexity", "search").to_json()) + "\n", encoding="utf-8")
    assert main(["rebuild", "--answers", str(unknown), "--out", str(tmp_path / "u")]) == 2
    assert "zz_01" in capsys.readouterr().err


def test_rebuild_with_the_published_prompts_csv_is_the_index_set(tmp_path):
    from aeo_index.outputs import ANSWERS_FILE
    from run_doubles import fact

    facts = tmp_path / ANSWERS_FILE
    rows = [fact("bhag_01", "perplexity", "search", citations=["https://sextantaeo.com/x"])]
    facts.write_text("".join(json.dumps(r.to_json()) + "\n" for r in rows), encoding="utf-8")
    published = REPO_ROOT / "data" / "2026-10-18" / "prompts.csv"
    out = tmp_path / "o"
    assert main(["rebuild", "--answers", str(facts), "--questions", str(published), "--out", str(out)]) == 0
    doc = _doc(out)
    assert doc["producer"]["question_set"] == "sextant-index-v1" and doc["top_domains"][0]["sextant"] is True


def test_a_rejected_judge_key_exits_3(tmp_path, env_with_two_keys, use_transport, capsys):
    from run_doubles import is_judge_request

    class RejectJudge(RunTransport):
        def _handle(self, request):
            if is_judge_request(request):
                return httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})
            return super()._handle(request)

    use_transport(RejectJudge())
    out = tmp_path / "r"
    assert main(["run", "--yes", "--engines", "openai", "--samples", "1", "--out", str(out)]) == 3
    assert "the remaining answers are not judged" in capsys.readouterr().err
    assert all(row["tools_named"] is None for row in _rows(out))


def test_compare_prints_no_control_characters_from_a_crafted_edition(tmp_path, capsys):
    from run_doubles import build, fact

    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "index.json").write_text(json.dumps(build([fact("q01", "anthropic", "api")])), encoding="utf-8")
    crafted = build([fact("q01", "anthropic", "api", tools=("Evil\x1b[31mTool​",))])
    path = tmp_path / "crafted.json"
    path.write_text(json.dumps(crafted), encoding="utf-8")
    assert main(["compare", str(mine), "--against", str(path)]) == 0
    captured = capsys.readouterr()
    for stream in (captured.out, captured.err):
        assert "\x1b" not in stream and "​" not in stream
    assert "| EvilTool [31m |" not in captured.out and "Evil[31mTool" in captured.out


# ------------------------------------------------------------------ fix round 1


def test_rebuild_reads_the_goal_row_each_day_recorded(tmp_path):
    from aeo_index.outputs import ANSWERS_FILE
    from run_doubles import fact

    rows = []
    # 10-14: the model engines answer from training knowledge, the retrieval engines search.
    for engine in ("anthropic", "openai", "gemini", "grok"):
        rows.append(fact("bhag_01", engine, "api", run_date="2026-10-14"))
    for engine in ("perplexity", "google_aio"):
        rows.append(fact("bhag_01", engine, "search", run_date="2026-10-14"))
    # 10-18: the model engines also search.
    for engine in ("anthropic", "openai", "gemini", "grok"):
        rows.append(fact("bhag_01", engine, "api", run_date="2026-10-18"))
        rows.append(fact("bhag_01", engine, "search", run_date="2026-10-18"))
    for engine in ("perplexity", "google_aio"):
        rows.append(fact("bhag_01", engine, "search", run_date="2026-10-18"))
    path = tmp_path / ANSWERS_FILE
    path.write_text("".join(json.dumps(r.to_json()) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "o"
    assert main(["rebuild", "--answers", str(path), "--out", str(out)]) == 0
    doc = _doc(out)
    assert [(r["run_date"], r["goal_mode"]) for r in doc["runs"]] == [("2026-10-14", "api"), ("2026-10-18", "search")]
    assert doc["goal_mode"] == "search" and validate_edition(doc) == []


def test_an_api_run_records_the_api_goal_row(tmp_path, monkeypatch, use_transport):
    from conftest import RUNNER_ENV_VARS
    from vendor_doubles import FAKE_ENV as ENV

    for name in RUNNER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.setenv(name, ENV[name])
    use_transport(RunTransport())
    out = tmp_path / "r"
    args = ["run", "--yes", "--engines", "openai,perplexity", "--mode", "api", "--samples", "1", "--out", str(out)]
    assert main(args) == 0
    doc = _doc(out)
    assert doc["goal_mode"] == "api" and doc["runs"][0]["goal_mode"] == "api"


@pytest.fixture
def read_only_dir(tmp_path):
    import os
    import stat

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can write into a read-only folder")
    folder = tmp_path / "ro"
    folder.mkdir()
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    yield folder
    folder.chmod(stat.S_IRWXU)


def test_a_failed_write_still_prints_the_summary(read_only_dir, env_with_two_keys, fixtures_transport, capsys):
    assert main(["run", "--yes", *TWO_ENGINES, "--out", str(read_only_dir)]) == 3
    err = capsys.readouterr().err
    assert "Could not write the run to" in err and "Actual cost $" in err and "Run completed" in err
    assert "Exit 3: the files could not be written." in err
    assert FAKE_ENV["OPENAI_API_KEY"] not in err


def test_a_failed_write_keeps_a_budget_stop_at_4(read_only_dir, env_with_two_keys, use_transport, monkeypatch, capsys):
    tiny = {key: (Decimal("0.0001"), Decimal("0.0001")) for key in pricing.TYPICAL_CALL_USD}
    monkeypatch.setattr("aeo_index.pricing.TYPICAL_CALL_USD", tiny)
    use_transport(RunTransport())
    args = ["run", "--yes", "--engines", "anthropic", "--mode", "api", "--samples", "1", "--budget", "0.07"]
    assert main([*args, "--out", str(read_only_dir)]) == 4
    assert "It stopped at the budget." in capsys.readouterr().err


def test_an_interrupt_reports_the_spend_and_exits_130(tmp_path, env_with_two_keys, use_transport, capsys):
    class Interrupting(RunTransport):
        def _handle(self, request):
            if self.calls:  # the first call is answered and billed, the second is interrupted
                raise KeyboardInterrupt
            return super()._handle(request)

    use_transport(Interrupting())
    out = tmp_path / "r"
    assert main(["run", "--yes", "--engines", "anthropic", "--mode", "api", "--samples", "1", "--out", str(out)]) == 130
    captured = capsys.readouterr()
    assert "Interrupted: nothing was written; $0.01 was spent on 1 paid call." in captured.err
    assert "Traceback" not in captured.err + captured.out
    assert not out.exists() or not any(out.iterdir())


def test_compare_refuses_a_crafted_count_with_exit_2(tmp_path, capsys):
    from run_doubles import build, sample_facts

    mine = tmp_path / "mine.json"
    mine.write_text(json.dumps(build(sample_facts())), encoding="utf-8")
    crafted = build(sample_facts())
    crafted["modes"]["search"]["leaderboard"] = [{"tool": "X", "first_count": "5"}]
    path = tmp_path / "crafted.json"
    path.write_text(json.dumps(crafted), encoding="utf-8")
    assert main(["compare", str(mine), "--against", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("Could not compare:") and len(captured.err.strip().splitlines()) == 1
    assert "Traceback" not in captured.err


def test_an_interrupt_during_the_write_names_the_files_already_written(
    tmp_path, env_with_two_keys, fixtures_transport, monkeypatch, capsys
):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    # index.json and table.md are written before the first CSV.
    monkeypatch.setattr("aeo_index.outputs._write_csv", interrupted)
    out = tmp_path / "r"
    assert main(["run", "--yes", "--engines", "anthropic", "--mode", "api", "--samples", "1", "--out", str(out)]) == 130
    captured = capsys.readouterr()
    assert f"Interrupted: only index.json and table.md were written to {out}, so the run is incomplete;" in captured.err
    assert "nothing was written" not in captured.err
    assert "Traceback" not in captured.err + captured.out
    assert _files(out) == {"index.json", "table.md"}


def test_an_interrupt_at_the_confirmation_exits_130_quietly(
    env_with_two_keys, monkeypatch, capsys, no_network, tmp_path
):
    import io

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    def interrupted(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("sys.stdin", Terminal(""))
    monkeypatch.setattr("builtins.input", interrupted)
    out = tmp_path / "x"
    assert main(["run", "--engines", "openai", "--samples", "1", "--out", str(out)]) == 130
    captured = capsys.readouterr()
    assert "Interrupted: nothing was asked and nothing was spent." in captured.err
    assert "Traceback" not in captured.err + captured.out
    assert not out.exists()


def test_compare_with_a_corrupt_compressed_body_exits_2_with_one_line(tmp_path, use_transport, capsys):
    from run_doubles import build, sample_facts

    mine = tmp_path / "mine.json"
    mine.write_text(json.dumps(build(sample_facts())), encoding="utf-8")
    use_transport(
        httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=b"not gzip at all")
        )
    )
    assert main(["compare", str(mine), "--against", "2026-10-18"]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("Could not compare:") and len(captured.err.strip().splitlines()) == 1
    assert "could not be decoded" in captured.err
    assert "Traceback" not in captured.err + captured.out


def test_an_interrupt_mid_file_leaves_no_truncated_file(
    tmp_path, env_with_two_keys, fixtures_transport, monkeypatch, capsys
):
    import aeo_index.outputs as outputs

    real = outputs._write_csv

    def half_then_interrupt(path, columns, rows):
        if path.name.startswith("cited_domains"):
            path.write_text("mode,dom", encoding="utf-8")  # cut short, as Ctrl-C would leave it
            raise KeyboardInterrupt
        return real(path, columns, rows)

    monkeypatch.setattr("aeo_index.outputs._write_csv", half_then_interrupt)
    out = tmp_path / "r"
    assert main(["run", "--yes", "--engines", "anthropic", "--mode", "api", "--samples", "1", "--out", str(out)]) == 130
    err = capsys.readouterr().err
    assert "only index.json, table.md and leaderboard.csv were written" in err
    assert _files(out) == {"index.json", "table.md", "leaderboard.csv"}
