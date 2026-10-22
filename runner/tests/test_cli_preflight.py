import io
import re

import pytest

from aeo_index.cli import build_parser, main


@pytest.fixture
def no_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))


def test_dry_run_prints_plan_and_cost_and_calls_nothing(env_with_two_keys, capsys, no_network):
    assert main(["run", "--dry-run", "--engines", "openai,anthropic", "--samples", "1"]) == 0
    out = capsys.readouterr().out
    assert "20 answers plus up to 20 judge calls; typical $" in out and "high $" in out
    # Engines left out with --engines are not "not configured".
    assert "not configured" not in out
    assert "ChatGPT" in out and "Claude" in out and "judge" in out
    assert "measured as of 2026-10-21" in out and "not yet measured" not in out


def test_dry_run_lists_each_engine_and_mode_row(env_all_keys, capsys, no_network):
    assert main(["run", "--dry-run", "--mode", "both", "--samples", "1"]) == 0
    out = capsys.readouterr().out
    # Four search engines in two modes, two retrieval engines in search: 10 rows, 100 calls.
    assert "100 answers plus up to 100 judge calls" in out and "not configured" not in out
    assert "Google AI Overview, results page via DataForSEO" in out


def test_high_estimate_over_budget_refuses_with_the_budget_to_pass(env_all_keys, capsys, no_network):
    assert main(["run", "--yes", "--budget", "0.01"]) == 2
    assert re.search(r"--budget \d+\.\d{2}", capsys.readouterr().err)


def test_non_terminal_without_yes_refuses(env_with_two_keys, no_terminal, capsys, no_network):
    assert main(["run", "--engines", "openai", "--samples", "1", "--out", "unused-dir"]) == 2
    assert "--yes" in capsys.readouterr().err


def test_missing_judge_key_refuses(env_openai_only, capsys, no_network):
    assert main(["run", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "--judge" in err


def test_another_judge_with_its_key_is_accepted(env_openai_only, capsys, no_network):
    assert main(["run", "--dry-run", "--judge", "openai:gpt-4o-mini"]) == 0


def test_unpriced_model_refused_unless_price_given(env_with_two_keys, capsys, no_network):
    assert main(["run", "--dry-run", "--model", "openai=gpt-9"]) == 2
    err = capsys.readouterr().err
    assert "openai:gpt-9" in err and "--price" in err
    assert main(["run", "--dry-run", "--model", "openai=gpt-9", "--price", "openai:gpt-9=0.01,0.02"]) == 0


def test_a_model_variant_is_refused_with_the_price_line_to_add(env_with_two_keys, capsys, no_network):
    assert main(["run", "--dry-run", "--engines", "openai", "--model", "openai=gpt-5.5-pro"]) == 2
    err = capsys.readouterr().err
    assert "no price for openai:gpt-5.5-pro; pass --price openai:gpt-5.5-pro=IN,OUT" in err
    assert main(["run", "--dry-run", "--engines", "openai", "--model", "openai=gpt-5.5-2026-04-23"]) == 0


def test_country_other_than_us_refused(env_all_keys, capsys, no_network):
    assert main(["run", "--dry-run", "--country", "GB"]) == 2
    assert "US only" in capsys.readouterr().err
    assert main(["run", "--dry-run", "--country", "us"]) == 0


def test_no_engine_configured_refuses(env_none, capsys, no_network):
    assert main(["run", "--dry-run"]) == 2
    assert "no engine" in capsys.readouterr().err.lower()


@pytest.mark.parametrize(
    "argv",
    [
        ["--engines", "openai,bing"],
        ["--samples", "6"],
        ["--samples", "0"],
        ["--budget", "-1"],
        ["--budget", "nan"],
        ["--mode", "deep"],
        ["--engine-mode", "perplexity=api"],
        ["--engine-mode", "openai"],
        ["--model", "openai="],
        ["--model", "gemini=models/../files"],
        ["--price", "openai:gpt-9=abc"],
        ["--judge", "perplexity:sonar"],
        ["--judge", "gemini:models/../files"],
        ["--brand-alias", "Acme"],
    ],
)
def test_bad_flags_are_refused_with_exit_2(env_all_keys, capsys, no_network, argv):
    assert main(["run", "--dry-run", *argv]) == 2
    assert capsys.readouterr().err.strip()


def test_engine_mode_and_samples_change_the_plan(env_with_two_keys, capsys, no_network):
    assert main(["run", "--dry-run", "--engines", "openai", "--engine-mode", "openai=both", "--samples", "2"]) == 0
    assert "40 answers plus up to 40 judge calls" in capsys.readouterr().out


def test_env_file_supplies_keys(env_none, tmp_path, capsys, no_network):
    path = tmp_path / "keys.env"
    path.write_text("OPENAI_API_KEY=k-test-123456\nANTHROPIC_API_KEY=k-test-anthropic-1\n", encoding="utf-8")
    assert main(["run", "--dry-run", "--env-file", str(path), "--samples", "1"]) == 0
    out = capsys.readouterr().out
    assert "20 answers plus up to 20 judge calls" in out and "k-test-123456" not in out


def test_a_bad_env_file_refuses(env_none, tmp_path, capsys, no_network):
    assert main(["run", "--dry-run", "--env-file", str(tmp_path / "absent.env")]) == 2


def test_brand_override_is_checked(env_with_two_keys, capsys, no_network):
    # A brand the questions name makes them branded: refused, naming it.
    assert main(["run", "--dry-run", "--brand", "AEO platforms"]) == 2
    assert "AEO platforms" in capsys.readouterr().err
    assert main(["run", "--dry-run", "--brand", "AEO platforms", "--allow-branded"]) == 0


def test_a_used_output_dir_refuses_unless_forced(env_with_two_keys, tmp_path, capsys, no_network):
    out = tmp_path / "run"
    out.mkdir()
    (out / "index.json").write_text("{}", encoding="utf-8")
    assert main(["run", "--yes", "--engines", "openai", "--samples", "1", "--out", str(out)]) == 2
    assert "--force" in capsys.readouterr().err


def test_an_engine_asked_for_without_a_key_is_named(env_with_two_keys, capsys, no_network):
    assert main(["run", "--dry-run", "--samples", "1"]) == 0
    assert "not configured: grok (set XAI_API_KEY)" in capsys.readouterr().out


def test_the_prompt_states_the_estimate_and_the_budget_cap(
    env_with_two_keys, monkeypatch, capsys, no_network, tmp_path
):
    monkeypatch.setattr("sys.stdin", _Terminal("n\n"))
    assert main(["run", "--engines", "openai", "--samples", "1", "--out", str(tmp_path / "x")]) == 2
    out = capsys.readouterr().out
    assert (
        "Ask 10 answers (plus up to 10 judge calls; typical $" in out and "; capped at the $20.00 budget)? [y/N]" in out
    )


def test_terminal_answer_other_than_yes_asks_nothing(env_with_two_keys, monkeypatch, capsys, no_network, tmp_path):
    monkeypatch.setattr("sys.stdin", _Terminal("n\n"))
    assert main(["run", "--engines", "openai", "--samples", "1", "--out", str(tmp_path / "x")]) == 2
    assert "nothing was asked" in capsys.readouterr().err.lower()


def test_parser_has_the_run_flags():
    commands = build_parser()._subparsers._group_actions[0]  # type: ignore[union-attr]
    run = commands.choices["run"]
    flags = {opt for action in run._actions for opt in action.option_strings}
    assert {
        "--questions",
        "--engines",
        "--samples",
        "--mode",
        "--engine-mode",
        "--model",
        "--judge",
        "--env-file",
        "--out",
        "--force",
        "--no-text",
        "--budget",
        "--yes",
        "--dry-run",
        "--price",
        "--brand",
        "--brand-alias",
        "--brand-domain",
        "--allow-branded",
        "--country",
        "--compare",
    } <= flags


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_a_dry_run_over_budget_exits_0_and_names_the_budget_it_needs(env_with_two_keys, capsys, no_network):
    assert main(["run", "--dry-run", "--budget", "0.01"]) == 0
    assert re.search(r"--budget \d+\.\d{2}", capsys.readouterr().out)


def test_the_default_budget_is_20():
    from decimal import Decimal

    from aeo_index.preflight import DEFAULT_BUDGET_USD

    assert DEFAULT_BUDGET_USD == Decimal("20.00")
    commands = build_parser()._subparsers._group_actions[0]  # type: ignore[union-attr]
    assert "default 20.00" in commands.choices["run"].format_help()


@pytest.mark.parametrize("bad", ["k-test 123456", "k-test-12345é", "k-test\t123456"])
def test_a_key_with_a_space_or_a_non_ascii_character_is_refused_by_name(monkeypatch, capsys, no_network, bad):
    monkeypatch.setenv("OPENAI_API_KEY", bad)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-test-anthropic-1")
    assert main(["run", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err and "space" in err and bad not in err


def test_a_malformed_key_of_an_engine_not_asked_is_ignored(env_with_two_keys, monkeypatch, capsys, no_network):
    monkeypatch.setenv("XAI_API_KEY", "k-test 123456")
    assert main(["run", "--dry-run", "--engines", "openai", "--samples", "1"]) == 0
    assert main(["run", "--dry-run", "--engines", "openai,grok", "--samples", "1"]) == 2
    assert "XAI_API_KEY holds a space" in capsys.readouterr().err
