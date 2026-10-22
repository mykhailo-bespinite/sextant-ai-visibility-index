import subprocess
import sys

from aeo_index.cli import build_parser, main
from conftest import write_toml


def test_questions_prints_the_default_set(capsys):
    assert main(["questions"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 10 and out[0].startswith("bhag_01")
    assert "What are the best AEO platforms for marketing agencies?" in out[0]


def test_questions_validates_a_file(tmp_path, capsys):
    path = write_toml(tmp_path, questions=[("q01", "Which CRM fits a small agency?"), ("q02", "Is Profound any good?")])
    assert main(["questions", "--questions", str(path)]) == 2
    err = capsys.readouterr().err
    assert "q02" in err and "Profound" in err and "--allow-branded" in err
    assert main(["questions", "--questions", str(path), "--allow-branded"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "experimental" in out[1] and "experimental" not in out[0]


def test_aliases_prints_one_tool_per_line(capsys):
    assert main(["aliases"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 12
    assert out[1] == "Peec AI: Peec AI, Peec"
    assert out[-1] == "Sextant AEO: Sextant AEO, Sextant, sextantaeo.com"


def test_no_subcommand_prints_help_and_exits_2(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_parser_has_the_subcommands():
    help_text = build_parser().format_help()
    assert "questions" in help_text and "aliases" in help_text


def test_python_dash_m_runs_the_cli():
    done = subprocess.run([sys.executable, "-m", "aeo_index", "aliases"], capture_output=True, text=True, check=False)
    assert done.returncode == 0 and done.stdout.startswith("Profound: Profound")


def _flags_named(text: str) -> set[str]:
    import re

    return set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", text))


def _readme_sections() -> dict[str, str]:
    """The README's text under each "### `aeo-index <command>`" heading, up to
    the next heading of that level or higher, keyed by command; `questions`
    and `aliases` share one section."""
    import re
    from pathlib import Path

    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    for match in re.finditer(r"^### ([^\n]+)\n(.*?)(?=^#{2,3} |\Z)", readme, flags=re.M | re.S):
        for command in re.findall(r"`aeo-index (\w+)`", match.group(1)):
            sections[command] = match.group(2)
    return sections


def test_the_readme_documents_every_flag_under_its_own_command():
    from pathlib import Path

    from aeo_index.cli import build_parser

    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # type: ignore[union-attr]
    sections = _readme_sections()
    assert set(sections) == set(commands), f"README sections {sorted(sections)} vs commands {sorted(commands)}"
    own = {
        name: {opt for action in sub._actions for opt in action.option_strings if opt.startswith("--")} - {"--help"}
        for name, sub in commands.items()
    }
    for name in commands:
        # A section shared by two commands documents the flags of both.
        sharing = [other for other in commands if sections[other] == sections[name]]
        flags = set().union(*(own[other] for other in sharing))
        named = _flags_named(sections[name])
        assert named == flags, (
            f"{name}: README names {sorted(named - flags)} it lacks and leaves out {sorted(flags - named)}"
        )
    top = {opt for action in parser._actions for opt in action.option_strings if opt.startswith("--")} - {"--help"}
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    assert top <= _flags_named(readme)
    every = top | {opt for sub in commands.values() for action in sub._actions for opt in action.option_strings}
    assert _flags_named(readme) <= every, (
        f"the README names flags no command has: {sorted(_flags_named(readme) - every)}"
    )
