import os

import pytest

from aeo_index.envfile import EnvFileError, load_env_file


def test_reads_keys_comments_and_quotes(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# my keys\n"
        "OPENAI_API_KEY=k-file-1\n"
        "\n"
        'ANTHROPIC_API_KEY="k-file-2"\n'
        "XAI_API_KEY='k-file-3'\n"
        "export PERPLEXITY_API_KEY=k-file-4  # trailing note\n"
        "GEMINI_API_KEY=\n",
        encoding="utf-8",
    )
    merged = load_env_file(path, {})
    assert merged == {
        "OPENAI_API_KEY": "k-file-1",
        "ANTHROPIC_API_KEY": "k-file-2",
        "XAI_API_KEY": "k-file-3",
        "PERPLEXITY_API_KEY": "k-file-4",
        "GEMINI_API_KEY": "",
    }


def test_a_variable_already_in_the_environment_wins(tmp_path):
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=k-file\nXAI_API_KEY=k-file-x\n", encoding="utf-8")
    environ = {"OPENAI_API_KEY": "k-env", "HOME": "/h"}
    merged = load_env_file(path, environ)
    assert merged == {"OPENAI_API_KEY": "k-env", "XAI_API_KEY": "k-file-x", "HOME": "/h"}
    assert environ == {"OPENAI_API_KEY": "k-env", "HOME": "/h"}


def test_os_environ_is_never_touched(tmp_path, monkeypatch):
    monkeypatch.delenv("AEO_INDEX_TEST_ONLY", raising=False)
    path = tmp_path / ".env"
    path.write_text("AEO_INDEX_TEST_ONLY=1\n", encoding="utf-8")
    load_env_file(path, os.environ)
    assert "AEO_INDEX_TEST_ONLY" not in os.environ


def test_a_bad_line_names_its_number_not_its_value(tmp_path):
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=ok\nthis is not a line k-secret-value\n", encoding="utf-8")
    with pytest.raises(EnvFileError) as e:
        load_env_file(path, {})
    assert "line 2" in str(e.value) and "k-secret-value" not in str(e.value)


def test_a_missing_file_is_an_env_file_error(tmp_path):
    with pytest.raises(EnvFileError):
        load_env_file(tmp_path / "absent.env", {})


def test_a_byte_order_mark_is_ignored(tmp_path):
    path = tmp_path / ".env"
    path.write_bytes("\ufeffOPENAI_API_KEY=k-file-1\n".encode())
    assert load_env_file(path, {}) == {"OPENAI_API_KEY": "k-file-1"}


def test_a_variable_set_empty_in_the_environment_does_not_hide_the_file(tmp_path):
    path = tmp_path / "keys.env"
    path.write_text("OPENAI_API_KEY=from-the-file\n", encoding="utf-8")
    assert load_env_file(path, {"OPENAI_API_KEY": "", "PATH": "/bin"})["OPENAI_API_KEY"] == "from-the-file"
    assert load_env_file(path, {"OPENAI_API_KEY": "  "})["OPENAI_API_KEY"] == "from-the-file"
    assert load_env_file(path, {"OPENAI_API_KEY": "from-the-shell"})["OPENAI_API_KEY"] == "from-the-shell"
