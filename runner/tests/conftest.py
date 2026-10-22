"""Shared test helpers.

pytest-asyncio runs in auto mode (`asyncio_mode = "auto"` in
`pyproject.toml`), so an `async def test_...` needs no marker. Sockets are
disabled for every test (`--disable-socket` in `addopts`); unix sockets stay
allowed because the asyncio event loop needs a socketpair to start.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pytest

DEFAULT_FIELDS: dict[str, object] = {
    "id": "crm-for-agencies-v1",
    "subject": "CRM software for marketing agencies",
    "purpose": "running a marketing agency's client relationships",
}


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"no TOML form for {value!r}")


def toml_text(
    *,
    questions: Iterable[Sequence[object]] = (("q01", "Which CRM fits a small agency?"),),
    aliases: Mapping[str, Sequence[str]] | None = None,
    omit: Iterable[str] = (),
    **fields: object,
) -> str:
    """A question file. `questions` rows are (key, text) or (key, text,
    experimental); `fields` override or add top-level fields."""
    top = {**DEFAULT_FIELDS, **fields}
    for name in omit:
        top.pop(name, None)
    lines = [f"{name} = {_toml_value(value)}" for name, value in top.items()]
    if aliases:
        lines.append("")
        lines.append("[aliases]")
        lines.extend(f"{json.dumps(k)} = {_toml_value(list(v))}" for k, v in aliases.items())
    for row in questions:
        lines.append("")
        lines.append("[[question]]")
        lines.append(f"key = {_toml_value(row[0])}")
        lines.append(f"text = {_toml_value(row[1])}")
        if len(row) > 2:
            lines.append(f"experimental = {_toml_value(row[2])}")
    return "\n".join(lines) + "\n"


def write_toml(tmp_path: Path, name: str = "questions.toml", **kwargs: object) -> Path:
    path = tmp_path / name
    path.write_text(toml_text(**kwargs), encoding="utf-8")  # type: ignore[arg-type]
    return path


@pytest.fixture
def toml_writer(tmp_path: Path):
    def _write(name: str = "questions.toml", **kwargs: object) -> Path:
        return write_toml(tmp_path, name, **kwargs)

    return _write


# Every variable the runner reads. The CLI tests clear them all first, so a
# developer's own keys never leak into a test run.
RUNNER_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
    "GROK_API_KEY",
    "PERPLEXITY_API_KEY",
    "DATAFORSEO_LOGIN",
    "DATAFORSEO_PASSWORD",
)


@pytest.fixture
def recorder():
    from vendor_doubles import Recorder

    return Recorder()


def _set_env(monkeypatch: pytest.MonkeyPatch, names: Iterable[str]) -> dict[str, str]:
    from vendor_doubles import FAKE_ENV

    for name in RUNNER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    chosen = {name: FAKE_ENV[name] for name in names}
    for name, value in chosen.items():
        monkeypatch.setenv(name, value)
    return chosen


@pytest.fixture
def env_none(monkeypatch):
    return _set_env(monkeypatch, ())


@pytest.fixture
def env_openai_only(monkeypatch):
    return _set_env(monkeypatch, ("OPENAI_API_KEY",))


@pytest.fixture
def env_with_two_keys(monkeypatch):
    return _set_env(monkeypatch, ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"))


@pytest.fixture
def env_all_keys(monkeypatch):
    from vendor_doubles import FAKE_ENV

    return _set_env(monkeypatch, tuple(FAKE_ENV))


@pytest.fixture
def no_network(monkeypatch):
    """Fails the test if anything tries to send an HTTP request."""
    import httpx

    def _refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("an HTTP request was sent")

    monkeypatch.setattr(httpx.AsyncClient, "send", _refuse)
    monkeypatch.setattr(httpx.Client, "send", _refuse)
