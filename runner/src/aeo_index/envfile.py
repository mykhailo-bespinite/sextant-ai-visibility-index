"""`--env-file`: read `KEY=VALUE` lines without touching the process
environment.

A line is `KEY=VALUE` or `export KEY=VALUE`; blank lines and lines starting
with `#` are skipped; a value may be wrapped in single or double quotes
(taken as written inside them); an unquoted value ends at ` #`. A variable
already set (not blank) in the environment wins over the file. Nothing is
loaded unless the user names the file.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final

MAX_ENV_FILE_BYTES: Final[int] = 100_000
_LINE: Final[re.Pattern[str]] = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)")


class EnvFileError(ValueError):
    """An env file that cannot be read. The message never holds a value."""


def load_env_file(path: Path, environ: Mapping[str, str]) -> dict[str, str]:
    """The file's variables merged under `environ` (a variable already in
    `environ` wins). Returns a new mapping; `environ` is not changed."""
    try:
        if path.stat().st_size > MAX_ENV_FILE_BYTES:
            raise EnvFileError(f"{path}: the file is larger than {MAX_ENV_FILE_BYTES} bytes")
        text = path.read_text(encoding="utf-8-sig")
    except EnvFileError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        reason = getattr(exc, "strerror", None) or exc.__class__.__name__
        raise EnvFileError(f"{path}: cannot read the file ({reason})") from exc
    from_file: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.fullmatch(line)
        if match is None:
            raise EnvFileError(f"{path}: line {number} is not KEY=VALUE")
        from_file[match.group(1)] = _value(match.group(2), path, number)
    # A variable set but empty in the environment does not hide the file's value.
    return {**from_file, **{name: value for name, value in environ.items() if value.strip() or name not in from_file}}


def _value(raw: str, path: Path, number: int) -> str:
    if raw[:1] in ("'", '"'):
        quote = raw[0]
        end = raw.find(quote, 1)
        if end == -1:
            raise EnvFileError(f"{path}: line {number} opens a quote it does not close")
        rest = raw[end + 1 :].strip()
        if rest and not rest.startswith("#"):
            raise EnvFileError(f"{path}: line {number} has text after its closing quote")
        return raw[1:end]
    comment = raw.find(" #")
    return (raw[:comment] if comment != -1 else raw).strip()
