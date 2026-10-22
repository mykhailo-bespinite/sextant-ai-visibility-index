"""Keeps secrets out of everything the runner stores or prints.

Every error string passes a `Redactor` before it reaches an answer row,
the terminal or a file: each configured key value of 8 or more
characters, the key shapes the vendors issue (`sk-`, `xai-`, `pplx-`,
`AIza`) and `Bearer` and `Basic` credentials all become `[redacted]`.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Iterable, Mapping
from typing import Final

REDACTED: Final[str] = "[redacted]"
# A configured value shorter than this is not treated as a secret: it would
# redact ordinary words.
MIN_SECRET_CHARS: Final[int] = 8

# The variables whose values are secrets.
SECRET_ENV_VARS: Final[tuple[str, ...]] = (
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

_KEY_SHAPES: Final[re.Pattern[str]] = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|xai-[A-Za-z0-9_-]{8,}|pplx-[A-Za-z0-9_-]{8,}|AIza[0-9A-Za-z_-]{20,})"
)
_CREDENTIALS: Final[re.Pattern[str]] = re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+")


class Redactor:
    """`redactor(text)` is `text` with every secret replaced by `[redacted]`."""

    def __init__(self, secrets: Iterable[str]) -> None:
        kept = {s for s in (value.strip() for value in secrets) if len(s) >= MIN_SECRET_CHARS}
        # Longest first, so a secret that contains another is replaced whole.
        self._secrets: tuple[str, ...] = tuple(sorted(kept, key=len, reverse=True))

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        text = _CREDENTIALS.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
        return _KEY_SHAPES.sub(REDACTED, text)


def secret_values(env: Mapping[str, str]) -> list[str]:
    """The secret values in `env`: every key variable's value, and the
    DataForSEO login and password in the base64 form a Basic header
    carries."""
    values = [env[name] for name in SECRET_ENV_VARS if env.get(name)]
    login, password = env.get("DATAFORSEO_LOGIN"), env.get("DATAFORSEO_PASSWORD")
    if login and password:
        values.append(base64.b64encode(f"{login}:{password}".encode()).decode())
    return values
