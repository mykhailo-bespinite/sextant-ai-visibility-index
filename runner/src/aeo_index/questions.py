"""Question sets: what is asked, and the words the judge uses about it.

A question set is a TOML file (spec section 5). The default set is the ten
unbranded questions the index publishes, packaged as
`aeo_index/data/sextant-index-v1.toml`. A user supplies their own category
in the same format.

An index question must not name a tool: a question that names one (any
spelling in the alias table, the file's own aliases, or the tracked brand)
is refused at load, naming its key and the name found. `allow_branded`
loads such a question as experimental instead, so it is answered and
scored but kept out of the leader board. The names of the AI assistants
(ChatGPT, Gemini and the rest) are not brands here.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Final

from aeo_index.aliases import AliasTable

DEFAULT_SET_ID: Final[str] = "sextant-index-v1"
_DEFAULT_SET_FILE: Final[str] = f"{DEFAULT_SET_ID}.toml"

# The judge's category words. The defaults render the index's own judge
# prompt word for word; a question file may replace any of them.
DEFAULT_SUBJECT: Final[str] = "AEO (Answer Engine Optimization) or AI visibility tools"
DEFAULT_PURPOSE: Final[str] = "AEO or AI visibility"
DEFAULT_GENERIC_EXAMPLE: Final[str] = "an AI visibility tool"
DEFAULT_EXCLUDE: Final[tuple[str, ...]] = ("ChatGPT", "Claude", "Perplexity", "Gemini", "Grok")

MIN_QUESTIONS: Final[int] = 1
MAX_QUESTIONS: Final[int] = 50
MIN_TEXT_CHARS: Final[int] = 10
MAX_TEXT_CHARS: Final[int] = 500
MAX_FIELD_CHARS: Final[int] = 200
MAX_LIST_ITEMS: Final[int] = 50
MAX_FILE_BYTES: Final[int] = 1_000_000

_KEY_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9_]{1,32}")
_SET_ID_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_DOMAIN_RE: Final[re.Pattern[str]] = re.compile(r"(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}")

_REQUIRED_FIELDS: Final[tuple[str, ...]] = ("id", "subject", "purpose")
_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "subject",
        "purpose",
        "generic_example",
        "exclude",
        "brand",
        "brand_aliases",
        "brand_domain",
        "aliases",
        "question",
    }
)
_QUESTION_FIELDS: Final[frozenset[str]] = frozenset({"key", "text", "experimental"})


class QuestionFileError(ValueError):
    """A question file that cannot be used, with the reason."""


@dataclass(frozen=True)
class Question:
    """One question: its key, its text, and whether it is kept out of the leader board."""

    key: str
    text: str
    experimental: bool


@dataclass(frozen=True)
class Brand:
    """The brand whose counts fill the edition's own-brand fields.

    Build one from user input with `make_brand`, which checks it."""

    name: str
    aliases: tuple[str, ...]
    domain: str | None

    def spellings(self) -> tuple[str, ...]:
        """The brand's name, its aliases and its domain, without repeats."""
        seen: dict[str, str] = {}
        for spelling in (self.name, *self.aliases, *((self.domain,) if self.domain else ())):
            seen.setdefault(spelling.lower(), spelling)
        return tuple(seen.values())


SEXTANT_BRAND: Final[Brand] = Brand("Sextant AEO", ("Sextant", "sextantaeo.com"), "sextantaeo.com")


@dataclass(frozen=True)
class QuestionSet:
    """A loaded and checked question set.

    `aliases` holds only the file's own `[aliases]`; `alias_table()` adds the
    published table and the brand. To change the brand use `with_brand`,
    never `dataclasses.replace`, which skips the branded-question check.
    Not hashable: its mappings are not."""

    __hash__ = None  # type: ignore[assignment]

    id: str
    subject: str
    purpose: str
    generic_example: str
    exclude: tuple[str, ...]
    brand: Brand | None
    aliases: Mapping[str, tuple[str, ...]]
    questions: tuple[Question, ...]
    _table: AliasTable = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        frozen = MappingProxyType({k: tuple(v) for k, v in self.aliases.items()})
        object.__setattr__(self, "aliases", frozen)
        object.__setattr__(self, "_table", _set_alias_table(frozen, self.brand))

    def alias_table(self) -> AliasTable:
        """The published table plus this set's aliases and its brand."""
        return self._table

    def with_brand(self, brand: Brand | None, *, allow_branded: bool = False, source: str = "--brand") -> QuestionSet:
        """This set tracking `brand` instead (None: no brand). The questions
        are checked again against the new alias table: one that names the
        brand is refused, or marked experimental with `allow_branded`. A
        question already experimental stays so. A brand spelling that
        another tool already owns raises QuestionFileError."""
        try:
            rebuilt = replace(self, brand=brand)
        except ValueError as exc:
            raise QuestionFileError(f"{source}: {exc}") from exc
        return _check_branding(rebuilt, source=source, allow_branded=allow_branded)


def _set_alias_table(aliases: Mapping[str, tuple[str, ...]], brand: Brand | None) -> AliasTable:
    table = AliasTable.default().merged(aliases)
    if brand is not None:
        table = table.merged({brand.name: brand.spellings()})
    return table


def make_brand(name: str, aliases: Sequence[str] = (), domain: str | None = None, *, source: str = "--brand") -> Brand:
    """A checked Brand from user input (the `--brand`, `--brand-alias` and
    `--brand-domain` flags). Raises QuestionFileError with the reason."""
    try:
        return _checked_brand(name, list(aliases) if aliases else None, domain)
    except ValueError as exc:
        raise QuestionFileError(f"{source}: {exc}") from exc


def branded_names(text: str, names: Iterable[str]) -> list[str]:
    """The names in `names` that `text` contains, word-bounded and
    case-insensitive, in the order given and without repeats."""
    found: list[str] = []
    seen: set[str] = set()
    for name in names:
        stripped = name.strip()
        if not stripped or stripped.lower() in seen:
            continue
        if re.search(rf"(?<!\w){re.escape(stripped)}(?!\w)", text, re.IGNORECASE):
            seen.add(stripped.lower())
            found.append(stripped)
    return found


def default_questions() -> QuestionSet:
    """The ten questions the index publishes, tracking Sextant AEO."""
    source = resources.files("aeo_index") / "data" / _DEFAULT_SET_FILE
    return _parse(source.read_text(encoding="utf-8"), source=_DEFAULT_SET_FILE, allow_branded=False)


def load_questions(path: Path, *, allow_branded: bool = False) -> QuestionSet:
    """Load and check a question file. Raises QuestionFileError naming the
    file and the reason."""
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise QuestionFileError(
                f"{path}: the file is {size} bytes; the most a question file may be is {MAX_FILE_BYTES}"
            )
        text = path.read_text(encoding="utf-8")
    except QuestionFileError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        reason = getattr(exc, "strerror", None) or exc.__class__.__name__
        raise QuestionFileError(f"{path}: cannot read the file ({reason})") from exc
    return _parse(text, source=str(path), allow_branded=allow_branded)


def _parse(text: str, *, source: str, allow_branded: bool) -> QuestionSet:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise QuestionFileError(f"{source}: not valid TOML ({exc})") from exc
    try:
        qset = _build(data)
    except (ValueError, TypeError) as exc:
        raise QuestionFileError(f"{source}: {exc}") from exc
    return _check_branding(qset, source=source, allow_branded=allow_branded)


def _build(data: Mapping[str, object]) -> QuestionSet:
    unknown = sorted(set(data) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(
            f"unknown field {unknown[0]!r}; the fields a question file may have are "
            f"{', '.join(sorted(_TOP_LEVEL_FIELDS))}"
        )
    for name in _REQUIRED_FIELDS:
        if name not in data:
            raise ValueError(f"the field {name!r} is required")
    set_id = _text(data["id"], "id")
    if not _SET_ID_RE.fullmatch(set_id):
        raise ValueError(
            f"id {set_id!r} must be 1 to 64 lowercase letters, digits, '-', '_' or '.', starting with a letter or digit"
        )
    return QuestionSet(
        id=set_id,
        subject=_text(data["subject"], "subject"),
        purpose=_text(data["purpose"], "purpose"),
        generic_example=_text(data.get("generic_example", DEFAULT_GENERIC_EXAMPLE), "generic_example"),
        exclude=_names(data.get("exclude", list(DEFAULT_EXCLUDE)), "exclude"),
        brand=_brand(data),
        aliases=_aliases(data.get("aliases", {})),
        questions=_questions(data.get("question", [])),
    )


def _text(value: object, what: str, *, limit: int = MAX_FIELD_CHARS) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    cleaned = value.strip()
    if len(cleaned) > limit:
        raise ValueError(f"{what} is {len(cleaned)} characters; the most it may be is {limit}")
    return cleaned


def _names(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{what} must be a list of one or more names")
    if len(value) > MAX_LIST_ITEMS:
        raise ValueError(f"{what} has {len(value)} names; the most it may have is {MAX_LIST_ITEMS}")
    return tuple(_text(item, f"each name in {what}") for item in value)


def _brand(data: Mapping[str, object]) -> Brand | None:
    if "brand" not in data:
        for name in ("brand_aliases", "brand_domain"):
            if name in data:
                raise ValueError(f'{name} needs a brand; add brand = "..." or remove {name}')
        return None
    return _checked_brand(data["brand"], data.get("brand_aliases"), data.get("brand_domain"))


def _checked_brand(name: object, aliases: object | None, domain: object | None) -> Brand:
    checked_name = _text(name, "brand")
    checked_aliases = _names(aliases, "brand_aliases") if aliases is not None else ()
    checked_domain: str | None = None
    if domain is not None:
        checked_domain = _text(domain, "brand_domain").lower()
        if not _DOMAIN_RE.fullmatch(checked_domain):
            raise ValueError(
                f"brand_domain {checked_domain!r} must be a bare domain such as example.com, with no scheme or path"
            )
    return Brand(checked_name, checked_aliases, checked_domain)


def _aliases(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise ValueError("aliases must be a table of tool name = [spellings]")
    if len(value) > MAX_LIST_ITEMS:
        raise ValueError(f"aliases has {len(value)} tools; the most it may have is {MAX_LIST_ITEMS}")
    return {
        _text(canonical, "each tool name in aliases"): _names(spellings, f"the spellings of {canonical!r}")
        for canonical, spellings in value.items()
    }


def _questions(value: object) -> tuple[Question, ...]:
    if not isinstance(value, list):
        raise ValueError("questions must be [[question]] tables")
    if not MIN_QUESTIONS <= len(value) <= MAX_QUESTIONS:
        raise ValueError(f"the file has {len(value)} questions; it needs {MIN_QUESTIONS} to {MAX_QUESTIONS}")
    questions: list[Question] = []
    seen: set[str] = set()
    for index, entry in enumerate(value, start=1):
        questions.append(_question(entry, index, seen))
    return tuple(questions)


def _question(entry: object, index: int, seen: set[str]) -> Question:
    if not isinstance(entry, dict):
        raise ValueError(f"question {index} must be a [[question]] table")
    unknown = sorted(set(entry) - _QUESTION_FIELDS)
    if unknown:
        raise ValueError(
            f"question {index} has the unknown field {unknown[0]!r}; a question has key, text and experimental"
        )
    key = entry.get("key")
    if not isinstance(key, str) or not _KEY_RE.fullmatch(key):
        raise ValueError(f"question {index} needs a key of 1 to 32 lowercase letters, digits or '_', got {key!r}")
    if key in seen:
        raise ValueError(f"the key {key!r} is used twice; keys must be unique")
    seen.add(key)
    text = entry.get("text")
    if not isinstance(text, str):
        raise ValueError(f"question {key} needs a text")
    text = text.strip()
    if not MIN_TEXT_CHARS <= len(text) <= MAX_TEXT_CHARS:
        raise ValueError(
            f"question {key} is {len(text)} characters; a question needs {MIN_TEXT_CHARS} to {MAX_TEXT_CHARS}"
        )
    experimental = entry.get("experimental", False)
    if not isinstance(experimental, bool):
        raise ValueError(f"question {key}: experimental must be true or false")
    return Question(key, text, experimental)


def _check_branding(qset: QuestionSet, *, source: str, allow_branded: bool) -> QuestionSet:
    names = qset.alias_table().names()
    questions: list[Question] = []
    for question in qset.questions:
        found = branded_names(question.text, names)
        if not found:
            questions.append(question)
        elif allow_branded:
            questions.append(replace(question, experimental=True))
        else:
            raise QuestionFileError(
                f"{source}: question {question.key} names {found[0]!r}. An index question must not name a tool; "
                "rephrase it, or pass --allow-branded to load it as experimental"
            )
    return replace(qset, questions=tuple(questions))
