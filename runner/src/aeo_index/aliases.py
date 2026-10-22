"""The alias table: how spellings of one tool fold to one name.

A model asked to list the tools an answer recommends spells the same product
differently from call to call ("Semrush" and "Semrush AI Toolkit",
"sextantaeo.com" and "Sextant AEO"). `normalise` folds every known spelling
to one canonical name, so a leader board is not split across near
duplicates. A name the table does not know is kept as written (trimmed,
never re-cased): the table never guesses.

Matching is case-insensitive and ignores surrounding whitespace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

# Canonical name -> every known spelling, the canonical name first.
KNOWN_TOOLS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "Profound": ("Profound",),
        "Peec AI": ("Peec AI", "Peec"),
        "Otterly": ("Otterly", "Otterly.AI", "Otterly AI"),
        "Scrunch": ("Scrunch", "Scrunch AI"),
        "RankSpot": ("RankSpot",),
        "OpenSEO": ("OpenSEO",),
        "Semrush AI Toolkit": ("Semrush AI Toolkit", "Semrush"),
        "Ahrefs Brand Radar": ("Ahrefs Brand Radar", "Ahrefs"),
        "Similarweb": ("Similarweb",),
        "Conductor": ("Conductor",),
        "BrightEdge": ("BrightEdge",),
        "Sextant AEO": ("Sextant AEO", "Sextant", "sextantaeo.com"),
    }
)


def _fold(name: str) -> str:
    return name.strip().lower()


def _clean(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value.strip()


def _canonical_first(canonical: str, spellings: Sequence[str]) -> tuple[str, ...]:
    """`spellings` with `canonical` first and case-duplicates removed."""
    kept: dict[str, str] = {}
    for spelling in (canonical, *spellings):
        kept.setdefault(_fold(spelling), spelling)
    return tuple(kept.values())


def _build_lookup(tools: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, spellings in tools.items():
        for spelling in spellings:
            owner = lookup.setdefault(_fold(spelling), canonical)
            if owner != canonical:
                raise ValueError(
                    f"the spelling {spelling!r} is claimed by both {canonical!r} and {owner!r}; "
                    "give it to one tool only"
                )
    return lookup


@dataclass(frozen=True)
class AliasTable:
    """An alias table: canonical name -> spellings, the canonical name
    first (added when a caller leaves it out).

    Immutable: `merged` returns a new table. Not hashable."""

    tools: Mapping[str, tuple[str, ...]]
    _lookup: Mapping[str, str] = field(init=False, repr=False, compare=False)

    __hash__ = None  # type: ignore[assignment]  # mappings are not hashable

    def __post_init__(self) -> None:
        frozen = MappingProxyType({k: _canonical_first(k, v) for k, v in self.tools.items()})
        object.__setattr__(self, "tools", frozen)
        object.__setattr__(self, "_lookup", MappingProxyType(_build_lookup(frozen)))

    @classmethod
    def default(cls) -> AliasTable:
        """The published table."""
        return _DEFAULT_TABLE

    def merged(self, extra: Mapping[str, Sequence[str]]) -> AliasTable:
        """A new table with `extra`'s spellings added. A canonical name
        already in the table gains the new spellings; a new one is added
        after the existing tools. A spelling that would belong to two
        tools raises ValueError."""
        tools: dict[str, tuple[str, ...]] = dict(self.tools)
        for raw_canonical, raw_spellings in extra.items():
            canonical = _clean(raw_canonical, "a tool name")
            if isinstance(raw_spellings, str):
                raise ValueError(f"the spellings of {canonical!r} must be a list of names")
            spellings = [_clean(s, f"a spelling of {canonical!r}") for s in raw_spellings]
            # The new table drops case-duplicates, keeping the first spelling.
            tools[canonical] = (*tools.get(canonical, ()), *spellings)
        return AliasTable(tools)

    def normalise(self, name: str) -> str:
        """`name`'s canonical form, or `name` trimmed when the table does
        not know it."""
        candidate = (name or "").strip()
        if not candidate:
            return candidate
        return self._lookup.get(candidate.lower(), candidate)

    def spellings(self, canonical: str) -> tuple[str, ...]:
        """Every spelling of `canonical` (itself first); empty for a name
        the table does not know. `canonical` must be the exact canonical
        form, as `normalise` returns it. To ground a raw name against an
        answer, check `{raw, *spellings(normalise(raw))}`."""
        return self.tools.get(canonical, ())

    def names(self) -> tuple[str, ...]:
        """Every spelling of every tool, canonical names included, in table
        order."""
        return tuple(spelling for spellings in self.tools.values() for spelling in spellings)


_DEFAULT_TABLE: Final[AliasTable] = AliasTable(KNOWN_TOOLS)


def normalise(name: str) -> str:
    """`name`'s canonical form in the published table."""
    return _DEFAULT_TABLE.normalise(name)
