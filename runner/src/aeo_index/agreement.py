"""The `agreement` block of an edition: how often the samples of one prompt
agree.

Every engine is asked every question several times a night. A *cell* is one
(run date, engine, mode, question); its samples are the answers with status
`success` that the judge listed tools for. Two measures per cell:

- `first_agreement`: 1.0 when every sample names the same tool first (after
  the alias table folds the spelling), 0.5 when exactly two of three do, 0.0
  otherwise; in general (largest group - 1) / (samples - 1). A sample that
  names no tool counts as the value "none". Undefined (None) with fewer than
  two samples, or when no sample names a tool first.
- `set_agreement`: the mean pairwise Jaccard index of the sets of tools the
  samples name (two empty sets are 1.0; an empty and a non-empty one 0.0).
  None with fewer than two samples.

A successful answer the judge never listed tools for (`tools_named` is None)
is not a sample: an unjudged answer is not an empty one.

Aggregates are the mean of the defined cells, with the count: `cells` counts
the cells with two or more samples (what `set_agreement` averages) and
`first_cells` those where `first_agreement` is defined. Means are exact
(fractions) and rounded to three decimals, so the block does not depend on
the order of the facts. Every question counts, experimental ones included
(the leaderboards count the fixed questions only). Groups are per (engine,
mode), per (question, mode) and per mode: an api answer and a search answer
are never pooled. The Sextant backend computes the same block with the
same definitions; `tests/fixtures/agreement_cases.json` holds the two equal.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from itertools import combinations
from typing import Any, Final

from aeo_index.aliases import normalise
from aeo_index.modes import DISPLAY_MODES, ENGINES, STATUS_SUCCESS
from aeo_index.records import AnswerFacts

DECIMALS: Final[int] = 3
# The value a sample that names no tool counts as; no tool name is empty.
NO_TOOL: Final[str] = ""

DEFINITION: Final[Mapping[str, str]] = {
    "first_agreement": (
        "For one prompt asked of one engine in one mode on one night, over the answers that succeeded and whose "
        "tools the judge listed: 1.0 when every such answer names the same tool first, 0.5 when two of three do, "
        "0.0 otherwise; not defined with fewer than two such answers or when none of them names a tool."
    ),
    "set_agreement": (
        "For the same cell and the same answers, the average overlap (Jaccard index) between the sets of tools each "
        "pair of answers names; 1.0 means identical sets and 0.0 means nothing in common; not defined with fewer "
        "than two such answers."
    ),
}

_Cell = tuple[Fraction | None, Fraction | None]
_ROSTER: Final[Mapping[str, int]] = {engine: index for index, engine in enumerate(ENGINES)}
_MODE_ORDER: Final[Mapping[str, int]] = {mode: index for index, mode in enumerate(DISPLAY_MODES)}


def _samples(cell: Iterable[AnswerFacts]) -> list[AnswerFacts]:
    return [f for f in cell if f.status == STATUS_SUCCESS and f.tools_named is not None]


def _first(samples: Sequence[AnswerFacts]) -> Fraction | None:
    if len(samples) < 2:
        return None
    named = [normalise(f.first_named) if f.first_named else NO_TOOL for f in samples]
    if all(value == NO_TOOL for value in named):
        return None
    return Fraction(max(Counter(named).values()) - 1, len(samples) - 1)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> Fraction:
    return Fraction(len(a & b), len(a | b)) if a | b else Fraction(1)


def _set(samples: Sequence[AnswerFacts]) -> Fraction | None:
    if len(samples) < 2:
        return None
    sets = [frozenset(normalise(tool) for tool in f.tools_named or ()) for f in samples]
    pairs = list(combinations(sets, 2))
    return sum((_jaccard(a, b) for a, b in pairs), Fraction(0)) / len(pairs)


def cell_agreement(cell: Iterable[AnswerFacts]) -> _Cell:
    """(first_agreement, set_agreement) of one cell's answers; each None
    where undefined."""
    samples = _samples(cell)
    return _first(samples), _set(samples)


def _mean(values: Sequence[Fraction]) -> float | None:
    return round(float(sum(values, Fraction(0)) / len(values)), DECIMALS) if values else None


def _row(cells: Sequence[_Cell], **head: str) -> dict[str, Any]:
    firsts = [first for first, _ in cells if first is not None]
    sets = [set_ for _, set_ in cells if set_ is not None]
    return {
        **head,
        "first_agreement": _mean(firsts),
        "set_agreement": _mean(sets),
        "cells": len(sets),
        "first_cells": len(firsts),
    }


def _rows(
    cells: Sequence[tuple[tuple[str, str, str], _Cell]],
    names: Sequence[str],
    pick: Callable[[str, str, str], tuple[str, ...]],
    order: Callable[[tuple[str, ...]], tuple],
) -> list[dict[str, Any]]:
    """One row per group of cells: `pick` takes (engine, mode, key) to the
    group's identity, `names` name its parts, `order` sorts the groups."""
    grouped: dict[tuple[str, ...], list[_Cell]] = defaultdict(list)
    for ident, value in cells:
        grouped[pick(*ident)].append(value)
    return [_row(grouped[group], **dict(zip(names, group, strict=True))) for group in sorted(grouped, key=order)]


def _engine_rank(engine: str) -> tuple[int, str]:
    return _ROSTER.get(engine, len(_ROSTER)), engine


def _mode_rank(mode: str) -> tuple[int, str]:
    return _MODE_ORDER.get(mode, len(_MODE_ORDER)), mode


def agreement_block(facts: Iterable[AnswerFacts]) -> dict[str, Any]:
    """The `agreement` block of an edition's facts. Engines list in roster
    order, modes in display order (web search first), questions by key."""
    by_cell: dict[tuple[str, str, str, str], list[AnswerFacts]] = defaultdict(list)
    for fact in facts:
        by_cell[(fact.run_date, fact.engine, fact.mode, fact.key)].append(fact)
    cells = [((engine, mode, key), cell_agreement(rows)) for (_, engine, mode, key), rows in by_cell.items()]
    return {
        "definition": dict(DEFINITION),
        "by_engine": _rows(
            cells, ("engine", "mode"), lambda e, m, k: (e, m), lambda g: (_mode_rank(g[1]), _engine_rank(g[0]))
        ),
        "by_prompt": _rows(cells, ("key", "mode"), lambda e, m, k: (k, m), lambda g: (g[0], _mode_rank(g[1]))),
        "by_mode": _rows(cells, ("mode",), lambda e, m, k: (m,), lambda g: _mode_rank(g[0])),
    }
