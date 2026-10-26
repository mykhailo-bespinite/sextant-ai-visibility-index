"""The checks the research page makes before it shows an edition, as a
function that returns every failure as a sentence instead of throwing on
the first one. A legacy edition (no `modes`) is checked on its legacy keys
only."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from aeo_index.modes import MODES


def is_legacy(doc: Mapping[str, Any]) -> bool:
    """An edition from before the per-mode blocks (no `modes`)."""
    return "modes" not in doc


class _Checker:
    """The research page's checks, collecting every failure as a sentence."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def fail(self, what: str) -> None:
        self.errors.append(what)

    def list(self, obj: Mapping[str, Any], key: str, where: str = "snapshot") -> list[Any]:
        value = obj.get(key)
        if not isinstance(value, list):
            self.fail(f'{where}: "{key}" is not a list')
            return []
        return value

    def rows(self, obj: Mapping[str, Any], key: str, where: str = "snapshot") -> list[Mapping[str, Any]]:
        out: list[Mapping[str, Any]] = []
        for index, row in enumerate(self.list(obj, key, where)):
            if isinstance(row, Mapping):
                out.append(row)
            else:
                self.fail(f"{where}.{key}[{index}] is not an object")
        return out

    def text(self, row: Mapping[str, Any], key: str, where: str) -> str | None:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            self.fail(f"{where}.{key} is not a non-empty string")
            return None
        return value

    def count(self, row: Mapping[str, Any], key: str, where: str) -> int:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            self.fail(f"{where}.{key} is not a count")
            return 0
        return value

    def flag(self, row: Mapping[str, Any], key: str, where: str) -> bool:
        value = row.get(key)
        if not isinstance(value, bool):
            self.fail(f"{where}.{key} is not a boolean")
            return False
        return value

    def iso_date(self, row: Mapping[str, Any], key: str, where: str) -> str | None:
        value = self.text(row, key, where)
        if value is None:
            return None
        try:
            date.fromisoformat(value)
        except ValueError:
            self.fail(f"{where}.{key} is not a YYYY-MM-DD date")
            return None
        if len(value) != len("YYYY-MM-DD"):
            self.fail(f"{where}.{key} is not a YYYY-MM-DD date")
            return None
        return value

    def optional_date(self, row: Mapping[str, Any], key: str, where: str) -> str | None:
        return None if row.get(key) is None else self.iso_date(row, key, where)

    def share(self, row: Mapping[str, Any], key: str, where: str) -> None:
        """A share between 0 and 1, or null (undefined)."""
        value = row.get(key)
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
            self.fail(f"{where}.{key} is not a share between 0 and 1")

    def mode(self, value: object, where: str) -> str | None:
        if not isinstance(value, str) or value not in MODES:
            self.fail(f'{where}: unknown mode "{value}"')
            return None
        return value


def _check_legacy_keys(c: _Checker, doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    generated = c.text(doc, "generated_at", "snapshot")
    if generated is not None:
        try:
            datetime.fromisoformat(generated)
        except ValueError:
            c.fail("generated_at is not a date")
    if "edition" in doc or not is_legacy(doc):
        c.iso_date(doc, "edition", "snapshot")
    if "engines_under_review" in doc or not is_legacy(doc):
        for index, value in enumerate(c.list(doc, "engines_under_review")):
            if not isinstance(value, str) or not value:
                c.fail(f"engines_under_review[{index}] is not a non-empty string")
    for run in c.rows(doc, "runs"):
        c.text(run, "run_date", "runs")
        c.text(run, "status", "runs")
        c.count(run, "engines_answered", "runs")
    for engine in c.rows(doc, "engines"):
        c.text(engine, "engine", "engines")
        for key in ("answers", "ok", "sextant_named", "sextant_first"):
            c.count(engine, key, "engines")
    prompts = []
    for prompt in c.rows(doc, "prompts"):
        prompts.append(
            {
                "key": c.text(prompt, "key", "prompts"),
                "text": c.text(prompt, "prompt_text", "prompts"),
                "experimental": c.flag(prompt, "experimental", "prompts"),
            }
        )
    for row in c.rows(doc, "leaderboard"):
        c.text(row, "tool", "leaderboard")
        c.count(row, "first_count", "leaderboard")
    for row in c.rows(doc, "leaderboard_by_engine"):
        c.text(row, "engine", "leaderboard_by_engine")
        c.text(row, "tool", "leaderboard_by_engine")
        c.count(row, "c", "leaderboard_by_engine")
    for row in c.rows(doc, "top_domains"):
        c.text(row, "domain", "top_domains")
        c.count(row, "n", "top_domains")
        c.flag(row, "sextant", "top_domains")
    for row in c.rows(doc, "top_urls"):
        c.text(row, "url", "top_urls")
        c.count(row, "n", "top_urls")
        c.text(row, "engines", "top_urls")
    return prompts


def _check_modes(c: _Checker, doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    modes = doc.get("modes")
    if not isinstance(modes, Mapping):
        c.fail("snapshot.modes is not an object")
        return []
    rows: list[dict[str, Any]] = []
    for mode_id, block in modes.items():
        mode = c.mode(mode_id, "modes")
        if mode is None:
            continue
        if not isinstance(block, Mapping):
            c.fail(f"modes.{mode_id} is not an object")
            continue
        where = f"modes.{mode_id}.engines"
        for engine in c.rows(block, "engines", f"modes.{mode_id}"):
            if "mode" in engine and engine["mode"] != mode_id:
                c.fail(f'{where} has a row for mode "{engine["mode"]}"')
            rows.append(
                {
                    "mode": mode,
                    "engine": c.text(engine, "engine", where),
                    **{key: c.count(engine, key, where) for key in ("answers", "ok", "sextant_named", "sextant_first")},
                }
            )
        if "top_domains" in block:
            for row in c.rows(block, "top_domains", f"modes.{mode_id}"):
                c.text(row, "domain", f"modes.{mode_id}.top_domains")
                c.count(row, "n", f"modes.{mode_id}.top_domains")
                c.flag(row, "sextant", f"modes.{mode_id}.top_domains")
    return rows


def _check_cell(c: _Checker, row: Mapping[str, Any]) -> dict[str, Any] | None:
    where = "by_prompt"
    before = len(c.errors)
    cell = {
        "key": c.text(row, "key", where),
        "mode": c.mode(row.get("mode"), where),
        "engine": c.text(row, "engine", where),
        **{key: c.count(row, key, where) for key in ("answers", "ok", "sextant_named", "sextant_first", "days")},
        "first_run": c.optional_date(row, "first_run", where),
        "last_run": c.optional_date(row, "last_run", where),
    }
    ranks = row.get("sextant_ranks")
    if not isinstance(ranks, list) or not all(isinstance(r, int) and not isinstance(r, bool) and r >= 1 for r in ranks):
        c.fail(f"{where}.sextant_ranks has a rank that is not a positive integer")
        ranks = []
    if len(c.errors) > before:
        return None
    name = f"{cell['key']} {cell['mode']} {cell['engine']}"
    if len(ranks) != cell["sextant_named"]:
        c.fail(f"{where} {name}: sextant_ranks has {len(ranks)} entries for sextant_named {cell['sextant_named']}")
    if sum(1 for r in ranks if r == 1) != cell["sextant_first"]:
        c.fail(f"{where} {name}: sextant_ranks does not hold sextant_first ranks of 1")
    if cell["ok"] > cell["answers"] or cell["sextant_named"] > cell["ok"]:
        c.fail(f"{where} {name}: more answered or named than asked")
    if cell["answers"] > 0 and (cell["first_run"] is None or cell["last_run"] is None):
        c.fail(f"{where} {name}: first_run/last_run missing on a cell with answers")
    return cell


def _reconcile(
    c: _Checker,
    prompts: Sequence[Mapping[str, Any]],
    mode_engines: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    board: Sequence[Mapping[str, Any]],
) -> None:
    keys: set[str] = set()
    for prompt in prompts:
        if prompt["key"] in keys:
            c.fail(f'duplicate prompt key "{prompt["key"]}"')
        keys.add(prompt["key"])
    by_id: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        if cell["key"] not in keys:
            c.fail(f'by_prompt names prompt "{cell["key"]}", which the edition does not list')
        name = f"{cell['key']} {cell['mode']} {cell['engine']}"
        if name in by_id:
            c.fail(f"by_prompt has {name} twice")
        by_id[name] = cell
    fields = ("answers", "ok", "sextant_named", "sextant_first")
    sums: dict[str, Counter[str]] = defaultdict(Counter)
    for cell in cells:
        for f in fields:
            sums[f"{cell['mode']} {cell['engine']}"][f] += cell[f]
    blocks: set[str] = set()
    for row in mode_engines:
        name = f"{row['mode']} {row['engine']}"
        if name in blocks:
            c.fail(f"modes.{row['mode']}.engines has {row['engine']} twice")
        blocks.add(name)
        for f in fields:
            if sums[name][f] != row[f]:
                c.fail(
                    f"by_prompt rows for {name} do not add up to modes.{row['mode']}: "
                    f"{f} {sums[name][f]} against {row[f]}"
                )
    for name in sums:
        if name not in blocks:
            c.fail(f"by_prompt rows for {name} do not add up to modes: the mode block has no such engine")
    named: Counter[str] = Counter()
    for row in board:
        name = f"{row['key']} {row['mode']} {row['engine']}"
        if name not in by_id:
            c.fail(f"leaderboard_by_prompt has {name}, which by_prompt does not")
            continue
        named[name] += row["c"]
    for name, n in named.items():
        if n > by_id[name]["ok"]:
            c.fail(f"leaderboard_by_prompt names a tool first {n} times for {name}, more than its answers")


def _check_board(c: _Checker, doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    where = "leaderboard_by_prompt"
    out = []
    for row in c.rows(doc, where):
        before = len(c.errors)
        parsed = {
            "key": c.text(row, "key", where),
            "mode": c.mode(row.get("mode"), where),
            "engine": c.text(row, "engine", where),
            "tool": c.text(row, "tool", where),
            "c": c.count(row, "c", where),
        }
        if len(c.errors) == before:
            out.append(parsed)
    return out


_AGREEMENT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("by_engine", ("engine", "mode")),
    ("by_engine_overall", ("engine",)),
    ("by_prompt", ("key", "mode")),
    ("by_mode", ("mode",)),
)


def _check_agreement(c: _Checker, doc: Mapping[str, Any]) -> None:
    """The optional `agreement` block: each group a list of rows naming
    their group, two shares that may be null, and the cell counts."""
    block = doc.get("agreement")
    if not isinstance(block, Mapping):
        c.fail("snapshot.agreement is not an object")
        return
    definition = block.get("definition")
    if not isinstance(definition, Mapping):
        c.fail("agreement.definition is not an object")
    else:
        for key in ("first_agreement", "set_agreement"):
            c.text(definition, key, "agreement.definition")
    for group, names in _AGREEMENT_GROUPS:
        where = f"agreement.{group}"
        for row in c.rows(block, group, "agreement"):
            for name in names:
                if name == "mode":
                    c.mode(row.get("mode"), where)
                else:
                    c.text(row, name, where)
            c.share(row, "first_agreement", where)
            c.share(row, "set_agreement", where)
            c.count(row, "cells", where)
            c.count(row, "first_cells", where)


def validate_edition(doc: Mapping[str, Any]) -> list[str]:
    """Every reason the research page would refuse `doc`, as sentences;
    empty when it would show it. `sacs`, which only the publisher writes,
    is not required (but is checked when present). A legacy edition (no
    `modes`) is checked on its legacy keys only."""
    if not isinstance(doc, Mapping):
        return ["snapshot: not an object"]
    c = _Checker()
    prompts = _check_legacy_keys(c, doc)
    if "sacs" in doc:
        sacs = doc["sacs"]
        if isinstance(sacs, Mapping):
            c.count(sacs, "evaluations", "sacs")
        else:
            c.fail("snapshot.sacs is not an object")
    if "agreement" in doc:
        _check_agreement(c, doc)
    if is_legacy(doc):
        return c.errors
    mode_engines = _check_modes(c, doc)
    cells = [cell for row in c.rows(doc, "by_prompt") if (cell := _check_cell(c, row)) is not None]
    board = _check_board(c, doc)
    if not c.errors:
        _reconcile(c, prompts, mode_engines, cells, board)
    return c.errors


def validate_for_compare(doc: Mapping[str, Any]) -> list[str]:
    """`validate_edition`, plus the fields a comparison reads that the
    research page does not check: each mode block's `leaderboard` rows
    (a tool name and a count)."""
    errors = validate_edition(doc)
    if errors or is_legacy(doc):
        return errors
    c = _Checker()
    for mode_id, block in doc["modes"].items():
        where = f"modes.{mode_id}.leaderboard"
        for row in c.rows(block, "leaderboard", f"modes.{mode_id}"):
            c.text(row, "tool", where)
            c.count(row, "first_count", where)
    return c.errors
