"""`table.md`: the first-named table of a run, one table per mode.

Tools are rows and engines are columns, each engine's header carrying its
answered count; a cell counts the answers of that engine that named the
tool first. The search row and the API row are separate tables and are
never summed; "All engines" adds the engines of one mode only. An engine
with no successful answer is listed under its table as "no answers" with
its status counts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

from aeo_index.modes import DISPLAY_MODES, MODE_API, MODE_SEARCH, STATUS_SUCCESS, STATUSES, engine_label
from aeo_index.records import AnswerFacts
from aeo_index.sanitize import md_cell as md_cell
from aeo_index.sanitize import plain, strip_controls

MODE_HEADINGS: Final[Mapping[str, str]] = {
    MODE_SEARCH: "With web search",
    MODE_API: "API only, from training knowledge (no web search)",
}
DEFAULT_TOP: Final[int] = 15


def _status_counts(facts: Sequence[AnswerFacts], engine: str, mode: str) -> Counter[str]:
    return Counter(f.status for f in facts if f.engine == engine and f.mode == mode)


def _failures(counts: Counter[str]) -> str:
    return ", ".join(f"{status} {counts[status]}" for status in STATUSES if status != STATUS_SUCCESS and counts[status])


def _mode_table(block: Mapping[str, Any], facts: Sequence[AnswerFacts], mode: str, top: int) -> list[str]:
    engines = [row["engine"] for row in block["engines"]]
    answered = {row["engine"]: row["ok"] for row in block["engines"]}
    shown = [engine for engine in engines if answered[engine] > 0]
    counts: dict[tuple[str, str], int] = {(r["engine"], r["tool"]): r["c"] for r in block["leaderboard_by_engine"]}
    lines = [f"## {MODE_HEADINGS[mode]}", ""]
    board = block["leaderboard"]
    if not shown:
        lines.append("No engine answered in this mode.")
    elif not board:
        lines.append("No answer named a tool first.")
    else:
        header = ["Tool", *(f"{engine_label(e)} ({answered[e]})" for e in shown), "All engines"]
        lines.append("| " + " | ".join(md_cell(h) for h in header) + " |")
        lines.append("|---|" + "---:|" * (len(shown) + 1))
        for row in board[:top]:
            cells = [
                md_cell(row["tool"]),
                *(str(counts.get((e, row["tool"]), 0)) for e in shown),
                str(row["first_count"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        if len(board) > top:
            more = len(board) - top
            lines.append("")
            lines.append(f"{more} more tool{'s' if more != 1 else ''} named first; see leaderboard.csv.")
    notes = []
    for engine in engines:
        status = _status_counts(facts, engine, mode)
        total = sum(status.values())
        if answered[engine] == 0:
            notes.append(f"{engine_label(engine)}: no answers ({_failures(status) or 'none asked'})")
        elif answered[engine] < total:
            notes.append(f"{engine_label(engine)}: {answered[engine]} of {total} answered ({_failures(status)})")
    if notes:
        lines.append("")
        lines.extend(notes)
    return lines


def render_table(doc: Mapping[str, Any], facts: Sequence[AnswerFacts], *, top: int = DEFAULT_TOP) -> str:
    """The Markdown tables of an edition the runner built, search first.
    `facts` give the status counts of the answers that failed."""
    producer = doc.get("producer") or {}
    set_id = producer.get("question_set") if isinstance(producer, Mapping) else None
    intro = f"Edition {plain(doc.get('edition'))}" + (f", question set {plain(set_id)}" if set_id else "") + "."
    lines = [
        "# Which tool the engines name first",
        "",
        f"{intro} Each number counts the answers that named the tool first. "
        "Answers with web search and answers without it are counted apart and never added together.",
    ]
    for mode in DISPLAY_MODES:
        block = doc["modes"][mode]
        if not block["engines"]:
            continue
        lines.append("")
        lines.extend(_mode_table(block, facts, mode, top))
    return strip_controls("\n".join(lines)) + "\n"
