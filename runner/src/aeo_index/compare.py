"""`--compare`: a run's first-named table beside a published edition.

`resolve_edition` finds the edition: `latest` is the newest dated folder in
the repository's `data/`, a `YYYY-MM-DD` date is that folder's
`index.json`, anything else is a local path (a file, or a folder holding
`index.json`). These are the only calls the package makes outside the
vendor APIs, and only when asked to compare.

`render_compare` sets the two side by side per mode when the edition has a
`modes` block, and against the edition's top level, its basis stated,
when it is a legacy edition without one. A share is a tool's first-named
count over that side's own successful answers, so runs of different sizes
can be read together. When the two sides asked different questions the
header says so and only the shares are shown.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Final

import httpx

from aeo_index import __version__
from aeo_index.edition import MODE_BASIS, TOP_LEVEL_BASIS
from aeo_index.modes import DISPLAY_MODES, MODE_API, MODE_SEARCH, engine_label
from aeo_index.sanitize import md_cell, plain, strip_controls
from aeo_index.table import MODE_HEADINGS
from aeo_index.validate import is_legacy, validate_for_compare

REPO: Final[str] = "mykhailo-bespinite/sextant-ai-visibility-index"
BRANCH: Final[str] = "master"
CONTENTS_URL: Final[str] = f"https://api.github.com/repos/{REPO}/contents/data"
RAW_EDITION_URL: Final[str] = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data/{{date}}/index.json"
EDITION_FILE: Final[str] = "index.json"
# The largest edition file read; a published one is well under a megabyte.
MAX_EDITION_BYTES: Final[int] = 20_000_000
FETCH_TIMEOUT_S: Final[float] = 30.0
DEFAULT_TOP: Final[int] = 10
_ISO_DATE: Final[re.Pattern[str]] = re.compile(r"\d{4}-\d{2}-\d{2}")
_HEADERS: Final[Mapping[str, str]] = {
    "Accept": "application/vnd.github+json",
    "User-Agent": f"aeo-index/{__version__}",
}
_YOURS: Final[str] = "Your run"


class CompareError(Exception):
    """The edition to compare with could not be read; the message says why.
    It can quote the file, so its control characters are removed."""

    def __init__(self, message: str) -> None:
        super().__init__(strip_controls(message))


async def _get(client: httpx.AsyncClient, url: str, what: str, params: Mapping[str, str] | None = None) -> Any:
    """GET `url` as JSON, refusing a body over `MAX_EDITION_BYTES` while
    it downloads."""
    body = bytearray()
    try:
        async with client.stream(
            "GET", url, params=params, headers=dict(_HEADERS), timeout=FETCH_TIMEOUT_S
        ) as response:
            if response.status_code != 200:
                raise CompareError(f"{what} answered HTTP {response.status_code}")
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_EDITION_BYTES:
                    raise CompareError(f"{what} is larger than {MAX_EDITION_BYTES} bytes")
    except httpx.TransportError as exc:
        raise CompareError(f"{what} could not be reached ({exc.__class__.__name__})") from exc
    except httpx.DecodingError as exc:
        raise CompareError(f"{what} sent a body that could not be decoded ({exc.__class__.__name__})") from exc
    except httpx.HTTPError as exc:
        raise CompareError(f"{what} could not be read ({exc.__class__.__name__})") from exc
    try:
        return json.loads(bytes(body))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise CompareError(f"{what} is not JSON") from exc


async def _latest(client: httpx.AsyncClient) -> str:
    listing = await _get(client, CONTENTS_URL, "the edition list on GitHub", params={"ref": BRANCH})
    if not isinstance(listing, list):
        raise CompareError("the edition list on GitHub is not a list")
    dates = sorted(
        entry["name"]
        for entry in listing
        if isinstance(entry, Mapping)
        and isinstance(entry.get("name"), str)
        and entry.get("type", "dir") == "dir"
        and _is_date(entry["name"])
    )
    if not dates:
        raise CompareError("the repository's data folder holds no edition")
    return dates[-1]


def _is_date(text: str) -> bool:
    if not _ISO_DATE.fullmatch(text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _read_local(ref: str) -> dict[str, Any]:
    path = Path(ref)
    if path.is_dir():
        path = path / EDITION_FILE
    if not path.is_file():
        raise CompareError(f"there is no file {path}")
    try:
        if path.stat().st_size > MAX_EDITION_BYTES:
            raise CompareError(f"{path} is larger than {MAX_EDITION_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompareError(f"{path} cannot be read as JSON ({exc.__class__.__name__})") from exc


def _local_label(path: Path, doc: object) -> str:
    """`Edition <date>` from the file's own edition date, or from its folder
    name (a published edition sits in a folder named by its date)."""
    when = doc.get("edition") if isinstance(doc, dict) else None
    if not isinstance(when, str) or not _is_date(when):
        folder = path if path.is_dir() else path.parent
        when = folder.name if _is_date(folder.name) else path.name
    return f"Edition {when}"


async def resolve_edition(ref: str, *, client: httpx.AsyncClient) -> tuple[str, dict[str, Any]]:
    """(label, edition) for `latest`, a `YYYY-MM-DD` date, or a local path.
    A published edition is labelled `Sextant edition <date>`, a local file
    `Edition <date>`. Raises CompareError when it cannot be read or is not
    a valid edition."""
    if ref == "latest" or _ISO_DATE.fullmatch(ref):
        if ref != "latest" and not _is_date(ref):
            raise CompareError(f"{ref!r} is not a date")
        when = await _latest(client) if ref == "latest" else ref
        label = f"Sextant edition {when}"
        doc = await _get(client, RAW_EDITION_URL.format(date=when), f"the edition {when}")
    else:
        doc = _read_local(ref)
        label = _local_label(Path(ref), doc)
    if not isinstance(doc, dict):
        raise CompareError(f"{label} is not a valid edition (not a JSON object)")
    errors = validate_for_compare(doc)
    if errors:
        raise CompareError(f"{label} is not a valid edition: {errors[0]}")
    return label, doc


# ------------------------------------------------------------------ the table


def _side(block: Mapping[str, Any]) -> tuple[dict[str, int], int, list[str], list[str]]:
    """(first-named count per tool, successful answers, engines with an
    answer, "Engine (N asked)" for each engine with none) of one block: the
    top level or one mode's."""
    counts = {row["tool"]: row["first_count"] for row in block.get("leaderboard", [])}
    rows = block.get("engines", [])
    ok = sum(row["ok"] for row in rows)
    silent = [f"{_label(row['engine'])} ({row['answers']} asked)" for row in rows if row["ok"] == 0]
    return counts, ok, [row["engine"] for row in rows if row["ok"] > 0], silent


def _who(ok: int, engines: list[str], silent: list[str]) -> str:
    named = ", ".join(_label(e) for e in engines) or "no engine"
    missing = f"; no answers: {', '.join(silent)}" if silent else ""
    return f"{ok} answers from {named}{missing}."


def _label(engine: str) -> str:
    """The engine's label; an edition read from elsewhere may name an
    engine this version does not know, which is shown cleaned."""
    try:
        return engine_label(engine)
    except KeyError:
        return md_cell(engine)


def _share(part: int, whole: int) -> str:
    return "0.0%" if whole == 0 else f"{part / whole * 100:.1f}%"


def _tools(mine: Mapping[str, int], ours: Mapping[str, int], top: int) -> list[str]:
    ranked = [sorted(side, key=lambda tool: (-side[tool], tool))[:top] for side in (mine, ours)]
    return list(dict.fromkeys([*ranked[0], *ranked[1]]))


def _same_questions(mine: Mapping[str, Any], ours: Mapping[str, Any]) -> bool:
    def keyed(doc: Mapping[str, Any]) -> set[tuple[str, str]]:
        return {(p.get("key"), p.get("prompt_text")) for p in doc.get("prompts", []) if isinstance(p, Mapping)}

    return keyed(mine) == keyed(ours)


def _block_table(
    mine: Mapping[str, Any], ours: Mapping[str, Any], ours_label: str, *, counts_shown: bool, top: int
) -> list[str]:
    my_counts, my_ok, my_engines, my_silent = _side(mine)
    our_counts, our_ok, our_engines, our_silent = _side(ours)
    lines = [
        f"{_YOURS}: {_who(my_ok, my_engines, my_silent)}",
        f"{ours_label}: {_who(our_ok, our_engines, our_silent)}",
        "",
    ]
    tools = _tools(my_counts, our_counts, top)
    if not tools:
        return [*lines, "Neither side named a tool first."]
    if counts_shown:
        header = ["Tool", _YOURS, f"{_YOURS}, share", ours_label, f"{ours_label}, share"]
    else:
        header = ["Tool", f"{_YOURS}, share", f"{ours_label}, share"]
    lines.append("| " + " | ".join(md_cell(h) for h in header) + " |")
    lines.append("|---|" + "---:|" * (len(header) - 1))
    for tool in tools:
        mine_n, ours_n = my_counts.get(tool, 0), our_counts.get(tool, 0)
        if counts_shown:
            cells = [str(mine_n), _share(mine_n, my_ok), str(ours_n), _share(ours_n, our_ok)]
        else:
            cells = [_share(mine_n, my_ok), _share(ours_n, our_ok)]
        lines.append("| " + " | ".join([md_cell(tool), *cells]) + " |")
    return lines


# What an edition without a producer block was read with: the published
# editions before this package, which did not check names against the text.
LEGACY_EXTRACT_VERSION: Final[int] = 1
_EXTRACT_WORDS: Final[Mapping[int, str]] = {
    1: "which did not check",
    2: "which keeps a name only when the answer contains it",
}


def _extract_version(doc: Mapping[str, Any]) -> int:
    producer = doc.get("producer")
    version = producer.get("extract_version") if isinstance(producer, Mapping) else None
    return version if isinstance(version, int) and not isinstance(version, bool) else LEGACY_EXTRACT_VERSION


def _counting_lines(mine: Mapping[str, Any], ours: Mapping[str, Any], ours_label: str) -> list[str]:
    """What makes the two sides count differently: engines either side lists
    under review (counted in its shares), and different extraction versions."""
    lines: list[str] = []
    for label, doc in ((_YOURS, mine), (ours_label, ours)):
        review = [e for e in doc.get("engines_under_review") or [] if isinstance(e, str)]
        if review:
            names = ", ".join(_label(e) for e in review)
            their = "its" if len(review) == 1 else "their"
            lines.append(f"{label} lists {names} as under review; {their} answers are counted in that side's shares.")
    mine_v, ours_v = _extract_version(mine), _extract_version(ours)
    if mine_v != ours_v:

        def said(v: int) -> str:
            words = _EXTRACT_WORDS.get(v)
            return f"version {v}, {words}" if words else f"version {v}"

        lines.append(
            f"The two sides read the names differently: {_YOURS} with extraction {said(mine_v)}; "
            f"{ours_label} with {said(ours_v)}."
        )
    return lines


def _legacy_lines(mine: Mapping[str, Any], ours: Mapping[str, Any], ours_label: str, **kwargs: Any) -> list[str]:
    my_basis = plain(mine.get("basis", TOP_LEVEL_BASIS))
    lines = [f"One row per engine on both sides. {ours_label} counts: {TOP_LEVEL_BASIS}. Your run counts: {my_basis}."]
    if my_basis != TOP_LEVEL_BASIS:
        lines.append(
            "The two sides count different rows, so read the shares with that in mind; "
            "a run with --mode both has the same rows as this edition."
        )
    lines.append("")
    return lines + _block_table(mine, ours, ours_label, **kwargs)


def render_compare(mine: Mapping[str, Any], ours: Mapping[str, Any], *, ours_label: str, top: int = DEFAULT_TOP) -> str:
    """Your run's first-named table beside `ours`, per mode (or against a
    legacy edition's top level), as Markdown."""
    ours_label = md_cell(ours_label)
    same = _same_questions(mine, ours)
    lines = [f"# {_YOURS} beside {ours_label}", ""]
    if same:
        lines.append("Both sides asked the same questions. A share is first-named answers over that side's answers.")
    else:
        lines.append(
            f"These two tables answer different questions: {_YOURS.lower()} asked its own set, {ours_label} the "
            "index's. Only the shares are shown, each over that side's own answers."
        )
    lines.extend(plain(line) for line in _counting_lines(mine, ours, ours_label))
    kwargs = {"counts_shown": same, "top": top}
    if is_legacy(ours):
        lines.append("")
        lines.extend(_legacy_lines(mine, ours, ours_label, **kwargs))
        return strip_controls("\n".join(lines)) + "\n"
    for mode in DISPLAY_MODES:
        my_block = mine["modes"].get(mode) or {}
        if not my_block.get("engines"):
            continue
        lines += ["", f"## {MODE_HEADINGS[mode]} ({MODE_BASIS[mode]} on both sides)", ""]
        our_block = ours["modes"].get(mode) or {}
        if not our_block.get("engines"):
            row = "api" if mode == MODE_API else "search" if mode == MODE_SEARCH else mode
            lines.append(f"{ours_label} has no {row} row, so there is nothing to set beside yours.")
            continue
        lines.extend(_block_table(my_block, our_block, ours_label, **kwargs))
    return strip_controls("\n".join(lines)) + "\n"
