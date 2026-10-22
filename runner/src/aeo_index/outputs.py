"""The files a run writes, and reading them back.

`write_run` writes, into the run's folder:

  index.json         the edition (no answer text, ever);
  table.md           the first-named table, one table per mode;
  leaderboard.csv    mode, engine, tool, first_count;
  cited_domains.csv  mode, domain, citations, engines (`;`-joined), sextant
                     (whether it is the tracked brand's domain);
  prompts.csv        one row per (question, mode, engine asked in it):
                     key, prompt_text, experimental, mode, engine, answers,
                     ok, sextant_named, sextant_first;
  answers.jsonl      one line per answer: its facts (what the edition is
                     built from) plus model_version, latency_ms, cost_usd,
                     error_type, error, search_count, search_queries and,
                     unless left out, the answer text as `text`.

A CSV cell a spreadsheet would read as a formula (starting with `=`, `+`,
`-`, `@`, a tab or a carriage return) is written with a leading `'`.
`read_facts` and `read_prompts_csv` read `answers.jsonl` and a
`prompts.csv` (this format, or the published editions' older
`key,prompt,experimental`) back for `aeo-index rebuild`.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from aeo_index.edition import domain_rows
from aeo_index.questions import Question, QuestionSet
from aeo_index.records import Answer, AnswerFacts

INDEX_FILE: Final[str] = "index.json"
TABLE_FILE: Final[str] = "table.md"
LEADERBOARD_CSV: Final[str] = "leaderboard.csv"
DOMAINS_CSV: Final[str] = "cited_domains.csv"
PROMPTS_CSV: Final[str] = "prompts.csv"
ANSWERS_FILE: Final[str] = "answers.jsonl"
OUTPUT_FILES: Final[tuple[str, ...]] = (INDEX_FILE, TABLE_FILE, LEADERBOARD_CSV, DOMAINS_CSV, PROMPTS_CSV, ANSWERS_FILE)

LEADERBOARD_COLUMNS: Final[tuple[str, ...]] = ("mode", "engine", "tool", "first_count")
DOMAINS_COLUMNS: Final[tuple[str, ...]] = ("mode", "domain", "citations", "engines", "sextant")
PROMPTS_COLUMNS: Final[tuple[str, ...]] = (
    "key",
    "prompt_text",
    "experimental",
    "mode",
    "engine",
    "answers",
    "ok",
    "sextant_named",
    "sextant_first",
)
# A cell starting with one of these is a formula to a spreadsheet.
FORMULA_PREFIXES: Final[tuple[str, ...]] = ("=", "+", "-", "@", "\t", "\r")
# The largest answers.jsonl or prompts.csv read back.
MAX_READ_BYTES: Final[int] = 500_000_000
_TRUE: Final[frozenset[str]] = frozenset({"true", "1", "yes"})
_FALSE: Final[frozenset[str]] = frozenset({"false", "0", "no", ""})


def _csv_cell(value: object) -> object:
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row[key]) for key in columns})


def _answer_line(fact: AnswerFacts, answer: Answer | None, include_text: bool) -> str:
    row: dict[str, object] = fact.to_json()
    if answer is not None:
        if (answer.key, answer.engine, answer.mode, answer.sample) != (fact.key, fact.engine, fact.mode, fact.sample):
            raise ValueError(f"the answer to {answer.key} does not pair with the fact row for {fact.key}")
        row.update(
            model_version=answer.model_version,
            latency_ms=answer.latency_ms,
            cost_usd=str(answer.cost_usd),
            error_type=answer.error_type,
            error=answer.error,
            search_count=answer.search_count,
            search_queries=list(answer.search_queries) if answer.search_queries is not None else None,
        )
        if include_text:
            row["text"] = answer.text
    return json.dumps(row, ensure_ascii=False)


def _prompt_csv_rows(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    prompts = {p["key"]: p for p in doc["prompts"]}
    return [
        {**row, "prompt_text": prompts[row["key"]]["prompt_text"], "experimental": prompts[row["key"]]["experimental"]}
        for row in doc["by_prompt"]
    ]


# A file being written goes to `<name>.partial` first and takes its name
# only when it is whole.
PARTIAL_SUFFIX: Final[str] = ".partial"


def _write_whole(path: Path, write: Callable[[Path], object]) -> None:
    """Write `path` through a partial file renamed into place, so a file
    stopped part way (Ctrl-C, a full disk) is never left under its name."""
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    try:
        write(partial)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def write_run(
    out: Path,
    *,
    doc: dict[str, Any],
    facts: Sequence[AnswerFacts],
    answers: Sequence[Answer] | None,
    questions: QuestionSet,
    table_md: str,
    include_text: bool,
    on_write: Callable[[Path], None] | None = None,
) -> list[Path]:
    """Write the run's six files into `out` (created if missing) and return
    their paths in `OUTPUT_FILES` order. `answers` pair one to one with
    `facts`; None (a rebuild) writes the facts alone. Each file is whole or
    absent. `on_write` is called with each file's path once that file is
    written, so a caller that is interrupted part way knows which ones
    exist."""
    if answers is not None and len(answers) != len(facts):
        raise ValueError(f"{len(answers)} answers but {len(facts)} fact rows; they must pair one to one")
    lines = [
        _answer_line(fact, answers[i] if answers is not None else None, include_text) for i, fact in enumerate(facts)
    ]
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / name for name in OUTPUT_FILES]
    modes = doc["modes"]
    brand_domain = questions.brand.domain if questions.brand is not None else None
    writers: list[Callable[[Path], object]] = [
        lambda p: p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"),
        lambda p: p.write_text(table_md, encoding="utf-8"),
        lambda p: _write_csv(
            p,
            LEADERBOARD_COLUMNS,
            ({**row, "first_count": row["c"]} for mode in modes for row in modes[mode]["leaderboard_by_engine"]),
        ),
        lambda p: _write_csv(p, DOMAINS_COLUMNS, domain_rows(facts, brand_domain=brand_domain)),
        lambda p: _write_csv(p, PROMPTS_COLUMNS, _prompt_csv_rows(doc)),
        lambda p: p.write_text("".join(line + "\n" for line in lines), encoding="utf-8"),
    ]
    for path, write in zip(paths, writers, strict=True):
        _write_whole(path, write)
        if on_write is not None:
            on_write(path)
    return paths


# ------------------------------------------------------------------ reading back


def _check_size(path: Path) -> None:
    size = path.stat().st_size
    if size > MAX_READ_BYTES:
        raise ValueError(f"{path} is {size} bytes; the most read is {MAX_READ_BYTES}")


def read_facts(path: Path) -> list[AnswerFacts]:
    """The fact rows of an `answers.jsonl`, in order; blank lines are
    skipped. Raises ValueError naming the line that cannot be read, and
    OSError when the file cannot be opened."""
    _check_size(path)
    facts: list[AnswerFacts] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                facts.append(AnswerFacts.from_json(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}, line {number}: {exc}") from exc
    if not facts:
        raise ValueError(f"{path} holds no answer")
    return facts


def _unquote(value: str) -> str:
    """A cell as it was before the formula guard quoted it."""
    if value.startswith("'") and value[1:].startswith(FORMULA_PREFIXES):
        return value[1:]
    return value


def _flag(value: str, where: str) -> bool:
    folded = value.strip().lower()
    if folded in _TRUE:
        return True
    if folded in _FALSE:
        return False
    raise ValueError(f"{where}: experimental must be true or false, got {value!r}")


def read_prompts_csv(path: Path) -> list[Question]:
    """The questions of a `prompts.csv`, one per key in first-seen order:
    this package's columns (`prompt_text`) or the older published ones
    (`prompt`). Raises ValueError for a missing column, or a key given two
    different texts or flags."""
    _check_size(path)
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        text_column = "prompt_text" if "prompt_text" in fields else "prompt" if "prompt" in fields else None
        if "key" not in fields or text_column is None:
            raise ValueError(f"{path} needs the columns key and prompt_text (or prompt)")
        found: dict[str, Question] = {}
        for number, row in enumerate(reader, start=2):
            where = f"{path}, line {number}"
            key = _unquote(row.get("key") or "").strip()
            text = _unquote(row.get(text_column) or "").strip()
            if not key or not text:
                raise ValueError(f"{where}: a question needs a key and a text")
            question = Question(key, text, _flag(row.get("experimental") or "", where))
            if found.setdefault(key, question) != question:
                raise ValueError(f"{where}: the key {key!r} is given two different questions")
    if not found:
        raise ValueError(f"{path} holds no question")
    return list(found.values())
