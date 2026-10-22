"""`aeo-index rebuild`: an edition from per-answer facts, with no call.

The facts are an `answers.jsonl` (a run's, or a published edition's). The
questions are a question file (TOML), a `prompts.csv`, or by default the
index's ten. A `prompts.csv` holding exactly the index's questions is read
as the index's set, so its brand (Sextant AEO and its domain) applies, as
it did when the edition was made; any other `prompts.csv` tracks no brand
unless `--brand` names one. What each mode was asked of is read from the
rows: every asked cell leaves a row, success or not. Each run date is one
run, judged on what that date asked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from aeo_index import __version__
from aeo_index.edition import asked_from_facts, build_edition, run_info
from aeo_index.outputs import read_facts, read_prompts_csv, write_run
from aeo_index.questions import (
    DEFAULT_EXCLUDE,
    DEFAULT_GENERIC_EXAMPLE,
    DEFAULT_PURPOSE,
    DEFAULT_SUBJECT,
    QuestionFileError,
    QuestionSet,
    default_questions,
    load_questions,
    make_brand,
)
from aeo_index.records import AnswerFacts, RunInfo
from aeo_index.table import render_table

# The set id a `prompts.csv` that is not the index's own is recorded under.
CSV_SET_ID: Final[str] = "from-prompts-csv"


class RebuildError(Exception):
    """The edition cannot be rebuilt; the message says why."""


@dataclass(frozen=True)
class Rebuilt:
    doc: dict[str, object]
    facts: tuple[AnswerFacts, ...]
    questions: QuestionSet
    paths: tuple[Path, ...]


def _csv_set(path: Path) -> QuestionSet:
    questions = tuple(read_prompts_csv(path))
    index = default_questions()
    if set(questions) == set(index.questions):
        return index
    return QuestionSet(
        id=CSV_SET_ID,
        subject=DEFAULT_SUBJECT,
        purpose=DEFAULT_PURPOSE,
        generic_example=DEFAULT_GENERIC_EXAMPLE,
        exclude=DEFAULT_EXCLUDE,
        brand=None,
        aliases={},
        questions=questions,
    )


def load_set(
    path: Path | None, *, brand: str | None, brand_aliases: Sequence[str], brand_domain: str | None
) -> QuestionSet:
    """The questions to rebuild with, tracking `brand` when given. A
    question that names a tool is loaded as experimental, as the run that
    asked it must have (`--allow-branded`)."""
    try:
        if path is None:
            qset = default_questions()
        elif path.suffix.lower() == ".csv":
            qset = _csv_set(path)
        else:
            qset = load_questions(path, allow_branded=True)
        if brand is None:
            if brand_aliases or brand_domain:
                raise RebuildError("--brand-alias and --brand-domain need --brand")
            return qset
        return qset.with_brand(make_brand(brand, brand_aliases, brand_domain), allow_branded=True)
    except (QuestionFileError, ValueError, OSError) as exc:
        raise RebuildError(str(exc)) from exc


def _runs(facts: Sequence[AnswerFacts]) -> list[RunInfo]:
    """One run per date, each judged on what that date asked."""
    return [
        run_info(facts, asked_from_facts(f for f in facts if f.run_date == day), run_date=day)
        for day in sorted({f.run_date for f in facts})
    ]


def _producer(answers: Path, qset: QuestionSet) -> dict[str, object]:
    brand = qset.brand
    return {
        "name": "aeo-index",
        "version": __version__,
        "rebuilt_from": answers.name,
        "brand": brand.name if brand is not None else None,
        "question_set": qset.id,
    }


def rebuild(
    answers: Path,
    qset: QuestionSet,
    out: Path,
    *,
    edition: str | None = None,
    engines_under_review: Sequence[str] = (),
) -> Rebuilt:
    """Build and write the edition of `answers` into `out`. Only the facts
    dated on or before `edition` (default: the newest date) are read."""
    try:
        facts = read_facts(answers)
    except (ValueError, OSError) as exc:
        raise RebuildError(str(exc)) from exc
    dates = sorted({f.run_date for f in facts})
    edition = edition or dates[-1]
    kept = tuple(f for f in facts if f.run_date <= edition)
    if not kept:
        raise RebuildError(f"no answer is dated on or before {edition}")
    runs = _runs(kept)
    brand = qset.brand
    # The questions the answers are to, as an edition lists only what was asked.
    answered = {f.key for f in kept}
    asked_questions = [q for q in qset.questions if q.key in answered]
    try:
        doc = build_edition(
            kept,
            asked_questions,
            runs,
            asked=asked_from_facts(kept),
            edition=edition,
            generated_at=datetime.now(UTC).isoformat(),
            brand_domain=brand.domain if brand is not None else None,
            engines_under_review=list(engines_under_review),
            producer=_producer(answers, qset),
        )
    except ValueError as exc:
        raise RebuildError(str(exc)) from exc
    table_md = render_table(doc, kept)
    try:
        paths = write_run(out, doc=doc, facts=kept, answers=None, questions=qset, table_md=table_md, include_text=False)
    except OSError as exc:
        raise RebuildError(f"cannot write to {out}: {exc}") from exc
    return Rebuilt(doc=doc, facts=kept, questions=qset, paths=tuple(paths))
