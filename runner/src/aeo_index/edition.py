"""The edition: the `index.json` the index's research page reads, built from
per-answer facts.

The shape is the published edition format. A top level reads one row per
engine (its api row when that engine was asked in api, else its search
row; `basis` says which), a `modes` block holds the same figures for each
mode on its own, `by_prompt` and `leaderboard_by_prompt` break them down
per question, `agreement` says how often the samples of one question agree,
and `runs` and `prompts` describe what was asked. A search
answer and an api answer are never added together. No answer text is
ever written: only counts, ranks, run dates, tool names, question texts
and cited URLs.

`build_edition` is pure, so the same facts always give the same edition:
a run's `answers.jsonl` rebuilds its `index.json` (`aeo-index rebuild`),
except the fields only a publisher writes (`strip_publisher_fields`).
`validate_edition` holds an edition to the checks the research page makes
before it shows one.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from aeo_index.agreement import agreement_block
from aeo_index.aliases import AliasTable, normalise
from aeo_index.citations import citation_domains, citation_urls, domain_matches
from aeo_index.modes import (
    DISPLAY_MODES,
    ENGINES,
    MODE_API,
    MODE_SEARCH,
    MODES,
    SEARCH_ENGINES,
    STATUS_SUCCESS,
    TOP_DOMAINS,
    TOP_URLS,
)
from aeo_index.questions import Brand, Question
from aeo_index.records import Answer, AnswerFacts, Judged, RunInfo
from aeo_index.validate import is_legacy as is_legacy
from aeo_index.validate import validate_edition as validate_edition

# Fields only the publisher of an edition writes; two editions built from the
# same facts are equal once these are stripped.
PUBLISHER_ONLY: Final[tuple[str, ...]] = ("generated_at", "sacs", "engines_under_review_note", "producer")
PUBLISHER_ONLY_RUN_FIELDS: Final[tuple[str, ...]] = ("goal_met", "streak_days")

# What the top level counts, stated in the file.
TOP_LEVEL_BASIS: Final[str] = "api row per engine; search row for retrieval engines"
BASIS_SEARCH_ONLY: Final[str] = "search row per engine"
BASIS_API_WHERE_ASKED: Final[str] = "api row where asked; search row otherwise"
MODE_BASIS: Final[Mapping[str, str]] = {MODE_API: "api row only", MODE_SEARCH: "search row only"}

RUN_COMPLETED: Final[str] = "completed"
RUN_PARTIAL: Final[str] = "partial"
RUN_FAILED: Final[str] = "failed"

BY_PROMPT_COLUMNS: Final[tuple[str, ...]] = (
    "key",
    "mode",
    "engine",
    "answers",
    "ok",
    "sextant_named",
    "sextant_first",
    "sextant_ranks",
    "first_run",
    "last_run",
    "days",
)
_ROSTER: Final[Mapping[str, int]] = {engine: index for index, engine in enumerate(ENGINES)}


# ------------------------------------------------------------------ facts and runs


def facts_from_answers(
    answers: Sequence[Answer],
    judged: Sequence[Judged],
    *,
    run_date: str,
    brand: Brand | None,
    table: AliasTable,
) -> list[AnswerFacts]:
    """One fact row per answer, in order. `sextant_rank` is the 1-based
    position of `brand` in the tools the judge listed (None without a
    brand, or when the brand is not listed); a mention the judge did not
    list is not a rank."""
    if len(answers) != len(judged):
        raise ValueError(f"{len(answers)} answers but {len(judged)} judged results; they must pair one to one")
    canonical = table.normalise(brand.name) if brand is not None else None
    facts: list[AnswerFacts] = []
    for answer, verdict in zip(answers, judged, strict=True):
        tools = verdict.tools if answer.status == STATUS_SUCCESS else None
        rank = None
        if tools and canonical is not None and canonical in tools:
            rank = tools.index(canonical) + 1
        facts.append(
            AnswerFacts(
                key=answer.key,
                engine=answer.engine,
                mode=answer.mode,
                sample=answer.sample,
                run_date=run_date,
                status=answer.status,
                tools_named=tools,
                first_named=tools[0] if tools else None,
                sextant_rank=rank,
                citations=answer.citations,
            )
        )
    return facts


def asked_from_facts(facts: Iterable[AnswerFacts]) -> dict[str, list[str]]:
    """The engines each mode was asked of, read from the rows (every asked
    cell leaves a row, success or not), in roster order."""
    seen = {(f.mode, f.engine) for f in facts}
    return {mode: [engine for engine in ENGINES if (mode, engine) in seen] for mode in MODES}


def goal_mode(asked: Mapping[str, Sequence[str]]) -> str:
    """The row a run's goal is read on: `search` when one of the model
    engines (the ones with a web search tool) was asked in search mode,
    else `api`. The retrieval engines always search, so they never make it
    the search row on their own; a run that asked the model engines from
    training knowledge only is an api run."""
    searched = asked.get(MODE_SEARCH, ())
    return MODE_SEARCH if any(engine in SEARCH_ENGINES for engine in searched) else MODE_API


def run_info(facts: Sequence[AnswerFacts], asked: Mapping[str, Sequence[str]], *, run_date: str) -> RunInfo:
    """The run dated `run_date`: `completed` when every asked (question,
    engine, mode) cell has a successful answer, `failed` when none does,
    `partial` otherwise. Its goal row is `goal_mode(asked)`."""
    mine = [f for f in facts if f.run_date == run_date]
    keys = sorted({f.key for f in mine})
    succeeded = {(f.key, f.engine, f.mode) for f in mine if f.status == STATUS_SUCCESS}
    cells = [(key, engine, mode) for key in keys for mode in MODES for engine in asked.get(mode, ())]
    if not succeeded:
        status = RUN_FAILED
    elif cells and all(cell in succeeded for cell in cells):
        status = RUN_COMPLETED
    else:
        status = RUN_PARTIAL
    answered = {engine for (_, engine, _) in succeeded}
    return RunInfo(
        run_date=run_date,
        status=status,
        engines_answered=len(answered),
        goal_mode=goal_mode(asked),
    )


def _in_roster_order(engines: Iterable[str]) -> list[str]:
    return sorted(set(engines), key=lambda engine: (_ROSTER.get(engine, len(_ROSTER)), engine))


def _source_rows(facts: Iterable[AnswerFacts], brand_domain: str | None) -> list[dict[str, Any]]:
    """Per cited domain: the answers citing it (each answer once), the
    engines citing it in roster order, and whether it is the brand's own
    domain or a subdomain of it. Most cited first, ties by domain."""
    counts: Counter[str] = Counter()
    engines: dict[str, set[str]] = defaultdict(set)
    for fact in facts:
        for domain in citation_domains(fact.citations):
            counts[domain] += 1
            engines[domain].add(fact.engine)
    rows = [
        {
            "domain": domain,
            "citations": n,
            "engines": _in_roster_order(engines[domain]),
            "sextant": brand_domain is not None and domain_matches(domain, brand_domain),
        }
        for domain, n in counts.items()
    ]
    rows.sort(key=lambda row: (-row["citations"], row["domain"]))
    return rows


def domain_rows(facts: Sequence[AnswerFacts], *, brand_domain: str | None) -> list[dict[str, Any]]:
    """Every cited domain per mode, uncapped (`cited_domains.csv`): mode,
    domain, citations, the engines joined with `;`, and the brand flag."""
    return [
        {
            "mode": mode,
            "domain": row["domain"],
            "citations": row["citations"],
            "engines": ";".join(row["engines"]),
            "sextant": row["sextant"],
        }
        for mode in DISPLAY_MODES
        for row in _source_rows((f for f in facts if f.mode == mode), brand_domain)
    ]


# ------------------------------------------------------------------ the builder


class _View:
    """Which rows a block reads: one mode, or (mode None) the top level's
    one row per engine."""

    def __init__(self, mode: str | None, top_modes: Mapping[str, str]) -> None:
        self.mode = mode
        self.top_modes = top_modes

    def row_mode(self, engine: str) -> str:
        return self.mode if self.mode is not None else self.top_modes[engine]

    def holds(self, fact: AnswerFacts) -> bool:
        if self.mode is not None:
            return fact.mode == self.mode
        return self.top_modes.get(fact.engine) == fact.mode


def _engine_row(rows: Sequence[AnswerFacts], engine: str, mode: str) -> dict[str, Any]:
    ok = [f for f in rows if f.status == STATUS_SUCCESS]
    return {
        "engine": engine,
        "mode": mode,
        "answers": len(rows),
        "ok": len(ok),
        "sextant_named": sum(1 for f in ok if f.sextant_rank is not None),
        "sextant_first": sum(1 for f in ok if f.sextant_rank == 1),
    }


def _count_firsts(facts: Iterable[AnswerFacts], fixed: frozenset[str], mode: str) -> Counter[tuple[str, str]]:
    """The leader board's counting rule: one count per successful `mode`
    answer to a non-experimental question that named a tool first, keyed by
    (engine, tool)."""
    return Counter(
        (f.engine, normalise(f.first_named))
        for f in facts
        if f.mode == mode and f.status == STATUS_SUCCESS and f.first_named and f.key in fixed
    )


def _by_roster(counts: Counter[tuple[str, str]]) -> list[tuple[tuple[str, str], int]]:
    return sorted(counts.items(), key=lambda t: (_ROSTER.get(t[0][0], len(_ROSTER)), -t[1], t[0][1]))


def _figures(
    facts: Sequence[AnswerFacts],
    engines: Sequence[str],
    view: _View,
    fixed: frozenset[str],
    brand_domain: str | None,
    basis: str,
) -> dict[str, Any]:
    mine = [f for f in facts if view.holds(f)]
    by_engine: Counter[tuple[str, str]] = Counter()
    for row_mode in MODES:
        by_engine.update(_count_firsts(mine, fixed, row_mode))
    totals: Counter[str] = Counter()
    for (_, tool), n in by_engine.items():
        totals[tool] += n
    tag = {"mode": view.mode} if view.mode is not None else {}
    leaderboard = [
        {"tool": tool, "first_count": n, **tag} for tool, n in sorted(totals.items(), key=lambda t: (-t[1], t[0]))
    ]
    per_engine = [
        {"engine": engine, "mode": view.row_mode(engine), "tool": tool, "c": n}
        for (engine, tool), n in _by_roster(by_engine)
    ]
    url_counts: Counter[str] = Counter()
    url_engines: dict[str, set[str]] = defaultdict(set)
    for fact in mine:
        for url in citation_urls(fact.citations):
            url_counts[url] += 1
            url_engines[url].add(fact.engine)
    top_domains = [
        {"n": row["citations"], "domain": row["domain"], "sextant": row["sextant"], **tag}
        for row in _source_rows(mine, brand_domain)
    ]
    top_urls = [
        {"n": n, "url": url, "engines": ",".join(_in_roster_order(url_engines[url])), **tag}
        for url, n in sorted(url_counts.items(), key=lambda t: (-t[1], t[0]))
    ]
    return {
        "basis": basis,
        "engines": [
            _engine_row([f for f in mine if f.engine == engine], engine, view.row_mode(engine)) for engine in engines
        ],
        "leaderboard": leaderboard,
        "leaderboard_by_engine": per_engine,
        "top_domains": top_domains[:TOP_DOMAINS],
        "top_urls": top_urls[:TOP_URLS],
    }


def _top_modes(asked: Mapping[str, Sequence[str]]) -> tuple[dict[str, str], str]:
    """Each engine's top-level row (api where asked, else search) and the
    basis that states it."""
    in_api = set(asked.get(MODE_API, ()))
    asked_any = {engine for mode in MODES for engine in asked.get(mode, ())}
    top = {engine: MODE_API if engine in in_api else MODE_SEARCH for engine in ENGINES if engine in asked_any}
    searchers = [engine for engine in top if engine in SEARCH_ENGINES]
    if not in_api:
        basis = BASIS_SEARCH_ONLY
    elif all(engine in in_api for engine in searchers):
        basis = TOP_LEVEL_BASIS
    else:
        basis = BASIS_API_WHERE_ASKED
    return top, basis


def _cell_detail(cell: Sequence[AnswerFacts]) -> dict[str, Any]:
    dates = sorted({f.run_date for f in cell})
    ranks = sorted(f.sextant_rank for f in cell if f.status == STATUS_SUCCESS and f.sextant_rank is not None)
    return {
        "sextant_ranks": ranks,
        "first_run": dates[0] if dates else None,
        "last_run": dates[-1] if dates else None,
        "days": len(dates),
    }


def prompt_rows(
    facts: Sequence[AnswerFacts], questions: Sequence[Question], asked: Mapping[str, Sequence[str]]
) -> list[dict[str, Any]]:
    """One row per (question, mode, engine asked in that mode), by question
    key: the `by_prompt` fields plus the question's text and flag, which
    `prompts.csv` reads."""
    rows: list[dict[str, Any]] = []
    for question in sorted(questions, key=lambda q: q.key):
        mine = [f for f in facts if f.key == question.key]
        for mode in DISPLAY_MODES:
            for engine in asked.get(mode, ()):
                cell = [f for f in mine if f.engine == engine and f.mode == mode]
                rows.append(
                    {
                        "key": question.key,
                        "prompt_text": question.text,
                        "experimental": question.experimental,
                        **_engine_row(cell, engine, mode),
                        **_cell_detail(cell),
                    }
                )
    return rows


def _prompt_leaderboards(facts: Sequence[AnswerFacts], questions: Sequence[Question]) -> list[dict[str, Any]]:
    fixed = frozenset(q.key for q in questions if not q.experimental)
    rows: list[dict[str, Any]] = []
    for question in sorted(questions, key=lambda q: q.key):
        mine = [f for f in facts if f.key == question.key]
        for mode in DISPLAY_MODES:
            rows.extend(
                {"key": question.key, "mode": mode, "engine": engine, "tool": tool, "c": n}
                for (engine, tool), n in _by_roster(_count_firsts(mine, fixed, mode))
            )
    return rows


def _check_inputs(
    facts: Sequence[AnswerFacts], questions: Sequence[Question], asked: Mapping[str, Sequence[str]]
) -> None:
    keys = [q.key for q in questions]
    duplicated = sorted({key for key in keys if keys.count(key) > 1})
    if duplicated:
        raise ValueError(f"the question key {duplicated[0]!r} is listed twice")
    unknown_modes = sorted(set(asked) - set(MODES))
    if unknown_modes:
        raise ValueError(f"unknown mode {unknown_modes[0]!r} in what was asked")
    known = set(keys)
    for fact in facts:
        if fact.key not in known:
            raise ValueError(f"an answer is to question {fact.key!r}, which the question set does not list")
        if fact.engine not in asked.get(fact.mode, ()):
            raise ValueError(f"an answer is from {fact.engine} in {fact.mode} mode, which was not asked")


def _run_row(run: RunInfo) -> dict[str, Any]:
    return {
        "run_date": run.run_date,
        "status": run.status,
        "engines_answered": run.engines_answered,
        "goal_mode": run.goal_mode,
        **run.extra,
    }


def build_edition(
    facts: Sequence[AnswerFacts],
    questions: Sequence[Question],
    runs: Sequence[RunInfo],
    *,
    asked: Mapping[str, Sequence[str]],
    edition: str,
    generated_at: str,
    brand_domain: str | None,
    engines_under_review: Sequence[str] = (),
    producer: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """The edition of `facts`. `asked` names, per mode, the engines asked in
    it; `runs` are oldest first. Raises ValueError for a fact whose
    question is not in `questions` or whose engine and mode were not
    asked, and for an edition with no run."""
    _check_inputs(facts, questions, asked)
    if not runs:
        raise ValueError("an edition needs at least one run")
    ordered_asked = {mode: [engine for engine in ENGINES if engine in asked.get(mode, ())] for mode in MODES}
    top_modes, basis = _top_modes(ordered_asked)
    fixed = frozenset(q.key for q in questions if not q.experimental)
    rows = prompt_rows(facts, questions, ordered_asked)
    doc: dict[str, Any] = {
        "generated_at": generated_at,
        "edition": edition,
        "goal_mode": runs[-1].goal_mode,
        "engines_under_review": list(engines_under_review),
        "runs": [_run_row(run) for run in runs],
        "prompts": [
            {"key": q.key, "prompt_text": q.text, "experimental": q.experimental}
            for q in sorted(questions, key=lambda q: q.key)
        ],
        **_figures(facts, list(top_modes), _View(None, top_modes), fixed, brand_domain, basis),
        "modes": {
            mode: _figures(facts, ordered_asked[mode], _View(mode, top_modes), fixed, brand_domain, MODE_BASIS[mode])
            for mode in DISPLAY_MODES
        },
        "by_prompt": [{column: row[column] for column in BY_PROMPT_COLUMNS} for row in rows],
        "leaderboard_by_prompt": _prompt_leaderboards(facts, questions),
        "agreement": agreement_block(facts),
    }
    if producer is not None:
        doc["producer"] = dict(producer)
    return doc


def strip_publisher_fields(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of `doc` without the fields only a publisher writes."""
    stripped = {key: value for key, value in doc.items() if key not in PUBLISHER_ONLY}
    if isinstance(stripped.get("runs"), list):
        # A run that answered nothing (a lost night the publisher still lists,
        # such as 2026-10-19) leaves no answer behind, so a rebuild from the
        # answers cannot know it existed: it is publisher metadata too.
        stripped["runs"] = [
            {k: v for k, v in run.items() if k not in PUBLISHER_ONLY_RUN_FIELDS}
            for run in stripped["runs"]
            if not isinstance(run, Mapping) or int(run.get("engines_answered") or 0) > 0
        ]
    return stripped
