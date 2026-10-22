"""`aeo-index run` after the plan is confirmed: ask, judge, build, write.

The order is fixed: every cell through `dispatch.ask_all` (the judge's
high figure held back in the budget's reserve, so asking cannot spend the
judge's share), then the judge over every successful answer, then the
unused reserve released, then the facts, the edition, the table and the
files, then the summary and, when asked, the side-by-side with a
published edition. Progress goes to stderr; the table and the summary to
stdout.

Exit codes: 0 when every asked (question, engine, mode) cell has a
successful answer and every one of them was judged; 3 when the run is
partial or failed, the judge could not list the tools of some answer
(an unscored answer makes the table partial), or the files could not be
written; 4 when the budget stopped the asking or the judging (a stop at
the budget is reported as such even when the run is also partial). The
summary's last line says why a run did not exit 0. When the files cannot
be written, the summary, with the actual cost, goes to stderr.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import httpx

from aeo_index import __version__
from aeo_index.compare import CompareError, render_compare, resolve_edition
from aeo_index.dispatch import Budget, Progress, ask_all
from aeo_index.edition import RUN_COMPLETED, RUN_FAILED, build_edition, facts_from_answers, run_info
from aeo_index.judge import EXTRACT_VERSION, JudgeOutcome, judge_answers
from aeo_index.modes import MODES, STATUS_SUCCESS, STATUSES, engine_label
from aeo_index.outputs import OUTPUT_FILES, write_run
from aeo_index.preflight import Plan, money
from aeo_index.providers.base import Provider, build_providers
from aeo_index.records import Answer, AnswerFacts, RunInfo
from aeo_index.table import render_table

EXIT_OK: Final[int] = 0
EXIT_REFUSED: Final[int] = 2
EXIT_PARTIAL: Final[int] = 3
EXIT_BUDGET: Final[int] = 4
# The longest the judge's progress goes unreported.
JUDGE_PROGRESS_INTERVAL_S: Final[float] = 10.0
PACKAGE_NAME: Final[str] = "aeo-index"


def _duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    if seconds < 60:
        return f"{seconds} s"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes} min {rest} s" if rest else f"{minutes} min"


def _time_left(elapsed: float, done: int, total: int) -> str:
    if done >= total:
        return "done"
    if done == 0 or elapsed <= 0:
        return "estimating the time left"
    return f"about {_duration(elapsed / done * (total - done))} left"


class ProgressPrinter:
    """Prints progress lines to stderr: one per engine as `ask_all` reports
    it (every 10 s), and the judge's every 10 s by `clock`."""

    def __init__(self, plan: Plan, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.totals: Counter[str] = Counter(cell.engine for cell in plan.cells)
        self.clock = clock
        self.judge_started: float | None = None
        self.judge_last = 0.0
        self.judge_spent: Callable[[], Decimal] = lambda: Decimal("0")

    def dispatch(self, progress: Progress) -> None:
        total = self.totals[progress.engine]
        print(
            f"{engine_label(progress.engine)}: asked {progress.asked} of {total}, answered {progress.answered}, "
            f"failed {progress.failed}, {money(progress.spent)} so far, {_duration(progress.elapsed_s)} elapsed, "
            f"{_time_left(progress.elapsed_s, progress.asked, total)}",
            file=sys.stderr,
        )

    def judge_start(self, count: int, judge: str, spent: Callable[[], Decimal]) -> None:
        self.judge_started = self.judge_last = self.clock()
        self.judge_spent = spent
        print(f"Judging {count} answers with {judge}.", file=sys.stderr)

    def judge(self, done: int, total: int) -> None:
        now = self.clock()
        if done < total and now - self.judge_last < JUDGE_PROGRESS_INTERVAL_S:
            return
        self.judge_last = now
        elapsed = now - (self.judge_started if self.judge_started is not None else now)
        print(
            f"Judge: {done} of {total} answers, {money(self.judge_spent())} spent in all, "
            f"{_duration(elapsed)} elapsed, {_time_left(elapsed, done, total)}",
            file=sys.stderr,
        )


def _asked(plan: Plan) -> dict[str, list[str]]:
    seen = {(cell.mode, cell.engine) for cell in plan.cells}
    return {mode: [engine for engine in plan.providers if (mode, engine) in seen] for mode in MODES}


def _status_line(engine: str, mode: str, answers: Sequence[Answer]) -> str:
    counts = Counter(a.status for a in answers if a.engine == engine and a.mode == mode)
    parts = ", ".join(f"{status} {counts[status]}" for status in STATUSES if counts[status])
    return f"  {engine_label(engine)}, {mode}: {parts}"


def render_summary(
    answers: Sequence[Answer],
    outcome: JudgeOutcome,
    *,
    info: RunInfo,
    budget: Budget,
    exit_code: int,
    asked: Mapping[str, Sequence[str]],
    write_failed: bool = False,
) -> str:
    """The summary printed after a run: its status, the actual cost (the
    budget's spend, which includes the ceiling charged for each timed-out
    call), `judge_ungrounded`, the status counts per engine and mode, and,
    for an exit other than 0, a last line saying why."""
    ok = sum(1 for a in answers if a.status == STATUS_SUCCESS)
    skipped = sum(1 for a in answers if a.status == "skipped")
    failed = len(answers) - ok - skipped
    answers_cost = sum((a.cost_usd for a in answers), Decimal("0"))
    judge_cost = sum((j.cost_usd for j in outcome.judged), Decimal("0"))
    held = budget.spent - answers_cost - judge_cost
    stopped = " It stopped at the budget." if exit_code == EXIT_BUDGET else ""
    lines = [
        f"Run {info.status}: {ok} of {len(answers)} answers succeeded, {failed} failed, {skipped} skipped.{stopped}",
        f"Actual cost {money(budget.spent)} of the {money(budget.cap)} budget "
        f"(answers {money(answers_cost)}, judge {money(judge_cost)}).",
    ]
    # Calls charged their ceiling with $0 on the row: timeouts, and replies that could not be read.
    unknown = sum(1 for a in answers if a.status == "timeout" or (a.error_type == "bad_response" and a.cost_usd == 0))
    if held > 0:
        lines.append(
            f"That includes {money(held, up=True)} counted for {unknown} call{'s' if unknown != 1 else ''} "
            "that timed out or sent a reply that could not be read, which the vendor may have billed; "
            "their rows record $0."
        )
    judged = sum(1 for j in outcome.judged if j.tools is not None)
    ungrounded = sum(len(j.ungrounded) for j in outcome.judged)
    lines.append(
        f"Judge: {judged} of {ok} answers judged; judge_ungrounded {ungrounded} "
        "(names the judge listed that the answer does not contain, dropped)."
    )
    lines.append("Answers per engine and mode:")
    lines.extend(_status_line(engine, mode, answers) for mode in MODES for engine in asked.get(mode, ()))
    reasons = _exit_reasons(exit_code, info, outcome, ok=ok, judged=judged, skipped=skipped, write_failed=write_failed)
    if reasons:
        lines.append(f"Exit {exit_code}: {'; '.join(reasons)}.")
    return "\n".join(lines)


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _exit_reasons(
    code: int, info: RunInfo, outcome: JudgeOutcome, *, ok: int, judged: int, skipped: int, write_failed: bool
) -> list[str]:
    """Why the run exits `code`, as clauses; none for exit 0."""
    if code == EXIT_OK:
        return []
    reasons: list[str] = []
    if code == EXIT_BUDGET:
        if skipped:
            reasons.append(f"the budget was reached; {_plural(skipped, 'cell was', 'cells were')} not asked")
        if outcome.skipped_for_budget:
            n = outcome.skipped_for_budget
            reasons.append(f"the budget was reached; {_plural(n, 'answer was', 'answers were')} not judged")
    if write_failed:
        reasons.append("the files could not be written")
    if info.status == RUN_FAILED:
        reasons.append("the run failed (no answer succeeded)")
    elif info.status != RUN_COMPLETED and code != EXIT_BUDGET:
        reasons.append("the run is partial (some question, engine and mode has no answer)")
    unscored = ok - judged - outcome.skipped_for_budget
    if unscored > 0:
        reasons.append(
            f"the judge could not list the tools of {_plural(unscored, 'answer', 'answers')}, so the table is partial"
        )
    return reasons


def _exit_code(info: RunInfo, budget: Budget, outcome: JudgeOutcome, *, write_failed: bool = False) -> int:
    """4 when the budget stopped the asking or the judging; else 3 when the
    run is partial or failed, an answer went unscored, or the files could
    not be written; else 0."""
    if budget.skipped or outcome.skipped_for_budget:
        return EXIT_BUDGET
    if info.status != RUN_COMPLETED or outcome.failed or outcome.rejected or write_failed:
        return EXIT_PARTIAL
    return EXIT_OK


def _producer(plan: Plan, ungrounded: int) -> dict[str, object]:
    brand = plan.questions.brand
    return {
        "name": PACKAGE_NAME,
        "version": __version__,
        "judge": str(plan.judge),
        "models": dict(plan.models),
        "brand": brand.name if brand is not None else None,
        "question_set": plan.questions.id,
        "judge_ungrounded": ungrounded,
        "extract_version": EXTRACT_VERSION,
    }


async def _compare(
    ref: str, doc: Mapping[str, object], client: httpx.AsyncClient, redact: Callable[[str], str]
) -> None:
    try:
        label, ours = await resolve_edition(ref, client=client)
    except CompareError as exc:
        print(redact(f"Could not compare: {exc}"), file=sys.stderr)
        return
    print()
    print(render_compare(doc, ours, ours_label=label), end="")


@dataclass
class RunTally:
    """What a run has spent and written so far, for a caller that must
    report it if the run is interrupted."""

    budget: Budget
    written: bool = False
    # The run's files written so far, by name, in the order written.
    files: list[str] = field(default_factory=list)


def new_tally(plan: Plan) -> RunTally:
    """The run's money: the plan's budget, with the judge's high figure held
    back so asking cannot spend the judge's share."""
    return RunTally(Budget(plan.budget, reserved=plan.estimate.judge_high))


def _names(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def interrupted_line(tally: RunTally, out_dir: Path) -> str:
    """What was written and spent when the run was stopped: every file, the
    files written before the stop, or nothing."""
    budget = tally.budget
    if tally.written:
        where = f"the files are in {out_dir}"
    elif tally.files:
        verb = "was" if len(tally.files) == 1 else "were"
        where = f"only {_names(tally.files)} {verb} written to {out_dir}, so the run is incomplete"
    else:
        where = "nothing was written"
    calls = _plural(budget.charges, "paid call", "paid calls")
    return f"Interrupted: {where}; {money(budget.spent, up=True)} was spent on {calls}."


async def _ask_and_judge(
    plan: Plan, budget: Budget, client: httpx.AsyncClient, printer: ProgressPrinter, judge: Provider
) -> tuple[list[Answer], JudgeOutcome]:
    answers = await ask_all(
        plan.cells,
        plan.providers,
        budget=budget,
        card=plan.card,
        redactor=plan.redactor,
        client=client,
        on_progress=printer.dispatch,
    )
    printer.judge_start(sum(1 for a in answers if a.status == STATUS_SUCCESS), str(plan.judge), lambda: budget.spent)
    outcome = await judge_answers(
        answers,
        spec=plan.judge,
        provider=judge,
        questions=plan.questions,
        table=plan.questions.alias_table(),
        budget=budget,
        card=plan.card,
        client=client,
        redactor=plan.redactor,
        on_progress=printer.judge,
    )
    budget.release(budget.reserved)
    return answers, outcome


def _edition(
    plan: Plan, answers: Sequence[Answer], outcome: JudgeOutcome, run_date: str
) -> tuple[list[AnswerFacts], RunInfo, dict[str, Any]]:
    brand = plan.questions.brand
    table = plan.questions.alias_table()
    facts = facts_from_answers(answers, outcome.judged, run_date=run_date, brand=brand, table=table)
    asked = _asked(plan)
    info = run_info(facts, asked, run_date=run_date)
    doc = build_edition(
        facts,
        plan.questions.questions,
        [info],
        asked=asked,
        edition=run_date,
        generated_at=datetime.now(UTC).isoformat(),
        brand_domain=brand.domain if brand is not None else None,
        producer=_producer(plan, sum(len(j.ungrounded) for j in outcome.judged)),
    )
    return facts, info, doc


async def execute_run(
    plan: Plan,
    *,
    include_text: bool,
    compare_ref: str | None,
    client: httpx.AsyncClient,
    tally: RunTally | None = None,
    progress: ProgressPrinter | None = None,
) -> int:
    """Ask, judge, build and write a confirmed plan; returns the exit code.
    `tally` (default: `new_tally(plan)`) records the spend as it happens."""
    judges, missing = build_providers(plan.env, [plan.judge.provider], {plan.judge.provider: plan.judge.model})
    if missing:
        print(f"the judge {plan.judge} has no key: {missing[plan.judge.provider]}", file=sys.stderr)
        return EXIT_REFUSED
    tally = tally or new_tally(plan)
    budget = tally.budget
    run_date = datetime.now(UTC).date().isoformat()
    printer = progress or ProgressPrinter(plan)
    answers, outcome = await _ask_and_judge(plan, budget, client, printer, judges[plan.judge.provider])
    facts, info, doc = _edition(plan, answers, outcome, run_date)
    table_md = render_table(doc, facts)
    try:
        write_run(
            plan.out_dir,
            doc=doc,
            facts=facts,
            answers=answers,
            questions=plan.questions,
            table_md=table_md,
            include_text=include_text,
            on_write=lambda path: tally.files.append(path.name),
        )
        tally.written = True
    except OSError as exc:
        print(plan.redactor(f"Could not write the run to {plan.out_dir}: {exc}"), file=sys.stderr)
    failed_write = not tally.written
    code = _exit_code(info, budget, outcome, write_failed=failed_write)
    summary = render_summary(
        answers, outcome, info=info, budget=budget, exit_code=code, asked=_asked(plan), write_failed=failed_write
    )
    print(table_md)
    if failed_write:
        print(plan.redactor(summary), file=sys.stderr)
        return code
    print(summary)
    print(f"Wrote {', '.join(OUTPUT_FILES)} to {plan.out_dir}.")
    if compare_ref is not None:
        await _compare(compare_ref, doc, client, plan.redactor)
    return code
