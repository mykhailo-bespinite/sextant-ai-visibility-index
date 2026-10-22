from decimal import Decimal

from aeo_index.dispatch import Budget, Progress
from aeo_index.execute import EXIT_BUDGET, EXIT_OK, ProgressPrinter, render_summary
from aeo_index.judge import JudgeOutcome
from aeo_index.records import Judged, RunInfo
from run_doubles import answer


def _outcome(*judged: Judged) -> JudgeOutcome:
    return JudgeOutcome(judged=judged, failed=0, skipped_for_budget=0, rejected=False)


def test_actual_cost_is_the_budgets_spend_with_the_timeout_allowance():
    answers = [answer("q01", cost="0.10"), answer("q01", sample=2, status="timeout", cost="0")]
    outcome = _outcome(Judged(("Profound",), ("Ghost",), Decimal("0.002")), Judged(None, (), Decimal("0")))
    budget = Budget(Decimal("5"))
    budget.charge(Decimal("0.10") + Decimal("0.30") + Decimal("0.002"))  # the timed-out call charged its ceiling
    text = render_summary(
        answers,
        outcome,
        info=RunInfo("2026-10-21", "partial", 1, "api"),
        budget=budget,
        exit_code=3,
        asked={"api": ["anthropic"], "search": []},
    )
    assert "Actual cost $0.40 of the $5.00 budget (answers $0.10, judge $0.00)." in text
    assert "That includes $0.30 counted for 1 call that timed out or sent a reply that could not be read" in text
    assert "Run partial: 1 of 2 answers succeeded, 1 failed, 0 skipped." in text
    assert "judge_ungrounded 1" in text and "Claude, api: success 1, timeout 1" in text


def test_an_unreadable_reply_is_counted_in_the_allowance_line():
    from dataclasses import replace

    unreadable = replace(answer("q01", sample=2, status="error", cost="0"), error_type="bad_response")
    answers = [answer("q01", cost="0.10"), unreadable, answer("q01", sample=3, status="timeout", cost="0")]
    outcome = _outcome(
        Judged(("Profound",), (), Decimal("0")), Judged(None, (), Decimal("0")), Judged(None, (), Decimal("0"))
    )
    budget = Budget(Decimal("5"))
    budget.charge(Decimal("0.10") + Decimal("0.60"))
    text = render_summary(
        answers,
        outcome,
        info=RunInfo("2026-10-21", "partial", 1, "api"),
        budget=budget,
        exit_code=3,
        asked={"api": ["anthropic"], "search": []},
    )
    assert "That includes $0.60 counted for 2 calls that timed out or sent a reply that could not be read" in text


def test_a_budget_stop_is_said():
    text = render_summary(
        [answer(), answer(sample=2, status="skipped")],
        _outcome(Judged(("Profound",), (), Decimal("0")), Judged(None, (), Decimal("0"))),
        info=RunInfo("2026-10-21", "partial", 1, "api"),
        budget=Budget(Decimal("1")),
        exit_code=EXIT_BUDGET,
        asked={"api": ["anthropic"]},
    )
    assert "It stopped at the budget." in text and "1 skipped" in text
    assert "timed out" not in text


def test_no_timeout_line_without_an_allowance():
    budget = Budget(Decimal("1"))
    budget.charge(Decimal("0.01"))
    text = render_summary(
        [answer()],
        _outcome(Judged(("Profound",), (), Decimal("0"))),
        info=RunInfo("2026-10-21", "completed", 1, "api"),
        budget=budget,
        exit_code=EXIT_OK,
        asked={"api": ["anthropic"]},
    )
    assert "timed out" not in text and "Run completed" in text


class _Plan:
    def __init__(self, cells):
        self.cells = cells


class _Cell:
    def __init__(self, engine):
        self.engine = engine


def test_progress_lines_show_counts_spend_elapsed_and_time_left(capsys):
    printer = ProgressPrinter(_Plan([_Cell("anthropic")] * 30))
    printer.dispatch(Progress("anthropic", 0, 0, 0, Decimal("0"), 0.0))
    printer.dispatch(Progress("anthropic", 10, 9, 1, Decimal("0.5"), 40.0))
    printer.dispatch(Progress("anthropic", 30, 29, 1, Decimal("1.5"), 125.0))
    lines = capsys.readouterr().err.splitlines()
    assert (
        lines[0] == "Claude: asked 0 of 30, answered 0, failed 0, $0.00 so far, 0 s elapsed, estimating the time left"
    )
    assert lines[1].endswith("$0.50 so far, 40 s elapsed, about 1 min 20 s left")
    assert lines[2].endswith("2 min 5 s elapsed, done")


def test_judge_progress_is_throttled(capsys):
    now = [0.0]
    printer = ProgressPrinter(_Plan([]), clock=lambda: now[0])
    printer.judge_start(3, "anthropic:claude-haiku-4-5-20251001", lambda: Decimal("0.01"))
    printer.judge(1, 3)  # under 10 s: quiet
    now[0] = 12.0
    printer.judge(2, 3)
    printer.judge(3, 3)  # the last one is always said
    err = capsys.readouterr().err.splitlines()
    assert err[0] == "Judging 3 answers with anthropic:claude-haiku-4-5-20251001."
    assert len(err) == 3 and err[1].startswith("Judge: 2 of 3 answers") and err[2].endswith("done")


def test_a_completed_run_that_exits_3_says_why():
    text = render_summary(
        [answer(), answer(sample=2)],
        JudgeOutcome(
            judged=(Judged(("Profound",), (), Decimal("0")), Judged(None, (), Decimal("0"))),
            failed=1,
            skipped_for_budget=0,
            rejected=False,
        ),
        info=RunInfo("2026-10-21", "completed", 1, "api"),
        budget=Budget(Decimal("1")),
        exit_code=3,
        asked={"api": ["anthropic"]},
    )
    assert text.splitlines()[-1] == "Exit 3: the judge could not list the tools of 1 answer, so the table is partial."


def test_a_budget_stop_says_why_on_the_last_line():
    text = render_summary(
        [answer(), answer(sample=2, status="skipped")],
        _outcome(Judged(("Profound",), (), Decimal("0")), Judged(None, (), Decimal("0"))),
        info=RunInfo("2026-10-21", "partial", 1, "api"),
        budget=Budget(Decimal("1")),
        exit_code=EXIT_BUDGET,
        asked={"api": ["anthropic"]},
    )
    assert text.splitlines()[-1] == "Exit 4: the budget was reached; 1 cell was not asked."


def test_a_partial_run_says_why_on_the_last_line():
    text = render_summary(
        [answer(), answer(sample=2, status="error")],
        _outcome(Judged(("Profound",), (), Decimal("0")), Judged(None, (), Decimal("0"))),
        info=RunInfo("2026-10-21", "partial", 1, "api"),
        budget=Budget(Decimal("1")),
        exit_code=3,
        asked={"api": ["anthropic"]},
    )
    assert text.splitlines()[-1] == "Exit 3: the run is partial (some question, engine and mode has no answer)."


def test_the_interrupt_line_names_what_was_written():
    from decimal import Decimal
    from pathlib import Path

    from aeo_index.dispatch import Budget
    from aeo_index.execute import RunTally, interrupted_line

    tally = RunTally(Budget(Decimal("20")))
    assert interrupted_line(tally, Path("out")).startswith("Interrupted: nothing was written;")
    tally.files.append("index.json")
    assert interrupted_line(tally, Path("out")).startswith(
        "Interrupted: only index.json was written to out, so the run is incomplete;"
    )
    tally.files.append("table.md")
    assert "only index.json and table.md were written to out" in interrupted_line(tally, Path("out"))
    tally.written = True
    assert interrupted_line(tally, Path("out")).startswith("Interrupted: the files are in out;")
