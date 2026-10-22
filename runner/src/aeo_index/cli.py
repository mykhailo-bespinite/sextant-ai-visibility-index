"""The `aeo-index` command line.

Exit codes: 0 done, 2 refused before asking (bad flags, a bad question
file, no engine configured, the judge's key missing, a budget below the
high estimate, an output folder in use). A run adds 3 (partial or failed,
answers the judge could not list, or files that could not be written),
4 (stopped at the budget) and 130 (interrupted with Ctrl-C: the files
written so far, if any, and the spend so far are printed; at the
confirmation prompt, nothing was asked or spent).

`run` asks and writes a run; `rebuild` builds an edition from a run's
`answers.jsonl` with no call; `compare` sets a run beside a published
edition; `questions` and `aliases` print what the method uses.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import httpx

from aeo_index import __version__
from aeo_index.aliases import AliasTable
from aeo_index.compare import CompareError, render_compare, resolve_edition
from aeo_index.execute import RunTally, execute_run, interrupted_line, new_tally
from aeo_index.modes import DEFAULT_SAMPLES, ENGINES, MAX_SAMPLES, MIN_SAMPLES, MODE_SEARCH, REQUESTABLE_MODES
from aeo_index.outputs import INDEX_FILE
from aeo_index.preflight import (
    DEFAULT_BUDGET_USD,
    DEFAULT_JUDGE,
    Plan,
    Refusal,
    budget_needed,
    check_budget,
    check_folder,
    check_out_dir,
    money,
    prepare,
    render_plan,
)
from aeo_index.questions import QuestionFileError, QuestionSet, default_questions, load_questions
from aeo_index.rebuild import RebuildError, load_set, rebuild
from aeo_index.sanitize import plain

EXIT_OK: Final[int] = 0
EXIT_REFUSED: Final[int] = 2
# A run stopped by Ctrl-C (128 + SIGINT), as shells report it.
EXIT_INTERRUPTED: Final[int] = 130

_FLAG_WIDTH: Final[int] = len("experimental")
_ISO_DATE: Final[re.Pattern[str]] = re.compile(r"\d{4}-\d{2}-\d{2}")
USER_AGENT: Final[str] = f"aeo-index/{__version__}"
# Connections kept open at once: the dispatch limit plus the judge's.
_MAX_CONNECTIONS: Final[int] = 16

# A subcommand's handler: takes the parsed arguments, returns the exit code.
Handler = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    """The parser every subcommand is added to.

    Each subparser registers its handler with `set_defaults(handler=fn)`,
    where `fn(args) -> int` returns the exit code (a `Handler`)."""
    parser = argparse.ArgumentParser(
        prog="aeo-index",
        description="Ask AI engines unbranded questions and see which tool each one names first.",
    )
    parser.add_argument("--version", action="version", version=f"aeo-index {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="command")

    questions = commands.add_parser("questions", help="print and check a question set")
    questions.add_argument(
        "--questions", type=Path, metavar="FILE", help="a question file (default: the ten index questions)"
    )
    questions.add_argument(
        "--allow-branded",
        action="store_true",
        help="load a question that names a tool as experimental instead of refusing it",
    )
    questions.set_defaults(handler=_cmd_questions)

    aliases = commands.add_parser("aliases", help="print the alias table")
    aliases.set_defaults(handler=_cmd_aliases)
    _add_run(commands)
    _add_rebuild(commands)
    _add_compare(commands)
    return parser


def http_client() -> httpx.AsyncClient:
    """The one HTTP client a command uses. Each provider sets its own
    deadline per call; redirects are never followed."""
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
        limits=httpx.Limits(max_connections=_MAX_CONNECTIONS),
    )


def _add_rebuild(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    rebuild_cmd = commands.add_parser(
        "rebuild",
        help="build an edition from a run's answers.jsonl, with no call",
        description="Rebuild index.json, table.md and the CSV files from per-answer facts.",
    )
    rebuild_cmd.add_argument("--answers", type=Path, required=True, metavar="FILE", help="an answers.jsonl")
    rebuild_cmd.add_argument(
        "--questions",
        type=Path,
        metavar="FILE",
        help="a question file (.toml) or a prompts.csv (default: the ten index questions)",
    )
    rebuild_cmd.add_argument("--brand", metavar="NAME", help="the brand whose counts fill the own-brand fields")
    rebuild_cmd.add_argument("--brand-alias", action="append", default=[], metavar="X", help="another spelling")
    rebuild_cmd.add_argument("--brand-domain", metavar="D", help="the brand's domain, such as example.com")
    rebuild_cmd.add_argument("--edition", metavar="YYYY-MM-DD", help="the last run date read (default: the newest)")
    rebuild_cmd.add_argument(
        "--under-review", action="append", default=[], metavar="ENGINE", help="an engine not read as a finding"
    )
    rebuild_cmd.add_argument("--out", type=Path, required=True, metavar="DIR", help="where the edition is written")
    rebuild_cmd.add_argument("--force", action="store_true", help="write into --out even when it is not empty")
    rebuild_cmd.set_defaults(handler=_cmd_rebuild)


def _add_compare(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    compare_cmd = commands.add_parser(
        "compare",
        help="print a run's table beside a published edition",
        description="Set a run's first-named table beside a Sextant edition (latest by default).",
    )
    compare_cmd.add_argument("run_dir", type=Path, metavar="RUN_DIR", help="a run's folder (holding index.json)")
    compare_cmd.add_argument(
        "--against",
        default="latest",
        metavar="latest|YYYY-MM-DD|PATH",
        help="the edition to compare with (default: latest)",
    )
    compare_cmd.set_defaults(handler=_cmd_compare)


def _add_run(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run = commands.add_parser(
        "run",
        help="ask the engines the questions and build the table",
        description="Ask the engines with your own keys. The plan and its cost are printed before any call.",
    )
    asked = run.add_argument_group("what is asked")
    asked.add_argument(
        "--questions", type=Path, metavar="FILE", help="a question file (default: the ten index questions)"
    )
    asked.add_argument("--engines", metavar="LIST", help="comma-separated engines (default: every configured engine)")
    asked.add_argument(
        "--samples", type=int, default=DEFAULT_SAMPLES, help=f"answers per cell, {MIN_SAMPLES} to {MAX_SAMPLES}"
    )
    asked.add_argument("--mode", default=MODE_SEARCH, help=f"{' | '.join(REQUESTABLE_MODES)} (default {MODE_SEARCH})")
    asked.add_argument("--engine-mode", action="append", default=[], metavar="ENGINE=MODE", help="one engine's mode")
    asked.add_argument("--model", action="append", default=[], metavar="ENGINE=MODEL", help="one engine's model")
    asked.add_argument("--country", default="US", help="where the results page is read from (US only in v0.1)")
    asked.add_argument("--brand", metavar="NAME", help="the brand whose counts fill the own-brand fields")
    asked.add_argument("--brand-alias", action="append", default=[], metavar="X", help="another spelling of --brand")
    asked.add_argument("--brand-domain", metavar="D", help="the brand's domain, such as example.com")
    asked.add_argument("--allow-branded", action="store_true", help="load a question that names a tool as experimental")
    _add_run_money(run)
    output = run.add_argument_group("output")
    output.add_argument("--out", type=Path, metavar="DIR", help="where the run is written")
    output.add_argument("--force", action="store_true", help="write into --out even when it is not empty")
    output.add_argument("--no-text", action="store_true", help="leave the answer text out of answers.jsonl")
    output.add_argument(
        "--compare",
        nargs="?",
        const="latest",
        metavar="latest|YYYY-MM-DD|PATH",
        help="print the table beside a published edition",
    )
    run.set_defaults(handler=_cmd_run)


def _add_run_money(run: argparse.ArgumentParser) -> None:
    judged = run.add_argument_group("the judge and the money")
    judged.add_argument(
        "--judge",
        default=DEFAULT_JUDGE,
        metavar="PROVIDER:MODEL",
        help=f"the model that lists the tools (default {DEFAULT_JUDGE})",
    )
    judged.add_argument(
        "--budget",
        default=str(DEFAULT_BUDGET_USD),
        metavar="USD",
        help=f"the cap for everything (default {DEFAULT_BUDGET_USD})",
    )
    judged.add_argument(
        "--price", action="append", default=[], metavar="ENGINE:MODEL=IN,OUT", help="USD per 1K tokens for a model"
    )
    judged.add_argument("--env-file", type=Path, metavar="PATH", help="read KEY=VALUE lines from this file")
    judged.add_argument("--yes", action="store_true", help="ask without waiting for y")
    judged.add_argument("--dry-run", action="store_true", help="print the plan and its cost; call nothing")


def _confirmed(plan: Plan, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        raise Refusal("not on a terminal, so there is no one to say y; pass --yes to run without asking")
    # The consent line states the cap the run enforces, not only the estimate.
    estimate = plan.estimate
    prompt = (
        f"Ask {len(plan.cells)} answers (plus up to {len(plan.cells)} judge calls; typical {money(estimate.typical)}, "
        f"high {money(estimate.high, up=True)}; capped at the {money(plan.budget)} budget)? [y/N] "
    )
    try:
        answer = input(prompt)
    except EOFError:
        answer = ""
    return answer.strip().lower() in ("y", "yes")


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        plan = prepare(args)
        print(render_plan(plan), flush=True)
        if args.dry_run:
            if plan.estimate.high > plan.budget:
                print(f"The high figure is over the budget: a real run needs --budget {budget_needed(plan)}.")
            return EXIT_OK
        check_budget(plan)
        check_out_dir(plan, args.force)
        if not _confirmed(plan, args.yes):
            print("Nothing was asked.", file=sys.stderr)
            return EXIT_REFUSED
    except Refusal as exc:
        print(exc, file=sys.stderr)
        return EXIT_REFUSED
    except KeyboardInterrupt:
        # The line starts on its own, after the prompt the interrupt cut short.
        print("\nInterrupted: nothing was asked and nothing was spent.", file=sys.stderr)
        return EXIT_INTERRUPTED
    tally = new_tally(plan)
    try:
        return asyncio.run(_execute(plan, args, tally))
    except KeyboardInterrupt:
        print(interrupted_line(tally, plan.out_dir), file=sys.stderr)
        return EXIT_INTERRUPTED


async def _execute(plan: Plan, args: argparse.Namespace, tally: RunTally) -> int:
    async with http_client() as client:
        return await execute_run(
            plan, include_text=not args.no_text, compare_ref=args.compare, client=client, tally=tally
        )


def _cmd_rebuild(args: argparse.Namespace) -> int:
    try:
        if args.edition is not None and not _ISO_DATE.fullmatch(args.edition):
            raise RebuildError(f"--edition must be a date as YYYY-MM-DD, got {args.edition!r}")
        unknown = [engine for engine in args.under_review if engine not in ENGINES]
        if unknown:
            raise RebuildError(f"--under-review: unknown engine {unknown[0]!r}; the engines are {', '.join(ENGINES)}")
        check_folder(args.out, args.force)
        qset = load_set(
            args.questions, brand=args.brand, brand_aliases=args.brand_alias, brand_domain=args.brand_domain
        )
        done = rebuild(args.answers, qset, args.out, edition=args.edition, engines_under_review=args.under_review)
    except (RebuildError, Refusal) as exc:
        print(exc, file=sys.stderr)
        return EXIT_REFUSED
    print(f"Rebuilt edition {done.doc['edition']} from {len(done.facts)} answers into {args.out}.")
    return EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    path = args.run_dir / INDEX_FILE if args.run_dir.is_dir() else args.run_dir
    return asyncio.run(_compare(path, args.against))


async def _compare(path: Path, against: str) -> int:
    async with http_client() as client:
        try:
            _, mine = await resolve_edition(str(path), client=client)
            label, ours = await resolve_edition(against, client=client)
        except CompareError as exc:
            print(f"Could not compare: {exc}", file=sys.stderr)
            return EXIT_REFUSED
    if "modes" not in mine:
        print(plain(f"Could not compare: {path} is not a run's edition (it has no modes block)"), file=sys.stderr)
        return EXIT_REFUSED
    print(render_compare(mine, ours, ours_label=label), end="")
    return EXIT_OK


def _load_set(path: Path | None, *, allow_branded: bool) -> QuestionSet:
    if path is None:
        return default_questions()
    return load_questions(path, allow_branded=allow_branded)


def _cmd_questions(args: argparse.Namespace) -> int:
    try:
        qset = _load_set(args.questions, allow_branded=args.allow_branded)
    except QuestionFileError as exc:
        print(exc, file=sys.stderr)
        return EXIT_REFUSED
    for question in qset.questions:
        flag = "experimental" if question.experimental else "fixed"
        print(f"{question.key}  {flag:<{_FLAG_WIDTH}}  {question.text}")
    count = len(qset.questions)
    print(f"{count} question{'s' if count != 1 else ''} in {qset.id}.", file=sys.stderr)
    return EXIT_OK


def _cmd_aliases(args: argparse.Namespace) -> int:
    for canonical, spellings in AliasTable.default().tools.items():
        print(f"{canonical}: {', '.join(spellings)}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line on `argv` (default: `sys.argv[1:]`) and return
    the exit code. With no subcommand it prints the help and returns 2."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return EXIT_REFUSED
    return handler(args)
