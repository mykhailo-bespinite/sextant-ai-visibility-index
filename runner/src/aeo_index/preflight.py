"""`aeo-index run` before the first call: read the flags and keys, plan the
cells, price them, and refuse anything that cannot run safely.

`prepare` returns a `Plan` or raises `Refusal` (exit 2) with the reason;
it never makes a network call. `render_plan` is the text printed before
any money is spent: per engine and mode the cells, a typical and a high
figure, the judge's share, the totals, and the engines not configured.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, ROUND_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Final

from aeo_index.dispatch import Cell, plan_cells
from aeo_index.envfile import EnvFileError, load_env_file
from aeo_index.modes import (
    ENGINES,
    MAX_SAMPLES,
    MIN_SAMPLES,
    MODE_API,
    REQUESTABLE_MODES,
    RETRIEVAL_ENGINES,
    engine_label,
)
from aeo_index.pricing import TYPICAL_AS_OF, Estimate, RateCard, estimate, unpriced
from aeo_index.providers.base import (
    AIO_LOGIN_VAR,
    AIO_PASSWORD_VAR,
    DEFAULT_MODELS,
    KEY_ENV_VARS,
    Provider,
    build_providers,
    check_model,
)
from aeo_index.questions import QuestionFileError, QuestionSet, default_questions, load_questions, make_brand
from aeo_index.records import JudgeSpec
from aeo_index.redact import Redactor, secret_values

DEFAULT_JUDGE: Final[str] = "anthropic:claude-haiku-4-5-20251001"
DEFAULT_BUDGET_USD: Final[Decimal] = Decimal("20.00")
SUPPORTED_COUNTRIES: Final[tuple[str, ...]] = ("US",)
DEFAULT_RUNS_DIR: Final[Path] = Path("aeo-index-runs")
_CENT: Final[Decimal] = Decimal("0.01")


class Refusal(Exception):
    """The run cannot start; the message says why and what to change."""


@dataclass(frozen=True)
class Plan:
    """A run that passed every check and is ready to ask."""

    questions: QuestionSet
    cells: tuple[Cell, ...]
    providers: Mapping[str, Provider]
    not_configured: Mapping[str, str]
    models: Mapping[str, str]
    judge: JudgeSpec
    card: RateCard
    estimate: Estimate
    budget: Decimal
    redactor: Redactor
    env: Mapping[str, str]
    out_dir: Path
    mode: str
    samples: int


def _split_pair(value: str, sep: str, flag: str, shape: str) -> tuple[str, str]:
    left, found, right = value.partition(sep)
    if not found or not left.strip() or not right.strip():
        raise Refusal(f"{flag} takes {shape}, got {value!r}")
    return left.strip(), right.strip()


def _engine(name: str, flag: str) -> str:
    if name not in ENGINES:
        raise Refusal(f"{flag}: unknown engine {name!r}; the engines are {', '.join(ENGINES)}")
    return name


def _engines(value: str | None) -> list[str]:
    if value is None:
        return list(ENGINES)
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise Refusal("--engines needs at least one engine")
    return [_engine(name, "--engines") for name in dict.fromkeys(names)]


def _engine_modes(values: Sequence[str]) -> dict[str, str]:
    chosen: dict[str, str] = {}
    for value in values:
        engine, mode = _split_pair(value, "=", "--engine-mode", "ENGINE=MODE")
        _engine(engine, "--engine-mode")
        if mode not in REQUESTABLE_MODES:
            raise Refusal(f"--engine-mode: {mode!r} is not a mode; use {', '.join(REQUESTABLE_MODES)}")
        if engine in RETRIEVAL_ENGINES and mode == MODE_API:
            raise Refusal(f"--engine-mode: {engine} always searches, so it is asked in search mode only")
        chosen[engine] = mode
    return chosen


def _models(values: Sequence[str]) -> dict[str, str]:
    chosen = dict(DEFAULT_MODELS)
    for value in values:
        engine, model = _split_pair(value, "=", "--model", "ENGINE=MODEL")
        _engine(engine, "--model")
        try:
            chosen[engine] = check_model(model)
        except ValueError as exc:
            raise Refusal(f"--model: {exc}") from exc
    return chosen


def _decimal(text: str, flag: str) -> Decimal:
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise Refusal(f"{flag}: {text!r} is not a number") from exc
    if not number.is_finite() or number < 0:
        raise Refusal(f"{flag}: {text!r} must be a number of 0 or more")
    return number


def _card(values: Sequence[str]) -> RateCard:
    card = RateCard.default()
    for value in values:
        target, prices = _split_pair(value, "=", "--price", "ENGINE:MODEL=IN,OUT (USD per 1K tokens)")
        engine, model = _split_pair(target, ":", "--price", "ENGINE:MODEL=IN,OUT (USD per 1K tokens)")
        _engine(engine, "--price")
        input_price, output_price = _split_pair(prices, ",", "--price", "ENGINE:MODEL=IN,OUT (USD per 1K tokens)")
        card = card.with_override(engine, model, _decimal(input_price, "--price"), _decimal(output_price, "--price"))
    return card


def _budget(text: str) -> Decimal:
    budget = _decimal(text, "--budget")
    if budget <= 0:
        raise Refusal("--budget must be more than 0")
    return budget


def _samples(value: int) -> int:
    if not MIN_SAMPLES <= value <= MAX_SAMPLES:
        raise Refusal(f"--samples must be {MIN_SAMPLES} to {MAX_SAMPLES}, got {value}")
    return value


def _questions(args: argparse.Namespace) -> QuestionSet:
    try:
        qset = (
            load_questions(args.questions, allow_branded=args.allow_branded) if args.questions else default_questions()
        )
        if args.brand is None:
            if args.brand_alias or args.brand_domain:
                raise Refusal("--brand-alias and --brand-domain need --brand")
            return qset
        brand = make_brand(args.brand, args.brand_alias or (), args.brand_domain)
        return qset.with_brand(brand, allow_branded=args.allow_branded)
    except QuestionFileError as exc:
        raise Refusal(str(exc)) from exc


def _env(env_file: Path | None, environ: Mapping[str, str]) -> dict[str, str]:
    if env_file is None:
        return dict(environ)
    try:
        return load_env_file(env_file, environ)
    except EnvFileError as exc:
        raise Refusal(str(exc)) from exc


# Each engine's key variables, the results page's login and password included.
_KEY_VARIABLES_OF: Final[Mapping[str, tuple[str, ...]]] = {
    **KEY_ENV_VARS,
    "google_aio": (AIO_LOGIN_VAR, AIO_PASSWORD_VAR),
}


def _check_keys(env: Mapping[str, str], engines: Sequence[str]) -> None:
    """Refuse a key of an engine asked (or of the judge) with a space or a
    character outside ASCII inside it (a paste gone wrong, which no vendor
    accepts and an HTTP header cannot carry), naming the variable and never
    the value. Keys of engines not asked are not read."""
    names = [name for engine in engines for name in _KEY_VARIABLES_OF.get(engine, ())]
    for name in names:
        value = (env.get(name) or "").strip()
        if value and (not value.isascii() or any(ch.isspace() for ch in value)):
            raise Refusal(f"{name} holds a space or a character outside ASCII; paste the key again")


def _judge(value: str, env: Mapping[str, str]) -> JudgeSpec:
    try:
        judge = JudgeSpec.parse(value)
    except ValueError as exc:
        raise Refusal(str(exc)) from exc
    names = KEY_ENV_VARS[judge.provider]
    if not any((env.get(name) or "").strip() for name in names):
        raise Refusal(
            f"the judge {judge} needs {names[0]}; set it, or choose a judge you have a key for with "
            "--judge PROVIDER:MODEL"
        )
    return judge


def _out_dir(args: argparse.Namespace, set_id: str) -> Path:
    if args.out is not None:
        return Path(args.out)
    return DEFAULT_RUNS_DIR / f"{datetime.now(UTC).date().isoformat()}-{set_id}"


def _check_out_dir(path: Path, force: bool) -> None:
    if force or not path.exists():
        return
    if not path.is_dir():
        raise Refusal(f"--out {path} is a file, not a folder")
    if any(path.iterdir()):
        raise Refusal(f"--out {path} is in use (not empty); choose another folder or pass --force")


def prepare(args: argparse.Namespace, environ: Mapping[str, str] | None = None) -> Plan:
    """Check everything a run needs, without a network call."""
    country = (args.country or "").strip().upper()
    if country not in SUPPORTED_COUNTRIES:
        raise Refusal("--country accepts US only in v0.1 (DataForSEO location 2840)")
    if args.mode not in REQUESTABLE_MODES:
        raise Refusal(f"--mode must be one of {', '.join(REQUESTABLE_MODES)}, got {args.mode!r}")
    samples = _samples(args.samples)
    budget = _budget(args.budget)
    requested = _engines(args.engines)
    engine_modes = _engine_modes(args.engine_mode)
    models = _models(args.model)
    card = _card(args.price)
    env = _env(args.env_file, os.environ if environ is None else environ)
    qset = _questions(args)
    providers, missing = build_providers(env, ENGINES, models)
    asked = {engine: provider for engine, provider in providers.items() if engine in requested}
    if not asked:
        wanted = ", ".join(f"{engine} ({missing[engine]})" for engine in requested if engine in missing)
        raise Refusal(f"no engine is configured: {wanted}")
    judge = _judge(args.judge, env)
    _check_keys(env, [*requested, judge.provider])
    asked_models = {engine: models[engine] for engine in ENGINES if engine in asked}
    missing_prices = unpriced(asked_models, judge, card)
    if missing_prices:
        first = missing_prices[0]
        raise Refusal(
            f"no price for {', '.join(missing_prices)}; pass --price {first}=IN,OUT (USD per 1K tokens) "
            "so the budget can hold"
        )
    cells = tuple(plan_cells(qset, list(asked), args.mode, engine_modes, samples))
    return Plan(
        questions=qset,
        cells=cells,
        providers=asked,
        # Only the engines asked for: one left out with --engines is not "not configured".
        not_configured={engine: missing[engine] for engine in ENGINES if engine in missing and engine in requested},
        models=asked_models,
        judge=judge,
        card=card,
        estimate=estimate(cells, asked_models, judge, card),
        budget=budget,
        redactor=Redactor(secret_values(env)),
        env=env,
        out_dir=_out_dir(args, qset.id),
        mode=args.mode,
        samples=samples,
    )


def money(amount: Decimal, *, up: bool = False) -> str:
    """`amount` as dollars and cents; `up` rounds up, so a ceiling is never
    printed below itself."""
    return f"${amount.quantize(_CENT, rounding=ROUND_UP if up else ROUND_HALF_UP)}"


def budget_needed(plan: Plan) -> Decimal:
    """The smallest `--budget` the high figure fits under."""
    return plan.estimate.high.quantize(_CENT, rounding=ROUND_UP)


def check_budget(plan: Plan) -> None:
    if plan.estimate.high > plan.budget:
        raise Refusal(
            f"the high estimate {money(plan.estimate.high, up=True)} is over the budget {money(plan.budget)}; "
            f"to run it anyway pass --budget {budget_needed(plan)}"
        )


def check_out_dir(plan: Plan, force: bool) -> None:
    _check_out_dir(plan.out_dir, force)


def check_folder(path: Path, force: bool) -> None:
    """Refuse a folder that is a file, or is not empty, unless `force`."""
    _check_out_dir(path, force)


def render_plan(plan: Plan) -> str:
    """The plan and its cost, as printed before the first call."""
    qcount = len(plan.questions.questions)
    lines = [
        f"Question set {plan.questions.id}: {qcount} question{'s' if qcount != 1 else ''}, "
        f"{plan.samples} sample{'s' if plan.samples != 1 else ''} each, mode {plan.mode}.",
    ]
    lines.extend(f"not configured: {engine} ({reason})" for engine, reason in plan.not_configured.items())
    rows = [("engine", "mode", "model", "calls", "typical", "high")]
    for engine, mode, cells, typical, high in plan.estimate.rows:
        rows.append((engine_label(engine), mode, plan.models[engine], str(cells), money(typical), money(high, up=True)))
    answers = len(plan.cells)
    rows.append(("judge", "", str(plan.judge), f"{answers} answers", "", money(plan.estimate.judge_high, up=True)))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines.extend("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in rows)
    lines.append(
        f"{answers} answers plus up to {answers} judge calls; typical {money(plan.estimate.typical)}, "
        f"high {money(plan.estimate.high, up=True)} "
        f"with the judge; budget {money(plan.budget)}."
    )
    lines.append(f"Typical and high figures per call measured as of {TYPICAL_AS_OF}.")
    return "\n".join(lines)
