"""Asking every cell: the plan, the limits, the retries and the budget.

A cell is one (question, engine, mode, sample). `ask_all` asks them all
and returns one `Answer` per cell, in cell order, and never raises: every
failure is an answer with its status and error type.

The limits (spec D9):
  - at most `MAX_IN_FLIGHT` calls in flight in total and `MAX_PER_ENGINE`
    per engine;
  - each engine's first call goes alone, as a probe, so a key the vendor
    rejects costs one call: a 401 or 403 is never retried, stops that
    engine (its remaining cells become `error` rows of type `auth`,
    without a call), prints one line naming the key's variable, and the
    other engines continue;
  - a 429 or a 5xx is retried twice, after `RETRY_DELAYS` seconds with
    jitter, or after the vendor's `Retry-After` up to `MAX_RETRY_AFTER`;
    then it is `rate_limited` (429) or `error` (5xx);
  - every call has a deadline (`modes.CALL_TIMEOUT_SECONDS`, the AI
    Overview's own for results pages); past it the answer is `timeout`
    with cost 0 on its row, but the budget is charged that call's ceiling
    (`pricing.ceiling_call_usd`), since the vendor may have billed the
    generation, so the cap still holds;
  - before a cell is asked, the budget must have money left: once the
    spend (plus the judge's reserve) reaches the cap, the remaining cells
    are `skipped` without a call. Calls already in flight finish, so the
    cap can be overshot by at most the calls in flight at that moment.

A row's `cost_usd` is everything its cell was billed, retries included,
so the rows add up to the budget's spend (timeouts aside, as above).
Progress: one report per engine at the start, one every
`PROGRESS_INTERVAL_S` for each engine still running, and one when an
engine's last cell is done.
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

import httpx

from aeo_index.modes import (
    ENGINES,
    MAX_ANSWER_CITATIONS,
    MAX_SAMPLES,
    MAX_SEARCH_QUERIES,
    MAX_SEARCH_QUERY_CHARS,
    MIN_SAMPLES,
    MODES,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_RATE_LIMITED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
    engine_label,
)
from aeo_index.modes import (
    engine_modes as modes_for,
)
from aeo_index.pricing import RateCard, call_cost, ceiling_call_usd
from aeo_index.providers.base import Provider, ProviderError, ProviderReply, call_timeout, is_unreadable
from aeo_index.questions import QuestionSet
from aeo_index.records import Answer
from aeo_index.redact import Redactor

MAX_IN_FLIGHT: Final[int] = 8
MAX_PER_ENGINE: Final[int] = 4
RETRY_DELAYS: Final[tuple[float, ...]] = (1.0, 4.0)
MAX_RETRY_AFTER: Final[float] = 60.0
# Up to this fraction of a backoff delay is added at random, so retries
# from parallel calls do not arrive together.
RETRY_JITTER: Final[float] = 0.25
# The longest an engine's progress goes unreported.
PROGRESS_INTERVAL_S: Final[float] = 10.0

_RETRYABLE: Final[frozenset[str]] = frozenset({"rate_limited", "server"})
_STATUS_BY_ERROR: Final[Mapping[str, str]] = {
    "rate_limited": STATUS_RATE_LIMITED,
    "timeout": STATUS_TIMEOUT,
    "empty": STATUS_EMPTY,
}


@dataclass(frozen=True)
class Cell:
    """One call to make: question `key` with `text`, asked of `engine` in
    `mode`, sample number `sample` (from 1)."""

    key: str
    text: str
    engine: str
    mode: str
    sample: int


def plan_cells(
    questions: QuestionSet, engines: Sequence[str], mode: str, engine_modes: Mapping[str, str], samples: int
) -> list[Cell]:
    """Every cell of a run, ordered by question, then engine (in `ENGINES`
    order), then mode (in `MODES` order), then sample. `mode` is the run's
    `--mode`; `engine_modes` overrides it per engine."""
    if not MIN_SAMPLES <= samples <= MAX_SAMPLES:
        raise ValueError(f"samples must be {MIN_SAMPLES} to {MAX_SAMPLES}, got {samples}")
    unknown = [engine for engine in engines if engine not in ENGINES]
    if unknown:
        raise ValueError(f"unknown engine {unknown[0]!r}; expected one of {', '.join(ENGINES)}")
    ordered = [engine for engine in ENGINES if engine in engines]
    asked = {engine: modes_for(engine, engine_modes.get(engine, mode)) for engine in ordered}
    return [
        Cell(question.key, question.text, engine, cell_mode, sample)
        for question in questions.questions
        for engine in ordered
        for cell_mode in MODES
        if cell_mode in asked[engine]
        for sample in range(1, samples + 1)
    ]


class Budget:
    """A run's money: `cap` for everything, `reserved` held back for the
    judge, `spent` so far, `charges` (the calls paid for), and `skipped`,
    the cells left unasked at the cap.
    One event loop, so no lock."""

    def __init__(self, cap: Decimal, reserved: Decimal = Decimal("0")) -> None:
        if reserved < 0:
            raise ValueError("the reserve cannot be negative")
        self.cap = cap
        self.reserved = reserved
        self.spent = Decimal("0")
        self.skipped = 0
        self.charges = 0  # the charges of more than $0: the calls paid for

    @property
    def available(self) -> Decimal:
        return self.cap - self.reserved - self.spent

    def charge(self, amount: Decimal) -> None:
        if amount < 0:
            raise ValueError("a charge cannot be negative")
        self.spent += amount
        if amount > 0:
            self.charges += 1

    def release(self, amount: Decimal) -> None:
        """Hand `amount` of the reserve back (the judge spends from it)."""
        if amount < 0 or amount > self.reserved:
            raise ValueError(f"cannot release {amount} of a {self.reserved} reserve")
        self.reserved -= amount


@dataclass(frozen=True)
class Progress:
    """One engine so far: cells asked, answered (success) and failed
    (any other status but skipped), its spend, and the run's elapsed time."""

    engine: str
    asked: int
    answered: int
    failed: int
    spent: Decimal
    elapsed_s: float


@dataclass(frozen=True)
class _Outcome:
    reply: ProviderReply | None
    error: ProviderError | None
    cost: Decimal  # what the row records for this attempt
    latency_ms: int
    charge: Decimal | None = None  # what the budget is charged, when not `cost`


class _EngineState:
    def __init__(self, total: int) -> None:
        self.slots = asyncio.Semaphore(MAX_PER_ENGINE)
        self.probe_claimed = False
        self.probed = asyncio.Event()
        self.rejected: int | None = None  # the status code of the auth failure
        self.total = total
        self.done = self.asked = self.answered = self.failed = 0
        self.spent = Decimal("0")


class _Dispatch:
    def __init__(
        self,
        cells: Sequence[Cell],
        providers: Mapping[str, Provider],
        budget: Budget,
        card: RateCard,
        redactor: Redactor,
        client: httpx.AsyncClient,
        on_progress: Callable[[Progress], None] | None,
        sleep: Callable[[float], Awaitable[None]],
        clock: Callable[[], float],
    ) -> None:
        self.providers = providers
        self.budget = budget
        self.card = card
        self.redact = redactor
        self.client = client
        self.on_progress = on_progress
        self.sleep = sleep
        self.clock = clock
        self.started = clock()
        self.slots = asyncio.Semaphore(MAX_IN_FLIGHT)
        self.progress_broken = False
        totals: dict[str, int] = {}
        for cell in cells:
            totals[cell.engine] = totals.get(cell.engine, 0) + 1
        self.engines = {engine: _EngineState(total) for engine, total in totals.items()}

    async def ask(self, cell: Cell) -> Answer:
        try:
            answer = await self._ask(cell)
        except Exception as exc:  # never raise: an unforeseen failure is a row too
            answer = _failed(cell, self._model(cell), "bad_response", self.redact(f"{exc.__class__.__name__}: {exc}"))
        self._record(cell.engine, answer)
        return answer

    def _model(self, cell: Cell) -> str:
        provider = self.providers.get(cell.engine)
        return provider.model if provider is not None else cell.engine

    async def _ask(self, cell: Cell) -> Answer:
        provider = self.providers.get(cell.engine)
        if provider is None:
            return _failed(cell, cell.engine, "client", f"no provider for {cell.engine}; it was not configured")
        state = self.engines[cell.engine]
        is_probe = not state.probe_claimed
        state.probe_claimed = True
        if not is_probe:
            await state.probed.wait()
        try:
            async with state.slots:
                return await self._attempts(cell, provider, state)
        finally:
            if is_probe:
                state.probed.set()

    async def _attempts(self, cell: Cell, provider: Provider, state: _EngineState) -> Answer:
        billed = Decimal("0")  # every attempt's cost, so the row matches the spend
        for attempt in range(len(RETRY_DELAYS) + 1):
            if state.rejected is not None:
                return self._not_asked_auth(cell, provider, state.rejected)
            async with self.slots:
                if attempt == 0 and self.budget.available <= 0:
                    self.budget.skipped += 1
                    return _failed(
                        cell, provider.model, "budget", "the run's budget was reached; not asked", STATUS_SKIPPED
                    )
                if attempt == 0:
                    state.asked += 1
                outcome = await self._call(cell, provider)
            charged = outcome.cost if outcome.charge is None else outcome.charge
            self.budget.charge(charged)
            state.spent += charged
            billed += outcome.cost
            error = outcome.error
            if error is None and outcome.reply is not None:
                return self._success(cell, outcome.reply, outcome, billed)
            assert error is not None
            if error.error_type in _RETRYABLE and attempt < len(RETRY_DELAYS):
                await self.sleep(_retry_delay(error, attempt))
                continue
            if error.error_type == "auth":
                self._reject(provider, state, error.status_code)
            return self._failure(cell, provider, error, outcome, billed)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _call(self, cell: Cell, provider: Provider) -> _Outcome:
        deadline = call_timeout(cell.engine, cell.mode)
        start = self.clock()
        try:
            reply = await asyncio.wait_for(
                provider.ask(cell.text, mode=cell.mode, client=self.client), timeout=deadline
            )
            error: ProviderError | None = None
        except TimeoutError:
            reply, error = None, ProviderError(f"no answer within {deadline:g} s", error_type="timeout")
        except ProviderError as exc:
            reply, error = None, exc
        except Exception as exc:  # noqa: BLE001 - a reply that broke the provider is unreadable, not a crash
            reply, error = (
                None,
                ProviderError(f"the reply could not be read ({exc.__class__.__name__})", error_type="bad_response"),
            )
        elapsed = self.clock() - start
        if elapsed >= deadline or (error is not None and error.error_type == "timeout"):
            message = str(error) if error is not None else f"no answer within {deadline:g} s"
            timed_out = ProviderError(message, error_type="timeout")
            return _Outcome(None, timed_out, Decimal("0"), int(deadline * 1000), self._ceiling(cell, provider))
        latency_ms = int(elapsed * 1000)
        if error is not None:
            if is_unreadable(error):
                # Billed, perhaps, but not readable: the row records $0 and the budget its ceiling.
                return _Outcome(None, error, Decimal("0"), latency_ms, self._ceiling(cell, provider))
            billed = error.cost_usd
            if error.reply is not None:
                billed = call_cost(cell.engine, provider.model, cell.mode, error.reply, self.card)
            return _Outcome(None, error, billed, latency_ms)
        assert reply is not None
        return _Outcome(reply, None, call_cost(cell.engine, provider.model, cell.mode, reply, self.card), latency_ms)

    def _ceiling(self, cell: Cell, provider: Provider) -> Decimal:
        try:
            return ceiling_call_usd(cell.engine, provider.model, cell.mode, self.card)
        except ValueError:
            return Decimal("0")

    def _success(self, cell: Cell, reply: ProviderReply, outcome: _Outcome, billed: Decimal) -> Answer:
        return Answer(
            key=cell.key,
            engine=cell.engine,
            mode=cell.mode,
            sample=cell.sample,
            status=STATUS_SUCCESS,
            text=reply.text,
            model_version=reply.model_version,
            latency_ms=outcome.latency_ms,
            cost_usd=billed,
            citations=tuple(reply.citations or ())[:MAX_ANSWER_CITATIONS],
            search_count=reply.search_count,
            search_queries=_bounded_queries(reply.search_queries),
            error_type=None,
            error=None,
        )

    def _failure(
        self, cell: Cell, provider: Provider, error: ProviderError, outcome: _Outcome, billed: Decimal
    ) -> Answer:
        partial = error.reply
        return Answer(
            key=cell.key,
            engine=cell.engine,
            mode=cell.mode,
            sample=cell.sample,
            status=_STATUS_BY_ERROR.get(error.error_type, STATUS_ERROR),
            text=None,
            model_version=partial.model_version if partial is not None else provider.model,
            latency_ms=outcome.latency_ms,
            cost_usd=billed,
            citations=(),
            search_count=partial.search_count if partial is not None else None,
            search_queries=_bounded_queries(partial.search_queries) if partial is not None else None,
            error_type=error.error_type,
            error=self.redact(str(error)),
        )

    def _reject(self, provider: Provider, state: _EngineState, status_code: int | None) -> None:
        if state.rejected is not None:
            return
        state.rejected = status_code or 0
        code = status_code if status_code is not None else "rejected"
        print(
            f"{provider.env_var} was rejected ({code}); {engine_label(provider.engine)} is skipped, "
            "the other engines continue.",
            file=sys.stderr,
        )

    def _not_asked_auth(self, cell: Cell, provider: Provider, status_code: int) -> Answer:
        message = f"{provider.env_var} was rejected ({status_code}) earlier in this run; not asked"
        return _failed(cell, provider.model, "auth", message)

    def _record(self, engine: str, answer: Answer) -> None:
        state = self.engines.get(engine)
        if state is None:
            return
        state.done += 1
        if answer.status == STATUS_SUCCESS:
            state.answered += 1
        elif answer.status != STATUS_SKIPPED:
            state.failed += 1
        if state.done == state.total:
            self.report(engine)

    def report(self, engine: str) -> None:
        """Tell `on_progress` where `engine` is. A callback that raises loses
        its reports (said once on stderr), never the run."""
        if self.on_progress is None or self.progress_broken:
            return
        state = self.engines[engine]
        progress = Progress(engine, state.asked, state.answered, state.failed, state.spent, self.clock() - self.started)
        try:
            self.on_progress(progress)
        except Exception as exc:
            self.progress_broken = True
            print(self.redact(f"progress reports stopped: {exc.__class__.__name__}: {exc}"), file=sys.stderr)

    def running(self) -> list[str]:
        return [engine for engine, state in self.engines.items() if state.done < state.total]

    async def tick(self) -> None:
        """Report every engine at the start, then each running engine every
        `PROGRESS_INTERVAL_S`, until cancelled."""
        for engine in self.engines:
            self.report(engine)
        while True:
            await self.sleep(PROGRESS_INTERVAL_S)
            # Yield even if the injected sleep did not, so the calls run.
            await asyncio.sleep(0)
            for engine in self.running():
                self.report(engine)


def _failed(cell: Cell, model: str, error_type: str, message: str, status: str = STATUS_ERROR) -> Answer:
    return Answer(
        key=cell.key,
        engine=cell.engine,
        mode=cell.mode,
        sample=cell.sample,
        status=status,
        text=None,
        model_version=model,
        latency_ms=0,
        cost_usd=Decimal("0"),
        citations=(),
        search_count=None,
        search_queries=None,
        error_type=error_type,
        error=message,
    )


def _retry_delay(error: ProviderError, attempt: int) -> float:
    if error.retry_after is not None:
        return min(error.retry_after, MAX_RETRY_AFTER)
    base = RETRY_DELAYS[attempt]
    return base + random.uniform(0, base * RETRY_JITTER)


def _bounded_queries(queries: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if queries is None:
        return None
    return tuple(query[:MAX_SEARCH_QUERY_CHARS] for query in queries[:MAX_SEARCH_QUERIES])


async def ask_all(
    cells: Sequence[Cell],
    providers: Mapping[str, Provider],
    *,
    budget: Budget,
    card: RateCard,
    redactor: Redactor,
    client: httpx.AsyncClient,
    on_progress: Callable[[Progress], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[Answer]:
    """Ask every cell under the limits and the budget. One `Answer` per
    cell, in cell order; never raises. `on_progress` hears each engine once
    at the start, every `PROGRESS_INTERVAL_S` (by `sleep` and `clock`)
    while it runs, and once when its last cell is done."""
    run = _Dispatch(cells, providers, budget, card, redactor, client, on_progress, sleep, clock)
    ticker = asyncio.ensure_future(run.tick()) if on_progress is not None else None
    try:
        return list(await asyncio.gather(*(run.ask(cell) for cell in cells)))
    finally:
        if ticker is not None:
            await _stop_ticker(ticker, redactor)


async def _stop_ticker(ticker: asyncio.Future[None], redactor: Redactor) -> None:
    """Cancel the progress ticker. A ticker that already failed loses its
    reports, said once on stderr through the redactor, never the answers."""
    ticker.cancel()
    try:
        await ticker
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise  # the run itself is being cancelled, not just the ticker
    except Exception as exc:
        print(redactor(f"progress reports stopped: {exc.__class__.__name__}: {exc}"), file=sys.stderr)
