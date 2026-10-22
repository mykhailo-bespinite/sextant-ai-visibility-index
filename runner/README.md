# aeo-index

`aeo-index` runs the Sextant AI Visibility Index method on your own machine with your own API keys. It asks a set of unbranded questions of the AI engines, has a second model list the tools each answer recommends in order, and prints which tool each engine names first, per mode, with the cost shown before the first call. The default questions are the ten the index publishes, and a question file of your own replaces them for another category.

The code is MIT licensed. The published index data in the same repository is CC BY 4.0: https://github.com/mykhailo-bespinite/sextant-ai-visibility-index

## Install

Python 3.12 or newer. The only dependency is `httpx`.

```bash
uv tool install aeo-index
# or
pipx install aeo-index
# or, without installing
uvx aeo-index questions
```

## Keys

Every engine is asked with your key and billed to your account. Keys are read from the environment. Nothing is read from a file unless you pass `--env-file PATH`, which takes `KEY=VALUE` lines (`export KEY=VALUE` and quoted values work; a variable already in the environment wins over the file). `.env.example` in this folder lists every variable with no value.

| Variable | Engine |
|---|---|
| `OPENAI_API_KEY` | ChatGPT (`openai`) |
| `ANTHROPIC_API_KEY` | Claude (`anthropic`), and the default judge |
| `GEMINI_API_KEY`, or `GOOGLE_API_KEY` | Gemini (`gemini`) |
| `XAI_API_KEY`, or `GROK_API_KEY` | Grok (`grok`) |
| `PERPLEXITY_API_KEY` | Perplexity (`perplexity`) |
| `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` | Google AI Overview, results page via DataForSEO (`google_aio`) |

A key of an engine asked, or of the judge, with a space or a character outside ASCII inside it is refused, naming the variable. An engine without a key is not asked. The plan names it once (`not configured: grok (set XAI_API_KEY)`) and it produces no rows. Google AI Overview is a scraped results page read through DataForSEO, not a model answering, so every table labels it that way.

## A first run

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
aeo-index run --dry-run     # the plan and its cost, no call
aeo-index run --compare     # asks after you type y
```

A run checks the flags and keys, prices the plan and prints it, then asks "Ask N answers (plus up to N judge calls; typical $A, high $B; capped at the $C budget)? [y/N]" on a terminal (or takes `--yes`). It asks every cell and has the judge read every successful answer. Then it writes six files and prints the table, the summary, the files written and, with `--compare`, the side-by-side table. Progress goes to stderr; the plan, the table and the summary go to stdout.

Progress is one line per engine at the start, at most every 10 seconds while it runs, and when it finishes. The figures in these examples show the shape of the lines, not a measured run:

```
Claude: asked 12 of 30, answered 11, failed 1, $0.82 so far, 40 s elapsed, about 1 min 20 s left
Judge: 40 of 178 answers, $0.12 spent in all, 30 s elapsed, about 1 min left
```

## The flags

`aeo-index --version` prints the version. Each command has `--help`.

### `aeo-index run`

#### What is asked

| Flag | Meaning |
|---|---|
| `--questions FILE` | A question file (below). Default: the ten index questions, set `sextant-index-v1`. |
| `--engines LIST` | Comma-separated engines: `anthropic`, `openai`, `perplexity`, `gemini`, `grok`, `google_aio`. Default: every engine with a key. |
| `--samples N` | Answers per question, engine and mode, 1 to 5. Default 3. |
| `--mode MODE` | `search` (default), `api` or `both`. See "Modes". |
| `--engine-mode ENGINE=MODE` | One engine's mode, overriding `--mode`. Repeatable. |
| `--model ENGINE=MODEL` | One engine's model. Repeatable. Defaults: `openai` gpt-5.5, `anthropic` claude-sonnet-4-6, `gemini` models/gemini-3.5-flash, `grok` grok-4.3, `perplexity` sonar-pro. |
| `--country COUNTRY` | Where the results page is read from. `US` only in v0.1 (DataForSEO location 2840). |
| `--brand NAME` | The brand whose counts fill the own-brand fields of the edition. |
| `--brand-alias X` | Another spelling of `--brand`. Repeatable. |
| `--brand-domain D` | The brand's domain, such as example.com. |
| `--allow-branded` | Load a question that names a tool as experimental instead of refusing the file. |

#### The judge and the money

| Flag | Meaning |
|---|---|
| `--judge PROVIDER:MODEL` | The model that lists the tools. `PROVIDER` is `openai`, `anthropic`, `gemini` or `grok`. Default `anthropic:claude-haiku-4-5-20251001`. The judge's key must be set. |
| `--budget USD` | The cap for everything, asking and judging together. Default 20.00. |
| `--price ENGINE:MODEL=IN,OUT` | USD per 1K input and output tokens for a model the package has no price for. Repeatable. A model with no price is refused, because a budget cannot hold over an unknown price. A dated snapshot of a priced model (`gpt-5.5-2026-04-23`, `claude-haiku-4-5-20251001`) takes its price; any other variant (`gpt-5.5-pro`) is another model and needs `--price`. |
| `--env-file PATH` | Read `KEY=VALUE` lines from this file. |
| `--yes` | Ask without waiting for `y`. Needed when stdin is not a terminal. |
| `--dry-run` | Print the plan and its cost and call nothing. Over the budget, it also names the `--budget` a real run needs. |

#### Output

| Flag | Meaning |
|---|---|
| `--out DIR` | Where the run is written. Default `./aeo-index-runs/<UTC date>-<question set id>/`. A folder that exists and is not empty is refused. |
| `--force` | Write into `--out` even when it is not empty. |
| `--no-text` | Leave the answer text out of `answers.jsonl`. |
| `--compare [latest\|YYYY-MM-DD\|PATH]` | After the run, print its table beside a published edition. With no value, `latest`. See "Compare". |

### `aeo-index rebuild`

```
aeo-index rebuild --answers FILE --out DIR [--questions FILE] [--brand NAME] [--brand-alias X]
                  [--brand-domain D] [--edition YYYY-MM-DD] [--under-review ENGINE] [--force]
```

Builds the six files from a run's `answers.jsonl` with no call. `answers.jsonl` in the output holds the facts only.

| Flag | Meaning |
|---|---|
| `--answers FILE` | An `answers.jsonl`. Required. |
| `--out DIR` | Where the edition is written. Required; refused when not empty unless `--force`. |
| `--questions FILE` | A question file (`.toml`) or a `prompts.csv` (this package's columns, or the published `key,prompt,experimental`). Default: the ten index questions. A `prompts.csv` holding exactly the index questions is read as the index set, so the Sextant brand and domain apply; any other CSV tracks no brand unless `--brand` names one. |
| `--brand NAME`, `--brand-alias X`, `--brand-domain D` | As for `run`. |
| `--edition YYYY-MM-DD` | Keep only facts dated on or before this day. Default: the newest. |
| `--under-review ENGINE` | List an engine under review in the edition. Repeatable. |
| `--force` | Write into `--out` even when it is not empty. |

Only the questions that have answers are listed, and each run date is one run. It exits 0, or 2 with a reason that names the line for bad input.

### `aeo-index compare`

```
aeo-index compare RUN_DIR [--against latest|YYYY-MM-DD|PATH]
```

Prints a run's table (`RUN_DIR`, or the `index.json` inside it) beside an edition. `--against` defaults to `latest`. It exits 0, or 2 with one line saying why when either edition cannot be read.

### `aeo-index questions` and `aeo-index aliases`

`aeo-index questions [--questions FILE] [--allow-branded]` checks a question set and prints it, one question a line. `aeo-index aliases` prints the alias table: each tool's canonical name and the spellings folded into it.

## The question file

A TOML file. The example in the repository, `runner/examples/example-category.toml`:

```toml
id = "crm-for-agencies-v1"
subject = "CRM software for marketing agencies"
purpose = "running a marketing agency's client relationships"
generic_example = "a CRM tool"
exclude = ["ChatGPT", "Claude", "Perplexity", "Gemini", "Grok"]
brand = "Acme CRM"
brand_aliases = ["Acme", "acmecrm.com"]
brand_domain = "acmecrm.com"

[aliases]            # optional: more spellings to fold, canonical = [spellings]
"Zed CRM" = ["Zed", "zedcrm.io"]

[[question]]
key = "q01"
text = "What is the best CRM for a small marketing agency?"
experimental = false
```

```bash
aeo-index questions --questions my-category.toml
aeo-index run --questions my-category.toml
```

`id`, `subject`, `purpose` and at least one `[[question]]` are required. `subject`, `purpose`, `generic_example` and `exclude` are the words the judge's prompt uses about your category; left out, `generic_example` and `exclude` take the index's. Any other top-level field is refused. Keys are unique, 1 to 32 characters of `a-z`, `0-9` and `_`. A file holds 1 to 50 questions, each 10 to 500 characters.

A question that names a tool (any spelling in the alias table, the file's own `[aliases]`, or the brand) is refused at load, naming its key and the name found, because an index question must not name a tool. `--allow-branded` loads it as experimental instead: it is asked and judged, and kept out of the leader board. The assistants' names (ChatGPT, Gemini and the rest) are not tool names here.

The brand is `--brand`, else the file's `brand`. For the default set it is Sextant AEO with its spellings and sextantaeo.com, so a default run fills every field our edition has, our own line included. A file with no brand writes `producer.brand = null`, and the own-brand fields count nothing.

## Modes

`search` asks each engine with its web search on, which is what a person sees in the consumer apps. `api` asks the model without search, from what it learned in training. `both` asks both, and the two are always counted apart.

Perplexity and Google AI Overview always search, so they are asked in `search` only, whatever the mode; `--engine-mode perplexity=api` is refused. ChatGPT, Claude, Gemini and Grok can be asked either way. With search on, ChatGPT, Claude and Grok make at most three web-search tool calls per answer. Gemini's search takes no limit, and each query it runs is billed. The ceiling charged for a call that times out (and the high figure of a model with no measured figure) assumes three searches for Gemini too, so a Gemini answer that searches more can cost more than that ceiling. Each answer keeps at most ten citations.

## Cost and the budget

Before the first call the plan prints, per engine and mode, the calls, a typical figure and a high figure, then the judge's share and the totals. `--dry-run` prints it and stops.

`--budget` (default 20.00 USD) covers the asking and the judging together:

- the run refuses to start when the high figure is over the budget, and names the `--budget` it would need;
- the judge's high figure is held back from the start, so the asking cannot spend the judge's share;
- once the spend reaches the budget, the cells not yet asked are recorded `skipped` with no call, and answers not yet judged stay unjudged;
- a call that times out, or whose reply arrived but could not be read, is charged its high figure, because the vendor may have billed it, while its row records $0. This holds for the judge's calls and their retry too. The summary says how much of the actual cost that is.

Calls already in flight finish, with their retries, so the spend can pass the cap by at most those calls.

## How engines are asked

At most 8 calls are in flight, and at most 4 per engine. Each engine's first call goes alone, so a key the vendor rejects costs one call. A 401 or 403 is never retried: it stops that engine, its remaining cells become `error` rows of type `auth` with no call, one line names the variable to fix, and the other engines go on. A 429 or a 5xx is retried twice, after about 1 and 4 seconds, or after the vendor's `Retry-After` up to 60 seconds; then it is recorded `rate_limited` or `error`. Deadlines are 90 seconds a call without search, 180 seconds with it, and 120 seconds for a results page.

Every answer is a row with one of six statuses: `success`, `empty`, `error`, `timeout`, `rate_limited` or `skipped`. Failures are counted in `answers` and never in `ok`, and an engine with no successful answer appears in every table as having no answers, with its status counts.

## The judge and grounding

Each successful answer goes to the judge with a prompt that wraps it in `<answer>` tags as data, asks for `{"tools": [...]}` in the order the answer recommends them, and excludes generic terms and the assistants' own names. With the default question set the prompt is the one our published index uses. The judge gets 500 output tokens and one retry; at most 4 judge calls run at once.

Every name the judge lists is folded through the alias table (`aeo-index aliases`) plus the question file's aliases. The grounding check then keeps a name only when that spelling, or a spelling of its canonical name, occurs in the answer text, case-insensitive and on word boundaries. A domain-like name such as acmecrm.com counts only as that host, with or without `www.`, so `acmecrm.com.example.net` does not ground it. Dropped names are counted as `judge_ungrounded`, printed in the summary and recorded in `index.json`; the check removes names and never adds one.

`sextant_rank` is the position of the tracked brand in the judged list. A mention the judge did not list is not a rank.

A reasoning model used as the judge can spend its 500 tokens thinking and return no list. That answer counts as unjudged and the run exits 3. The default judge is not asked to reason.

## Outputs

`--out DIR` gets six files:

| File | Contents |
|---|---|
| `index.json` | The edition, in the format the index's research page reads: runs, prompts, a top level and a `modes` block with per-engine totals, leaderboard, leaderboard by engine, top cited domains and URLs, then `by_prompt` and `leaderboard_by_prompt`, plus `basis` and a `producer` block (package version, judge, models, brand, question set, `judge_ungrounded`, `extract_version` 2). Never any answer text. |
| `table.md` | The first-named table: one table per mode asked, "With web search" first, tools as rows, engines as columns with each engine's successful answers in its header, and an "All engines" column within that mode. At most 15 rows; engines with no or partial answers are noted under the table. |
| `leaderboard.csv` | `mode, engine, tool, first_count`. |
| `cited_domains.csv` | `mode, domain, citations, engines, sextant`, where `sextant` marks the tracked brand's domain. |
| `prompts.csv` | `key, prompt_text, experimental, mode, engine, answers, ok, sextant_named, sextant_first`. |
| `answers.jsonl` | One line per answer: the facts the edition is built from (`key, engine, mode, sample, run_date, status, tools_named, first_named, sextant_rank, citations`), then `model_version, latency_ms, cost_usd, error_type, error, search_count, search_queries` and the answer `text` unless `--no-text`. The text is your own vendor output and stays on your machine. |

The own-brand fields keep the names `sextant_named`, `sextant_first` and `sextant_ranks` so the file parses like ours; `producer.brand` says whose counts they are. Each run's `goal_mode` is `search` when ChatGPT, Claude, Gemini or Grok was asked with web search, else `api`. A CSV cell that a spreadsheet would read as a formula gets a leading `'`.

After the table the summary prints the run's status, the actual cost against the budget (answers and judge apart), the judged count and `judge_ungrounded`, the status counts per engine and mode, and, when the exit is not 0, a line saying why. A two-engine example, showing the shape of the lines rather than a measured run:

```
Run partial: 58 of 60 answers succeeded, 2 failed, 0 skipped.
Actual cost $8.62 of the $20.00 budget (answers $8.55, judge $0.07).
Judge: 56 of 58 answers judged; judge_ungrounded 3 (names the judge listed that the answer does not contain, dropped).
Answers per engine and mode:
  Claude, search: success 30
  ChatGPT, search: success 28, error 2
Exit 3: the run is partial (some question, engine and mode has no answer); the judge could not list the tools of 2 answers, so the table is partial.
```

## Compare

`--compare` on `run`, or `aeo-index compare RUN_DIR`, sets your first-named table beside an edition:

- `latest` is the newest dated folder in the repository's `data/`, read through the GitHub contents API (unauthenticated, so GitHub allows 60 requests an hour from one address);
- a date reads that edition's `index.json` from the repository;
- anything else is a local `index.json`, or a folder holding one.

Every edition is checked before use, and one that does not parse is refused with the reason. A published edition is labelled "Sextant edition <date>", a local one "Edition <date>".

Against an edition with a `modes` block, the tables are set side by side per mode, search first. Against an older edition without one, the two top levels are compared and both bases are stated; when they count different rows, the header says so, and a run with `--mode both` has the same rows. The rows are each side's ten tools named first most often. A share is a side's first-named count over its own successful answers, so runs of different sizes read together. Engines with no answers are named. The header also names each side's engines under review, whose answers are counted in that side's shares, and says so when the two sides read names with different extraction versions (an edition without a `producer` block is read as version 1, which did not check names against the answer). When the two sides asked different questions, the header says the tables answer different questions and only the shares are shown.

When the comparison fails after a run, the run's files and its exit code stand, and stderr says why.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Every asked question, engine and mode has a successful answer, and the judge listed the tools of every one. |
| 2 | Refused before asking: bad flags, a bad question file, a branded question, no engine configured, the judge's key missing, a budget below the high figure, an output folder in use, or a "no" at the prompt. `compare` and `rebuild` also exit 2 on input they cannot read. |
| 3 | The run is partial or failed, the judge could not list the tools of some answers, or the files could not be written (the table is still printed, and the summary goes to stderr). |
| 4 | The budget stopped the asking or the judging. This takes precedence over 3. |
| 130 | Ctrl-C. The line printed names the files written so far, if any, and the money spent on how many paid calls. At the `[y/N]` prompt nothing was asked and nothing was spent. |

When the exit is not 0, the summary's last line says why, for example "Exit 3: the run is partial (some question, engine and mode has no answer)."

## What it never does

- No telemetry, no analytics, no update check. Your runs never reach us.
- No network call except to the engines' APIs (api.openai.com, api.anthropic.com, generativelanguage.googleapis.com, api.x.ai, api.perplexity.ai, api.dataforseo.com) and, only for a comparison, api.github.com and raw.githubusercontent.com. Redirects are not followed.
- No key in any output. Every error string is redacted before it is stored or printed: each configured key value, the vendors' key shapes and `Bearer` and `Basic` credentials. The Gemini key travels in a header, never in a URL, and request URLs are not logged.
- No file is loaded unless you name it.

## Licence

MIT, copyright 2026 Shearwater Labs (`LICENSE`). The index data in the repository's `data/` is CC BY 4.0 (`LICENSE-DATA` at the repository root). The files in its `media/` are © Shearwater Labs, all rights reserved, and may be reproduced unaltered when citing or linking the index.
