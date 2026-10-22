# Sextant AI Visibility Index

Which AEO tools do the AI answer engines recommend when nobody names a company?

Every day Sextant AEO (https://sextantaeo.com) asks six AI engines (ChatGPT, Claude, Gemini, Grok, Perplexity, Google AI Overviews) ten fixed, unbranded questions such as "Which AEO tools should I use?", three samples per engine per prompt, and records which tool each answer names first, every tool it names, and every page it cites. This repository publishes the dataset weekly, including the weeks in which Sextant itself scores zero.

Live page and method: https://sextantaeo.com/research/ai-engines-aeo-tools

## Run it yourself

The code that runs the index method is in `runner/`, published as the Python package `aeo-index`. It needs Python 3.12 or newer and your own API keys, and two keys are enough to start.

```bash
uv tool install aeo-index      # or: pipx install aeo-index
export ANTHROPIC_API_KEY=...   # Claude, and the default judge
export OPENAI_API_KEY=...      # ChatGPT
aeo-index run --compare
```

The run prints its plan and the cost before the first call, then waits for `y`. With these two keys it asks Claude and ChatGPT the ten questions three times each, with web search on. It prints the first-named table, then the same table beside our latest edition. A key for Gemini, Grok or Perplexity, or a DataForSEO login, adds that engine. `aeo-index run --dry-run` prints the plan and its cost and calls nothing. The flags, the question file, the outputs and the exit codes are in `runner/README.md`.

## What the runner does

1. It asks every engine you have a key for the ten questions in `data/` (or a question file for your own category), three samples each, with web search unless you choose `--mode api`.
2. Before the first call it prints a typical and a high cost; `--budget` (20.00 USD by default) caps the asking and the judging together.
3. A second model, the judge, lists the tools each answer recommends, in the order the answer gives them.
4. A name the judge lists counts only when the answer's own text contains it.
5. It counts the tool each answer names first, per engine and per mode, and writes the four files this repository publishes for each edition (`index.json`, `leaderboard.csv`, `cited_domains.csv`, `prompts.csv`), plus `table.md` and `answers.jsonl`.
6. With `--compare` it prints your table beside our latest edition, each side as a share of its own answers.

## Data

`data/YYYY-MM-DD/` holds one edition: `index.json` (full snapshot: runs, per-engine totals, prompts, leaderboard, leaderboard by engine, top cited domains and URLs), `leaderboard.csv`, `cited_domains.csv`, `prompts.csv`.

Edition 2026-10-18: 720 answers (14 to 17 October 2026). Profound named first in 233. Sextant AEO named in 0. Google AI Overviews was the only engine returning citations in this edition (977); Gemini's answers were truncated by our collector in this period and its counts are under review.

## Method, briefly

- Ten prompts, fixed and versioned (`prompts.csv`), asked without any brand in the question.
- Six engines through their APIs; three samples per prompt per engine per day.
- "Named first": the first AEO tool a tool-extraction model lists for that answer. Spelling variants are grouped by vendor where stated.
- Citations: the pages each answer cited, when the engine returns them.
- We publish our own result unedited.

## Badges

Tools on the index may display their rank: `badges/<tool-slug>-<edition>.svg` (coming with the next edition) and the embed line on https://sextantaeo.com/research/ai-engines-aeo-tools.

## Licences

The code in `runner/` is MIT licensed (`LICENSE`, also shipped as `runner/LICENSE`, copyright 2026 Shearwater Labs). The data in `data/` is CC BY 4.0 (`LICENSE-DATA`). The files in `media/` are © Shearwater Labs, all rights reserved; they may be reproduced unaltered when citing or linking the index.

## Cite

Cite as: Shearwater Labs, "Sextant AI Visibility Index", edition YYYY-MM-DD, https://github.com/mykhailo-bespinite/sextant-ai-visibility-index (see `CITATION.cff`).
