# Sextant AI Visibility Index

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23645499.svg)](https://doi.org/10.5281/zenodo.23645499)

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

`data/YYYY-MM-DD/` holds one edition: `index.json` (full snapshot: runs, per-engine totals, prompts, leaderboard, leaderboard by engine, top cited domains and URLs), `leaderboard.csv`, `cited_domains.csv`, `prompts.csv`. From edition 2026-10-25 it also holds `answers.jsonl`: one line per answer with the prompt key, engine, mode, sample, run date, status, the tools named in order, the first-named tool, Sextant AEO's rank when named, and the cited URLs. Answer text is not published.

Edition 2026-10-25: 1,800 answers (20 to 25 October 2026; the 19 October run was lost to a deploy and holds no answers). With web search on, 1,080 answers from six engines: Profound named first in 287; Sextant AEO named in 24 and first in 8 (ChatGPT 6, Gemini 2; Grok named it in 14 answers, never first). From training knowledge, no web search, 720 answers from four engines: Profound named first in 446; Sextant AEO named in 0. The answers cited 9,149 pages; the most cited domains and URLs are in `cited_domains.csv` and `index.json`. The Gemini collector fix of 18 October is in force and no engine is under review.

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

Cite as: Shearwater Labs, "Sextant AI Visibility Index", edition YYYY-MM-DD, https://doi.org/10.5281/zenodo.23645499 (see `CITATION.cff`). The concept DOI 10.5281/zenodo.23645499 always resolves to the latest edition; each edition has its own DOI on Zenodo (2026-10-18: 10.5281/zenodo.23645500; 2026-10-25: 10.5281/zenodo.23668261).
