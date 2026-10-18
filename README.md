# Sextant AI Visibility Index

Which AEO tools do the AI answer engines recommend when nobody names a company?

Every day Sextant AEO (https://sextantaeo.com) asks six AI engines (ChatGPT, Claude, Gemini, Grok, Perplexity, Google AI Overviews) ten fixed, unbranded questions such as "Which AEO tools should I use?", three samples per engine per prompt, and records which tool each answer names first, every tool it names, and every page it cites. This repository publishes the dataset weekly, including the weeks in which Sextant itself scores zero.

Live page and method: https://sextantaeo.com/research/ai-engines-aeo-tools

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

## Licence and citation

Data: CC BY 4.0. Cite as: Shearwater Labs, "Sextant AI Visibility Index", edition YYYY-MM-DD, https://github.com/mykhailo-bespinite/sextant-ai-visibility-index (see `CITATION.cff`).
