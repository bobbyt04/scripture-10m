# Scripture-10M website

A single plain static page: one prompt box with two modes (find verses by topic, or let the trained model write in a
book's style, in the browser), the story of how it was built, and the parallels. Vite and TypeScript, no UI framework, no ML library, no analytics.
Fonts (Inter, Source Serif 4) are self-hosted from npm packages.

```bash
cd web
npm install
npm run dev        # http://localhost:5300
npm test           # unit tests (Vitest)
npm run typecheck
npm run build      # static site in web/dist
npm run preview    # serve the build on http://localhost:5301
npm run e2e        # Playwright smoke tests against the build (uses local Google Chrome; CI installs Chromium)
```

Every script first runs `node scripts/extract.ts` (Node 22.18 or later, for TypeScript type stripping).

## Where the numbers come from

`scripts/extract.ts` reads the repository at build time and writes `src/generated/repo.json` (not committed):

- `evals/results/model.json` and `evals/results/book_sampling.json`: every eval number on the page;
- `runs/verse/losses.jsonl` and `runs/book/losses.jsonl`: the training charts;
- `data/processed/stats.json` and `data/SOURCES.md`: verse counts and the Gutenberg errata;
- `scripture/model.py`: the attention excerpt, highlighted by a small built-in highlighter. Each annotation
  is matched to a line of code, and the build fails if the code changes under it.

The parameter breakdown is computed from the model config and must add up to `params` in `model.json`.
`src/story/advice-history.json` (the "What failed" numbers recorded only in the README, with sources) is the only
manually written data; `src/story/test-counts.json` is written by a script (below).

## The model in the browser

`src/model/` is a from-scratch TypeScript forward pass of `scripture/model.py`: token and position embeddings,
pre-LayerNorm blocks with causal multi-head attention and an exact-GELU MLP, a key-value cache, and
temperature sampling. It runs in a Web Worker (`src/generate/worker.ts`) and streams characters to the page.
The weights (`public/model/*.bin`) are fetched only when the visitor switches to "Let the model write" and
presses Write. The model continues from the book's tag followed by whatever the visitor typed
(`src/generate/start.ts`: characters outside the 67-token vocabulary are dropped and counted, line breaks become
spaces, and the start is capped at 120 characters so it fits well inside the 256-character window); an empty box
starts from the tag. Before the model writes, the typed text goes through the search's own `suggest()`, and a
crisis result shows the same note instead of any generated text. A "temperature" slider sets the sampling
temperature (0.2 to 1.5, default 0.8) for the next run. While it writes, the page shows the five most likely next
characters with their probabilities and marks the one sampled: `topChoices()` in `src/model/generate.ts` applies
the same temperature softmax that `sampleToken()` draws from, to the last step of each ~24 ms slice the worker
posts, and the bars are redrawn at most every 120 ms (600 ms with reduced motion).

The model has 256 learned positions. PyTorch's `generate` crops to the newest 256 tokens and recomputes every
step; the browser instead rebuilds its cache from the newest 128 tokens when the window fills. The two are
identical for the first 256 tokens.

### Export

```bash
# from the repository root, with the Python environment that trains the model
python web/scripts/export_model.py --ckpt out/best.pt            # add --device mps to measure faster
```

This reads the checkpoint (never writes it) and writes the model binary, `src/model/manifest.json`,
`src/model/vocab.json` and the test fixtures in `test/fixtures/`. The binary layout is documented at the
top of the script. Running it twice gives identical files.

Two formats were measured on all held-out verses with the eval harness's own scorer (`evals/run.py
model_logprobs`), weights dequantised back into the PyTorch model:

| Format | Size | Mean validation loss (nats) | Change |
|---|---|---|---|
| float32 checkpoint | 43.2 MB of weights | 0.90251 | |
| float16 | 21.6 MB | 0.90252 | +0.00001 |
| int8, one scale per tensor (LayerNorm and biases float32) | 10.9 MB | 0.90298 | +0.00047 |

The int8 file is published: half the download, for a loss change of at most 0.0006 nats on any book, about
3% of the width of the narrowest 95% interval in the eval (the Bible's, 0.020 nats). The exact numbers are in `src/model/manifest.json`.

### Correctness

`test/model.test.ts` runs the TypeScript forward pass on the published file against PyTorch logits computed
by the export script from exactly the same (dequantised) weights:

- logits agree within 1e-4 at every checked position, over prompts in all three books and a full 256-token
  window (the measured worst difference is about 1.3e-5);
- greedy generation matches PyTorch token for token for 160 tokens after each book's tag.

`test/tokenizer.test.ts` checks the tokenizer against Python's ids for tricky strings (partial tags, unknown
characters, non-ASCII). Measured speed (the figure the page reports after each run): about 80 characters per
second or more in Chrome on an Apple M2 that is otherwise idle, and about 30 in headless Chrome while other jobs
shared the machine. Older phones will be slower.

## The verse-topic search in the browser

`src/search/searcher.ts` is a plain TypeScript port of the site contract of `scripture/search.py`, following
`search/FORMAT.md` steps 0 to 5, 8 and 9: `suggest(text)` (a crisis note only; gentle topics for a sensitive
message; up to four topics; or none), `topic(id, {sensitive})` and `topics()`. Typing never selects a verse: when `suggest()` returns topics
and the first has `auto_show` (an allowlist in `search/themes.json`), the page opens it (exactly as tapping it would,
with its `sensitive` flag) and lists the others; otherwise it lists the topics and waits for a tap
(`src/advice/state.ts`).

`scripts/extract.ts` writes its data at build time: `src/generated/search.json` is
`search/artifacts/search.json` without what only the eval-history `search()` reads (BM25 terms, the bridge,
query stopwords): 385 kB, 104 kB gzipped. It is a separate chunk, fetched when the visitor first focuses the
text box (or asks for the topic list), never with the page. `src/generated/similarities.json` (23 kB, 8 kB
gzipped) is the reviewed parallels file, loaded right after the page.

`test/search.test.ts` checks the port against every `words`, `suggest`, `topic` and `topics` case of
`search/artifacts/fixtures.json`, then runs `scripts/search_parity.py` with Python (`PYTHON`, default
`python3`; no packages needed) and requires identical JSON from `suggest()` (with its crisis words, signal and
comfort cue) for about 4,700 messages: every stored probe and situation, theme and safety phrases with filler,
and seeded variants with odd Unicode, digits, emoji, typos and text past the 500-character cap; plus
`topics()` and `topic()` for every theme, sensitive or not.

The "What failed" chapter reads the review numbers from `evals/search_probes.json`, `evals/search_cases.json` and
`evals/results/search.json`; the few that exist only in the README are in `src/story/advice-history.json`,
each with its source. The test totals on the page are in `src/story/test-counts.json`, written by
`node scripts/count_tests.ts` (set `PYTHON` to an interpreter with PyTorch; under Rosetta use
`PYTHON="arch -arm64 ../.venv/bin/python"`).

## Deploying on Vercel

Import the repository and set **Root Directory** to `web`. `vercel.json` sets the Vite preset, `npm run build`
and the `dist` output, and long-lived caching for the hashed assets and model file. Keep "Include files outside
the root directory in the Build Step" on (the default), because the build reads `../evals`, `../runs`,
`../data`, `../scripture` and `../search`. No server code, environment variables or paid features are needed.
