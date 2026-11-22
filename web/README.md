# Scripture-10M website

A single static page: the verse-topic search, the parallels, the trained model generating text in the
browser, and the story of how it was built. Vite and TypeScript, no UI framework, no ML library, no analytics.
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
`src/story/process.json` (test counts per pull request, copied from the PR descriptions) and
`src/story/advice-history.json` (a placeholder chapter) are the only hand-written data.

## The model in the browser

`src/model/` is a hand-written forward pass of `scripture/model.py`: token and position embeddings,
pre-LayerNorm blocks with causal multi-head attention and an exact-GELU MLP, a key-value cache, and
temperature sampling. It runs in a Web Worker (`src/generate/worker.ts`) and streams characters to the page.
The weights (`public/model/*.bin`) are fetched only when the visitor presses Generate.

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
characters, non-ASCII). Measured speed: about 50 characters per second in Node on an Apple M2, and about 30 in
headless Chrome while other jobs shared the machine.

## Waiting for PR 4

The verse search and the parallels come from PR 4 (`search/`). Until it merges, both run on clearly marked
mock data with real verse texts (a test checks every one against `data/processed`):

- `src/advice/api.ts` is the contract the UI uses: `suggest(text)` returns a crisis note, up to four topics, or
  nothing; `topic(id)` returns verses per book, where an empty list means the book has nothing for that topic.
  Swap `mockAdviceApi` for the port of `search/FORMAT.md` there; the UI does not change. While the mock is in use
  the page says "Preview data".
- `src/similarities/api.ts` follows the shape of `search/artifacts/similarities.json`.
- `src/story/advice-history.json` holds the placeholder text of chapter 08.

## Deploying on Vercel

Import the repository and set **Root Directory** to `web`. `vercel.json` sets the Vite preset, `npm run build`
and the `dist` output, and long-lived caching for the hashed assets and model file. Keep "Include files outside
the root directory in the Build Step" on (the default), because the build reads `../evals`, `../runs`,
`../data` and `../scripture`. No server code, environment variables or paid features are needed.
