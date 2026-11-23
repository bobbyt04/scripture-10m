// Build step: read the real artefacts from the repository and write src/generated/repo.json, which the site
// imports. Runs before dev, build, typecheck and test (see package.json). Nothing here is typed in by hand:
// numbers come from the eval JSON, the training logs and the data stats; code comes from scripture/model.py.
//
//   node scripts/extract.ts

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Searcher, type SearchData } from "../src/search/searcher.ts";
import {
  adviceHistory, annotate, extractBlock, highlightPython, paramBreakdown, parseKjvErrors, parseLosses, summariseEvals,
  tokenizeExample, type EvalJson, type RepoData, type Stats, parseSample,
} from "./repo-data.ts";

const WEB = join(dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = join(WEB, "..");
const read = (p: string) => readFileSync(join(ROOT, p), "utf8");
const readJson = <T>(p: string) => JSON.parse(read(p)) as T;

const MODEL_FILE = "scripture/model.py";
const NOTES = [
  { match: "self.qkv = nn.Linear", note: "One matrix makes the query, key and value for all six heads at once: 384 numbers in, 1,152 out." },
  { match: "mask = torch.tril", note: "A lower-triangular mask: each position may look at itself and earlier positions, never later ones." },
  { match: "q, k, v = self.qkv(x).split", note: "Split into queries (what each position looks for), keys (what it offers) and values (what it passes on)." },
  { match: "att = (q @ k.transpose", note: "Every query is scored against every key. Dividing by the square root of the head size (64) keeps the scores in a range softmax handles well." },
  { match: "att = att.masked_fill", note: "Scores for future positions become minus infinity, so after softmax they get exactly zero weight." },
  { match: "y = att @ v", note: "Each position takes a weighted average of the values it attends to." },
  { match: "y = y.transpose(1, 2)", note: "The six heads' outputs are put side by side again, then mixed by one more linear layer." },
];
const EXAMPLE = { book: "quran" as const, ref: "94:5" };

function verseText(book: string, ref: string): string {
  for (const line of read(`data/processed/${book}.jsonl`).split("\n")) {
    if (!line) continue;
    const v = JSON.parse(line) as { ref: string; text: string };
    if (v.ref === ref) return v.text;
  }
  throw new Error(`${book} ${ref} not found`);
}

function gitHead(): string | null {
  try {
    return execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: ROOT, encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

export function build(): RepoData {
  const evalJson = readJson<EvalJson & { git: { commit: string } }>("evals/results/model.json");
  const bookJson = readJson<EvalJson>("evals/results/book_sampling.json");
  const verse = parseLosses(read("runs/verse/losses.jsonl"));
  const book = parseLosses(read("runs/book/losses.jsonl"));
  const vocab = JSON.parse(readFileSync(join(WEB, "src/model/vocab.json"), "utf8")) as string[];

  const src = read(MODEL_FILE);
  const block = extractBlock(src, "class CausalSelfAttention", "class MLP");
  const parts = paramBreakdown(evalJson.config);
  const total = parts.reduce((a, p) => a + p.count, 0);
  if (total !== evalJson.params) throw new Error(`parameter breakdown ${total} != model.json params ${evalJson.params}`);

  const best = verse.reduce((a, r) => (r.val.overall < a.val.overall ? r : a));
  const last = book[book.length - 1];
  const bookLoss = (name: string) => {
    const m = bookJson.loss.models.find((x) => x.name === name);
    if (!m) throw new Error(`book_sampling.json: no ${name} row`);
    return m.gita.nats;
  };
  const text = verseText(EXAMPLE.book, EXAMPLE.ref);
  const search = siteSearchData();

  return {
    stats: readJson<Stats>("data/processed/stats.json"),
    kjvErrors: parseKjvErrors(read("data/SOURCES.md")),
    samples: Object.fromEntries((["gita", "quran", "bible"] as const).map((b) => [b, parseSample(read(`samples/${b}.md`), `samples/${b}.md`)])) as RepoData["samples"],
    vocab,
    example: { ...EXAMPLE, text, tokens: tokenizeExample(vocab, EXAMPLE.book, text) },
    code: { file: MODEL_FILE, firstLine: block.firstLine, lines: highlightPython(block.lines), notes: annotate(block.lines, block.firstLine, NOTES) },
    params: { total, parts, config: evalJson.config },
    training: { verse, book, trainArgs: evalJson.train_args, bestStep: best.step },
    evals: summariseEvals(evalJson),
    bookRun: {
      step: last.step,
      gitaCopied: bookJson.memorisation.gita.generated.copied_from_any,
      gitaLongestRun: bookJson.memorisation.gita.generated.longest_run,
      gitaLoss: bookLoss("Scripture-10M"),
      bigramGitaLoss: bookLoss("2-gram"),
    },
    evalCommit: evalJson.git.commit.slice(0, 7),
    advice: adviceHistory(readJson("evals/search_probes.json"), readJson("evals/search_cases.json"),
      readJson("evals/results/search.json"), search as never, new Searcher(search as unknown as SearchData).topics().length,
      JSON.parse(readFileSync(join(WEB, "src/story/advice-history.json"), "utf8"))),
  };
}

/**
 * The verse-topic search data for the browser: search/artifacts/search.json without what only the eval-history
 * search() reads (each verse's BM25 terms, the bridge, the query stopwords) and without the build inputs.
 * src/search/searcher.ts runs on exactly this file, and so do its tests.
 */
export function siteSearchData(): Record<string, unknown> {
  const full = readJson<Record<string, any>>("search/artifacts/search.json");
  if (full.version !== 1) throw new Error(`search.json version ${full.version}, expected 1`);
  const { query_stopwords: _q, ...lexicon } = full.lexicon;
  const pool = Object.fromEntries(Object.entries(full.pool as Record<string, Record<string, unknown>[]>)
    .map(([book, verses]) => [book, verses.map(({ terms: _t, ...v }) => v)]));
  return { version: full.version, params: full.params, lexicon, vocabulary: full.vocabulary, themes: full.themes, safety: full.safety, pool };
}

function write(name: string, data: unknown): void {
  const out = join(WEB, "src/generated", name);
  mkdirSync(dirname(out), { recursive: true });
  const text = JSON.stringify(data) + "\n";
  writeFileSync(out, text);
  console.log(`wrote src/generated/${name} (${(Buffer.byteLength(text) / 1024).toFixed(1)} kB)`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  write("repo.json", build());
  write("search.json", siteSearchData());
  write("similarities.json", readJson("search/artifacts/similarities.json"));
  console.log(`repo at ${gitHead() ?? "unknown commit"}`);
}
