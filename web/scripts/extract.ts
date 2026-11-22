// Build step: read the real artefacts from the repository and write src/generated/repo.json, which the site
// imports. Runs before dev, build, typecheck and test (see package.json). Nothing here is typed in by hand:
// numbers come from the eval JSON, the training logs and the data stats; code comes from scripture/model.py.
//
//   node scripts/extract.ts

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  annotate, extractBlock, highlightPython, paramBreakdown, parseKjvErrors, parseLosses, summariseEvals,
  tokenizeExample, type EvalJson, type RepoData, type Stats,
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

  return {
    stats: readJson<Stats>("data/processed/stats.json"),
    kjvErrors: parseKjvErrors(read("data/SOURCES.md")),
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
  };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const data = build();
  const out = join(WEB, "src/generated/repo.json");
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(data) + "\n");
  console.log(`wrote src/generated/repo.json (${(JSON.stringify(data).length / 1024).toFixed(1)} kB, repo at ${gitHead() ?? "unknown commit"})`);
}
