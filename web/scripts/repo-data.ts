// Pure functions that turn files from the repository into the data the site's build story shows.
// scripts/extract.ts does the reading and writing; these are unit-tested in test/repo-data.test.ts.

import { CharTokenizer } from "../src/model/tokenizer.ts";

export type Book = "gita" | "quran" | "bible";
export const BOOK_IDS: Book[] = ["gita", "quran", "bible"];
type PerBook<T> = Record<Book, T>;

// ------------------------------------------------------------------ training logs

export interface LossRow {
  step: number;
  train: PerBook<number> & { overall: number };
  val: PerBook<number> & { overall: number };
  elapsed_s: number;
}

export function parseLosses(jsonl: string): LossRow[] {
  const rows = jsonl.split("\n").filter((l) => l.trim()).map((l, i) => {
    const r = JSON.parse(l) as LossRow;
    for (const split of ["train", "val"] as const) {
      for (const b of [...BOOK_IDS, "overall"] as const) {
        if (typeof r[split]?.[b] !== "number") throw new Error(`losses line ${i + 1}: missing ${split}.${b}`);
      }
    }
    if (typeof r.step !== "number") throw new Error(`losses line ${i + 1}: missing step`);
    return { step: r.step, train: r.train, val: r.val, elapsed_s: r.elapsed_s };
  });
  for (let i = 1; i < rows.length; i++) {
    if (rows[i].step <= rows[i - 1].step) throw new Error("losses are not in step order");
  }
  return rows;
}

// ------------------------------------------------------------------ code excerpts

export type Segment = [cls: "" | "k" | "s" | "c" | "n" | "f" | "b" | "d", text: string];

/** Lines from the one starting with `start` up to (not including) the one starting with `end`, trailing blanks cut. */
export function extractBlock(source: string, start: string, end: string): { firstLine: number; lines: string[] } {
  const all = source.split("\n");
  const i = all.findIndex((l) => l.startsWith(start));
  if (i < 0) throw new Error(`excerpt start not found: ${start}`);
  let j = all.findIndex((l, k) => k > i && l.startsWith(end));
  if (j < 0) j = all.length;
  const lines = all.slice(i, j);
  while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
  return { firstLine: i + 1, lines };
}

const KEYWORDS = new Set(("and as assert break class continue def del elif else except finally for from global if import in is " +
  "lambda nonlocal not or pass raise return try while with yield None True False").split(" "));
const BUILTINS = new Set("self super float int len range print isinstance sum min max list dict tuple set".split(" "));

/** A small Python highlighter: keywords, strings (with docstrings across lines), comments, numbers, names after def/class. */
export function highlightPython(lines: string[]): Segment[][] {
  const out: Segment[][] = [];
  let inDoc: string | null = null; // the open triple quote, if inside one
  for (const line of lines) {
    const segs: Segment[] = [];
    const push = (cls: Segment[0], text: string) => {
      if (!text) return;
      const last = segs[segs.length - 1];
      if (last && last[0] === cls) last[1] += text;
      else segs.push([cls, text]);
    };
    let i = 0;
    let prevWord = "";
    while (i < line.length) {
      if (inDoc) {
        const close = line.indexOf(inDoc, i);
        if (close < 0) { push("s", line.slice(i)); i = line.length; break; }
        push("s", line.slice(i, close + 3));
        i = close + 3;
        inDoc = null;
        continue;
      }
      const rest = line.slice(i);
      let m: RegExpMatchArray | null;
      if ((m = rest.match(/^("""|''')/))) { inDoc = m[1]; push("s", m[1]); i += 3; continue; }
      if (rest[0] === "#") { push("c", rest); break; }
      if ((m = rest.match(/^[rbfu]?("([^"\\]|\\.)*"|'([^'\\]|\\.)*')/i))) { push("s", m[0]); i += m[0].length; continue; }
      if ((m = rest.match(/^@[\w.]+/)) && !line.slice(0, i).trim()) { push("d", m[0]); i += m[0].length; continue; }
      if ((m = rest.match(/^\d[\d_]*(\.\d+)?(e-?\d+)?/i))) { push("n", m[0]); i += m[0].length; continue; }
      if ((m = rest.match(/^[A-Za-z_]\w*/))) {
        const w = m[0];
        push(prevWord === "def" || prevWord === "class" ? "f" : KEYWORDS.has(w) ? "k" : BUILTINS.has(w) ? "b" : "", w);
        prevWord = w;
        i += w.length;
        continue;
      }
      push("", rest[0]);
      if (rest[0] !== " ") prevWord = "";
      i += 1;
    }
    out.push(segs);
  }
  return out;
}

export interface Note {
  match: string;
  note: string;
}

/** Attach each note to the one excerpt line containing its `match`; fails loudly if the code has moved on. */
export function annotate(lines: string[], firstLine: number, notes: Note[]): { line: number; note: string }[] {
  return notes.map(({ match, note }) => {
    const hits = lines.flatMap((l, k) => (l.includes(match) ? [k] : []));
    if (hits.length !== 1) throw new Error(`annotation "${match}" matches ${hits.length} lines of the excerpt`);
    return { line: firstLine + hits[0], note };
  });
}

// ------------------------------------------------------------------ parameters

export interface ModelConfig {
  vocab_size: number;
  block_size: number;
  n_layer: number;
  n_head: number;
  n_embd: number;
}

export interface ParamPart {
  id: string;
  label: string;
  detail: string;
  count: number;
}

/** Parameter count by part, from the shapes in scripture/model.py. */
export function paramBreakdown(c: ModelConfig): ParamPart[] {
  const C = c.n_embd, L = c.n_layer;
  const ln = 2 * C;
  const attn = C * 3 * C + (C * C + C);
  const mlp = (C * 4 * C + 4 * C) + (4 * C * C + C);
  return [
    { id: "tok", label: "Token embeddings", detail: `${c.vocab_size} tokens × ${C}`, count: c.vocab_size * C },
    { id: "pos", label: "Position embeddings", detail: `${c.block_size} positions × ${C}`, count: c.block_size * C },
    { id: "attn", label: "Attention", detail: `${L} layers × (q, k, v and output projections)`, count: L * attn },
    { id: "mlp", label: "Feed-forward (MLP)", detail: `${L} layers × (${C} → ${4 * C} → ${C})`, count: L * mlp },
    { id: "ln", label: "Layer norms", detail: `${2 * L + 1} × (scale and shift)`, count: (2 * L + 1) * ln },
    { id: "head", label: "Output head", detail: `${C} → ${c.vocab_size} next-token scores`, count: C * c.vocab_size },
  ];
}

// ------------------------------------------------------------------ data sources

/** The Gutenberg #10 wording errors listed in data/SOURCES.md, as {wrong, right, ref}. */
export function parseKjvErrors(sourcesMd: string): { text: string }[] {
  const start = sourcesMd.indexOf("**Why not Project Gutenberg #10:**");
  if (start < 0) throw new Error("SOURCES.md: Gutenberg section not found");
  const after = sourcesMd.slice(start).split("\n").slice(1);
  const items: { text: string }[] = [];
  for (const l of after) {
    const m = l.match(/^\s{2,}- (.+)$/);
    if (m) items.push({ text: m[1].trim() });
    else if (items.length && l.trim()) break;
  }
  if (items.length < 3) throw new Error("SOURCES.md: expected a list of Gutenberg errors");
  return items;
}

// ------------------------------------------------------------------ tokenizer example

export function tokenizeExample(tokens: string[], book: Book, text: string): { token: string; id: number }[] {
  const tok = new CharTokenizer(tokens);
  return tok.encode(`<${book}>${text}\n`).map((id) => ({ token: tokens[id], id }));
}

// ------------------------------------------------------------------ evals

interface LossStat { nats: number; bpc: number; ci95_nats?: [number, number] }
interface EvalModelRow extends PerBook<LossStat> { name: string; overall: LossStat }
interface CopyStat { copied_from_any: number; longest_run: number; longest_run_source: string; texts: number }
interface StyleCell { n: number; counts: PerBook<number>; own_book: number }

export interface EvalJson {
  step: number;
  params: number;
  config: ModelConfig;
  train_args: Record<string, unknown>;
  eval_args: { samples: number; chars: number; temperature: number; window: number };
  loss: { models: EvalModelRow[]; best_baseline: PerBook<string> & { overall: string }; duplicates: PerBook<{ val_verses: number }> };
  memorisation: PerBook<{ generated: CopyStat; heldout_real: CopyStat }>;
  style: { rows: Record<string, PerBook<StyleCell>>; uniform_noise: { n: number; counts: PerBook<number> }; verdict: string };
  wellformed: PerBook<{ real_word_frac: number; mean_verse_chars: number; real_mean_verse_chars: number }>;
}

export interface EvalSummary {
  step: number;
  params: number;
  loss: PerBook<{ model: LossStat; baseline: LossStat & { name: string }; valVerses: number }>;
  memorisation: PerBook<{ generated: CopyStat; heldout: CopyStat }>;
  style: { rows: { label: string; cells: PerBook<StyleCell> }[]; noise: { n: number; counts: PerBook<number> }; verdict: string };
  words: PerBook<{ realWordFrac: number; meanVerseChars: number; realMeanVerseChars: number }>;
  samplesPerBook: number;
  sampleChars: number;
  temperature: number;
  window: number;
}

export function summariseEvals(e: EvalJson): EvalSummary {
  const byName = new Map(e.loss.models.map((m) => [m.name, m]));
  const model = byName.get("Scripture-10M");
  if (!model) throw new Error("model.json: no Scripture-10M row");
  function per<T>(f: (b: Book) => T): PerBook<T> {
    return { gita: f("gita"), quran: f("quran"), bible: f("bible") };
  }
  return {
    step: e.step,
    params: e.params,
    loss: per((b) => {
      const name = e.loss.best_baseline[b];
      const base = byName.get(name);
      if (!base) throw new Error(`model.json: baseline ${name} missing`);
      return { model: model[b], baseline: { name, ...base[b] }, valVerses: e.loss.duplicates[b].val_verses };
    }),
    memorisation: per((b) => ({ generated: e.memorisation[b].generated, heldout: e.memorisation[b].heldout_real })),
    style: {
      rows: Object.entries(e.style.rows).map(([label, cells]) => ({ label, cells })),
      noise: e.style.uniform_noise,
      verdict: e.style.verdict,
    },
    words: per((b) => ({
      realWordFrac: e.wellformed[b].real_word_frac,
      meanVerseChars: e.wellformed[b].mean_verse_chars,
      realMeanVerseChars: e.wellformed[b].real_mean_verse_chars,
    })),
    samplesPerBook: e.eval_args.samples,
    sampleChars: e.eval_args.chars,
    temperature: e.eval_args.temperature,
    window: e.eval_args.window,
  };
}

// ------------------------------------------------------------------ everything the site needs

export interface Stats extends PerBook<{ verses: number; characters: number; chapters: number }> {
  charset: string;
}

export interface RepoData {
  stats: Stats;
  kjvErrors: { text: string }[];
  vocab: string[];
  example: { book: Book; ref: string; text: string; tokens: { token: string; id: number }[] };
  code: { file: string; firstLine: number; lines: Segment[][]; notes: { line: number; note: string }[] };
  params: { total: number; parts: ParamPart[]; config: ModelConfig };
  training: {
    verse: LossRow[];
    book: LossRow[];
    trainArgs: Record<string, unknown>;
    bestStep: number;
  };
  evals: EvalSummary;
  bookRun: { step: number; gitaCopied: number; gitaLongestRun: number; gitaLoss: number; bigramGitaLoss: number };
  evalCommit: string;
}
