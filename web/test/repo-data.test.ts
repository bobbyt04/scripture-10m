import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";
import { build, siteSearchData } from "../scripts/extract.ts";
import {
  annotate, extractBlock, highlightPython, paramBreakdown, parseKjvErrors, parseLosses, tokenizeExample,
} from "../scripts/repo-data.ts";
import searchSite from "../src/generated/search.json";
import similarities from "../src/generated/similarities.json";
import vocab from "../src/model/vocab.json";

const root = (p: string) => fileURLToPath(new URL(`../../${p}`, import.meta.url));
const read = (p: string) => readFileSync(root(p), "utf8");

describe("training log loader", () => {
  const line = (step: number, v = 1) =>
    JSON.stringify({ step, lr: 0.001, train: { gita: v, quran: v, bible: v, overall: v }, val: { gita: v, quran: v, bible: v, overall: v }, elapsed_s: step });

  test("parses rows in step order and skips blank lines", () => {
    const rows = parseLosses([line(0, 4), "", line(500, 2), line(1000, 1.5)].join("\n") + "\n");
    expect(rows.map((r) => r.step)).toEqual([0, 500, 1000]);
    expect(rows[1].val.gita).toBe(2);
  });

  test("rejects a row with a missing book and rows out of order", () => {
    expect(() => parseLosses(JSON.stringify({ step: 0, train: { gita: 1 }, val: {} }))).toThrow(/missing train.quran/);
    expect(() => parseLosses([line(500), line(0)].join("\n"))).toThrow(/step order/);
  });

  test("reads both real runs", () => {
    const verse = parseLosses(read("runs/verse/losses.jsonl"));
    const book = parseLosses(read("runs/book/losses.jsonl"));
    expect(verse[verse.length - 1].step).toBe(12000);
    expect(book[book.length - 1].step).toBe(8500);
    expect(book[book.length - 1].train.gita).toBeLessThan(0.1);
    expect(book[book.length - 1].val.gita).toBeGreaterThan(2.3);
  });
});

describe("code excerpts", () => {
  const src = 'x = 1\n\nclass A:\n    """Doc\n    more doc"""\n    def f(self):  # note\n        return "s" + 2\n\n\nclass B:\n    pass\n';

  test("cuts a block between markers and drops trailing blank lines", () => {
    const b = extractBlock(src, "class A", "class B");
    expect(b.firstLine).toBe(3);
    expect(b.lines).toHaveLength(5);
    expect(() => extractBlock(src, "class C", "class B")).toThrow(/not found/);
  });

  test("highlights keywords, names, docstrings across lines, comments, strings and numbers", () => {
    const hl = highlightPython(extractBlock(src, "class A", "class B").lines);
    const kinds = (i: number) => hl[i].filter((s) => s[0]).map((s) => `${s[0]}:${s[1]}`);
    expect(kinds(0)).toEqual(["k:class", "f:A"]);
    expect(kinds(1)).toEqual(['s:"""Doc']);
    expect(kinds(2)).toEqual(['s:    more doc"""']);
    expect(kinds(3)).toEqual(["k:def", "f:f", "b:self", "c:# note"]);
    expect(kinds(4)).toEqual(["k:return", 's:"s"', "n:2"]);
    expect(hl.map((l) => l.map((s) => s[1]).join(""))).toEqual(extractBlock(src, "class A", "class B").lines);
  });

  test("annotations must match exactly one line", () => {
    const lines = ["a = 1", "b = 2", "a = 3"];
    expect(annotate(lines, 10, [{ match: "b =", note: "n" }])).toEqual([{ line: 11, note: "n" }]);
    expect(() => annotate(lines, 10, [{ match: "a =", note: "n" }])).toThrow(/matches 2 lines/);
    expect(() => annotate(lines, 10, [{ match: "zzz", note: "n" }])).toThrow(/matches 0 lines/);
  });
});

describe("parameter breakdown", () => {
  test("adds up to the real model's 10,790,400 parameters", () => {
    const parts = paramBreakdown({ vocab_size: 67, block_size: 256, n_layer: 6, n_head: 6, n_embd: 384 });
    expect(parts.reduce((a, p) => a + p.count, 0)).toBe(10_790_400);
  });
});

describe("data sources", () => {
  test("lists the Gutenberg wording errors from SOURCES.md", () => {
    const errs = parseKjvErrors(read("data/SOURCES.md"));
    expect(errs.length).toBeGreaterThanOrEqual(5);
    expect(errs[0].text).toContain("Jonah 1:15");
  });

  test("tokenizes the example verse with its tag and newline", () => {
    const t = tokenizeExample(vocab, "quran", "But lo!");
    expect(t[0]).toEqual({ token: "<quran>", id: 1 });
    expect(t[t.length - 1]).toEqual({ token: "\n", id: 3 });
    expect(t).toHaveLength(9);
  });
});

describe("site data built from the repository", () => {
  const data = build();
  const model = JSON.parse(read("evals/results/model.json"));
  const book = JSON.parse(read("evals/results/book_sampling.json"));

  test("eval numbers are the eval JSON's, not copies", () => {
    expect(data.evals.loss.gita.model.nats).toBe(model.loss.models[0].gita.nats);
    expect(data.evals.loss.bible.baseline.name).toBe(model.loss.best_baseline.bible);
    expect(data.evals.style.rows[0].cells.gita.counts.gita).toBe(model.style.rows.generated.gita.counts.gita);
    expect(data.evals.words.quran.realWordFrac).toBe(model.wellformed.quran.real_word_frac);
    expect(data.bookRun.gitaCopied).toBe(book.memorisation.gita.generated.copied_from_any);
    expect(data.params.total).toBe(model.params);
  });

  test("the code excerpt is the attention class from scripture/model.py", () => {
    const text = data.code.lines.map((l) => l.map((s) => s[1]).join("")).join("\n");
    expect(text.startsWith("class CausalSelfAttention")).toBe(true);
    expect(read("scripture/model.py")).toContain(text);
    expect(data.code.notes.length).toBeGreaterThanOrEqual(5);
  });

  test("verse counts come from data/processed/stats.json", () => {
    expect(data.stats.gita.verses).toBe(701);
    expect(data.stats.quran.verses).toBe(6236);
    expect(data.stats.bible.verses).toBe(31102);
  });
});

describe("the data the browser search and the parallels load", () => {
  const verses = new Map<string, string>();
  for (const b of ["gita", "quran", "bible"]) {
    for (const l of read(`data/processed/${b}.jsonl`).split("\n")) {
      if (l) { const v = JSON.parse(l); verses.set(`${b} ${v.ref}`, v.text); }
    }
  }

  test("every parallel is the published file's, with its verse texts verbatim from data/processed", () => {
    const sims = JSON.parse(read("search/artifacts/similarities.json"));
    expect(similarities).toEqual(sims);
    const pairs = sims.themes.flatMap((t: { pairs: { verses: { book: string; ref: string; text: string }[] }[] }) => t.pairs);
    expect(pairs.length).toBe(sims.kept + sims.hand_listed_kept);
    for (const v of pairs.flatMap((p: { verses: { book: string; ref: string; text: string }[] }) => p.verses)) {
      expect(verses.get(`${v.book} ${v.ref}`), `${v.book} ${v.ref}`).toBe(v.text);
    }
  });

  test("the site's search data is search.json without the eval-only fields", () => {
    const full = JSON.parse(read("search/artifacts/search.json"));
    expect(siteSearchData()).toEqual(searchSite);
    expect(Object.keys(searchSite)).toEqual(["version", "params", "lexicon", "vocabulary", "themes", "safety", "pool"]);
    for (const key of ["params", "vocabulary", "themes", "safety"] as const) expect((searchSite as Record<string, unknown>)[key]).toEqual(full[key]);
    for (const b of ["gita", "quran", "bible"]) {
      expect(searchSite.pool[b as "gita"]).toEqual(full.pool[b].map(({ terms: _t, ...v }: { terms: unknown }) => v));
    }
  });
});
