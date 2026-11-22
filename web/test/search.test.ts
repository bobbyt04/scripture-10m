// The TypeScript search against the Python one: every case of search/artifacts/fixtures.json, then thousands of
// varied messages run through both (web/scripts/search_parity.py prints Python's answers at test time).
// Set PYTHON to the interpreter to use (default python3); scripture/search.py needs no packages.

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";
import siteData from "../src/generated/search.json";
import { BOOKS, cap, Searcher, words, type SearchData } from "../src/search/searcher.ts";

const root = (p: string) => fileURLToPath(new URL(`../../${p}`, import.meta.url));
const fixtures = JSON.parse(readFileSync(root("search/artifacts/fixtures.json"), "utf8"));
const searcher = new Searcher(siteData as unknown as SearchData);

describe("fixtures.json", () => {
  test.each(fixtures.words as [string, string[]][])("words(%j)", (text, expected) => {
    expect(words(text)).toEqual(expected);
  });

  for (const c of fixtures.suggest) {
    test(`suggest(${JSON.stringify(c.query.slice(0, 60))})`, () => {
      expect(searcher.crisisWords(c.query)).toEqual(c.crisis_words);
      expect(searcher.signal(c.query)).toBe(c.signal);
      expect(searcher.comfortCue(c.query)).toBe(c.comfort_cue);
      expect(searcher.suggest(c.query)).toEqual({ category: c.category, note: c.note, topics: c.topics });
    });
  }

  for (const c of fixtures.topic) {
    test(`topic(${c.theme}, k=${c.k}, sensitive=${c.sensitive})`, () => {
      const r = searcher.topic(c.theme, { k: c.k, sensitive: c.sensitive });
      expect({ label: r.label, blurb: r.blurb, note: r.note }).toEqual({ label: c.label, blurb: c.blurb, note: c.note });
      for (const b of BOOKS) {
        expect({ verses: r.books[b].verses.map((v) => [v.ref, v.priority]), nothing: r.books[b].nothing }).toEqual(c.books[b]);
      }
    });
  }

  test("topics()", () => {
    expect(searcher.topics()).toEqual(fixtures.topics);
  });

  test("an unknown topic and a bad k are errors", () => {
    expect(() => searcher.topic("no_such_topic")).toThrow(/unknown theme/);
    expect(() => searcher.topic("grief", { k: 0 })).toThrow(/positive integer/);
    expect(() => searcher.topic("grief", { k: 1.5 })).toThrow(/positive integer/);
  });

  test("the cap counts code points, as Python's query[:500] does", () => {
    expect(cap("𝒜".repeat(600), 500)).toBe("𝒜".repeat(500));
    expect(cap("ab", 500)).toBe("ab");
  });

  test("words that are also names of object properties are plain words", () => {
    expect(searcher.termOf("constructor")).toBeNull();
    expect(searcher.suggest("constructor tostring valueof hasownproperty").category).toBe("none");
  });
});

describe("parity with the Python Searcher", () => {
  const [python, ...args] = (process.env.PYTHON ?? "python3").split(" "); // may carry arguments ("arch -arm64 python3")
  const expected = JSON.parse(execFileSync(python, [...args, root("web/scripts/search_parity.py")], {
    encoding: "utf8", maxBuffer: 64 << 20,
  }));

  test("covers enough varied messages", () => {
    expect(expected.cases.length).toBeGreaterThanOrEqual(2000);
    const categories = new Set(expected.cases.map((c: { suggest: { category: string } }) => c.suggest.category));
    expect([...categories].sort()).toEqual(["crisis", "general", "none", "sensitive"]);
  });

  test("suggest() gives identical JSON for every message", () => {
    const mismatches: unknown[] = [];
    for (const c of expected.cases) {
      const got = { crisis_words: searcher.crisisWords(c.query), signal: searcher.signal(c.query),
        comfort_cue: searcher.comfortCue(c.query), suggest: searcher.suggest(c.query) };
      const want = { crisis_words: c.crisis_words, signal: c.signal, comfort_cue: c.comfort_cue, suggest: c.suggest };
      if (JSON.stringify(got) !== JSON.stringify(want)) mismatches.push({ query: c.query, got, want });
    }
    expect(mismatches.slice(0, 5)).toEqual([]);
  });

  test("topics() and topic() for every theme, sensitive or not", () => {
    expect(JSON.stringify(searcher.topics())).toBe(JSON.stringify(expected.topics));
    for (const t of expected.topic) {
      expect(JSON.stringify(searcher.topic(t.id, { sensitive: t.sensitive }))).toBe(JSON.stringify(t.result));
    }
  });
});
