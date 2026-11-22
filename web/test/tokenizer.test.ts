import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";
import { CharTokenizer } from "../src/model/tokenizer.ts";
import vocab from "../src/model/vocab.json";

const fx = JSON.parse(readFileSync(fileURLToPath(new URL("./fixtures/tokenizer.json", import.meta.url)), "utf8")) as {
  tokens: string[];
  cases: { text: string; ids: number[] }[];
};

describe("tokenizer parity with scripture/tokenizer.py", () => {
  const tok = new CharTokenizer(vocab);

  test("same 67 tokens in the same order", () => {
    expect(vocab).toEqual(fx.tokens);
    expect(tok.vocabSize).toBe(67);
    expect(vocab.slice(0, 3)).toEqual(["<gita>", "<quran>", "<bible>"]);
  });

  for (const c of fx.cases) {
    test(`encode ${JSON.stringify(c.text)}`, () => {
      expect(tok.encode(c.text)).toEqual(c.ids);
    });
  }

  test("decode inverts encode for text inside the vocabulary", () => {
    const s = "<gita>The Blessed Lord said: Thou grievest for those that should not be grieved for.\n";
    expect(tok.decode(tok.encode(s))).toBe(s);
  });
});
