// The TypeScript forward pass against PyTorch, on fixtures written by web/scripts/export_model.py from
// exactly the weights in the published model file.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, test } from "vitest";
import { parseModel, type ModelWeights } from "../src/model/format.ts";
import { generate, sampleToken, seededRandom, topChoices } from "../src/model/generate.ts";
import { CharTokenizer } from "../src/model/tokenizer.ts";
import { Transformer } from "../src/model/transformer.ts";
import manifest from "../src/model/manifest.json";
import vocab from "../src/model/vocab.json";

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));

interface Parity {
  model_file: string;
  vocab_size: number;
  logit_cases: { prompt: string; ids: number[]; positions: number[]; logits: number[][] }[];
  greedy: { prompt: string; ids: number[]; tokens: number[]; text: string }[];
}
const parity = JSON.parse(readFileSync(here("./fixtures/parity.json"), "utf8")) as Parity;

/** Stated tolerance: logits agree to within 1e-4 absolute (they span roughly -20 to 20; measured worst is about 1.3e-5). */
const LOGIT_TOL = 1e-4;

let weights: ModelWeights;
let model: Transformer;

beforeAll(() => {
  const buf = readFileSync(here("../public/" + manifest.file));
  weights = parseModel(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  model = new Transformer(weights);
});

describe("model file", () => {
  test("fixtures were written for the published file", () => {
    expect(manifest.file.endsWith(parity.model_file)).toBe(true);
  });

  test("header matches the manifest and the parameter count", () => {
    expect(weights.format).toBe(manifest.format);
    expect(weights.config).toMatchObject(manifest.config);
    let n = 0;
    for (const t of weights.tensors.values()) n += t.length;
    expect(n).toBe(manifest.params);
    expect(n).toBe(10_790_400);
  });

  test("rejects a file that is not a model", () => {
    expect(() => parseModel(new Uint8Array(64).buffer)).toThrow(/not a Scripture-10M/);
  });
});

describe("forward pass parity with PyTorch", () => {
  for (const c of parity.logit_cases) {
    test(`logits for ${JSON.stringify(c.prompt.slice(0, 40))}`, () => {
      model.reset();
      let worst = 0;
      let checked = 0;
      for (let t = 0; t < c.ids.length; t++) {
        const logits = model.step(c.ids[t]);
        const k = c.positions.indexOf(t);
        if (k < 0) continue;
        const want = c.logits[k];
        expect(logits.length).toBe(want.length);
        for (let i = 0; i < want.length; i++) worst = Math.max(worst, Math.abs(logits[i] - want[i]));
        checked++;
      }
      expect(checked).toBe(c.positions.length);
      expect(worst).toBeLessThan(LOGIT_TOL);
    });
  }

  for (const g of parity.greedy) {
    test(`greedy generation matches PyTorch for ${g.tokens.length} tokens after ${g.prompt}`, () => {
      expect(g.tokens.length).toBeGreaterThanOrEqual(100);
      const out = [...generate(model, g.ids, { maxNewTokens: g.tokens.length, temperature: 0, random: Math.random })];
      expect(out).toEqual(g.tokens);
      expect(new CharTokenizer(vocab).decode(out)).toBe(g.text);
    });
  }
});

describe("generation", () => {
  test("refuses a token past the context window without a reset", () => {
    model.reset();
    for (let i = 0; i < model.blockSize; i++) model.step(1);
    expect(() => model.step(1)).toThrow(/context window full/);
  });

  test("keeps going past the 256-token window and stays deterministic for a seed", () => {
    const tok = new CharTokenizer(vocab);
    const run = () =>
      [...generate(model, tok.encode("<bible>"), { maxNewTokens: 300, temperature: 0.8, random: seededRandom(7) })];
    const a = run();
    expect(a.length).toBe(300);
    expect(run()).toEqual(a);
    expect(a.every((id) => id >= 0 && id < tok.vocabSize)).toBe(true);
  });

  test("the top-5 guesses shown while writing are the sampling softmax: they and the rest sum to 1", () => {
    const tok = new CharTokenizer(vocab);
    for (const temperature of [0.2, 0.8, 1.5]) {
      let shown: ReturnType<typeof topChoices> | null = null;
      const it = generate(model, tok.encode("<gita>"), {
        maxNewTokens: 40, temperature, random: seededRandom(3),
        onLogits: (l) => (shown = topChoices(l, temperature, 5)),
      });
      for (const id of it) {
        const c = shown!;
        expect(c.top).toHaveLength(5);
        expect(c.top.reduce((a, t) => a + t.p, 0) + c.rest).toBeCloseTo(1, 9);
        expect(c.top.every((t, i) => t.p >= 0 && (i === 0 || t.p <= c.top[i - 1].p))).toBe(true);
        expect(id >= 0 && id < tok.vocabSize && vocab.includes(tok.decode([id]))).toBe(true);
      }
    }
    const two = topChoices(new Float32Array([0, Math.log(3)]), 1, 5); // probabilities 1/4 and 3/4
    expect(two.top.map((t) => t.id)).toEqual([1, 0]);
    expect(two.top[0].p).toBeCloseTo(0.75, 6); // log(3) is stored as float32
    expect(two.rest).toBeCloseTo(0, 9);
  });

  test("temperature sampling follows the softmax and topP trims the tail", () => {
    const logits = new Float32Array([0, Math.log(3)]); // probabilities 1/4 and 3/4
    const rnd = seededRandom(1);
    let ones = 0;
    for (let i = 0; i < 20000; i++) ones += sampleToken(logits, 1, rnd);
    expect(ones / 20000).toBeCloseTo(0.75, 1);
    expect(sampleToken(logits, 0, rnd)).toBe(1);
    // With topP=0.5, only the top token (p=0.75 >= 0.5) stays in the nucleus.
    for (let i = 0; i < 50; i++) expect(sampleToken(logits, 1, rnd, 0.5)).toBe(1);
  });
});
