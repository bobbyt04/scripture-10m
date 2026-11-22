// Sampling on top of the hand-written forward pass, as in ScriptureLM.generate_stream (temperature, no top-k).

import type { Transformer } from "./transformer.ts";

/** mulberry32: a small seeded generator, so a seed always gives the same text. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Pick the next token. temperature <= 0 means greedy (argmax). */
export function sampleToken(logits: Float32Array, temperature: number, random: () => number): number {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  if (temperature <= 0) return best;
  const max = logits[best];
  const probs = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    probs[i] = Math.exp((logits[i] - max) / temperature);
    sum += probs[i];
  }
  let r = random() * sum;
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r < 0) return i;
  }
  return best;
}

export interface GenerateOptions {
  maxNewTokens: number;
  temperature: number;
  random: () => number;
}

/**
 * Yield new token ids one at a time after the prompt ids.
 *
 * The model has 256 learned positions. PyTorch's generate crops the context to the newest 256 tokens and
 * recomputes everything for each new token. Here, when the cache is full, it is rebuilt once from the newest
 * half window (128 tokens) and generation carries on; up to the first 256 tokens the two are identical.
 */
export function* generate(model: Transformer, prompt: readonly number[], opts: GenerateOptions): Generator<number> {
  const T = model.blockSize;
  const history = prompt.slice(-T);
  if (history.length === 0) throw new Error("prompt must have at least one token");
  model.reset();
  let logits: Float32Array = new Float32Array(0);
  for (const id of history) logits = model.step(id);
  for (let i = 0; i < opts.maxNewTokens; i++) {
    const next = sampleToken(logits, opts.temperature, opts.random);
    history.push(next);
    yield next;
    if (i === opts.maxNewTokens - 1) break;
    if (model.length >= T) {
      model.reset();
      for (const id of history.slice(-T / 2)) logits = model.step(id);
    } else {
      logits = model.step(next);
    }
  }
}
