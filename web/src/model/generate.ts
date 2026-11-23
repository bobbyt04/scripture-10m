// Sampling on top of the forward pass in transformer.ts, as in ScriptureLM.generate_stream (temperature, no top-k).

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

/** Pick the next token. temperature <= 0 means greedy (argmax). topP < 1 trims the low-probability tail. */
export function sampleToken(logits: Float32Array, temperature: number, random: () => number, topP = 1): number {
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
  if (topP > 0 && topP < 1) {
    const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
    const cutoff = topP * sum;
    let kept = 0;
    let n = 0;
    for (const idx of order) {
      kept += probs[idx];
      n++;
      if (kept >= cutoff) break;
    }
    let r = random() * kept;
    for (let i = 0; i < n; i++) {
      const idx = order[i];
      r -= probs[idx];
      if (r < 0) return idx;
    }
    return order[0];
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
  /** Nucleus sampling threshold (0 < topP <= 1). Defaults to 1 (full softmax). */
  topP?: number;
  /** Called with the logits of each step just before the next token is sampled from them. */
  onLogits?: (logits: Float32Array) => void;
}

/**
 * The k most likely next tokens under the same temperature softmax that sampleToken draws from, and the
 * probability left for all the others (so the k probabilities and `rest` sum to 1).
 */
export function topChoices(logits: Float32Array, temperature: number, k: number): { top: { id: number; p: number }[]; rest: number } {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const probs = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    probs[i] = temperature <= 0 ? (logits[i] === max ? 1 : 0) : Math.exp((logits[i] - max) / temperature);
    sum += probs[i];
  }
  const ids = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, k);
  const top = ids.map((id) => ({ id, p: probs[id] / sum }));
  return { top, rest: Math.max(0, 1 - top.reduce((a, t) => a + t.p, 0)) };
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
    opts.onLogits?.(logits);
    const next = sampleToken(logits, opts.temperature, opts.random, opts.topP ?? 1);
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
