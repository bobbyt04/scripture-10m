// The Scripture-10M forward pass, written out by hand to mirror scripture/model.py, for one token at a
// time with a key-value cache. Weights are float32 arrays in PyTorch's layout (a Linear weight is
// [out, in], so y = W x + b). No dropout: this is inference only.

import type { ModelWeights } from "./format.ts";

interface Layer {
  ln1w: Float32Array; ln1b: Float32Array;
  qkv: Float32Array; // [3C, C], no bias
  projW: Float32Array; projB: Float32Array; // [C, C]
  ln2w: Float32Array; ln2b: Float32Array;
  fcW: Float32Array; fcB: Float32Array; // [4C, C]
  outW: Float32Array; outB: Float32Array; // [C, 4C]
  kCache: Float32Array; vCache: Float32Array; // [T, C]
}

const LN_EPS = 1e-5; // nn.LayerNorm default

/** erf, Abramowitz and Stegun 7.1.26 refined: absolute error below 1.2e-7 (Numerical Recipes erfc). */
function erf(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - r : r - 1;
}

/** nn.GELU() default: the exact form x * Phi(x). */
function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

function layerNorm(x: Float32Array, w: Float32Array, b: Float32Array, out: Float32Array): void {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; }
  const inv = 1 / Math.sqrt(v / n + LN_EPS);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

/** out = W x + b, W is [rows, cols] row-major. Four rows at a time, sharing each x load. */
function matvec(W: Float32Array, x: Float32Array, rows: number, cols: number, out: Float32Array, b: Float32Array): void {
  let r = 0;
  for (; r + 3 < rows; r += 4) {
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    const o0 = r * cols, o1 = o0 + cols, o2 = o1 + cols, o3 = o2 + cols;
    for (let c = 0; c < cols; c++) {
      const xv = x[c];
      s0 += W[o0 + c] * xv;
      s1 += W[o1 + c] * xv;
      s2 += W[o2 + c] * xv;
      s3 += W[o3 + c] * xv;
    }
    out[r] = s0 + b[r]; out[r + 1] = s1 + b[r + 1]; out[r + 2] = s2 + b[r + 2]; out[r + 3] = s3 + b[r + 3];
  }
  for (; r < rows; r++) {
    let s0 = 0;
    const o = r * cols;
    for (let c = 0; c < cols; c++) s0 += W[o + c] * x[c];
    out[r] = s0 + b[r];
  }
}

export class Transformer {
  readonly vocabSize: number;
  readonly blockSize: number;
  private readonly C: number;
  private readonly nHead: number;
  private readonly tokEmb: Float32Array;
  private readonly posEmb: Float32Array;
  private readonly layers: Layer[];
  private readonly lnfW: Float32Array;
  private readonly lnfB: Float32Array;
  private readonly head: Float32Array;
  private pos = 0;
  // scratch buffers, allocated once
  private readonly x: Float32Array;
  private readonly h: Float32Array;
  private readonly qkvBuf: Float32Array;
  private readonly y: Float32Array;
  private readonly tmp: Float32Array;
  private readonly hidden: Float32Array;
  private readonly att: Float32Array;
  private readonly logits: Float32Array;
  private readonly noBias3C: Float32Array; // zeros, for the bias-free q/k/v projection
  private readonly noBiasV: Float32Array; // zeros, for the bias-free output head

  constructor(w: ModelWeights) {
    const cfg = w.config;
    const C = (this.C = cfg.n_embd);
    this.nHead = cfg.n_head;
    this.vocabSize = cfg.vocab_size;
    this.blockSize = cfg.block_size;
    const get = (name: string, size: number): Float32Array => {
      const t = w.tensors.get(name);
      if (!t) throw new Error(`model file has no tensor ${name}`);
      if (t.length !== size) throw new Error(`${name} has ${t.length} values, expected ${size}`);
      return t;
    };
    const T = cfg.block_size;
    this.tokEmb = get("tok_emb.weight", cfg.vocab_size * C);
    this.posEmb = get("pos_emb.weight", T * C);
    this.layers = [];
    for (let l = 0; l < cfg.n_layer; l++) {
      const p = `blocks.${l}.`;
      this.layers.push({
        ln1w: get(p + "ln1.weight", C), ln1b: get(p + "ln1.bias", C),
        qkv: get(p + "attn.qkv.weight", 3 * C * C),
        projW: get(p + "attn.proj.weight", C * C), projB: get(p + "attn.proj.bias", C),
        ln2w: get(p + "ln2.weight", C), ln2b: get(p + "ln2.bias", C),
        fcW: get(p + "mlp.net.0.weight", 4 * C * C), fcB: get(p + "mlp.net.0.bias", 4 * C),
        outW: get(p + "mlp.net.2.weight", 4 * C * C), outB: get(p + "mlp.net.2.bias", C),
        kCache: new Float32Array(T * C), vCache: new Float32Array(T * C),
      });
    }
    this.lnfW = get("ln_f.weight", C);
    this.lnfB = get("ln_f.bias", C);
    this.head = get("lm_head.weight", cfg.vocab_size * C);
    this.x = new Float32Array(C);
    this.h = new Float32Array(C);
    this.qkvBuf = new Float32Array(3 * C);
    this.y = new Float32Array(C);
    this.tmp = new Float32Array(C);
    this.hidden = new Float32Array(4 * C);
    this.att = new Float32Array(T);
    this.logits = new Float32Array(cfg.vocab_size);
    this.noBias3C = new Float32Array(3 * C);
    this.noBiasV = new Float32Array(cfg.vocab_size);
  }

  /** Number of tokens in the cache (the position the next token will take). */
  get length(): number {
    return this.pos;
  }

  reset(): void {
    this.pos = 0;
  }

  /**
   * Append one token and return the logits for the token after it (a view that the next call overwrites).
   * Equal to the last row of ScriptureLM.forward on every token fed since the last reset.
   */
  step(token: number): Float32Array {
    const { C, nHead, x, h, qkvBuf, y, tmp, hidden, att } = this;
    const t = this.pos;
    if (t >= this.blockSize) throw new Error("context window full; reset first");
    if (token < 0 || token >= this.vocabSize) throw new Error(`token id ${token} out of range`);
    const hs = C / nHead;
    const scale = 1 / Math.sqrt(hs);
    for (let i = 0; i < C; i++) x[i] = this.tokEmb[token * C + i] + this.posEmb[t * C + i];

    for (const L of this.layers) {
      // attention: tokens talk
      layerNorm(x, L.ln1w, L.ln1b, h);
      matvec(L.qkv, h, 3 * C, C, qkvBuf, this.noBias3C);
      L.kCache.set(qkvBuf.subarray(C, 2 * C), t * C);
      L.vCache.set(qkvBuf.subarray(2 * C, 3 * C), t * C);
      for (let hd = 0; hd < nHead; hd++) {
        const off = hd * hs;
        let max = -Infinity;
        for (let s = 0; s <= t; s++) { // causal: positions <= t only
          let dot = 0;
          const k = s * C + off;
          for (let d = 0; d < hs; d++) dot += qkvBuf[off + d] * L.kCache[k + d];
          const a = dot * scale;
          att[s] = a;
          if (a > max) max = a;
        }
        let sum = 0;
        for (let s = 0; s <= t; s++) { const e = Math.exp(att[s] - max); att[s] = e; sum += e; }
        for (let d = 0; d < hs; d++) y[off + d] = 0;
        for (let s = 0; s <= t; s++) {
          const p = att[s] / sum;
          const v = s * C + off;
          for (let d = 0; d < hs; d++) y[off + d] += p * L.vCache[v + d];
        }
      }
      matvec(L.projW, y, C, C, tmp, L.projB);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
      // MLP: tokens think
      layerNorm(x, L.ln2w, L.ln2b, h);
      matvec(L.fcW, h, 4 * C, C, hidden, L.fcB);
      for (let i = 0; i < 4 * C; i++) hidden[i] = gelu(hidden[i]);
      matvec(L.outW, hidden, C, 4 * C, tmp, L.outB);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
    }
    layerNorm(x, this.lnfW, this.lnfB, h);
    matvec(this.head, h, this.vocabSize, C, this.logits, this.noBiasV);
    this.pos = t + 1;
    return this.logits;
  }
}
