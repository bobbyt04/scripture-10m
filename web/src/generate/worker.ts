/// <reference lib="webworker" />
// Runs the model off the main thread: downloads the weights (reporting progress), then streams text.

import { parseModel } from "../model/format.ts";
import { generate, seededRandom, topChoices } from "../model/generate.ts";
import { CharTokenizer } from "../model/tokenizer.ts";
import { verifyDownload } from "./verify.ts";
import { Transformer } from "../model/transformer.ts";
import vocab from "../model/vocab.json";

export type WorkerRequest =
  | { type: "load"; url: string; bytes: number; sha256: string }
  /** prompt: the book tag and the visitor's typed start, as token ids (see start.ts). */
  | { type: "generate"; prompt: number[]; temperature: number; maxTokens: number; seed: number }
  | { type: "stop" };

export type WorkerReply =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; ms: number }
  /** think: the last step's top 5 next characters (post-temperature softmax), what was left, and what was sampled. */
  | { type: "text"; text: string; think: Think | null }
  | { type: "done"; tokens: number; ms: number; stopped: boolean }
  | { type: "error"; message: string };

export interface Think {
  top: { char: string; p: number }[];
  rest: number;
  sampled: string;
}

const post = (m: WorkerReply) => (self as unknown as DedicatedWorkerGlobalScope).postMessage(m);
const tok = new CharTokenizer(vocab);
let model: Transformer | null = null;
let stopRequested = false;

/** Let queued messages (a stop) through between slices of work. */
const yieldToMessages = () => new Promise<void>((r) => {
  const c = new MessageChannel();
  c.port1.onmessage = () => r();
  c.port2.postMessage(0);
});

async function load(url: string, expected: number, sha256: string): Promise<void> {
  const t0 = performance.now();
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`could not download the model (HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length")) || expected;
  const buf = new Uint8Array(Math.max(total, expected));
  let loaded = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > buf.length) throw new Error("model download is larger than expected");
    buf.set(value, loaded);
    loaded += value.length;
    post({ type: "progress", loaded, total });
  }
  const data = buf.buffer.slice(0, loaded);
  await verifyDownload(data, expected, sha256);
  model = new Transformer(parseModel(data));
  post({ type: "ready", ms: performance.now() - t0 });
}

async function run(req: Extract<WorkerRequest, { type: "generate" }>): Promise<void> {
  if (!model) throw new Error("model not loaded");
  stopRequested = false;
  const t0 = performance.now();
  // Only the last step of each slice is shown, so its logits are kept and turned into probabilities when posted.
  let last: Float32Array = new Float32Array(0);
  const it = generate(model, req.prompt, {
    maxNewTokens: req.maxTokens, temperature: req.temperature, random: seededRandom(req.seed),
    onLogits: (l) => (last = l),
  });
  const think = (id: number): Think => {
    const c = topChoices(last, req.temperature, 5);
    return { top: c.top.map((t) => ({ char: tok.decode([t.id]), p: t.p })), rest: c.rest, sampled: tok.decode([id]) };
  };
  let n = 0;
  let pending = "";
  let sliceStart = performance.now();
  for (const id of it) {
    pending += tok.decode([id]);
    n++;
    if (performance.now() - sliceStart > 24) {
      post({ type: "text", text: pending, think: think(id) });
      pending = "";
      await yieldToMessages();
      if (stopRequested) break;
      sliceStart = performance.now();
    }
  }
  if (pending) post({ type: "text", text: pending, think: null });
  post({ type: "done", tokens: n, ms: performance.now() - t0, stopped: stopRequested });
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const req = e.data;
  if (req.type === "stop") {
    stopRequested = true;
    return;
  }
  const job = req.type === "load" ? load(req.url, req.bytes, req.sha256) : run(req);
  job.catch((err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }));
};
