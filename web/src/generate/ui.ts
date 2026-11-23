import { BOOK_NAMES, type BookId } from "../advice/api.ts";
import { h, prefersReducedMotion } from "../dom.ts";
import manifest from "../model/manifest.json";
import type { Think, WorkerReply, WorkerRequest } from "./worker.ts";

const MAX_TOKENS = 400;
/** The "watch it think" bars are redrawn at most this often (less with reduced motion), so drawing never slows the text. */
const thinkEveryMs = () => (prefersReducedMotion() ? 600 : 120);
const TAG = /^<(gita|quran|bible)>/;

export const modelSizeMB = (manifest.bytes / 1e6).toFixed(1);

/** Turn raw model output into display lines. Tags are dropped; a switch to another book is flagged. */
export function toLines(raw: string, book: BookId): { text: string; switchedTo: BookId | null }[] {
  return raw.split("\n").flatMap((line) => {
    const m = line.match(TAG);
    const b = (m?.[1] as BookId | undefined) ?? book;
    const text = m ? line.slice(m[0].length) : line;
    return text ? [{ text, switchedTo: b === book ? null : b }] : [];
  });
}

/** A token as the visitor sees it in the bars: space and newline get visible symbols. */
export const showChar = (c: string) => (c === " " ? "␣" : c === "\n" ? "↵" : c);

/** What the visitor reads when loading or generating fails; the technical reason goes in data-error. */
export function errorMessage(modelLoaded: boolean): string {
  return modelLoaded
    ? "The model stopped unexpectedly. Press Write to try again."
    : "The model could not be downloaded completely. Check your connection and press Try again.";
}

export interface Generator {
  /** Start writing in the style of `book`, or stop if it is already writing. Loads the model on first use. */
  toggle(book: BookId, temperature: number): void;
  stop(): void;
}

export type GenPhase = "idle" | "loading" | "writing";

/** The generation panel. `onPhase` tells the prompt box what its button should say and do. */
export function mountGenerate(root: HTMLElement, onPhase: (phase: GenPhase, label: string) => void): Generator {
  const q = <T extends Element>(sel: string) => root.querySelector<T>(sel)!;
  const progress = q<HTMLElement>("[data-gen-progress]");
  const bar = q<HTMLElement>("[data-gen-bar]");
  const output = q<HTMLElement>("[data-gen-output]");
  const rate = q<HTMLElement>("[data-gen-rate]");
  const status = q<HTMLElement>("[data-gen-status]");
  const live = q<HTMLElement>("[data-gen-live]");
  const think = q<HTMLElement>("[data-gen-think]");
  const bars = q<HTMLElement>("[data-gen-bars]");

  let worker: Worker | null = null;
  let ready = false;
  let phase: GenPhase = "idle";
  let raw = "";
  let book: BookId = "gita";
  let announced = 0;
  let t0 = 0;
  let temperature = 0.8;
  let thinkDrawn = 0;

  const drawThink = (t: Think) => {
    think.hidden = false;
    bars.replaceChildren(...t.top.map((c) =>
      h("li", { class: `think__row${c.char === t.sampled ? " is-sampled" : ""}` },
        h("span", { class: "think__char" }, showChar(c.char)),
        h("span", { class: "think__bar", style: `--p:${c.p.toFixed(4)}` }),
        h("span", { class: "think__p" }, `${(100 * c.p).toFixed(1)}%`))),
    h("li", { class: "think__rest" }, `others ${(100 * t.rest).toFixed(1)}%`,
      t.top.some((c) => c.char === t.sampled) ? null : ` · sampled ${showChar(t.sampled)}`));
  };

  const draw = (generating: boolean) => {
    const lines = toLines(raw, book);
    output.replaceChildren(...lines.map((l, i) =>
      h("p", { class: "gen__line" },
        l.switchedTo ? h("span", { class: "gen__switch" }, `switched to ${BOOK_NAMES[l.switchedTo]} style`) : null,
        l.text,
        generating && i === lines.length - 1 ? h("span", { class: "gen__caret", "aria-hidden": "true" }) : null)));
    if (generating && lines.length === 0) output.append(h("p", { class: "gen__line" }, h("span", { class: "gen__caret", "aria-hidden": "true" })));
    // announce finished lines only, never single characters
    const done = raw.split("\n").length - 1;
    if (done > announced) {
      const finished = toLines(raw.split("\n").slice(announced, done).join("\n"), book);
      live.textContent = finished.map((l) => l.text).join(" ");
      announced = done;
    }
  };

  const setPhase = (p: GenPhase, label: string) => {
    phase = p;
    onPhase(p, label);
  };

  const send = (m: WorkerRequest) => worker!.postMessage(m);

  const onMessage = (e: MessageEvent<WorkerReply>) => {
    const m = e.data;
    if (m.type === "progress") {
      const frac = Math.min(1, m.loaded / m.total);
      bar.style.transform = `scaleX(${frac})`;
      progress.setAttribute("aria-valuenow", String(Math.round(100 * frac)));
      status.textContent = `Downloading the model: ${(m.loaded / 1e6).toFixed(1)} / ${(m.total / 1e6).toFixed(1)} MB`;
    } else if (m.type === "ready") {
      ready = true;
      progress.hidden = true;
      start();
    } else if (m.type === "text") {
      raw += m.text;
      const secs = (performance.now() - t0) / 1000;
      rate.textContent = `${raw.length} chars · ${secs > 0.2 ? Math.round(raw.length / secs) : "–"} chars/s`;
      draw(true);
      const now = performance.now();
      if (m.think && now - thinkDrawn >= thinkEveryMs()) {
        thinkDrawn = now;
        drawThink(m.think);
      }
    } else if (m.type === "done") {
      draw(false);
      setPhase("idle", "Write again");
      const secs = m.ms / 1000;
      const perSec = m.tokens / secs;
      rate.textContent = `${m.tokens} chars · ${perSec.toFixed(0)} chars/s · ${secs.toFixed(1)} s on your device`;
      status.textContent = m.stopped ? "Stopped." : "";
      root.dataset.tokensPerSecond = perSec.toFixed(1);
      live.textContent = m.stopped ? "Stopped." : "Finished.";
    } else if (m.type === "error") {
      progress.hidden = true;
      status.textContent = errorMessage(ready);
      status.dataset.error = m.message;
      if (!ready) {
        worker?.terminate();
        worker = null;
      }
      setPhase("idle", ready ? "Write" : "Try again");
    }
  };

  function start() {
    raw = "";
    announced = 0;
    status.textContent = "";
    rate.textContent = "";
    delete status.dataset.error;
    delete root.dataset.tokensPerSecond;
    output.dataset.book = book;
    draw(true);
    setPhase("writing", "Stop");
    t0 = performance.now();
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    send({ type: "generate", book, temperature, maxTokens: MAX_TOKENS, seed });
  }

  const stop = () => {
    if (phase === "writing") send({ type: "stop" });
  };

  return {
    stop,
    toggle(b: BookId, temp: number) {
      if (phase === "writing") return stop();
      if (phase === "loading") return;
      book = b;
      temperature = temp;
      if (ready) return start();
      setPhase("loading", "Loading…");
      if (!worker) {
        worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
        worker.onmessage = onMessage;
        worker.onerror = () => onMessage(new MessageEvent("message", { data: { type: "error", message: "the model worker failed to start" } }));
      }
      progress.hidden = false;
      bar.style.transform = "scaleX(0)";
      status.textContent = `Downloading the model: 0 / ${modelSizeMB} MB`;
      delete status.dataset.error;
      send({ type: "load", url: `${import.meta.env.BASE_URL}${manifest.file}`, bytes: manifest.bytes, sha256: manifest.sha256 });
    },
  };
}
