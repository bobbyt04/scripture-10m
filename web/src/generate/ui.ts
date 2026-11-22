import { BOOK_NAMES, type BookId } from "../advice/api.ts";
import { h } from "../dom.ts";
import manifest from "../model/manifest.json";
import type { WorkerReply, WorkerRequest } from "./worker.ts";

const MAX_TOKENS = 400;
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

/** What the visitor reads when loading or generating fails; the technical reason goes in data-error. */
export function errorMessage(modelLoaded: boolean): string {
  return modelLoaded
    ? "The model stopped unexpectedly. Press Generate to try again."
    : "The model could not be downloaded completely. Check your connection and press Try again.";
}

export function mountGenerate(root: HTMLElement): void {
  const q = <T extends Element>(sel: string) => root.querySelector<T>(sel)!;
  const button = q<HTMLButtonElement>("[data-gen-start]");
  const stop = q<HTMLButtonElement>("[data-gen-stop]");
  const temp = q<HTMLInputElement>("[data-gen-temp]");
  const tempOut = q<HTMLOutputElement>("[data-gen-temp-out]");
  const progress = q<HTMLElement>("[data-gen-progress]");
  const bar = q<HTMLElement>("[data-gen-bar]");
  const progressText = q<HTMLElement>("[data-gen-progress-text]");
  const output = q<HTMLElement>("[data-gen-output]");
  const status = q<HTMLElement>("[data-gen-status]");
  const live = q<HTMLElement>("[data-gen-live]");
  const sizeEls = root.querySelectorAll<HTMLElement>("[data-model-size]");
  sizeEls.forEach((el) => (el.textContent = `${modelSizeMB} MB`));

  let worker: Worker | null = null;
  let ready = false;
  let busy = false;
  let raw = "";
  let book: BookId = "gita";
  let announced = 0;

  const selectedBook = () => (root.querySelector<HTMLInputElement>("input[name=gen-book]:checked")?.value ?? "gita") as BookId;
  const showTemp = () => (tempOut.value = Number(temp.value).toFixed(1));
  temp.addEventListener("input", showTemp);
  showTemp();

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

  const setBusy = (b: boolean) => {
    busy = b;
    button.hidden = b;
    stop.hidden = !b;
    root.querySelectorAll<HTMLInputElement>("input[name=gen-book]").forEach((r) => (r.disabled = b));
    if (b) stop.focus();
    else if (document.activeElement === stop || document.activeElement === document.body) button.focus();
  };

  const send = (m: WorkerRequest) => worker!.postMessage(m);

  const onMessage = (e: MessageEvent<WorkerReply>) => {
    const m = e.data;
    if (m.type === "progress") {
      const pct = Math.min(100, (100 * m.loaded) / m.total);
      bar.style.transform = `scaleX(${pct / 100})`;
      progress.setAttribute("aria-valuenow", String(Math.round(pct)));
      progressText.textContent = `Downloading the model: ${(m.loaded / 1e6).toFixed(1)} of ${(m.total / 1e6).toFixed(1)} MB`;
    } else if (m.type === "ready") {
      ready = true;
      progress.hidden = true;
      progressText.textContent = "";
      button.disabled = false;
      setBusy(true);
      start();
    } else if (m.type === "text") {
      raw += m.text;
      draw(true);
    } else if (m.type === "done") {
      draw(false);
      setBusy(false);
      const secs = m.ms / 1000;
      const rate = m.tokens / secs;
      status.textContent = `${m.tokens} characters in ${secs.toFixed(1)} s (${rate.toFixed(0)} per second), computed on your device.` +
        (m.stopped ? " Stopped early." : "");
      root.dataset.tokensPerSecond = rate.toFixed(1);
      live.textContent = m.stopped ? "Stopped." : "Finished.";
      button.textContent = "Generate again";
    } else if (m.type === "error") {
      button.disabled = false;
      setBusy(false);
      progress.hidden = true;
      progressText.textContent = "";
      status.textContent = errorMessage(ready);
      status.dataset.error = m.message;
      if (!ready) {
        worker?.terminate();
        worker = null;
        button.textContent = "Try again";
      }
    }
  };

  function start() {
    book = selectedBook();
    raw = "";
    announced = 0;
    status.textContent = "";
    delete status.dataset.error;
    output.dataset.book = book;
    draw(true);
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    send({ type: "generate", book, temperature: Number(temp.value), maxTokens: MAX_TOKENS, seed });
  }

  button.addEventListener("click", () => {
    if (busy || button.disabled) return;
    if (ready) {
      setBusy(true);
      return start();
    }
    button.disabled = true;
    if (!worker) {
      worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
      worker.onmessage = onMessage;
      worker.onerror = () => onMessage(new MessageEvent("message", { data: { type: "error", message: "the model worker failed to start" } }));
    }
    progress.hidden = false;
    bar.style.transform = "scaleX(0)";
    progressText.textContent = `Downloading the model: 0 of ${modelSizeMB} MB`;
    status.textContent = "";
    send({ type: "load", url: `${import.meta.env.BASE_URL}${manifest.file}`, bytes: manifest.bytes, sha256: manifest.sha256 });
  });
  stop.addEventListener("click", () => send({ type: "stop" }));
}
