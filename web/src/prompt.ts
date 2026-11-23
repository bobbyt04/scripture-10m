// The prompt box in the hero. Two modes share it: "Find verses" sends the text to the verse-topic search (in
// the browser); "Let the model write" gives the text to the model as the start of a passage in a book's style,
// after the same crisis check the search runs. Nothing typed is put in the URL or sent anywhere.

import { MAX_INPUT_CHARS, type BookId } from "./advice/api.ts";
import { MAX_START_CHARS } from "./generate/start.ts";
import type { AdviceController } from "./advice/state.ts";
import type { Generator, GenPhase } from "./generate/ui.ts";

type Mode = "find" | "write";

const PLACEHOLDER: Record<Mode, string> = {
  find: "What’s on your mind?",
  write: "Start a sentence, or leave empty.",
};

export function mountPrompt(form: HTMLFormElement, results: HTMLElement, gen: HTMLElement,
  ctl: AdviceController, mountGen: (onPhase: (p: GenPhase, label: string) => void) => Generator,
  /** The crisis note for this text, or null: the search's own suggest(), run before the model writes. */
  crisisNote: (text: string) => Promise<string | null>): void {
  const input = form.querySelector("textarea")!;
  const counter = form.querySelector<HTMLElement>("[data-counter]")!;
  const submit = form.querySelector<HTMLButtonElement>("button[type=submit]")!;
  const modes = form.querySelectorAll<HTMLButtonElement>("[data-mode]");
  // The book choice and temperature sit under the box in write mode; "Run it live" is in the write-up's Model section.
  const books = form.querySelectorAll<HTMLButtonElement>("[data-gen-book]");
  const temp = form.querySelector<HTMLInputElement>("[data-gen-temp]")!;
  const tempOut = form.querySelector<HTMLOutputElement>("[data-gen-temp-out]")!;
  const runLive = document.querySelector<HTMLButtonElement>("[data-gen-run]");
  temp.addEventListener("input", () => (tempOut.value = Number(temp.value).toFixed(1)));
  input.maxLength = MAX_INPUT_CHARS;

  let mode: Mode = "find";
  let book = ([...books].find((b) => b.getAttribute("aria-pressed") === "true")?.dataset.genBook ?? "gita") as BookId;
  let phase: GenPhase = "idle";
  let writeLabel = "Write";
  const generator = mountGen((p, label) => {
    phase = p;
    writeLabel = label;
    books.forEach((b) => (b.disabled = p !== "idle"));
    temp.disabled = p !== "idle";
    update();
  });

  const ARROW = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const SQUARE = '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><rect x="5" y="5" width="14" height="14" rx="2" fill="currentColor"/></svg>';
  function update() {
    counter.textContent = `${input.value.length} / ${mode === "find" ? MAX_INPUT_CHARS : MAX_START_CHARS}`;
    const stop = mode === "write" && phase === "writing";
    submit.setAttribute("aria-label", mode === "find" ? "Find" : writeLabel);
    submit.innerHTML = stop ? SQUARE : ARROW;
    submit.disabled = mode === "find" ? input.value.trim().length === 0 : phase === "loading";
  }

  function setMode(next: Mode) {
    if (next === mode) return;
    mode = next;
    form.dataset.mode = mode;
    modes.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    form.querySelectorAll<HTMLElement>("[data-write-only], [data-note-write]").forEach((el) => (el.hidden = mode !== "write"));
    form.querySelector<HTMLElement>("[data-note-find]")!.hidden = mode !== "find";
    results.hidden = mode !== "find";
    gen.hidden = mode !== "write" || !genStarted;
    input.placeholder = PLACEHOLDER[mode];
    update();
  }

  modes.forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode as Mode)));
  books.forEach((b) => b.addEventListener("click", () => {
    book = b.dataset.genBook as BookId;
    books.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  }));
  input.addEventListener("focus", () => ctl.warm(), { once: true });
  input.addEventListener("input", update);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!submit.disabled) form.requestSubmit();
    }
  });
  const reveal = (el: HTMLElement) => el.scrollIntoView({ block: "start" });
  let checking = false;
  let genStarted = false;
  /** Write with the panel's book and temperature, starting from `text`, after the crisis check. */
  const write = async (text: string) => {
    if (phase === "writing") return generator.stop();
    if (checking || phase !== "idle") return;
    checking = true;
    try {
      const note = text ? await crisisNote(text) : null;
      genStarted = true;
      gen.hidden = false;
      if (note !== null) generator.note(note);
      else generator.toggle(book, Number(temp.value), text);
    } catch {
      generator.note("Something went wrong. Please try again.");
    } finally {
      checking = false;
    }
    reveal(gen);
  };
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (mode === "find") void ctl.submit(input.value);
    else void write(input.value.trim());
  });
  // "Run it live" switches to "Let the model write" and starts from the typed text there (a "Find verses" question is not a start).
  runLive?.addEventListener("click", () => {
    const start = mode === "write" ? input.value.trim() : "";
    setMode("write");
    void write(start);
  });
  update();
}
