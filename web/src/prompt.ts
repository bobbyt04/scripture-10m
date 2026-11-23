// The prompt box in the hero. Two modes share it: "Find verses" sends the text to the verse-topic search (in
// the browser); "Let the model write" gives the text to the model as the start of a passage in a book's style,
// after the same crisis check the search runs. Nothing typed is put in the URL or sent anywhere.

import { MAX_INPUT_CHARS, type BookId } from "./advice/api.ts";
import { prefersReducedMotion } from "./dom.ts";
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
  const books = form.querySelectorAll<HTMLButtonElement>("[data-gen-book]");
  const temp = form.querySelector<HTMLInputElement>("[data-gen-temp]")!;
  const tempOut = form.querySelector<HTMLOutputElement>("[data-gen-temp-out]")!;
  temp.addEventListener("input", () => (tempOut.value = Number(temp.value).toFixed(1)));
  input.maxLength = MAX_INPUT_CHARS;

  let mode: Mode = "find";
  let book: BookId = "gita";
  let phase: GenPhase = "idle";
  let writeLabel = "Write";
  const generator = mountGen((p, label) => {
    phase = p;
    writeLabel = label;
    books.forEach((b) => (b.disabled = p !== "idle"));
    temp.disabled = p !== "idle";
    update();
  });

  function update() {
    counter.textContent = `${input.value.length} / ${mode === "find" ? MAX_INPUT_CHARS : MAX_START_CHARS}`;
    if (mode === "find") {
      submit.textContent = "Find";
      submit.disabled = input.value.trim().length === 0;
    } else {
      submit.textContent = writeLabel;
      submit.disabled = phase === "loading";
    }
    submit.classList.toggle("is-stop", mode === "write" && phase === "writing");
  }

  function setMode(next: Mode) {
    if (next === mode) return;
    mode = next;
    form.dataset.mode = mode;
    modes.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    form.querySelectorAll<HTMLElement>("[data-write-only], [data-note-write]").forEach((el) => (el.hidden = mode !== "write"));
    form.querySelector<HTMLElement>("[data-note-find]")!.hidden = mode !== "find";
    results.hidden = mode !== "find";
    gen.hidden = mode !== "write";
    if (mode === "find") generator.stop();
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
  const reveal = (el: HTMLElement) =>
    el.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  let checking = false;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (mode === "find") return void ctl.submit(input.value);
    if (phase === "writing") return generator.stop();
    if (checking || phase !== "idle") return;
    const text = input.value.trim();
    checking = true;
    try {
      const note = text ? await crisisNote(text) : null;
      if (note !== null) generator.note(note);
      else generator.toggle(book, Number(temp.value), text);
    } catch {
      generator.note("Something went wrong. Please try again.");
    } finally {
      checking = false;
    }
    reveal(gen);
  });
  update();
}
