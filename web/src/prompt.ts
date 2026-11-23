// The prompt box in the hero. Two modes share it: "Find verses" sends the text to the verse-topic search (in
// the browser); "Let the model write" ignores the text and asks the model for a book's style. Nothing typed is
// put in the URL or sent anywhere.

import { MAX_INPUT_CHARS, type BookId } from "./advice/api.ts";
import type { AdviceController } from "./advice/state.ts";
import type { Generator, GenPhase } from "./generate/ui.ts";

type Mode = "find" | "write";

const PLACEHOLDER: Record<Mode, string> = {
  find: "What’s on your mind?",
  write: "No prompt: the model ignores typed text. Pick a book, press Write.",
};

export function mountPrompt(form: HTMLFormElement, results: HTMLElement, gen: HTMLElement,
  ctl: AdviceController, mountGen: (onPhase: (p: GenPhase, label: string) => void) => Generator): void {
  const input = form.querySelector("textarea")!;
  const counter = form.querySelector<HTMLElement>("[data-counter]")!;
  const submit = form.querySelector<HTMLButtonElement>("button[type=submit]")!;
  const modes = form.querySelectorAll<HTMLButtonElement>("[data-mode]");
  const books = form.querySelectorAll<HTMLButtonElement>("[data-gen-book]");
  input.maxLength = MAX_INPUT_CHARS;

  let mode: Mode = "find";
  let book: BookId = "gita";
  let phase: GenPhase = "idle";
  let writeLabel = "Write";
  let findText = "";
  const generator = mountGen((p, label) => {
    phase = p;
    writeLabel = label;
    books.forEach((b) => (b.disabled = p !== "idle"));
    update();
  });

  function update() {
    counter.textContent = `${input.value.length} / ${MAX_INPUT_CHARS}`;
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
    if (mode === "write") {
      findText = input.value;
      input.value = "";
      input.readOnly = true;
    } else {
      generator.stop();
      input.value = findText;
      input.readOnly = false;
    }
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
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (mode === "find") void ctl.submit(input.value);
    else generator.toggle(book);
  });
  update();
}
