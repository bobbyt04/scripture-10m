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
  // The book tabs, the temperature slider and the "Run it live" button live in the model panel.
  const books = gen.querySelectorAll<HTMLButtonElement>("[data-gen-book]");
  const temp = gen.querySelector<HTMLInputElement>("[data-gen-temp]")!;
  const tempOut = gen.querySelector<HTMLOutputElement>("[data-gen-temp-out]")!;
  const runLive = gen.querySelector<HTMLButtonElement>("[data-gen-run]")!;
  temp.addEventListener("input", () => (tempOut.value = Number(temp.value).toFixed(1)));
  input.maxLength = MAX_INPUT_CHARS;

  let mode: Mode = "find";
  let book = ([...books].find((b) => b.getAttribute("aria-pressed") === "true")?.dataset.genBook ?? "bible") as BookId;
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
    input.placeholder = PLACEHOLDER[mode];
    update();
  }

  modes.forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode as Mode)));
  books.forEach((b) => b.addEventListener("click", () => {
    book = b.dataset.genBook as BookId;
    books.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    generator.showSample(book);
  }));
  generator.showSample(book);
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
  /** Write with the panel's book and temperature, starting from `text`, after the crisis check. */
  const write = async (text: string) => {
    if (phase === "writing") return generator.stop();
    if (checking || phase !== "idle") return;
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
    if (matchMedia("(max-width: 900px)").matches) reveal(gen);
  };
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (mode === "find") void ctl.submit(input.value);
    else void write(input.value.trim());
  });
  // "Run it live" uses the typed text as the start only in "Let the model write"; a "Find verses" question is not a start.
  runLive.addEventListener("click", () => void write(mode === "write" ? input.value.trim() : ""));
  update();
}
