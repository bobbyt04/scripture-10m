// The prompt box in the hero. Two modes share it. "Model" (the default) gives the typed text to the model as the
// start of a passage in a book's style; "Verses" sends it to the verse-topic search. Both run in the browser, and
// the search's crisis check runs on typed text in both. Nothing typed is put in the URL or sent anywhere.

import { MAX_INPUT_CHARS, type BookId } from "./advice/api.ts";
import { h, prefersReducedMotion } from "./dom.ts";
import { MAX_START_CHARS } from "./generate/start.ts";
import type { AdviceController } from "./advice/state.ts";
import { looksLikeQuestion, type Generator, type GenPhase } from "./generate/ui.ts";

type Mode = "find" | "write";

/** Starts the Model placeholder types out, one after another (a single static one with reduced motion). */
export const EXAMPLE_STARTS = ["In the beginning", "O Arjuna,", "And when they came", "Blessed are", "Say: He is"];
const FIND_PLACEHOLDER = "A worry or a topic: money, grief, a hard decision";

export function mountPrompt(form: HTMLFormElement, results: HTMLElement, gen: HTMLElement,
  ctl: AdviceController,
  mountGen: (onPhase: (p: GenPhase, label: string) => void, onRate: (text: string) => void) => Generator,
  /** The crisis note for this text, or null: the search's own suggest(), run before the model writes. */
  crisisNote: (text: string) => Promise<string | null>): void {
  const input = form.querySelector("textarea")!;
  const composer = form.querySelector<HTMLElement>(".composer")!;
  const counter = form.querySelector<HTMLElement>("[data-counter]")!;
  const liveRate = form.querySelector<HTMLElement>("[data-live-rate]")!;
  const submit = form.querySelector<HTMLButtonElement>("button[type=submit]")!;
  const modes = form.querySelectorAll<HTMLButtonElement>("[data-mode]");
  // The book choice and temperature sit under the box in Model mode; "Run it live" is in the write-up's Model section.
  const books = form.querySelectorAll<HTMLButtonElement>("[data-gen-book]");
  const examples = form.querySelectorAll<HTMLButtonElement>("[data-gen-example]");
  const temp = form.querySelector<HTMLInputElement>("[data-gen-temp]")!;
  const tempOut = form.querySelector<HTMLOutputElement>("[data-gen-temp-out]")!;
  const hint = gen.querySelector<HTMLElement>("[data-gen-hint]");
  const runLive = document.querySelector<HTMLButtonElement>("[data-gen-run]");
  temp.addEventListener("input", () => (tempOut.value = Number(temp.value).toFixed(1)));
  input.maxLength = MAX_INPUT_CHARS;

  let mode = (form.dataset.mode === "find" ? "find" : "write") as Mode;
  let book = ([...books].find((b) => b.getAttribute("aria-pressed") === "true")?.dataset.genBook ?? "bible") as BookId;
  let phase: GenPhase = "idle";
  let writeLabel = "Continue";
  let checking = false;
  let genStarted = false;
  const generator = mountGen((p, label) => {
    phase = p;
    writeLabel = label;
    books.forEach((b) => (b.disabled = p !== "idle"));
    examples.forEach((b) => (b.disabled = p !== "idle"));
    temp.disabled = p !== "idle";
    if (p !== "writing") liveRate.textContent = "";
    update();
  }, (text) => (liveRate.textContent = phase === "writing" ? text : ""));

  function update() {
    counter.textContent = `${input.value.length} / ${mode === "find" ? MAX_INPUT_CHARS : MAX_START_CHARS}`;
    const label = mode === "find" ? "Find" : writeLabel;
    submit.setAttribute("aria-label", label);
    submit.textContent = mode === "write" && phase === "idle" && writeLabel === "Continue again" ? "Continue" : label;
    submit.disabled = mode === "find" ? input.value.trim().length === 0 : phase === "loading";
    composer.classList.toggle("is-busy", mode === "write" && phase !== "idle");
  }

  // ----- the Model placeholder: example starts typed out one character at a time, with a blinking caret
  let typer: number | undefined;
  const stopTyping = () => { clearTimeout(typer); typer = undefined; };
  function typeExamples() {
    stopTyping();
    if (mode !== "write") return;
    if (prefersReducedMotion()) { input.placeholder = EXAMPLE_STARTS[0]; return; }
    let i = 0, n = 0, blink = 0;
    const tick = () => {
      if (mode !== "write" || document.activeElement === input || input.value) {
        input.placeholder = EXAMPLE_STARTS[i];
        typer = window.setTimeout(tick, 400); // paused while the box has focus or text
        return;
      }
      const s = EXAMPLE_STARTS[i];
      if (n < s.length) {
        n++;
        input.placeholder = s.slice(0, n) + "|";
        typer = window.setTimeout(tick, 70);
      } else if (blink < 6) {
        blink++;
        input.placeholder = s + (blink % 2 ? "" : "|");
        typer = window.setTimeout(tick, 450);
      } else {
        i = (i + 1) % EXAMPLE_STARTS.length;
        n = 0;
        blink = 0;
        input.placeholder = "|";
        typer = window.setTimeout(tick, 300);
      }
    };
    tick();
  }

  function applyMode() {
    form.dataset.mode = mode;
    modes.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    form.querySelectorAll<HTMLElement>("[data-write-only], [data-note-write]").forEach((el) => (el.hidden = mode !== "write"));
    form.querySelector<HTMLElement>("[data-note-find]")!.hidden = mode !== "find";
    input.setAttribute("aria-describedby", mode === "find" ? "ask-note" : "ask-note-write");
    results.hidden = mode !== "find";
    gen.hidden = mode !== "write" || !genStarted;
    if (mode === "find") {
      stopTyping();
      input.placeholder = FIND_PLACEHOLDER;
    } else {
      typeExamples();
    }
    update();
  }
  function setMode(next: Mode) {
    if (next === mode) return;
    mode = next;
    applyMode();
  }

  const setBook = (next: BookId) => {
    book = next;
    books.forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.genBook === next)));
  };
  modes.forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode as Mode)));
  books.forEach((b) => b.addEventListener("click", () => setBook(b.dataset.genBook as BookId)));
  examples.forEach((b) => b.addEventListener("click", () => {
    if (phase !== "idle") return;
    const text = b.dataset.genExample ?? "";
    const targetBook = b.dataset.genExampleBook as BookId | undefined;
    if (targetBook) setBook(targetBook);
    input.value = text;
    update();
    void write(text);
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
  /** Write with the chosen book and temperature, starting from `text`, after the crisis check. */
  const write = async (text: string) => {
    if (phase === "writing") return generator.stop();
    if (checking || phase !== "idle") return;
    checking = true;
    try {
      const note = text ? await crisisNote(text) : null;
      genStarted = true;
      gen.hidden = false;
      if (hint) {
        if (note === null && looksLikeQuestion(text)) {
          hint.replaceChildren(
            "This is a 10M-parameter autocomplete model, not a chatbot—it only continues your words in scripture style. Looking for real verses? ",
            h("button", {
              type: "button",
              class: "textlink",
              onclick: () => {
                generator.stop();
                setMode("find");
                void ctl.submit(input.value);
              },
            }, "Switch to Verses"),
            ".",
          );
          hint.hidden = false;
        } else {
          hint.hidden = true;
          hint.replaceChildren();
        }
      }
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
  // "Run it live" switches to Model mode and starts from the typed text there (a Verses query is not a start).
  runLive?.addEventListener("click", () => {
    const start = mode === "write" ? input.value.trim() : "";
    setMode("write");
    void write(start);
  });
  applyMode();
}
