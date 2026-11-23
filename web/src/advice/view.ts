import { h, prefersReducedMotion } from "../dom.ts";
import { BOOKS, BOOK_NAMES, type BookId, type TopicLabel, type TopicResult, type TopicVerse } from "./api.ts";
import type { AdviceController, AdviceState } from "./state.ts";

export interface ResultHandlers {
  openTopic(id: string): void;
  browseAll(): void;
  back(): void;
}

const TRANSLATION: Record<BookId, string> = { gita: "Besant, 1922", quran: "Pickthall, 1930", bible: "King James Version" };

function chips(topics: TopicLabel[], current: string | null, on: ResultHandlers): HTMLElement {
  return h("ul", { class: "chips", role: "list" },
    topics.map((t) => h("li", {},
      h("button", { type: "button", class: "chip", "data-topic": t.id, title: t.blurb, "aria-pressed": String(t.id === current),
        onclick: () => on.openTopic(t.id) }, t.label))));
}

const quiet = (text: string, onclick: () => void, attrs: Record<string, string> = {}) =>
  h("button", { type: "button", class: "link-button", onclick, ...attrs }, text);

const refLabel = (b: BookId, v: TopicVerse) => (b === "bible" ? v.ref : `${BOOK_NAMES[b]} ${v.ref}`);

function verseFigure(b: BookId, v: TopicVerse): HTMLElement {
  return h("figure", { class: "verse", "data-verse": "" },
    h("blockquote", { class: "verse__text" }, v.text),
    h("figcaption", {}, h("span", { class: "verse__ref" }, refLabel(b, v)), h("span", { class: "verse__why" }, v.why)));
}

/** One book's verses: the hand-picked first ones (at least three), the rest behind a button. */
function bookSection(b: BookId, verses: TopicVerse[]): HTMLElement {
  const first = Math.max(3, verses.filter((v) => v.priority !== null).length);
  const rest = verses.slice(first);
  const more = h("div", { class: "book__more", hidden: true }, rest.map((v) => verseFigure(b, v)));
  const button = rest.length
    ? h("button", { type: "button", class: "link-button book__toggle",
      onclick: (e: Event) => {
        more.hidden = false;
        (e.currentTarget as HTMLElement).remove();
      } }, `${rest.length} more`)
    : null;
  return h("section", { class: `book book--${b}`, "aria-label": BOOK_NAMES[b], "data-book": b },
    h("h3", { class: "book__name" }, BOOK_NAMES[b], h("span", { class: "book__tr" }, TRANSLATION[b])),
    verses.slice(0, first).map((v) => verseFigure(b, v)),
    rest.length ? more : null,
    button);
}

function verses(r: TopicResult): Node[] {
  const full = BOOKS.filter((b) => r.books[b].verses.length > 0);
  const empty = BOOKS.filter((b) => r.books[b].verses.length === 0);
  return [
    r.note ? h("p", { class: "topic-note" }, r.note) : null,
    h("div", { class: `books books--${full.length}` }, full.map((b) => bookSection(b, r.books[b].verses))),
    empty.length ? h("div", { class: "books-none" },
      empty.map((b) => h("p", { class: `book-none book--${b}`, "data-empty": b }, r.books[b].nothing))) : null,
  ].filter((n) => n !== null);
}

const status = (text: string) => h("p", { class: "results__status" }, h("span", { class: "dots", "aria-hidden": "true" }), text);

/** The results area for one state. Pure apart from the handlers it wires up; tests render it directly. */
export function renderResults(state: AdviceState, on: ResultHandlers): Node[] {
  switch (state.kind) {
    case "idle":
      return [];
    case "suggesting":
      return [status("Searching…")];
    case "crisis":
      return [h("div", { class: "crisis", "data-crisis": "" }, h("p", {}, state.note))];
    case "answer": {
      // One compact row above the first verse: the topic, the other suggested topics, and the full list.
      const others = state.topics.length > 1 ? chips(state.topics, state.current.id, on) : null;
      return [
        h("div", { class: "answer__bar" },
          h("h2", { class: "answer__title", tabindex: "-1", "data-focus": "", title: state.current.blurb },
            h("span", { class: "answer__kicker" }, "Topic"), state.current.label),
          others, quiet("All topics", on.browseAll, { "data-browse": "" })),
        ...(state.result ? verses(state.result) : [status("Loading verses…")]),
      ];
    }
    case "browse":
      return [
        h("p", { class: "answer__line", tabindex: "-1", "data-focus": "" },
          state.reason === "no-match" ? "Nothing matched. Pick a topic:" : "All topics:"),
        chips(state.topics, null, on),
        state.back ? quiet("← Back", on.back) : null,
      ].filter((n) => n !== null);
    case "error":
      return [h("p", { class: "results__error", role: "alert" }, state.message)];
  }
}

/** Draw the controller's state into the results area. */
export function mountResults(results: HTMLElement, ctl: AdviceController): void {
  const handlers: ResultHandlers = {
    openTopic: (id) => void ctl.openTopic(id),
    browseAll: () => void ctl.browseAll(),
    back: () => ctl.back(),
  };
  let prev: AdviceState["kind"] = "idle";
  const render = (s: AdviceState) => {
    results.replaceChildren(...renderResults(s, handlers));
    results.dataset.state = s.kind;
    // Replay the reveal only when a new answer appears, not while a topic's verses load.
    const fresh = s.kind !== prev || (s.kind === "answer" && s.result !== null);
    if (fresh && !prefersReducedMotion()) {
      results.classList.remove("is-new");
      void results.offsetWidth;
      results.classList.add("is-new");
    }
    // A fresh answer is scrolled into view, but focus stays in the text box so a follow-up can be typed; the
    // live region announces it. Focus moves only when the reader navigated within the answer.
    if (prev === "suggesting" && s.kind !== "suggesting") {
      results.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" });
    }
    if ((s.kind === "browse" && s.reason === "asked") || (s.kind === "answer" && prev === "browse")) {
      results.querySelector<HTMLElement>("[data-focus]")?.focus({ preventScroll: true });
    }
    prev = s.kind;
  };
  ctl.subscribe(render);
  render(ctl.state);
}
