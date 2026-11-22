import { h, prefersReducedMotion } from "../dom.ts";
import { BOOKS, BOOK_NAMES, MAX_INPUT_CHARS, type BookId, type TopicLabel, type TopicListing, type TopicVerse } from "./api.ts";
import type { AdviceController, AdviceState } from "./state.ts";

export interface ResultHandlers {
  openTopic(id: string): void;
  browseAll(): void;
  back(): void;
}

const TRANSLATION: Record<BookId, string> = { gita: "Besant, 1922", quran: "Pickthall, 1930", bible: "King James Version" };
const SHORT: Record<BookId, string> = { gita: "Gita", quran: "Quran", bible: "Bible" };

/** A topic button. In the full list, three dots show which books have verses for it. */
function topicList(topics: (TopicLabel | TopicListing)[], on: ResultHandlers): HTMLElement {
  return h("ul", { class: "topic-list", role: "list" },
    topics.map((t) => {
      const counts = "verses" in t ? t.verses : null;
      return h("li", {},
        h("button", { type: "button", class: "topic", "data-topic": t.id, onclick: () => on.openTopic(t.id) },
          h("span", { class: "topic__label" }, t.label),
          h("span", { class: "topic__blurb" }, t.blurb),
          counts ? h("span", { class: "topic__books" },
            BOOKS.map((b) => h("span", { class: `topic__book book--${b}${counts[b] ? "" : " is-empty"}` }, SHORT[b])),
            h("span", { class: "visually-hidden" }, `: ${BOOKS.map((b) => `${counts[b]} ${SHORT[b]}`).join(", ")} verses`)) : null,
          h("span", { class: "topic__arrow", "aria-hidden": "true" }, "→")));
    }));
}

const browseLink = (on: ResultHandlers, text = "Browse all topics") =>
  h("button", { type: "button", class: "link-button", "data-browse": "", onclick: () => on.browseAll() }, text);

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
      } }, `Show ${rest.length} more from the ${BOOK_NAMES[b]}`)
    : null;
  return h("section", { class: `book book--${b}`, "aria-label": BOOK_NAMES[b], "data-book": b },
    h("h4", { class: "book__name" }, BOOK_NAMES[b], h("span", { class: "book__tr" }, TRANSLATION[b])),
    verses.slice(0, first).map((v) => verseFigure(b, v)),
    rest.length ? more : null,
    button);
}

/** The results area for one state. Pure apart from the handlers it wires up; tests render it directly. */
export function renderResults(state: AdviceState, on: ResultHandlers): Node[] {
  switch (state.kind) {
    case "idle":
      return [h("p", { class: "results__hint" }, "Nothing is sent anywhere: the search runs in your browser. ", browseLink(on))];
    case "suggesting":
      return [h("p", { class: "results__status" }, h("span", { class: "dots", "aria-hidden": "true" }), "Looking for topics…")];
    case "crisis":
      return [h("div", { class: "crisis", "data-crisis": "" }, h("p", {}, state.note))];
    case "suggestions":
      if (state.topics.length === 0) {
        return [h("p", { class: "results__hint" }, "I have no topic to suggest for this. ", browseLink(on))];
      }
      return [
        h("h3", { class: "results__title", tabindex: "-1", "data-focus": "" }, "Topics that may fit"),
        h("p", { class: "results__hint" }, "Pick one to read verses on it from each book."),
        topicList(state.topics, on),
        h("p", { class: "results__more" }, "Not quite it? ", browseLink(on)),
      ];
    case "browse":
      return [
        h("h3", { class: "results__title", tabindex: "-1", "data-focus": "" },
          state.reason === "no-match" ? "I couldn’t match that to a topic" : "All topics"),
        h("p", { class: "results__hint" }, state.reason === "no-match"
          ? "Here is every topic instead. Pick the one closest to what you mean; a dimmed book has no verse for it."
          : "Every topic with verses in the reviewed selection. A dimmed book has none for that topic."),
        topicList(state.topics, on),
      ];
    case "loadingTopic":
      return [
        h("button", { type: "button", class: "link-button back", onclick: () => on.back() }, "← Topics"),
        h("h3", { class: "results__title" }, state.topic.label),
        h("p", { class: "results__status" }, h("span", { class: "dots", "aria-hidden": "true" }), "Loading verses…"),
      ];
    case "topic": {
      const r = state.result;
      const full = BOOKS.filter((b) => r.books[b].verses.length > 0);
      const empty = BOOKS.filter((b) => r.books[b].verses.length === 0);
      return [
        h("button", { type: "button", class: "link-button back", onclick: () => on.back() }, "← Topics"),
        h("h3", { class: "results__title", tabindex: "-1", "data-focus": "" }, r.label),
        h("p", { class: "results__hint" }, r.blurb),
        r.note ? h("p", { class: "topic-note" }, r.note) : null,
        h("div", { class: `books books--${full.length}` }, full.map((b) => bookSection(b, r.books[b].verses))),
        empty.length ? h("div", { class: "books-none" },
          empty.map((b) => h("p", { class: `book-none book--${b}`, "data-empty": b }, r.books[b].nothing))) : null,
      ].filter((n) => n !== null);
    }
    case "error":
      return [h("p", { class: "results__error", role: "alert" }, state.message)];
  }
}

/** Wire the static form in index.html to the controller. */
export function mountAdvice(form: HTMLFormElement, results: HTMLElement, ctl: AdviceController): void {
  const input = form.querySelector("textarea")!;
  const counter = form.querySelector<HTMLElement>("[data-counter]")!;
  const submit = form.querySelector<HTMLButtonElement>("button[type=submit]")!;
  input.maxLength = MAX_INPUT_CHARS;

  const updateCount = () => {
    counter.textContent = `${input.value.length} / ${MAX_INPUT_CHARS}`;
    submit.disabled = input.value.trim().length === 0;
  };
  input.addEventListener("focus", () => ctl.warm(), { once: true });
  input.addEventListener("input", updateCount);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    void ctl.submit(input.value);
  });
  updateCount();

  const handlers: ResultHandlers = {
    openTopic: (id) => void ctl.openTopic(id),
    browseAll: () => void ctl.browseAll(),
    back: () => ctl.back(),
  };
  let prev: AdviceState["kind"] = "idle";
  const render = (s: AdviceState) => {
    results.replaceChildren(...renderResults(s, handlers));
    results.dataset.state = s.kind;
    // Move focus only when the reader navigated (opened a topic, went back, asked for the list); new
    // suggestions are announced by the live region instead, so typing is never interrupted.
    const navigated = s.kind === "topic" || (s.kind === "browse" && s.reason === "asked") ||
      ((s.kind === "suggestions" || s.kind === "browse") && (prev === "topic" || prev === "loadingTopic"));
    if (navigated) {
      results.querySelector<HTMLElement>("[data-focus]")?.focus({ preventScroll: true });
      results.scrollIntoView?.({ block: "nearest", behavior: prefersReducedMotion() ? "auto" : "smooth" });
    }
    prev = s.kind;
  };
  ctl.subscribe(render);
  render(ctl.state);
}
