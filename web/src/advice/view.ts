import { h } from "../dom.ts";
import { BOOKS, BOOK_NAMES, type BookId, type TopicLabel, type TopicResult, type TopicVerse } from "./api.ts";
import type { AdviceController, AdviceState } from "./state.ts";

export interface ResultHandlers {
  openTopic(id: string): void;
  browseAll(): void;
  back(): void;
}

const TRANSLATION: Record<BookId, string> = { gita: "Besant, 1922", quran: "Pickthall, 1930", bible: "King James Version" };

/** Topics as a plain inline list of text buttons; the open one is marked. */
function topicLinks(topics: TopicLabel[], current: string | null, on: ResultHandlers): HTMLElement {
  return h("ul", { class: "inline", role: "list" },
    topics.map((t) => h("li", {},
      h("button", { type: "button", class: "textlink", "data-topic": t.id, title: t.blurb, "aria-pressed": String(t.id === current),
        onclick: () => on.openTopic(t.id) }, t.label))));
}

const textButton = (text: string, onclick: () => void, attrs: Record<string, string> = {}) =>
  h("button", { type: "button", class: "textlink", onclick, ...attrs }, text);

const refLabel = (b: BookId, v: TopicVerse) => (b === "bible" ? v.ref : `${BOOK_NAMES[b]} ${v.ref}`);

function verseItem(b: BookId, v: TopicVerse): HTMLElement {
  return h("figure", { class: "verse", "data-verse": "" },
    h("blockquote", {}, v.text),
    h("figcaption", {}, h("span", { class: "ref" }, refLabel(b, v)), " ", v.why));
}

/** One book's verses: the manually prioritised first ones (at least three), the rest behind a button. */
function bookSection(b: BookId, verses: TopicVerse[]): HTMLElement {
  const first = Math.max(3, verses.filter((v) => v.priority !== null).length);
  const rest = verses.slice(first);
  const more = h("div", { class: "book__more", hidden: true }, rest.map((v) => verseItem(b, v)));
  const button = rest.length
    ? h("button", { type: "button", class: "textlink book__toggle",
      onclick: (e: Event) => {
        more.hidden = false;
        (e.currentTarget as HTMLElement).remove();
      } }, `${rest.length} more`)
    : null;
  return h("section", { class: "book", "aria-label": BOOK_NAMES[b], "data-book": b },
    h("h3", {}, BOOK_NAMES[b], h("span", { class: "muted" }, ` (${TRANSLATION[b]})`)),
    verses.slice(0, first).map((v) => verseItem(b, v)),
    rest.length ? more : null,
    button);
}

function verses(r: TopicResult): Node[] {
  const empty = BOOKS.filter((b) => r.books[b].verses.length === 0);
  return [
    r.note ? h("p", { class: "muted" }, r.note) : null,
    ...BOOKS.filter((b) => r.books[b].verses.length > 0).map((b) => bookSection(b, r.books[b].verses)),
    ...empty.map((b) => h("p", { class: "muted", "data-empty": b }, r.books[b].nothing)),
  ].filter((n) => n !== null);
}

const status = (text: string) => h("p", { class: "muted" }, text);

/** The results area for one state. Pure apart from the handlers it wires up; tests render it directly. */
export function renderResults(state: AdviceState, on: ResultHandlers): Node[] {
  switch (state.kind) {
    case "idle":
      return [];
    case "suggesting":
      return [status("Searching…")];
    case "crisis":
      return [h("p", { class: "crisis", "data-crisis": "" }, state.note)];
    case "choose":
      return [
        h("p", { class: "answer__line", tabindex: "-1", "data-focus": "" }, "Pick a topic:"),
        topicLinks(state.topics, null, on),
        h("p", {}, textButton("All topics", on.browseAll, { "data-browse": "" })),
      ];
    case "answer": {
      return [
        h("p", { class: "answer__bar" },
          "Topic: ", h("strong", { class: "answer__title", tabindex: "-1", "data-focus": "", title: state.current.blurb }, state.current.label),
          ". ", textButton("All topics", on.browseAll, { "data-browse": "" })),
        state.topics.length > 1 ? h("div", { class: "answer__bar" }, "Suggested: ", topicLinks(state.topics, state.current.id, on)) : null,
        ...(state.result ? verses(state.result) : [status("Loading verses…")]),
      ].filter((n) => n !== null);
    }
    case "browse":
      return [
        h("p", { class: "answer__line", tabindex: "-1", "data-focus": "" },
          state.reason === "no-match" ? "Nothing matched. Pick a topic:" : "All topics:"),
        topicLinks(state.topics, null, on),
        state.back ? h("p", {}, textButton("Back", on.back)) : null,
      ].filter((n) => n !== null);
    case "error":
      return [h("p", { role: "alert" }, state.message)];
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
    // A fresh answer is scrolled into view, but focus stays in the text box so a follow-up can be typed; the
    // live region announces it. Focus moves only when the reader navigated within the answer.
    if (prev === "suggesting" && s.kind !== "suggesting") results.scrollIntoView({ block: "start" });
    if ((s.kind === "browse" && s.reason === "asked") || (s.kind === "answer" && prev === "browse")) {
      results.querySelector<HTMLElement>("[data-focus]")?.focus({ preventScroll: true });
    }
    prev = s.kind;
  };
  ctl.subscribe(render);
  render(ctl.state);
}
