import { h, prefersReducedMotion } from "../dom.ts";
import { BOOKS, BOOK_NAMES, MAX_INPUT_CHARS, type TopicSummary } from "./api.ts";
import type { AdviceController, AdviceState } from "./state.ts";

export interface ResultHandlers {
  openTopic(id: string): void;
  browseAll(): void;
  back(): void;
}

const TRANSLATION: Record<string, string> = { gita: "Besant, 1922", quran: "Pickthall, 1930", bible: "King James Version" };

function topicList(topics: TopicSummary[], on: ResultHandlers): HTMLElement {
  return h("ul", { class: "topic-list", role: "list" },
    topics.map((t) =>
      h("li", {},
        h("button", { type: "button", class: "topic", "data-topic": t.id, onclick: () => on.openTopic(t.id) },
          h("span", { class: "topic__label" }, t.label),
          h("span", { class: "topic__blurb" }, t.blurb),
          h("span", { class: "topic__arrow", "aria-hidden": "true" }, "→")))));
}

const browseLink = (on: ResultHandlers) =>
  h("button", { type: "button", class: "link-button", "data-browse": "", onclick: () => on.browseAll() }, "Browse all topics");

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
      return [
        h("h3", { class: "results__title", tabindex: "-1", "data-focus": "" }, "Topics that may fit"),
        topicList(state.topics, on),
        h("p", { class: "results__more" }, "Not quite it? ", browseLink(on)),
      ];
    case "browse":
      return [
        h("h3", { class: "results__title", tabindex: "-1", "data-focus": "" },
          state.reason === "no-match" ? "I couldn’t match that to a topic" : "All topics"),
        ...(state.reason === "no-match" ? [h("p", { class: "results__hint" }, "Here is every topic instead. Pick the one closest to what you mean.")] : []),
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
      return [
        h("button", { type: "button", class: "link-button back", onclick: () => on.back() }, "← Topics"),
        h("h3", { class: "results__title", tabindex: "-1", "data-focus": "" }, r.label),
        h("p", { class: "results__hint" }, r.blurb),
        h("div", { class: "books" },
          BOOKS.map((b) =>
            h("section", { class: `book book--${b}`, "aria-label": BOOK_NAMES[b] },
              h("h4", { class: "book__name" }, BOOK_NAMES[b], h("span", { class: "book__tr" }, TRANSLATION[b])),
              r.books[b].length === 0
                ? h("p", { class: "book__empty", "data-empty": b }, `Nothing in the ${BOOK_NAMES[b]} for this topic in this selection.`)
                : r.books[b].map((v) =>
                  h("figure", { class: "verse", "data-verse": "" },
                    h("blockquote", { class: "verse__text" }, v.text),
                    h("figcaption", {},
                      h("span", { class: "verse__ref" }, b === "bible" ? v.ref : `${BOOK_NAMES[b]} ${v.ref}`),
                      h("span", { class: "verse__why" }, v.why))))))),
      ];
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
  const preview = document.querySelector<HTMLElement>("[data-advice-preview]");
  if (preview) preview.hidden = !ctl.isMock;

  const updateCount = () => {
    counter.textContent = `${input.value.length} / ${MAX_INPUT_CHARS}`;
    submit.disabled = input.value.trim().length === 0;
  };
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
