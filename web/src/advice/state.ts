// The answer area's state machine. It has no DOM code, so it can be tested on its own.
//
//   idle --submit--> suggesting --> crisis   (note only; nothing else is reachable)
//                               \-> answer   (topics matched: the first topic's verses, the others as chips)
//                               \-> browse   (nothing matched: every topic)
//   answer --pick a chip--> answer;  answer/idle --browseAll--> browse --pick--> answer
//
// Verses come only from topic(), for a topic that suggest() returned or that is in the full list. A sensitive
// suggestion opens its topics with {sensitive: true}; a topic picked from the full list never does. Every
// request carries a number; a reply that arrives after a newer request started is dropped, so a slow topic can
// never appear on top of a crisis note.

import { MAX_INPUT_CHARS, type AdviceApi, type TopicLabel, type TopicListing, type TopicResult } from "./api.ts";

export type AdviceState =
  | { kind: "idle" }
  | { kind: "suggesting"; query: string }
  | { kind: "crisis"; note: string }
  /** topics: the suggested topics shown as chips; result is null while `current` loads. */
  | { kind: "answer"; topics: TopicLabel[]; sensitive: boolean; current: TopicLabel; result: TopicResult | null }
  | { kind: "browse"; reason: "no-match" | "asked"; topics: TopicListing[]; back: AnswerState | null }
  | { kind: "error"; message: string };

export type AnswerState = Extract<AdviceState, { kind: "answer" }>;

export class AdviceController {
  private s: AdviceState = { kind: "idle" };
  private request = 0;
  private readonly listeners = new Set<(s: AdviceState) => void>();
  private readonly api: AdviceApi;

  constructor(api: AdviceApi) {
    this.api = api;
  }

  get state(): AdviceState {
    return this.s;
  }

  subscribe(fn: (s: AdviceState) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  /** Start loading the search data before the first question. */
  warm(): void {
    this.api.warm?.();
  }

  private set(next: AdviceState): void {
    this.s = next;
    for (const fn of this.listeners) fn(next);
  }

  /** Run a request; its result is applied only if no newer request has started meanwhile. */
  private async run<T>(start: AdviceState | null, work: () => Promise<T>, done: (r: T) => AdviceState): Promise<void> {
    const id = ++this.request;
    if (start) this.set(start);
    try {
      const r = await work();
      if (id === this.request) this.set(done(r));
    } catch {
      if (id === this.request) this.set({ kind: "error", message: "Something went wrong. Please try again." });
    }
  }

  /** Suggest topics for the text and, when there are some, open the first one straight away. */
  async submit(text: string): Promise<void> {
    const query = text.trim().slice(0, MAX_INPUT_CHARS);
    if (!query) return;
    await this.run({ kind: "suggesting", query }, async () => {
      const s = await this.api.suggest(query);
      if (s.category === "crisis") return { kind: "crisis", note: s.note ?? "" } as const;
      const topics = s.topics.slice(0, 4);
      if (s.category === "none" || topics.length === 0) {
        return { kind: "browse", reason: "no-match", topics: await this.api.topics(), back: null } as const;
      }
      const sensitive = s.category === "sensitive";
      const result = await this.api.topic(topics[0].id, { sensitive });
      return { kind: "answer", topics, sensitive, current: topics[0], result } as const;
    }, (next): AdviceState => next);
  }

  async browseAll(): Promise<void> {
    const s = this.s;
    if (s.kind === "suggesting" || s.kind === "crisis" || s.kind === "browse") return;
    const back = s.kind === "answer" && s.result ? s : null;
    await this.run(null, () => this.api.topics(), (topics): AdviceState => ({ kind: "browse", reason: "asked", topics, back }));
  }

  /** Open a topic: one of the answer's chips, or one from the full list. */
  async openTopic(id: string): Promise<void> {
    const s = this.s;
    let next: AnswerState;
    if (s.kind === "answer") {
      const current = s.topics.find((t) => t.id === id);
      if (!current) return;
      next = { ...s, current, result: null };
    } else if (s.kind === "browse") {
      const current = s.topics.find((t) => t.id === id);
      if (!current) return;
      next = { kind: "answer", topics: [], sensitive: false, current, result: null };
    } else {
      return; // topics are only reachable from a list
    }
    await this.run(next, () => this.api.topic(id, { sensitive: next.sensitive }),
      (result): AdviceState => ({ ...next, result }));
  }

  /** From the full list, return to the answer it was opened from. */
  back(): void {
    const s = this.s;
    if (s.kind === "browse" && s.back) {
      this.request++;
      this.set(s.back);
    }
  }

  reset(): void {
    this.request++;
    this.set({ kind: "idle" });
  }
}
