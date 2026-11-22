// The advice section's state machine. It has no DOM code, so it can be tested on its own.
//
//   idle --submit--> suggesting --> crisis            (note only; nothing else is reachable)
//                               \-> suggestions --pick--> loadingTopic --> topic
//                               \-> browse (nothing matched: every topic) --pick--> ...
//   any non-busy state --browseAll--> browse;  topic --back--> where it came from
//
// Verses exist only in the topic state, which is reached only by picking a topic from a list. Every request
// carries a number; a reply that arrives after a newer request started is dropped, so a slow topic can never
// appear on top of a crisis note.

import { MAX_INPUT_CHARS, type AdviceApi, type TopicLabel, type TopicListing, type TopicResult } from "./api.ts";

export type AdviceState =
  | { kind: "idle" }
  | { kind: "suggesting"; query: string }
  | { kind: "crisis"; note: string }
  /** sensitive: the topics carry gentle labels and open with topic(id, {sensitive: true}). */
  | { kind: "suggestions"; query: string; topics: TopicLabel[]; sensitive: boolean }
  | { kind: "browse"; reason: "no-match" | "asked"; topics: TopicListing[] }
  | { kind: "loadingTopic"; topic: TopicLabel; back: ListState }
  | { kind: "topic"; result: TopicResult; back: ListState }
  | { kind: "error"; message: string };

type ListState = Extract<AdviceState, { kind: "suggestions" | "browse" }>;

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

  async submit(text: string): Promise<void> {
    const query = text.trim().slice(0, MAX_INPUT_CHARS);
    if (!query) return;
    await this.run({ kind: "suggesting", query }, async () => {
      const s = await this.api.suggest(query);
      return { s, all: s.category === "none" ? await this.api.topics() : [] };
    }, ({ s, all }): AdviceState => {
      switch (s.category) {
        case "crisis":
          return { kind: "crisis", note: s.note ?? "" };
        case "none":
          return { kind: "browse", reason: "no-match", topics: all };
        default:
          return { kind: "suggestions", query, topics: s.topics.slice(0, 4), sensitive: s.category === "sensitive" };
      }
    });
  }

  async browseAll(): Promise<void> {
    if (this.s.kind === "suggesting" || this.s.kind === "loadingTopic") return;
    await this.run(null, () => this.api.topics(), (topics): AdviceState => ({ kind: "browse", reason: "asked", topics }));
  }

  async openTopic(id: string): Promise<void> {
    const s = this.s;
    if (s.kind !== "suggestions" && s.kind !== "browse") return; // topics are only reachable from a list
    const topic = s.topics.find((t) => t.id === id);
    if (!topic) return;
    const sensitive = s.kind === "suggestions" && s.sensitive;
    await this.run({ kind: "loadingTopic", topic, back: s }, () => this.api.topic(id, { sensitive }),
      (result): AdviceState => ({ kind: "topic", result, back: s }));
  }

  back(): void {
    const s = this.s;
    if (s.kind === "topic" || s.kind === "loadingTopic") {
      this.request++;
      this.set(s.back);
    }
  }

  reset(): void {
    this.request++;
    this.set({ kind: "idle" });
  }
}
