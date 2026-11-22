// The contract between the advice UI and the verse search.
//
// The real search (PR 4, search/FORMAT.md) is a port of a plain-Python scorer over a hand-reviewed pool of
// verses. Until it lands, `adviceApi` is the MOCK in ./mock.ts. To switch, implement AdviceApi with the port
// and change the one line at the bottom of this file; the UI only ever talks to this interface.

import { mockAdviceApi } from "./mock.ts";

export type BookId = "gita" | "quran" | "bible";
export const BOOKS: readonly BookId[] = ["gita", "quran", "bible"];
export const BOOK_NAMES: Record<BookId, string> = { gita: "Bhagavad Gita", quran: "Quran", bible: "Bible" };

/** Longest input the search accepts (FORMAT.md: max_query_chars). */
export const MAX_INPUT_CHARS = 500;

export interface TopicSummary {
  id: string;
  label: string;
  blurb: string;
}

export interface Verse {
  ref: string;
  text: string;
  /** One line on why this verse is listed under the topic, written by a reviewer. */
  why: string;
}

/**
 * What the search makes of a message. Typing never shows verses: a message only ever leads to topics.
 * - crisis: show the note and nothing else (no verses, no topics, no helpline numbers; the owner's decision).
 * - topics: one to four suggested topics, best first.
 * - none: nothing matched; the UI offers the full topic list.
 */
export type Suggestion =
  | { category: "crisis"; note: string }
  | { category: "topics"; topics: TopicSummary[] }
  | { category: "none" };

export interface TopicVerses extends TopicSummary {
  /** Verses per book. An empty list means the book has nothing for this topic, and the UI says so. */
  books: Record<BookId, Verse[]>;
}

export interface AdviceApi {
  /** True while the mock is in use; the UI shows a "preview data" label. */
  readonly isMock: boolean;
  suggest(text: string): Promise<Suggestion>;
  topic(id: string): Promise<TopicVerses>;
  allTopics(): Promise<TopicSummary[]>;
}

export const adviceApi: AdviceApi = mockAdviceApi;
