// The contract between the advice UI and the verse-topic search (search/FORMAT.md steps 8 and 9).
//
// Typing only ever leads to topics: suggest() returns a crisis note, gentle topics for a sensitive message, up
// to four topics, or none. Verses come only from topic(), after the visitor taps one. The search data (about
// 370 kB, 100 kB compressed) is loaded the first time it is needed, not with the page.

import { Searcher, type SearchData, type Suggestion, type TopicListing, type TopicResult } from "../search/searcher.ts";

export { BOOKS, BOOK_NAMES, type BookId, type Suggestion, type TopicLabel, type TopicListing, type TopicResult, type TopicVerse } from "../search/searcher.ts";

/** Longest input the search reads (FORMAT.md: max_query_chars). */
export const MAX_INPUT_CHARS = 500;

export interface AdviceApi {
  suggest(text: string): Promise<Suggestion>;
  topic(id: string, opts: { sensitive: boolean }): Promise<TopicResult>;
  topics(): Promise<TopicListing[]>;
  /** Start loading the search data early (when the visitor starts typing), so the first answer is quick. */
  warm?(): void;
}

let loading: Promise<Searcher> | null = null;

/** Load the search data once; later calls share the same promise. A failed load can be retried. */
export function loadSearcher(): Promise<Searcher> {
  loading ??= import("../generated/search.json")
    .then((data) => new Searcher(data.default as unknown as SearchData))
    .catch((e) => {
      loading = null;
      throw e;
    });
  return loading;
}

export const searchApi: AdviceApi = {
  suggest: async (text) => (await loadSearcher()).suggest(text),
  topic: async (id, { sensitive }) => (await loadSearcher()).topic(id, { sensitive }),
  topics: async () => (await loadSearcher()).topics(),
  warm: () => void loadSearcher().catch(() => {}),
};
