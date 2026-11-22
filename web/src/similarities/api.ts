// The contract for the side-by-side parallels. The shape follows search/artifacts/similarities.json in PR 4
// (themes -> pairs -> two verses and a one-line note). Until PR 4 lands, the data is the MOCK in ./mock.ts:
// replace the import below with that file (or a fetch of it) and set isMock to false.

import type { BookId } from "../advice/api.ts";
import { MOCK_SIMILARITIES } from "./mock.ts";

export interface PairVerse {
  book: BookId;
  ref: string;
  text: string;
}

export interface Pair {
  /** One line, written by the reviewer, on what the two passages share. */
  idea: string;
  /** "model, reviewed" or "hand-listed". */
  source: string;
  verses: [PairVerse, PairVerse];
}

export interface SimilarityTheme {
  name: string;
  pairs: Pair[];
}

export interface Similarities {
  isMock: boolean;
  themes: SimilarityTheme[];
}

export async function loadSimilarities(): Promise<Similarities> {
  return { isMock: true, themes: MOCK_SIMILARITIES };
}
