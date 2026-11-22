// The reviewed cross-book parallels: search/artifacts/similarities.json, copied into src/generated/ by
// scripts/extract.ts and loaded after the page (about 23 kB).

import type { BookId } from "../advice/api.ts";

export interface PairVerse {
  book: BookId;
  ref: string;
  text: string;
}

export interface Pair {
  /** One line, written by the reviewer, on what the two passages share. */
  idea: string;
  /** "model, reviewed": proposed by the word statistics, kept after review; "hand-listed": from a list of known parallels. */
  source: "model, reviewed" | "hand-listed";
  /** The statistics' similarity for the pair, as recorded in search/similarities_review.json. */
  score: number;
  verses: [PairVerse, PairVerse];
}

export interface SimilarityTheme {
  name: string;
  pairs: Pair[];
}

export interface Similarities {
  candidates_proposed: number;
  kept: number;
  kept_fraction: number;
  hand_listed_kept: number;
  hand_listed_left_out: number;
  themes: SimilarityTheme[];
}

export async function loadSimilarities(): Promise<Similarities> {
  return (await import("../generated/similarities.json")).default as unknown as Similarities;
}
