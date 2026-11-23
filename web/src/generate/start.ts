// The visitor's typed start for "Let the model write": the book tag, then the typed characters the model
// knows, so the model continues the visitor's sentence. Kept apart from the DOM so it can be tested alone.

import type { BookId } from "../advice/api.ts";
import type { CharTokenizer } from "../model/tokenizer.ts";

/** Longest typed start, in characters: with the tag it stays well inside the model's 256-character window. */
export const MAX_START_CHARS = 120;

export interface Start {
  /** Token ids for the model: the book tag, then `text`. */
  ids: number[];
  /** The typed characters that were kept, as the visitor will see them. */
  text: string;
  /** Characters left out because the model has no token for them. */
  dropped: number;
  /** Known characters left out past MAX_START_CHARS. */
  cut: number;
}

export function encodeStart(tok: CharTokenizer, book: BookId, typed: string): Start {
  let text = "";
  let dropped = 0;
  let cut = 0;
  // Line breaks become spaces: a newline ends a verse, and the visitor means one running start.
  for (const ch of typed.replace(/\s+/g, " ").trimStart()) {
    // Only single characters count; "<gita>" typed literally stays text (and "<" is not in the vocabulary).
    if (!tok.tokens.includes(ch) || [...ch].length !== 1 || ch === "\n") dropped++;
    else if (text.length >= MAX_START_CHARS) cut++;
    else text += ch;
  }
  return { ids: [tok.id(`<${book}>`), ...[...text].map((c) => tok.id(c))], text, dropped, cut };
}
