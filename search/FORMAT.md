# Meaning-search artefacts

`python search/build.py` writes these files to `search/artifacts/`. They hold
everything a browser needs to run the search; the verse texts come from
`data/processed/*.jsonl`. All binary files are little-endian, with no header.

Verses are numbered in one sequence: the Gita's 701 verses first, then the
Quran's 6,236, then the Bible's 31,102, each in the order of its `.jsonl` file.
`N` = 38,039 verses, `D` = vector size (`model.json` → `dim`), `T` = number of
terms, `V` = number of terms that have a vector.

| File | Contents |
|---|---|
| `model.json` | Vocabulary, counts, settings and verse refs (below). |
| `word_vectors.i8` | `int8[V][D]`. Row `t` is the vector of term `t`. Only the direction matters: divide each row by its length after loading. Terms `0 .. V-1` have vectors; terms `V .. T-1` (rare words) do not. |
| `verse_vectors.i8` | `int8[N][D]`. Row `i` is the meaning vector of verse `i`. Divide by its length after loading. |
| `verse_terms.u16` | `uint16[]`: the term ids of every verse, verse after verse, filler words removed. |
| `verse_offsets.u32` | `uint32[N + 1]`: verse `i`'s term ids are `verse_terms[offsets[i] : offsets[i + 1]]`. |
| `similarities.json` | Strongest matches across books, grouped by theme (made for display). |

`V` is the number of rows of `word_vectors.i8` (file size / `D`).

Vectors are quantised per row: `q = round(x / max(abs(x)) * 127)`. The scale is
not stored because only directions are used (every score is a cosine).

## model.json

```
{
  "version": 1,
  "dim": D,
  "method": "ppmi",                 # how the word vectors were trained
  "params": {                       # scoring settings, see scripture/search.py
    "sif_a", "bm25_k1", "bm25_b", "alpha", "bridge_weight", "similar_vector_weight"
  },
  "words": [...],                   # every distinct lower-case word in the three books, sorted
  "word_term": [...],               # term id for each entry of "words", or -1 for a filler word
  "terms": [...],                   # term strings, most frequent first (each is a word of the books)
  "count": [...],                   # how often each term occurs in the three books
  "principal": [D floats],          # shared direction removed from verse and query vectors
  "books": {"gita": {"refs": [...]}, "quran": {...}, "bible": {...}}
}
```

## Scoring a query (what a port must reproduce)

1. **Words.** Lower-case; spell out `won't`, `can't`, `ain't`, `n't`; drop
   `'m 're 's 've 'll 'd`; keep runs of `a-z`.
2. **Bridge.** Walk the words left to right. At each position try the longest
   phrase (up to the longest key) in `search/bridge.json`; a single word may
   also match through `candidates()` (e.g. "worries" → "worry"). Add each
   bridge word's term with weight `bridge_weight`. If a phrase of two or more
   words matched, skip its words. If a single word matched and it is listed in
   `_replace`, skip it too.
3. **Own words.** Every other word that is not a stopword is looked up in
   `words`/`word_term`; unknown words go through `normalise()` first. It gets
   weight 1. A term keeps its highest weight.
4. **BM25,** per book: `sum_t w_t * idf_t * tf (k1 + 1) / (tf + k1 (1 - b + b len / avglen))`
   with `idf_t = ln(1 + (n - df + 0.5) / (df + 0.5))`, where `n`, `df`, `len`
   and `avglen` are counted inside that book from `verse_terms`.
5. **Vector:** `q = sum_t w_t * a / (a + p_t) * unit(wordvec_t)` over terms that
   have vectors (`p_t = count_t / sum(count)`, `a = sif_a`), minus its
   projection on `principal`, scaled to length 1; the score is `q · unit(versevec)`.
6. **Hybrid,** per book: `alpha * bm25 / max(bm25) + (1 - alpha) * cos / max(cos)`.
   Return the top k of each book.

`similar(book, ref)` scores `similar_vector_weight * cos(verse vectors) + (1 -
similar_vector_weight) * cos(TF-IDF)`, with TF-IDF weights
`ln(1 + N / (1 + df_all))` counted over all three books.
