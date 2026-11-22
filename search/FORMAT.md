# Meaning-search artefacts

`python search/build.py` writes these files to `search/artifacts/`. They hold
everything a port of the search (for example to JavaScript in the browser)
needs. The only other input is the verse texts in `data/processed/*.jsonl`,
which are shown to the visitor. All binary files are little-endian and have
no header.

Verses are numbered in one sequence: the Gita's 701 verses first, then the
Quran's 6,236, then the Bible's 31,102, each in the order of its `.jsonl`
file (N = 38,039). A verse's number inside its book is its position in that
book's `refs` list. D is the vector size (`dim`), T the number of terms, and
V the number of terms that have a vector (the rows of `word_vectors.i8`).
Terms 0 to V-1 have vectors; rarer terms V to T-1 do not.

| File | Contents |
|---|---|
| `model.json` | All word rules, the bridge, the safety lists, the vocabulary, settings and refs (below). |
| `word_vectors.i8` | `int8[V][D]`. Row t is the vector of term t. |
| `verse_vectors.i8` | `int8[N][D]`. Row i is the vector of verse i. |
| `verse_terms.u16` | `uint16[]`: the term ids of every verse, one verse after another, filler words removed. |
| `verse_offsets.u32` | `uint32[N + 1]`: verse i's term ids are `verse_terms[offsets[i] : offsets[i + 1]]`. |
| `similarities.json` | The hand-reviewed parallels shown on the similarities page. |
| `fixtures.json` | 30 queries with the expected category, terms and top-3 refs and scores per book, for testing a port. |

Vectors are quantised per row: `q = round(x / max(abs(x)) * 127)`. The scale
is not stored, because every score uses only directions. **Divide every row by
its length after loading.**

The artefacts total 7.55 MB (under the 10 MB budget).

## model.json

```
{
  "version": 2,
  "dim": D,
  "method": "ppmi",                      how the word vectors were trained
  "params": {
    "sif_a": 0.001, "bm25_k1": 1.2, "bm25_b": 0.75,
    "method": "bm25+bridge",             what search() uses by default
    "alpha": 0.5,                        hybrid mix (only for method "hybrid")
    "bridge_weight": 0.8,
    "similar_vector_weight": 0.5
  },
  "lexicon": {
    "stopwords": [...],                  filler words, never terms
    "query_stopwords": [...],            also dropped from what a visitor types
    "irregular": {"hath": "have", ...},  applied before the suffix rules
    "suffix_rules": [...],               see "Normalising a word"
    "not_inflected": [...]               never shortened (includes every always-capitalised name)
  },
  "bridge": {"stressed": ["troubled", ...], "laid off": [...], ...},
  "bridge_replace": ["job", ...],        bridge keys whose own word is not searched
  "safety": {
    "crisis":    {"suicide": [phrase, ...], "self_harm": [...], ...},
    "sensitive": {"sexuality_gender": [...], ...}
  },
  "gentle": {"gita": [i, ...], "quran": [...], "bible": [...]},
                                         verse numbers (inside each book) of the gentle set, in curated order
  "blocked": "base64...",                one bit per verse in the global numbering:
                                         verse i is bit (i % 8) of byte (i // 8), least significant bit first
  "words": [...],                        every distinct lower-case word of the three books, sorted
  "word_term": [...],                    term id of each entry of "words", or -1 for a filler word
  "terms": [...],                        term strings (each a word of the books), most frequent first
  "count": [...],                        occurrences of each term in the three books
  "principal": [D floats],               the shared direction removed from verse and query vectors
  "books": {"gita": {"refs": [...]}, "quran": {...}, "bible": {...}},
  "inputs": {corpus and source-file checksums, verse counts}
}
```

## Step by step

### 1. Words

Lower-case the text and turn `’` into `'`. Replace `won't` → `will not`,
`can't` → `can not`, `ain't` → `is not`, then any other `n't` → ` not`.
Delete `'m 're 's 've 'll 'd`. The words are the runs of `a-z`.

### 2. Safety category

The query is "crisis" if any phrase in `safety.crisis` matches the words,
otherwise "sensitive" if any phrase in `safety.sensitive` matches, otherwise
"general". The topic is the key of the first list that matched (checked in
file order). A phrase is a space-separated list of slots. Each slot is one or
more alternatives separated by `|`, and an alternative ending in `*` matches
any word starting with what comes before the `*`. A phrase matches if its
slots match consecutive words. So `he|she hits|hit me` matches "she hit me".

- crisis: return no verses at all. The site shows help instead.
- sensitive: return only verses of the gentle set (step 7).

### 3. Normalising a word

`term_of(w)`: if w is in `words`, its term is `word_term[...]`, which is -1
for filler words. Otherwise:

1. Replace w by `irregular[w]` if there is one.
2. Up to two times: take the candidates of w (below). The first candidate
   that is in `words` replaces w. If the new w is a stopword, stop.
3. Look the result up in `words`.

Candidates of w: none if w is in `not_inflected`. Otherwise use the **first**
rule in `suffix_rules` with `len(w) >= min_len`, where w ends with one of
`ends` and with none of `not_ends`. For each entry of its `try` list, in
order: skip the entry if it has `only_if_ends` and w does not end with one of
those. Then stem = w without its last `strip` letters. If `undouble` is set,
skip the entry unless the stem's last two letters are equal, and drop the last
one. Append `add`. Keep the stem if it has at least 3 letters.

Examples: "loveth" → love, "sinned" → sin, "blessings" → blessing → bless,
"telling" → tell, which is a stopword, so the word is dropped.

### 4. Expanding the query

Walk the words left to right. At position i, look for the longest phrase
`words[i..i+n]` (n from the longest bridge key down to 1) that is exactly a
key of `bridge`. Each target of a matched key gets weight `bridge_weight`,
remembered with the key as its source.

- If a phrase of two or more words matched, skip all its words.
- If one word matched, also use the word itself (weight 1), unless the key is
  in `bridge_replace`. Then move on one word.
- With no match, use the word itself (weight 1), unless it is a stopword or a
  query stopword.

A term reached more than once keeps its highest weight.

### 5. Scores, per book

Count the BM25 statistics inside the book from `verse_terms`. For each term:
df_t is the number of the book's verses that contain it, and
`idf_t = ln(1 + (n - df_t + 0.5) / (df_t + 0.5))`. For each verse, len is its
number of terms; avglen is the mean of len.

- **bm25(verse)** = `sum over query terms t of w_t * idf_t * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen))`,
  where tf is how often t occurs in the verse.
- **cosine(verse)** = `unit(q) · unit(versevec)`, where q is the sum over
  query terms t < V of `w_t * a / (a + count_t / sum(count)) * unit(wordvec_t)`,
  minus `(q · principal) * principal`, and `a = sif_a`. If q is all zeros,
  every cosine is 0.
- Method "bm25+bridge" (the default): the score is bm25.
- Method "hybrid": the score is `alpha * bm25 / max(bm25) + (1 - alpha) * cosine / max(cosine)`.
  A part whose maximum is not positive is used as it is.

### 6. Picking the results, per book

- Blocked verses (bit set in `blocked`) are never returned.
- For a sensitive query, only gentle verses are allowed. Each gets a small
  boost so that the curated order decides when the scores tie:
  `0.01 * (1 - position / len(gentle[book]))`.
- Round every score to 4 decimals, sort by score (ties: the earlier verse first), take the k best allowed
  verses and drop any whose score is not above 0.
- `no_match` is true when no book returned anything. A query of only filler
  or unknown words returns nothing.

### 7. similar(book, ref)

`similar_vector_weight * cos(verse vectors) + (1 - similar_vector_weight) * cos(TF-IDF)`.
TF-IDF weights a term by its count in the verse times
`ln(1 + N / (1 + df_all))`, where df_all is counted over all three books.
Blocked verses and the verse itself are left out; rounding and ties as in step 6.

## Worked examples

From `fixtures.json` (k = 3; scores rounded):

1. `I got laid off and I'm scared about money` → category `general`.
   Terms: afraid, bread, dread, fear, gold, idle, labour, lucre, mammon, money, need, poverty, provide, rich, scare, silver, sustenance, terror, treasure, tremble, wage, wealth.
   - gita: 11:49 (7.4167); 7:2 (5.5086); 1:33 (5.0791)
   - quran: 23:55 (10.4105); 17:103 (9.4423); 13:21 (8.3068)
   - bible: Ezekiel 28:4 (17.0091); Proverbs 10:15 (16.7233); Matthew 10:9 (16.0505)

2. `with hardship goeth ease` → category `general`.
   Terms: ease, hardship.
   - quran: 94:5 (17.9187); 94:6 (17.9187); 87:8 (8.7808)
   - bible: 2 Corinthians 8:13 (9.3929); Job 7:13 (8.8770); Job 21:23 (8.4148)

3. `I'm gay` → category `sensitive` (`sexuality_gender`).
   Terms: gay.
   - gita: 9:29 (0.0100); 6:30 (0.0097); 12:13 (0.0094)
   - quran: 49:13 (0.0100); 17:70 (0.0097); 93:3 (0.0094)
   - bible: Psalms 139:14 (0.0100); 1 John 4:16 (0.0098); Galatians 3:28 (0.0095)

4. `I want to die` → category `crisis` (`suicide`).
   Terms: comfort, despair, hope, life, mercy, perish, soul.
   No verses in any book.

5. `xyzzy` → category `general`, `no_match`.
   No terms, so no verses.
