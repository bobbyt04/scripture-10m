# Advice-search artefacts

`python search/build.py --site` writes the files the website loads to
`search/artifacts/`. They hold everything a port of the advice search (for
example to JavaScript in the browser) needs, including the verse texts to
show. The scorer is plain arithmetic in double precision; no vectors are
needed for it.

| File | Size | Contents |
|---|---|---|
| `search.json` | 540 kB | The reviewed verse pool (display text, themes, note, BM25 terms), themes and their phrases, word rules, bridge, safety lists, settings. |
| `pool_vectors.i8` | 123 kB | `int8[P][D]`: one row per published pool verse (Gita, then Quran, then Bible, each in `pool` order), for `similar()`. |
| `similarities.json` | 43 kB | The hand-reviewed cross-book parallels for the similarities page. |
| `fixtures.json` | 81 kB | Expected outputs of 60 queries, 4 theme chips and 3 `similar()` calls, for testing a port. |

Total 787 kB. (The full-corpus model in `search/model/`, 6.6 MB, is used
offline only: to propose parallels, score them, and give pool verses their
vectors. The site never loads it.)

Vectors are quantised per row (`q = round(x / max|x| * 127)`); only the
direction matters. **Divide each row by its length after loading, and leave
an all-zero row as zeros** (its cosine with anything is 0, never NaN).

## search.json

```
{
  "version": 1,
  "dim": D,                                   width of pool_vectors.i8
  "params": {
    "bm25_k1": 1.2, "bm25_b": 0.75,
    "bridge_weight": 0.8,                     weight of a bridge word (a typed word has 1)
    "theme_weight": 1.0, "lex_weight": 0.2,   score = theme_weight * theme part + lex_weight * BM25
    "min_score": 0.3,                         verses scoring below this are not shown
    "sensitive_theme_score": 0.5,             score of a sensitive topic's gentle themes
    "min_theme_verses": 3                     build setting (already applied to "pool")
  },
  "lexicon": {
    "stopwords": [...], "query_stopwords": [...],
    "irregular": {"wives": "wife", ...}, "suffix_rules": [...], "not_inflected": [...]
  },
  "vocabulary": {"word": "term", ...},        every word of the pool documents -> its term ("" = filler word)
  "bridge": {"stressed": ["troubled", ...], "laid off": [...], ...},
  "bridge_replace": ["job", ...],             bridge keys whose own word is not searched
  "themes": {
    "grief": {"name": "Grief and loss", "description": "...", "note": "...",   note is optional
              "phrases": ["passed away", "lost my mom|dad|...", ...]}, ...
  },
  "safety": {
    "fuzzy": ["suicide", "suicidal", "myself", "unalive"],
    "idioms": ["killing it", "cut back", ...],
    "crisis": {"suicide": [phrase, ...], "self_harm": [...], "abuse": [...], "harm_to_others": [...]},
    "sensitive": {"sexuality_gender": {"themes": ["self_worth", ...], "phrases": [...]}, ...}
  },
  "pool": {
    "gita": [{"ref": "2:47", "text": "...", "themes": [...], "why": "...", "sensitive": false,
              "terms": ["business", "action", ...]}, ...],
    "quran": [...], "bible": [...]
  },
  "inputs": {checksums of the source files, pool sizes}
}
```

- `text` is the display text: the source verse with hand-checked typo fixes,
  a leading heading or conjunction removed ("A Psalm of David. ", "And "),
  trailing words removed ("Selah."), the first letter capitalised and a final
  `, ; : -` turned into a full stop. `data/processed` is never changed.
- `themes` lists only themes the book offers (at least 3 verses for them).
- `terms` is the verse's BM25 document, in order: the words of `text`, then
  of `why`, then of its themes' names, each mapped through `vocabulary`,
  with filler words left out. A port does not need to rebuild it.
- `sensitive` marks the hand-picked verses that are the only ones shown for
  sensitive topics (`search/pool/sensitive_set.json`).

## Step by step

All arithmetic in IEEE double precision, in the order given (JavaScript
numbers are doubles; sum in the stated order to match exactly).

### 1. Words: `words(text)`

1. Lower-case; turn `’` into `'`.
2. Replace `won't` → `will not`, `can't` → `can not`, `ain't` → `is not`
   wherever they start a word (regex `\b(won't|can't|ain't)`).
3. Replace every `n't` that ends a word (regex `n't\b`) with ` not`.
4. Replace `'m 're 's 've 'll 'd` that end a word (regex `'(m|re|s|ve|ll|d)\b`)
   with a space. So "I'm" → "i", "friend's" → "friend".
5. The words are the runs of `[a-z]+` (digits and other characters split
   words and are dropped: "70 hour weeks" → hour, weeks).
6. Contractions typed without an apostrophe are expanded: cant → can not,
   dont → do not, didnt, doesnt, isnt, wasnt, arent, werent, havent, hasnt,
   hadnt, couldnt, wouldnt, shouldnt → … not; im, ive → i; youre → you;
   theyre → they; thats → that; whats → what; hes → he; shes → she. ("wont"
   and "ill" are real words and stay.)

### 2. Phrases

A phrase (themes, idioms, crisis, sensitive) is a list of slots separated by
spaces; a slot lists alternatives separated by `|`. An alternative matches a
word if it is equal, or if it ends in `*` and the word starts with what comes
before the `*` (so a lone `*` matches any word). A phrase matches at position
i if slot j matches word i + j for every slot.

### 3. Safety category: `classify(query)`

Let W = words(query).

1. Crisis words: copy W. For every idiom phrase and every position where it
   matches **W**, replace those words in the copy by "" (the empty word).
   Then replace each word of 5 or more letters that is exactly one edit away
   from a `fuzzy` word by that word. One edit: one letter added, removed or
   changed, or two neighbouring letters swapped (a word equal to the fuzzy
   word is left as it is).
2. If any crisis phrase matches the crisis words → `("crisis", topic)`, the
   first topic in file order with a match. Return no verses, no themes.
3. Else if any sensitive phrase matches **W** → `("sensitive", topic)`.
4. Else `("general", null)`.

### 4. Query themes: `query_themes(query)`

Find every (start, end, theme) where a phrase of that theme matches W (end =
start + number of slots); identical triples count once. Drop a match if
another match (of any theme) spans it and is strictly longer:
`s2 <= s and e <= e2 and e2 - s2 > e - s`. Each remaining match adds
`e - s` to its theme. Divide by the largest total, so the top theme scores
1. Order: score descending, then theme id ascending.

For a sensitive query, each of the topic's `themes` not already present is
appended (in list order) with score `sensitive_theme_score`.

`notes` in the result: `{theme, note}` for each query theme (in this order)
whose theme has a `note`.

### 5. Normalising a word: `term_of(w)`

If w is a key of `vocabulary`, its term is `vocabulary[w]`, where "" means
a filler word (no term). Otherwise:

1. Replace w by `irregular[w]` if there is one.
2. Up to two times: take the candidates of w (below); the first candidate
   that is a key of `vocabulary` replaces w; if the new w is a stopword, stop.
3. The term is `vocabulary[w]` if w is a key (and not ""), else none.

Candidates of w: none if w is in `not_inflected`. Otherwise use the **first**
rule of `suffix_rules` with `len(w) >= min_len`, where w ends with one of
`ends` and with none of `not_ends`. For each entry of its `try` list in
order: skip it if it has `only_if_ends` and w ends with none of them; stem =
w without its last `strip` letters; if `undouble` is set, skip unless the
stem's last two letters are equal, then drop the last one; append `add`;
keep the stem if it has at least 3 letters.

### 6. Expanding the query: `expand(query)`

Walk W left to right; Q is an ordered map term → (weight, source), and
`add(term, weight, source)` sets Q[term] only if term is not none and weight
is greater than the weight already there (a new key goes at the end; an
updated key keeps its place).

At position i, find the longest n (from the longest bridge key's word count
down to 1) such that `W[i..i+n-1]` joined by single spaces is exactly a
bridge key.

- If a key matched: for each target t, in order, `add(term_of(t),
  bridge_weight, key)`. Targets are words, so they go through `term_of`.
- Then, if no key matched, or the key is one word and not in
  `bridge_replace`: let w = W[i]; unless w is a stopword or a query stopword,
  `add(term_of(w), 1.0, w)`.
- Move on n words if a key matched, else 1.

### 7. Scores, per book

BM25 statistics come from the book's pool verses' `terms`: n = number of
verses; df_t = number of verses containing t; `idf_t = ln(1 + (n - df_t +
0.5) / (df_t + 0.5))`; len = number of terms of the verse; avglen = mean len.

For each verse v of the book (in pool order):

- If the query is sensitive and v is not `sensitive`, skip v.
- matched themes M = the themes of v (in v's order) that are query themes.
- If the query has any themes and M is empty, skip v (off-topic).
- theme part = sum over M (in that order) of the query's theme score.
- bm25 = sum over Q (in Q's order) of terms t that occur in v of
  `w_t * idf_t * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen))`.
- score = `theme_weight * theme part + lex_weight * bm25`.

Rounding: `r = sign(x) * floor(|x| * 10000 + 0.5) / 10000` (halves away from
zero; JavaScript: `Math.sign(x) * Math.floor(Math.abs(x) * 1e4 + 0.5) / 1e4`).
Sort by rounded score descending, ties by pool order; take the first k; stop
at the first whose rounded score is below `min_score`. A book may return
nothing; `no_match` is true when no book returns anything.

Each hit: ref, text, themes, why, score (rounded), and
`matched = {"themes": M, "words": [{"word": t, "from": source}]}` for the
terms of Q (in Q's order) that occur in v.

`Math.log` and Python's `math.log` can differ in the last bit; scores agree
to about 1e-12, so a different order is possible only for verses whose
scores tie before rounding to within that, which the fixtures do not contain.

### 8. `by_theme(theme, k)`

Per book, the verses whose themes include the theme, scored `1 / (position
of the theme in the verse's list + 1)` and rounded as above; sort by score
descending, then pool order; take k. A book that does not offer the theme
returns nothing.

### 9. `similar(book, ref, k)`

The source must be a pool verse. Per book, every other pool verse (and,
with `sensitive_only`, only sensitive ones) scores the dot product of the
unit-length rows of `pool_vectors.i8` (all-zero rows stay zero, so they score
0). Round, sort by score descending then pool order, take k.

## fixtures.json

- `words`: `[text, words(text)]` pairs for step 1.
- `search`: for each query, `search(query, k=3)`:
  - `words`: words(query); `category`, `topic`, `no_match` as in step 3 and 7;
  - `themes`: `[theme, rounded score]` in order (step 4, with sensitive defaults);
  - `terms`: Q from step 6 as `[term, weight, source]` in Q's order;
  - `results`: per book, `[ref, rounded score, matched themes, matched terms]`.
- `by_theme`: per theme, per book `[ref, rounded score]` (k = 3).
- `similar`: per source verse, per book `[ref, rounded score]` (k = 3).

A port should match every field exactly.
