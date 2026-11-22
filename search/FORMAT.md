# Advice-search artefacts

`python search/build.py --site` writes the files the website loads to
`search/artifacts/`. They hold everything a port of the advice search (for
example to JavaScript in the browser) needs, including the verse texts to
show. The ranking is plain comparisons and double-precision arithmetic; no
vectors are needed for it.

| File | Size | Contents |
|---|---|---|
| `search.json` | 534 kB | The reviewed verse pool (display text, themes, hand-set priorities, note, BM25 terms), themes and their phrases, word rules, bridge, safety lists (crisis phrases, distress gate, bereavement and idiom blanks, sensitive topics, the crisis message), settings. |
| `pool_vectors.i8` | 109 kB | `int8[P][D]`: one row per published pool verse (Gita, then Quran, then Bible, each in `pool` order), for `similar()`. |
| `similarities.json` | 34 kB | The hand-reviewed cross-book parallels for the similarities page. |
| `fixtures.json` | 151 kB | Expected outputs of 89 searches (several with k other than 3, the distress gate, bereavement, typos and non-ASCII input), 4 theme chips and 3 `similar()` calls with and without `sensitive_only`, for testing a port. |

Total 827 kB. (The full-corpus model in `search/model/`, 6.6 MB, is used
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
    "bridge_weight": 0.8,                     weight of a bridge word (a typed word has 1); BM25 tie break only
    "min_theme_score": 0.5,                   query themes below this (the top one scores 1) choose no verses
    "exclusive_themes": ["grief", "gratitude"],  the first of these that matches is the only theme kept
    "theme_fuzzy_min_len": 6,                 unknown words this long may be read as a theme word one typo away
    "max_query_chars": 500,                   only this much of a query is read
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
    "crisis_message": "...",                  the kind note shown instead of verses (no phone numbers, by the owner's decision)
    "idioms": ["killing it", "cut back", ...],
    "bereavement": ["died by suicide", "lost my|our|his|her * to suicide|...", ...],
    "crisis": {"suicide": [phrase, ...], "self_harm": [...], "abuse": [...], "harm_to_others": [...]},
    "sensitive": {"sexuality_gender": {"themes": ["self_worth", ...], "phrases": [...]}, ...},
    "distress": {"window": 7, "rules": [rule, ...]}
  },
  "pool": {
    "gita": [{"ref": "2:47", "text": "...", "themes": [...], "why": "...", "sensitive": false,
              "priority": {"work_effort": 1, ...},      optional
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
- `priority` maps some of the verse's themes to a rank (1 = show first). A
  reviewer chose the 3 to 5 verses each theme of each book should show first.
- `terms` is the verse's BM25 document, in order: the words of `text`, then
  of `why`, then of its themes' names, each mapped through `vocabulary`,
  with filler words left out. A port does not need to rebuild it.
- `sensitive` marks the hand-picked verses that are the only ones shown for
  sensitive topics (`search/pool/sensitive_set.json`).

## Step by step

All arithmetic in IEEE double precision, in the order given (JavaScript
numbers are doubles; sum in the stated order to match exactly).

### 0. The query

Only the first `max_query_chars` (500) characters of the query are read, in
every step below (`query[:500]` in Python; in JavaScript count UTF-16 code
units the same way only if your strings are UTF-16 and the text has no
characters outside the Basic Multilingual Plane, otherwise slice by code
point: `Array.from(q).slice(0, 500).join("")`). The fixtures include a
query whose crisis words come after character 500 and must not be read.

### 1. Words: `words(text)`

1. Lower-case; turn `’` into `'`.
2. Replace `won't` → `will not`, `can't` → `can not`, `ain't` → `is not`
   wherever they start a word (regex `\b(won't|can't|ain't)`).
3. Replace every `n't` that ends a word (regex `n't\b`) with ` not`.
4. Replace `'m 're 's 've 'll 'd` that end a word (regex `'(m|re|s|ve|ll|d)\b`)
   with a space. So "I'm" → "i", "friend's" → "friend".
5. The words are the runs of `[a-z]+` (digits and other characters split
   words and are dropped: "70 hour weeks" → hour, weeks; "café" → caf).
6. Contractions typed without an apostrophe are expanded: cant → can not,
   dont → do not, didnt, doesnt, isnt, wasnt, arent, werent, havent, hasnt,
   hadnt, couldnt, wouldnt, shouldnt → … not; im, ive → i; youre → you;
   theyre → they; thats → that; whats → what; hes → he; shes → she. ("wont"
   and "ill" are real words and stay.)

**Word boundaries are Unicode.** Python's `\b` treats every letter, digit
and underscore of any script as a word character, so in "éwon't" there is no
word boundary before "won't": step 2 does not apply, step 3 makes it "éwo
not", and the words are `["wo", "not"]` (and "_can't" gives `["ca",
"not"]`). JavaScript's plain `\b` is ASCII-only, so a port must not use it. Use lookarounds with the `u`
flag: `(?<![\p{L}\p{N}_])` for a leading `\b` and `(?![\p{L}\p{N}_])` for
a trailing one. The fixtures' `words` list has non-ASCII cases.

### 2. Phrases

A phrase (themes, idioms, bereavement, crisis, sensitive, and the parts of a
distress rule) is a list of slots separated by spaces; a slot lists
alternatives separated by `|`. An alternative matches a word if it is equal,
or if it ends in `*` and the word starts with what comes before the `*`. A
lone `*` therefore matches any word, but **the empty word "" (a blanked
word, step 3) never matches anything, not even `*`**. A phrase matches at
position i if slot j matches word i + j for every slot.

### 3. Safety category: `classify(query)`

Let W = words(query).

1. Crisis words C: copy W. For every phrase of `idioms` and then of
   `bereavement`, and every position where it matches **W**, replace those
   words in C by "" (the empty word). Then replace each word of 5 or more
   letters that is exactly one edit away from a `fuzzy` word by the first
   such fuzzy word in list order. One edit: one letter added, removed or
   changed, or two neighbouring letters swapped (a word equal to the fuzzy
   word is left as it is). The `bereavement` phrases are a third person's
   death ("my dad passed away", "died by suicide", "lost my brother to an
   overdose"): blanking them sends a bereaved visitor to the grief theme
   instead of the crisis note, while anything else in the message, such as
   "and i want to die too", is still read.
2. If any crisis phrase matches C → `("crisis", topic)`, the first topic in
   file order with a match.
3. Else if any distress rule fires on C → `("crisis", "distress")`.
4. Else if any sensitive phrase matches **W** → `("sensitive", topic)`, the
   first topic in file order.
5. Else `("general", null)`.

**Distress rules.** A rule is a string: groups separated by ` + `; a group
lists alternative phrases separated by `, `; after the groups come vetoes,
each a phrase after ` ! `. A rule may start with `~N ` to use a window of N
words instead of `distress.window`. Example:
`~10 he|she|dad + hits me|us, hit me ! hit me up`.

For each group, its matches are the (start, end) word positions of every
match of every one of its phrases (end = start + number of slots). The rule
fires if there is a choice of one match per group such that, with lo the
smallest start and hi the largest end, `hi - lo <= window`, and no veto
match (s, e) satisfies `lo - window <= s and e <= hi + window`. The groups
may come in any order. A rule with one group fires on any match of it that
no veto is near.

### 4. Query themes: `query_themes(query)` and `selection_themes`

Theme words T: copy W. Each word of `theme_fuzzy_min_len` (6) or more
letters that is unknown is replaced by the first word, in sorted (code
point) order, of the theme words that is one edit away from it, if any.
Theme words: the alternatives without `*` of every theme phrase that has a
single slot. Unknown: not an alternative (without `*`) of any slot of any
theme phrase, not a stopword, and with no term (step 5: `term_of(w)` is
none). So "lonley" → "lonely", but "honestly" stays (its base "honest" is a
pool term).

Find every (start, end, theme) where a phrase of that theme matches T (end =
start + number of slots); identical triples count once. Drop a match if
another match (of any theme) spans it and is strictly longer:
`s2 <= s and e <= e2 and e2 - s2 > e - s`. Each remaining match adds
`e - s` to its theme. Divide by the largest total, so the top theme scores
1. Order: score descending, then theme id ascending. **These scores are not
rounded**; every comparison below uses the unrounded values.

The themes that choose verses (`selection_themes`), in this order:

1. Keep the query themes with score >= `min_theme_score`.
2. If any of `exclusive_themes` is kept, keep only the first one in that
   list, with score 1 (grief before gratitude).
3. For a sensitive query, append each of the topic's `themes` not already
   there (in list order) with score `sensitive_theme_score`.

`themes` in the result: `{id, score rounded}` for these, in this order.
`notes`: `{theme, note}` for each of them (in this order) whose theme has a
`note`.

### 5. Normalising a word: `term_of(w)`

If w is a key of `vocabulary`, its term is `vocabulary[w]`, where "" means
a filler word (no term). Otherwise:

1. Replace w by `irregular[w]` if there is one.
2. Up to two times: take the candidates of w (below); the first candidate
   that is a key of `vocabulary` replaces w; if the new w is a stopword, stop.
3. The term is `vocabulary[w]` if w is a key (and not ""), else none.

Candidates of w: none if w is in `not_inflected`. Otherwise use the **first**
rule of `suffix_rules` with `len(w) >= min_len`, where w ends with one of
`ends` and with none of `not_ends` (only that rule; later rules are not
tried even if it yields nothing). For each entry of its `try` list in order:

1. skip the entry if it has `only_if_ends` and w ends with none of them;
2. stem = w without its last `strip` letters;
3. if `undouble` is set: skip the entry unless the stem has at least 2
   letters and its last two letters are equal; then drop its last letter;
4. append `add` (if any);
5. keep the stem only if it now has at least 3 letters (this check comes
   last, after `undouble` and `add`).

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

Q only breaks ties between verses (step 7); it never chooses one.

### 7. Choosing and ordering the verses, per book

If the category is crisis, stop: `results` are empty for every book,
`themes` and `notes` are empty, `message` is `crisis_message` and `no_match`
is **false** (a crisis is not a failed search). Otherwise S =
selection_themes (step 4), in its order; if S is empty, every book returns
nothing.

BM25 statistics come from the book's pool verses' `terms`: n = number of
verses; df_t = number of verses containing t; `idf_t = ln(1 + (n - df_t +
0.5) / (df_t + 0.5))`; len = number of terms of the verse; avglen = mean len.

For each verse v of the book (in pool order; i is its position):

- If the query is sensitive and v is not `sensitive`, skip v.
- M = the themes of v (in v's order) that are in S. If M is empty, skip v:
  **a verse is shown only if it carries a theme the query matched**.
- best = the theme of M that comes first in S.
- p = `v.priority[best]` if v has one and the query is not sensitive
  (sensitive verses come from a short hand-picked list and are ordered by
  the query's words instead), else none.
- bm25 = sum over Q (in Q's order) of terms t that occur in v of
  `w_t * idf_t * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen))`.
- key = `(position of best in S, p or 1000, -r(sum of S[t] over M, in M's
  order), -r(bm25), i)`.

Rounding: `r(x) = sign(x) * floor(|x| * 10000 + 0.5) / 10000` (halves away
from zero; JavaScript: `Math.sign(x) * Math.floor(Math.abs(x) * 1e4 + 0.5) /
1e4`). Sort by key ascending and take the first k. So the top theme's verses
come before a secondary theme's, the hand-picked verses of a theme come
first, and the query's words only decide among equals.

Each hit: ref, text, themes, why, `theme` (best), `priority` (p, or null),
`score` (r(bm25)), and `matched = {"themes": M, "words": [{"word": t,
"from": source}]}` for the terms of Q (in Q's order) that occur in v.
`no_match` is true when no book returns anything (and the category is not
crisis).

`Math.log` and Python's `math.log` can differ in the last bit; BM25 values
agree to about 1e-12, so a different order is possible only for verses whose
keys tie before rounding to within that, which the fixtures do not contain.

### 8. `search(query, k=3)`

k must be a positive integer (booleans and floats are rejected); the same
rule holds for `by_theme` and `similar`. It returns:

```
{"category": "general" | "sensitive" | "crisis",
 "topic": null | sensitive topic | crisis topic | "distress",
 "message": null | crisis_message,
 "no_match": bool,
 "themes": [{"id", "score"}],           selection themes, rounded scores
 "notes": [{"theme", "note"}],
 "results": {"gita": [hit, ...], "quran": [...], "bible": [...]}}
```

### 9. `by_theme(theme, k)`

For the theme chips. Per book, the verses whose themes include the theme.
Sort by: priority for this theme (1, 2, ...; none counts as 1000), then
score = r(1 / (position of the theme in the verse's list + 1)) descending,
then pool order; take k. Each hit has `theme` and `priority` as in step 7.
A book that does not offer the theme returns nothing. An unknown theme is an
error.

### 10. `similar(book, ref, k, sensitive_only=false)`

The source must be a pool verse. Per book, every other pool verse (with
`sensitive_only`, only verses marked `sensitive`; the source itself may be
any pool verse) scores the dot product of the unit-length rows of
`pool_vectors.i8` (all-zero rows stay zero, so they score 0). Round, sort by
score descending then pool order, take k.

## fixtures.json

- `words`: `[text, words(text)]` pairs for step 1, including non-ASCII text.
- `search`: for each case, `search(query, k)`:
  - `query`, `k`, `words` (W), `crisis_words` (C of step 3),
    `theme_words` (T of step 4);
  - `category`, `topic`, `message`, `no_match`;
  - `themes`: `[theme, rounded score]` in order (selection themes);
    `notes`: the themes whose note is shown;
  - `terms`: Q from step 6 as `[term, weight, source]` in Q's order. It is
    given for every case, **crisis ones included**, although a crisis search
    never uses it;
  - `results`: per book, `[ref, theme, priority, score, matched themes,
    matched terms]`.
- `by_theme`: per theme, per book `[ref, score, priority]` (k = 3).
- `similar` and `similar_sensitive_only`: per source verse, per book
  `[ref, score]` (k = 3).

A port should match every field exactly.
