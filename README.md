# Scripture-10M

A small language model trained from scratch on three scriptures: the Bhagavad Gita, the Quran and the Bible.

Everything here is trained in this repo. No pre-trained models, no paid APIs.

- **Advice**: describe a situation and get the real verses that fit it, from all three books.
- **Similarities**: the closest passages across the three books, side by side.
- **Generation**: a ~10M-parameter transformer, written by hand in PyTorch, that writes in each book's style.

## Data

Three public-domain English translations: the King James Bible, Pickthall's Quran (1930) and Annie Besant's Bhagavad Gita (1922).
`python data/prepare.py` downloads them and writes one cleaned verse per line to `data/processed/` (31,102 + 6,236 + 701 verses).
Sources, licences and cleaning steps are in [data/SOURCES.md](data/SOURCES.md).

Work in progress. Each part lands as its own pull request.

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Advice search

Describe a situation in everyday English and the site shows real verses from each book. **The advice search only ever shows verses from a hand-reviewed pool of 960 verses: 108 from the Gita, 236 from the Quran and 616 from the Bible.** Nothing outside the pool can be shown. It matches topics; it does not understand situations.

```bash
python -m scripture.search "I got laid off and I'm scared about money"   # top 3 verses from each book
python search/build.py --site   # rebuild search/artifacts/ after editing the pool or the lists (seconds)
python search/build.py          # also retrain the full-corpus model in search/model/ (under a minute on a CPU)
python evals/search_eval.py     # benchmark; writes evals/results/search.json
```

### The pool, and who reviewed it

- **Chosen.** Three curators read the books by theme and picked 1,054 verses (109 / 277 / 668), each with one or more of 40 themes (grief, money_worry, overwork_rest, ...), a one-line plain-English note on why it may help, and a flag for verses gentle enough for any reader.
- **Reviewed.** An independent reviewer then read every verse as the people most likely to be hurt by a wrong one (someone bereaved, suicidal, abused by a partner or parent, gay or trans, after a pregnancy loss, disabled, facing racism, of another faith or none). That removed 64 verses, dropped themes from 106, took 108 out of the sensitive pool, rewrote 24 notes and asked for display fixes on 371 verses (Psalm headings, a leading "And" or "Therefore", "Selah", bracketed glosses). A second pass re-read every verse under forgiveness, patience, suffering_endurance, marriage, family_parents, betrayal and anger as someone being harmed at home, and every money_worry verse as someone being evicted, and checked every note against the verse text in our translation (no wording from other translations, no added promises). The owner then dropped two themes entirely (mortality, laziness_discipline). Every decision is recorded in [search/pool/](search/pool/) (`REVIEW.md` and the review files) and applied by `search/apply_review.py`.
- **A book offers a theme only if it has at least three verses for it.** So the Gita offers nothing for grief, money worries, family, illness, marriage, injustice, betrayal or parenting, and the Quran nothing for marriage, betrayal or patience; for such a query that book says it has nothing fitting.

### How a query is answered

Details in [scripture/search.py](scripture/search.py) (plain Python, no dependencies) and [search/FORMAT.md](search/FORMAT.md).

1. **Safety first** ([search/safety.json](search/safety.json), hand-written).
   - **Crisis** (suicide, self-harm, abuse, threats of violence): no verses at all; the site shows a short kind note. Matching tolerates missing apostrophes and one-letter typos in key words, and everyday idioms ("killing it at work", "cut back on sugar", "my brother beat me at chess") are blanked first so they do not trigger it.
   - **Sensitive** (sexual orientation, gender identity, pregnancy and pregnancy loss, disability, race, religion or none, caste, eating and body image): only verses from a short hand-picked list ([search/pool/sensitive_set.json](search/pool/sensitive_set.json): 6 Gita, 19 Quran, 23 Bible verses about dignity, being known, not being forsaken and comfort), ranked by the same scorer with the topic's gentle themes added, so results still depend on what was written.
2. **Themes.** About 1,300 hand-written modern phrases ([search/themes.json](search/themes.json)) spot the query's themes: "passed away" → grief, "70 hour weeks" → overwork_rest. A longer phrase hides shorter ones inside it, so "lost my job" is not read as a bereavement. Some themes carry a short note shown with the results ("If someone is hurting you, your safety comes first.").
3. **Words.** Query words are reduced to base forms found in the pool ("drinking" → "drink"); a hand-written bridge ([search/bridge.json](search/bridge.json), 641 entries) adds scripture words for modern ones ("stressed" → troubled, burden, weary).
4. **Score.** For each pool verse: the sum of the query's theme scores over the verse's themes, plus 0.2 × BM25 of the query words against the verse text, its note and its theme names. If the query has themes, a verse must share one. A book whose best verse scores below 0.3 shows nothing rather than a weak match; if no book shows anything, the site offers the themes as chips.

**From scratch:** the word normaliser, BM25 ranking, and the word vectors (PPMI + SVD trained on the three books) used for similarities. **Hand-written:** the pool and its notes, the themes and their phrases, the bridge, and the crisis and sensitive lists. No pre-trained models, outside word lists or APIs.

The site loads 787 kB ([search/FORMAT.md](search/FORMAT.md) describes every field and walks through the scorer step by step); `search/artifacts/fixtures.json` holds expected outputs to test a port.

### Benchmark

[evals/search_cases.json](evals/search_cases.json) has 90 everyday situations, each with verses that fit, hand-picked from the whole corpus before the pool existed, and the themes a good answer should match.

**History, honestly.** The first design searched all 38,039 verses. Its first reported number (hit@10 0.57) was contaminated: the bridge had been written with those queries in view. A reviewer then wrote 40 blind situations; there the full-corpus design scored hit@10 0.43, and when a person judged the verses it actually showed, only 14% fitted (59% were plainly wrong, some harmful). That is why the design changed to a reviewed pool. All three case sets (dev, dev2, and the former blind test, now dev3) have since been seen while building the pool design, so **every number below is a development number**. A new blind set (`final_blind`, empty for now) will be written by someone else after the scorer is frozen, and the verses shown for it will be judged by hand; that precision will be the number to quote.

Metrics: hit@3 = is one of the three verses shown for a book in its answer key (only key verses inside the pool count; 1,451 of 2,810 key verses are outside it, and a book with no key verse in the pool is skipped); theme accuracy = is the query's top theme one of the hand-labelled ones; empty-book rate = how often a book shows nothing. Intervals are 95% bootstraps over cases.

| Cases | Gita hit@3 | Quran hit@3 | Bible hit@3 | All hit@3 (95% CI) | Theme accuracy | Empty-book rate |
|---|---|---|---|---|---|---|
| dev (33) | 0.71 | 0.70 | 0.78 | 0.73 (0.61–0.84) | 0.91 | 0.08 |
| dev2 (17) | 0.53 | 0.69 | 0.71 | 0.65 (0.49–0.81) | 0.88 | 0.06 |
| dev3 (40, formerly blind) | 0.58 | 0.54 | 0.73 | 0.62 (0.50–0.73) | 0.98 | 0.10 |

What each part adds (all 90 cases):

| Scorer | hit@3 (95% CI) | Empty-book rate |
|---|---|---|
| BM25 on the verse text only | 0.18 (0.12–0.24) | 0.25 |
| + bridge | 0.46 (0.39–0.52) | 0.03 |
| + notes and theme names in the document | 0.55 (0.48–0.61) | 0.02 |
| **+ theme matching (used)** | **0.67 (0.59–0.74)** | 0.09 |

The weights (0.2 for BM25, 0.3 threshold, 0.8 for bridge words) were chosen on these cases; nearby values change hit@3 by about ±0.02, within the noise. Word vectors were not added to the advice scorer: they are not needed to rank within a small, hand-themed pool, and leaving them out keeps the site's download small.

**Crisis detection** was tested on phrasings written after the lists were frozen: the first held-out round of 60 caught 28 (47%) and the second of 40 caught 21 (53%), with no false alarms on 100 innocent idioms. Both rounds were then added to the lists and to the tests. Phrase lists generalise poorly to new wording; because the pool holds only gentle verses, a missed crisis message gets gentle verses rather than harmful ones, but it does not get the kind note.

### Similarities

[search/artifacts/similarities.json](search/artifacts/similarities.json) shows 66 parallels in 7 themes, **chosen by a person, not by the model alone**:

- **Model, reviewed (33).** The model proposed 267 candidate pairs (each verse's closest verse in another book, by word vectors and shared words). A person read every one and kept 33 (12%) real parallels in story or teaching, each with a one-line note. The rest shared words or images but not an idea; that low rate is itself a measure of the model.
- **Hand-listed (33)** from [evals/search_parallels.json](evals/search_parallels.json); 28 were left out.
- Pairs with threatening or theologically contentious verses were left out (for example Isaac and the unnamed son of Abraham, Christ and Allah paired as the same God, the camel and the needle in its Quran setting), notes that misstated a verse were corrected, and each verse appears in at most one pair. Every decision is in [search/similarities_review.json](search/similarities_review.json).

For each anchor of the 61 known parallels (122 lookups, whole corpus), how often is the partner among the 10 closest verses of the other book?

| Similarity | All | Stories | Teachings |
|---|---|---|---|
| TF-IDF only | 0.37 | 0.48 | 0.24 |
| Vectors only | 0.47 | 0.53 | 0.40 |
| Mix (used) | 0.48 | 0.56 | 0.38 |
