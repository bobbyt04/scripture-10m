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

## Meaning search

The advice chat and the similarities page run on a small search engine trained from scratch on the three books. It uses no pre-trained models, embeddings, word lists or outside APIs, only PyTorch and the Python standard library. Every word list in it was written by hand in this repo or derived from the three books.

```bash
python -m scripture.search "I got laid off and I'm scared about money"   # top 3 verses from each book
python search/build.py        # retrain and rewrite search/artifacts/ (about a minute on a CPU)
python evals/search_eval.py   # benchmark; writes evals/results/search.json
```

How it works (details in [scripture/search.py](scripture/search.py), [search/build.py](search/build.py) and [search/FORMAT.md](search/FORMAT.md)):

- **Safety first.** Hand-written phrase lists ([search/safety.json](search/safety.json)) sort each query.
  - "Crisis" queries (suicide, self-harm, abuse, sexual violence, harming others) get **no verses**, so the site can show help instead.
  - "Sensitive" queries (sexual orientation and gender, abortion, disability, race, religion as identity, eating and body, addiction relapse, pregnancy loss, widowhood) get verses only from a hand-picked set of gentle verses: 34 from the Gita, 35 from the Quran and 42 from the Bible, each with a one-line reason.
  - A blocklist keeps verses about violence, curses and hell, sexual violence, bodily horror, polemic against other faiths or peoples, punishment law, slavery and caste, and the subordination of women out of every situational answer. That is 58 of the 701 Gita verses, 1,373 of 6,236 in the Quran and 5,444 of 31,102 in the Bible. It is built from word patterns and verse ranges, and 118 verses that only mention such a word kindly are kept (Psalm 23:4, "love your enemies", "do not despair of the mercy of God").
  - A query with no known words returns nothing.
- **Words.** Old forms are reduced to a base word that occurs in the books ("loveth", "loved", "loving" → "love"). Names are never shortened. Filler words are the 120 words found in the most verses (each book weighted equally), minus 22 content words (lord, god, allah, ...), plus a short hand-written list of archaic and modern fillers ([search/lexicon.json](search/lexicon.json)).
- **Bridge.** [search/bridge.json](search/bridge.json) is a hand-written table of about 640 modern words and phrases, each mapped to words the books do use: "stressed" → troubled, burden, weary; "laid off" → idle, wages, poverty, bread. Words that mean something else in scripture ("job", "kids", "tired") are searched only through their bridge words. Words with a common innocent sense ("passed", "high", "moved", "down") fire only as phrases ("passed away", "got high"). The tests check that every target occurs in the books.
- **BM25.** Scores how many of the query's words a verse contains, giving rare words more weight. It is computed inside each book, so each book returns its own top verses and the 31,102-verse Bible cannot crowd out the 701-verse Gita. **This is what the site uses** (see below).
- **Word vectors.** Trained on the three books by counting which words appear near each other (PPMI) and compressing the table with a truncated SVD. A verse vector is a weighted average of its word vectors. The vectors power "similar verses" across books and the optional hybrid ranking.

The artefacts total 7.55 MB. [search/FORMAT.md](search/FORMAT.md) describes them well enough to port the scorer, and `search/artifacts/fixtures.json` holds 30 queries with their expected results to test a port.

### Benchmark

[evals/search_cases.json](evals/search_cases.json) has 90 everyday situations ("my dad has cancer", "I can't forgive my brother"), each with hand-picked verses that fit, per book: 50 in "dev" and "dev2", and 40 in `blind_test`.

**Only the blind test is a held-out result.** I saw the dev and dev2 sets while writing the bridge and choosing the settings. The 17 "dev2" situations were first meant as a test set, but the bridge was written after those queries existed and covers many of their exact phrases.

Each number is the share of situation–book pairs whose hand-picked answer key has a verse in that book's top 10 (hit@10) or top 3, or the mean of 1 / rank of the first such verse (MRR). The 95% intervals come from a bootstrap that resamples whole situations. Blocklisted verses are never returned, as on the site, so blocklisted verses were taken out of the answer keys (62 in dev and dev2, 21 in the blind test).

Blind test (40 situations, 113 pairs). The situations and their answer keys were written by an independent reviewer, not the author, who had not seen the bridge, the lexicon, the safety lists or the dev cases, and the bridge was not changed afterwards:

| Method | Gita hit@10 | Quran hit@10 | Bible hit@10 | All hit@3 | All hit@10 (95% CI) | All MRR (95% CI) |
|---|---|---|---|---|---|---|
| Random order | 0.09 | 0.02 | 0.01 | 0.01 | 0.03 (0.03–0.04) | 0.02 (0.02–0.02) |
| BM25, no bridge | 0.21 | 0.10 | 0.08 | 0.04 | 0.12 (0.06–0.20) | 0.04 (0.02–0.06) |
| **BM25 + bridge (used on the site)** | 0.45 | 0.33 | 0.53 | 0.23 | 0.43 (0.34–0.52) | 0.23 (0.16–0.30) |
| Vectors, no bridge | 0.30 | 0.05 | 0.15 | 0.04 | 0.16 (0.08–0.25) | 0.05 (0.03–0.08) |
| Vectors + bridge | 0.45 | 0.35 | 0.38 | 0.23 | 0.39 (0.28–0.50) | 0.20 (0.14–0.26) |
| Hybrid (BM25 + vectors) | 0.64 | 0.35 | 0.53 | 0.28 | 0.50 (0.40–0.60) | 0.25 (0.18–0.31) |

Dev (33 situations, 95 pairs):

| Method | Gita hit@10 | Quran hit@10 | Bible hit@10 | All hit@3 | All hit@10 (95% CI) | All MRR (95% CI) |
|---|---|---|---|---|---|---|
| Random order | 0.13 | 0.02 | 0.01 | 0.02 | 0.05 (0.04–0.06) | 0.03 (0.02–0.03) |
| BM25, no bridge | 0.28 | 0.18 | 0.24 | 0.15 | 0.23 (0.14–0.33) | 0.13 (0.08–0.19) |
| **BM25 + bridge (used on the site)** | 0.62 | 0.64 | 0.67 | 0.47 | 0.64 (0.54–0.74) | 0.36 (0.29–0.44) |
| Vectors, no bridge | 0.38 | 0.15 | 0.21 | 0.13 | 0.24 (0.14–0.35) | 0.13 (0.07–0.20) |
| Vectors + bridge | 0.55 | 0.52 | 0.67 | 0.34 | 0.58 (0.46–0.69) | 0.33 (0.25–0.41) |
| Hybrid (BM25 + vectors) | 0.72 | 0.61 | 0.70 | 0.43 | 0.67 (0.57–0.77) | 0.37 (0.29–0.45) |

Dev2 (17 situations, 51 pairs; formerly called "test", not held out):

| Method | Gita hit@10 | Quran hit@10 | Bible hit@10 | All hit@3 | All hit@10 (95% CI) | All MRR (95% CI) |
|---|---|---|---|---|---|---|
| Random order | 0.12 | 0.02 | 0.01 | 0.01 | 0.05 (0.04–0.05) | 0.02 (0.02–0.03) |
| BM25, no bridge | 0.24 | 0.18 | 0.06 | 0.08 | 0.16 (0.06–0.29) | 0.06 (0.02–0.11) |
| **BM25 + bridge (used on the site)** | 0.71 | 0.65 | 0.47 | 0.29 | 0.61 (0.45–0.77) | 0.28 (0.18–0.37) |
| Vectors, no bridge | 0.47 | 0.12 | 0.12 | 0.10 | 0.24 (0.12–0.35) | 0.08 (0.03–0.14) |
| Vectors + bridge | 0.59 | 0.41 | 0.65 | 0.37 | 0.55 (0.39–0.71) | 0.34 (0.23–0.47) |
| Hybrid (BM25 + vectors) | 0.65 | 0.53 | 0.65 | 0.37 | 0.61 (0.45–0.78) | 0.36 (0.23–0.49) |

What this says:

- **On unseen situations the site finds a fitting verse in the top 10 for 43% of pairs, not about 60%.** The drop from dev (0.64) to the blind test (0.43) is what to expect when the bridge was written with the dev queries in view (the blind answer keys also come from a different person); the blind number is the one to quote.
- **The hand-written bridge is what makes modern questions work.** Without it, a fitting verse is in the top 10 only 16–24% of the time; with it, about 60%.
- **The site uses BM25 + bridge, not the hybrid.** Over all 50 dev situations, the hybrid minus BM25 + bridge is +0.021 hit@10 (95% interval -0.034 to +0.077) and +0.033 MRR (-0.019 to +0.079). That is no measurable gain, so the simpler method is used. (In the first version of this PR, the hybrid was even the worst of the three bridged methods on dev2 hit@10; now it ties BM25 + bridge there.)
- **Vectors alone are the weakest bridged method.** The hybrid beats vectors + bridge by +0.082 hit@10 (+0.028 to +0.144).
- **The answer keys are incomplete,** so these numbers are a lower bound: several unlisted top hits still fit.
- **It matches topics, not advice.** "I can't stop drinking" finds verses about wine, not about breaking a habit. Situations about feelings with no word in the books (purpose, burnout) are the weakest.

### Similarities

[search/artifacts/similarities.json](search/artifacts/similarities.json) shows 85 parallels in 7 themes, **chosen by a person, not by the model alone**:

- **Model, reviewed (43).** The model proposed 267 candidate pairs (each verse's closest verse in another book). I read every one and kept 43 (16%) that are real parallels in story or teaching, each with a one-line note on the shared idea. The other 84% shared words or images but not an idea. Every decision is in [search/similarities_review.json](search/similarities_review.json).
- **Hand-listed (42).** The rest come from the hand-listed parallels below; 19 of those were left out because a verse is on the blocklist or the pair was already there.

The 16% kept rate is itself a measure: on its own, the model's nearest-verse search mostly finds surface coincidences.

[evals/search_parallels.json](evals/search_parallels.json) lists 61 known parallels: 32 shared Bible–Quran stories (Joseph's dream, Moses' staff, Noah's ark, the annunciation to Mary) and 29 shared teachings, including thematic Gita parallels (the soul is not slain; offer every act to God). For each anchor verse, how often is the parallel among the 10 closest verses of the other book? (This measures the model, so blocklisted verses are allowed here.)

| Similarity | All (122 lookups) | Stories | Teachings |
|---|---|---|---|
| TF-IDF only | 0.37 | 0.48 | 0.24 |
| Vectors only | 0.47 | 0.53 | 0.40 |
| Mix (used) | 0.48 | 0.56 | 0.38 |

Word vectors were also trained with skip-gram and negative sampling (`python search/build.py --method sgns`). On the dev set it was worse (vectors-only hit@10 0.44 against 0.60 for PPMI + SVD, in an earlier round) and took minutes instead of seconds to train, so PPMI + SVD is used.
