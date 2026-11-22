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

The advice chat and the similarities page run on a small search engine trained from scratch on the three books. There are no pre-trained models, embeddings or word lists; it uses only PyTorch and the standard library.

```bash
python -m scripture.search "I got laid off and I'm scared about money"   # top 3 verses from each book
python search/build.py        # retrain and rewrite search/artifacts/ (about a minute on a CPU)
python evals/search_eval.py   # benchmark; writes evals/results/search.json
```

How it works (details in [scripture/search.py](scripture/search.py) and [search/build.py](search/build.py)):

- **Words.** Old forms are reduced to a base word that occurs in the books ("loveth", "loved", "loving" → "love"). Filler words are dropped with a hand-written list.
- **Bridge.** [search/bridge.json](search/bridge.json) is a hand-written table of about 650 modern words and phrases, each mapped to words the scriptures do use ("stressed" → troubled, burden, weary; "laid off" → idle, wages, poverty, bread). The tests check that every target word occurs in the books.
- **BM25.** Scores how many of the query's words a verse contains, giving rare words more weight. It is computed inside each book, so each book returns its own top verses and the 31,102-verse Bible cannot crowd out the 701-verse Gita.
- **Word vectors.** Trained on the three books by counting which words appear near each other (PPMI) and compressing the table with a truncated SVD. A verse vector is a weighted average of its word vectors. This finds verses that share meaning but not words.
- **Hybrid.** The final score mixes BM25 and vector similarity. The same vectors, mixed with TF-IDF, give the closest verses across books for the similarities page (`search/artifacts/similarities.json`).

The artefacts the browser loads are 7.5 MB in total; the format is in [search/FORMAT.md](search/FORMAT.md).

**Benchmark.** [evals/search_cases.json](evals/search_cases.json) has 50 everyday situations ("my dad has cancer", "I can't forgive my brother"), each with hand-picked verses that fit, per book. The 17 test situations were written before any search code and were never used for tuning; the 33 dev situations were used to choose settings. hit@10 is the share of situation–book pairs where a fitting verse is in that book's top 10; MRR is the average of 1 / rank of the first fitting verse.

Held-out test set (17 situations, 51 situation–book pairs):

| Method | Gita hit@10 | Quran hit@10 | Bible hit@10 | All hit@3 | All hit@10 | All MRR |
|---|---|---|---|---|---|---|
| Random order | 0.11 | 0.01 | 0.00 | 0.01 | 0.04 | 0.02 |
| BM25, no bridge | 0.24 | 0.18 | 0.12 | 0.06 | 0.18 | 0.05 |
| BM25 + bridge | 0.71 | 0.71 | 0.41 | 0.33 | 0.61 | 0.28 |
| Vectors, no bridge | 0.35 | 0.12 | 0.12 | 0.08 | 0.20 | 0.08 |
| Vectors + bridge | 0.65 | 0.53 | 0.59 | 0.35 | 0.59 | 0.34 |
| **Hybrid (used on the site)** | 0.59 | 0.65 | 0.47 | 0.39 | 0.57 | 0.34 |

Dev set (33 situations, 95 pairs; used for tuning, so expect it to look better):

| Method | Gita hit@10 | Quran hit@10 | Bible hit@10 | All hit@3 | All hit@10 | All MRR |
|---|---|---|---|---|---|---|
| Random order | 0.12 | 0.02 | 0.01 | 0.01 | 0.05 | 0.02 |
| BM25, no bridge | 0.28 | 0.18 | 0.24 | 0.14 | 0.23 | 0.13 |
| BM25 + bridge | 0.62 | 0.58 | 0.70 | 0.44 | 0.63 | 0.36 |
| Vectors, no bridge | 0.38 | 0.21 | 0.24 | 0.15 | 0.27 | 0.15 |
| Vectors + bridge | 0.55 | 0.58 | 0.67 | 0.38 | 0.60 | 0.33 |
| **Hybrid (used on the site)** | 0.72 | 0.64 | 0.73 | 0.43 | 0.69 | 0.36 |

What this says:

- The hand-written bridge is what makes modern questions work: without it, both methods find a fitting verse in the top 10 only about 20% of the time; with it, about 60%.
- On the held-out set, hybrid, BM25 + bridge and vectors + bridge are within noise of each other on hit@10 (each pair is 2 points). Hybrid has the best hit@3 and ties for the best MRR, so it is kept, but the test set does not show it clearly beating its parts.
- The hand-picked verse lists are incomplete, so these numbers are a lower bound: many top hits that are not on the list still fit (for "I cheated on my partner", Malachi 2:14 and Jeremiah 3:20 on dealing treacherously with a wife).
- It matches topics, not advice. "I can't stop drinking" finds verses about wine, not about breaking a habit; "my mother passed away" can return "And last of all the woman died also." Situations about feelings with no word in the books (purpose, burnout) are the weakest.

**Parallels.** [evals/search_parallels.json](evals/search_parallels.json) lists 61 known parallels: 32 shared stories of the Bible and the Quran (Joseph's dream, Moses' staff, Noah's ark, the annunciation to Mary) and 29 shared teachings or images, including thematic Gita parallels (the soul is not slain, offer every act to God). Looking up each anchor verse, how often is the parallel among the 10 closest verses of the other book?

| Similarity | All (122 lookups) | Stories | Teachings |
|---|---|---|---|
| TF-IDF only | 0.36 | 0.47 | 0.24 |
| Vectors only | 0.43 | 0.55 | 0.29 |
| Hybrid (used) | 0.47 | 0.56 | 0.36 |

Word vectors were also trained with skip-gram and negative sampling (`python search/build.py --method sgns`). On the dev set it was worse (vectors-only hit@10 0.44 vs 0.60 for PPMI + SVD) and took minutes instead of seconds to train, so PPMI + SVD is used.
