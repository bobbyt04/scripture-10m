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

## Train

```bash
python train.py            # a few hours on an Apple-silicon Mac; writes out/ckpt.pt, out/best.pt, out/losses.jsonl
python train.py --resume   # continue an interrupted run from out/ckpt.pt
```

Each verse is fed as `<book>text`. By default every training sequence picks its book in proportion to the book's size, so every verse is seen equally often. `--sampling book` instead picks each book equally often; we tried that first, but the small Gita was then seen about 40 times as often per verse as the Bible and the model memorised it (its validation loss rose while its training loss kept falling). 5% of each book's verses (at least 50) are held out for validation. `python train.py --help` lists every setting.

`--resume` continues from the last checkpoint with the same settings, step, optimizer state and learning-rate schedule. Training batches depend only on the step, so the resumed run sees the same data; it matches an uninterrupted run exactly only when dropout is 0, because the dropout random state is not saved.

## Sample

```bash
python sample.py --book gita --prompt "Krishna said:" --tokens 400 --temperature 0.8
```

Writes text in the style of one book (`gita`, `quran` or `bible`), starting from its tag.

## Evals

```bash
python evals/run.py --ckpt out/best.pt
```

Writes `evals/results/model.json` and prints a table: validation loss (with bootstrap intervals, and again without held-out verses that also occur in training) next to uniform, unigram and 2- to 8-gram baselines fitted on the same text; how much generated text is copied verbatim from each book's training text; whether a simple classifier recognises each book's style, with control texts that check the classifier is fair; and how many generated words are real.

## Results

The model is `out/best.pt` from the run in [runs/verse](runs/README.md): 10,790,400 parameters, step 11,000 of
12,000, about 10 hours on an Apple M2 (inflated by other jobs on the same machine). All numbers below come from
[evals/results/model.json](evals/results/model.json), written by `python evals/run.py --ckpt out/best.pt`, and
for the abandoned run from [evals/results/book_sampling.json](evals/results/book_sampling.json).
`python evals/readme_tables.py` rebuilds the tables and a test checks that they match the JSON.

`best.pt` was picked by its loss on the same validation verses used here; there is no separate test split, so the
model's numbers are slightly optimistic. The validation set is 5% of each book: 50 Gita verses, 312 Quran verses
and 1,555 Bible verses.

### Validation loss

Mean loss per character on the held-out verses, in nats and in bits per character (lower is better), next to the
best of the uniform, unigram and 2- to 8-gram character models fitted on the same book's training text. Intervals
are 95% bootstrap intervals over held-out verses.

| Book | Scripture-10M, nats (bits/char) | 95% interval | Best baseline | Baseline, nats (bits/char) | 95% interval |
|---|---|---|---|---|---|
| Gita | 1.103 (1.591) | 1.040-1.168 | 5-gram | 1.371 (1.978) | 1.293-1.451 |
| Quran | 0.831 (1.198) | 0.804-0.855 | 7-gram | 1.092 (1.575) | 1.059-1.122 |
| Bible | 0.774 (1.117) | 0.764-0.784 | 7-gram | 1.039 (1.499) | 1.028-1.049 |
| Mean of the three | 0.903 (1.302) |  | 6-gram | 1.188 (1.714) |  |

The model beats the best n-gram baseline on every book, by 0.26 to 0.27 nats, and the intervals do not overlap.
The baselines use Witten-Bell smoothing; Kneser-Ney smoothing would make them somewhat stronger. Leaving out the
held-out verses that also occur word for word in training (0 Gita, 7 Quran, 14 Bible) changes the model's numbers
by at most 0.005 nats. These losses are lower than the validation losses in the training log (0.944 overall at
step 11,000) because the eval scores every held-out character with at least half a window of context, while
training estimates loss on random windows.

The abandoned run that sampled each book equally often (step 8,500) scores 2.525 nats on the Gita: worse than
the 2-gram baseline (2.369), because it memorised the Gita's training verses instead of learning to generalise.

### Memorisation

Share of 50-character windows of generated text that appear verbatim in the training text of any book, and the
longest copied run in characters. Each book gets 50 samples of 400 characters at temperature 1.0. The last row is
the same measure on real held-out verses: scripture repeats its own phrases, so it is not zero.

| Text | Gita | Quran | Bible |
|---|---|---|---|
| Generated, verse-sampled run (step 11000) | 0.0% (longest 35) | 0.1% (longest 72) | 0.3% (longest 95) |
| Generated, book-sampled run (step 8500) | 90.2% (longest 399) | 2.1% (longest 314) | 0.0% (longest 50) |
| Real held-out verses | 0.0% (longest 47) | 1.9% (longest 115) | 1.0% (longest 106) |

The final model copies no more than real unseen verses already overlap with the training text. The abandoned run recites the
Gita: 90% of its Gita windows are copied, and one sample is a 399-character stretch of training text. Its
Quran-tagged samples also drift into copied Gita verses (the 314-character run comes from the Gita).

### Style

A 5-gram character classifier per book, each fitted on the same amount of training text, guesses which book each
text comes from. The cells count texts sent to the book they come from or imitate; the last row shows where pure
noise goes.

| Texts | Gita | Quran | Bible |
|---|---|---|---|
| generated | 39/50 | 50/50 | 49/50 |
| real held-out text | 64/64 | 100/100 | 100/100 |
| real held-out, 10% of characters corrupted | 64/64 | 100/100 | 99/100 |
| unigram text | 27/50 | 10/50 | 26/50 |
| bigram text | 47/50 | 38/50 | 36/50 |
| uniform noise (share sent to each book) | 22/50 | 24/50 | 4/50 |

Quran and Bible samples are recognised almost always. Gita samples are recognised 39 times out of 50; 7 are taken
for Quran and 4 for Bible text, and reading them shows why: the Gita is the smallest book, and its samples borrow
words and phrases from the other two. The classifier is not fully fair, as the eval itself reports: uniform noise
almost never goes to the Bible (4/50), and unigram text imitating the Quran goes to the Quran only 10 times out
of 50. Both biases affect texts without word-level structure; text that has it (held-out verses,
even with 10% of characters corrupted) is classified correctly almost every time. The real held-out chunks overlap
by three quarters, so they are not independent.

### Words and verse length

| | Gita | Quran | Bible |
|---|---|---|---|
| Generated words found in the training text | 95.6% | 98.8% | 99.3% |
| Mean length of generated verses, characters | 150 | 94 | 140 |
| Mean length of real held-out verses | 135 | 120 | 133 |

Most generated words are real words; the invented ones look like "merity" and "denuised". Verse lengths count
only verses finished within a 400-character sample, so long verses are under-represented.

### Samples

Machine-generated imitations, not scripture. Each is the first verse of the seed-1 sample (500 characters,
temperature 0.8) in [samples/](samples/), unedited; the files have three unprompted and one prompted sample per
book.

> **Gita style**: Those who invent a lie concerning what will be before you, in doubt concerning the Activity, and of the Praised Lord of the Worlds.

> **Quran style**: Those who lead the Scripture with truth with their Lord, they do but know.

> **Bible style**: Thou art old and rock and drinkard, and the years of their men shall see their faces a glorious name: for they are dear to come upon the earth.

The Gita sample shows the problem from the style section: "Lord of the Worlds" is a Quran phrase. The model can
also switch to another book's tag in the middle of a sample (Gita seed 3 ends in Bible verses).

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
