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

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
