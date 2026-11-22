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

Each verse is fed as `<book>text`, and every training sequence picks its book uniformly, so the small Gita is seen as often as the Bible. 5% of each book's verses (at least 50) are held out for validation. `python train.py --help` lists every setting.

## Sample

```bash
python sample.py --book gita --prompt "Krishna said:" --tokens 400 --temperature 0.8
```

Writes text in the style of one book (`gita`, `quran` or `bible`), starting from its tag.

## Evals

```bash
python evals/run.py --ckpt out/best.pt
```

Writes `evals/results/model.json` and prints a table: validation loss next to uniform, unigram and 2- to 6-gram baselines fitted on the same text, how much generated text is copied verbatim from training, whether a simple classifier recognises each book's style, and how many generated words are real.

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
