# Scripture-3M

A small language model trained from scratch on three scriptures: the Bhagavad Gita, the Quran and the Bible.

Everything here is trained in this repo. No pre-trained models, no paid APIs.

- **Advice**: describe a situation and get the real verses that fit it, from all three books.
- **Similarities**: the closest passages across the three books, side by side.
- **Generation**: a ~3M-parameter transformer, written by hand in PyTorch, that writes in each book's style.

Work in progress. Each part lands as its own pull request.

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
