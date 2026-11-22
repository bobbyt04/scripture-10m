"""Training data: the three books as token tensors, split into train and validation by verse.

Each verse is written as "<book>text\n", where "<book>" is a single special token, so the model
learns which book it is writing and where verses end.
"""
import json
import random
from pathlib import Path

import torch

from scripture.tokenizer import CharTokenizer

BOOKS = ("gita", "quran", "bible")
TAGS = tuple(f"<{b}>" for b in BOOKS)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SPLIT_SEED = 1337  # fixed, so every script holds out the same verses


def load_verses(data_dir=DATA_DIR):
    """{book: [verse text, ...]} in book order."""
    verses = {}
    for book in BOOKS:
        with open(Path(data_dir) / f"{book}.jsonl", encoding="utf-8") as f:
            verses[book] = [json.loads(line)["text"] for line in f]
    return verses


def split_verses(texts, val_frac=0.05, min_val=50, seed=SPLIT_SEED):
    """Hold out a random 5% of verses (at least min_val). Both parts keep the original verse order."""
    n_val = max(min_val, round(val_frac * len(texts)))
    val_idx = set(random.Random(seed).sample(range(len(texts)), n_val))
    train = [t for i, t in enumerate(texts) if i not in val_idx]
    val = [t for i, t in enumerate(texts) if i in val_idx]
    return train, val


def render(book, texts):
    return "".join(f"<{book}>{t}\n" for t in texts)


def load_splits(data_dir=DATA_DIR):
    """Returns (tokenizer, texts, data).

    texts[split][book] is the rendered text and data[split][book] its 1-D tensor of token ids,
    for split in ("train", "val").
    """
    texts = {"train": {}, "val": {}}
    for book, verses in load_verses(data_dir).items():
        train, val = split_verses(verses)
        texts["train"][book] = render(book, train)
        texts["val"][book] = render(book, val)
    # the character set comes from all the text, so no held-out character is unknown
    all_text = "".join(t for split in texts.values() for t in split.values())
    tok = CharTokenizer.from_text(all_text, TAGS)
    data = {s: {b: torch.tensor(tok.encode(t), dtype=torch.long) for b, t in texts[s].items()} for s in texts}
    return tok, texts, data


def get_batch(split_data, batch_size, block_size, generator, book=None, weights=None):
    """Random windows of block_size + 1 tokens. x is the window, y is the same window shifted by one.

    Each row picks its book (unless `book` is given), then a random window inside that book.
    Without `weights` every book is equally likely, whatever its size. With `weights` (one number
    per book) a book is picked in proportion to its weight: passing the books' lengths means every
    verse of every book is seen equally often. Returns x, y and the book index of each row.
    """
    if book is not None:
        book_ids = torch.full((batch_size,), BOOKS.index(book))
    elif weights is not None:
        book_ids = torch.multinomial(weights, batch_size, replacement=True, generator=generator)
    else:
        book_ids = torch.randint(len(BOOKS), (batch_size,), generator=generator)
    rows = []
    for b in book_ids.tolist():
        d = split_data[BOOKS[b]]
        i = torch.randint(len(d) - block_size, (1,), generator=generator).item()
        rows.append(d[i : i + block_size + 1])
    batch = torch.stack(rows)
    return batch[:, :-1], batch[:, 1:], book_ids
