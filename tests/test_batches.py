"""The train/val split, the rendered text and the batch sampler in scripture/data.py."""
import pytest
import torch

from scripture.data import BOOKS, TAGS, get_batch, load_splits, render, split_verses
from tests.conftest import SIZES


@pytest.mark.parametrize("n, n_val", [(100, 50), (1000, 50), (2000, 100), (6236, 312)])
def test_split_is_disjoint_complete_and_sized(n, n_val):
    verses = [f"verse {i}" for i in range(n)]
    train, val = split_verses(verses)
    assert len(val) == n_val
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == sorted(verses)
    # both parts keep the book's order
    assert train == [v for v in verses if v in set(train)]
    assert val == [v for v in verses if v in set(val)]


def test_split_is_deterministic_and_random():
    verses = [f"verse {i}" for i in range(1000)]
    assert split_verses(verses) == split_verses(verses)
    assert split_verses(verses)[1] != split_verses(verses, seed=1)[1]
    assert split_verses(verses)[1] != verses[-50:]  # not just the last verses


def test_render():
    assert render("quran", ["In the name.", "Praise."]) == "<quran>In the name.\n<quran>Praise.\n"


def test_tags_are_single_tokens_and_mark_every_verse(tiny_data_dir):
    tok, texts, data = load_splits(tiny_data_dir)
    for tag in TAGS:
        assert len(tok.encode(tag)) == 1
    for split in ("train", "val"):
        for book in BOOKS:
            ids = data[split][book].tolist()
            n_verses = texts[split][book].count("\n")
            assert ids.count(tok.stoi[f"<{book}>"]) == n_verses
            assert ids[0] == tok.stoi[f"<{book}>"]
            assert sum(ids.count(tok.stoi[t]) for t in TAGS) == n_verses  # no other book's tag
            assert tok.decode(ids) == texts[split][book]
    assert texts["val"]["gita"].count("\n") == 50
    assert texts["train"]["bible"].count("\n") == SIZES["bible"] - 60


def test_val_text_is_not_in_train(tiny_data_dir):
    _, texts, _ = load_splits(tiny_data_dir)
    for book in BOOKS:
        train_verses = set(texts["train"][book].split("\n"))
        val_verses = texts["val"][book].split("\n")[:-1]
        assert len(val_verses) >= 50
        # the made-up verses can repeat by chance, but most held-out verses must be unseen
        assert sum(v not in train_verses for v in val_verses) > 0.9 * len(val_verses)


def fake_books(sizes=(300, 3000, 30000)):
    """Each book's tensor holds its own index, offset so values never collide across books."""
    return {b: torch.arange(n) + 100_000 * i for i, (b, n) in enumerate(zip(BOOKS, sizes))}


def test_batch_books_are_balanced_despite_sizes():
    data = fake_books()
    gen = torch.Generator().manual_seed(0)
    x, y, book_ids = get_batch(data, 6000, 8, gen)
    counts = torch.bincount(book_ids, minlength=3).float() / 6000
    assert torch.all((counts - 1 / 3).abs() < 0.03), counts
    # each row really comes from the book it reports
    assert torch.equal(x[:, 0] // 100_000, book_ids)


def test_weighted_batches_follow_book_sizes():
    data = fake_books()
    sizes = torch.tensor([300.0, 3000.0, 30000.0])
    _, _, book_ids = get_batch(data, 20000, 8, torch.Generator().manual_seed(5), weights=sizes)
    counts = torch.bincount(book_ids, minlength=3).float() / 20000
    assert torch.allclose(counts, sizes / sizes.sum(), atol=0.01), counts
    assert counts[0] > 0  # the small book still appears


def test_targets_are_inputs_shifted_by_one():
    data = fake_books()
    x, y, _ = get_batch(data, 64, 16, torch.Generator().manual_seed(1))
    assert x.shape == y.shape == (64, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y, x + 1)  # consecutive tokens of the book


def test_fixed_book_and_windows_stay_inside():
    data = fake_books()
    x, _, book_ids = get_batch(data, 500, 16, torch.Generator().manual_seed(2), book="gita")
    assert torch.all(book_ids == 0)
    assert x.min() >= 0 and x.max() + 1 <= 299  # the target of the last position is still in the book


def test_every_window_can_be_drawn():
    # a book of block_size + 2 tokens has exactly two windows, starting at 0 and at 1
    data = {b: torch.arange(10) for b in BOOKS}
    x, _, _ = get_batch(data, 200, 8, torch.Generator().manual_seed(3))
    assert set(x[:, 0].tolist()) == {0, 1}


def test_same_seed_same_batch():
    data = fake_books()
    a = get_batch(data, 32, 8, torch.Generator().manual_seed(4))
    b = get_batch(data, 32, 8, torch.Generator().manual_seed(4))
    assert all(torch.equal(p, q) for p, q in zip(a, b))
