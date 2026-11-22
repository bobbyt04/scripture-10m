"""The n-gram baselines and the metrics in evals/metrics.py."""
import math
import random
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from evals.run import model_nll
from evals.metrics import NGram, classify, complete_verses, memorisation, strip_tags, words

ALPHABET = "abcdefgh .\n"


def wordy_text(seed, n_words, vocab=("the", "lord", "said", "unto", "him", "and", "behold", "fear")):
    rng = random.Random(seed)
    return " ".join(rng.choice(vocab) for _ in range(n_words))


def test_witten_bell_by_hand():
    m = NGram("abab", 2, vocab_size=5)
    p_a, p_b = 3 / 9, 3 / 9  # add-one unigram: (2 + 1) / (4 + 5)
    assert m.prob("", "a") == pytest.approx(p_a)
    assert m.prob("x", "z") == pytest.approx(1 / 9)  # unseen history and symbol
    # after "a": seen twice as a history, always followed by "b", so one type
    assert m.prob("a", "b") == pytest.approx((2 + 1 * p_b) / (2 + 1))
    assert m.prob("a", "a") == pytest.approx((0 + 1 * p_a) / (2 + 1))
    # "b" is followed once (the final "b" has no successor)
    assert m.prob("b", "a") == pytest.approx((1 + 1 * p_a) / (1 + 1))
    assert m.prob("ab", "a", order=1) == pytest.approx(p_a)


def test_witten_bell_weights_backoff_by_number_of_types():
    m = NGram("abac", 2, vocab_size=5)
    p_b, p_d = 2 / 9, 1 / 9
    # after "a": seen twice, followed by two different symbols ("b" and "c")
    assert m.prob("a", "b") == pytest.approx((1 + 2 * p_b) / (2 + 2))
    assert m.prob("a", "d") == pytest.approx((0 + 2 * p_d) / (2 + 2))


@pytest.mark.parametrize("order", [1, 2, 3, 5])
@pytest.mark.parametrize("history", ["", "the l", "zzzz", "d unt", "q"])
def test_probabilities_sum_to_one(order, history):
    text = wordy_text(0, 500)
    m = NGram(text, 5, vocab_size=30)
    symbols = sorted(set(text)) + [chr(200 + i) for i in range(30 - len(set(text)))]
    assert len(symbols) == 30
    assert sum(m.prob(history, c, order) for c in symbols) == pytest.approx(1.0)


def test_higher_order_beats_unigram_on_held_out_text():
    m = NGram(wordy_text(1, 4000), 4, vocab_size=30)
    held_out = wordy_text(2, 400)
    nll = {n: m.nll(held_out, order=n) for n in (1, 2, 3, 4)}
    assert nll[2] < nll[1] - 0.3
    assert nll[4] < nll[2]
    assert nll[1] < math.log(30)


def test_nll_skips_first_symbol_and_averages():
    m = NGram("abab", 2, vocab_size=5)
    expected = -(math.log(m.prob("a", "b")) + math.log(m.prob("b", "a"))) / 2
    assert m.nll("aba") == pytest.approx(expected)
    assert m.nll("aba", start=0) == pytest.approx((2 * expected - math.log(m.prob("", "a"))) / 3)


TRAIN = wordy_text(3, 3000)


def test_memorisation_of_copied_text_is_one():
    copied = TRAIN[1000:1400]
    r = memorisation(copied, TRAIN, window=50)
    assert r["windows"] == r["memorised"] == 351
    assert r["longest"] == 400 and r["longest_text"] == copied


def test_memorisation_of_shuffled_text_is_near_zero():
    chars = list(TRAIN[1000:1400])
    random.Random(0).shuffle(chars)
    r = memorisation("".join(chars), TRAIN, window=50)
    assert r["memorised"] == 0
    assert r["longest"] < 15


def test_memorisation_finds_the_copied_span():
    generated = "#" * 30 + TRAIN[500:570] + "#" * 30  # "#" never appears in TRAIN
    r = memorisation(generated, TRAIN, window=50)
    assert r["windows"] == 130 - 50 + 1
    assert r["memorised"] == 70 - 50 + 1
    assert r["longest_text"] == TRAIN[500:570]


def test_classify_picks_the_matching_book():
    models = {"a": NGram(wordy_text(4, 3000, ("arjuna", "krishna", "yoga")), 3, 40),
              "b": NGram(wordy_text(5, 3000, ("moses", "israel", "king")), 3, 40)}
    assert classify("krishna yoga arjuna", models) == "a"
    assert classify("israel king moses", models) == "b"


def test_text_helpers():
    sample = "<gita>Krishna said: Arise.\n<gita>The wise don't grie"
    assert strip_tags(sample) == "Krishna said: Arise.\nThe wise don't grie"
    assert words(sample) == ["krishna", "said", "arise", "the", "wise", "don't", "grie"]
    assert complete_verses(sample) == ["Krishna said: Arise."]


class Bigram(torch.nn.Module):
    """Logits depend only on the current token, so the exact loss of any text is easy to compute."""

    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.cfg = SimpleNamespace(block_size=block_size)
        self.table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        assert idx.shape[1] <= self.cfg.block_size
        return self.table(idx), None


@pytest.mark.parametrize("length", [5, 9, 50, 51])
def test_model_nll_scores_every_token_once(length):
    torch.manual_seed(0)
    model = Bigram(7, block_size=8)
    ids = torch.randint(0, 7, (length,))
    expected = F.cross_entropy(model.table(ids[:-1]), ids[1:]).item()
    assert model_nll(model, ids, "cpu") == pytest.approx(expected, rel=1e-5)
