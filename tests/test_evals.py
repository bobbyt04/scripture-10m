"""The n-gram baselines and the metrics in evals/metrics.py."""
import math
import random
from collections import Counter

import pytest

from evals.metrics import (NGram, Source, classify, complete_verses, corrupt, markov_samples, memorisation,
                           strip_tags, subsample, suffix_array, words)


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
    assert strip_tags("<quran>a\n<bible>b\n<gita>c<other>") == "a\nb\nc<other>"
    assert words(sample) == ["krishna", "said", "arise", "the", "wise", "don't", "grie"]
    assert complete_verses(sample) == ["Krishna said: Arise."]


def test_restricted_counts_score_the_same():
    train, val = wordy_text(6, 2000), wordy_text(7, 200)
    full, small = NGram(train, 6, 30), NGram(train, 6, 30, only=val)
    for a, b in zip(small.logprobs(val), full.logprobs(val)):
        assert a == pytest.approx(b)
    assert sum(len(c) for c in small.next[1:]) < sum(len(c) for c in full.next[1:])


def test_logprobs_match_prob_for_every_order():
    m = NGram(wordy_text(8, 1000), 4, 30)
    text = "the lord said"
    lps = m.logprobs(text)
    assert len(lps) == 4 and all(len(lp) == len(text) - 1 for lp in lps)
    for order in range(1, 5):
        for i in range(1, len(text)):
            assert lps[order - 1][i - 1] == pytest.approx(math.log(m.prob(text[max(0, i - 3) : i], text[i], order)))


def test_suffix_array_is_sorted_suffixes():
    rng = random.Random(0)
    for text in ["banana", "aaaa", "abcab", "".join(rng.choice("ab ") for _ in range(300))]:
        assert suffix_array(text).tolist() == sorted(range(len(text)), key=lambda i: text[i:])


def test_source_answers_like_a_string():
    rng = random.Random(1)
    src = Source(TRAIN)
    queries = ["", "the", "lord said", "zzz", TRAIN[100:160], TRAIN[100:160] + "q", TRAIN[-5:], "behold fear the"]
    queries += [TRAIN[i : i + rng.randint(1, 30)] + rng.choice("abt ") for i in range(0, 2000, 37)]
    for q in queries:
        assert (q in src) == (q in TRAIN), q
    generated = wordy_text(9, 80)
    assert memorisation(generated, src, 20) == memorisation(generated, TRAIN, 20)


def test_subsample_takes_whole_verses_up_to_size():
    verses = [f"verse number {i}" for i in range(100)]
    a = subsample(verses, 300, seed=0)
    picked = a.split("\n")
    assert all(v in verses for v in picked) and len(set(picked)) == len(picked)
    assert 300 <= len(a) + 1 < 300 + len("verse number 99") + 1
    assert subsample(verses, 300, seed=0) == a and subsample(verses, 300, seed=1) != a
    assert subsample(verses, 10**6, seed=0) == "\n".join(verses)


def test_markov_samples_follow_the_source_statistics():
    rng = random.Random(0)
    source = "ab" * 500 + "c" * 1000
    uni = markov_samples(source, 1, 3000, 1, rng)[0]
    counts = Counter(uni)
    assert len(uni) == 3000 and set(uni) == {"a", "b", "c"}
    assert counts["c"] / 3000 == pytest.approx(0.5, abs=0.05)
    two = markov_samples(source, 2, 3000, 2, rng)
    assert len(two) == 2 and two[0] != two[1]
    bi = two[0]
    seen = {source[i : i + 2] for i in range(len(source) - 1)}
    assert all(bi[i : i + 2] in seen for i in range(len(bi) - 1))  # every step follows a seen pair
    assert "ac" not in bi and "ba" in bi


def test_corrupt_changes_about_the_given_share():
    rng = random.Random(0)
    text = ("abcdefghij" * 99 + "\n") * 10
    out = corrupt(text, 0.1, ["X"], rng)
    assert len(out) == len(text) and out.count("\n") == text.count("\n")
    assert out.count("X") == pytest.approx(0.1 * len(text), rel=0.05)
