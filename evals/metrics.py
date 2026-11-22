"""Baselines and metrics for the evals. Plain Python, no model needed.

Text is handled as strings where every character is one token.
"""
import math
import re
from collections import Counter

TAG = re.compile(r"<(?:gita|quran|bible)>")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")


class NGram:
    """Character n-gram model with interpolated Witten-Bell smoothing.

    The order-1 model is an add-one unigram over vocab_size symbols. Each higher order mixes its
    own counts with the order below:

        P(c | h) = (count(h c) + T(h) * P(c | shorter h)) / (count(h) + T(h))

    where T(h) is the number of different symbols seen after h. Histories never seen fall back to
    the shorter history. Every distribution sums to 1 over the vocabulary.
    """

    def __init__(self, text, n, vocab_size):
        self.n, self.vocab_size = n, vocab_size
        self.grams = [None] + [Counter(text[i : i + k] for i in range(len(text) - k + 1)) for k in range(1, n + 1)]
        self.total = len(text)
        # for each history h of length k >= 1: how often it is followed by something, and by how many symbols
        self.hist, self.types = [None], [None]
        for k in range(2, n + 1):
            hist, types = Counter(), Counter()
            for g, c in self.grams[k].items():
                hist[g[:-1]] += c
                types[g[:-1]] += 1
            self.hist.append(hist)
            self.types.append(types)

    def prob(self, history, c, order=None):
        order = order or self.n
        p = (self.grams[1][c] + 1) / (self.total + self.vocab_size)
        for k in range(1, min(order, len(history) + 1)):
            h = history[len(history) - k :]
            seen = self.hist[k][h]
            if seen == 0:
                break
            t = self.types[k][h]
            p = (self.grams[k + 1][h + c] + t * p) / (seen + t)
        return p

    def nll(self, text, order=None, start=1):
        """Mean negative log-likelihood in nats of text[start:], each symbol given the ones before it."""
        order = order or self.n
        total = 0.0
        for i in range(start, len(text)):
            total -= math.log(self.prob(text[max(0, i - order + 1) : i], text[i], order))
        return total / (len(text) - start)


def memorisation(generated, train_text, window=50):
    """How much of `generated` is copied verbatim from `train_text`.

    Returns the number of window-length substrings of `generated`, how many of them occur in
    `train_text`, and the longest substring of `generated` that occurs in `train_text`.
    """
    i, hits, best = 0, 0, (0, 0)
    for j in range(1, len(generated) + 1):
        # move i forward until generated[i:j] is the longest suffix of generated[:j] found in the training text
        while generated[i:j] not in train_text:
            i += 1
        if j - i >= window:
            hits += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
    return {"windows": max(0, len(generated) - window + 1), "memorised": hits,
            "longest": best[1] - best[0], "longest_text": generated[best[0] : best[1]]}


def strip_tags(text):
    return TAG.sub("", text)


def words(text):
    return [w.lower() for w in WORD.findall(strip_tags(text))]


def complete_verses(sample):
    """Verses that the sample finished: every line except the last, without book tags."""
    return [strip_tags(line) for line in sample.split("\n")[:-1]]


def classify(text, models):
    """The key of the model in {book: NGram} that gives `text` the highest likelihood."""
    return min(models, key=lambda book: models[book].nll(text, start=0))
