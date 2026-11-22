"""Baselines and metrics for the evals. No trained model needed.

Text is handled as strings where every character is one token.
"""
import bisect
import math
import random
import re
from array import array
from collections import Counter

import torch

TAG = re.compile(r"<(?:gita|quran|bible)>")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")


class NGram:
    """Character n-gram model with interpolated Witten-Bell smoothing.

    The order-1 model is an add-one unigram over vocab_size symbols. Each higher order mixes its
    own counts with the order below:

        P(c | h) = (count(h c) + T(h) * P(c | shorter h)) / (count(h) + T(h))

    where T(h) is the number of different symbols seen after h. Histories never seen fall back to
    the shorter history. Every distribution sums to 1 over the vocabulary.

    With `only`, it keeps just the counts needed to score that one text, which saves memory on a
    large training text; its probabilities for that text are unchanged.
    """

    def __init__(self, text, n, vocab_size, only=None):
        self.n, self.vocab_size = n, vocab_size
        self.unigram, self.total = Counter(text), len(text)
        # per history length k: counts of (history + next symbol), how often each history is followed
        # by something, and by how many different symbols
        self.next, self.hist, self.types = [None], [None], [None]
        for k in range(1, n):
            grams = (text[i : i + k + 1] for i in range(len(text) - k))
            if only is not None:
                needed = {only[i : i + k] for i in range(len(only) - k + 1)}
                grams = (g for g in grams if g[:-1] in needed)
            nxt, hist, types = Counter(grams), Counter(), Counter()
            for g, c in nxt.items():
                hist[g[:-1]] += c
                types[g[:-1]] += 1
            self.next.append(nxt)
            self.hist.append(hist)
            self.types.append(types)

    def probs(self, history, c):
        """[P(c | history) for orders 1 to n], each order using the last order - 1 symbols of history."""
        p = (self.unigram[c] + 1) / (self.total + self.vocab_size)
        out = [p]
        for k in range(1, self.n):
            h = history[len(history) - k :]
            seen = self.hist[k][h] if len(history) >= k else 0
            if seen:
                t = self.types[k][h]
                p = (self.next[k][h + c] + t * p) / (seen + t)
            out.append(p)
        return out

    def prob(self, history, c, order=None):
        return self.probs(history, c)[(order or self.n) - 1]

    def logprobs(self, text, start=1):
        """For each order 1 to n, the log-probability of every symbol of text[start:] given the ones before."""
        out = [[] for _ in range(self.n)]
        for i in range(start, len(text)):
            for k, p in enumerate(self.probs(text[max(0, i - self.n + 1) : i], text[i])):
                out[k].append(math.log(p))
        return out

    def nll(self, text, order=None, start=1):
        """Mean negative log-likelihood in nats of text[start:]."""
        lp = self.logprobs(text, start)[(order or self.n) - 1]
        return -sum(lp) / len(lp)


def suffix_array(text):
    """The start positions of all suffixes of text, in sorted order.

    Prefix doubling: rank the suffixes by their first character, then repeatedly by their first 2k
    characters, using the ranks for the first k at i and at i + k, until no two ranks are equal.
    """
    n = len(text)
    codes = torch.frombuffer(bytearray(text, "utf-32-le"), dtype=torch.int32)
    rank = torch.unique(codes, return_inverse=True)[1]  # 0, 1, 2, ... in character order
    k = 1
    while True:
        after = torch.zeros(n, dtype=torch.long)  # rank of the suffix k further on; 0 when it runs past the end
        after[: n - k] = rank[k:] + 1
        key = rank * (n + 1) + after
        sa = torch.argsort(key)
        new = torch.cat([torch.zeros(1, dtype=torch.long), (key[sa][1:] != key[sa][:-1]).long()]).cumsum(0)
        rank = torch.empty_like(rank)
        rank[sa] = new
        if new[-1] == n - 1:
            return sa
        k *= 2


class Source:
    """A text that answers `s in source` like the string itself, by binary search in its suffix array
    instead of scanning the whole text."""

    def __init__(self, text):
        self.text = text
        self.sa = array("q", suffix_array(text).tolist())

    def __contains__(self, s):
        text, sa = self.text, self.sa
        i = bisect.bisect_left(range(len(sa)), s, key=lambda j: text[sa[j] : sa[j] + len(s)])
        return i < len(sa) and text.startswith(s, sa[i])


def memorisation(generated, source, window=50):
    """How much of `generated` is copied verbatim from `source` (a string or a Source).

    Returns, for each window-length substring of `generated`, whether it occurs in the source
    ("flags"), how many do, and the longest substring of `generated` that occurs in the source.
    """
    i, flags, best = 0, [], (0, 0)
    for j in range(1, len(generated) + 1):
        # move i forward until generated[i:j] is the longest suffix of generated[:j] found in the source
        while generated[i:j] not in source:
            i += 1
        if j >= window:
            flags.append(j - i >= window)
        if j - i > best[1] - best[0]:
            best = (i, j)
    return {"flags": flags, "windows": len(flags), "memorised": sum(flags),
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


def subsample(verses, n_chars, seed):
    """Random whole verses, joined by newlines, until there are at least n_chars characters."""
    order = list(range(len(verses)))
    random.Random(seed).shuffle(order)
    picked, size = [], 0
    for i in order:
        if size >= n_chars:
            break
        picked.append(i)
        size += len(verses[i]) + 1
    return "\n".join(verses[i] for i in sorted(picked))


def markov_samples(text, order, length, count, rng):
    """`count` random texts with the character statistics of `text`: order 1 draws each character from
    the unigram counts, order 2 from the counts of characters seen after the previous one, and so on."""
    follow = {}
    for i in range(order - 1, len(text)):
        follow.setdefault(text[i - order + 1 : i], Counter())[text[i]] += 1
    choices = {ctx: (list(c), list(c.values())) for ctx, c in follow.items()}
    contexts = list(follow)
    out = []
    for _ in range(count):
        sample = ""
        while len(sample) < length:
            nxt = choices.get(sample[len(sample) - order + 1 :]) if len(sample) >= order - 1 else None
            if nxt:
                sample += rng.choices(nxt[0], weights=nxt[1])[0]
            else:  # at the start, or after a context seen only at the very end of the text
                sample += rng.choice(contexts)
        out.append(sample[:length])
    return out


def corrupt(text, frac, charset, rng):
    """Replace a random `frac` of the characters (not newlines) with random characters from charset."""
    chars = list(text)
    for i in rng.sample(range(len(chars)), round(frac * len(chars))):
        if chars[i] != "\n":
            chars[i] = rng.choice(charset)
    return "".join(chars)
