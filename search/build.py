"""Train the meaning-search model from the three books and write its artefacts.

    python search/build.py            # writes search/artifacts/ (about a minute on CPU)

Steps:
1. Vocabulary. Every word of the three books is reduced to a base form
   (scripture.search.normalise) and filler words are dropped. Each verse
   becomes a list of term ids.
2. Word vectors (PPMI + SVD). Count how often two terms appear within a few
   words of each other (inside one chapter), turn the counts into positive
   pointwise mutual information ("do these two words meet more often than
   chance?"), and compress that table with a truncated SVD. Terms used in
   similar contexts end up with similar vectors. The Gita and the Quran are
   much smaller than the Bible, so their counts are up-weighted so their
   word usage still shapes the vectors.
3. Verse vectors. A weighted average of the verse's word vectors (rare words
   count more), minus the direction shared by all verses, scaled to length 1.
4. Artefacts, quantised to int8 (see search/FORMAT.md), plus
   search/artifacts/similarities.json: the strongest matches across books.

`--method sgns` trains the word vectors with skip-gram and negative sampling
instead (written here in PyTorch); the benchmark compares the two.
"""

import argparse
import json
import sys
from array import array
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripture.search import BOOKS, BRIDGE, STOPWORDS, Searcher, normalise, words  # noqa: E402

DATA = ROOT / "data" / "processed"
OUT = ROOT / "search" / "artifacts"

PARAMS = {
    "dim": 128,            # word and verse vector size
    "min_count": 3,        # terms seen fewer times get no vector (they still count for BM25)
    "window": 5,           # co-occurrence window, in content words
    "book_balance": 0.5,   # up-weight smaller books by (bible tokens / book tokens) ** this
    "cds": 0.75,           # context-distribution smoothing for PPMI
    "svd_power": 0.5,      # word vector = U * S ** this
    "sif_a": 1e-3,         # word weight a / (a + p(word)) when averaging
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
    "alpha": 0.5,          # hybrid score = alpha * BM25 + (1 - alpha) * cosine (each scaled by its best)
    "bridge_weight": 0.8,  # weight of a bridge word relative to a word typed by the visitor
    "similar_vector_weight": 0.5,  # similar(): mix of vector cosine and TF-IDF cosine
}


def load_verses():
    verses = {}
    for book in BOOKS:
        with open(DATA / f"{book}.jsonl") as f:
            verses[book] = [json.loads(line) for line in f]
    return verses


def build_vocab(verses, min_count):
    """Map every corpus word to a term; order terms by frequency."""
    raw = Counter(w for book in BOOKS for v in verses[book] for w in words(v["text"]))
    known = set(raw)
    base = {}
    for w in raw:
        t = normalise(w, known)
        base[w] = None if (w in STOPWORDS or t in STOPWORDS) else t
    count = Counter()
    for w, c in raw.items():
        if base[w]:
            count[base[w]] += c
    terms = sorted(count, key=lambda t: (-count[t], t))
    term_id = {t: i for i, t in enumerate(terms)}
    corpus_words = sorted(raw)
    word_term = [term_id[base[w]] if base[w] else -1 for w in corpus_words]
    n_vec = sum(1 for t in terms if count[t] >= min_count)
    lookup = dict(zip(corpus_words, word_term))
    verse_terms = [torch.tensor([lookup[w] for w in words(v["text"]) if lookup[w] >= 0], dtype=torch.long)
                   for book in BOOKS for v in verses[book]]
    return {
        "terms": terms, "count": [count[t] for t in terms], "words": corpus_words,
        "word_term": word_term, "n_vec": n_vec, "verse_terms": verse_terms,
    }


def chapter_streams(verses, verse_terms, n_vec):
    """One stream of term ids per chapter (terms with vectors only), with its book."""
    streams = []
    i = 0
    for book in BOOKS:
        current, key = [], None
        for v in verses[book]:
            k = (v["section"], v["chapter"])
            if k != key and current:
                streams.append((book, torch.cat(current)))
                current = []
            key = k
            t = verse_terms[i]
            current.append(t[t < n_vec])
            i += 1
        if current:
            streams.append((book, torch.cat(current)))
    return streams


def book_weights(verses, verse_terms, power):
    sizes, i = {}, 0
    for book in BOOKS:
        n = len(verses[book])
        sizes[book] = sum(len(t) for t in verse_terms[i:i + n])
        i += n
    return {b: (sizes["bible"] / sizes[b]) ** power for b in BOOKS}


def ppmi_svd_vectors(streams, weights, n_vec, p):
    """Word vectors from a truncated SVD of the PPMI co-occurrence matrix."""
    counts = torch.zeros(n_vec * n_vec)
    for book, ids in streams:
        for d in range(1, p["window"] + 1):
            if len(ids) <= d:
                break
            a, b = ids[:-d], ids[d:]
            w = torch.full((len(a),), weights[book])
            counts.index_add_(0, a * n_vec + b, w)
            counts.index_add_(0, b * n_vec + a, w)
    c = counts.view(n_vec, n_vec)
    row = c.sum(1, keepdim=True)
    col = c.sum(0, keepdim=True) ** p["cds"]
    # PMI = log P(w, c) / (P(w) P(c)), with the context distribution smoothed by cds.
    ppmi = c.mul_(col.sum()).div_(row * col).add_(1e-12).log_().clamp_(min=0)
    torch.manual_seed(0)
    u, s, _ = torch.svd_lowrank(ppmi, q=p["dim"], niter=6)
    return u * s ** p["svd_power"]


def sgns_vectors(streams, weights, counts, n_vec, p, epochs=3, neg=5, lr=0.002, batch=4096):
    """Word vectors from skip-gram with negative sampling (word2vec), in PyTorch."""
    torch.manual_seed(0)
    pairs, pw = [], []
    for book, ids in streams:
        for d in range(1, p["window"] + 1):
            if len(ids) <= d:
                break
            for a, b in ((ids[:-d], ids[d:]), (ids[d:], ids[:-d])):
                pairs.append(torch.stack([a, b], 1))
                pw.append(torch.full((len(a),), weights[book]))
    pairs, pw = torch.cat(pairs), torch.cat(pw)
    noise = torch.tensor(counts[:n_vec], dtype=torch.float32) ** 0.75
    emb_in = torch.nn.Embedding(n_vec, p["dim"])
    emb_out = torch.nn.Embedding(n_vec, p["dim"])
    torch.nn.init.uniform_(emb_in.weight, -0.5 / p["dim"], 0.5 / p["dim"])
    torch.nn.init.zeros_(emb_out.weight)
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=lr)
    for _ in range(epochs):
        order = torch.randperm(len(pairs))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            a, b, w = pairs[idx, 0], pairs[idx, 1], pw[idx]
            negs = torch.multinomial(noise, len(idx) * neg, replacement=True).view(len(idx), neg)
            va = emb_in(a)
            pos = torch.nn.functional.logsigmoid((va * emb_out(b)).sum(1))
            ng = torch.nn.functional.logsigmoid(-(emb_out(negs) @ va.unsqueeze(2)).squeeze(2)).sum(1)
            loss = -((pos + ng) * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return emb_in.weight.detach() + emb_out.weight.detach()


def verse_vectors(verse_terms, wv, counts, a):
    """SIF-weighted average of unit word vectors, minus the shared direction."""
    unit = wv / wv.norm(dim=1, keepdim=True).clamp(min=1e-9)
    p = torch.tensor(counts, dtype=torch.float32)
    sif = a / (a + p / p.sum())
    n_vec = len(wv)
    out = torch.zeros(len(verse_terms), wv.shape[1])
    for i, t in enumerate(verse_terms):
        t = t[t < n_vec]
        if len(t):
            out[i] = (sif[t].unsqueeze(1) * unit[t]).sum(0)
    _, _, v = torch.svd_lowrank(out, q=1, niter=6)
    principal = v[:, 0]
    out -= (out @ principal).unsqueeze(1) * principal
    return out / out.norm(dim=1, keepdim=True).clamp(min=1e-9), principal


def train(verses=None, params=None, method="ppmi"):
    """Build the full-precision model. Returns (model dict, word vectors, verse vectors, verse terms)."""
    p = {**PARAMS, **(params or {})}
    verses = verses or load_verses()
    vocab = build_vocab(verses, p["min_count"])
    n_vec = vocab["n_vec"]
    streams = chapter_streams(verses, vocab["verse_terms"], n_vec)
    weights = book_weights(verses, vocab["verse_terms"], p["book_balance"])
    if method == "ppmi":
        wv = ppmi_svd_vectors(streams, weights, n_vec, p)
    else:
        wv = sgns_vectors(streams, weights, vocab["count"], n_vec, p)
    vv, principal = verse_vectors(vocab["verse_terms"], wv, vocab["count"], p["sif_a"])
    model = {
        "version": 1,
        "dim": p["dim"],
        "method": method,
        "params": {k: p[k] for k in ("sif_a", "bm25_k1", "bm25_b", "alpha", "bridge_weight", "similar_vector_weight")},
        "words": vocab["words"],
        "word_term": vocab["word_term"],
        "terms": vocab["terms"],
        "count": vocab["count"],
        "principal": principal.tolist(),
        "books": {b: {"refs": [v["ref"] for v in verses[b]]} for b in BOOKS},
    }
    return model, wv, vv, vocab["verse_terms"]


def searcher(model, wv, vv, verse_terms, verses=None, bridge_path=BRIDGE):
    """An in-memory Searcher (full precision), for tuning and tests."""
    verses = verses or load_verses()
    texts = {b: [v["text"] for v in verses[b]] for b in BOOKS}
    return Searcher(model, wv, vv, verse_terms, json.loads(Path(bridge_path).read_text()), texts)


def quantise(m):
    """int8 per row: the row scaled so its largest entry is +-127 (only the direction is kept)."""
    scale = m.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    return torch.round(m / scale * 127).to(torch.int8)


def write(model, wv, vv, verse_terms, out=OUT):
    """Write the artefacts described in search/FORMAT.md (little-endian, no header)."""
    assert sys.byteorder == "little" and len(model["terms"]) < 65536
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    (out / "word_vectors.i8").write_bytes(array("b", quantise(wv).flatten().tolist()).tobytes())
    (out / "verse_vectors.i8").write_bytes(array("b", quantise(vv).flatten().tolist()).tobytes())
    (out / "verse_terms.u16").write_bytes(array("H", torch.cat(verse_terms).tolist()).tobytes())
    offsets = torch.tensor([0] + [len(t) for t in verse_terms]).cumsum(0)
    (out / "verse_offsets.u32").write_bytes(array("I", offsets.tolist()).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", choices=["ppmi", "sgns"], default="ppmi")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    torch.set_num_threads(3)
    model, wv, vv, vt = train(method=args.method)
    write(model, wv, vv, vt, args.out)
    from search.similarities import build_similarities  # noqa: E402
    s = Searcher.load(args.out)
    sims = build_similarities(s)
    (Path(args.out) / "similarities.json").write_text(json.dumps(sims, indent=1, ensure_ascii=False) + "\n")
    for f in sorted(Path(args.out).iterdir()):
        print(f"{f.name:>20}  {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
