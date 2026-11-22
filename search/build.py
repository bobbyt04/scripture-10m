"""Train the meaning-search model from the three books and write its artefacts.

    python search/build.py            # writes search/artifacts/ (about a minute on CPU)

Steps:
1. Word rules (search/lexicon.json). Stopwords are the words found in the
   largest share of verses, minus a hand-written list of content words, plus
   a hand-written list of fillers. Words that are always capitalised in the
   books (names such as Judas, Annas) are never shortened.
2. Vocabulary. Every word of the three books is reduced to a base form
   (Lexicon.normalise) and filler words are dropped. Each verse becomes a
   list of term ids.
3. Word vectors (PPMI + SVD). Count how often two terms appear within a few
   words of each other (inside one chapter), turn the counts into positive
   pointwise mutual information ("do these two words meet more often than
   chance?"), and compress that table with a truncated SVD. Terms used in
   similar contexts end up with similar vectors. The Gita and the Quran are
   much smaller than the Bible, so their counts are up-weighted so their
   word usage still shapes the vectors.
4. Verse vectors. A weighted average of the verse's word vectors (rare words
   count more), minus the direction shared by all verses, scaled to length 1.
5. Safety data (search/safety.json): the blocklist is turned into one bit per
   verse, the gentle verse set into verse numbers.
6. Artefacts, quantised to int8, with everything a browser port needs
   (search/FORMAT.md), plus similarities.json built from the hand-reviewed
   pairs in search/similarities_review.json and fixtures.json for a port's tests.

`--method sgns` trains the word vectors with skip-gram and negative sampling
instead (written here in PyTorch); the benchmark compares the two.
"""

import argparse
import base64
import hashlib
import json
import re
import sys
from array import array
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripture.search import BOOKS, Lexicon, Searcher, words  # noqa: E402

DATA = ROOT / "data" / "processed"
OUT = ROOT / "search" / "artifacts"
LEXICON = ROOT / "search" / "lexicon.json"
BRIDGE = ROOT / "search" / "bridge.json"
SAFETY = ROOT / "search" / "safety.json"

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
    "method": "bm25+bridge",  # what search() uses by default: as good as hybrid on dev, and simpler
    "alpha": 0.5,          # hybrid score = alpha * BM25 + (1 - alpha) * cosine (each scaled by its best)
    "bridge_weight": 0.8,  # weight of a bridge word relative to a word typed by the visitor
    "similar_vector_weight": 0.5,  # similar(): mix of vector cosine and TF-IDF cosine
}
SEARCH_PARAMS = ("sif_a", "bm25_k1", "bm25_b", "method", "alpha", "bridge_weight", "similar_vector_weight")


def load_json(path):
    return json.loads(Path(path).read_text())


def load_verses():
    verses = {}
    for book in BOOKS:
        with open(DATA / f"{book}.jsonl") as f:
            verses[book] = [json.loads(line) for line in f]
    return verses


def corpus_sha256(verses):
    """A checksum of the refs and texts the model was built from."""
    h = hashlib.sha256()
    for book in BOOKS:
        for v in verses[book]:
            h.update(f'{book}\t{v["ref"]}\t{v["text"]}\n'.encode())
    return h.hexdigest()


def derive_stopwords(verses, cfg):
    """Words in the largest share of verses (each book weighted equally), minus 'keep', plus 'add'."""
    share = Counter()
    for book in BOOKS:
        n = len(verses[book])
        df = Counter(w for v in verses[book] for w in set(words(v["text"])))
        for w, c in df.items():
            share[w] += c / n / len(BOOKS)
    top = [w for w, _ in sorted(share.items(), key=lambda x: (-x[1], x[0]))[:cfg["top_by_verse_share"]]]
    return sorted((set(top) - set(cfg["keep"])) | set(cfg["add"]))


def proper_names(verses):
    """Lower-cased words that are capitalised every time they occur (names such as Judas)."""
    caps, lower = set(), set()
    for book in BOOKS:
        for v in verses[book]:
            for w in re.findall(r"[A-Za-z]+", v["text"]):
                (caps if w[0].isupper() else lower).add(w.lower())
    return caps - lower


def build_lexicon(verses, cfg):
    return {
        "stopwords": derive_stopwords(verses, cfg["stopwords"]),
        "query_stopwords": sorted(cfg["query_stopwords"]),
        "irregular": cfg["irregular"],
        "suffix_rules": cfg["suffix_rules"],
        "not_inflected": sorted(set(cfg["not_inflected"]) | proper_names(verses)),
    }


def build_vocab(verses, min_count, lex):
    """Map every corpus word to a term; order terms by frequency."""
    raw = Counter(w for book in BOOKS for v in verses[book] for w in words(v["text"]))
    known = set(raw)
    base = {}
    for w in raw:
        t = lex.normalise(w, known)
        base[w] = None if (w in lex.stopwords or t in lex.stopwords) else t
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


def cooccurrence(streams, weights, n_vec, window):
    """Symmetric counts of terms at most `window` positions apart, each pair weighted by its book."""
    counts = torch.zeros(n_vec * n_vec)
    for book, ids in streams:
        for d in range(1, window + 1):
            if len(ids) <= d:
                break
            a, b = ids[:-d], ids[d:]
            w = torch.full((len(a),), weights[book])
            counts.index_add_(0, a * n_vec + b, w)
            counts.index_add_(0, b * n_vec + a, w)
    return counts.view(n_vec, n_vec)


def ppmi(c, cds):
    """Positive PMI: max(0, log P(w, c) / (P(w) P(c))), with the context distribution smoothed by cds."""
    row = c.sum(1, keepdim=True)
    col = c.sum(0, keepdim=True) ** cds
    return (c * col.sum() / (row * col).clamp(min=1e-12) + 1e-12).log().clamp(min=0)


def ppmi_svd_vectors(streams, weights, n_vec, p):
    """Word vectors from a truncated SVD of the PPMI co-occurrence matrix."""
    m = ppmi(cooccurrence(streams, weights, n_vec, p["window"]), p["cds"])
    torch.manual_seed(0)
    u, s, _ = torch.svd_lowrank(m, q=min(p["dim"], n_vec), niter=6)
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


# ----- safety data -----

def _ref_order(verses, book):
    return {v["ref"]: i for i, v in enumerate(verses[book])}


def blocklist(verses, safety):
    """One flag per verse (all books in order) and, per verse, why it is blocked ('' if not)."""
    rules = [(p["name"], re.compile(p["regex"], re.IGNORECASE if p.get("ignore_case", True) else 0))
             for p in safety["block"]["patterns"]]
    keep = {(k["book"], k["ref"]) for k in safety["block"]["keep"]}
    ranges = {}
    for r in safety["block"]["refs"]:
        order = _ref_order(verses, r["book"])
        if r["from"] in order and r["to"] in order:
            for i in range(order[r["from"]], order[r["to"]] + 1):
                ranges[(r["book"], i)] = r["reason"]
    reasons = []
    for book in BOOKS:
        for i, v in enumerate(verses[book]):
            why = ranges.get((book, i), "")
            if not why and (book, v["ref"]) not in keep:
                why = next((name for name, rx in rules if rx.search(v["text"])), "")
            reasons.append(why)
    return reasons


def gentle_indices(verses, safety):
    out = {}
    for book in BOOKS:
        order = _ref_order(verses, book)
        out[book] = [order[ref] for ref, _ in safety["gentle"][book] if ref in order]
    return out


def pack_bits(flags):
    raw = bytearray((len(flags) + 7) // 8)
    for i, f in enumerate(flags):
        if f:
            raw[i >> 3] |= 1 << (i & 7)
    return base64.b64encode(bytes(raw)).decode()


def bridge_tables(bridge):
    entries = {k: v for k, v in bridge.items() if not k.startswith("_")}
    return entries, sorted(bridge.get("_replace", []))


# ----- the whole model -----

def train(verses=None, params=None, method="ppmi", lexicon=None, bridge=None, safety=None):
    """Build the full-precision model. Returns (model dict, word vectors, verse vectors, verse terms)."""
    p = {**PARAMS, **(params or {})}
    verses = verses or load_verses()
    lexicon = lexicon or load_json(LEXICON)
    bridge = bridge or load_json(BRIDGE)
    safety = safety or load_json(SAFETY)
    lex_data = build_lexicon(verses, lexicon)
    vocab = build_vocab(verses, p["min_count"], Lexicon(lex_data))
    n_vec = vocab["n_vec"]
    streams = chapter_streams(verses, vocab["verse_terms"], n_vec)
    weights = book_weights(verses, vocab["verse_terms"], p["book_balance"])
    if method == "ppmi":
        wv = ppmi_svd_vectors(streams, weights, n_vec, p)
    else:
        wv = sgns_vectors(streams, weights, vocab["count"], n_vec, p)
    vv, principal = verse_vectors(vocab["verse_terms"], wv, vocab["count"], p["sif_a"])
    entries, replace = bridge_tables(bridge)
    model = {
        "version": 2,
        "dim": p["dim"],
        "method": method,
        "params": {k: p[k] for k in SEARCH_PARAMS},
        "lexicon": lex_data,
        "bridge": entries,
        "bridge_replace": replace,
        "safety": {"crisis": safety["crisis"], "sensitive": safety["sensitive"]},
        "gentle": gentle_indices(verses, safety),
        "blocked": pack_bits([bool(r) for r in blocklist(verses, safety)]),
        "words": vocab["words"],
        "word_term": vocab["word_term"],
        "terms": vocab["terms"],
        "count": vocab["count"],
        "principal": principal.tolist(),
        "books": {b: {"refs": [v["ref"] for v in verses[b]]} for b in BOOKS},
        "inputs": {
            "corpus_sha256": corpus_sha256(verses),
            "verses": {b: len(verses[b]) for b in BOOKS},
            "lexicon_sha256": hashlib.sha256(json.dumps(lexicon, sort_keys=True).encode()).hexdigest(),
            "bridge_sha256": hashlib.sha256(json.dumps(bridge, sort_keys=True).encode()).hexdigest(),
            "safety_sha256": hashlib.sha256(json.dumps(safety, sort_keys=True).encode()).hexdigest(),
        },
    }
    return model, wv, vv, vocab["verse_terms"]


def searcher(model, wv, vv, verse_terms, verses=None):
    """An in-memory Searcher (full precision), for tuning and tests."""
    verses = verses or load_verses()
    texts = {b: [v["text"] for v in verses[b]] for b in BOOKS}
    return Searcher(model, wv, vv, verse_terms, texts)


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


FIXTURE_QUERIES = [
    "I got laid off and I'm scared about money", "my girlfriend broke up with me", "I feel burnt out",
    "my dad has cancer", "I'm anxious about my exams", "I can't forgive my brother", "my mother passed away",
    "I feel like my life has no purpose", "I'm jealous of my friend's success", "I keep procrastinating",
    "I want to help the poor", "I'm grateful for my life", "I feel guilty about my past",
    "I want inner peace and a calm mind", "should I lie to my boss", "I'm drowning in debt",
    "the Lord is my shepherd", "love your enemies", "with hardship goeth ease",
    "action without attachment to its fruits", "I'm getting married next month", "I'm afraid of dying",
    "nobody appreciates my work", "I'm stressed out at work", "I'm gay", "I'm blind", "I want to die",
    "my husband hits me", "xyzzy", "!!!",
]


def fixtures(s):
    """Expected results for a browser port's tests."""
    cases = []
    for q in FIXTURE_QUERIES:
        r = s.search(q, k=3)
        cases.append({
            "query": q, "category": r["category"], "topic": r["topic"], "no_match": r["no_match"],
            "terms": sorted(s.terms[t] for t in s.expand(q)),
            "results": {b: [[h["ref"], h["score"]] for h in r["results"][b]] for b in BOOKS},
        })
    return {"about": "search(query, k=3) on the committed artefacts; scores rounded to 4 places. "
                     "A port should match the refs exactly and the scores to about 1e-3.", "cases": cases}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", choices=["ppmi", "sgns"], default="ppmi")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    torch.set_num_threads(3)
    model, wv, vv, vt = train(method=args.method)
    write(model, wv, vv, vt, args.out)
    from search.similarities import REVIEW, build_similarities  # noqa: E402
    s = Searcher.load(args.out)
    if REVIEW.exists():
        sims = build_similarities(s)
        (Path(args.out) / "similarities.json").write_text(json.dumps(sims, indent=1, ensure_ascii=False) + "\n")
    (Path(args.out) / "fixtures.json").write_text(json.dumps(fixtures(s), indent=1, ensure_ascii=False) + "\n")
    total = 0
    for f in sorted(Path(args.out).iterdir()):
        total += f.stat().st_size
        print(f"{f.name:>20}  {f.stat().st_size / 1e6:6.2f} MB")
    print(f"{'total':>20}  {total / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
