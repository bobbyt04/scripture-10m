"""Meaning search over the Gita, the Quran and the Bible, trained from scratch.

A visitor types a situation in everyday English ("I got laid off and I'm
scared about money") and gets real verses from each of the three books.

How a query is scored, in four steps:

1. Words. The query is split into words. Each word is reduced to a base form
   that occurs in the scriptures ("loveth", "loved", "loving" -> "love"), and
   filler words ("the", "thou", "unto") are dropped.
2. Bridge. Many modern words never occur in old scripture English ("stressed",
   "girlfriend", "laid off"). A hand-written table, search/bridge.json, maps
   them to words that do ("troubled", "burden", "weary"). Those words are added
   to the query with a lower weight.
3. Two scores per verse. BM25 counts how many query words a verse contains,
   weighting rare words more (computed separately inside each book). Vector
   similarity compares the meaning of the query and the verse using word
   vectors trained on the three books in search/build.py.
4. Mix. The two scores are mixed, and the top verses of each book are returned
   separately, so the Bible (31,102 verses) cannot crowd out the Gita (701).

Everything the browser needs is in search/artifacts/ (see search/FORMAT.md).
Run `python -m scripture.search "your situation"` to try it.
"""

import json
import re
import sys
from pathlib import Path

import torch

BOOKS = ("gita", "quran", "bible")
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "search" / "artifacts"
BRIDGE = ROOT / "search" / "bridge.json"

# Filler words, written by hand: modern and archaic function words, plus a few
# very common scripture words that carry no meaning on their own ("lo", "unto").
STOPWORDS = set("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can cannot could did do does doing
down during each either else even ever every few for from further had has have
having he her here hers herself him himself his how i if in into is it its
itself just let me more most much must my myself neither no nor not now of off
oft often on once only or other our ours ourselves out over own same shall she
should so some such than that the their theirs them themselves then there these
they this those through thus to too under until up upon us very was we were what
when where whether which while who whom whose why will with would yet you your
yours yourself yourselves
thee thou thy thine ye unto hath hast doth dost art wast wert shalt wilt canst
couldest wouldest shouldest mayest didst hadst o oh lo verily yea nay behold
thereof therein thereon thereto thereby therefore wherefore whereof wherein
whereby whereupon hither thither whither hence thence whence ere amongst among
whatsoever whosoever whoso whomsoever wheresoever whithersoever whatever whoever
wherever however howbeit thereupon therewith wherewith herein hereby
also say said saith saying says came come cometh comest go goeth goest went gone
one two may might
like make made let
feel feeling felt think know knew really just get got getting gets keep
keeps kept always anymore lot lots stuff thing things something anything
everything someone anyone everyone maybe okay ok want wanted wants wanna gonna
going tell told im dont cant ive
""".split())

# Words dropped from what a visitor types, but kept in the scriptures, where
# they carry meaning ("a well of water", "the month of Ramadan").
QUERY_STOPWORDS = set("""
well next week weeks month months lately recently today tonight completely
totally literally actually basically seriously right
""".split())

# Words that end like "-est" or "-eth" but are not verb forms (modest, priest).
NOT_INFLECTED = set("""
modest honest earnest interest harvest request priest guest chest conquest
tempest manifest unmanifest protest behest nest rest best west test jest quest
arrest invest contest suggest digest detest attest infest molest crest wrest
blest teeth
""".split())

# Archaic forms that suffix rules cannot handle.
IRREGULAR = {
    "hath": "have", "hast": "have", "hadst": "have", "doth": "do", "dost": "do",
    "doeth": "do", "doest": "do", "didst": "do", "saith": "say", "sayest": "say",
    "spake": "speak", "art": "are", "wast": "was", "wert": "were", "shalt": "shall",
    "wilt": "will", "canst": "can",
}


def candidates(word):
    """Shorter forms a word may come from, most likely first.

    These are only candidates: normalise() keeps the first one that is a real
    word in the scriptures, so "loving" -> "love" but "evening" stays as is.
    """
    w = word
    out = []
    if w in NOT_INFLECTED:
        pass
    elif len(w) > 4 and (w.endswith("eth") or w.endswith("est")):
        if w[-4] == "i":
            out.append(w[:-4] + "y")                     # crieth -> cry
        out += [w[:-3], w[:-3] + "e"]                    # walketh -> walk, giveth -> give
        if len(w) > 5 and w[-4] == w[-5]:
            out.append(w[:-4])                           # sinneth -> sin
    elif len(w) > 4 and w.endswith("ing"):
        out += [w[:-3], w[:-3] + "e"]                    # walking, loving
        if w[-4] == w[-5]:
            out.append(w[:-4])                           # sinning
        if w.endswith("ying"):
            out.append(w[:-4] + "ie")                    # dying -> die
    elif len(w) > 3 and w.endswith("ed") and not w.endswith("eed"):
        if w.endswith("ied"):
            out.append(w[:-3] + "y")                     # cried -> cry
        out += [w[:-1], w[:-2]]                          # loved -> love, walked -> walk
        if len(w) > 4 and w[-3] == w[-4]:
            out.append(w[:-3])                           # sinned -> sin
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        if w.endswith("ies"):
            out.append(w[:-3] + "y")                     # cries -> cry
        out.append(w[:-1])                               # loves -> love
        if w.endswith("es"):
            out.append(w[:-2])                           # riches -> rich
    return [c for c in out if len(c) >= 3 and c not in STOPWORDS]


def normalise(word, known):
    """Reduce a lower-case word to a base form found in `known` (a set of words).

    Up to two steps, so "blessings" -> "blessing" -> "bless".
    """
    word = IRREGULAR.get(word, word)
    for _ in range(2):
        for c in candidates(word):
            if c in known:
                word = c
                break
        else:
            break
    return word


def words(text):
    """Lower-case words, with modern contractions spelled out."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"\b(won't|can't|ain't)", lambda m: {"won't": "will not", "can't": "can not", "ain't": "is not"}[m.group(1)], t)
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"'(m|re|s|ve|ll|d)\b", " ", t)
    return re.findall(r"[a-z]+", t)


class Searcher:
    """Loads the artefacts written by search/build.py and answers queries."""

    def __init__(self, model, word_vectors, verse_vectors, verse_terms, bridge, verses):
        self.params = model["params"]
        self.terms = model["terms"]
        self.word_term = dict(zip(model["words"], model["word_term"]))
        self.known = set(model["words"])
        self.count = torch.tensor(model["count"], dtype=torch.float32)
        self.wv = _unit(word_vectors)                    # [terms with vectors, dim]
        self.principal = torch.tensor(model["principal"])
        self.bridge = {k.lower(): v for k, v in bridge.items() if not k.startswith("_")}
        self.replace = set(bridge.get("_replace", []))
        self.max_phrase = max(len(k.split()) for k in self.bridge)
        self.books = {}
        start = 0
        for book in BOOKS:
            info = model["books"][book]
            n = len(info["refs"])
            terms_b = verse_terms[start:start + n]
            self.books[book] = {
                "refs": info["refs"],
                "index": {r: i for i, r in enumerate(info["refs"])},
                "text": verses[book] if verses else None,
                "vectors": _unit(verse_vectors[start:start + n]),
                "terms": terms_b,
                **_bm25_stats(terms_b, len(self.terms)),
            }
            start += n
        df = sum(b["df"] for b in self.books.values())
        self.global_idf = torch.log(1 + start / (1 + df))    # IDF over all three books, for similar()

    # ----- loading -----

    @classmethod
    def load(cls, directory=ARTIFACTS, bridge_path=BRIDGE, data_dir=ROOT / "data" / "processed"):
        d = Path(directory)
        model = json.loads((d / "model.json").read_text())
        dim = model["dim"]
        wv = torch.frombuffer(bytearray((d / "word_vectors.i8").read_bytes()), dtype=torch.int8)
        vv = torch.frombuffer(bytearray((d / "verse_vectors.i8").read_bytes()), dtype=torch.int8)
        ids = torch.frombuffer(bytearray((d / "verse_terms.u16").read_bytes()), dtype=torch.int16)
        offsets = torch.frombuffer(bytearray((d / "verse_offsets.u32").read_bytes()), dtype=torch.int32)
        ids = ids.to(torch.int64) & 0xFFFF
        offsets = offsets.tolist()
        verse_terms = [ids[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
        bridge = json.loads(Path(bridge_path).read_text())
        verses = None
        if data_dir and Path(data_dir).exists():
            verses = {}
            for book in BOOKS:
                with open(Path(data_dir) / f"{book}.jsonl") as f:
                    verses[book] = [json.loads(line)["text"] for line in f]
        return cls(model, wv.float().view(-1, dim), vv.float().view(-1, dim), verse_terms, bridge, verses)

    # ----- queries -----

    def term_of(self, word):
        """Term id of a lower-case word, or None for filler or unknown words."""
        t = self.word_term.get(word)
        if t is None:                                   # not in the books: try a base form that is
            t = self.word_term.get(normalise(word, self.known))
        return None if t is None or t < 0 else t

    def expand(self, query, bridge=True):
        """Turn a query into weighted terms.

        Returns {term id: (weight, source word)}. Words found in the scriptures
        get weight 1; words added by the bridge get the bridge weight.
        """
        toks = words(query)
        out = {}

        def add(term, weight, source):
            if term is not None and weight > out.get(term, (0, ""))[0]:
                out[term] = (weight, source)

        i = 0
        beta = self.params["bridge_weight"]
        while i < len(toks):
            key, n = None, 1
            if bridge:
                for n in range(min(self.max_phrase, len(toks) - i), 0, -1):
                    key = self._bridge_key(toks[i:i + n])
                    if key:
                        break
            if key:
                for target in self.bridge[key]:
                    add(self.term_of(target), beta, key)
            # A phrase ("laid off") or a word whose scripture meaning differs
            # ("job" is the prophet Job) is searched only through its bridge words.
            if not key or (n == 1 and key not in self.replace):
                if toks[i] not in STOPWORDS and toks[i] not in QUERY_STOPWORDS:
                    add(self.term_of(toks[i]), 1.0, toks[i])
            i += n if key else 1
        return out

    def _bridge_key(self, toks):
        phrase = " ".join(toks)
        if phrase in self.bridge:
            return phrase
        if len(toks) == 1:
            for c in candidates(toks[0]):
                if c in self.bridge:
                    return c
        return None

    def scores(self, query, book, lexical=True, vectors=True, bridge=True):
        """Score every verse of one book. Returns (scores tensor, expanded query)."""
        q = self.expand(query, bridge=bridge)
        b = self.books[book]
        n = len(b["refs"])
        lex = self._bm25(q, b) if lexical else torch.zeros(n)
        vec = self._cosine(q, b) if vectors else torch.zeros(n)
        if lexical and vectors:
            a = self.params["alpha"]
            s = a * _scale(lex) + (1 - a) * _scale(vec)
        else:
            s = lex if lexical else vec
        return s, q

    def search(self, query, k=3, method="hybrid"):
        """Top-k verses from each book: {"gita": [hit, ...], "quran": [...], "bible": [...]}.

        Each hit has ref, text, score, and `matched`: the query words found in
        the verse, with the word of the query they came from.
        """
        opts = METHODS[method]
        out = {}
        for book in BOOKS:
            s, q = self.scores(query, book, **opts)
            top = torch.topk(s, min(k, len(s)))
            hits = []
            for score, i in zip(top.values.tolist(), top.indices.tolist()):
                b = self.books[book]
                present = set(b["terms"][i].tolist())
                matched = [{"word": self.terms[t], "from": src}
                           for t, (w, src) in sorted(q.items(), key=lambda x: -x[1][0]) if t in present]
                hits.append({
                    "ref": b["refs"][i],
                    "text": b["text"][i] if b["text"] else None,
                    "score": round(score, 4),
                    "matched": matched,
                })
            out[book] = hits
        return out

    def similar(self, book, ref, k=5, books=BOOKS):
        """The verses closest in meaning to one verse, per book (the verse itself excluded)."""
        src = self.books[book]
        i = src["index"][ref]
        v = src["vectors"][i]
        lex = self._tfidf(src["terms"][i])
        lam = self.params["similar_vector_weight"]
        out = {}
        for other in books:
            b = self.books[other]
            s = lam * (b["vectors"] @ v) + (1 - lam) * self._tfidf_scores(lex, b)
            if other == book:
                s[i] = -1
            top = torch.topk(s, min(k, len(s)))
            out[other] = [{"ref": b["refs"][j], "text": b["text"][j] if b["text"] else None,
                           "score": round(x, 4)} for x, j in zip(top.values.tolist(), top.indices.tolist())]
        return out

    # ----- scoring pieces -----

    def _bm25(self, q, b):
        k1, bb = self.params["bm25_k1"], self.params["bm25_b"]
        s = torch.zeros(len(b["refs"]))
        for t, (w, _) in q.items():
            rows = b["postings"].get(t)
            if rows is None:
                continue
            docs, tf = rows
            idf = b["idf"][t]
            norm = tf + k1 * (1 - bb + bb * b["length"][docs] / b["avgdl"])
            s.index_add_(0, docs, w * idf * tf * (k1 + 1) / norm)
        return s

    def query_vector(self, q):
        v = torch.zeros(self.wv.shape[1])
        for t, (w, _) in q.items():
            if t < self.wv.shape[0]:
                v += w * self.sif(t) * self.wv[t]
        v -= (v @ self.principal) * self.principal
        return v / v.norm() if v.norm() > 0 else v

    def sif(self, t):
        a = self.params["sif_a"]
        return a / (a + self.count[t] / self.count.sum())

    def _cosine(self, q, b):
        return b["vectors"] @ self.query_vector(q)

    def _tfidf(self, ids):
        tf = torch.bincount(ids, minlength=len(self.terms)).float()
        v = tf * self.global_idf
        return v / v.norm() if v.norm() > 0 else v

    def _tfidf_scores(self, lex, b):
        if "tfidf" not in b:
            rows = torch.cat([torch.full((len(t),), i) for i, t in enumerate(b["terms"])])
            cols = torch.cat(b["terms"])
            vals = self.global_idf[cols]
            m = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (len(b["terms"]), len(self.terms))).coalesce()
            norms = torch.zeros(len(b["terms"])).index_add_(0, m.indices()[0], m.values() ** 2).sqrt().clamp(min=1e-9)
            b["tfidf"] = (m, norms)
        m, norms = b["tfidf"]
        return torch.sparse.mm(m, lex.unsqueeze(1)).squeeze(1) / norms


# Which scoring pieces each method uses (the benchmark compares them).
METHODS = {
    "bm25": dict(lexical=True, vectors=False, bridge=False),
    "bm25+bridge": dict(lexical=True, vectors=False, bridge=True),
    "vectors": dict(lexical=False, vectors=True, bridge=True),
    "vectors-no-bridge": dict(lexical=False, vectors=True, bridge=False),
    "hybrid": dict(lexical=True, vectors=True, bridge=True),
}


def _unit(m):
    n = m.norm(dim=1, keepdim=True)
    return m / n.clamp(min=1e-9)


def _scale(s):
    """Divide by the best score so BM25 and cosine can be mixed."""
    top = s.max()
    return s / top if top > 0 else s


def _bm25_stats(verse_terms, n_terms):
    """Per-book document frequencies, verse lengths and postings for BM25."""
    length = torch.tensor([len(t) for t in verse_terms], dtype=torch.float32)
    rows = torch.cat([torch.full((len(t),), i) for i, t in enumerate(verse_terms)]) if verse_terms else torch.zeros(0, dtype=torch.long)
    cols = torch.cat(verse_terms) if verse_terms else torch.zeros(0, dtype=torch.long)
    pairs = torch.unique(rows * n_terms + cols, return_counts=True)
    keys, tf = pairs
    docs, terms = keys // n_terms, keys % n_terms
    df = torch.bincount(terms, minlength=n_terms).float()
    n = len(verse_terms)
    idf = torch.log(1 + (n - df + 0.5) / (df + 0.5))
    order = torch.argsort(terms, stable=True)
    docs, terms, tf = docs[order], terms[order], tf[order].float()
    postings = {}
    bounds = torch.searchsorted(terms, torch.arange(n_terms + 1))
    for t in torch.nonzero(df).squeeze(1).tolist():
        lo, hi = bounds[t].item(), bounds[t + 1].item()
        postings[t] = (docs[lo:hi], tf[lo:hi])
    return {"length": length, "avgdl": length.mean().item(), "df": df, "idf": idf, "postings": postings}


def main(argv):
    if not argv:
        print('usage: python -m scripture.search "describe your situation" [k]')
        return
    torch.set_num_threads(3)
    k = int(argv[1]) if len(argv) > 1 else 3
    s = Searcher.load()
    res = s.search(argv[0], k=k)
    for book in BOOKS:
        print(f"\n== {book} ==")
        for h in res[book]:
            why = ", ".join(m["word"] if m["word"] == m["from"] else f'{m["word"]} ({m["from"]})' for m in h["matched"])
            print(f'{h["ref"]:>22}  {h["score"]:.3f}  {h["text"]}')
            if why:
                print(f'{"":>30}matched: {why}')


if __name__ == "__main__":
    main(sys.argv[1:])
