"""Meaning search over the Gita, the Quran and the Bible, trained from scratch.

A visitor types a situation in everyday English ("I got laid off and I'm
scared about money") and gets real verses from each of the three books.

How a query is answered:

1. Safety first. Hand-written phrase lists (search/safety.json) sort the
   query into "crisis" (suicide, self-harm, abuse, violence: no verses are
   returned, so the site can show help instead), "sensitive" (identity,
   disability, abortion and the like: only a hand-picked set of gentle verses
   can be returned) or "general".
2. Words. The query is split into words. Each word is reduced to a base form
   that occurs in the scriptures ("loveth", "loved", "loving" -> "love"), and
   filler words ("the", "thou", "unto") are dropped (search/lexicon.json).
3. Bridge. Many modern words never occur in old scripture English ("stressed",
   "girlfriend", "laid off"). A hand-written table, search/bridge.json, maps
   them to words that do ("troubled", "burden", "weary"). Those words are added
   to the query with a lower weight.
4. Scores. BM25 counts how many query words a verse contains, weighting rare
   words more (computed inside each book). Vector similarity compares the
   meaning of query and verse using word vectors trained on the three books
   (search/build.py). The two are mixed.
5. Results. The top verses of each book are returned separately, so the
   Bible (31,102 verses) cannot crowd out the Gita (701). Verses on the
   blocklist (violence, curses, polemic, punishment law and the like) are
   never returned for a situation. A query with no known words returns nothing.

Everything the scorer needs is in search/artifacts/ (see search/FORMAT.md).
Run `python -m scripture.search "your situation"` to try it.
"""

import base64
import json
import re
import sys
from pathlib import Path

import torch

BOOKS = ("gita", "quran", "bible")
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "search" / "artifacts"

# Which scoring pieces each method uses (the benchmark compares them).
METHODS = {
    "bm25": dict(lexical=True, vectors=False, bridge=False),
    "bm25+bridge": dict(lexical=True, vectors=False, bridge=True),
    "vectors": dict(lexical=False, vectors=True, bridge=True),
    "vectors-no-bridge": dict(lexical=False, vectors=True, bridge=False),
    "hybrid": dict(lexical=True, vectors=True, bridge=True),
}


def words(text):
    """Lower-case words, with modern contractions spelled out."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"\b(won't|can't|ain't)", lambda m: {"won't": "will not", "can't": "can not", "ain't": "is not"}[m.group(1)], t)
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"'(m|re|s|ve|ll|d)\b", " ", t)
    return re.findall(r"[a-z]+", t)


class Lexicon:
    """Stopwords, irregular forms and suffix rules (all data, from search/lexicon.json)."""

    def __init__(self, d):
        self.stopwords = set(d["stopwords"])
        self.query_stopwords = set(d["query_stopwords"])
        self.irregular = d["irregular"]
        self.rules = d["suffix_rules"]
        self.not_inflected = set(d["not_inflected"])

    def candidates(self, w):
        """Shorter forms a word may come from, most likely first (only the first matching rule is used)."""
        if w in self.not_inflected:
            return []
        for rule in self.rules:
            if len(w) < rule["min_len"] or not w.endswith(tuple(rule["ends"])) \
                    or w.endswith(tuple(rule.get("not_ends", []))):
                continue
            out = []
            for c in rule["try"]:
                if "only_if_ends" in c and not w.endswith(tuple(c["only_if_ends"])):
                    continue
                stem = w[:-c["strip"]]
                if c.get("undouble"):
                    if len(stem) < 2 or stem[-1] != stem[-2]:
                        continue
                    stem = stem[:-1]
                stem += c.get("add", "")
                if len(stem) >= 3:
                    out.append(stem)
            return out
        return []

    def normalise(self, word, known):
        """Reduce a lower-case word to a base form found in `known` (a set of words).

        The first candidate that is a known word wins, up to two steps
        ("blessings" -> "blessing" -> "bless"). If that base form is a
        stopword ("telling" -> "tell"), the result is that stopword, so the
        word is dropped as filler.
        """
        word = self.irregular.get(word, word)
        for _ in range(2):
            nxt = next((c for c in self.candidates(word) if c in known), None)
            if nxt is None:
                break
            word = nxt
            if word in self.stopwords:
                break
        return word


def compile_phrases(phrases):
    """'he|she hits me' -> [['he', 'she'], ['hits'], ['me']]; a trailing * matches a prefix."""
    return [[alt.split("|") for alt in p.split()] for p in phrases]


def phrase_in(tokens, phrase):
    def ok(tok, alts):
        return any(tok == a or (a.endswith("*") and tok.startswith(a[:-1])) for a in alts)
    n = len(phrase)
    return any(all(ok(tokens[i + j], phrase[j]) for j in range(n)) for i in range(len(tokens) - n + 1))


def read_u16(path):
    """A little-endian uint16 file as int64 (torch has no uint16 reader, so read int16 and unwrap)."""
    raw = torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.int16)
    return raw.to(torch.int64) & 0xFFFF


def unpack_bits(b64, n):
    """Bitmask (base64, bit i of byte i // 8, least significant bit first) -> bool tensor of length n."""
    raw = base64.b64decode(b64)
    return torch.tensor([(raw[i >> 3] >> (i & 7)) & 1 == 1 for i in range(n)], dtype=torch.bool)


class Searcher:
    """Answers queries from the artefacts written by search/build.py."""

    def __init__(self, model, word_vectors, verse_vectors, verse_terms, verses=None):
        self.params = model["params"]
        self.lex = Lexicon(model["lexicon"])
        self.terms = model["terms"]
        self.word_term = dict(zip(model["words"], model["word_term"]))
        self.known = set(model["words"])
        self.count = torch.tensor(model["count"], dtype=torch.float32)
        self.wv = _unit(word_vectors)                    # [terms with vectors, dim]
        self.principal = torch.tensor(model["principal"])
        self.bridge = model["bridge"]
        self.replace = set(model["bridge_replace"])
        self.max_phrase = max(len(k.split()) for k in self.bridge)
        self.crisis = {t: compile_phrases(p) for t, p in model["safety"]["crisis"].items()}
        self.sensitive = {t: compile_phrases(p) for t, p in model["safety"]["sensitive"].items()}
        n_all = sum(len(model["books"][b]["refs"]) for b in BOOKS)
        blocked = unpack_bits(model["blocked"], n_all)
        self.books = {}
        start = 0
        for book in BOOKS:
            refs = model["books"][book]["refs"]
            n = len(refs)
            terms_b = verse_terms[start:start + n]
            gentle = model["gentle"][book]
            prior = torch.zeros(n)
            for rank, i in enumerate(gentle):            # curated order breaks ties
                prior[i] = 0.01 * (1 - rank / len(gentle))
            self.books[book] = {
                "refs": refs,
                "index": {r: i for i, r in enumerate(refs)},
                "text": verses[book] if verses else None,
                "vectors": _unit(verse_vectors[start:start + n]),
                "terms": terms_b,
                "blocked": blocked[start:start + n],
                "gentle": prior > 0,
                "prior": prior,
                **_bm25_stats(terms_b, len(self.terms)),
            }
            start += n
        df = sum(b["df"] for b in self.books.values())
        self.global_idf = torch.log(1 + start / (1 + df))    # IDF over all three books, for similar()

    # ----- loading -----

    @classmethod
    def load(cls, directory=ARTIFACTS, data_dir=ROOT / "data" / "processed"):
        d = Path(directory)
        model = json.loads((d / "model.json").read_text())
        dim = model["dim"]
        wv = torch.frombuffer(bytearray((d / "word_vectors.i8").read_bytes()), dtype=torch.int8)
        vv = torch.frombuffer(bytearray((d / "verse_vectors.i8").read_bytes()), dtype=torch.int8)
        ids = read_u16(d / "verse_terms.u16")
        offsets = torch.frombuffer(bytearray((d / "verse_offsets.u32").read_bytes()), dtype=torch.int32).tolist()
        verse_terms = [ids[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
        verses = None
        if data_dir and Path(data_dir).exists():
            verses = {}
            for book in BOOKS:
                with open(Path(data_dir) / f"{book}.jsonl") as f:
                    verses[book] = [json.loads(line)["text"] for line in f]
        return cls(model, wv.float().view(-1, dim), vv.float().view(-1, dim), verse_terms, verses)

    # ----- queries -----

    def classify(self, query):
        """("crisis" | "sensitive" | "general", topic or None)."""
        toks = words(query)
        for category, lists in (("crisis", self.crisis), ("sensitive", self.sensitive)):
            for topic, phrases in lists.items():
                if any(phrase_in(toks, p) for p in phrases):
                    return category, topic
        return "general", None

    def term_of(self, word):
        """Term id of a lower-case word, or None for filler or unknown words."""
        t = self.word_term.get(word)
        if t is None:                                   # not in the books: try a base form that is
            t = self.word_term.get(self.lex.normalise(word, self.known))
        return None if t is None or t < 0 else t

    def expand(self, query, bridge=True):
        """Turn a query into weighted terms.

        Returns {term id: (weight, source word)}. Words found in the scriptures
        get weight 1; words added by the bridge get the bridge weight. A term
        reached twice keeps its highest weight.
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
            if bridge:                                   # longest bridge phrase starting here
                for n in range(min(self.max_phrase, len(toks) - i), 0, -1):
                    if " ".join(toks[i:i + n]) in self.bridge:
                        key = " ".join(toks[i:i + n])
                        break
            if key:
                for target in self.bridge[key]:
                    add(self.term_of(target), beta, key)
            # A phrase ("laid off") or a word whose scripture meaning differs
            # ("job" is the prophet Job) is searched only through its bridge words.
            if not key or (n == 1 and key not in self.replace):
                w = toks[i]
                if w not in self.lex.stopwords and w not in self.lex.query_stopwords:
                    add(self.term_of(w), 1.0, w)
            i += n if key else 1
        return out

    def scores(self, query, book, lexical=True, vectors=True, bridge=True):
        """Score every verse of one book (no safety rules). Returns (scores tensor, expanded query)."""
        q = self.expand(query, bridge=bridge)
        return self._scores(q, book, lexical, vectors), q

    def _scores(self, q, book, lexical=True, vectors=True):
        b = self.books[book]
        n = len(b["refs"])
        lex = self._bm25(q, b) if lexical else torch.zeros(n)
        vec = self._cosine(q, b) if vectors else torch.zeros(n)
        if lexical and vectors:
            a = self.params["alpha"]
            return a * _scale(lex) + (1 - a) * _scale(vec)
        return lex if lexical else vec

    def search(self, query, k=3, method=None):
        """Top-k verses from each book, after the safety rules.

        Returns {"category", "topic", "no_match", "results": {book: [hit, ...]}}.
        Each hit has ref, text, score (always > 0) and `matched`: the query
        words found in the verse, with the word of the query they came from.
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        method = method or self.params["method"]
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
        category, topic = self.classify(query)
        out = {"category": category, "topic": topic, "no_match": False, "results": {b: [] for b in BOOKS}}
        if category == "crisis":
            return out
        opts = METHODS[method]
        q = self.expand(query, bridge=opts["bridge"])
        for book in BOOKS:
            b = self.books[book]
            s = self._scores(q, book, opts["lexical"], opts["vectors"])
            allowed = ~b["blocked"]
            if category == "sensitive":
                allowed &= b["gentle"]
                s = s + b["prior"]
            s = torch.where(allowed, s, torch.full_like(s, float("-inf")))
            for score, i in _top(s, k):
                if not score > 0:
                    break
                present = set(b["terms"][i].tolist())
                matched = [{"word": self.terms[t], "from": src}
                           for t, (w, src) in sorted(q.items(), key=lambda x: -x[1][0]) if t in present]
                out["results"][book].append({
                    "ref": b["refs"][i],
                    "text": b["text"][i] if b["text"] else None,
                    "score": score,
                    "matched": matched,
                })
        out["no_match"] = not any(out["results"].values())
        return out

    def similar(self, book, ref, k=5, books=BOOKS, allow_blocked=False):
        """The verses closest in meaning to one verse, per book (the verse itself excluded).

        Blocklisted verses are left out unless allow_blocked=True.
        """
        if book not in self.books:
            raise ValueError(f"unknown book {book!r}; choose from {BOOKS}")
        if ref not in self.books[book]["index"]:
            raise ValueError(f"unknown verse {book} {ref!r}")
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        for other in books:
            if other not in self.books:
                raise ValueError(f"unknown book {other!r}; choose from {BOOKS}")
        src = self.books[book]
        i = src["index"][ref]
        v = src["vectors"][i]
        lex = self._tfidf(src["terms"][i])
        lam = self.params["similar_vector_weight"]
        out = {}
        for other in books:
            b = self.books[other]
            s = lam * (b["vectors"] @ v) + (1 - lam) * self._tfidf_scores(lex, b)
            if not allow_blocked:
                s = torch.where(b["blocked"], torch.full_like(s, float("-inf")), s)
            if other == book:
                s[i] = float("-inf")
            out[other] = [{"ref": b["refs"][j], "text": b["text"][j] if b["text"] else None, "score": x}
                          for x, j in _top(s, k) if x > float("-inf")]
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
        """Weighted sum of unit word vectors (weight x SIF), minus the shared direction, length 1."""
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
            m = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (len(b["terms"]), len(self.terms)),
                                        check_invariants=False).coalesce()
            norms = torch.zeros(len(b["terms"])).index_add_(0, m.indices()[0], m.values() ** 2).sqrt().clamp(min=1e-9)
            b["tfidf"] = (m, norms)
        m, norms = b["tfidf"]
        return torch.sparse.mm(m, lex.unsqueeze(1)).squeeze(1) / norms


def _top(s, k):
    """The k best (score, index) pairs. Scores are rounded to 4 decimals and ties go to the
    earlier verse, so every platform (and a port) returns the same order."""
    s = torch.round(s * 1e4) / 1e4
    order = torch.sort(s, descending=True, stable=True).indices[:k]
    return [(round(s[i].item(), 4), i.item()) for i in order]


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
    rows = torch.cat([torch.full((len(t),), i) for i, t in enumerate(verse_terms)])
    cols = torch.cat(verse_terms)
    keys, tf = torch.unique(rows * n_terms + cols, return_counts=True)
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
    res = Searcher.load().search(argv[0], k=k)
    print(f'category: {res["category"]}' + (f' ({res["topic"]})' if res["topic"] else ""))
    if res["category"] == "crisis":
        print("No verses for this one. If you are in danger or thinking of harming yourself, please reach out "
              "to someone you trust or a local emergency or crisis line now.")
        return
    if res["no_match"]:
        print("No matching verses: none of these words occur in the books or in the bridge.")
        return
    for book in BOOKS:
        print(f"\n== {book} ==")
        for h in res["results"][book]:
            why = ", ".join(m["word"] if m["word"] == m["from"] else f'{m["word"]} ({m["from"]})' for m in h["matched"])
            print(f'{h["ref"]:>22}  {h["score"]:.3f}  {h["text"]}')
            if why:
                print(f'{"":>30}matched: {why}')


if __name__ == "__main__":
    main(sys.argv[1:])
