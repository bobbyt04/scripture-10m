"""The full-corpus model: word rules and verse vectors for all 38,039 verses.

The advice search never uses it to pick verses (it only searches the
hand-reviewed pool). It is used offline to propose cross-book parallels
(search/similarities.py), to score the published similarity pairs, to give
each pool verse its vector, and in the benchmark of similar().

The files are in search/model/ (written by `python search/build.py`):
model.json (word rules, vocabulary, refs), verse_vectors.i8 (int8[N][D]),
verse_terms.u16 and verse_offsets.u32 (each verse's term ids).
"""

import json
from pathlib import Path

import torch

from scripture.search import BOOKS, Lexicon

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "search" / "model"


def read_u16(path):
    """A little-endian uint16 file as int64 (torch has no uint16 reader, so read int16 and unwrap)."""
    raw = torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.int16)
    return raw.to(torch.int64) & 0xFFFF


class Corpus:
    """Verse vectors and TF-IDF over the whole of the three books."""

    def __init__(self, model, verse_vectors, verse_terms, verses=None):
        self.params = model["params"]
        self.lex = Lexicon(model["lexicon"])
        self.terms = model["terms"]
        self.n_vec = model["n_vec"]                    # terms 0 .. n_vec - 1 had a word vector
        self.word_term = dict(zip(model["words"], model["word_term"]))
        self.known = set(model["words"])
        self.books = {}
        start = 0
        for book in BOOKS:
            refs = model["books"][book]["refs"]
            n = len(refs)
            self.books[book] = {
                "refs": refs,
                "index": {r: i for i, r in enumerate(refs)},
                "text": verses[book] if verses else None,
                "vectors": _unit(verse_vectors[start:start + n]),
                "terms": verse_terms[start:start + n],
            }
            start += n
        df = torch.zeros(len(self.terms))
        for t in verse_terms:
            df[torch.unique(t)] += 1
        self.global_idf = torch.log(1 + start / (1 + df))

    @classmethod
    def load(cls, directory=MODEL, data_dir=ROOT / "data" / "processed"):
        d = Path(directory)
        model = json.loads((d / "model.json").read_text())
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
        return cls(model, vv.float().view(-1, model["dim"]), verse_terms, verses)

    def term_of(self, word):
        """Term id of a lower-case word, or None for filler or unknown words."""
        t = self.word_term.get(word)
        if t is None:
            t = self.word_term.get(self.lex.normalise(word, self.known))
        return None if t is None or t < 0 else t

    def similar(self, book, ref, k=5, books=BOOKS):
        """The verses of the whole corpus closest to one verse, per book (the verse itself excluded).

        score = w * cos(verse vectors) + (1 - w) * cos(TF-IDF), w = similar_vector_weight.
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
        lex = self.tfidf(src["terms"][i])
        lam = self.params["similar_vector_weight"]
        out = {}
        for other in books:
            b = self.books[other]
            s = lam * (b["vectors"] @ src["vectors"][i]) + (1 - lam) * self.tfidf_scores(lex, b)
            if other == book:
                s[i] = float("-inf")
            order = torch.sort(s, descending=True, stable=True).indices[:k]
            out[other] = [{"ref": b["refs"][j], "text": b["text"][j] if b["text"] else None, "score": s[j].item()}
                          for j in order.tolist()]
        return out

    def pair_score(self, a, b):
        """The similar() score of one pair, each given as 'book ref'."""
        (ba, ra), (bb, rb) = a.split(" ", 1), b.split(" ", 1)
        i, j = self.books[ba]["index"][ra], self.books[bb]["index"][rb]
        lam = self.params["similar_vector_weight"]
        vec = (self.books[ba]["vectors"][i] @ self.books[bb]["vectors"][j]).item()
        lex = self.tfidf_scores(self.tfidf(self.books[ba]["terms"][i]), self.books[bb])[j].item()
        return round(lam * vec + (1 - lam) * lex, 3)

    def tfidf(self, ids):
        tf = torch.bincount(ids, minlength=len(self.terms)).float()
        v = tf * self.global_idf
        return v / v.norm() if v.norm() > 0 else v

    def tfidf_scores(self, lex, b):
        """Cosine of `lex` with every verse of one book (sparse matrix, built once per book)."""
        if "tfidf" not in b:
            rows = torch.cat([torch.full((len(t),), i) for i, t in enumerate(b["terms"])])
            cols = torch.cat(b["terms"])
            m = torch.sparse_coo_tensor(torch.stack([rows, cols]), self.global_idf[cols],
                                        (len(b["terms"]), len(self.terms)), check_invariants=False).coalesce()
            norms = torch.zeros(len(b["terms"])).index_add_(0, m.indices()[0], m.values() ** 2).sqrt().clamp(min=1e-9)
            b["tfidf"] = (m, norms)
        m, norms = b["tfidf"]
        return torch.sparse.mm(m, lex.unsqueeze(1)).squeeze(1) / norms


def _unit(m):
    """Rows divided by their length; all-zero rows stay zero."""
    return m / m.norm(dim=1, keepdim=True).clamp(min=1e-9)
