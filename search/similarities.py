"""Cross-book parallels for the website's similarities section.

Two steps:

1. The model proposes candidates (`propose`). For each pair of books it
   compares every verse of one with every verse of the other (vector cosine
   mixed with TF-IDF cosine, as in Corpus.similar), takes each verse's best
   match, and keeps pairs that share several informative words or are very
   close in vector space. Refrains and very short verses are left out.
2. A person reads every candidate, and every hand-listed parallel of
   evals/search_parallels.json. search/similarities_review.json records each
   decision; kept pairs get a theme and a one-line note on the shared idea.
   Pairs with threatening or theologically contentious verses are left out,
   and each verse appears in at most one published pair. Only kept pairs
   are published, in search/artifacts/similarities.json, with the model's
   score and the share of candidates that were kept.

    python search/similarities.py --propose    # rewrite the candidate list for review
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripture.search import BOOKS  # noqa: E402
from search.corpus import Corpus  # noqa: E402

REVIEW = ROOT / "search" / "similarities_review.json"
PARALLELS = ROOT / "evals" / "search_parallels.json"
PAIRS = (("bible", "quran"), ("bible", "gita"), ("quran", "gita"))
POOL = {("bible", "quran"): 160, ("bible", "gita"): 70, ("quran", "gita"): 70}
MIN_TERMS = 5           # verses with fewer content words are too thin to compare
MIN_SHARED = 2          # informative words both verses use
COMMON = 150            # the most frequent terms (lord, god, people, ...) are not informative
VECTOR_ONLY = 0.55      # or: this close in vector space, whatever the shared words
MAX_PER_CHAPTER = 3     # keep the candidate list varied


def _key(book, ref):
    return f"{book} {ref}"


def _informative(c, book, i):
    return {t for t in c.books[book]["terms"][i].tolist() if COMMON <= t < c.n_vec}


def _best_matches(c, a, b, lam, chunk=2048):
    """For every verse of a: best verse of b and the score; and the reverse."""
    va, vb = c.books[a]["vectors"], c.books[b]["vectors"]
    c.tfidf_scores(torch.zeros(len(c.terms)), c.books[a])      # build the cached TF-IDF matrices
    c.tfidf_scores(torch.zeros(len(c.terms)), c.books[b])
    ta, na = c.books[a]["tfidf"]
    tb, nb = c.books[b]["tfidf"]
    tb_t = tb.t().coalesce()
    ab = (torch.full((len(va),), -1.0), torch.zeros(len(va), dtype=torch.long))
    ba = (torch.full((len(vb),), -1.0), torch.zeros(len(vb), dtype=torch.long))
    for lo in range(0, len(va), chunk):
        hi = min(lo + chunk, len(va))
        sub = torch.index_select(ta, 0, torch.arange(lo, hi))
        lex = torch.sparse.mm(sub, tb_t).to_dense() / na[lo:hi, None] / nb[None, :]
        sim = lam * (va[lo:hi] @ vb.t()) + (1 - lam) * lex
        v, i = sim.max(1)
        ab[0][lo:hi], ab[1][lo:hi] = v, i
        v, i = sim.max(0)
        better = v > ba[0]
        ba[0][better], ba[1][better] = v[better], i[better] + lo
    return ab, ba


def _refrains(c, book):
    """Verses whose set of content words occurs 3+ times in the book (refrains, formulas)."""
    keys = [frozenset(t.tolist()) for t in c.books[book]["terms"]]
    n = Counter(keys)
    return [n[k] >= 3 for k in keys]


def propose(c):
    """Candidate pairs for review, best first per pair of books."""
    lam = c.params["similar_vector_weight"]
    out = []
    for a, b in PAIRS:
        (sab, iab), (sba, iba) = _best_matches(c, a, b, lam)
        ok = {}
        for book in (a, b):
            refrain = _refrains(c, book)
            ok[book] = [len(_informative(c, book, i)) >= MIN_TERMS - 2 and len(set(t.tolist())) >= MIN_TERMS
                        and not refrain[i] for i, t in enumerate(c.books[book]["terms"])]
        cands = {(i, iab[i].item()): sab[i].item() for i in range(len(sab))}
        cands.update({(iba[j].item(), j): sba[j].item() for j in range(len(sba))})
        rows = []
        for (i, j), score in cands.items():
            if not (ok[a][i] and ok[b][j]):
                continue
            shared = _informative(c, a, i) & _informative(c, b, j)
            cos = (c.books[a]["vectors"][i] @ c.books[b]["vectors"][j]).item()
            if len(shared) >= MIN_SHARED or cos >= VECTOR_ONLY:
                rows.append((score, i, j, shared))
        rows.sort(key=lambda r: -r[0])
        per_ch, used, kept = Counter(), set(), 0
        for score, i, j, shared in rows:
            ra, rb = c.books[a]["refs"][i], c.books[b]["refs"][j]
            ca, cb = (a, ra.rsplit(":", 1)[0]), (b, rb.rsplit(":", 1)[0])
            if (a, i) in used or (b, j) in used or per_ch[ca] >= MAX_PER_CHAPTER or per_ch[cb] >= MAX_PER_CHAPTER:
                continue
            used |= {(a, i), (b, j)}
            per_ch[ca] += 1
            per_ch[cb] += 1
            out.append({"a": _key(a, ra), "b": _key(b, rb), "score": round(score, 3),
                        "shared_words": sorted(c.terms[t] for t in shared)})
            kept += 1
            if kept >= POOL[(a, b)]:
                break
    return out


def _verses(c, keys):
    out = []
    for key in keys:
        book, ref = key.split(" ", 1)
        out.append({"book": book, "ref": ref, "text": c.books[book]["text"][c.books[book]["index"][ref]]})
    return out


def build_similarities(c, review_path=REVIEW, parallels_path=PARALLELS):
    """similarities.json from the reviewed candidates and the reviewed hand-listed parallels."""
    review = json.loads(Path(review_path).read_text())
    themes = {}

    def publish(a, b, theme, idea, source):
        themes.setdefault(theme, []).append({"idea": idea, "score": c.pair_score(a, b), "source": source,
                                             "verses": _verses(c, (a, b))})

    for cand in review["candidates"]:
        if cand["keep"]:
            publish(cand["a"], cand["b"], cand["theme"], cand["idea"], "model, reviewed")
    decisions = review["hand_listed"]
    for p in json.loads(Path(parallels_path).read_text())["parallels"]:
        d = decisions[p["id"]]
        if d["keep"]:
            publish(_key(p["a"]["book"], p["a"]["refs"][0]), _key(p["b"]["book"], p["b"]["refs"][0]),
                    d["theme"], d.get("idea", p["label"] + "."), "hand-listed")
    n = len(review["candidates"])
    kept = sum(cand["keep"] for cand in review["candidates"])
    per_pair = Counter("-".join(sorted((cand["a"].split()[0], cand["b"].split()[0])))
                       for cand in review["candidates"] if cand["keep"])
    return {
        "about": ("Parallels between the books, chosen by a person. Two sources: (1) 'model, reviewed': the search "
                  "model proposed candidate pairs (each verse's closest verse in another book); a person read every "
                  "candidate and kept only real parallels in story or teaching, wrote the one-line note and chose "
                  "the theme; (2) 'hand-listed': known parallels from evals/search_parallels.json, each read and kept "
                  "or left out. Pairs with threatening or theologically contentious verses are left out, and each "
                  "verse is in at most one pair. 'score' is the model's similarity for the pair."),
        "candidates_proposed": n,
        "kept": kept,
        "kept_fraction": round(kept / n, 3) if n else None,
        "kept_per_book_pair": dict(sorted(per_pair.items())),
        "hand_listed_kept": sum(d["keep"] for d in decisions.values()),
        "hand_listed_left_out": sum(not d["keep"] for d in decisions.values()),
        "themes": [{"name": name, "pairs": sorted(pairs, key=lambda p: -p["score"])}
                   for name, pairs in themes.items()],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--propose", action="store_true", help="rewrite the candidates in the review file "
                    "(decisions already made for the same pair are kept)")
    args = ap.parse_args()
    torch.set_num_threads(3)
    c = Corpus.load()
    if args.propose:
        review = json.loads(REVIEW.read_text()) if REVIEW.exists() else {"hand_listed": {}}
        old = {(x["a"], x["b"]): x for x in review.get("candidates", [])}
        cands = []
        for x in propose(c):
            prev = old.get((x["a"], x["b"]), {})
            cands.append({**x, "keep": prev.get("keep", False), "theme": prev.get("theme", ""),
                          "idea": prev.get("idea", "")})
        review["candidates"] = cands
        REVIEW.write_text(json.dumps(review, indent=1, ensure_ascii=False) + "\n")
        print(len(cands), "candidates written to", REVIEW)


if __name__ == "__main__":
    main()
