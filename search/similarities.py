"""The strongest matches across the three books, for the website's similarities section.

For each pair of books we compare every verse of one with every verse of the
other (vector cosine mixed with TF-IDF cosine, as in Searcher.similar) and take
each verse's best match. Then we drop trivial matches (refrains, very short
verses, pairs held together by a single rare word), keep the list varied, and
sort the pairs under a handful of hand-named themes.

Called by search/build.py; writes search/artifacts/similarities.json.
"""

from collections import Counter

import torch

from scripture.search import BOOKS

# How many pairs to keep for each pair of books.
QUOTA = {("bible", "quran"): 150, ("bible", "gita"): 60, ("quran", "gita"): 60}
MIN_TERMS = 6           # verses with fewer content words are too thin to compare
MAX_PER_CHAPTER = 2     # at most this many pairs from one chapter on either side
MIN_SHARED = 2          # a pair must share at least two informative words (not one rare word)
COMMON = 60             # the most frequent terms (lord, god, people, ...) do not count as informative

# Themes, named and seeded by hand. Each pair goes to the theme whose seed
# words are closest to it in the word-vector space.
THEMES = {
    "Moses, Pharaoh and the Exodus": "moses pharaoh egypt staff rod sea serpent magicians israel",
    "Joseph and his brothers": "joseph brethren dream jacob kine prison brother",
    "Noah, Abraham and the prophets of old": "noah ark flood abraham isaac lot sodom jonah prophet",
    "Mary, Jesus and John": "mary jesus zacharias john angel child womb",
    "Creation and the signs of nature": "heavens earth created rain sun moon night stars mountains",
    "Mercy and forgiveness": "mercy forgive forgiveness merciful pardon repent sins",
    "Giving and the poor": "alms poor needy orphan give spend widow",
    "Patience, trust and hardship": "patience patient trust affliction endure steadfast",
    "Death, judgment and the life to come": "death die resurrection judgment hell paradise grave",
    "The soul and the Self": "soul self spirit body eternal birth born",
    "Devotion and worship": "worship pray prayer praise devotion sacrifice",
    "Desire, anger and self-control": "desire lust anger wrath senses greed restrain",
    "Wisdom and knowledge": "wisdom knowledge understanding wise fool ignorant",
    "Truth, justice and right conduct": "truth justice righteous judge measure balance",
}


def _tfidf_matrix(s, book):
    b = s.books[book]
    s._tfidf_scores(torch.zeros(len(s.terms)), b)          # builds and caches the sparse matrix
    m, norms = b["tfidf"]
    return m, norms


def _best_matches(s, a, b, lam, chunk=2048):
    """For every verse of a: best verse of b and its score; and the reverse."""
    va, vb = s.books[a]["vectors"], s.books[b]["vectors"]
    ta, na = _tfidf_matrix(s, a)
    tb, nb = _tfidf_matrix(s, b)
    tb_t = tb.t().coalesce()
    best_ab = torch.full((len(va),), -1.0), torch.zeros(len(va), dtype=torch.long)
    best_ba = torch.full((len(vb),), -1.0), torch.zeros(len(vb), dtype=torch.long)
    for lo in range(0, len(va), chunk):
        hi = min(lo + chunk, len(va))
        rows = torch.arange(lo, hi)
        sub = torch.index_select(ta, 0, rows)
        lex = torch.sparse.mm(sub, tb_t).to_dense() / na[lo:hi, None] / nb[None, :]
        sim = lam * (va[lo:hi] @ vb.t()) + (1 - lam) * lex
        v, i = sim.max(1)
        best_ab[0][lo:hi], best_ab[1][lo:hi] = v, i
        v, i = sim.max(0)
        better = v > best_ba[0]
        best_ba[0][better], best_ba[1][better] = v[better], i[better] + lo
    return best_ab, best_ba


def _boilerplate(s, book):
    """Verses whose set of content words occurs 3+ times in the book (refrains, formulas)."""
    keys = [frozenset(t.tolist()) for t in s.books[book]["terms"]]
    c = Counter(keys)
    return [c[k] >= 3 for k in keys]


def _informative_shared(s, a, i, b, j):
    """Words both verses use, leaving out the most common terms and terms without vectors."""
    n_vec = s.wv.shape[0]
    ta = {t for t in s.books[a]["terms"][i].tolist() if COMMON <= t < n_vec}
    tb = {t for t in s.books[b]["terms"][j].tolist() if COMMON <= t < n_vec}
    return ta & tb


def _chapter(ref):
    return ref.rsplit(":", 1)[0]


def build_similarities(s):
    lam = s.params["similar_vector_weight"]
    theme_vecs = {}
    for name, seeds in THEMES.items():
        q = {s.term_of(w): (1.0, w) for w in seeds.split() if s.term_of(w) is not None}
        theme_vecs[name] = s.query_vector(q)
    names = list(THEMES)
    theme_m = torch.stack([theme_vecs[n] for n in names])
    grouped = {n: [] for n in names}
    n_vec = s.wv.shape[0]
    for (a, b), quota in QUOTA.items():
        (sab, iab), (sba, iba) = _best_matches(s, a, b, lam)
        ok = {}
        for book in (a, b):
            terms = s.books[book]["terms"]
            boiler = _boilerplate(s, book)
            ok[book] = [len(set(t[t < n_vec].tolist())) >= MIN_TERMS and not boiler[i] for i, t in enumerate(terms)]
        # Each verse's best match in the other book, from both sides.
        cands = {(i, iab[i].item()): sab[i].item() for i in range(len(sab))}
        cands.update({(iba[j].item(), j): sba[j].item() for j in range(len(sba))})
        cands = [(score, i, j) for (i, j), score in cands.items()
                 if ok[a][i] and ok[b][j] and len(_informative_shared(s, a, i, b, j)) >= MIN_SHARED]
        cands.sort(reverse=True)
        used, per_ch, kept = set(), Counter(), 0
        for score, i, j in cands:
            ra, rb = s.books[a]["refs"][i], s.books[b]["refs"][j]
            ca, cb = (a, _chapter(ra)), (b, _chapter(rb))
            if (a, i) in used or (b, j) in used or per_ch[ca] >= MAX_PER_CHAPTER or per_ch[cb] >= MAX_PER_CHAPTER:
                continue
            used |= {(a, i), (b, j)}
            per_ch[ca] += 1
            per_ch[cb] += 1
            ta, tb = set(s.books[a]["terms"][i].tolist()), set(s.books[b]["terms"][j].tolist())
            shared = sorted(ta & tb, key=lambda t: -s.global_idf[t].item())[:6]
            mid = s.books[a]["vectors"][i] + s.books[b]["vectors"][j]
            theme = names[int((theme_m @ mid).argmax())]
            grouped[theme].append({
                "score": round(score, 3),
                "a": {"book": a, "ref": ra, "text": s.books[a]["text"][i]},
                "b": {"book": b, "ref": rb, "text": s.books[b]["text"][j]},
                "shared_words": [s.terms[t] for t in shared],
            })
            kept += 1
            if kept >= quota:
                break
    themes = [{"name": n, "pairs": sorted(grouped[n], key=lambda p: -p["score"])} for n in names if grouped[n]]
    return {
        "about": (
            "Closest passages across the books. Each pair is one verse's best match in the other book, scored "
            "by a mix of word-vector cosine and TF-IDF cosine. Refrains, very short verses and pairs that share "
            f"fewer than {MIN_SHARED} informative words are left out, each verse is used once, and no chapter "
            "appears more than "
            f"{MAX_PER_CHAPTER} times per pair of books. Themes are hand-named; each pair is put under the "
            "closest one by word vectors, so a few placements are rough."
        ),
        "quota": {f"{a}-{b}": q for (a, b), q in QUOTA.items()},
        "themes": themes,
    }
