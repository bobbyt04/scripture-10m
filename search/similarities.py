"""Cross-book parallels for the website's similarities section.

Two steps:

1. The model proposes candidates (`propose`). For each pair of books it
   compares every verse of one with every verse of the other (vector cosine
   mixed with TF-IDF cosine, as in Searcher.similar), takes each verse's best
   match, and keeps pairs that share several informative words or are very
   close in vector space. Blocklisted verses, refrains and very short verses
   are left out.
2. A person reads every candidate. search/similarities_review.json records
   the decision for each one; kept pairs get a theme and a one-line note on
   the shared idea. Only kept pairs are published, in
   search/artifacts/similarities.json, with the model's score and the share
   of candidates that were kept.

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

from scripture.search import BOOKS, Searcher  # noqa: E402

REVIEW = ROOT / "search" / "similarities_review.json"
PARALLELS = ROOT / "evals" / "search_parallels.json"
HAND_THEMES = {
    "alms_secret": "Humility, giving and self-control", "camel_needle": "Humility, giving and self-control",
    "friend_and_foe": "Humility, giving and self-control", "no_return_gift": "Humility, giving and self-control",
    "no_return_gift_q": "Humility, giving and self-control", "moderation_food": "Humility, giving and self-control",
    "anger_folly": "Humility, giving and self-control", "senses_control_q": "Humility, giving and self-control",
    "eye_for_eye": "Guidance and wisdom", "seed_grows": "Guidance and wisdom", "equanimity_qg": "Guidance and wisdom",
    "scroll": "Death and the life to come", "taste_death": "Death and the life to come",
    "fleeting_grass": "Death and the life to come", "soul_not_slain": "Death and the life to come",
    "soul_living_q": "Death and the life to come", "body_garment": "Death and the life to come",
    "first_last": "God's knowledge and care", "mercy_all_sins": "God's knowledge and care",
    "god_light": "God's knowledge and care", "beginning_end": "God's knowledge and care",
    "beginning_end_q": "God's knowledge and care", "god_in_heart": "God's knowledge and care",
    "thousand_suns": "God's knowledge and care", "one_father": "God's knowledge and care",
    "do_all_for_god": "Worship and devotion", "do_all_for_god_q": "Worship and devotion",
    "devotion_offering": "Worship and devotion", "devotion_offering_b": "Worship and devotion",
}
PAIRS = (("bible", "quran"), ("bible", "gita"), ("quran", "gita"))
POOL = {("bible", "quran"): 160, ("bible", "gita"): 70, ("quran", "gita"): 70}
MIN_TERMS = 5           # verses with fewer content words are too thin to compare
MIN_SHARED = 2          # informative words both verses use
COMMON = 150            # the most frequent terms (lord, god, people, ...) are not informative
VECTOR_ONLY = 0.55      # or: this close in vector space, whatever the shared words
MAX_PER_CHAPTER = 3     # keep the candidate list varied


def _key(book, ref):
    return f"{book} {ref}"


def _informative(s, book, i):
    n_vec = s.wv.shape[0]
    return {t for t in s.books[book]["terms"][i].tolist() if COMMON <= t < n_vec}


def _best_matches(s, a, b, lam, chunk=2048):
    """For every verse of a: best verse of b and the score; and the reverse."""
    va, vb = s.books[a]["vectors"], s.books[b]["vectors"]
    s._tfidf_scores(torch.zeros(len(s.terms)), s.books[a])     # build the cached TF-IDF matrices
    s._tfidf_scores(torch.zeros(len(s.terms)), s.books[b])
    ta, na = s.books[a]["tfidf"]
    tb, nb = s.books[b]["tfidf"]
    tb_t = tb.t().coalesce()
    bad_a, bad_b = s.books[a]["blocked"], s.books[b]["blocked"]
    ab = (torch.full((len(va),), -1.0), torch.zeros(len(va), dtype=torch.long))
    ba = (torch.full((len(vb),), -1.0), torch.zeros(len(vb), dtype=torch.long))
    for lo in range(0, len(va), chunk):
        hi = min(lo + chunk, len(va))
        sub = torch.index_select(ta, 0, torch.arange(lo, hi))
        lex = torch.sparse.mm(sub, tb_t).to_dense() / na[lo:hi, None] / nb[None, :]
        sim = lam * (va[lo:hi] @ vb.t()) + (1 - lam) * lex
        sim[bad_a[lo:hi]] = -1
        sim[:, bad_b] = -1
        v, i = sim.max(1)
        ab[0][lo:hi], ab[1][lo:hi] = v, i
        v, i = sim.max(0)
        better = v > ba[0]
        ba[0][better], ba[1][better] = v[better], i[better] + lo
    return ab, ba


def _refrains(s, book):
    """Verses whose set of content words occurs 3+ times in the book (refrains, formulas)."""
    keys = [frozenset(t.tolist()) for t in s.books[book]["terms"]]
    c = Counter(keys)
    return [c[k] >= 3 for k in keys]


def propose(s):
    """Candidate pairs for review, best first per pair of books."""
    lam = s.params["similar_vector_weight"]
    out = []
    for a, b in PAIRS:
        (sab, iab), (sba, iba) = _best_matches(s, a, b, lam)
        ok = {}
        for book in (a, b):
            refrain = _refrains(s, book)
            ok[book] = [len(_informative(s, book, i)) >= MIN_TERMS - 2 and len(set(t.tolist())) >= MIN_TERMS
                        and not refrain[i] and not s.books[book]["blocked"][i]
                        for i, t in enumerate(s.books[book]["terms"])]
        cands = {(i, iab[i].item()): sab[i].item() for i in range(len(sab))}
        cands.update({(iba[j].item(), j): sba[j].item() for j in range(len(sba))})
        rows = []
        for (i, j), score in cands.items():
            if not (ok[a][i] and ok[b][j]):
                continue
            shared = _informative(s, a, i) & _informative(s, b, j)
            cos = (s.books[a]["vectors"][i] @ s.books[b]["vectors"][j]).item()
            if len(shared) >= MIN_SHARED or cos >= VECTOR_ONLY:
                rows.append((score, i, j, shared))
        rows.sort(key=lambda r: -r[0])
        per_ch, used, kept = Counter(), set(), 0
        for score, i, j, shared in rows:
            ra, rb = s.books[a]["refs"][i], s.books[b]["refs"][j]
            ca, cb = (a, ra.rsplit(":", 1)[0]), (b, rb.rsplit(":", 1)[0])
            if (a, i) in used or (b, j) in used or per_ch[ca] >= MAX_PER_CHAPTER or per_ch[cb] >= MAX_PER_CHAPTER:
                continue
            used |= {(a, i), (b, j)}
            per_ch[ca] += 1
            per_ch[cb] += 1
            out.append({"a": _key(a, ra), "b": _key(b, rb), "score": round(score, 3),
                        "shared_words": sorted(s.terms[t] for t in shared)})
            kept += 1
            if kept >= POOL[(a, b)]:
                break
    return out


def pair_score(s, a, b):
    """The model's similarity for one pair, as in Searcher.similar."""
    (ba, ra), (bb, rb) = a.split(" ", 1), b.split(" ", 1)
    i, j = s.books[ba]["index"][ra], s.books[bb]["index"][rb]
    lam = s.params["similar_vector_weight"]
    vec = (s.books[ba]["vectors"][i] @ s.books[bb]["vectors"][j]).item()
    lex = s._tfidf_scores(s._tfidf(s.books[ba]["terms"][i]), s.books[bb])[j].item()
    return round(lam * vec + (1 - lam) * lex, 3)


def _verses(s, keys):
    out = []
    for key in keys:
        book, ref = key.split(" ", 1)
        out.append({"book": book, "ref": ref, "text": s.books[book]["text"][s.books[book]["index"][ref]]})
    return out


def build_similarities(s, review_path=REVIEW, parallels_path=PARALLELS):
    """similarities.json from the reviewed candidates plus the hand-listed parallels."""
    review = json.loads(Path(review_path).read_text())
    themes, seen = {}, set()
    for c in review["candidates"]:
        if not c["keep"]:
            continue
        seen.add(frozenset((c["a"], c["b"])))
        themes.setdefault(c["theme"], []).append({
            "idea": c["idea"], "score": pair_score(s, c["a"], c["b"]), "source": "model, reviewed",
            "verses": _verses(s, (c["a"], c["b"]))})
    hand, skipped = 0, 0
    for p in json.loads(Path(parallels_path).read_text())["parallels"]:
        a, b = _key(p["a"]["book"], p["a"]["refs"][0]), _key(p["b"]["book"], p["b"]["refs"][0])
        blocked = any(s.books[k.split(" ", 1)[0]]["blocked"][s.books[k.split(" ", 1)[0]]["index"][k.split(" ", 1)[1]]]
                      for k in (a, b))
        if blocked or frozenset((a, b)) in seen:
            skipped += 1
            continue
        theme = "Prophets and their stories" if p["kind"] == "story" else HAND_THEMES[p["id"]]
        themes.setdefault(theme, []).append({
            "idea": p["label"] + ".", "score": pair_score(s, a, b), "source": "hand-listed",
            "verses": _verses(s, (a, b))})
        hand += 1
    n = len(review["candidates"])
    kept = sum(c["keep"] for c in review["candidates"])
    per_pair = Counter()
    for c in review["candidates"]:
        if c["keep"]:
            per_pair["-".join(sorted((c["a"].split()[0], c["b"].split()[0])))] += 1
    return {
        "about": ("Parallels between the books, chosen by a person. Two sources: (1) 'model, reviewed': the search "
                  "model proposed candidate pairs (each verse's closest verse in another book); a person read every "
                  "candidate and kept only real parallels in story or teaching, wrote the one-line note and chose "
                  "the theme; (2) 'hand-listed': the known parallels of evals/search_parallels.json (anchor verses), "
                  "left out if a verse is on the blocklist. 'score' is the model's similarity for the pair."),
        "candidates_proposed": n,
        "kept": kept,
        "kept_fraction": round(kept / n, 3) if n else None,
        "kept_per_book_pair": dict(sorted(per_pair.items())),
        "hand_listed_added": hand,
        "hand_listed_skipped": skipped,
        "themes": [{"name": name, "pairs": sorted(pairs, key=lambda p: -p["score"])}
                   for name, pairs in themes.items()],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--propose", action="store_true", help="rewrite the candidates in the review file "
                    "(decisions already made for the same pair are kept)")
    args = ap.parse_args()
    torch.set_num_threads(3)
    s = Searcher.load()
    if args.propose:
        old = {}
        if REVIEW.exists():
            old = {(c["a"], c["b"]): c for c in json.loads(REVIEW.read_text())["candidates"]}
        cands = []
        for c in propose(s):
            prev = old.get((c["a"], c["b"]), {})
            cands.append({**c, "keep": prev.get("keep", False), "theme": prev.get("theme", ""),
                          "idea": prev.get("idea", "")})
        REVIEW.write_text(json.dumps({"about": "Every candidate the model proposed, with the human decision.",
                                      "candidates": cands}, indent=1, ensure_ascii=False) + "\n")
        print(len(cands), "candidates written to", REVIEW)


if __name__ == "__main__":
    main()
