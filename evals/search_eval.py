"""Benchmark for the meaning search.

    python evals/search_eval.py      # prints tables, writes evals/results/search.json

1. Situations (evals/search_cases.json): for each modern-language situation
   and each book, where does the first hand-picked relevant verse land in the
   ranking? We report hit@3 (one is in the top 3), hit@10 and MRR (mean of
   1 / rank of the first relevant verse), per book and overall. The "test"
   cases were written before any search code and never used for tuning; the
   "dev" cases were used to choose settings.
2. Parallels (evals/search_parallels.json): known parallel passages across
   books. For each one we look up the 10 verses of the other book closest to
   the anchor verse and check whether a parallel is among them.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripture.search import BOOKS, METHODS, Searcher  # noqa: E402

CASES = ROOT / "evals" / "search_cases.json"
PARALLELS = ROOT / "evals" / "search_parallels.json"
RESULTS = ROOT / "evals" / "results" / "search.json"
ALL_METHODS = ["random", "bm25", "bm25+bridge", "vectors-no-bridge", "vectors", "hybrid"]
SHOW = ["laid_off", "breakup", "burnout", "dad_cancer", "exams", "forgive_brother", "mother_died",
        "no_purpose", "alcohol", "cheated_partner", "new_city", "jealous_friend"]


# ----- metrics -----

def first_rank(scores, relevant):
    """1-based rank of the best-placed relevant index (ties go against us)."""
    best = scores[relevant].max()
    ahead = (scores > best).sum() + (scores == best).sum() - (scores[relevant] == best).sum()
    return int(ahead) + 1


def random_hit(n, r, k):
    """Chance that a random order of n verses puts one of r relevant ones in the top k."""
    miss = 1.0
    for i in range(min(k, n)):
        miss *= max(n - r - i, 0) / (n - i)
    return 1 - miss


def random_mrr(n, r):
    """Expected 1 / rank of the first relevant verse in a random order."""
    total, none_yet = 0.0, 1.0
    for rank in range(1, n - r + 2):
        p_here = none_yet * r / (n - rank + 1)
        total += p_here / rank
        none_yet *= (n - rank + 1 - r) / (n - rank + 1)
    return total


def summarise(ranks):
    """hit@3, hit@10 and MRR from a list of first-relevant ranks."""
    n = len(ranks)
    if n == 0:
        return {"n": 0, "hit@3": None, "hit@10": None, "mrr": None}
    return {
        "n": n,
        "hit@3": sum(r <= 3 for r in ranks) / n,
        "hit@10": sum(r <= 10 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks) / n,
    }


# ----- situations -----

def evaluate(searcher, cases, method):
    """Metrics per book and overall (every case-book pair with relevant verses counts once)."""
    per_book = {b: [] for b in BOOKS}
    rows = []
    for c in cases:
        for book in BOOKS:
            rel = c["relevant"][book]
            if not rel:
                continue
            idx = torch.tensor([searcher.books[book]["index"][r] for r in rel])
            n = len(searcher.books[book]["refs"])
            if method == "random":
                rows.append((book, random_hit(n, len(rel), 3), random_hit(n, len(rel), 10), random_mrr(n, len(rel))))
                continue
            s, _ = searcher.scores(c["query"], book, **METHODS[method])
            per_book[book].append(first_rank(s, idx))
    if method == "random":      # expected values, not a sample
        out = {}
        for book in BOOKS + ("overall",):
            rs = [r for r in rows if book == "overall" or r[0] == book]
            n = len(rs)
            out[book] = {"n": n, "hit@3": sum(r[1] for r in rs) / n if n else None,
                         "hit@10": sum(r[2] for r in rs) / n if n else None,
                         "mrr": sum(r[3] for r in rs) / n if n else None}
        return out
    out = {b: summarise(per_book[b]) for b in BOOKS}
    out["overall"] = summarise([r for b in BOOKS for r in per_book[b]])
    return out


# ----- parallels -----

def evaluate_parallels(searcher, parallels, vector_weight=None, k=10):
    """Share of parallels (both directions) whose partner is in the other book's top k."""
    saved = searcher.params["similar_vector_weight"]
    if vector_weight is not None:
        searcher.params["similar_vector_weight"] = vector_weight
    hits = {"story": [], "teaching": []}
    detail = []
    try:
        for p in parallels:
            for src, dst in ((p["a"], p["b"]), (p["b"], p["a"])):
                top = searcher.similar(src["book"], src["refs"][0], k=k, books=[dst["book"]])[dst["book"]]
                refs = [h["ref"] for h in top]
                rank = next((i + 1 for i, r in enumerate(refs) if r in dst["refs"]), None)
                hits[p["kind"]].append(rank is not None)
                detail.append({"id": p["id"], "from": f'{src["book"]} {src["refs"][0]}', "rank": rank})
    finally:
        searcher.params["similar_vector_weight"] = saved
    allh = hits["story"] + hits["teaching"]
    return {
        f"hit@{k}": sum(allh) / len(allh),
        f"story_hit@{k}": sum(hits["story"]) / len(hits["story"]),
        f"teaching_hit@{k}": sum(hits["teaching"]) / len(hits["teaching"]),
        "trials": len(allh),
        "detail": detail,
    }


# ----- report -----

def table(title, results):
    cols = [f"{b} {m}" for b in BOOKS + ("overall",) for m in ("h@3", "h@10", "MRR")]
    lines = [f"\n{title}", f'{"method":<18}' + "".join(f"{c:>13}" for c in cols)]
    for method, r in results.items():
        cells = []
        for b in BOOKS + ("overall",):
            cells += [r[b]["hit@3"], r[b]["hit@10"], r[b]["mrr"]]
        lines.append(f"{method:<18}" + "".join(f"{x:>13.3f}" for x in cells))
    return "\n".join(lines)


def run(searcher, out_path=RESULTS):
    cases = json.loads(CASES.read_text())["cases"]
    parallels = json.loads(PARALLELS.read_text())["parallels"]
    report = {"situations": {}, "parallels": {}, "examples": []}
    for split in ("dev", "test"):
        sel = [c for c in cases if c["split"] == split]
        res = {m: evaluate(searcher, sel, m) for m in ALL_METHODS}
        report["situations"][split] = res
        print(table(f"{split.upper()} set ({len(sel)} situations)", res))
    for name, w in (("tf-idf only", 0.0), ("vectors only", 1.0), ("hybrid", None)):
        r = evaluate_parallels(searcher, parallels, w)
        report["parallels"][name] = r
        print(f"\nparallels, {name:<12}: hit@10 {r['hit@10']:.3f}  (stories {r['story_hit@10']:.3f}, "
              f"teachings {r['teaching_hit@10']:.3f}, {r['trials']} lookups)")
    print("\nExamples (held-out test situations, top 3 per book):")
    for c in cases:
        if c["id"] not in SHOW:
            continue
        res = searcher.search(c["query"], k=3)
        ex = {"query": c["query"], "split": c["split"], "results": {}}
        print(f'\n> {c["query"]}  [{c["split"]}]')
        for book in BOOKS:
            ex["results"][book] = []
            for h in res[book]:
                good = h["ref"] in c["relevant"][book]
                ex["results"][book].append({"ref": h["ref"], "text": h["text"], "in_gold": good})
                print(f'  {"*" if good else " "} {book:<5} {h["ref"]:<20} {h["text"][:110]}')
        report["examples"].append(ex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"\n(* = in the hand-picked relevant set)  wrote {out_path.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    torch.set_num_threads(3)
    run(Searcher.load())
