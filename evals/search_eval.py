"""Benchmark for the meaning search.

    python evals/search_eval.py      # prints tables, writes evals/results/search.json

1. Situations (evals/search_cases.json): for each modern-language situation
   and each book, where does the first hand-picked relevant verse land in the
   ranking? We report hit@3 (one is in the top 3), hit@10 and MRR (mean of
   1 / rank of the first relevant verse), per book and overall, with a 95%
   bootstrap interval that resamples whole situations.
   Splits: "dev" and "dev2" were both seen while the bridge and settings were
   written (dev2 was first meant as a test set, but the bridge was written
   after its queries existed, so it is not held out). "blind_test" is for
   situations written later by someone who has not seen the bridge.
   As on the site, blocklisted verses are never returned; relevant verses
   that are blocklisted are dropped from the answer key (the count is reported).
2. Parallels (evals/search_parallels.json): known parallel passages across
   books. For each one we look up the 10 verses of the other book closest to
   the anchor verse and check whether a parallel is among them.
"""

import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripture.search import BOOKS, METHODS, Searcher  # noqa: E402

CASES = ROOT / "evals" / "search_cases.json"
PARALLELS = ROOT / "evals" / "search_parallels.json"
RESULTS = ROOT / "evals" / "results" / "search.json"
SPLITS = ("dev", "dev2", "blind_test")
ALL_METHODS = ["random", "bm25", "bm25+bridge", "vectors-no-bridge", "vectors", "hybrid"]
BOOTSTRAP = 1000


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


def pair_metrics(rank):
    """hit@3, hit@10 and reciprocal rank for one situation-book pair."""
    return {"hit@3": float(rank <= 3), "hit@10": float(rank <= 10), "mrr": 1.0 / rank}


def mean_metrics(rows):
    """Average a list of per-pair metric dicts."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "hit@3": None, "hit@10": None, "mrr": None}
    return {"n": n, **{m: sum(r[m] for r in rows) / n for m in ("hit@3", "hit@10", "mrr")}}


def bootstrap(per_case, reps=BOOTSTRAP, seed=0):
    """95% intervals for overall metrics, resampling situations (each with all its book pairs)."""
    per_case = [rows for rows in per_case if rows]
    if not per_case:
        return None
    rnd = random.Random(seed)
    draws = {m: [] for m in ("hit@3", "hit@10", "mrr")}
    for _ in range(reps):
        sample = [r for _ in per_case for r in rnd.choice(per_case)]
        means = mean_metrics(sample)
        for m in draws:
            draws[m].append(means[m])
    out = {}
    for m, xs in draws.items():
        xs.sort()
        out[m] = [round(xs[int(0.025 * reps)], 3), round(xs[int(0.975 * reps) - 1], 3)]
    return out


# ----- situations -----

def relevant_indices(searcher, case, book):
    """Indices of the case's relevant verses in one book, without blocklisted ones."""
    b = searcher.books[book]
    idx = [b["index"][r] for r in case["relevant"][book]]
    return [i for i in idx if not b["blocked"][i]], len(idx)


def evaluate(searcher, cases, method):
    """Metrics per book, overall, and a bootstrap interval for overall."""
    per_book = {b: [] for b in BOOKS}
    per_case, dropped = [], 0
    for c in cases:
        rows = []
        for book in BOOKS:
            idx, total = relevant_indices(searcher, c, book)
            dropped += total - len(idx)
            if not idx:
                continue
            b = searcher.books[book]
            n_open = int((~b["blocked"]).sum())
            if method == "random":
                row = {"hit@3": random_hit(n_open, len(idx), 3), "hit@10": random_hit(n_open, len(idx), 10),
                       "mrr": random_mrr(n_open, len(idx))}
            else:
                s, _ = searcher.scores(c["query"], book, **METHODS[method])
                s = torch.where(b["blocked"], torch.full_like(s, float("-inf")), s)
                row = pair_metrics(first_rank(s, torch.tensor(idx)))
            per_book[book].append(row)
            rows.append(row)
        per_case.append(rows)
    out = {b: mean_metrics(per_book[b]) for b in BOOKS}
    out["overall"] = mean_metrics([r for b in BOOKS for r in per_book[b]])
    out["overall_ci95"] = bootstrap(per_case)
    out["relevant_dropped_as_blocked"] = dropped
    return out


def case_rows(searcher, cases, method):
    """Per situation, the metric rows of its book pairs (blocklist applied as in evaluate)."""
    out = []
    for c in cases:
        rows = []
        for book in BOOKS:
            idx, _ = relevant_indices(searcher, c, book)
            if not idx:
                continue
            b = searcher.books[book]
            s, _ = searcher.scores(c["query"], book, **METHODS[method])
            s = torch.where(b["blocked"], torch.full_like(s, float("-inf")), s)
            rows.append(pair_metrics(first_rank(s, torch.tensor(idx))))
        out.append(rows)
    return out


def paired_difference(searcher, cases, m1, m2, reps=BOOTSTRAP, seed=0):
    """Mean of (m1 - m2) per pair, with a 95% interval resampling situations."""
    a, b = case_rows(searcher, cases, m1), case_rows(searcher, cases, m2)
    diffs = [[{m: x[m] - y[m] for m in x} for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
    point = mean_metrics([r for rows in diffs for r in rows])
    ci = bootstrap(diffs, reps, seed)
    return {m: {"mean": round(point[m], 3), "ci95": ci[m]} for m in ("hit@3", "hit@10", "mrr")}


# ----- parallels -----

def evaluate_parallels(searcher, parallels, vector_weight=None, k=10):
    """Share of parallels (both directions) whose partner is in the other book's top k.

    This measures the model, so blocklisted verses are allowed here.
    """
    saved = searcher.params["similar_vector_weight"]
    if vector_weight is not None:
        searcher.params["similar_vector_weight"] = vector_weight
    hits = {"story": [], "teaching": []}
    detail = []
    try:
        for p in parallels:
            for src, dst in ((p["a"], p["b"]), (p["b"], p["a"])):
                top = searcher.similar(src["book"], src["refs"][0], k=k, books=[dst["book"]], allow_blocked=True)
                refs = [h["ref"] for h in top[dst["book"]]]
                rank = next((i + 1 for i, r in enumerate(refs) if r in dst["refs"]), None)
                hits[p["kind"]].append(rank is not None)
                detail.append({"id": p["id"], "from": f'{src["book"]} {src["refs"][0]}', "rank": rank})
    finally:
        searcher.params["similar_vector_weight"] = saved
    allh = hits["story"] + hits["teaching"]
    return {
        f"hit@{k}": sum(allh) / len(allh) if allh else None,
        f"story_hit@{k}": sum(hits["story"]) / len(hits["story"]) if hits["story"] else None,
        f"teaching_hit@{k}": sum(hits["teaching"]) / len(hits["teaching"]) if hits["teaching"] else None,
        "trials": len(allh),
        "detail": detail,
    }


# ----- report -----

def fmt(x):
    return "   -" if x is None else f"{x:.3f}"


def table(title, results):
    cols = [f"{b} {m}" for b in BOOKS + ("overall",) for m in ("h@3", "h@10", "MRR")]
    lines = [f"\n{title}", f'{"method":<18}' + "".join(f"{c:>13}" for c in cols) + "   overall h@10 95% CI   MRR 95% CI"]
    for method, r in results.items():
        cells = []
        for b in BOOKS + ("overall",):
            cells += [r[b]["hit@3"], r[b]["hit@10"], r[b]["mrr"]]
        ci = r["overall_ci95"]
        lines.append(f"{method:<18}" + "".join(f"{fmt(x):>13}" for x in cells)
                     + (f"   [{ci['hit@10'][0]:.2f}, {ci['hit@10'][1]:.2f}]        [{ci['mrr'][0]:.2f}, {ci['mrr'][1]:.2f}]" if ci else ""))
    return "\n".join(lines)


def run(searcher, out_path=RESULTS):
    cases = json.loads(CASES.read_text())["cases"]
    parallels = json.loads(PARALLELS.read_text())["parallels"]
    report = {"situations": {}, "parallels": {}, "examples": []}
    for split in SPLITS:
        sel = [c for c in cases if c["split"] == split]
        if not sel:
            report["situations"][split] = None
            print(f"\n{split.upper()} set: no situations yet")
            continue
        res = {m: evaluate(searcher, sel, m) for m in ALL_METHODS}
        report["situations"][split] = res
        print(table(f"{split.upper()} set ({len(sel)} situations; "
                    f"{res['hybrid']['relevant_dropped_as_blocked']} blocklisted relevant verses dropped)", res))
    dev_all = [c for c in cases if c["split"] in ("dev", "dev2")]
    for other in ("bm25+bridge", "vectors"):
        d = paired_difference(searcher, dev_all, "hybrid", other)
        report.setdefault("hybrid_minus", {})[other] = d
        print(f"\nhybrid minus {other} on dev + dev2 ({len(dev_all)} situations): "
              + ", ".join(f"{m} {d[m]['mean']:+.3f} [{d[m]['ci95'][0]:+.3f}, {d[m]['ci95'][1]:+.3f}]" for m in d))
    for name, w in (("tf-idf only", 0.0), ("vectors only", 1.0), ("hybrid", None)):
        r = evaluate_parallels(searcher, parallels, w)
        report["parallels"][name] = r
        print(f"\nparallels, {name:<12}: hit@10 {r['hit@10']:.3f}  (stories {r['story_hit@10']:.3f}, "
              f"teachings {r['teaching_hit@10']:.3f}, {r['trials']} lookups)")
    print("\nExamples (search() with the safety rules, top 3 per book):")
    for c in cases:
        res = searcher.search(c["query"], k=3)
        ex = {"query": c["query"], "split": c["split"], "category": res["category"], "results": {}}
        print(f'\n> {c["query"]}  [{c["split"]}, {res["category"]}]')
        for book in BOOKS:
            ex["results"][book] = []
            for h in res["results"][book]:
                good = h["ref"] in c["relevant"][book]
                ex["results"][book].append({"ref": h["ref"], "text": h["text"], "in_answer_key": good})
                print(f'  {"*" if good else " "} {book:<5} {h["ref"]:<20} {h["text"][:110]}')
        report["examples"].append(ex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"\n(* = in the hand-picked answer key)  wrote {out_path.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    torch.set_num_threads(3)
    run(Searcher.load())
