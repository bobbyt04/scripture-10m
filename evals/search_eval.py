"""Benchmark for the advice search (the reviewed pool) and for similar().

    python evals/search_eval.py      # prints tables, writes evals/results/search.json

1. Situations (evals/search_cases.json). Every case has a modern-language
   situation, hand-picked relevant verses per book (picked from the whole
   corpus, before the pool existed) and the themes a good answer should match.
   All filled splits (dev, dev2, dev3, blind_1) were seen during development.
   - hit@3: for each case and book, is one of the shown verses (top 3) in the
     answer key? Only answer-key verses that are in the published pool count
     (the search cannot show anything else); a book whose key has none is
     skipped, and the number of key verses outside the pool is reported.
   - theme accuracy: is the query's top theme one of the expected themes?
   - empty-book rate: the share of case-book pairs where the book shows nothing.
   - ablation: verses chosen by theme only, in pool order; + the BM25 tie
     break; + the hand-set priorities (final).
   A 95% bootstrap interval resamples whole cases.
2. Parallels (evals/search_parallels.json): known parallel passages across
   books. For each one, is the partner among the 10 verses of the other book
   closest to the anchor verse (search/corpus.py, the whole corpus)?
3. blind_1: 50 situations an independent judge wrote and judged by hand in
   round 5 ("blind_1_hand_judged" in search_cases.json, kept as history); it
   has since been used for development. final_blind: 50 situations from an
   independent blind review, judged verse by verse once
   ("final_blind_hand_judged", kept as history); spent in round 6.
4. Safety probes (evals/search_probes.json): for each probe set, how many
   crisis messages get the crisis category, and how many innocent or
   bereaved messages are wrongly sent there. Every set has been used for
   development; the numbers from each set's first run are in "first_look"
   (judge5_held_out's and judge6_held_out's in their own "first_run").
5. suggest(), the site contract since round 7 (type, then pick a topic):
   - topic hit@4: for the labelled everyday splits, is an expected theme
     among the (at most 4) topics suggest() offers?
   - for every stored crisis message the gate misses, what suggest()
     returns (nothing, or which topics); and, as a stress test of the
     suggestion layer alone, the same for every crisis message as if the
     gate had missed it.
   Every set was used to write the lists, so these are development numbers.
   Items 1 to 4 use search(), which the site no longer calls; they are kept
   so the recorded history can be reproduced.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripture.search import BOOKS, Searcher, phrase_starts, words  # noqa: E402
from search import build  # noqa: E402

CASES = ROOT / "evals" / "search_cases.json"
PARALLELS = ROOT / "evals" / "search_parallels.json"
RESULTS = ROOT / "evals" / "results" / "search.json"
SPLITS = ("dev", "dev2", "dev3", "blind_1", "final_blind")
PROBES = ROOT / "evals" / "search_probes.json"
BOOTSTRAP = 1000


# ----- metrics -----

def case_rows(searcher, case, k=3):
    """Per book: {"book", "hit" (None if no key verse is in the pool), "empty", "outside"}."""
    res = searcher.search(case["query"], k=k)
    rows = []
    for book in BOOKS:
        in_pool = [r for r in case["relevant"][book] if r in searcher.books[book]["index"]]
        shown = [h["ref"] for h in res["results"][book]]
        rows.append({"book": book, "hit": float(bool(set(shown) & set(in_pool))) if in_pool else None,
                     "empty": not shown, "outside": len(case["relevant"][book]) - len(in_pool)})
    return rows


def top_theme_ok(searcher, case):
    themes = searcher.query_themes(case["query"])
    return float(bool(themes) and next(iter(themes)) in case["expected_themes"])


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def bootstrap(per_case, reps=BOOTSTRAP, seed=0):
    """95% interval of the mean over all values, resampling cases (each a list of values)."""
    per_case = [xs for xs in per_case if xs]
    if not per_case:
        return None
    rnd = random.Random(seed)
    draws = sorted(mean([x for _ in per_case for x in rnd.choice(per_case)]) for _ in range(reps))
    return [round(draws[int(0.025 * reps)], 3), round(draws[int(0.975 * reps) - 1], 3)]


def evaluate(searcher, cases):
    """hit@3 per book and overall (with interval), theme accuracy, empty-book rate, key verses outside the pool."""
    per_case = [case_rows(searcher, c) for c in cases]
    hits = {b: [r["hit"] for rows in per_case for r in rows if r["book"] == b and r["hit"] is not None] for b in BOOKS}
    out = {b: {"pairs": len(hits[b]), "hit@3": mean(hits[b])} for b in BOOKS}
    case_hits = [[r["hit"] for r in rows if r["hit"] is not None] for rows in per_case]
    out["overall"] = {"pairs": sum(len(h) for h in case_hits), "hit@3": mean([x for h in case_hits for x in h]),
                      "hit@3_ci95": bootstrap(case_hits)}
    out["theme_accuracy"] = mean([top_theme_ok(searcher, c) for c in cases])
    out["empty_book_rate"] = mean([float(r["empty"]) for rows in per_case for r in rows])
    out["empty_book_rate_per_book"] = {b: mean([float(r["empty"]) for rows in per_case for r in rows if r["book"] == b])
                                       for b in BOOKS}
    out["key_verses_outside_pool"] = sum(r["outside"] for rows in per_case for r in rows)
    out["key_verses_total"] = sum(len(c["relevant"][b]) for c in cases for b in BOOKS)
    return out


# ----- ablation -----

def variant(name):
    """A Searcher with only some parts of the final ranking, built from the same pool."""
    verses = build.load_verses()
    pool = build.load_pool()
    lexicon = build.load_json(build.MODEL / "model.json")["lexicon"]
    lists = [build.load_json(x) for x in (build.THEMES, build.BRIDGE, build.SAFETY, build.SENSITIVE_SET)]
    no_priority = {b: [{k: v for k, v in x.items() if k != "priority"} for x in pool[b]] for b in BOOKS}
    fields, pl = {
        "themes, pool order": ((), no_priority),
        "+ BM25 tie break": (build.DOC_FIELDS, no_priority),
        "+ hand-set priority (final)": (build.DOC_FIELDS, pool),
    }[name]
    return Searcher(build.build_site(pl, verses, lexicon, *lists, fields=fields))


ABLATION = ("themes, pool order", "+ BM25 tie break", "+ hand-set priority (final)")


def probe_report(searcher, probes):
    """Per probe set: crisis messages caught, and innocent or bereaved messages sent to crisis."""
    out = {}
    for name, d in probes.items():
        if not isinstance(d, dict) or "crisis" not in d:
            continue
        crisis = [p["text"] for p in d["crisis"]]
        row = {"crisis": len(crisis), "crisis_caught": sum(searcher.classify(q)[0] == "crisis" for q in crisis)}
        for g in ("innocent", "bereavement", "not_crisis"):
            if g in d:
                row[g] = len(d[g])
                row[g + "_false_alarms"] = sum(searcher.classify(q)[0] == "crisis" for q in d[g])
        if "sensitive" in d:
            row["sensitive"] = len(d["sensitive"])
            row["sensitive_routed"] = sum(searcher.classify(q)[0] == "sensitive" for q in d["sensitive"])
        out[name] = row
    return out


# ----- suggest(): the site contract since round 7 -----

def topic_hit(searcher, cases, n=4):
    """Share of cases where one of the expected themes is among the (at most n) topics suggest() offers."""
    hits = [bool({t["id"] for t in searcher.suggest(c["query"])["topics"][:n]} & set(c["expected_themes"]))
            for c in cases]
    return {"cases": len(cases), f"topic_hit@{n}": mean([float(h) for h in hits]),
            "no_topic": sum(not searcher.suggest(c["query"])["topics"] for c in cases)}


def as_if_missed(searcher, query):
    """What suggest() would offer for a message if the crisis gate had not fired (its sensitive routing kept)."""
    toks = words(query[:searcher.params["max_query_chars"]])
    topic = next((t for t, ps in searcher.sensitive.items() if any(phrase_starts(toks, p) for p in ps)), None)
    return searcher.offer(query, "sensitive" if topic else "general", topic)


def suggest_probe_report(searcher, probes):
    """For every stored crisis message: what suggest() returns when the gate misses it.

    "missed": the messages the current gate misses, and what suggest()
    offers them. "as_if_missed": every crisis message as if the gate had
    missed it (a stress test of the suggestion layer alone): how many get no
    topic, how many get topics, and which topics.
    """
    missed, sim = [], []
    for name, d in probes.items():
        if not isinstance(d, dict) or "crisis" not in d:
            continue
        for p in d["crisis"]:
            res = searcher.suggest(p["text"])
            if res["category"] != "crisis":
                missed.append({"set": name, "text": p["text"], "topics": [t["id"] for t in res["topics"]]})
            sim.append([t["id"] for t in as_if_missed(searcher, p["text"])["topics"]])

    def summary(rows):
        counts = Counter(t for r in rows for t in r)
        return {"messages": len(rows), "nothing_suggested": sum(not r for r in rows),
                "topics_suggested": sum(bool(r) for r in rows), "topics": dict(counts.most_common())}
    return {"missed": {**summary([m["topics"] for m in missed]), "detail": missed}, "as_if_missed": summary(sim)}


# ----- parallels -----

def evaluate_parallels(corpus, parallels, vector_weight=None, k=10):
    """Share of parallels (both directions) whose partner is in the other book's top k (whole corpus)."""
    saved = corpus.params["similar_vector_weight"]
    if vector_weight is not None:
        corpus.params["similar_vector_weight"] = vector_weight
    hits = {"story": [], "teaching": []}
    detail = []
    try:
        for p in parallels:
            for src, dst in ((p["a"], p["b"]), (p["b"], p["a"])):
                top = corpus.similar(src["book"], src["refs"][0], k=k, books=[dst["book"]])
                refs = [h["ref"] for h in top[dst["book"]]]
                rank = next((i + 1 for i, r in enumerate(refs) if r in dst["refs"]), None)
                hits[p["kind"]].append(rank is not None)
                detail.append({"id": p["id"], "from": f'{src["book"]} {src["refs"][0]}', "rank": rank})
    finally:
        corpus.params["similar_vector_weight"] = saved
    allh = hits["story"] + hits["teaching"]
    return {
        f"hit@{k}": mean([float(x) for x in allh]),
        f"story_hit@{k}": mean([float(x) for x in hits["story"]]),
        f"teaching_hit@{k}": mean([float(x) for x in hits["teaching"]]),
        "trials": len(allh),
        "detail": detail,
    }


# ----- report -----

def fmt(x):
    return "   -" if x is None else f"{x:.3f}"


def run(searcher=None, corpus=None, out_path=RESULTS, ablation=True):
    from search.corpus import Corpus
    searcher = searcher or Searcher.load()
    corpus = corpus or Corpus.load()
    case_file = json.loads(CASES.read_text())
    cases = case_file["cases"]
    parallels = json.loads(PARALLELS.read_text())["parallels"]
    report = {"situations": {}, "ablation": {}, "parallels": {},
              "blind_1_hand_judged": case_file.get("blind_1_hand_judged"),
              "probes": {}, "probes_first_look": {},
              "suggest": {"topic_hit@4": {}, "crisis_probes": {}},
              "examples": []}
    for split in SPLITS:
        sel = [c for c in cases if c["split"] == split]
        if not sel:
            report["situations"][split] = None
            print(f"\n{split}: no situations yet")
            continue
        r = evaluate(searcher, sel)
        report["situations"][split] = {"cases": len(sel), **r}
        print(f"\n{split} ({len(sel)} cases): hit@3 " + "  ".join(f"{b} {fmt(r[b]['hit@3'])} ({r[b]['pairs']})" for b in BOOKS)
              + f"  overall {fmt(r['overall']['hit@3'])} {r['overall']['hit@3_ci95']}"
              + f"\n   theme accuracy {fmt(r['theme_accuracy'])}  empty-book rate {fmt(r['empty_book_rate'])}"
              + f"  key verses outside the pool {r['key_verses_outside_pool']} of {r['key_verses_total']}")
    if ablation:
        dev_all = [c for c in cases if c["split"] in ("dev", "dev2", "dev3")]
        print(f"\nAblation on all {len(dev_all)} development cases:")
        for name in ABLATION:
            r = evaluate(variant(name), dev_all)
            report["ablation"][name] = {"hit@3": r["overall"]["hit@3"], "hit@3_ci95": r["overall"]["hit@3_ci95"],
                                        "theme_accuracy": r["theme_accuracy"], "empty_book_rate": r["empty_book_rate"]}
            print(f"   {name:<30} hit@3 {fmt(r['overall']['hit@3'])} {r['overall']['hit@3_ci95']}  "
                  f"empty-book rate {fmt(r['empty_book_rate'])}")
    probes = json.loads(PROBES.read_text())
    report["probes"] = probe_report(searcher, probes)
    report["probes_first_look"] = probes["first_look"]
    print("\nSafety probes (all used for development):")
    for name, row in report["probes"].items():
        print(f"   {name:<8} crisis caught {row['crisis_caught']}/{row['crisis']}"
              + "".join(f"  {g} false alarms {row[g + '_false_alarms']}/{row[g]}" for g in ("innocent", "bereavement") if g in row)
              + (f"  sensitive routed {row['sensitive_routed']}/{row['sensitive']}" if "sensitive" in row else ""))
    print("\nsuggest(), the site contract (development numbers: every set below was used to write the lists):")
    for split in SPLITS:
        sel = [c for c in cases if c["split"] == split]
        r = topic_hit(searcher, sel)
        report["suggest"]["topic_hit@4"][split] = r
        print(f"   {split:<12} topic hit@4 {fmt(r['topic_hit@4'])} ({r['cases']} cases, {r['no_topic']} with no topic)")
    sp = suggest_probe_report(searcher, probes)
    report["suggest"]["crisis_probes"] = sp
    for key, label in (("missed", "crisis messages the gate misses"),
                       ("as_if_missed", "every crisis message as if the gate missed it")):
        r = sp[key]
        top = ", ".join(f"{t} {n}" for t, n in list(r["topics"].items())[:6]) or "-"
        print(f"   {label}: {r['messages']}; nothing suggested {r['nothing_suggested']}, "
              f"topics suggested {r['topics_suggested']} (most often: {top})")
    for name, w in (("tf-idf only", 0.0), ("vectors only", 1.0), ("mix (used)", None)):
        r = evaluate_parallels(corpus, parallels, w)
        report["parallels"][name] = r
        print(f"parallels, {name:<12}: hit@10 {r['hit@10']:.3f}  (stories {r['story_hit@10']:.3f}, "
              f"teachings {r['teaching_hit@10']:.3f}, {r['trials']} lookups)")
    for c in cases:
        res = searcher.search(c["query"], k=3)
        report["examples"].append({
            "query": c["query"], "split": c["split"], "category": res["category"],
            "themes": [t["id"] for t in res["themes"]][:3],
            "results": {b: [{"ref": h["ref"], "text": h["text"], "in_answer_key": h["ref"] in c["relevant"][b]}
                            for h in res["results"][b]] for b in BOOKS}})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    import torch
    torch.set_num_threads(3)
    run()
