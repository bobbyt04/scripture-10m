"""Expected outputs of the Python Searcher for the TypeScript parity test (test/search.test.ts).

    python web/scripts/search_parity.py > out.json

Prints JSON: a few thousand varied messages with suggest(), the crisis words, the signal and the comfort cue of
each, plus topics() and topic(id, sensitive=...) for every theme. The messages are every stored probe and
situation, the fixtures' queries, theme phrases with filler words, and seeded random variants of all of these
(case, apostrophes, typos, digits, emoji, non-ASCII letters next to contractions, astral characters, text past
the 500-character cap). Plain Python, no dependencies; the same seed gives the same messages.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripture.search import Searcher  # noqa: E402

SEED = 10


def stored_messages():
    """Every probe, situation and fixture query in the repository, in file order, without repeats."""
    out = []
    probes = json.loads((ROOT / "evals/search_probes.json").read_text())
    for name, s in probes.items():
        if not isinstance(s, dict):
            continue
        for kind in ("crisis", "bereavement", "innocent", "sensitive"):
            for m in s.get(kind, []):
                out.append(m["text"] if isinstance(m, dict) else m)
        for m in s.get("missed_crisis", []):
            out.append(m["text"])
    cases = json.loads((ROOT / "evals/search_cases.json").read_text())
    out += [c["query"] for c in cases["cases"]]
    for key in ("judge6_blind_hand_judged", "judge7_blind_hand_judged"):
        out += [s["query"] for s in cases[key]["situations"]]
    fixtures = json.loads((ROOT / "search/artifacts/fixtures.json").read_text())
    out += [c["query"] for c in fixtures["suggest"]] + [w[0] for w in fixtures["words"]]
    seen = set()
    return [m for m in out if not (m in seen or seen.add(m))]


def phrase_messages(model, rng):
    """One message per theme, sensitive, comfort-cue and signal phrase family, filled in at random."""
    def fill(p):
        ws = []
        for slot in p.split():
            alt = rng.choice(slot.split("|"))
            ws.append(alt[:-1] + rng.choice(["", "s", "ing", "ed", "ness"]) if alt.endswith("*") else alt)
        return " ".join(ws)
    pre = ["", "i feel ", "lately ", "my friend says ", "honestly ", "i think "]
    post = ["", " and i dont know what to do", " at work", " lol", ", any advice?", " again today"]
    phrases = [p for t in model["themes"].values() for p in t["phrases"]]
    phrases += [p for d in model["safety"]["sensitive"].values() for p in d["phrases"]]
    phrases += [p for g in model["safety"]["suggest"]["comfort_only_cues"].values() for p in g]
    phrases += model["safety"]["suggest"]["signals"] + model["safety"]["idioms"] + model["safety"]["bereavement"]
    out = [rng.choice(pre) + fill(p) + rng.choice(post) for p in rng.sample(phrases, min(900, len(phrases)))]
    # Two themes in one message, to exercise scores, exclusive themes and the topic cap.
    out += [fill(rng.choice(phrases)) + rng.choice([" and ", ". ", ", also "]) + fill(rng.choice(phrases)) for _ in range(500)]
    return out


ODD = ["é", "_", "١", "²", "Ⅻ", "K", "İ", "́", "𝒜", "我", "ß", "Σ", "ǅ", "ﬁ", "'", "’", "-", "😢", "🔫", "💊", "🪢", "⚰️", "7", "14", "007"]


def variant(m, rng):
    """A random change to a message that the port has to handle exactly as Python does."""
    k = rng.randrange(12)
    if k == 0:
        return m.upper()
    if k == 1:
        return m.replace("'", "’")
    if k == 2:
        return m.replace("'", "")
    if k == 3 and len(m) > 6:                                    # a typo: swap, drop or double a letter
        i = rng.randrange(1, len(m) - 2)
        return rng.choice([m[:i] + m[i + 1] + m[i] + m[i + 2:], m[:i] + m[i + 1:], m[:i] + m[i] + m[i:]])
    if k == 4:                                                   # an odd character before a word (contractions, \b)
        ws = m.split(" ")
        i = rng.randrange(len(ws))
        ws[i] = rng.choice(ODD) + ws[i]
        return " ".join(ws)
    if k == 5:                                                   # an odd character after a word
        ws = m.split(" ")
        i = rng.randrange(len(ws))
        ws[i] = ws[i] + rng.choice(ODD)
        return " ".join(ws)
    if k == 6:
        return m + " im " + str(rng.choice([1, 9, 12, 14, 16, 17, 18, 21, 40]))
    if k == 7:
        return rng.choice(["🙂", "😢 ", "🔫", "💊💊 ", "🪢 ", "🩸"]) + m
    if k == 8:                                                   # past the 500-character cap, counted in code points
        pad = rng.choice(["x ", "𝒜 ", "我", "é "]) * rng.randrange(150, 300)
        return pad + m if rng.random() < 0.5 else m + " " + pad + m
    if k == 9:
        return m.replace(" ", rng.choice(["  ", "\n", "\t", " - ", "..."]))
    if k == 10:
        return "".join(c.upper() if rng.random() < 0.3 else c for c in m)
    return m + rng.choice([" can't", " won't", " ain't", " isn't", "n't", " i'm", " we're", " dont", " cant", " im"])


def main():
    searcher = Searcher.load()
    model = json.loads((ROOT / "search/artifacts/search.json").read_text())
    rng = random.Random(SEED)
    base = stored_messages() + phrase_messages(model, rng)
    messages = base + [variant(rng.choice(base), rng) for _ in range(1500)]
    messages += ["", " ", "'", "n't", "_can't", "éwon't", "ma'am", "🔫", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18"]
    seen = set()
    messages = [m for m in messages if not (m in seen or seen.add(m))]
    out = {
        "seed": SEED,
        "cases": [{"query": q, "crisis_words": searcher.crisis_words(q), "signal": searcher.suggest_signal(q),
                   "comfort_cue": searcher.comfort_cue(q), "suggest": searcher.suggest(q)} for q in messages],
        "topics": searcher.topics(),
        "topic": [{"id": t, "sensitive": s, "result": searcher.topic(t, sensitive=s)}
                  for t in model["themes"] for s in (False, True)],
    }
    json.dump(out, sys.stdout, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
