"""Advice search: real verses for an everyday situation, from a hand-reviewed pool.

A visitor types a situation in modern English ("I got laid off and I'm scared
about money") and gets verses from the Bhagavad Gita, the Quran and the Bible.

The search can only ever return verses from a hand-curated pool
(search/pool/*.json). Every pool verse was chosen and read by a person, has
one or more themes (grief, money_worry, ...) and a one-line note on why it
may help. A book offers a theme only if it has at least three verses for it.
Nothing outside the pool can be shown.

How a query is answered:

1. Safety first (search/safety.json, hand-written).
   - "crisis" (suicide, self-harm, abuse, threats of violence): no verses at
     all. The site shows a short kind note instead.
   - "sensitive" (sexual orientation, gender, pregnancy, disability, race,
     religion, caste, eating and body image): only verses from a short
     hand-picked list (search/pool/sensitive_set.json), ranked as usual,
     with the topic's gentle themes (loneliness, self_worth, ...) added.
   - otherwise "general".
2. Themes. Hand-written phrases (search/themes.json) spot the themes of the
   query: "passed away" -> grief, "burnt out" -> overwork_rest. A longer
   phrase hides the shorter ones inside it ("lost my job" is not "lost my").
3. Words. The query words are reduced to base forms found in the pool
   ("drinking" -> "drink"); a hand-written bridge adds scripture words for
   modern ones ("stressed" -> troubled, burden, weary).
4. Score, for each pool verse of each book:
       theme_weight * (sum of the query's theme scores over the verse's themes)
     + lex_weight * BM25(query words, verse text + note + theme names)
   If the query has themes, a verse must share one of them. A book whose
   best verse scores below min_score returns nothing rather than a weak
   match. If no book returns anything, the result is no_match (the site
   then offers the themes as chips: by_theme()).

This matches topics; it does not understand situations. Everything it needs
is in search/artifacts/search.json (see search/FORMAT.md). Plain Python, no
dependencies: run `python -m scripture.search "your situation"` to try it.
"""

import json
import math
import re
import sys
from array import array
from pathlib import Path

BOOKS = ("gita", "quran", "bible")
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "search" / "artifacts"

# Contractions typed without an apostrophe ("dont", "im"). "wont" and "ill" are left alone: both are real words.
NO_APOSTROPHE = {
    "cant": "can not", "dont": "do not", "didnt": "did not", "doesnt": "does not", "isnt": "is not",
    "wasnt": "was not", "arent": "are not", "werent": "were not", "havent": "have not", "hasnt": "has not",
    "hadnt": "had not", "couldnt": "could not", "wouldnt": "would not", "shouldnt": "should not",
    "im": "i", "ive": "i", "youre": "you", "theyre": "they", "thats": "that", "whats": "what",
    "hes": "he", "shes": "she",
}


def words(text):
    """Lower-case words, with modern contractions spelled out ("can't" -> "can not", "I'm" -> "i")."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"\b(won't|can't|ain't)", lambda m: {"won't": "will not", "can't": "can not", "ain't": "is not"}[m.group(1)], t)
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"'(m|re|s|ve|ll|d)\b", " ", t)
    return " ".join(NO_APOSTROPHE.get(w, w) for w in re.findall(r"[a-z]+", t)).split()


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


# ----- phrases -----

def compile_phrase(p):
    """'he|she hits me' -> [['he', 'she'], ['hits'], ['me']]; a trailing * matches any word with that start."""
    return [alt.split("|") for alt in p.split()]


def phrase_starts(tokens, phrase):
    """Positions where a compiled phrase matches consecutive tokens."""
    def ok(tok, alts):
        return any(tok == a or (a.endswith("*") and tok.startswith(a[:-1])) for a in alts)
    n = len(phrase)
    return [i for i in range(len(tokens) - n + 1) if all(ok(tokens[i + j], phrase[j]) for j in range(n))]


def one_edit_apart(a, b):
    """True if b is a with one letter added, removed, changed, or two neighbours swapped."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def round4(x):
    """Round to 4 decimals, halves away from zero (JavaScript: Math.sign(x) * Math.floor(Math.abs(x) * 1e4 + 0.5) / 1e4)."""
    return math.copysign(math.floor(abs(x) * 1e4 + 0.5), x) / 1e4


# ----- the searcher -----

class Searcher:
    """Answers queries from search/artifacts/search.json (written by search/build.py)."""

    def __init__(self, model, vectors=None):
        p = self.params = model["params"]
        self.lex = Lexicon(model["lexicon"])
        self.vocabulary = model["vocabulary"]            # pool word -> term ("" for a filler word)
        self.bridge = model["bridge"]
        self.replace = set(model["bridge_replace"])
        self.max_bridge = max((len(k.split()) for k in self.bridge), default=1)
        self.theme_info = model["themes"]
        self.theme_phrases = {t: [compile_phrase(x) for x in d["phrases"]] for t, d in model["themes"].items()}
        safety = model["safety"]
        self.fuzzy = safety["fuzzy"]
        self.idioms = [compile_phrase(x) for x in safety["idioms"]]
        self.crisis = {t: [compile_phrase(x) for x in ps] for t, ps in safety["crisis"].items()}
        self.sensitive = {t: [compile_phrase(x) for x in d["phrases"]] for t, d in safety["sensitive"].items()}
        self.sensitive_themes = {t: d["themes"] for t, d in safety["sensitive"].items()}
        self.books = {}
        row = 0
        for book in BOOKS:
            verses = model["pool"][book]
            b = {"verses": verses, "index": {v["ref"]: i for i, v in enumerate(verses)},
                 **bm25_index([v["terms"] for v in verses])}
            if vectors is not None:
                b["vectors"] = [unit(vectors[row + i]) for i in range(len(verses))]
            row += len(verses)
            self.books[book] = b
        self.k1, self.b = p["bm25_k1"], p["bm25_b"]

    @classmethod
    def load(cls, directory=ARTIFACTS):
        d = Path(directory)
        model = json.loads((d / "search.json").read_text())
        vectors = None
        if (d / "pool_vectors.i8").exists():
            raw = array("b", (d / "pool_vectors.i8").read_bytes())
            dim = model["dim"]
            vectors = [raw[i:i + dim].tolist() for i in range(0, len(raw), dim)]
        return cls(model, vectors)

    # ----- understanding the query -----

    def classify(self, query):
        """("crisis" | "sensitive" | "general", topic or None)."""
        toks = words(query)
        crisis_toks = list(toks)
        for idiom in self.idioms:                        # "killing it at work" is not a threat
            for i in phrase_starts(toks, idiom):
                crisis_toks[i:i + len(idiom)] = [""] * len(idiom)
        crisis_toks = [next((f for f in self.fuzzy if len(t) >= 5 and one_edit_apart(t, f)), t) for t in crisis_toks]
        for topic, phrases in self.crisis.items():
            if any(phrase_starts(crisis_toks, p) for p in phrases):
                return "crisis", topic
        for topic, phrases in self.sensitive.items():
            if any(phrase_starts(toks, p) for p in phrases):
                return "sensitive", topic
        return "general", None

    def query_themes(self, query):
        """{theme: score in (0, 1]} from the theme phrases; the best theme scores 1.

        Each match counts its number of words. A match that lies inside a
        longer match is dropped, so "can't stop crying" (grief) hides
        "can't stop" (temptation).
        """
        toks = words(query)
        found = {(i, i + len(p), t) for t, phrases in self.theme_phrases.items()
                 for p in phrases for i in phrase_starts(toks, p)}
        raw = {}
        for s, e, t in found:
            if not any(s2 <= s and e <= e2 and e2 - s2 > e - s for s2, e2, _ in found):
                raw[t] = raw.get(t, 0) + (e - s)
        top = max(raw.values(), default=0)
        return {t: raw[t] / top for t in sorted(raw, key=lambda t: (-raw[t], t))}

    def term_of(self, word):
        """Pool term of a lower-case word, or None for filler or unknown words."""
        t = self.vocabulary.get(word)
        if t is None:                                    # not in the pool: try a base form that is
            t = self.vocabulary.get(self.lex.normalise(word, self.vocabulary))
        return t or None

    def expand(self, query, bridge=True):
        """Query words as pool terms: {term: (weight, source word)}.

        A typed word has weight 1, a bridge word bridge_weight. A term reached
        twice keeps its highest weight (the first source on a tie).
        """
        toks = words(query)
        out = {}

        def add(term, weight, source):
            if term is not None and weight > out.get(term, (0, ""))[0]:
                out[term] = (weight, source)

        i = 0
        while i < len(toks):
            key, n = None, 1
            if bridge:                                   # longest bridge phrase starting here
                for n in range(min(self.max_bridge, len(toks) - i), 0, -1):
                    if " ".join(toks[i:i + n]) in self.bridge:
                        key = " ".join(toks[i:i + n])
                        break
            if key:
                for target in self.bridge[key]:
                    add(self.term_of(target), self.params["bridge_weight"], key)
            # A phrase ("laid off") or a word whose scripture meaning differs
            # ("job" is the prophet Job) is searched only through its bridge words.
            if not key or (n == 1 and key not in self.replace):
                w = toks[i]
                if w not in self.lex.stopwords and w not in self.lex.query_stopwords:
                    add(self.term_of(w), 1.0, w)
            i += n if key else 1
        return out

    # ----- scoring -----

    def score(self, book, verse_i, themes, q):
        """(score, matched themes, matched terms) of one pool verse."""
        v = self.books[book]["verses"][verse_i]
        hit_themes = [t for t in v["themes"] if t in themes]
        theme_part = sum(themes[t] for t in hit_themes)
        lex_part = self.bm25(book, verse_i, q)
        return (self.params["theme_weight"] * theme_part + self.params["lex_weight"] * lex_part,
                hit_themes, [t for t in q if t in self.books[book]["tf"][verse_i]])

    def bm25(self, book, i, q):
        b = self.books[book]
        total = 0.0
        for t, (w, _) in q.items():
            tf = b["tf"][i].get(t, 0)
            if tf:
                norm = tf + self.k1 * (1 - self.b + self.b * b["length"][i] / b["avgdl"])
                total += w * b["idf"][t] * tf * (self.k1 + 1) / norm
        return total

    def search(self, query, k=3):
        """Top-k pool verses from each book.

        Returns {"category", "topic", "no_match", "themes": [{"id", "score"}],
        "notes": [{"theme", "note"}], "results": {book: [hit, ...]}}. "notes"
        are the short notes of the query's themes that have one (shown with
        the results). Each hit has ref, text (for display),
        themes, why, score, and matched = {"themes": [...], "words": [{"word",
        "from"}]}: the query's themes the verse carries and the query words
        found in it, with the word of the query each came from.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        category, topic = self.classify(query)
        out = {"category": category, "topic": topic, "no_match": False, "themes": [], "notes": [],
               "results": {b: [] for b in BOOKS}}
        if category == "crisis":
            return out
        themes = self.query_themes(query)
        if category == "sensitive":                     # gentle default themes for the topic, below typed ones
            for t in self.sensitive_themes[topic]:
                themes.setdefault(t, self.params["sensitive_theme_score"])
        out["themes"] = [{"id": t, "score": round4(s)} for t, s in themes.items()]
        out["notes"] = [{"theme": t, "note": self.theme_info[t]["note"]} for t in themes if "note" in self.theme_info[t]]
        q = self.expand(query)
        for book in BOOKS:
            scored = []
            for i, v in enumerate(self.books[book]["verses"]):
                if category == "sensitive" and not v["sensitive"]:
                    continue
                s, hit_themes, hit_terms = self.score(book, i, themes, q)
                if themes and not hit_themes:           # off-topic for this query
                    continue
                scored.append((round4(s), i, hit_themes, hit_terms))
            scored.sort(key=lambda x: (-x[0], x[1]))
            for s, i, hit_themes, hit_terms in scored[:k]:
                if s < self.params["min_score"]:
                    break
                out["results"][book].append(self._hit(book, i, s, {
                    "themes": hit_themes, "words": [{"word": t, "from": q[t][1]} for t in hit_terms]}))
        out["no_match"] = not any(out["results"].values())
        return out

    def by_theme(self, theme, k=3):
        """Pool verses of one theme per book, for the theme chips.

        A verse scores 1 / (position of the theme in its list + 1): verses
        that list the theme first come first; ties keep pool order.
        """
        if theme not in self.theme_info:
            raise ValueError(f"unknown theme {theme!r}")
        out = {}
        for book in BOOKS:
            rows = [(round4(1 / (v["themes"].index(theme) + 1)), i)
                    for i, v in enumerate(self.books[book]["verses"]) if theme in v["themes"]]
            rows.sort(key=lambda x: (-x[0], x[1]))
            out[book] = [self._hit(book, i, s, {"themes": [theme], "words": []}) for s, i in rows[:k]]
        return out

    def themes(self):
        """Every theme with its display name, description and pool verse count per book."""
        return [{"id": t, "name": d["name"], "description": d["description"],
                 "verses": {b: sum(t in v["themes"] for v in self.books[b]["verses"]) for b in BOOKS}}
                for t, d in self.theme_info.items()]

    def similar(self, book, ref, k=3, sensitive_only=False):
        """The pool verses closest in meaning to one pool verse, per book (cosine of verse vectors).

        Verse vectors come from word vectors trained on the three books. A
        verse with no vector (all zeros) scores 0 against everything.
        """
        if book not in self.books or ref not in self.books[book]["index"]:
            raise ValueError(f"not a pool verse: {book} {ref!r}")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        src = self.books[book]["vectors"][self.books[book]["index"][ref]]
        out = {}
        for other in BOOKS:
            b = self.books[other]
            rows = [(round4(sum(x * y for x, y in zip(src, vec))), j) for j, vec in enumerate(b["vectors"])
                    if not (other == book and b["verses"][j]["ref"] == ref)
                    and (b["verses"][j]["sensitive"] or not sensitive_only)]
            rows.sort(key=lambda x: (-x[0], x[1]))
            out[other] = [self._hit(other, j, s, {"themes": [], "words": []}) for s, j in rows[:k]]
        return out

    def _hit(self, book, i, score, matched):
        v = self.books[book]["verses"][i]
        return {"ref": v["ref"], "text": v["text"], "themes": v["themes"], "why": v["why"],
                "score": score, "matched": matched}


def bm25_index(docs):
    """Per-book BM25 statistics: term counts per verse, lengths, and idf = ln(1 + (n - df + 0.5) / (df + 0.5))."""
    tf = []
    for terms in docs:
        c = {}
        for t in terms:
            c[t] = c.get(t, 0) + 1
        tf.append(c)
    df = {}
    for c in tf:
        for t in c:
            df[t] = df.get(t, 0) + 1
    n = len(docs)
    length = [len(d) for d in docs]
    return {"tf": tf, "length": length, "avgdl": sum(length) / n if n else 1.0,
            "idf": {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}}


def unit(v):
    """v divided by its length; an all-zero vector stays all zeros (never NaN)."""
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else [0.0] * len(v)


def main(argv):
    if not argv:
        print('usage: python -m scripture.search "describe your situation" [k]')
        return
    k = int(argv[1]) if len(argv) > 1 else 3
    res = Searcher.load().search(argv[0], k=k)
    print(f'category: {res["category"]}' + (f' ({res["topic"]})' if res["topic"] else ""))
    if res["category"] == "crisis":
        print("No verses for this one. You matter, and you don't have to carry this alone: "
              "please talk to someone you trust today.")
        return
    print("themes: " + (", ".join(f'{t["id"]} {t["score"]:.2f}' for t in res["themes"]) or "none"))
    if res["no_match"]:
        print("Nothing in the pool fits these words. Try one of the themes.")
        return
    for book in BOOKS:
        print(f"\n== {book} ==")
        if not res["results"][book]:
            print("  (nothing fitting in this book)")
        for h in res["results"][book]:
            print(f'{h["ref"]:>18}  {h["score"]:.3f}  {h["text"]}')
            print(f'{"":>20}why: {h["why"]}  [{", ".join(h["matched"]["themes"] + [m["word"] for m in h["matched"]["words"]])}]')


if __name__ == "__main__":
    main(sys.argv[1:])
