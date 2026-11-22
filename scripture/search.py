"""Advice search: real verses for an everyday situation, from a reviewed pool.

Since round 7 the site works as "type, then pick a topic". A visitor types a
situation in modern English ("I got laid off and I'm scared about money");
suggest() answers with topics only (money_worry, fear_anxiety), never verses,
or with a kind note and no topics if the message may be a crisis. Verses from
the Bhagavad Gita, the Quran and the Bible appear only after the visitor taps a
topic: topic() returns that topic's verses in the hand-set order. The site
never shows a verse chosen from free text. Three blind reviews in a row found
that keyword rules miss a few percent of crisis messages in unusual wording,
and a verse picked for such a message can do harm; a topic list the visitor
chooses from cannot hand them a verse they did not ask for.

search(), described below, is the free-text verse search of rounds 4 to 6. It
is kept only so the recorded eval history can be reproduced; the site does not
use it.

The search can only ever return verses from a hand-curated pool
(search/pool/*.json). Every pool verse was reviewed before it was kept, has
one or more themes (grief, money_worry, ...) and a one-line note on why it
may help. A book offers a theme only if it has at least three verses for it.
Nothing outside the pool can be shown.

How a message is read (steps 0 to 2 are shared by suggest() and search()):

0. Only the first 500 characters of the query are read.
1. Safety first (search/safety.json, hand-written).
   - "crisis": no topics, no verses; the site shows a short kind note
     (safety.json "crisis_message") instead. Two layers fire it:
     the crisis phrase lists (suicide, self-harm, abuse, harm to others),
     and a deliberately over-sensitive "distress" gate: rules that fire when
     a few words occur close together in any order ("pills" near "how
     many", "he" near "grabs", "better off" near "without me"). Everyday
     idioms ("killing it at work") and a third person's death ("my dad
     passed away", "her husband died by suicide") are blanked first, so
     bereavement goes to the grief theme instead.
   - "sensitive" (sexual orientation, gender, pregnancy, disability, race,
     religion, caste, eating and body image, mental health, addiction,
     immigration, divorce): only topics with verses from a short
     hand-picked list (search/pool/sensitive_set.json), under gentle labels,
     with the topic's gentle themes (loneliness, self_worth, ...) added after
     the typed ones; topic(sensitive=True) then shows only those verses.
   - otherwise "general".
2. Themes. Hand-written phrases (search/themes.json) spot the themes of the
   query: "passed away" -> grief, "burnt out" -> overwork_rest. A longer
   phrase hides the shorter ones inside it ("lost my job" is not "lost my").
   A word with a one-letter typo of a theme word counts as it ("lonley").
   Themes scoring below min_theme_score are dropped; if grief matched, only
   grief is kept (exclusive_themes; grief counts at any score).
3. suggest() offers up to 4 of these themes as topics, best first. If the
   message has any first-person harm, plan, secrecy, age-under-18 or
   adult-contact word (safety.json "suggest"), it never offers courage,
   love, friendship, marriage, family or forgiveness, and at most 2 topics,
   even if the gate did not fire. If it names another person (he, her,
   husband, teacher, someone ...), an event done to the writer ("something
   happened", "said no"), mental health or anger at God, it offers only the
   comfort topics (themes.json "kind": grief, fear, loneliness, illness,
   doubt and faith, hope, self-worth), never one whose verses ask the reader
   to change their own behaviour or bear something (anger, guilt,
   self-control, patience ...). A gentle topic for a sensitive message
   needs at least 3 sensitive verses. With no topic, category "none": the
   site shows the full topic list (topics()); but if no theme matched at
   all and the message has a signal word that does not also occur in
   everyday idioms (safety.json "safeguard_exempt"), the crisis note
   instead (the no-topic safeguard).
   search() (eval history only) instead lets the themes choose verses:
   a verse can be shown only if it carries one
   of the query's themes; with no theme there are no verses (no_match; the
   site then offers the themes as chips: by_theme()). Verses are ordered by
   the best-ranked query theme they carry, then by the hand-set priority of
   the verse for that theme (the 3-5 verses a reviewer chose to show first),
   then by how many query themes they carry, and only then, as a weak tie
   break, by BM25 of the query words against the verse text, note and theme
   names (query words reduced to base forms found in the pool; a
   hand-written bridge adds scripture words for modern ones).

This matches topics in a reviewed selection; it does not understand
situations. Everything it needs is in search/artifacts/search.json (see
search/FORMAT.md). Plain Python, no dependencies: run
`python -m scripture.search "your situation"` for the suggested topics and
`python -m scripture.search --topic grief` for a topic's verses.
"""

import json
import math
import re
import sys
from array import array
from pathlib import Path

BOOKS = ("gita", "quran", "bible")
BOOK_NAMES = {"gita": "Bhagavad Gita", "quran": "Quran", "bible": "Bible"}
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


# Numbers written in digits, read as words for the safety check only: 1-17 are spelled out, so a stated age
# ("i'm 14") can be read; any other number becomes the word "num", so "took 30 tylenol" can be read too.
NUMBER_WORDS = dict(zip(map(str, range(1, 18)), (
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen").split()))


def spell_numbers(text):
    """Each run of ASCII digits becomes " <word> ": its English word if it is exactly 1-17 (no leading zero), else "num"."""
    return re.sub(r"[0-9]+", lambda m: f" {NUMBER_WORDS.get(m.group(), 'num')} ", text)


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
    """'he|she hits me' -> one slot per word: (alternatives, exact words, prefixes); a trailing * matches any word with that start.

    For example "hit*|beat me" -> [(['hit*', 'beat'], {'beat'}, ('hit',)), (['me'], {'me'}, ())].
    """
    out = []
    for slot in p.split():
        alts = slot.split("|")
        out.append((alts, frozenset(a for a in alts if not a.endswith("*")),
                    tuple(a[:-1] for a in alts if a.endswith("*"))))
    return out


def phrase_starts(tokens, phrase, tokset=None):
    """Positions where a compiled phrase matches consecutive tokens. The empty word "" never matches.

    `tokset` (set(tokens)) is only a speed-up: a phrase with a slot of exact
    words none of which occurs in the query cannot match.
    """
    if tokset is not None and any(not s[2] and s[1].isdisjoint(tokset) for s in phrase):
        return []

    def ok(tok, slot):
        return bool(tok) and (tok in slot[1] or tok.startswith(slot[2]))
    n = len(phrase)
    return [i for i in range(len(tokens) - n + 1) if all(ok(tokens[i + j], phrase[j]) for j in range(n))]


def compile_rule(r):
    """'pills|meds + how many, how much ! vitamins' -> (groups, vetoes).

    Groups are separated by " + "; a group lists alternative phrases separated
    by ", ". Each veto (after " ! ") is one phrase. A rule may start with
    "~N " to use a window of N words instead of the default.
    """
    window = None
    if r.startswith("~"):
        n, r = r[1:].split(" ", 1)
        window = int(n)
    parts = r.split(" ! ")
    return ([[compile_phrase(a) for a in g.split(", ")] for g in parts[0].split(" + ")],
            [compile_phrase(v) for v in parts[1:]], window)


def rule_fires(tokens, rule, window, tokset=None):
    """True if every group matches somewhere, all within `window` words, with no veto near that span.

    A group matches where any of its phrases matches. The span runs from the
    first word of the earliest chosen match to the last word of the latest
    one, so the groups may come in any order; its length (end - start) must
    be at most `window`. A veto vetoes a span if its whole match lies within
    the span widened by `window` words on each side. The rule fires if some
    choice of one match per group gives a span that is short enough and not vetoed.
    """
    groups, vetoes, own = rule
    window = window if own is None else own
    hits = []
    for g in groups:
        hits.append([(i, i + len(ph)) for ph in g for i in phrase_starts(tokens, ph, tokset)])
        if not hits[-1]:
            return False
    veto_hits = [(i, i + len(v)) for v in vetoes for i in phrase_starts(tokens, v, tokset)]

    def walk(k, lo, hi):
        if k == len(hits):
            return hi - lo <= window and not any(lo - window <= s and e <= hi + window for s, e in veto_hits)
        return any(walk(k + 1, min(lo, s), max(hi, e)) for s, e in hits[k])
    return walk(0, len(tokens), 0)


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
        # Theme words a typo can stand for: single-word phrases without a wildcard, in sorted order.
        self.theme_words = sorted({d for x in self.theme_phrases.values() for ph in x if len(ph) == 1 for d in ph[0][1]})
        self.known_words = {a for x in self.theme_phrases.values() for ph in x
                            for slot in ph for a in slot[1]} | self.lex.stopwords
        safety = model["safety"]
        self.crisis_message = safety["crisis_message"]
        self.fuzzy = safety["fuzzy"]
        self.emoji = safety.get("emoji", {})
        self.idioms = [compile_phrase(x) for x in safety["idioms"]]
        self.bereavement = [compile_phrase(x) for x in safety["bereavement"]]
        self.distress_window = safety["distress"]["window"]
        self.distress = [compile_rule(r) for r in safety["distress"]["rules"]]
        self.crisis = {t: [compile_phrase(x) for x in ps] for t, ps in safety["crisis"].items()}
        self.sensitive = {t: [compile_phrase(x) for x in d["phrases"]] for t, d in safety["sensitive"].items()}
        self.sensitive_themes = {t: d["themes"] for t, d in safety["sensitive"].items()}
        # Sensitive-set verses a topic never shows ({"gita": ["2:40"]}), e.g. a verse that reads wrongly for abortion.
        self.sensitive_exclude = {t: {b: set(d.get("exclude", {}).get(b, ())) for b in BOOKS}
                                  for t, d in safety["sensitive"].items()}
        # topic(sensitive=True) does not know which sensitive topic led there, so it leaves out every topic's excludes.
        self.sensitive_exclude_any = {b: set().union(*(e[b] for e in self.sensitive_exclude.values())) for b in BOOKS}
        sug = safety.get("suggest", {})
        self.suggest_signals = [compile_phrase(x) for x in sug.get("signals", [])]
        self.suggest_never = set(sug.get("never_after_signal", []))
        # The signals that also count for the no-topic safeguard: all but those that occur in everyday idioms.
        exempt = set(sug.get("safeguard_exempt", []))
        self.safeguard_signals = [compile_phrase(x) for x in sug.get("signals", []) if x not in exempt]
        # Wording that names another person, an event done to the writer, mental health or anger at God: then
        # only "comfort" topics (themes.json "kind") are offered, never one that asks the reader to change or bear.
        self.comfort_cues = [compile_phrase(x) for g in sug.get("comfort_only_cues", {}).values() for x in g]
        self.comfort_topics = {t for t, d in model["themes"].items() if d.get("kind") == "comfort"}
        self.suggest_max = sug.get("max_topics", 4)
        self.suggest_min_sensitive = sug.get("min_sensitive_verses", 1)
        self.suggest_max_signal = sug.get("max_topics_after_signal", self.suggest_max)
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

    def crisis_words(self, query):
        """words(spell_numbers(query)) with idioms and a third person's death blanked to "", then crisis typos fixed.

        A few emoji (safety.json "emoji": "🔫" -> "emojigun") are replaced by
        a word first, then numbers 1-17 typed as digits are spelled out ("i'm
        14" -> "i fourteen"). Idioms ("killing it at work") and bereavement phrases ("my
        dad passed away", "died by suicide") are matched on these words; every
        word they cover becomes "". Then each word of 5 or more letters one
        edit away from a `fuzzy` word becomes that word (the first in list order).
        """
        query = query[:self.params["max_query_chars"]]
        for ch, word in self.emoji.items():               # a few emoji that stand for a means of harm ("🔫")
            query = query.replace(ch, f" {word} ")
        toks = words(spell_numbers(query))
        out, tokset = list(toks), set(toks)
        for phrase in self.idioms + self.bereavement:
            for i in phrase_starts(toks, phrase, tokset):
                out[i:i + len(phrase)] = [""] * len(phrase)
        return [next((f for f in self.fuzzy if len(t) >= 5 and one_edit_apart(t, f)), t) for t in out]

    def classify(self, query):
        """("crisis" | "sensitive" | "general", topic or None). The distress gate's topic is "distress"."""
        query = query[:self.params["max_query_chars"]]
        crisis_toks = self.crisis_words(query)
        cset = set(crisis_toks)
        for topic, phrases in self.crisis.items():
            if any(phrase_starts(crisis_toks, p, cset) for p in phrases):
                return "crisis", topic
        if any(rule_fires(crisis_toks, r, self.distress_window, cset) for r in self.distress):
            return "crisis", "distress"
        toks = words(query)
        tset = set(toks)
        for topic, phrases in self.sensitive.items():
            if any(phrase_starts(toks, p, tset) for p in phrases):
                return "sensitive", topic
        return "general", None

    def theme_tokens(self, query):
        """words(query), with each unknown word of theme_fuzzy_min_len (6)+ letters one edit from a theme word replaced by it.

        Unknown = not a word of any theme phrase, not a stopword, and with no
        pool term (term_of finds none, so "honestly" -> "honest" is known).
        The first theme word in sorted order wins ("lonley" -> "lonely").
        """
        n = self.params["theme_fuzzy_min_len"]
        return [t if len(t) < n or t in self.known_words or self.term_of(t) is not None
                else next((w for w in self.theme_words if one_edit_apart(t, w)), t)
                for t in words(query[:self.params["max_query_chars"]])]

    def query_themes(self, query):
        """{theme: score in (0, 1]} from the theme phrases; the best theme scores 1.

        Each match counts its number of words. A match that lies inside a
        longer match is dropped, so "can't stop crying" (grief) hides
        "can't stop" (temptation). Scores are not rounded here.
        """
        toks = self.theme_tokens(query)
        tset = set(toks)
        found = {(i, i + len(p), t) for t, phrases in self.theme_phrases.items()
                 for p in phrases for i in phrase_starts(toks, p, tset)}
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

    def selection_themes(self, query, category, topic):
        """The themes that choose verses, best first: {theme: score}.

        The query's themes scoring at least min_theme_score. If one of
        exclusive_themes (grief, then gratitude) is among them (grief at any
        score: exclusive_any_score), only the first such theme in that list is
        kept, with score 1. For a sensitive query,
        the topic's gentle themes are then appended (in list order) with score
        sensitive_theme_score.
        """
        p = self.params
        found = self.query_themes(query)
        themes = {t: s for t, s in found.items() if s >= p["min_theme_score"]}
        exclusive = next((t for t in p["exclusive_themes"]
                          if t in themes or (t in found and t in p["exclusive_any_score"])), None)
        if exclusive:
            themes = {exclusive: 1.0}
        if category == "sensitive":
            for t in self.sensitive_themes[topic]:
                themes.setdefault(t, p["sensitive_theme_score"])
        return themes

    def rank_key(self, book, verse_i, themes, q, use_priority=True):
        """(sort key, best theme, priority, matched themes, matched terms) of one pool verse, or None if off-topic.

        Sort key, ascending: (position of the verse's best query theme in
        `themes`, the verse's priority for that theme or 1000 if it has none,
        minus the rounded sum of the query's scores of all its matched themes,
        minus the rounded BM25, pool position). Without use_priority (sensitive
        queries, whose verses come from the short sensitive set) every
        priority counts as none, so the query's words order the verses.
        """
        v = self.books[book]["verses"][verse_i]
        hit = [t for t in v["themes"] if t in themes]
        if not hit:
            return None
        order = list(themes)
        best = min(hit, key=order.index)
        prio = v.get("priority", {}).get(best) if use_priority else None
        bm = round4(self.bm25(book, verse_i, q))
        key = (order.index(best), 1000 if prio is None else prio, -round4(sum(themes[t] for t in hit)), -bm, verse_i)
        return key, best, prio, hit, [t for t in q if t in self.books[book]["tf"][verse_i]]

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
        """EVAL HISTORY ONLY, NOT USED BY THE SITE: top-k verses chosen from free text.

        Kept so evals/search_eval.py can reproduce the recorded numbers of
        rounds 4 to 6. The site never shows a verse chosen from free text: it
        calls suggest() (topics only) and then topic() for the topic the
        visitor taps. Only the first max_query_chars (500) characters are read.

        Returns {"category", "topic", "message", "no_match", "themes": [{"id",
        "score"}], "notes": [{"theme", "note"}], "results": {book: [hit, ...]}}.
        For a crisis: "message" is the kind note to show, and themes, notes
        and results are empty (no_match is false: it is not a failed search).
        Otherwise "message" is None, "themes" are the themes that chose the
        verses (selection_themes, scores rounded) and "notes" the notes of
        those themes that have one. Each hit has ref, text (for display),
        themes, why, theme (the query theme it was chosen for), priority (its
        hand-set rank for that theme, or None), score (the rounded BM25 tie
        break) and matched = {"themes": [...], "words": [{"word", "from"}]}:
        the query's themes the verse carries and the query words found in it,
        with the word of the query each came from.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        query = query[:self.params["max_query_chars"]]
        category, topic = self.classify(query)
        out = {"category": category, "topic": topic, "message": None, "no_match": False, "themes": [],
               "notes": [], "results": {b: [] for b in BOOKS}}
        if category == "crisis":
            out["message"] = self.crisis_message
            return out
        themes = self.selection_themes(query, category, topic)
        out["themes"] = [{"id": t, "score": round4(s)} for t, s in themes.items()]
        out["notes"] = [{"theme": t, "note": self.theme_info[t]["note"]} for t in themes if "note" in self.theme_info[t]]
        q = self.expand(query)
        for book in BOOKS:
            rows = []
            for i, v in enumerate(self.books[book]["verses"]):
                if category == "sensitive" and (not v["sensitive"] or v["ref"] in self.sensitive_exclude[topic][book]):
                    continue
                r = self.rank_key(book, i, themes, q, use_priority=category != "sensitive")
                if r is not None:
                    rows.append(r)
            rows.sort(key=lambda r: r[0])
            for key, best, prio, hit_themes, hit_terms in rows[:k]:
                h = self._hit(book, key[-1], -key[3], {
                    "themes": hit_themes, "words": [{"word": t, "from": q[t][1]} for t in hit_terms]})
                out["results"][book].append({**h, "theme": best, "priority": prio})
        out["no_match"] = not any(out["results"].values())
        return out

    # ----- what the site uses: suggest() then topic() -----

    def suggest_signal(self, query):
        """True if the message has any first-person harm, plan, secrecy, age-under-18 or adult-contact word.

        Matched on crisis_words (so "i'm 14" and "🔫" count, and idioms and a
        third person's death are blanked) with the phrases of safety.json
        "suggest" "signals". Deliberately broad: it only makes suggest() offer
        fewer topics.
        """
        toks = self.crisis_words(query)
        tset = set(toks)
        return any(phrase_starts(toks, p, tset) for p in self.suggest_signals)

    def comfort_cue(self, query):
        """True if the message names another person or an event done to the writer, or has mental-health wording.

        Matched on words(query) with the phrases of safety.json "suggest"
        "comfort_only_cues": a third-person pronoun or person word (he, her,
        husband, dad, teacher, friend, someone, people ...), passive-event
        wording (something happened, said no, made me, touched ...),
        mental-health wording (depressed, trauma, panic attack ...) or anger at
        God. Then suggest() offers only the "comfort" topics.
        """
        toks = words(query[:self.params["max_query_chars"]])
        tset = set(toks)
        return any(phrase_starts(toks, p, tset) for p in self.comfort_cues)

    def suggest(self, query):
        """Topics to offer for a typed message. The only free-text entry point the site uses; it never returns verses.

        Returns {"category", "note", "topics": [{"id", "label", "blurb"}]}:
        - "crisis" (a crisis phrase or the distress gate fired): note is the
          kind crisis message and topics is empty;
        - "sensitive": the message's themes and then the sensitive topic's
          gentle themes, keeping only topics with sensitive-set verses, under
          their gentle labels where they have one; the site then calls
          topic(id, sensitive=True);
        - "general": the message's themes as the theme matching ranks them
          (selection_themes), keeping only topics some book offers;
        - "none": nothing matched; topics is empty and the site shows the
          full list (topics()). If no theme matched at all and the message
          has a signal word that is not exempt (safeguard), "crisis" instead.
        If the message has a signal word (suggest_signal), the topics in
        safety.json "suggest" "never_after_signal" (courage, love,
        friendship, marriage, family, forgiveness) are never offered and at
        most max_topics_after_signal (2) are; otherwise at most max_topics (4).
        If the message names another person, an event done to the writer,
        mental health or anger at God (comfort_cue), only the "comfort" topics
        (themes.json "kind") are offered: never one whose verses ask the reader
        to change their own behaviour or to bear something.
        Only the first max_query_chars (500) characters are read.
        """
        query = query[:self.params["max_query_chars"]]
        category, topic = self.classify(query)
        if category == "crisis":
            return {"category": "crisis", "note": self.crisis_message, "topics": []}
        out = self.offer(query, category, topic)
        if out["category"] == "none" and self.safeguard(query):
            return {"category": "crisis", "note": self.crisis_message, "topics": []}
        return out

    def safeguard(self, query):
        """True if no theme phrase matches the message at all and it has a signal word that is not exempt.

        Then suggest() shows the crisis note rather than the full topic list:
        when in doubt it fires. The exempt signals (safety.json "suggest"
        "safeguard_exempt": kill*, cut, last, tonight, pills ...) also occur
        in everyday idioms ("killing it last night", "cut from the team").
        """
        query = query[:self.params["max_query_chars"]]
        if self.query_themes(query):
            return False
        toks = self.crisis_words(query)
        tset = set(toks)
        return any(phrase_starts(toks, p, tset) for p in self.safeguard_signals)

    def offer(self, query, category, topic):
        """suggest() after the safety check, for a "general" or "sensitive" message.

        The benchmark also calls it for crisis messages, as if the gate had
        missed them, to see what the suggestion layer alone would offer.
        """
        query = query[:self.params["max_query_chars"]]
        sensitive = category == "sensitive"
        # A topic needs a verse to show; a gentle topic for a sensitive message needs at least min_sensitive_verses.
        need = self.suggest_min_sensitive if sensitive else 1
        ids = [t for t in self.selection_themes(query, category, topic) if self.topic_count(t, sensitive) >= need]
        cap = self.suggest_max
        if self.suggest_signal(query):
            ids = [t for t in ids if t not in self.suggest_never]
            cap = self.suggest_max_signal
        if self.comfort_cue(query):
            ids = [t for t in ids if t in self.comfort_topics]
        topics = [self._topic_label(t, sensitive) for t in ids[:cap]]
        if not topics and not sensitive:
            category = "none"
        return {"category": category, "note": None, "topics": topics}

    def topic(self, theme, k=None, sensitive=False):
        """The verses of one topic per book, in the hand-set priority order: what the site shows after a tap.

        Per book: verses with a hand-set priority for the topic first (1, 2,
        ...), then by 1 / (position of the topic in the verse's list + 1),
        then pool order; the first k (all if k is None). With sensitive=True
        only sensitive-set verses (and none that any sensitive topic
        excludes). Returns {"id", "label", "blurb", "note", "sensitive",
        "books": {book: {"verses": [{"ref", "text", "why", "themes",
        "priority"}], "nothing": None, or a sentence saying the book has no
        verse for this topic}}}.
        """
        rows = self._theme_rows(theme, k, sensitive)
        out = {**self._topic_label(theme, sensitive), "note": self.theme_info[theme].get("note"),
               "sensitive": sensitive, "books": {}}
        for book in BOOKS:
            verses = []
            for p, _, i in rows[book]:
                v = self.books[book]["verses"][i]
                verses.append({"ref": v["ref"], "text": v["text"], "why": v["why"], "themes": v["themes"],
                               "priority": None if p == 1000 else p})
            out["books"][book] = {"verses": verses, "nothing": None if verses else
                                  f"The {BOOK_NAMES[book]} has no verse for this topic in the reviewed selection."}
        return out

    def topics(self):
        """Every topic that has a verse in some book, with its label, blurb and verse counts per book.

        "verses" counts what topic() shows, "sensitive_verses" what
        topic(sensitive=True) shows. A theme with no verse left in any book
        (marriage, since round 7) is not listed.
        """
        return [{**self._topic_label(t, False),
                 "verses": {b: self.topic_count(t, False, b) for b in BOOKS},
                 "sensitive_verses": {b: self.topic_count(t, True, b) for b in BOOKS}}
                for t in self.theme_info if self.topic_count(t)]

    def topic_count(self, theme, sensitive=False, book=None):
        """How many verses topic() can show for a theme, in one book or in all of them."""
        return sum(len(rows) for rows in self._theme_rows(theme, None, sensitive, [book] if book else BOOKS).values())

    def _topic_label(self, theme, sensitive):
        d = self.theme_info[theme]
        if sensitive and "gentle_label" in d:
            return {"id": theme, "label": d["gentle_label"], "blurb": d["gentle_blurb"]}
        return {"id": theme, "label": d.get("label", d["name"]), "blurb": d.get("blurb", d["description"])}

    def _theme_rows(self, theme, k, sensitive, books=BOOKS):
        """{book: [(priority or 1000, score, pool index), ...]}, sorted, the first k (all if k is None)."""
        if theme not in self.theme_info:
            raise ValueError(f"unknown theme {theme!r}")
        if k is not None and (not isinstance(k, int) or isinstance(k, bool) or k < 1):
            raise ValueError(f"k must be a positive integer, got {k!r}")
        out = {}
        for book in books:
            rows = [(v.get("priority", {}).get(theme, 1000), round4(1 / (v["themes"].index(theme) + 1)), i)
                    for i, v in enumerate(self.books[book]["verses"]) if theme in v["themes"]
                    and (not sensitive or (v["sensitive"] and v["ref"] not in self.sensitive_exclude_any[book]))]
            rows.sort(key=lambda x: (x[0], -x[1], x[2]))
            out[book] = rows if k is None else rows[:k]
        return out

    def by_theme(self, theme, k=3):
        """topic()'s order in search()'s hit format (score, matched); kept for the fixtures and the eval history."""
        if k is None:
            raise ValueError("k must be a positive integer, got None")
        rows = self._theme_rows(theme, k, False)
        return {book: [{**self._hit(book, i, s, {"themes": [theme], "words": []}), "theme": theme,
                        "priority": None if p == 1000 else p} for p, s, i in rows[book]] for book in BOOKS}

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
        print('usage: python -m scripture.search "describe your situation"   # topics to tap (what the site shows)')
        print('       python -m scripture.search --topic grief [--sensitive]    # the verses of one topic')
        return
    s = Searcher.load()
    if argv[0] == "--topic":
        res = s.topic(argv[1], sensitive="--sensitive" in argv)
        print(f'{res["label"]}: {res["blurb"]}' + (f'\n{res["note"]}' if res["note"] else ""))
        for book in BOOKS:
            print(f"\n== {BOOK_NAMES[book]} ==")
            if res["books"][book]["nothing"]:
                print("  " + res["books"][book]["nothing"])
            for h in res["books"][book]["verses"]:
                print(f'{h["ref"]:>18}  #{h["priority"] or "-"}  {h["text"]}')
                print(f'{"":>20}why: {h["why"]}')
        return
    res = s.suggest(argv[0])
    print(f'category: {res["category"]}')
    if res["note"]:
        print(res["note"])
    for t in res["topics"]:
        print(f'  {t["id"]:<24} {t["label"]}: {t["blurb"]}')
    if res["category"] == "none":
        print("No topic fits these words; the site shows the full list of topics.")


if __name__ == "__main__":
    main(sys.argv[1:])
