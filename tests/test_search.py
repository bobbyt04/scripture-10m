"""Tests for the advice search (scripture/search.py), its build (search/) and its benchmark (evals/search_eval.py).

Unit tests use a tiny hand-built pool where every number can be worked out by
hand. Integration tests use the committed artefacts in search/artifacts/.
"""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from evals import search_eval as E
from scripture.search import (ARTIFACTS, BOOKS, Lexicon, Searcher, bm25_index, compile_phrase, one_edit_apart,
                              phrase_starts, round4, unit, words)
from search import apply_review as AR
from search import build
from search import similarities as SIM
from search.corpus import Corpus, read_u16

ROOT = Path(__file__).resolve().parent.parent
torch.set_num_threads(2)


@pytest.fixture(scope="module")
def s():
    return Searcher.load()


@pytest.fixture(scope="module")
def corpus():
    return Corpus.load()


@pytest.fixture(scope="module")
def texts():
    return AR.load_texts()


# ----- a tiny pool -----

def _verse(ref, text, section="S"):
    ch, v = ref.split(":")
    return {"ref": ref, "chapter": int(ch), "verse": int(v), "section": section, "text": text}


TINY = {
    "gita": [_verse("1:1", "The soul is eternal; the soul is never slain."),
             _verse("1:2", "Do thy duty without attachment to fruit."),
             _verse("1:3", "The wise grieve not for the dead.")],
    "quran": [_verse("1:1", "Be patient, for with hardship cometh ease."),
              _verse("1:2", "The soul shall taste of death."),
              _verse("1:3", "Give alms to the poor and the orphan."),
              _verse("2:1", "Patience and prayer, patience and prayer.")],
    "bible": [_verse("1:1", "A Psalm. Take no thought for the morrow, for the morrow is weary;"),
              _verse("1:2", "Blessed are the poor in spirit."),
              _verse("2:1", "The LORD is my shepherd; I shall not want.")],
}
TINY_LEXICON = {
    "stopwords": ["the", "is", "a", "and", "for", "of", "to", "in", "be", "do", "thy", "not", "no", "my", "i", "shall",
                  "with", "cometh", "are", "about", "tell", "take", "want", "never", "you", "your"],
    "query_stopwords": ["well"], "irregular": {"wives": "wife"},
    "suffix_rules": build.load_json(build.LEXICON)["suffix_rules"], "not_inflected": ["evening"],
}
TINY_THEMES = {
    "_about": "test",
    "grief": {"name": "Grief", "description": "d", "note": "Grief has no timetable.", "phrases": ["passed away", "grief", "died"]},
    "patience": {"name": "Patience", "description": "d", "phrases": ["waiting", "sick of waiting", "impatient"]},
    "money_worry": {"name": "Money", "description": "d", "phrases": ["laid off", "money", "rent"]},
    "work_effort": {"name": "Work", "description": "d", "phrases": ["my duty", "work"]},
    "temptation_self_control": {"name": "Temptation", "description": "d", "phrases": ["can not stop"]},
}
TINY_POOL = {
    "gita": [{"ref": "1:2", "themes": ["work_effort"], "why": "Do your duty.", "universal": True},
             {"ref": "1:3", "themes": ["grief"], "why": "Mourning.", "universal": True},
             {"ref": "1:1", "themes": ["grief", "work_effort"], "why": "The soul lives on.", "universal": False}],
    "quran": [{"ref": "1:1", "themes": ["patience", "money_worry"], "why": "Ease after hardship.", "universal": True},
              {"ref": "2:1", "themes": ["patience"], "why": "Patience and prayer.", "universal": True},
              {"ref": "1:3", "themes": ["money_worry"], "why": "Give to the poor.", "universal": True},
              {"ref": "1:2", "themes": ["patience", "money_worry"], "why": "Death comes.", "universal": True}],
    "bible": [{"ref": "1:1", "themes": ["money_worry", "patience"], "why": "Do not worry about tomorrow.",
               "universal": True, "trim": "A Psalm. "},
              {"ref": "2:1", "themes": ["money_worry"], "why": "Cared for like sheep.", "universal": True},
              {"ref": "1:2", "themes": ["money_worry"], "why": "The poor are blessed.", "universal": True}],
}
TINY_BRIDGE = {"_replace": ["job"], "stressed": ["weary", "patience", "nonexistentword"], "laid off": ["poor"],
               "job": ["duty"], "sick of waiting": ["patience"], "sick": ["death"], "tired": ["weary"]}
TINY_SAFETY = {
    "fuzzy": ["suicide", "myself"], "crisis_message": "Kind note.", "idioms": ["killing it", "cut back"],
    "bereavement": ["died by suicide", "lost my * to suicide"],
    "crisis": {"suicide": ["kill myself", "suicide", "want to die"], "self_harm": ["cut myself", "been cutting"]},
    "sensitive": {"identity": {"themes": ["patience"], "phrases": ["gay"]}},
    "distress": {"window": 3, "rules": ["pills|meds + how many, how much ! vitamins", "noose", "~6 he|she + hits + me"]},
}
TINY_SENSITIVE = {"gita": [{"ref": "1:2", "why": "x"}], "quran": [{"ref": "2:1", "why": "x"}], "bible": []}
TINY_PARAMS = {"min_theme_verses": 2}


def tiny_site(**kw):
    args = dict(pool=TINY_POOL, verses=TINY, lexicon=TINY_LEXICON, themes=TINY_THEMES, bridge=TINY_BRIDGE,
                safety=TINY_SAFETY, sensitive_set=TINY_SENSITIVE, params=TINY_PARAMS)
    args.update(kw)
    return build.build_site(**args)


@pytest.fixture(scope="module")
def tiny():
    return Searcher(tiny_site(), vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0],
                                          [3.0, 4.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


# ----- words and normalising -----

KNOWN = {"love", "walk", "give", "sin", "cry", "bless", "blessing", "die", "evening", "even",
         "forest", "for", "naked", "seed", "see", "rich", "hate", "horse", "tell", "tel"}
LEX = Lexicon({"stopwords": ["even", "for", "tell"], "query_stopwords": [], "irregular": {"art": "are"},
               "suffix_rules": build.load_json(build.LEXICON)["suffix_rules"], "not_inflected": ["evening", "forest"]})


@pytest.mark.parametrize("word, base", [
    ("loveth", "love"), ("loved", "love"), ("loving", "love"), ("loves", "love"), ("lovest", "love"),
    ("walketh", "walk"), ("walked", "walk"), ("giveth", "give"), ("sinned", "sin"), ("sinneth", "sin"),
    ("sinning", "sin"), ("crieth", "cry"), ("cried", "cry"), ("cries", "cry"), ("blessings", "bless"),
    ("dying", "die"), ("died", "die"), ("riches", "rich"), ("hateth", "hate"), ("horses", "horse"),
])
def test_normalise_reduces_archaic_and_inflected_forms(word, base):
    assert LEX.normalise(word, KNOWN) == base


@pytest.mark.parametrize("word", ["evening", "forest", "naked", "seed", "love", "unknownword"])
def test_normalise_leaves_words_that_only_look_inflected(word):
    assert LEX.normalise(word, KNOWN) == word


def test_normalise_stops_at_a_stopword_and_applies_irregulars_first():
    assert LEX.normalise("telling", KNOWN) == "tell" and LEX.normalise("telleth", KNOWN) == "tell"
    assert LEX.normalise("art", set()) == "are"
    assert LEX.normalise("lies", {"ly", "lie"}) == "lie"         # candidates under 3 letters are ignored
    lex = Lexicon({"stopwords": ["blessing"], "query_stopwords": [], "irregular": {},
                   "suffix_rules": build.load_json(build.LEXICON)["suffix_rules"], "not_inflected": []})
    assert lex.normalise("blessings", {"bless", "blessing"}) == "blessing"


def test_words_spells_out_contractions_with_or_without_apostrophes():
    assert words("I can't sleep, I'm scared & he won't call") == \
        ["i", "can", "not", "sleep", "i", "scared", "he", "will", "not", "call"]
    assert words("Don’t") == ["do", "not"]
    assert words("im ok, dont worry, i cant") == ["i", "ok", "do", "not", "worry", "i", "can", "not"]
    assert words("my friend's dog") == ["my", "friend", "dog"]
    assert words("as he was wont, ill") == ["as", "he", "was", "wont", "ill"]     # real words stay
    assert words("70 hour weeks") == ["hour", "weeks"]
    assert words("") == [] and words("!!! ...") == []


def test_phrases_slots_prefixes_and_wildcards():
    toks = words("my husband hit me last night")
    assert phrase_starts(toks, compile_phrase("husband|wife hit|hits me")) == [1]
    assert phrase_starts(toks, compile_phrase("wife hit me")) == []
    assert phrase_starts(words("I feel suicidal"), compile_phrase("suicid*")) == [2]
    assert phrase_starts(words("I feel sui"), compile_phrase("suicid*")) == []
    assert phrase_starts(words("punch my boss in the face"), compile_phrase("punch * * in the face")) == [0]


@pytest.mark.parametrize("a, b, ok", [
    ("suicide", "sucide", True), ("suicide", "suicidee", True), ("suicide", "suicdie", True), ("suicide", "suixide", True),
    ("suicide", "suicide", False), ("suicide", "sucidi", False), ("myself", "mysefl", True), ("myself", "himself", False),
])
def test_one_edit_apart(a, b, ok):
    assert one_edit_apart(a, b) is ok and one_edit_apart(b, a) is ok


def test_round4_is_half_away_from_zero():
    assert round4(0.00005) == 0.0001 and round4(-0.00005) == -0.0001 and round4(1.23444) == 1.2344
    assert round4(2.00015) == 2.0002 or abs(round4(2.00015) - 2.0001) < 1e-12   # float edge; JS gives the same
    assert unit([0.0, 0.0]) == [0.0, 0.0] and unit([3.0, 4.0]) == [0.6, 0.8]


# ----- building the site model -----

def test_display_text_trims_fixes_and_tidies():
    v = {"trim": "And ", "trim_end": " Selah.", "fixes": [["he world", "the world"]]}
    assert build.display_text("And look at he world, my soul; Selah.", v) == "Look at the world, my soul."
    assert build.display_text("Fear not;", {}) == "Fear not."
    with pytest.raises(ValueError):
        build.display_text("Fear not", {"trim": "And "})
    with pytest.raises(ValueError):
        build.display_text("Fear not", {"fixes": [["xyz", "abc"]]})


def test_site_model_from_the_tiny_pool(tiny):
    m = tiny_site()
    bible = {v["ref"]: v for v in m["pool"]["bible"]}
    assert bible["1:1"]["text"] == "Take no thought for the morrow, for the morrow is weary."
    assert "psalm" not in bible["1:1"]["terms"]                  # the trimmed heading is not searched
    # gita offers work_effort (2 verses) and grief (2 verses); themes with fewer than 2 verses are dropped
    assert {v["ref"]: v["themes"] for v in m["pool"]["gita"]} == {"1:2": ["work_effort"], "1:3": ["grief"],
                                                                 "1:1": ["grief", "work_effort"]}
    assert all(v["themes"] == ["money_worry"] for v in m["pool"]["bible"] if v["ref"] != "1:1")
    assert bible["1:1"]["themes"] == ["money_worry"]             # bible has one patience verse: not offered
    assert [v["sensitive"] for v in m["pool"]["quran"]] == [False, True, False, False]
    assert m["bridge"]["stressed"] == ["weary", "patience"]      # a target that is not a pool word is dropped
    assert m["bridge_replace"] == ["job"]
    assert set(m["vocabulary"]["patience"]) and m["vocabulary"]["the"] == ""


def test_build_site_rejects_a_sensitive_verse_outside_the_pool():
    with pytest.raises(ValueError):
        tiny_site(sensitive_set={**TINY_SENSITIVE, "bible": [{"ref": "9:9", "why": "x"}]})


def test_verse_with_no_offered_theme_is_not_published():
    pool = {**TINY_POOL, "bible": TINY_POOL["bible"] + [{"ref": "1:2", "themes": ["grief"], "why": "x", "universal": True}]}
    pool["bible"] = [v for v in pool["bible"] if v["themes"] != ["money_worry"] or v["ref"] != "1:2"]
    m = tiny_site(pool=pool)
    assert "1:2" not in {v["ref"] for v in m["pool"]["bible"]}


def test_document_fields_for_the_ablation():
    text_only = tiny_site(fields=("text",))
    full = tiny_site()
    g = {v["ref"]: v["terms"] for v in text_only["pool"]["gita"]}
    f = {v["ref"]: v["terms"] for v in full["pool"]["gita"]}
    assert "work" not in g["1:2"] and "work" in f["1:2"]          # from the theme name "Work"
    assert "duty" in g["1:2"]


# ----- understanding the query -----

def test_query_themes_longest_match_and_scores(tiny):
    assert tiny.query_themes("my grandma passed away") == {"grief": 1.0}
    assert tiny.query_themes("sick of waiting for money") == {"patience": 1.0, "money_worry": 1 / 3}
    assert tiny.query_themes("xyzzy") == {}
    s2 = Searcher(tiny_site(themes={**TINY_THEMES, "grief": {**TINY_THEMES["grief"], "phrases": ["can not stop crying"]}}))
    assert s2.query_themes("I can't stop crying") == {"grief": 1.0}    # hides "can not stop" (temptation)


def test_expand_bridge_replace_and_weights(tiny):
    q = tiny.expand("weary and stressed")
    assert q[tiny.term_of("weary")] == (1.0, "weary")             # typed first (1.0), then bridged (0.8): keeps 1.0
    assert q[tiny.term_of("patience")] == (tiny.params["bridge_weight"], "stressed")
    assert {src for _, src in tiny.expand("I got laid off").values()} == {"laid off"}
    q = tiny.expand("sick of waiting")
    assert tiny.term_of("patience") in q and tiny.term_of("death") not in q
    assert set(tiny.expand("my job")) == {tiny.term_of("duty")}
    assert tiny.expand("my job", bridge=False) == {}
    assert tiny.expand("stresseds") == {}                          # bridge keys match exactly
    assert tiny.expand("the and of") == {}
    replaced = Searcher(tiny_site(bridge={"_replace": ["patience"], "patience": ["prayer"]}))
    assert set(replaced.expand("patience")) == {replaced.term_of("prayer")}       # the pool word itself is skipped
    kept = Searcher(tiny_site(bridge={"patience": ["prayer"]}))
    assert set(kept.expand("patience")) == {kept.term_of("prayer"), kept.term_of("patience")}


def test_classify_crisis_sensitive_idioms_and_typos(tiny):
    assert tiny.classify("I want to kill myself") == ("crisis", "suicide")
    assert tiny.classify("i want to kill mysefl") == ("crisis", "suicide")     # fuzzy word
    assert tiny.classify("thinking about sucide") == ("crisis", "suicide")
    assert tiny.classify("I've been cutting again") == ("crisis", "self_harm")
    assert tiny.classify("I need to cut back") == ("general", None)            # idiom blanked
    assert tiny.classify("I'm gay") == ("sensitive", "identity")
    assert tiny.classify("I'm gay and I want to die") == ("crisis", "suicide")
    assert tiny.classify("killing it") == ("general", None)


# ----- scoring -----

def test_bm25_matches_the_formula(tiny):
    b = tiny.books["quran"]
    k1, bb = tiny.k1, tiny.b
    i = b["index"]["2:1"]
    t = tiny.term_of("patience")
    n = len(b["verses"])
    df = sum(t in tf for tf in b["tf"])
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    tf = b["tf"][i][t]
    want = idf * tf * (k1 + 1) / (tf + k1 * (1 - bb + bb * b["length"][i] / b["avgdl"]))
    assert tiny.bm25("quran", i, {t: (1.0, "x")}) == pytest.approx(want)
    assert tiny.bm25("quran", i, {t: (0.5, "x")}) == pytest.approx(0.5 * want)
    assert bm25_index([["a", "b", "a"], ["b"]])["idf"]["a"] == pytest.approx(math.log(1 + 1.5 / 1.5))


def test_rank_key_orders_by_theme_then_priority_then_themes_then_bm25(tiny):
    themes = {"patience": 1.0, "money_worry": 0.5}
    q = tiny.expand("patience")
    i = tiny.books["quran"]["index"]["1:1"]
    key, best, prio, hit, terms = tiny.rank_key("quran", i, themes, q)
    assert hit == ["patience", "money_worry"] and best == "patience" and prio is None
    assert key == (0, 1000, -1.5, -round4(tiny.bm25("quran", i, q)), i)
    assert tiny.rank_key("quran", i, {"grief": 1.0}, q) is None                 # off-topic: never shown


def test_search_returns_only_verses_of_the_query_themes(tiny):
    r = tiny.search("sick of waiting", k=5)
    assert r["category"] == "general" and r["themes"] == [{"id": "patience", "score": 1.0}] and r["message"] is None
    assert [h["ref"] for h in r["results"]["quran"]] == ["2:1", "1:2", "1:1"]      # 2:1 has the word twice; 1:2 is shorter than 1:1
    assert r["results"]["gita"] == [] and r["results"]["bible"] == []             # no patience verses there
    h = r["results"]["quran"][0]
    assert h["matched"] == {"themes": ["patience"], "words": [{"word": "patience", "from": "sick of waiting"}]}
    assert h["why"] == "Patience and prayer." and h["text"].startswith("Patience")
    assert h["theme"] == "patience" and h["priority"] is None
    assert [x["score"] for x in r["results"]["quran"]] == sorted((x["score"] for x in r["results"]["quran"]), reverse=True)


def test_hand_set_priority_beats_words_and_the_top_theme_comes_first():
    pool = {**TINY_POOL, "quran": [dict(v) for v in TINY_POOL["quran"]]}
    pool["quran"][3]["priority"] = {"patience": 1}                               # 1:2, which has no matching word
    s = Searcher(tiny_site(pool=pool))
    r = s.search("sick of waiting", k=5)
    assert [h["ref"] for h in r["results"]["quran"]] == ["1:2", "2:1", "1:1"]
    assert r["results"]["quran"][0]["priority"] == 1
    r = s.search("sick of waiting and money", k=5)                                # patience 1.0 (3 words), money 1/3
    assert [t["id"] for t in r["themes"]] == ["patience"]                         # below min_theme_score: dropped
    r = s.search("waiting for money", k=5)                                        # a tie: 1.0 each, id order
    assert [t["id"] for t in r["themes"]] == ["money_worry", "patience"]
    assert [h["theme"] for h in r["results"]["quran"]] == ["money_worry"] * 3 + ["patience"]


def test_search_notes_ties_and_no_match(tiny):
    r = tiny.search("my mother passed away", k=3)
    assert r["notes"] == [{"theme": "grief", "note": "Grief has no timetable."}]
    assert [h["ref"] for h in r["results"]["gita"]] == ["1:3", "1:1"]             # equal keys: pool order
    for q in ["", "!!!", "xyzzy qwerty", "the and of", "shepherd"]:               # no theme: no verses, even if words match
        r = tiny.search(q)
        assert r["no_match"] and r["themes"] == [] and all(v == [] for v in r["results"].values())


def test_grief_is_exclusive(tiny):
    r = tiny.search("my rent money is gone since my mother passed away", k=5)
    assert r["themes"] == [{"id": "grief", "score": 1.0}]
    assert r["results"]["quran"] == [] and r["results"]["bible"] == []            # no grief verses there: nothing else instead


def test_theme_typos_and_the_500_character_cap(tiny):
    assert tiny.query_themes("so impatinet") == {"patience": 1.0}                 # one swap from a theme word (6+ letters)
    assert tiny.query_themes("my duty") == {"work_effort": 1.0}
    assert tiny.theme_tokens("griefs")[0] == "griefs"                              # a pool word's form is not a typo
    long_q = "x " * 300 + "I want to die"
    assert tiny.classify(long_q) == ("general", None)                              # beyond 500 characters: not read
    assert tiny.search(long_q)["no_match"]


def test_distress_gate_rules_window_vetoes_and_bereavement(tiny):
    assert tiny.classify("how many pills is it") == ("crisis", "distress")         # any order inside the window
    assert tiny.classify("pills, how many") == ("crisis", "distress")
    assert tiny.classify("how many pills of vitamins") == ("general", None)        # vetoed
    assert tiny.classify("how many days until i refill my pills") == ("general", None)   # too far apart (window 3)
    assert tiny.classify("he always hits me") == ("crisis", "distress")            # own window of 6
    assert tiny.classify("I bought a noose") == ("crisis", "distress")
    assert tiny.classify("my brother died by suicide") == ("general", None)        # bereavement blanked
    assert tiny.classify("lost my uncle to suicide last year") == ("general", None)
    assert tiny.classify("my brother died by suicide and I think about suicide") == ("crisis", "suicide")
    r = tiny.search("I bought a noose")
    assert r["message"] == "Kind note." and r["notes"] == [] and not r["no_match"]


def test_search_crisis_and_sensitive(tiny):
    r = tiny.search("I want to die")
    assert r["category"] == "crisis" and r["themes"] == [] and all(v == [] for v in r["results"].values())
    assert r["message"] == "Kind note." and r["no_match"] is False
    r = tiny.search("I'm gay", k=5)
    assert r["category"] == "sensitive" and r["themes"] == [{"id": "patience", "score": 0.5}]
    assert [h["ref"] for h in r["results"]["quran"]] == ["2:1"]                   # only the sensitive set
    assert r["results"]["gita"] == []                                             # its set verse has no patience theme


def test_search_rejects_bad_k(tiny):
    for k in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            tiny.search("sad", k=k)
        with pytest.raises(ValueError):
            tiny.by_theme("grief", k=k)


def test_by_theme_themes_and_similar(tiny):
    r = tiny.by_theme("grief", k=3)
    assert [(h["ref"], h["score"]) for h in r["gita"]] == [("1:3", 1.0), ("1:1", 1.0)]
    assert [(h["ref"], h["score"]) for h in tiny.by_theme("work_effort")["gita"]] == [("1:2", 1.0), ("1:1", 0.5)]
    pool = {**TINY_POOL, "gita": [dict(v) for v in TINY_POOL["gita"]]}
    pool["gita"][2]["priority"] = {"work_effort": 1}
    r = Searcher(tiny_site(pool=pool)).by_theme("work_effort")["gita"]
    assert [(h["ref"], h["priority"]) for h in r] == [("1:1", 1), ("1:2", None)]   # a hand-set priority comes first
    with pytest.raises(ValueError):
        tiny.by_theme("nope")
    t = {x["id"]: x for x in tiny.themes()}
    assert t["patience"]["verses"] == {"gita": 0, "quran": 3, "bible": 0}
    sim = tiny.similar("gita", "1:2", k=3)
    assert "1:2" not in [h["ref"] for h in sim["gita"]]
    assert [(h["ref"], h["score"]) for h in sim["gita"]] == [("1:1", 0.7071), ("1:3", 0.0)]
    assert sim["quran"][-1]["score"] == 0.0                       # the all-zero vector scores 0, not NaN
    assert [h["ref"] for h in tiny.similar("gita", "1:2", sensitive_only=True)["quran"]] == ["2:1"]
    for args in [("gita", "9:9"), ("torah", "1:1")]:
        with pytest.raises(ValueError):
            tiny.similar(*args)


# ----- applying a review -----

def test_apply_review_actions(texts):
    pool = {"gita": [{"ref": "2:47", "themes": ["work_effort", "purpose"], "why": "a", "universal": True},
                     {"ref": "2:48", "themes": ["work_effort"], "why": "b", "universal": True},
                     {"ref": "2:14", "themes": ["patience"], "why": "c", "universal": True}],
            "quran": [{"ref": "94:5", "themes": ["hope"], "why": "d", "universal": True}], "bible": []}
    review = {"gita": [{"ref": "2:47", "action": "retag", "drop_themes": ["purpose"], "reason": "r"},
                       {"ref": "2:48", "action": "retag", "drop_themes": ["work_effort"], "reason": "r"},
                       {"ref": "2:14", "action": "not_universal", "reason": "r"},
                       {"ref": "2:14", "action": "fix_why", "new_why": "new", "reason": "r"},
                       {"ref": "9:99", "action": "remove", "reason": "r"}],
              "quran": [{"ref": "94:5", "action": "display", "trim": "But lo! ", "reason": "r"},
                        {"ref": "94:5", "action": "remove", "reason": "r"}]}
    applied, skipped = AR.apply(review, pool, texts)
    assert pool["gita"] == [{"ref": "2:47", "themes": ["work_effort"], "why": "a", "universal": True},
                            {"ref": "2:14", "themes": ["patience"], "why": "new", "universal": False}]
    assert pool["quran"] == []
    assert len(applied) == 6 and [(b, i["ref"]) for b, i, _ in skipped] == [("gita", "9:99")]
    with pytest.raises(ValueError):
        AR.apply({"bible": [{"ref": "Psalms 23:1", "action": "display", "trim": "Not the start"}]},
                 {"gita": [], "quran": [], "bible": [{"ref": "Psalms 23:1", "themes": ["hope"], "why": "", "universal": True}]},
                 texts)
    with pytest.raises(ValueError):
        AR.apply({"gita": [{"ref": "2:47", "action": "explode"}]}, {"gita": [{"ref": "2:47"}], "quran": [], "bible": []}, texts)


def test_apply_review_empty_trim_and_the_sensitive_set(texts):
    pool = {"gita": [{"ref": "2:47", "themes": ["work_effort", "hope"], "why": "a", "universal": True}],
            "quran": [{"ref": "94:5", "themes": ["hope"], "why": "d", "universal": True, "trim": "Lo! "}], "bible": []}
    AR.apply({"quran": [{"ref": "94:5", "action": "display", "trim": "", "reason": "r"}]}, pool, texts)
    assert "trim" not in pool["quran"][0]                         # an empty value removes the field
    sens = {"gita": [{"ref": "2:47", "why": "x"}], "quran": [{"ref": "94:5", "why": "y"}, {"ref": "9:9", "why": "z"}],
            "bible": []}
    review = {"sensitive_set": [{"ref": "quran 94:5", "action": "remove", "reason": "r"},
                                {"ref": "gita 2:47", "action": "retag", "drop_themes": ["hope"], "reason": "r"},
                                {"ref": "bible Psalms 23:1", "action": "remove", "reason": "r"}]}
    applied, skipped = AR.apply_sensitive(review, pool, sens)
    assert sens == {"gita": [{"ref": "2:47", "why": "x"}], "quran": [], "bible": []}   # 9:9 is not in the pool: dropped too
    assert pool["gita"][0]["themes"] == ["work_effort"] and len(applied) == 2 and len(skipped) == 1


# ----- the committed pool and artefacts -----

def test_pool_files_are_well_formed(texts):
    themes = {k for k in build.load_json(build.THEMES) if not k.startswith("_")}
    assert "mortality" not in themes and "laziness_discipline" not in themes and len(themes) == 38
    for book in BOOKS:
        pool = build.load_pool()[book]
        assert len({v["ref"] for v in pool}) == len(pool) >= 70          # Gita 72 after the judge-5 sweep
        for v in pool:
            assert v["ref"] in texts[book] and v["why"].strip() and v["themes"]
            assert set(v["themes"]) <= themes, (book, v["ref"])
            build.display_text(texts[book][v["ref"]], v)


def test_sensitive_set_follows_its_rules():
    sens = build.load_json(build.SENSITIVE_SET)
    pools = build.load_pool()
    banned = {"guilt_repentance", "temptation_self_control", "forgiveness", "family_parents", "marriage", "illness"}
    for book in BOOKS:
        rows = {v["ref"]: v for v in pools[book]}
        assert len(sens[book]) >= (3 if book == "gita" else 15) and len(sens[book]) <= 25
        for x in sens[book]:
            v = rows[x["ref"]]
            assert v["universal"] and not set(v["themes"]) & banned and x["why"], (book, x["ref"])


def test_artefacts_match_their_inputs_and_are_small(s):
    model = json.loads((ARTIFACTS / "search.json").read_text())
    inputs = model["inputs"]
    for name, path in (("lexicon", build.LEXICON), ("bridge", build.BRIDGE), ("safety", build.SAFETY),
                       ("themes", build.THEMES), ("sensitive_set", build.SENSITIVE_SET)):
        h = hashlib.sha256(json.dumps(build.load_json(path), sort_keys=True).encode()).hexdigest()
        assert inputs[f"{name}_sha256"] == h, f"{path.name} changed: rerun python search/build.py --site"
    assert inputs["pool_sha256"] == build.sha256(build.load_pool()), "pool changed: rerun python search/build.py --site"
    assert inputs["corpus_sha256"] == build.corpus_sha256(build.load_verses())
    n = sum(len(model["pool"][b]) for b in BOOKS)
    assert (ARTIFACTS / "pool_vectors.i8").stat().st_size == n * model["dim"]
    assert sum(f.stat().st_size for f in ARTIFACTS.iterdir()) < 1.5e6
    offered = build.offered_themes(build.load_pool(), model["params"]["min_theme_verses"])
    for book in BOOKS:
        for v in model["pool"][book]:
            assert v["themes"] and set(v["themes"]) <= offered[book]
    assert [(k, t) for k, ts in model["bridge"].items() for t in ts if s.term_of(t) is None] == []


def test_offered_themes_follow_the_three_verse_rule(s):
    t = {x["id"]: x["verses"] for x in s.themes()}
    for theme, counts in t.items():
        for book, c in counts.items():
            assert c == 0 or c >= 3, (theme, book, c)
    assert t["grief"]["gita"] == 0 and t["marriage"]["quran"] == 0


def test_corpus_model_matches_its_inputs(corpus):
    model = json.loads((build.MODEL / "model.json").read_text())
    assert model["inputs"]["corpus_sha256"] == build.corpus_sha256(build.load_verses())
    h = hashlib.sha256(json.dumps(build.load_json(build.LEXICON), sort_keys=True).encode()).hexdigest()
    assert model["inputs"]["lexicon_sha256"] == h
    n = sum(len(model["books"][b]["refs"]) for b in BOOKS)
    assert n == 701 + 6236 + 31102
    assert (build.MODEL / "verse_vectors.i8").stat().st_size == n * model["dim"]
    assert (build.MODEL / "verse_offsets.u32").stat().st_size == 4 * (n + 1)


def test_fixtures_match_the_committed_artefacts(s):
    fx = json.loads((ARTIFACTS / "fixtures.json").read_text())
    assert len(fx["search"]) >= 85
    for text, toks in fx["words"]:
        assert words(text) == toks
    for c in fx["search"]:
        r = s.search(c["query"], k=c["k"])
        assert (r["category"], r["topic"], r["message"], r["no_match"]) == \
            (c["category"], c["topic"], c["message"], c["no_match"]), c["query"]
        assert s.crisis_words(c["query"]) == c["crisis_words"] and s.theme_tokens(c["query"]) == c["theme_words"]
        assert [[t["id"], t["score"]] for t in r["themes"]] == c["themes"]
        assert [n["theme"] for n in r["notes"]] == c["notes"]
        assert [[t, w, src] for t, (w, src) in s.expand(c["query"]).items()] == c["terms"]
        for book in BOOKS:
            got = [[h["ref"], h["theme"], h["priority"], h["score"], h["matched"]["themes"],
                    [m["word"] for m in h["matched"]["words"]]] for h in r["results"][book]]
            assert got == c["results"][book], (c["query"], book)
    for c in fx["by_theme"]:
        assert {b: [[h["ref"], h["score"], h["priority"]] for h in v] for b, v in s.by_theme(c["theme"]).items()} == c["results"]
    for c in fx["similar"]:
        assert {b: [[h["ref"], h["score"]] for h in v] for b, v in s.similar(c["book"], c["ref"]).items()} == c["results"]
    for c in fx["similar_sensitive_only"]:
        got = s.similar(c["book"], c["ref"], sensitive_only=True)
        assert {b: [[h["ref"], h["score"]] for h in v] for b, v in got.items()} == c["results"]
    cats = {c["category"] for c in fx["search"]}
    assert cats == {"general", "sensitive", "crisis"} and any(c["no_match"] for c in fx["search"])
    assert {c["k"] for c in fx["search"]} > {3}                                  # k other than 3 is covered
    assert any(c["topic"] == "distress" for c in fx["search"])                    # the distress gate is covered
    assert any(c["notes"] for c in fx["search"])
    assert any(w != f for c in fx["search"] for w, f in zip(c["words"], c["theme_words"]))   # a fuzzy theme word
    assert any(not c["query"].isascii() for c in fx["search"])


def test_display_text_never_shows_psalm_headings(s):
    for v in s.books["bible"]["verses"]:
        assert not v["text"].startswith(("A Psalm", "To the chief Musician", "A Song")), v["ref"]
        assert not v["text"].endswith("Selah."), v["ref"]


@pytest.mark.parametrize("query, book, expected", [
    ("I'm drowning in debt", "bible", {"Philippians 4:19", "Psalms 23:1"}),
    ("so anxious about everything", "bible", {"Isaiah 41:10", "1 Peter 5:7"}),
    ("my mother passed away", "quran", {"12:86", "13:28"}),
    ("I keep procrastinating", "gita", {"2:47", "3:8"}),
])
def test_sanity_queries(s, query, book, expected):
    assert expected & {h["ref"] for h in s.search(query, k=3)["results"][book]}


# ----- the full-corpus model and similarities -----

def test_corpus_similar(corpus):
    for args in [("torah", "1:1"), ("gita", "99:99")]:
        with pytest.raises(ValueError):
            corpus.similar(*args)
    with pytest.raises(ValueError):
        corpus.similar("gita", "2:47", k=0)
    res = corpus.similar("bible", "Genesis 37:9", k=7)
    assert all(len(v) == 7 for v in res.values())
    assert "Genesis 37:9" not in [h["ref"] for h in res["bible"]]
    assert "12:4" in [h["ref"] for h in corpus.similar("bible", "Genesis 37:9", k=10, books=["quran"])["quran"]]
    lam = corpus.params["similar_vector_weight"]
    a, b = corpus.books["bible"], corpus.books["quran"]
    i, j = a["index"]["Genesis 37:9"], b["index"]["12:4"]
    want = lam * (a["vectors"][i] @ b["vectors"][j]).item() + (1 - lam) * corpus.tfidf_scores(corpus.tfidf(a["terms"][i]), b)[j].item()
    assert corpus.pair_score("bible Genesis 37:9", "quran 12:4") == round(want, 3)


def test_similarities_are_reviewed_and_well_formed(texts):
    sims = json.loads((ARTIFACTS / "similarities.json").read_text())
    review = json.loads(SIM.REVIEW.read_text())
    assert sims["kept"] == sum(c["keep"] for c in review["candidates"])
    pairs = [p for t in sims["themes"] for p in t["pairs"]]
    assert 40 <= len(pairs) <= 120
    seen = []
    for p in pairs:
        assert p["idea"] and p["source"] in ("model, reviewed", "hand-listed")
        a, b = p["verses"]
        assert a["book"] != b["book"]
        for v in (a, b):
            assert texts[v["book"]][v["ref"]] == v["text"]
            seen.append((v["book"], v["ref"]))
    assert max(seen.count(x) for x in seen) == 1                  # each verse in at most one pair
    removed = {("bible", "Mark 4:31"), ("quran", "7:40"), ("bible", "Genesis 22:2"), ("quran", "3:59"),
               ("quran", "42:50"), ("bible", "Colossians 1:16"), ("bible", "Revelation 1:8"), ("quran", "57:3")}
    assert not removed & set(seen)
    ideas = " ".join(p["idea"] for p in pairs)
    assert "two women at Midian" not in ideas and "not sacrifice" not in ideas
    # round 5: wrong pairs, threatening lines and Krishna's "Me" rendered as "God" are out; scores are the review's
    gone = {("bible", "Mark 4:39"), ("quran", "22:32"), ("bible", "1 John 4:4"), ("bible", "Ephesians 2:19"),
            ("quran", "11:37"), ("quran", "44:24"), ("quran", "20:39"), ("gita", "15:12"), ("gita", "9:34"), ("gita", "9:26"),
            ("gita", "9:27"), ("gita", "14:4"), ("gita", "15:6"), ("gita", "9:31")}
    assert not gone & set(seen)
    judge5 = {("bible", "Hebrews 9:27"), ("gita", "2:22"), ("bible", "John 11:13"), ("quran", "17:19"), ("gita", "2:38"),
              ("bible", "Matthew 5:44"), ("gita", "18:61")}                 # judge 5's seven pairs are left out
    assert not judge5 & set(seen)
    for p in pairs:
        if {v["book"] for v in p["verses"]} == {"bible", "quran"} and "Pharaoh" in p["idea"]:
            assert any("Pharaoh" in v["text"] for v in p["verses"] if v["book"] == "quran"), p["idea"]
    scores = {frozenset((c["a"], c["b"])): c["score"] for c in review["candidates"]}
    par = {p["id"]: p for p in json.loads(E.PARALLELS.read_text())["parallels"]}
    for pid, d in review["hand_listed"].items():
        p = par[pid]
        if d["keep"]:
            scores[frozenset((f'{p["a"]["book"]} {p["a"]["refs"][0]}', f'{p["b"]["book"]} {p["b"]["refs"][0]}'))] = d["score"]
    for p in pairs:
        a, b = p["verses"]
        assert p["score"] == scores[frozenset((f'{a["book"]} {a["ref"]}', f'{b["book"]} {b["ref"]}'))]


def test_read_u16_unwraps_large_ids(tmp_path):
    from array import array
    (tmp_path / "x.u16").write_bytes(array("H", [1, 32767, 32768, 40000, 65535]).tobytes())
    assert read_u16(tmp_path / "x.u16").tolist() == [1, 32767, 32768, 40000, 65535]


# ----- corpus build pieces -----

def test_stopwords_are_derived_from_frequency_plus_hand_lists():
    cfg = {"top_by_verse_share": 3, "keep": ["for"], "add": ["extra"]}
    assert build.derive_stopwords(TINY, cfg) == ["extra", "is", "the"]


def test_proper_names_are_always_capitalised_words():
    corpus = {"gita": [_verse("1:1", "Judas went. The judge came.")], "quran": [_verse("1:1", "The Judge")],
              "bible": [_verse("1:1", "Annas and the high priest")]}
    names = build.proper_names(corpus)
    assert {"judas", "annas"} <= names and "judge" not in names and "the" not in names


def test_cooccurrence_counts_window_and_book_weight():
    streams = [("gita", torch.tensor([0, 1, 2])), ("bible", torch.tensor([0, 2]))]
    c = build.cooccurrence(streams, {"gita": 2.0, "bible": 1.0}, 3, window=2)
    assert c[0, 1] == 2.0 and c[1, 2] == 2.0 and c[0, 2] == 2.0 + 1.0
    assert torch.equal(c, c.t())
    assert build.cooccurrence(streams, {"gita": 2.0, "bible": 1.0}, 3, window=1)[0, 2] == 1.0


def test_ppmi_formula_with_context_smoothing():
    c = torch.tensor([[0., 4., 1.], [4., 0., 2.], [1., 2., 0.]])
    got = build.ppmi(c, 0.75)
    row = c.sum(1, keepdim=True)
    col = c.sum(0, keepdim=True) ** 0.75
    expect = torch.log(c * col.sum() / (row * col)).nan_to_num(neginf=0).clamp(min=0)
    assert torch.allclose(got, expect, atol=1e-5) and got.min() >= 0
    assert not torch.allclose(got, build.ppmi(c, 1.0))


def test_book_weights_and_chapter_streams():
    vt = [torch.tensor([1, 2])] * 3 + [torch.tensor([1])] * 4 + [torch.tensor([1, 2, 3, 4])] * 3
    w = build.book_weights(TINY, vt, 0.5)
    assert w["bible"] == 1.0 and w["gita"] == pytest.approx((12 / 6) ** 0.5) and w["quran"] == pytest.approx((12 / 4) ** 0.5)
    streams = build.chapter_streams(TINY, [torch.tensor([i]) for i in range(10)], 100)
    assert [(b, x.tolist()) for b, x in streams] == [
        ("gita", [0, 1, 2]), ("quran", [3, 4, 5]), ("quran", [6]), ("bible", [7, 8]), ("bible", [9])]


def test_vocab_drops_words_whose_base_is_a_stopword_and_counts_min():
    corpus = {**TINY, "gita": TINY["gita"] + [_verse("1:4", "He was telling the truth; tell it.")]}
    lex = Lexicon({**TINY_LEXICON, "not_inflected": []})
    v = build.build_vocab(corpus, 1, lex)
    assert "telling" not in v["terms"] and "tell" not in v["terms"] and "truth" in v["terms"]
    v2 = build.build_vocab(corpus, 2, lex)
    assert v2["n_vec"] < v["n_vec"] == len(v["terms"]) and all(v2["count"][t] >= 2 for t in range(v2["n_vec"]))


def test_verse_vectors_are_sif_weighted_and_orthogonal_to_the_shared_direction():
    wv = torch.eye(3)
    counts = [10, 1, 1]
    vt = [torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 2]), torch.tensor([0])]
    vv, principal = build.verse_vectors(vt, wv, counts, 0.5)
    assert torch.allclose(vv @ principal, torch.zeros(4), atol=1e-5)
    p = torch.tensor(counts, dtype=torch.float32)
    sif = 0.5 / (0.5 + p / p.sum())
    raw = torch.stack([sif[t].unsqueeze(1).mul(wv[t]).sum(0) for t in vt])
    raw -= (raw @ principal).unsqueeze(1) * principal
    assert torch.allclose(vv, raw / raw.norm(dim=1, keepdim=True), atol=1e-5)


def test_quantise_is_per_row():
    q = build.quantise(torch.tensor([[0.001, -0.002, 0.0005], [10.0, 5.0, -2.0]]))
    assert q.abs().max(dim=1).values.tolist() == [127, 127]
    torch.manual_seed(0)
    r = torch.randn(50, 64)
    assert torch.nn.functional.cosine_similarity(r, build.quantise(r).float(), dim=1).min() > 0.999


def test_train_is_deterministic_and_writes_loadable_files(tmp_path):
    p = {"dim": 4, "min_count": 1, "window": 2}
    a, b = build.train(TINY, p, TINY_LEXICON_SRC), build.train(TINY, p, TINY_LEXICON_SRC)
    assert json.dumps(a[0], sort_keys=True) == json.dumps(b[0], sort_keys=True) and torch.equal(a[2], b[2])
    build.write_corpus(a[0], a[2], a[3], tmp_path)
    c = Corpus.load(tmp_path, data_dir=None)
    assert c.terms == a[0]["terms"] and len(c.books["quran"]["refs"]) == 4


TINY_LEXICON_SRC = {"stopwords": {"top_by_verse_share": 0, "keep": [], "add": TINY_LEXICON["stopwords"]},
                    "query_stopwords": ["well"], "irregular": {}, "suffix_rules": TINY_LEXICON["suffix_rules"],
                    "not_inflected": []}


# ----- eval maths -----

def test_case_rows_and_evaluate_on_the_tiny_pool(tiny):
    case = {"query": "sick of waiting", "expected_themes": ["patience"],
            "relevant": {"gita": ["1:2"], "quran": ["1:2", "9:9"], "bible": ["2:1"]}}
    rows = E.case_rows(tiny, case)
    assert [(r["book"], r["hit"], r["empty"], r["outside"]) for r in rows] == [
        ("gita", 0.0, True, 0), ("quran", 1.0, False, 1), ("bible", 0.0, True, 0)]
    miss = {**case, "query": "shepherd", "expected_themes": ["grief"], "relevant": {"gita": [], "quran": ["9:9"], "bible": ["2:1"]}}
    r = E.evaluate(tiny, [case, miss])
    assert r["quran"] == {"pairs": 1, "hit@3": 1.0}                # the shepherd case has no quran key in the pool
    assert r["bible"] == {"pairs": 2, "hit@3": 0.0}                # "shepherd" has no theme, so no verses
    assert r["overall"]["pairs"] == 4 and r["overall"]["hit@3"] == pytest.approx(1 / 4)
    assert r["theme_accuracy"] == 0.5
    assert r["empty_book_rate"] == pytest.approx(5 / 6)
    assert r["key_verses_outside_pool"] == 2 and r["key_verses_total"] == 6


def test_bootstrap_interval():
    per_case = [[1.0], [0.0], [1.0, 1.0]]
    ci = E.bootstrap(per_case, reps=500)
    assert ci[0] <= 0.75 <= ci[1] and ci[0] < ci[1]
    assert E.bootstrap([[1.0]] * 5) == [1.0, 1.0] and E.bootstrap([]) is None
    per_case = [[i / 10] for i in range(10)]                     # the 2.5th and 97.5th percentiles exactly
    rnd = __import__("random").Random(0)
    draws = sorted(E.mean([x for _ in per_case for x in rnd.choice(per_case)]) for _ in range(500))
    assert E.bootstrap(per_case, reps=500) == [round(draws[12], 3), round(draws[486], 3)]
    assert draws[486] < draws[-1]
    assert E.mean([]) is None


def test_ablation_variants_differ_only_in_their_parts():
    full, bm, bare = E.variant("+ hand-set priority (final)"), E.variant("+ BM25 tie break"), E.variant("themes, pool order")
    assert any("priority" in v for v in full.books["bible"]["verses"])
    assert not any("priority" in v for v in bm.books["bible"]["verses"] + bare.books["bible"]["verses"])
    assert all(v["terms"] == [] for v in bare.books["bible"]["verses"]) and bm.books["bible"]["verses"][0]["terms"]
    for s in (full, bm, bare):                                    # the same verses can be shown: only the order differs
        assert {h["ref"] for h in s.search("my mother passed away", k=500)["results"]["bible"]} == \
            {h["ref"] for h in full.search("my mother passed away", k=500)["results"]["bible"]}


def test_probe_report_counts(tiny):
    probes = {"x": {"crisis": [{"kind": "suicide", "text": "I want to die"}, {"kind": "suicide", "text": "so sad"}],
                    "innocent": ["killing it"], "sensitive": ["I'm gay"]}, "first_look": {}}
    assert E.probe_report(tiny, probes) == {"x": {"crisis": 2, "crisis_caught": 1, "innocent": 1,
                                                  "innocent_false_alarms": 0, "sensitive": 1, "sensitive_routed": 1}}


def test_evaluate_parallels_restores_settings(corpus):
    par = [{"id": "x", "kind": "story", "a": {"book": "bible", "refs": ["Genesis 37:9"]}, "b": {"book": "quran", "refs": ["12:4"]}}]
    before = corpus.params["similar_vector_weight"]
    r = E.evaluate_parallels(corpus, par, vector_weight=0.3, k=10)
    assert corpus.params["similar_vector_weight"] == before and r["trials"] == 2 and r["hit@10"] == 1.0


def test_eval_cases_are_labelled_and_point_at_real_verses(texts):
    doc = json.loads(E.CASES.read_text())
    cases = doc["cases"]
    themes = {k for k in build.load_json(build.THEMES) if not k.startswith("_")}
    assert {c["split"] for c in cases} == set(E.SPLITS) and set(doc["splits"]) == set(E.SPLITS)
    assert len(cases) == 190 and len({c["id"] for c in cases}) == 190
    assert sum(c["split"] == "blind_1" for c in cases) == 50
    assert sum(c["split"] == "final_blind" for c in cases) == 50    # judge 5's held-out set, judged once by hand
    assert doc["blind_1_hand_judged"]["precision_of_shown_verses"] == 0.373
    assert doc["final_blind_hand_judged"]["precision_of_shown_verses"] == 0.573
    for c in cases:
        assert c["expected_themes"] and set(c["expected_themes"]) <= themes, c["id"]
        for book in BOOKS:
            for r in c["relevant"][book]:
                assert r in texts[book], (c["id"], r)
    for p in json.loads(E.PARALLELS.read_text())["parallels"]:
        for side in ("a", "b"):
            for r in p[side]["refs"]:
                assert r in texts[p[side]["book"]], (p["id"], r)
