"""Tests for the meaning search (scripture/search.py, search/, evals/search_eval.py).

Unit tests use a tiny hand-built corpus where every number can be worked out
by hand. Integration tests use the committed artefacts in search/artifacts/.
"""

import hashlib
import itertools
import json
import math
from pathlib import Path

import pytest
import torch

from evals import search_eval as E
from scripture.search import ARTIFACTS, BOOKS, Lexicon, Searcher, compile_phrases, phrase_in, read_u16, unpack_bits, words
from search import build
from search import similarities as SIM

ROOT = Path(__file__).resolve().parent.parent
torch.set_num_threads(2)


# ----- fixtures -----

@pytest.fixture(scope="module")
def s():
    return Searcher.load()


@pytest.fixture(scope="module")
def refs():
    out = {}
    for book in BOOKS:
        with open(ROOT / "data" / "processed" / f"{book}.jsonl") as f:
            rows = [json.loads(line) for line in f]
        out[book] = {r["ref"]: r["text"] for r in rows}
    return out


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
    "bible": [_verse("1:1", "Take no thought for the morrow, for the morrow is weary."),
              _verse("1:2", "Blessed are the poor in spirit."),
              _verse("2:1", "The LORD is my shepherd; I shall not want.")],
}
TINY_LEXICON = {
    "stopwords": {"top_by_verse_share": 0, "keep": [], "add": [
        "the", "is", "a", "and", "for", "of", "to", "in", "be", "do", "thy", "not", "no", "my", "i", "shall",
        "with", "cometh", "are", "about", "tell", "take", "want", "never"]},
    "query_stopwords": ["well"],
    "irregular": {"art": "are", "hath": "have", "wives": "wife"},
    "suffix_rules": build.load_json(build.LEXICON)["suffix_rules"],
    "not_inflected": ["evening"],
}
TINY_BRIDGE = {"_replace": ["job"], "stressed": ["weary", "patience"], "laid off": ["poor"],
               "job": ["duty"], "sick of waiting": ["patience"], "sick": ["death"], "pay rent": ["poor"],
               "tired": ["weary"]}
TINY_SAFETY = {
    "crisis": {"suicide": ["kill myself", "suicid*"], "abuse": ["he|she hits|hit me"]},
    "sensitive": {"identity": ["gay"]},
    "gentle": {"gita": [["1:1", "eternal soul"]], "quran": [["2:1", "patience"], ["1:1", "ease"]],
               "bible": [["2:1", "shepherd"]]},
    "block": {"patterns": [{"name": "violence", "regex": "\\bslain\\b"}],
              "refs": [{"book": "bible", "from": "1:2", "to": "1:2", "reason": "test range"}],
              "keep": []},
}


def tiny_model(params=None, corpus=TINY, **kw):
    p = {"dim": 4, "min_count": 1, "window": 2, **(params or {})}
    return build.train(corpus, p, lexicon=kw.get("lexicon", TINY_LEXICON), bridge=kw.get("bridge", TINY_BRIDGE),
                       safety=kw.get("safety", TINY_SAFETY))


@pytest.fixture(scope="module")
def tiny():
    return build.searcher(*tiny_model(), TINY)


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


def test_normalise_stops_at_a_stopword_instead_of_falling_through():
    # "telling" -> "tell" (a stopword): the word is filler; it must not fall through to "tel".
    assert LEX.normalise("telling", KNOWN) == "tell"
    assert LEX.normalise("telleth", KNOWN) == "tell"


def test_normalise_does_not_shorten_past_a_stopword():
    lex = Lexicon({"stopwords": ["blessing"], "query_stopwords": [], "irregular": {},
                   "suffix_rules": build.load_json(build.LEXICON)["suffix_rules"], "not_inflected": []})
    assert lex.normalise("blessings", {"bless", "blessing"}) == "blessing"


def test_candidates_shorter_than_three_letters_are_ignored():
    assert LEX.normalise("lies", {"ly", "lie"}) == "lie"


def test_normalise_applies_irregular_forms_first():
    assert LEX.normalise("art", set()) == "are"


def test_committed_lexicon_fixes(s):
    for w in ["telling", "telleth", "tellest"]:
        assert s.term_of(w) is None, w
    for w in ["judas", "annas", "jonas", "joses"]:           # names are never shortened
        assert s.terms[s.term_of(w)] == w
    assert s.terms[s.term_of("wives")] == s.terms[s.term_of("wife")] == "wife"
    assert s.terms[s.term_of("judgement")] == "judgment"
    assert s.terms[s.term_of("news")] == "news"
    assert s.terms[s.term_of("neighbor")] == "neighbour"
    assert s.term_of("hath") is None and s.term_of("thou") is None


def test_irregular_and_names_point_at_real_words(s):
    lex = build.load_json(build.LEXICON)
    for src, dst in lex["irregular"].items():
        assert dst in s.known, (src, dst)


def test_stopwords_are_derived_from_frequency_plus_hand_lists():
    cfg = {"top_by_verse_share": 3, "keep": ["for"], "add": ["extra"]}
    sw = build.derive_stopwords(TINY, cfg)
    # By share of verses (each book weighted equally): the 0.72, is 0.33, for 0.31, then soul, and, ...
    assert sw == ["extra", "is", "the"]


def test_proper_names_are_always_capitalised_words():
    corpus = {"gita": [_verse("1:1", "Judas went. The judge came.")], "quran": [_verse("1:1", "The Judge")],
              "bible": [_verse("1:1", "Annas and the high priest")]}
    names = build.proper_names(corpus)
    assert {"judas", "annas"} <= names and "judge" not in names and "the" not in names


def test_lexicon_adds_names_to_not_inflected():
    corpus = {**TINY, "bible": TINY["bible"] + [_verse("3:1", "And Annas sent him bound.")]}
    assert "annas" in build.build_lexicon(corpus, TINY_LEXICON)["not_inflected"]


def test_blocklist_patterns_ranges_and_keep_list():
    safety = {**TINY_SAFETY, "block": {**TINY_SAFETY["block"], "keep": [{"book": "gita", "ref": "1:1", "why": "t"}]}}
    reasons = build.blocklist(TINY, safety)
    assert reasons[0] == ""                                   # gita 1:1 matches "slain" but is kept
    assert reasons[3 + 4 + 1] == "test range"                # bible 1:2 by range
    assert sum(bool(r) for r in reasons) == 1
    assert build.blocklist(TINY, TINY_SAFETY)[0] == "violence"


def test_words_spells_out_contractions():
    assert words("I can't sleep, I'm scared & he won't call") == \
        ["i", "can", "not", "sleep", "i", "scared", "he", "will", "not", "call"]
    assert words("Don’t") == ["do", "not"]
    assert words("") == [] and words("!!! ...") == []


def test_phrase_matching_slots_and_prefixes():
    toks = words("my husband hit me last night")
    assert phrase_in(toks, compile_phrases(["husband|wife hit|hits me"])[0])
    assert not phrase_in(toks, compile_phrases(["wife hit me"])[0])
    assert phrase_in(words("I feel suicidal"), compile_phrases(["suicid*"])[0])
    assert not phrase_in(words("I feel sui"), compile_phrases(["suicid*"])[0])


# ----- expand and the bridge -----

def test_expand_typed_word_beats_a_later_bridge_weight(tiny):
    q = tiny.expand("weary and stressed")
    assert q[tiny.term_of("weary")][0] == 1.0          # typed first (1.0), then bridged (0.8): keeps 1.0
    assert q[tiny.term_of("patience")] == (tiny.params["bridge_weight"], "stressed")


def test_expand_phrase_hides_its_own_words(tiny):
    q = tiny.expand("pay rent")
    assert set(q) == {tiny.term_of("poor")}
    q = tiny.expand("I got laid off")
    assert {src for _, src in q.values()} == {"laid off"}


def test_expand_prefers_the_longest_phrase(tiny):
    q = tiny.expand("sick of waiting")
    assert tiny.term_of("patience") in q and tiny.term_of("death") not in q


def test_expand_replace_list_drops_the_literal_word(tiny):
    assert set(tiny.expand("my job")) == {tiny.term_of("duty")}
    assert tiny.expand("my job", bridge=False) == {}


def test_committed_replace_list_hides_job_the_prophet(s):
    assert s.term_of("job") is not None                       # Job is in the books
    q = s.expand("I hate my job")
    assert s.term_of("job") not in q and s.term_of("work") in q
    assert s.term_of("job") in s.expand("I hate my job", bridge=False)


def test_crisis_wins_over_sensitive(tiny, s):
    assert tiny.classify("I'm gay and I want to kill myself") == ("crisis", "suicide")
    assert s.classify("I'm gay and I want to die")[0] == "crisis"


def test_expand_drops_query_stopwords_but_keeps_them_in_the_books():
    corpus = {**TINY, "gita": TINY["gita"] + [_verse("1:4", "A well of water.")]}
    t = build.searcher(*tiny_model(corpus=corpus), corpus)
    assert t.term_of("well") is not None
    assert t.expand("well") == {}


def test_bridge_matching_is_exact(tiny):
    # "kidding" must not reach a "kid" key through suffix stripping.
    assert tiny.expand("stresseds") == {}


def test_committed_bridge_targets_are_all_in_the_corpus(s):
    bridge = build.load_json(build.BRIDGE)
    entries = {k: v for k, v in bridge.items() if not k.startswith("_")}
    assert len(entries) >= 300
    assert [(k, t) for k, v in entries.items() for t in v if s.term_of(t) is None] == []
    assert not [t for v in entries.values() for t in v if t in s.lex.stopwords]
    assert all(k == k.lower() and k.strip() == k and k == " ".join(words(k)) for k in entries)
    assert set(bridge["_replace"]) <= set(entries)


@pytest.mark.parametrize("query, absent", [
    ("I passed my driving test", ["died", "death"]),
    ("high school was hard", ["wine"]),
    ("I was moved to tears", ["stranger", "sojourn"]),
    ("I'm all fired up", ["idle", "wages", "poverty"]),
    ("my friend let me down", ["sorrow", "heaviness"]),
    ("I'm just kidding", ["child", "children"]),
])
def test_innocent_senses_do_not_fire_the_bridge(s, query, absent):
    got = {s.terms[t] for t in s.expand(query)}
    assert not got & set(absent), got


# ----- BM25, vectors and the hybrid -----

def test_bm25_matches_the_formula_with_query_weights(tiny):
    k1, b = tiny.params["bm25_k1"], tiny.params["bm25_b"]
    t = tiny.term_of("patience")
    lengths = [len(x) for x in tiny.books["quran"]["terms"]]
    avg = sum(lengths) / len(lengths)
    idf = math.log(1 + (4 - 1 + 0.5) / (1 + 0.5))     # "patience" is in 1 of 4 Quran verses ("patient" is another word)
    tf = 2
    one = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[3] / avg))
    s1 = tiny._scores({t: (1.0, "x")}, "quran", vectors=False)
    s2 = tiny._scores({t: (0.5, "x")}, "quran", vectors=False)
    assert s1[3].item() == pytest.approx(one, rel=1e-5)
    assert s2[3].item() == pytest.approx(0.5 * one, rel=1e-5)
    assert s1[:3].abs().sum() == 0


def test_bm25_idf_is_counted_inside_each_book(tiny):
    t = tiny.term_of("soul")       # in 1 of 3 Gita verses, 1 of 4 Quran verses, no Bible verse
    assert tiny.books["gita"]["idf"][t].item() == pytest.approx(math.log(1 + 2.5 / 1.5))
    assert tiny.books["quran"]["idf"][t].item() == pytest.approx(math.log(1 + 3.5 / 1.5))
    assert tiny.books["bible"]["idf"][t].item() == pytest.approx(math.log(1 + 3.5 / 0.5))


def test_global_idf_for_similar(tiny):
    t = tiny.term_of("soul")       # 2 of the 10 verses
    assert tiny.global_idf[t].item() == pytest.approx(math.log(1 + 10 / 3))


def test_query_vector_uses_sif_weights_and_removes_the_shared_direction(tiny):
    q = tiny.expand("soul death")
    v = torch.zeros(tiny.wv.shape[1])
    total = tiny.count.sum()
    for t, (w, _) in q.items():
        v += w * 1e-3 / (1e-3 + tiny.count[t] / total) * tiny.wv[t]
    v -= (v @ tiny.principal) * tiny.principal
    v = v / v.norm()
    got = tiny.query_vector(q)
    assert torch.allclose(got, v, atol=1e-5)
    assert got.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert (got @ tiny.principal).item() == pytest.approx(0.0, abs=1e-5)
    assert tiny.query_vector({}).abs().sum() == 0


def test_hybrid_is_the_scaled_mix_of_its_parts(s):
    q = "afraid of the future"
    a = s.params["alpha"]
    lex, _ = s.scores(q, "quran", vectors=False)
    vec, _ = s.scores(q, "quran", lexical=False)
    mix, _ = s.scores(q, "quran")
    assert torch.allclose(mix, a * lex / lex.max() + (1 - a) * vec / vec.max(), atol=1e-6)


# ----- search() -----

def test_search_returns_k_valid_hits_per_book(s, refs):
    res = s.search("I feel burnt out and alone", k=5)
    assert res["category"] == "general" and not res["no_match"]
    for book in BOOKS:
        hits = res["results"][book]
        assert len(hits) == 5 and len({h["ref"] for h in hits}) == 5
        for h in hits:
            assert refs[book][h["ref"]] == h["text"] and h["score"] > 0
        assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)


def test_search_rejects_bad_arguments(s):
    for k in (0, -1, 2.5):
        with pytest.raises(ValueError):
            s.search("sad", k=k)
    with pytest.raises(ValueError):
        s.search("sad", method="nope")


def test_search_no_match_returns_nothing(s):
    for q in ["", "   ", "!!! ???", "xyzzy qwerty", "the and of unto thee", "12345"]:
        r = s.search(q, k=3)
        assert r["no_match"] and r["category"] == "general", q
        assert all(r["results"][b] == [] for b in BOOKS)


def test_search_matched_words_are_only_words_in_the_verse(s):
    r = s.search("I'm drowning in debt and afraid", k=3)
    for book in BOOKS:
        for h in r["results"][book]:
            verse_terms = set(s.books[book]["terms"][s.books[book]["index"][h["ref"]]].tolist())
            assert h["matched"]
            for m in h["matched"]:
                assert s.term_of(m["word"]) in verse_terms


def test_search_small_book_is_not_crowded_out(s):
    res = s.search("Pharaoh Egypt Moses sea", k=3)
    assert len(res["results"]["quran"]) == 3
    assert any("Pharaoh" in h["text"] or "sea" in h["text"] for h in res["results"]["quran"])


@pytest.mark.parametrize("query, book, expected, k", [
    ("shepherd", "bible", {"Psalms 23:1", "John 10:11", "John 10:14"}, 10),
    ("the Lord is my shepherd", "bible", {"Psalms 23:1"}, 10),
    ("love your enemies", "bible", {"Matthew 5:44", "Luke 6:27", "Luke 6:35"}, 10),
    ("strong drink and games of chance", "quran", {"5:90", "2:219", "5:91"}, 10),
    ("with hardship goeth ease", "quran", {"94:5", "94:6"}, 3),
    ("action without attachment to its fruits", "gita", {"2:47", "3:19", "4:20", "5:12", "18:9", "6:1", "18:6", "18:23"}, 10),
])
def test_search_sanity_queries(s, query, book, expected, k):
    got = [h["ref"] for h in s.search(query, k=k)["results"][book]]
    assert expected & set(got), got


def test_search_respects_blocklist_and_gentle_set(tiny):
    r = tiny.search("soul", k=3)                 # gita 1:1 mentions "slain": blocked
    assert r["results"]["gita"] == []
    assert [h["ref"] for h in r["results"]["quran"]] == ["1:2"]
    r = tiny.search("poor", k=3)                 # bible 1:2 is blocked by ref range
    assert r["results"]["bible"] == []
    r = tiny.search("I'm gay", k=3)              # sensitive: only the gentle set, curated order
    assert r["category"] == "sensitive" and r["topic"] == "identity"
    assert [h["ref"] for h in r["results"]["quran"]] == ["2:1", "1:1"]
    assert r["results"]["gita"] == []            # its only gentle verse (1:1) is also blocked
    assert tiny.search("he hit me", k=3)["category"] == "crisis"
    assert all(v == [] for v in tiny.search("I want to kill myself", 3)["results"].values())


def test_gentle_verse_that_is_blocked_is_never_shown(tiny):
    # gita 1:1 is both gentle and blocked: blocking wins.
    assert tiny.books["gita"]["blocked"][0] and tiny.books["gita"]["gentle"][0]
    assert tiny.search("gay soul", k=3)["results"]["gita"] == []


# ----- similar() -----

def test_similar_score_is_the_documented_mix(tiny):
    lam = tiny.params["similar_vector_weight"]
    b, i = tiny.books["quran"], 1
    target = tiny.books["gita"]
    sims = tiny.similar("quran", "1:2", k=3, books=["gita"], allow_blocked=True)["gita"]
    lex = tiny._tfidf(b["terms"][i])
    for h in sims:
        j = target["index"][h["ref"]]
        vec = (target["vectors"][j] @ b["vectors"][i]).item()
        tf = torch.bincount(target["terms"][j], minlength=len(tiny.terms)).float() * tiny.global_idf
        cos = (tf @ lex).item() / tf.norm().item() if tf.norm() > 0 else 0.0
        assert h["score"] == pytest.approx(lam * vec + (1 - lam) * cos, abs=1e-3)


def test_similar_errors_and_exclusions(s):
    for args in [("torah", "1:1"), ("gita", "99:99")]:
        with pytest.raises(ValueError):
            s.similar(*args)
    with pytest.raises(ValueError):
        s.similar("gita", "2:47", k=0)
    with pytest.raises(ValueError):
        s.similar("gita", "2:47", books=["torah"])
    res = s.similar("bible", "Genesis 37:9", k=7)
    assert all(len(v) == 7 for v in res.values())
    assert "Genesis 37:9" not in [h["ref"] for h in res["bible"]]
    for book in BOOKS:
        for h in res[book]:
            assert not s.books[book]["blocked"][s.books[book]["index"][h["ref"]]]


@pytest.mark.parametrize("a, b", [
    (("bible", "Genesis 37:9"), ("quran", "12:4")),             # Joseph's dream
    (("bible", "Matthew 19:24"), ("quran", "7:40")),            # camel and the needle
])
def test_similar_finds_known_parallels_both_ways(s, a, b):
    ab = [h["ref"] for h in s.similar(a[0], a[1], k=10, books=[b[0]], allow_blocked=True)[b[0]]]
    ba = [h["ref"] for h in s.similar(b[0], b[1], k=10, books=[a[0]], allow_blocked=True)[a[0]]]
    assert b[1] in ab and a[1] in ba


# ----- build pieces -----

def test_cooccurrence_counts_window_and_book_weight():
    streams = [("gita", torch.tensor([0, 1, 2])), ("bible", torch.tensor([0, 2]))]
    c = build.cooccurrence(streams, {"gita": 2.0, "bible": 1.0}, 3, window=2)
    assert c[0, 1] == 2.0 and c[1, 2] == 2.0 and c[0, 2] == 2.0 + 1.0
    assert torch.equal(c, c.t())
    c1 = build.cooccurrence(streams, {"gita": 2.0, "bible": 1.0}, 3, window=1)
    assert c1[0, 2] == 1.0                      # distance 2 in gita is outside window 1


def test_ppmi_formula_with_context_smoothing():
    c = torch.tensor([[0., 4., 1.], [4., 0., 2.], [1., 2., 0.]])
    got = build.ppmi(c, 0.75)
    row = c.sum(1, keepdim=True)
    col = c.sum(0, keepdim=True) ** 0.75
    expect = torch.log(c * col.sum() / (row * col)).nan_to_num(neginf=0).clamp(min=0)
    assert torch.allclose(got, expect, atol=1e-5)
    assert got.min() >= 0
    assert not torch.allclose(got, build.ppmi(c, 1.0))


def test_word_vectors_are_u_times_s_to_the_power():
    model, wv, _, _ = tiny_model({"svd_power": 0.5})
    vocab = build.build_vocab(TINY, 1, Lexicon(model["lexicon"]))
    streams = build.chapter_streams(TINY, vocab["verse_terms"], vocab["n_vec"])
    weights = build.book_weights(TINY, vocab["verse_terms"], build.PARAMS["book_balance"])
    m = build.ppmi(build.cooccurrence(streams, weights, vocab["n_vec"], 2), build.PARAMS["cds"])
    torch.manual_seed(0)
    u, sv, _ = torch.svd_lowrank(m, q=4, niter=6)
    assert torch.allclose(wv, u * sv ** 0.5, atol=1e-5)
    assert not torch.allclose(wv, u * sv, atol=1e-3)


def test_book_weights_balance_small_books():
    vt = [torch.tensor([1, 2])] * 3 + [torch.tensor([1])] * 4 + [torch.tensor([1, 2, 3, 4])] * 3
    w = build.book_weights(TINY, vt, 0.5)
    assert w["bible"] == 1.0
    assert w["gita"] == pytest.approx((12 / 6) ** 0.5) and w["quran"] == pytest.approx((12 / 4) ** 0.5)


def test_chapter_streams_do_not_cross_chapters():
    vt = [torch.tensor([i]) for i in range(10)]
    streams = build.chapter_streams(TINY, vt, 100)
    assert [(b, s.tolist()) for b, s in streams] == [
        ("gita", [0, 1, 2]), ("quran", [3, 4, 5]), ("quran", [6]), ("bible", [7, 8]), ("bible", [9])]


def test_min_count_limits_terms_with_vectors():
    lex = Lexicon(build.build_lexicon(TINY, TINY_LEXICON))
    v1, v2 = build.build_vocab(TINY, 1, lex), build.build_vocab(TINY, 2, lex)
    assert v2["n_vec"] < v1["n_vec"] == len(v1["terms"])
    assert all(v2["count"][t] >= 2 for t in range(v2["n_vec"]))


def test_vocab_drops_words_whose_base_is_a_stopword():
    corpus = {**TINY, "gita": TINY["gita"] + [_verse("1:4", "He was telling the truth; tell it.")]}
    lex = Lexicon(build.build_lexicon(corpus, TINY_LEXICON))
    v = build.build_vocab(corpus, 1, lex)
    assert "telling" not in v["terms"] and "tell" not in v["terms"] and "truth" in v["terms"]


def test_verse_vectors_are_sif_weighted_and_orthogonal_to_the_shared_direction():
    wv = torch.eye(3)
    counts = [10, 1, 1]
    vt = [torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 2]), torch.tensor([0])]
    vv, principal = build.verse_vectors(vt, wv, counts, 0.5)
    assert torch.allclose(vv @ principal, torch.zeros(4), atol=1e-5)
    # rebuild by hand
    p = torch.tensor(counts, dtype=torch.float32)
    sif = 0.5 / (0.5 + p / p.sum())
    raw = torch.stack([sif[t].unsqueeze(1).mul(wv[t]).sum(0) for t in vt])
    raw -= (raw @ principal).unsqueeze(1) * principal
    assert torch.allclose(vv, raw / raw.norm(dim=1, keepdim=True), atol=1e-5)


def test_quantise_is_per_row():
    m = torch.tensor([[0.001, -0.002, 0.0005], [10.0, 5.0, -2.0]])
    q = build.quantise(m)
    assert q.abs().max(dim=1).values.tolist() == [127, 127]
    torch.manual_seed(0)
    r = torch.randn(50, 64)
    cos = torch.nn.functional.cosine_similarity(r, build.quantise(r).float(), dim=1)
    assert cos.min() > 0.999


def test_read_u16_unwraps_large_ids(tmp_path):
    from array import array
    (tmp_path / "x.u16").write_bytes(array("H", [1, 32767, 32768, 40000, 65535]).tobytes())
    assert read_u16(tmp_path / "x.u16").tolist() == [1, 32767, 32768, 40000, 65535]


def test_bitmask_round_trip():
    flags = [True, False, False, True, True, False, False, False, True, False, True]
    assert unpack_bits(build.pack_bits(flags), len(flags)).tolist() == flags


def test_build_is_deterministic():
    a, b = tiny_model(), tiny_model()
    assert json.dumps(a[0], sort_keys=True) == json.dumps(b[0], sort_keys=True)
    assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


def test_quantised_artefacts_rank_like_full_precision(tmp_path):
    verses = build.load_verses()
    small = {b: verses[b][:400] for b in BOOKS}
    model, wv, vv, vt = build.train(small, {"dim": 32, "min_count": 2})
    full = build.searcher(model, wv, vv, vt, small)
    build.write(model, wv, vv, vt, tmp_path)
    loaded = Searcher.load(tmp_path, data_dir=None)
    assert loaded.terms == full.terms
    for q in ["the beginning of creation", "war and battle", "sorrow and grief", "a righteous man"]:
        for book in BOOKS:
            a, _ = full.scores(q, book, lexical=False)
            b, _ = loaded.scores(q, book, lexical=False)
            assert len(set(torch.topk(a, 10).indices.tolist()) & set(torch.topk(b, 10).indices.tolist())) >= 8
            assert torch.corrcoef(torch.stack([a, b]))[0, 1] > 0.99
            assert torch.allclose(full.scores(q, book, vectors=False)[0], loaded.scores(q, book, vectors=False)[0])


# ----- committed artefacts -----

def test_artefacts_are_small_and_consistent_with_the_inputs(s):
    total = sum(f.stat().st_size for f in ARTIFACTS.iterdir())
    assert total < 10e6
    model = json.loads((ARTIFACTS / "model.json").read_text())
    verses = build.load_verses()
    assert model["inputs"]["corpus_sha256"] == build.corpus_sha256(verses)
    for name, path in (("lexicon", build.LEXICON), ("bridge", build.BRIDGE), ("safety", build.SAFETY)):
        h = hashlib.sha256(json.dumps(build.load_json(path), sort_keys=True).encode()).hexdigest()
        assert model["inputs"][f"{name}_sha256"] == h, f"{path.name} changed: rerun python search/build.py"
    n = sum(len(model["books"][b]["refs"]) for b in BOOKS)
    assert n == 701 + 6236 + 31102
    assert (ARTIFACTS / "verse_vectors.i8").stat().st_size == n * model["dim"]
    assert (ARTIFACTS / "verse_offsets.u32").stat().st_size == 4 * (n + 1)
    assert len(model["words"]) == len(model["word_term"])
    assert s.wv.shape[0] <= len(model["terms"]) < 65536


def test_fixtures_match_the_committed_artefacts(s):
    fx = json.loads((ARTIFACTS / "fixtures.json").read_text())
    assert len(fx["cases"]) >= 30
    for c in fx["cases"]:
        r = s.search(c["query"], k=3)
        assert (r["category"], r["topic"], r["no_match"]) == (c["category"], c["topic"], c["no_match"])
        assert sorted(s.terms[t] for t in s.expand(c["query"])) == c["terms"]
        for book in BOOKS:
            got = r["results"][book]
            assert [h["ref"] for h in got] == [x[0] for x in c["results"][book]]
            assert all(abs(h["score"] - x[1]) < 2e-3 for h, x in zip(got, c["results"][book]))


def test_gentle_set_is_curated_and_never_blocked(s, refs):
    safety = build.load_json(build.SAFETY)
    for book in BOOKS:
        g = safety["gentle"][book]
        assert 20 <= len(g) <= 45
        for ref, why in g:
            assert ref in refs[book] and why
            assert not s.books[book]["blocked"][s.books[book]["index"][ref]]
        assert int(s.books[book]["gentle"].sum()) == len(g)


def test_blocklist_blocks_the_reported_harmful_verses(s):
    for book, ref in [("quran", "7:124"), ("quran", "17:31"),
                      ("bible", "Romans 7:2"), ("bible", "Exodus 34:7"), ("quran", "4:16"), ("quran", "81:8"),
                      ("bible", "Genesis 19:36"), ("quran", "5:51"), ("quran", "9:34"), ("bible", "Titus 1:14"),
                      ("quran", "2:88"), ("quran", "17:72"), ("quran", "2:18"), ("bible", "Psalms 109:9"),
                      ("quran", "43:17"), ("bible", "Psalms 137:9")]:
        assert s.books[book]["blocked"][s.books[book]["index"][ref]], (book, ref)
    for book, ref in [("bible", "Psalms 23:4"), ("bible", "Matthew 5:44"), ("quran", "39:53"), ("quran", "93:3"),
                      ("gita", "2:20"), ("bible", "Galatians 3:28")]:
        assert not s.books[book]["blocked"][s.books[book]["index"][ref]], (book, ref)
    for book in BOOKS:
        assert int(s.books[book]["blocked"].sum()) > 0


def test_blocklist_keep_entries_point_at_real_verses(refs):
    safety = build.load_json(build.SAFETY)
    for k in safety["block"]["keep"]:
        assert k["ref"] in refs[k["book"]] and k["why"]
    for r in safety["block"]["refs"]:
        assert r["from"] in refs[r["book"]] and r["to"] in refs[r["book"]] and r["reason"]


# ----- similarities -----

def test_similarities_are_reviewed_and_well_formed(s, refs):
    sims = json.loads((ARTIFACTS / "similarities.json").read_text())
    review = json.loads(SIM.REVIEW.read_text())
    kept = [c for c in review["candidates"] if c["keep"]]
    assert sims["candidates_proposed"] == len(review["candidates"]) and sims["kept"] == len(kept)
    assert all(c["theme"] and c["idea"] for c in kept)
    pairs = [p for t in sims["themes"] for p in t["pairs"]]
    assert 60 <= len(pairs) <= 120
    books_paired = set()
    for p in pairs:
        assert p["idea"] and p["source"] in ("model, reviewed", "hand-listed")
        (a, b) = p["verses"]
        assert a["book"] != b["book"]
        books_paired.add(frozenset((a["book"], b["book"])))
        for v in (a, b):
            assert refs[v["book"]][v["ref"]] == v["text"]
            assert not s.books[v["book"]]["blocked"][s.books[v["book"]]["index"][v["ref"]]]
    assert len(books_paired) == 3


# ----- eval maths -----

def test_first_rank_counts_ties_against_us():
    s = torch.tensor([0.1, 0.9, 0.5, 0.5, 0.3])
    assert E.first_rank(s, torch.tensor([2])) == 3
    assert E.first_rank(s, torch.tensor([2, 3])) == 2
    assert E.first_rank(s, torch.tensor([1])) == 1
    assert E.first_rank(s, torch.tensor([0, 4])) == 4


def test_pair_metrics_boundaries():
    assert E.pair_metrics(10) == {"hit@3": 0.0, "hit@10": 1.0, "mrr": 0.1}
    assert E.pair_metrics(3)["hit@3"] == 1.0 and E.pair_metrics(4)["hit@3"] == 0.0
    assert E.pair_metrics(11)["hit@10"] == 0.0


def test_mean_metrics_and_empty():
    r = E.mean_metrics([E.pair_metrics(x) for x in (1, 2, 4, 11)])
    assert r["hit@3"] == 0.5 and r["hit@10"] == 0.75 and r["n"] == 4
    assert math.isclose(r["mrr"], (1 + 1 / 2 + 1 / 4 + 1 / 11) / 4)
    assert E.mean_metrics([])["hit@10"] is None


def test_random_baseline_matches_brute_force():
    n, r = 6, 2
    ranks = [min(perm.index(i) for i in range(r)) + 1 for perm in itertools.permutations(range(n))]
    for k in (1, 3, 5):
        assert math.isclose(E.random_hit(n, r, k), sum(x <= k for x in ranks) / len(ranks))
    assert math.isclose(E.random_mrr(n, r), sum(1 / x for x in ranks) / len(ranks))


def test_bootstrap_interval_contains_the_mean_and_resamples_situations():
    per_case = [[E.pair_metrics(1)], [E.pair_metrics(20)], [E.pair_metrics(2), E.pair_metrics(2)]]
    ci = E.bootstrap(per_case, reps=500)
    point = E.mean_metrics([r for rows in per_case for r in rows])["hit@10"]
    assert ci["hit@10"][0] <= point <= ci["hit@10"][1]
    assert ci["hit@10"][0] < ci["hit@10"][1]
    assert E.bootstrap([[E.pair_metrics(1)]] * 5)["mrr"] == [1.0, 1.0]
    assert E.bootstrap([]) is None


def test_evaluate_on_constructed_cases(tiny):
    cases = [
        {"query": "patience", "split": "dev", "relevant": {"gita": [], "quran": ["2:1"], "bible": []}},
        {"query": "shepherd", "split": "dev", "relevant": {"gita": ["1:3"], "quran": [], "bible": ["2:1"]}},
    ]
    r = E.evaluate(tiny, cases, "bm25")
    assert r["quran"]["n"] == 1 and r["quran"]["mrr"] == 1.0
    assert r["bible"]["mrr"] == 1.0
    assert r["gita"]["n"] == 1 and r["gita"]["hit@10"] == 1.0 and r["gita"]["mrr"] < 1.0
    assert r["overall"]["n"] == 3
    assert r["overall"]["mrr"] == pytest.approx((1 + 1 + r["gita"]["mrr"]) / 3)
    rnd = E.evaluate(tiny, cases, "random")
    assert rnd["gita"]["mrr"] == pytest.approx(E.random_mrr(2, 1))         # counted over the 2 open verses
    assert rnd["bible"]["mrr"] == pytest.approx(E.random_mrr(2, 1))
    assert rnd["gita"]["hit@3"] == pytest.approx(E.random_hit(2, 1, 3))      # gita 1:1 is blocked: 2 open verses
    assert rnd["quran"]["hit@3"] == pytest.approx(E.random_hit(4, 1, 3))
    assert rnd["bible"]["hit@3"] == pytest.approx(E.random_hit(2, 1, 3))
    # a blocked relevant verse is dropped from the key
    blocked_case = [{"query": "soul", "split": "dev", "relevant": {"gita": ["1:1"], "quran": [], "bible": []}}]
    assert E.evaluate(tiny, blocked_case, "bm25")["relevant_dropped_as_blocked"] == 1


def test_evaluate_parallels_counts_only_the_other_side_and_restores_settings(tiny):
    before = tiny.params["similar_vector_weight"]
    order = [h["ref"] for h in tiny.similar("quran", "1:2", k=3, books=["gita"], allow_blocked=True)["gita"]]
    # The parallel lists the top gita verse only on the quran side (same ref string): it must not count.
    par = [{"id": "x", "kind": "story", "a": {"book": "quran", "refs": ["1:2", order[0]]},
            "b": {"book": "gita", "refs": [order[1]]}}]
    r = E.evaluate_parallels(tiny, par, vector_weight=0.3, k=1)
    assert tiny.params["similar_vector_weight"] == before
    assert r["trials"] == 2 and "hit@1" in r
    first = next(d for d in r["detail"] if d["from"] == "quran 1:2")
    assert first["rank"] is None               # order[1] is 2nd, so not in the top 1
    r2 = E.evaluate_parallels(tiny, par, k=2)
    first = next(d for d in r2["detail"] if d["from"] == "quran 1:2")
    assert first["rank"] == 2
    assert all(d["rank"] is None or d["rank"] <= 2 for d in r2["detail"])


def test_eval_cases_and_parallels_point_at_real_verses(refs):
    doc = json.loads(E.CASES.read_text())
    cases = doc["cases"]
    assert {c["split"] for c in cases} <= set(E.SPLITS) and "test" not in {c["split"] for c in cases}
    assert len(cases) >= 40 and len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        for book in BOOKS:
            for r in c["relevant"][book]:
                assert r in refs[book], (c["id"], r)
    for p in json.loads(E.PARALLELS.read_text())["parallels"]:
        for side in ("a", "b"):
            for r in p[side]["refs"]:
                assert r in refs[p[side]["book"]], (p["id"], r)
