"""Tests for the meaning search (scripture/search.py, search/, evals/search_eval.py).

Most tests use the committed artefacts in search/artifacts/. The round-trip
test trains a small model on a slice of the corpus.
"""

import itertools
import json
import math
from pathlib import Path

import pytest
import torch

from evals import search_eval as E
from scripture.search import ARTIFACTS, BOOKS, BRIDGE, STOPWORDS, Searcher, candidates, normalise, words
from search import build

ROOT = Path(__file__).resolve().parent.parent
torch.set_num_threads(2)


@pytest.fixture(scope="module")
def s():
    return Searcher.load()


@pytest.fixture(scope="module")
def refs():
    out = {}
    for book in BOOKS:
        with open(ROOT / "data" / "processed" / f"{book}.jsonl") as f:
            out[book] = {json.loads(line)["ref"]: json.loads(line)["text"] for line in f}
    return out


# ----- words and normalising -----

KNOWN = {"love", "walk", "give", "sin", "cry", "bless", "blessing", "die", "evening", "even",
         "forest", "for", "naked", "seed", "see", "rich", "hate", "horse"}


@pytest.mark.parametrize("word, base", [
    ("loveth", "love"), ("loved", "love"), ("loving", "love"), ("loves", "love"), ("lovest", "love"),
    ("walketh", "walk"), ("walked", "walk"), ("giveth", "give"), ("sinned", "sin"), ("sinneth", "sin"),
    ("sinning", "sin"), ("crieth", "cry"), ("cried", "cry"), ("cries", "cry"), ("blessings", "bless"),
    ("dying", "die"), ("died", "die"), ("riches", "rich"), ("hateth", "hate"), ("horses", "horse"),
])
def test_normalise_reduces_archaic_and_inflected_forms(word, base):
    assert normalise(word, KNOWN) == base


@pytest.mark.parametrize("word", ["evening", "forest", "naked", "seed", "love", "unknownword"])
def test_normalise_leaves_words_that_only_look_inflected(word):
    # "evening" -> "even" and "forest" -> "for" would land on stopwords; "seed" is not "see" + "d".
    assert normalise(word, KNOWN) == word


def test_normalise_maps_irregular_archaic_verbs():
    assert normalise("hath", set()) == "have"
    assert normalise("spake", set()) == "speak"
    assert normalise("doeth", set()) == "do"


def test_candidates_never_offer_stopwords_or_short_stems():
    for w in ["evening", "being", "sing", "seed", "doeth", "teeth"]:
        for c in candidates(w):
            assert c not in STOPWORDS and len(c) >= 3


def test_words_spells_out_contractions():
    assert words("I can't sleep, I'm scared & he won't call") == \
        ["i", "can", "not", "sleep", "i", "scared", "he", "will", "not", "call"]
    assert words("Don’t") == ["do", "not"]


# ----- bridge -----

def test_bridge_targets_are_all_in_the_corpus_vocabulary(s):
    bridge = json.loads(BRIDGE.read_text())
    entries = {k: v for k, v in bridge.items() if not k.startswith("_")}
    assert len(entries) >= 300
    missing = [(k, t) for k, v in entries.items() for t in v if s.term_of(t) is None]
    assert missing == []
    # Every target is a real scripture word, not a filler word.
    assert not [t for v in entries.values() for t in v if t in STOPWORDS]


def test_bridge_keys_are_lower_case_and_replace_list_is_valid():
    bridge = json.loads(BRIDGE.read_text())
    keys = [k for k in bridge if not k.startswith("_")]
    assert all(k == k.lower() and k.strip() == k for k in keys)
    assert set(bridge["_replace"]) <= set(keys)


def test_bridge_covers_the_modern_words_it_was_written_for(s):
    # Words that never occur in the books: everything they find comes from the bridge.
    for word in ["girlfriend", "cancer", "burnout", "insomnia", "coworker", "porn", "unemployed"]:
        assert s.expand(word, bridge=False) == {}, word
        q = s.expand(word)
        assert q and all(w == s.params["bridge_weight"] for w, _ in q.values()), word


def test_expand_weights_typed_words_above_bridge_words(s):
    q = s.expand("stressed about money")
    money = s.term_of("money")
    assert q[money][0] == 1.0                       # typed, and a scripture word
    assert q[s.term_of("weary")] == (s.params["bridge_weight"], "stressed")
    assert s.params["bridge_weight"] < 1.0


def test_expand_phrase_hides_its_own_words(s):
    q = s.expand("I got laid off")
    sources = {src for _, src in q.values()}
    assert sources == {"laid off"}
    assert s.term_of("laid") not in q


def test_expand_replace_list_drops_the_literal_word(s):
    # "job" in scripture is the prophet Job: the word itself must not be searched.
    q = s.expand("my job")
    assert s.term_of("job") not in q
    assert s.term_of("work") in q
    # without the bridge the literal word is used
    assert s.term_of("job") in s.expand("my job", bridge=False)


def test_expand_without_bridge_ignores_modern_words(s):
    assert s.expand("girlfriend", bridge=False) == {}
    assert s.expand("the and of unto thee") == {}


# ----- search -----

def test_search_returns_k_valid_hits_per_book(s, refs):
    res = s.search("I feel burnt out and alone", k=5)
    assert set(res) == set(BOOKS)
    for book in BOOKS:
        hits = res[book]
        assert len(hits) == 5
        assert len({h["ref"] for h in hits}) == 5
        for h in hits:
            assert refs[book][h["ref"]] == h["text"]
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)


def test_search_small_book_is_not_crowded_out(s):
    # A query full of Bible names still gets Gita and Quran verses of its own.
    res = s.search("Pharaoh Egypt Moses Israel", k=3)
    assert len(res["gita"]) == 3 and len(res["quran"]) == 3
    assert any("Pharaoh" in h["text"] for h in res["quran"])


@pytest.mark.parametrize("query, book, expected, k", [
    ("shepherd", "bible", {"Psalms 23:1", "John 10:11", "John 10:14"}, 10),
    ("the Lord is my shepherd I shall not want", "bible", {"Psalms 23:1"}, 10),
    ("love your enemies", "bible", {"Matthew 5:44", "Luke 6:27", "Luke 6:35"}, 10),
    ("strong drink and games of chance", "quran", {"5:90", "2:219", "5:91"}, 10),
    ("with hardship goeth ease", "quran", {"94:5", "94:6"}, 3),
    ("action without attachment to its fruits", "gita", {"2:47", "3:19", "4:20", "5:12", "18:9", "6:1"}, 10),
])
def test_search_sanity_queries(s, query, book, expected, k):
    got = [h["ref"] for h in s.search(query, k=k)[book]]
    assert expected & set(got), got


def test_search_matched_words_explain_the_hit(s):
    hit = s.search("I'm drowning in debt", k=1)["bible"][0]
    assert hit["matched"], hit
    for m in hit["matched"]:
        assert m["word"] in s.terms
        assert m["word"] in hit["text"].lower() or m["from"] != m["word"]


def test_methods_differ_and_bridge_matters(s):
    q = "my girlfriend dumped me"
    plain, _ = s.scores(q, "bible", lexical=True, vectors=False, bridge=False)
    assert plain.abs().sum() == 0                   # no scripture words at all without the bridge
    bridged, _ = s.scores(q, "bible", lexical=True, vectors=False, bridge=True)
    assert bridged.max() > 0


def test_hybrid_is_a_mix_of_its_parts(s):
    q = "afraid of the future"
    a = s.params["alpha"]
    lex, _ = s.scores(q, "quran", vectors=False)
    vec, _ = s.scores(q, "quran", lexical=False)
    mix, _ = s.scores(q, "quran")
    assert torch.allclose(mix, a * lex / lex.max() + (1 - a) * vec / vec.max(), atol=1e-6)


# ----- similar -----

def test_similar_excludes_itself_and_returns_k(s):
    res = s.similar("bible", "Genesis 37:9", k=7)
    assert set(res) == set(BOOKS)
    assert all(len(v) == 7 for v in res.values())
    assert "Genesis 37:9" not in [h["ref"] for h in res["bible"]]


@pytest.mark.parametrize("a, b", [
    (("bible", "Genesis 37:9"), ("quran", "12:4")),             # Joseph's dream
    (("bible", "Matthew 19:24"), ("quran", "7:40")),            # camel and the needle
    (("bible", "Exodus 21:24"), ("quran", "5:45")),             # eye for an eye
])
def test_similar_finds_known_parallels_both_ways(s, a, b):
    ab = [h["ref"] for h in s.similar(a[0], a[1], k=10, books=[b[0]])[b[0]]]
    ba = [h["ref"] for h in s.similar(b[0], b[1], k=10, books=[a[0]])[a[0]]]
    assert b[1] in ab and a[1] in ba


def test_similar_score_is_symmetric(s):
    x = s.similar("bible", "Genesis 37:9", k=1, books=["quran"])["quran"][0]
    back = s.similar("quran", x["ref"], k=len(s.books["bible"]["refs"]) - 1, books=["bible"])["bible"]
    score_back = next(h["score"] for h in back if h["ref"] == "Genesis 37:9")
    assert abs(score_back - x["score"]) < 1e-3


# ----- artefacts -----

def test_artefacts_are_small_and_consistent(s):
    total = sum(f.stat().st_size for f in ARTIFACTS.iterdir())
    assert total < 10e6
    model = json.loads((ARTIFACTS / "model.json").read_text())
    n = sum(len(model["books"][b]["refs"]) for b in BOOKS)
    assert n == 701 + 6236 + 31102
    assert (ARTIFACTS / "verse_vectors.i8").stat().st_size == n * model["dim"]
    assert len(model["words"]) == len(model["word_term"])
    assert s.wv.shape[0] <= len(model["terms"]) < 65536


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
            top_a = set(torch.topk(a, 10).indices.tolist())
            top_b = set(torch.topk(b, 10).indices.tolist())
            assert len(top_a & top_b) >= 8, (q, book)
            assert torch.corrcoef(torch.stack([a, b]))[0, 1] > 0.99
            # BM25 does not depend on the vectors and survives exactly
            la, _ = full.scores(q, book, vectors=False)
            lb, _ = loaded.scores(q, book, vectors=False)
            assert torch.allclose(la, lb)


def test_quantise_keeps_direction():
    torch.manual_seed(0)
    m = torch.randn(50, 64)
    q = build.quantise(m).float()
    assert q.abs().max() == 127
    cos = torch.nn.functional.cosine_similarity(m, q, dim=1)
    assert cos.min() > 0.999


# ----- eval maths -----

def test_first_rank_counts_ties_against_us():
    s = torch.tensor([0.1, 0.9, 0.5, 0.5, 0.3])
    assert E.first_rank(s, torch.tensor([2])) == 3      # tied with index 3, counted ahead
    assert E.first_rank(s, torch.tensor([2, 3])) == 2   # both relevant: first of them is rank 2
    assert E.first_rank(s, torch.tensor([1])) == 1
    assert E.first_rank(s, torch.tensor([0, 4])) == 4


def test_summarise_hit_and_mrr():
    r = E.summarise([1, 2, 4, 11])
    assert r["hit@3"] == 0.5 and r["hit@10"] == 0.75
    assert math.isclose(r["mrr"], (1 + 1 / 2 + 1 / 4 + 1 / 11) / 4)


def test_random_baseline_matches_brute_force():
    n, r = 6, 2
    ranks = []
    for perm in itertools.permutations(range(n)):
        ranks.append(min(perm.index(i) for i in range(r)) + 1)
    for k in (1, 3, 5):
        assert math.isclose(E.random_hit(n, r, k), sum(x <= k for x in ranks) / len(ranks))
    assert math.isclose(E.random_mrr(n, r), sum(1 / x for x in ranks) / len(ranks))


def test_evaluate_on_a_constructed_case(s):
    case = {"query": "the Lord is my shepherd I shall not want", "split": "dev",
            "relevant": {"gita": [], "quran": [], "bible": ["Psalms 23:1", "Genesis 1:1"]}}
    r = E.evaluate(s, [case], "bm25")
    scores, _ = s.scores(case["query"], "bible", vectors=False, bridge=False)
    idx = torch.tensor([s.books["bible"]["index"]["Psalms 23:1"], 0])
    rank = E.first_rank(scores, idx)
    assert r["bible"]["n"] == 1 and r["gita"]["n"] == 0
    assert r["bible"]["mrr"] == 1 / rank and r["overall"]["mrr"] == 1 / rank
    rnd = E.evaluate(s, [case], "random")
    assert math.isclose(rnd["bible"]["hit@10"], E.random_hit(31102, 2, 10))


def test_eval_cases_and_parallels_point_at_real_verses(refs):
    cases = json.loads(E.CASES.read_text())["cases"]
    assert sum(c["split"] == "test" for c in cases) >= 15 and len(cases) >= 40
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        for book in BOOKS:
            for r in c["relevant"][book]:
                assert r in refs[book], (c["id"], r)
    for p in json.loads(E.PARALLELS.read_text())["parallels"]:
        for side in ("a", "b"):
            for r in p[side]["refs"]:
                assert r in refs[p[side]["book"]], (p["id"], r)


def test_similarities_file_is_well_formed(refs):
    sims = json.loads((ARTIFACTS / "similarities.json").read_text())
    pairs = [p for t in sims["themes"] for p in t["pairs"]]
    assert len(pairs) >= 150
    seen = set()
    for p in pairs:
        assert p["a"]["book"] != p["b"]["book"]
        for side in ("a", "b"):
            assert refs[p[side]["book"]][p[side]["ref"]] == p[side]["text"]
        key = (p["a"]["book"], p["a"]["ref"], p["b"]["book"])
        assert key not in seen
        seen.add(key)
    assert not [p for p in pairs if "spake unto Moses, saying" in p["a"]["text"] + p["b"]["text"]]


# ----- BM25 and vectors on a tiny made-up corpus -----

def _verse(ref, text):
    ch, v = ref.split(":")
    return {"ref": ref, "chapter": int(ch), "verse": int(v), "section": "S", "text": text}


@pytest.fixture(scope="module")
def tiny():
    corpus = {
        "gita": [_verse("1:1", "The soul is eternal; the soul is never slain."),
                 _verse("1:2", "Do thy duty without attachment to fruit."),
                 _verse("1:3", "The wise grieve not for the dead.")],
        "quran": [_verse("1:1", "Be patient, for with hardship cometh ease."),
                  _verse("1:2", "The soul shall taste of death."),
                  _verse("1:3", "Give alms to the poor and the orphan."),
                  _verse("1:4", "Patience and prayer, patience and prayer.")],
        "bible": [_verse("1:1", "Take no thought for the morrow."),
                  _verse("1:2", "Blessed are the poor in spirit."),
                  _verse("1:3", "The LORD is my shepherd; I shall not want.")],
    }
    model, wv, vv, vt = build.train(corpus, {"dim": 4, "min_count": 1})
    return build.searcher(model, wv, vv, vt, corpus)


def test_bm25_matches_the_formula(tiny):
    k1, b = tiny.params["bm25_k1"], tiny.params["bm25_b"]
    scores, q = tiny.scores("patience", "quran", vectors=False, bridge=False)
    assert len(q) == 1
    lengths = [len(t) for t in tiny.books["quran"]["terms"]]
    avg = sum(lengths) / len(lengths)
    df, n, tf = 1, 4, 2                  # "patience" is twice in quran 1:4 only ("patient" is another word)
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[3] / avg))
    assert scores[3].item() == pytest.approx(expected, rel=1e-5)
    assert scores[:3].abs().sum() == 0


def test_bm25_idf_is_counted_inside_each_book(tiny):
    # "soul" is in 1 of 3 Gita verses, 1 of 4 Quran verses and no Bible verse.
    t = tiny.term_of("soul")
    assert tiny.books["gita"]["idf"][t].item() == pytest.approx(math.log(1 + 2.5 / 1.5))
    assert tiny.books["quran"]["idf"][t].item() == pytest.approx(math.log(1 + 3.5 / 1.5))
    assert tiny.books["bible"]["idf"][t].item() == pytest.approx(math.log(1 + 3.5 / 0.5))


def test_query_vector_is_unit_and_orthogonal_to_the_shared_direction(tiny):
    v = tiny.query_vector(tiny.expand("soul and death"))
    assert v.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert (v @ tiny.principal).item() == pytest.approx(0.0, abs=1e-5)
    assert tiny.query_vector({}).abs().sum() == 0


def test_filler_words_never_become_terms(tiny):
    for w in ["the", "thy", "unto", "is", "shall", "not", "cometh"]:
        assert tiny.term_of(w) is None, w
    assert tiny.term_of("patient") is not None and tiny.term_of("patient") != tiny.term_of("patience")
