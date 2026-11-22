"""Adversarial queries for the public search: crisis, identity, abuse and odd inputs.

For every query: the category is as expected, a crisis returns no verses, and
nothing returned is on the blocklist or contains a hard-banned word.
"""

import json
import random
import re

import pytest

from scripture.search import BOOKS, Searcher
from search import build


@pytest.fixture(scope="module")
def s():
    return Searcher.load()


CRISIS = [
    "I've been cutting myself", "I want to die", "my husband hits me", "should I hit my child",
    "I'm thinking about suicide", "I was sexually abused as a child", "I was raped", "I want to kill myself",
    "I want to end my life", "kms", "I just want to end it all", "my wife hits me", "my stepfather abuses me",
    "he beats me when he is drunk", "I hate my neighbor and want to hurt him", "should I spank my kids",
    "I think about hurting myself", "i feel suicidal", "sucide", "I wish I was dead", "I'm better off dead",
    "my boyfriend hit me last night", "my dad beats us", "I'm afraid to go home", "I was molested",
    "I want to hurt someone", "my husband won't let me work", "nobody would care if I took my own life",
    "I don't want to be alive anymore", "I self harm",
]
SENSITIVE = [
    "I had an abortion and feel guilty", "my friend is a Christian and I'm Muslim", "I'm Jewish",
    "I'm an atheist", "I'm blind", "I'm a widow", "I'm black and face racism", "I'm transgender",
    "I'm a lesbian", "I'm worried about my sister who has an eating disorder", "I want to lose weight",
    "my daughter is marrying someone of another faith", "I'm gay", "I'm gay and my parents rejected me",
    "my best friend is Jewish", "I'm Hindu and I'm curious about other religions", "I'm pregnant and not married",
    "I'm disabled and use a wheelchair", "my baby died", "I relapsed after two years sober", "I'm Muslim and proud",
    "I had a miscarriage", "I'm autistic", "I face discrimination at work", "my son came out as gay",
]
GENERAL = [
    "I got laid off and I'm scared about money", "my dog died", "I'm a single mom", "I want revenge on my ex",
    "I'm sexually attracted to my coworker", "I love cooking for my family", "I passed my driving test",
    "high school was hard", "I feel ugly", "I'm being bullied online", "I'm fasting for Ramadan",
    "I feel like a slave at my job", "I'm going on a trip with my friends", "I'm starting a new business",
    "I was moved to tears", "I'm all fired up", "calm down", "my friend let me down", "good news, I got the job",
    "I'm so tired", "I feel lost", "my child disobeys me", "how should I treat my wife",
    "my parents are getting divorced", "I'm going to church on Sunday",
]
NO_MATCH = ["I'm just kidding", "", "!!!", "xyzzy", "   ", "12345", "the and of"]

# Words that must never appear in a verse shown for a situation.
BANNED = re.compile(r"\b(whore\w*|harlot\w*|crucif\w*|dung|rape\w*|ravish\w*|scourg\w*|stoned|infidels?|idolat\w*|"
                    r"sodomites?|incest|concubines?|behead\w*|massacre\w*|foreskins?|eunuchs?|menstru\w*|"
                    r"slay not your children|buried alive)\b", re.IGNORECASE)


def check_hits(s, res):
    for book in BOOKS:
        for h in res["results"][book]:
            b = s.books[book]
            assert not b["blocked"][b["index"][h["ref"]]], (book, h["ref"])
            assert not BANNED.search(h["text"]), (book, h["ref"], h["text"])
            assert h["score"] > 0


def test_there_are_at_least_60_adversarial_queries():
    assert len(CRISIS) + len(SENSITIVE) + len(GENERAL) + len(NO_MATCH) >= 60


@pytest.mark.parametrize("q", CRISIS)
def test_crisis_queries_return_no_verses(s, q):
    res = s.search(q, k=3)
    assert res["category"] == "crisis", (q, res["category"])
    assert all(res["results"][b] == [] for b in BOOKS)


@pytest.mark.parametrize("q", SENSITIVE)
def test_sensitive_queries_return_only_gentle_verses(s, q):
    res = s.search(q, k=3)
    assert res["category"] == "sensitive", (q, res["category"])
    for book in BOOKS:
        assert len(res["results"][book]) == 3
        for h in res["results"][book]:
            assert s.books[book]["gentle"][s.books[book]["index"][h["ref"]]]
    check_hits(s, res)


@pytest.mark.parametrize("q", GENERAL)
def test_general_queries_return_only_allowed_verses(s, q):
    res = s.search(q, k=3)
    assert res["category"] == "general", (q, res["category"])
    check_hits(s, res)


@pytest.mark.parametrize("q", NO_MATCH)
def test_queries_without_known_words_return_nothing(s, q):
    res = s.search(q, k=3)
    assert res["no_match"] and res["category"] == "general"
    assert all(res["results"][b] == [] for b in BOOKS)


def test_gay_does_not_match_gay_clothing(s):
    res = s.search("I'm gay", k=10)
    assert "James 2:3" not in [h["ref"] for h in res["results"]["bible"]]


def test_reported_harmful_results_are_gone(s):
    cases = {
        "I've been cutting myself": None, "I want to die": None, "my husband hits me": None,
        "I had an abortion and feel guilty": ["Exodus 34:7", "4:16"], "should I hit my child": None,
        "my friend is a Christian and I'm Muslim": ["5:51"], "I'm Jewish": ["9:34", "Titus 1:14"],
        "I'm an atheist": ["2:88"], "I'm blind": ["17:72", "2:18"], "I'm a widow": ["Psalms 109:9"],
        "I'm black and face racism": ["43:17"], "I got laid off and I'm scared about money": ["17:31"],
    }
    for q, bad in cases.items():
        res = s.search(q, k=10)
        got = [h["ref"] for b in BOOKS for h in res["results"][b]]
        if bad is None:
            assert got == [], q
        else:
            assert not set(bad) & set(got), (q, got)


def test_every_bridge_key_as_a_query_returns_only_allowed_verses(s):
    keys = [k for k in build.load_json(build.BRIDGE) if not k.startswith("_")]
    for q in random.Random(0).sample(keys, 120):
        check_hits(s, s.search(q, k=3))


def test_safety_lists_are_well_formed():
    safety = build.load_json(build.SAFETY)
    for cat in ("crisis", "sensitive"):
        for topic, phrases in safety[cat].items():
            assert phrases, topic
            for p in phrases:
                assert p == p.lower() and p.strip() == p and "  " not in p, p
    for pat in safety["block"]["patterns"]:
        re.compile(pat["regex"])
        assert pat["name"]
