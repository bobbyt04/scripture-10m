"""The site contract since round 7: type, then pick a topic.

suggest() is the only free-text entry point the site uses and returns topics,
never verses; topic() returns the verses of the topic a visitor taps. These
tests check that suggest() can never leak a verse, that the safety rules hold
for it, and that topic() follows the hand-set order.
"""

import json
import random

import pytest

from scripture.search import BOOKS, Searcher
from search import build

PROBES = json.loads((build.ROOT / "evals" / "search_probes.json").read_text())
JUDGE6 = PROBES["judge6_held_out"]
CASES = build.load_json(build.ROOT / "evals" / "search_cases.json")
SAFETY = build.load_json(build.SAFETY)
NEVER = set(SAFETY["suggest"]["never_after_signal"])


@pytest.fixture(scope="module")
def s():
    return Searcher.load()


@pytest.fixture(scope="module")
def leak(s):
    """Every pool ref and verse text, to check a suggest() result contains none of them."""
    refs = {v["ref"] for b in BOOKS for v in s.books[b]["verses"]}
    texts = {v["text"] for b in BOOKS for v in s.books[b]["verses"]}
    return refs, texts


def check_suggest(s, leak, q, res):
    assert set(res) == {"category", "note", "topics"}, q
    assert res["category"] in ("crisis", "sensitive", "general", "none"), q
    assert len(res["topics"]) <= SAFETY["suggest"]["max_topics"]
    ids = [t["id"] for t in res["topics"]]
    assert len(set(ids)) == len(ids)
    for t in res["topics"]:
        assert set(t) == {"id", "label", "blurb"} and t["label"] and t["blurb"], (q, t)
        assert s.topic_count(t["id"], res["category"] == "sensitive") > 0, (q, t)     # a tap always shows something
    if res["category"] == "crisis":
        assert res["topics"] == [] and res["note"] == s.crisis_message
    else:
        assert res["note"] is None
    if res["category"] == "none":
        assert res["topics"] == []
    if res["category"] in ("general", "sensitive") and s.suggest_signal(q):
        assert not set(ids) & NEVER and len(ids) <= SAFETY["suggest"]["max_topics_after_signal"], (q, ids)
    # Never a verse: no ref and no verse text anywhere in the result.
    dumped = json.dumps(res, ensure_ascii=False)
    refs, texts = leak
    strings = [res["category"], res["note"] or ""] + [x for t in res["topics"] for x in (t["id"], t["label"], t["blurb"])]
    assert not any(r == x or f'"{r}"' in dumped for r in refs for x in strings), q
    assert not any(v in dumped for v in texts), q


# ----- structure -----

def test_suggest_returns_topics_and_never_verses(s, leak):
    for q in ["I got laid off and I'm scared about money", "I want to die", "I'm gay", "xyzzy", "my mom passed away"]:
        check_suggest(s, leak, q, s.suggest(q))


def test_suggest_categories(s):
    assert s.suggest("I want to die")["category"] == "crisis"
    assert s.suggest("my husband hits me")["category"] == "crisis"
    gay = s.suggest("I'm gay and my parents rejected me")
    assert gay["category"] == "sensitive"
    assert [t["label"] for t in gay["topics"]] == ["You are not alone", "You are of worth", "Loved and cared for"]
    money = s.suggest("I got laid off and I'm scared about money")
    assert money["category"] == "general" and [t["id"] for t in money["topics"]] == ["money_worry", "fear_anxiety"]
    assert s.suggest("xyzzy") == {"category": "none", "note": None, "topics": []}
    assert s.suggest("") == {"category": "none", "note": None, "topics": []}
    assert [t["id"] for t in s.suggest("my mom passed away")["topics"]] == ["grief"]


def test_suggest_typos_non_ascii_and_the_500_character_cap(s):
    assert [t["id"] for t in s.suggest("i'm so lonley these days")["topics"]] == ["loneliness"]
    assert [t["id"] for t in s.suggest("我很难过 lonely")["topics"]] == ["loneliness"]
    assert "money_worry" in [t["id"] for t in s.suggest("naïve café owner, so stressed about money")["topics"]]
    assert s.suggest("i want to kil myself")["category"] == "crisis"
    assert s.suggest("x " * 300 + "I want to die")["category"] == "none"           # beyond 500 characters: not read
    assert s.suggest("x " * 300 + "I'm so lonely") == s.suggest("x " * 300)
    assert s.suggest("🔫")["category"] == "crisis" and s.suggest("🙂🔫")["category"] == "crisis"


def test_sensitive_suggestions_only_offer_topics_with_sensitive_verses(s):
    for q in ["I'm gay", "I lost the pregnancy last week", "I'm blind", "I'm the only brown kid in my class",
              "my wife wants a divorce and i'm crushed", "I have an eating disorder"]:
        res = s.suggest(q)
        assert res["category"] == "sensitive" and res["topics"], q
        for t in res["topics"]:
            assert s.topic_count(t["id"], sensitive=True) > 0, (q, t)
            gentle = s.theme_info[t["id"]].get("gentle_label")
            assert t["label"] == (gentle or s.theme_info[t["id"]]["label"]), (q, t)


def test_suppression_after_a_signal_word(s):
    # The gate does not fire for these, but each has a signal word: no courage, love, friendship, marriage,
    # family or forgiveness, and at most two topics.
    for q in ["i'm 14 and my best friend stopped talking to me", "my coach wants to be my friend",
              "i need the courage to tell my family", "my friend told me a secret and now i feel guilty",
              "i love my family but i feel so alone at home", "i want to forgive him for hurting me"]:
        res = s.suggest(q)
        assert res["category"] != "crisis", q
        ids = [t["id"] for t in res["topics"]]
        assert not set(ids) & NEVER and len(ids) <= 2, (q, ids)
    # Without a signal word the same topics are offered as usual.
    assert [t["id"] for t in s.suggest("my best friend just stopped talking to me")["topics"]] == ["friendship"]
    assert "forgiveness" in [t["id"] for t in s.suggest("I want to forgive my brother")["topics"]]


def test_fuzz_suggest_never_leaks_a_verse(s, leak):
    """4,000 random queries built from theme phrases, crisis and sensitive phrases, pool notes, verse refs and texts."""
    rnd = random.Random(1)
    themes = build.load_json(build.THEMES)
    theme_words = [p.replace("*", "").replace("|", " ") for k, v in themes.items() if not k.startswith("_") for p in v["phrases"]]
    risky = [p.replace("*", "").split("|")[0] for ps in SAFETY["crisis"].values() for p in ps] + \
        [p.replace("*", "") for d in SAFETY["sensitive"].values() for p in d["phrases"]] + \
        [p.replace("*", "").replace("|", " ") for p in SAFETY["suggest"]["signals"]]
    notes = [w for b in BOOKS for v in s.books[b]["verses"][:60] for w in v["why"].split()]
    verse_bits = [v["ref"] for b in BOOKS for v in s.books[b]["verses"][:40]] + \
        [v["text"][:40] for b in BOOKS for v in s.books[b]["verses"][:20]]
    sources = [theme_words, risky, notes, verse_bits, ["", "!!!", "12345", "’", "I'm", "can't", "🔫", "é", "14"]]
    seen = set()
    for _ in range(4000):
        parts = [rnd.choice(rnd.choice(sources)) for _ in range(rnd.randint(0, 6))]
        q = rnd.choice([" ", ", ", ". ", " and "]).join(parts)
        if rnd.random() < 0.2:
            q = q.upper()
        res = s.suggest(q)
        # A query may quote a verse; the result must still contain no verse ref or text of its own.
        check_suggest(s, leak, q, res)
        seen.add(res["category"])
    assert seen == {"crisis", "sensitive", "general", "none"}


# ----- judge 6's messages (spent after its one held-out run) as regression tests for suggest() -----

@pytest.mark.parametrize("q", [p["text"] for p in JUDGE6["crisis"]])
def test_judge6_crisis_messages_get_the_crisis_note(s, leak, q):
    res = s.suggest(q)
    assert res["category"] == "crisis", q
    check_suggest(s, leak, q, res)


@pytest.mark.parametrize("q", JUDGE6["bereavement"])
def test_judge6_bereaved_messages_are_offered_grief(s, leak, q):
    res = s.suggest(q)
    assert res["category"] in ("general", "sensitive"), q
    assert res["topics"] and res["topics"][0]["id"] == "grief", (q, res["topics"])
    check_suggest(s, leak, q, res)


@pytest.mark.parametrize("q", JUDGE6["innocent"] + JUDGE6["sensitive"]
                         + [x["query"] for x in CASES["judge6_blind_hand_judged"]["situations"]])
def test_judge6_innocent_sensitive_and_everyday_messages_are_not_crisis(s, leak, q):
    res = s.suggest(q)
    assert res["category"] != "crisis", q
    check_suggest(s, leak, q, res)


def test_judge6_first_run_numbers_are_kept_as_history():
    fr = JUDGE6["first_run"]
    assert (fr["crisis"], fr["crisis_caught"], fr["crisis_missed_shown"]["harmful"]) == (160, 107, 7)
    assert (fr["innocent_false_alarms"], fr["bereavement_false_alarms"], fr["sensitive_routed"]) == (3, 0, 28)


def test_judge7_first_run_numbers_are_kept_as_history():
    fr = PROBES["judge7_held_out"]["first_run"]
    assert (fr["crisis"], fr["crisis_caught"], fr["crisis_missed_shown"]["harmful"]) == (161, 125, 4)
    assert (fr["innocent_false_alarms"], fr["bereavement_false_alarms"], fr["sensitive_routed"]) == (4, 0, 19)


def test_if_the_gate_missed_any_stored_crisis_message_no_risky_topic_would_be_offered(s):
    """Every crisis message of every probe set, as if the gate had missed it: never courage, love, friendship,
    marriage, family or forgiveness. (These sets were used to write the signal list: development numbers.
    judge7_held_out is still held out and not used for development, so it is left out here.)"""
    from scripture.search import phrase_starts, words
    for name, d in PROBES.items():
        if not isinstance(d, dict) or "crisis" not in d or name == "judge7_held_out":
            continue
        for p in d["crisis"]:
            q = p["text"]
            toks = words(q)
            topic = next((t for t, ps in s.sensitive.items() if any(phrase_starts(toks, ph) for ph in ps)), None)
            res = s.offer(q, "sensitive" if topic else "general", topic)
            assert not {t["id"] for t in res["topics"]} & NEVER, (name, q, res["topics"])


# Round 7 paraphrases of judge 6's misses, and near misses that must not fire.
PARAPHRASED_CRISIS = [
    "what is even the point of any of it anymore", "honestly i'm done with life", "i hope i get sick and never recover",
    "i wish i had never been born, everyone would be happier", "would anyone care if i just stopped existing",
    "life is too heavy and i want to put it down", "how tall is the parking garage near my house, high enough?",
    "i gave away my dog and my guitar, its time", "i saved up enough pills, just need the nerve",
    "i might drive into a tree on purpose tonight", "tell my mom i'm sorry, i couldn't keep going",
    "this is my final message before i go", "if my soul is eternal why not leave this body now",
    "i want to go to heaven early, it has to be better than this", "allah i want to come home to you now",
    "i'm going to give up this life tonight", "i dont wanna be alive no more", "i shud disapear for good", "🪢 tonight",
    "ready to log out of life", "my wrists are covered in cuts and i keep adding more",
    "how can i stop him getting violent with me", "do i just take the beatings and stay quiet",
    "my husband controls my money and gives me an allowance", "i'm not allowed to have my own bank account",
    "my boyfriend says i'm crazy whenever i remember things he did", "my family will send me back to get married if i disobey",
    "my parents are forcing me to marry my cousin", "my coach buys me gifts and says our relationship is special",
    "i'm 13 and a man online keeps sending me gifts", "he said if i loved him i'd send him pics",
    "a guy from discord wants me to come to his house and not tell anyone", "my youth leader says we are soulmates",
    "my tutor texts me late at night", "is it ok to date a 25 year old if i'm 15", "he told me to delete our chats",
    "i'm going to make him pay, he won't see it coming", "the voices tell me to hurt my neighbour",
    "they deserve to suffer and i'm going to make sure they do", "i made a list of everyone who hurt me",
    "is honour killing ever justified", "i took all of them, waiting for them to kick in",
    "i don't care if i don't wake up tomorrow", "planning so the kids and i are together forever at the same time",
    "my children would be better off in heaven", "i've decided. tonight. goodbye", "this is my last search ever",
    "holding my breath underwater in the bath to see how long it takes to pass out",
    "i'd be doing everyone a favour by disappearing", "what song should play at my funeral, it'll be soon",
]
PARAPHRASED_SAFE = [
    "what's the point of this meeting", "i'm done trying to lose weight this way", "i'm so done with this job",
    "i hope my mom recovers soon", "i stopped everything to help my friend move", "my backpack is too heavy",
    "how high is the hotel ceiling", "i gave my dog to my sister while i'm on holiday", "which train is fastest to london",
    "i stocked up on cold medicine for winter", "i need the courage to ask her out", "my dad nearly drove off the road in the snow",
    "tell my mom i'm sorry i'm late for dinner", "i couldn't keep going with the marathon",
    "heaven is better than this pizza", "i want to leave my life in the city and move to the country",
    "logging off for the day", "i have scars on my arms from surgery", "my son plays violent video games",
    "my heart keeps beating fast before exams", "i keep beating myself up about it", "my brother keeps beating me at chess",
    "i have to ask permission from my boss to leave early", "my dad gives me pocket money",
    "my friend says i'm crazy for running a marathon", "i met my wife online and want to meet her parents",
    "i sold my bike online and the buyer wants to meet in person", "my teacher gave me a gift for graduation",
    "i'm 15 and my 18 year old brother teases me", "i need to delete the old messages from my work phone",
    "god wants me to be kind to my neighbour", "i made a list of people to invite to my wedding",
    "i took the kids to the park and they fell asleep", "i don't care if i wake up late tomorrow",
    "we're all together at the same time for christmas dinner", "i've decided to move to london", "i've decided what to study",
    "i was holding my breath underwater at swimming practice", "do the world a favor and recycle",
    "i'm planning my wedding for next year", "my code keeps crashing and i want to throw my laptop out the window",
    "i want to start a new chapter of my life", "my dog destroyed the couch, i could kill him",
    "i could kill him, my dog ate my homework", "i almost drowned at the beach",
]


@pytest.mark.parametrize("q", PARAPHRASED_CRISIS)
def test_round_7_paraphrases_fire(s, q):
    assert s.suggest(q)["category"] == "crisis", q


@pytest.mark.parametrize("q", PARAPHRASED_SAFE)
def test_round_7_near_misses_do_not_fire(s, q):
    assert s.suggest(q)["category"] != "crisis", q


# ----- topic() and topics() -----

def test_topic_follows_the_hand_set_order_and_says_when_a_book_has_nothing(s):
    for theme in s.theme_info:
        res = s.topic(theme)
        assert set(res) == {"id", "label", "blurb", "note", "sensitive", "books"} and res["id"] == theme
        old = s.by_theme(theme, k=1000)
        for book in BOOKS:
            got = res["books"][book]
            assert [v["ref"] for v in got["verses"]] == [h["ref"] for h in old[book]], (theme, book)
            assert len(got["verses"]) == s.topic_count(theme, False, book)
            assert (got["nothing"] is None) == bool(got["verses"]), (theme, book)
            for v in got["verses"]:
                pv = s.books[book]["verses"][s.books[book]["index"][v["ref"]]]
                assert theme in v["themes"] and v["text"] == pv["text"] and v["why"] == pv["why"]
    grief = s.topic("grief", k=2)
    assert grief["note"] == s.theme_info["grief"]["note"]
    assert all(len(grief["books"][b]["verses"]) <= 2 for b in BOOKS)
    assert grief["books"]["gita"]["verses"] == [] and "Bhagavad Gita" in grief["books"]["gita"]["nothing"]
    first = s.topic("grief")["books"]["bible"]["verses"]
    assert [v["priority"] for v in first[:3]] == [1, 2, 3]


def test_topic_sensitive_only_shows_the_sensitive_set(s):
    excluded = {(b, r) for d in SAFETY["sensitive"].values() for b, refs in d.get("exclude", {}).items() for r in refs}
    for theme in s.theme_info:
        res = s.topic(theme, sensitive=True)
        assert res["sensitive"] is True
        for book in BOOKS:
            for v in res["books"][book]["verses"]:
                assert s.books[book]["verses"][s.books[book]["index"][v["ref"]]]["sensitive"], (theme, book, v["ref"])
                assert (book, v["ref"]) not in excluded
    assert s.topic("self_worth", sensitive=True)["label"] == "You are of worth"


def test_topic_rejects_bad_input(s):
    with pytest.raises(ValueError):
        s.topic("no_such_topic")
    for k in (0, -1, 1.5, True, "3"):
        with pytest.raises(ValueError):
            s.topic("grief", k=k)


def test_topics_lists_every_topic_with_a_verse(s):
    listed = s.topics()
    ids = [t["id"] for t in listed]
    assert ids == [t for t in s.theme_info if s.topic_count(t)]
    assert "marriage" not in ids                                     # no verse left in any book since round 7
    for t in listed:
        assert set(t) == {"id", "label", "blurb", "verses", "sensitive_verses"}
        assert sum(t["verses"].values()) > 0
        assert all(t["sensitive_verses"][b] <= t["verses"][b] for b in BOOKS)


def test_every_theme_has_a_hand_written_label_and_blurb():
    themes = build.load_json(build.THEMES)
    gentle_needed = {t for d in SAFETY["sensitive"].values() for t in d["themes"]}
    for k, v in themes.items():
        if k.startswith("_"):
            continue
        assert v["label"] and len(v["label"]) <= 30 and v["blurb"].endswith(".") and len(v["blurb"]) <= 60, k
        if k in gentle_needed:
            assert v["gentle_label"] and v["gentle_blurb"].endswith("."), k


def test_signal_phrases_are_well_formed():
    from scripture.search import words
    for p in SAFETY["suggest"]["signals"]:
        assert p == p.lower() and p.strip() == p and "  " not in p, p
        for slot in p.split():
            for alt in slot.split("|"):
                assert alt == "*" or words(alt.rstrip("*")) == [alt.rstrip("*")], (p, alt)
    assert NEVER == {"courage", "love_compassion", "friendship", "marriage", "family_parents", "forgiveness"}
