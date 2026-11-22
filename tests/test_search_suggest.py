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
JUDGE7 = PROBES["judge7_held_out"]
CASES = build.load_json(build.ROOT / "evals" / "search_cases.json")
SAFETY = build.load_json(build.SAFETY)
NEVER = set(SAFETY["suggest"]["never_after_signal"])
THEMES = build.load_json(build.THEMES)
COMFORT = {k for k, v in THEMES.items() if not k.startswith("_") and v["kind"] == "comfort"}
SELF_DIRECTED = {k for k, v in THEMES.items() if not k.startswith("_") and v["kind"] == "self_directed"}


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
    if s.comfort_cue(q):
        assert set(ids) <= COMFORT, (q, ids)
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
    assert [t["label"] for t in gay["topics"]] == ["You are not alone", "You are of worth"]    # "parents": comfort only
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
    # Without a signal word (or another person) the same topics are offered as usual.
    assert "forgiveness" in [t["id"] for t in s.suggest("how do i forgive myself")["topics"]]


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


def as_if_missed(s, q):
    """What suggest() would offer if the crisis gate had not fired (its sensitive routing kept)."""
    from scripture.search import phrase_starts, words
    toks = words(q)
    topic = next((t for t, ps in s.sensitive.items() if any(phrase_starts(toks, ph) for ph in ps)), None)
    return s.offer(q, "sensitive" if topic else "general", topic)


def test_if_the_gate_missed_any_stored_crisis_message_no_risky_topic_would_be_offered(s):
    """Every crisis message of every probe set (judge 7's included), as if the gate had missed it: never courage,
    love, friendship, marriage, family or forgiveness; and when the message names another person, an event done
    to the writer, mental health or anger at God, never a self-directed topic (anger, guilt, self-control,
    patience ...), only comfort ones. (These sets were used to write the lists: development numbers.)"""
    cued = 0
    for name, d in PROBES.items():
        if not isinstance(d, dict) or "crisis" not in d:
            continue
        for p in d["crisis"]:
            q = p["text"]
            ids = {t["id"] for t in as_if_missed(s, q)["topics"]}
            assert not ids & NEVER, (name, q, ids)
            if s.comfort_cue(q):
                cued += 1
                assert not ids & SELF_DIRECTED, (name, q, ids)
    assert cued > 300


def test_every_sensitive_topic_keeps_a_comfort_topic_with_sensitive_verses(s):
    # So a sensitive message that names another person is still offered something gentle.
    for topic, themes in s.sensitive_themes.items():
        assert any(t in COMFORT and s.topic_count(t, sensitive=True) for t in themes), topic


def test_topic_kinds_split_every_topic_in_two():
    assert COMFORT | SELF_DIRECTED == {k for k in THEMES if not k.startswith("_")} and not COMFORT & SELF_DIRECTED
    assert COMFORT == {"grief", "fear_anxiety", "loneliness", "illness", "doubt_faith", "hope", "self_worth"}
    assert NEVER <= SELF_DIRECTED
    assert {"anger", "guilt_repentance", "temptation_self_control", "patience", "pride_humility", "forgiveness",
            "speech_gossip", "work_effort", "suffering_endurance", "peace_of_mind"} <= SELF_DIRECTED


# Words that tell a reader to endure, wait, submit, keep quiet, forgive or try harder. A tripwire, not a proof:
# every verse of every comfort topic was read by hand; a new verse with one of these words fails here until a
# person has read it and listed it below.
DIRECTIVE = ["endure", "endureth", "endured", "submit", "obey", "obedient", "forgive", "forgiveth", "forgiving",
             "silent", "silence", "quiet", "quieted", "patient", "patience", "patiently", "longsuffering", "bear",
             "beareth", "bearing", "wait", "waiteth", "waited", "strive", "effort", "practice", "diligent", "diligence",
             "repent", "repenteth", "confess", "humble", "humbly", "meek", "deserve", "punish", "duty", "serve",
             "servant", "try", "harder", "deed", "deeds", "good works", "courage", "be still", "hold thy peace",
             "keep thy", "keep the covenant", "oath", "oaths"]
DIRECTIVE_OK = {   # (book, ref): why the word is not an instruction to the reader
    ("gita", "2:40"): "'no loss of effort': nothing done is wasted",
    ("bible", "Isaiah 61:1"): "'good tidings unto the meek': good news for the downtrodden",
    ("bible", "Isaiah 46:4"): "'I have made, and I will bear': God carries the reader",
    ("bible", "Psalms 126:6"): "'bearing precious seed ... shall doubtless come again with rejoicing'",
    ("bible", "Isaiah 40:31"): "'they that wait upon the LORD shall renew their strength': a promise of strength",
    ("bible", "Micah 7:7"): "'I will wait for the God of my salvation: my God will hear me': a promise of being heard",
    ("bible", "Psalms 130:5"): "'my soul doth wait, and in his word do I hope': hope, said in the first person",
    ("bible", "Psalms 138:8"): "'thy mercy, O LORD, endureth for ever': it is God's mercy that endures",
    ("bible", "Mark 4:39"): "'Peace, be still': said to the storm, which then calms",
}


def test_comfort_topics_have_no_directive_verses(s):
    import re
    pats = [re.compile(r"\b" + re.escape(w) + r"\b", re.I) for w in DIRECTIVE]
    found = set()
    for theme in COMFORT:
        for book, rows in s.topic(theme)["books"].items():
            for v in rows["verses"]:
                hits = [w for w, pat in zip(DIRECTIVE, pats) if pat.search(v["text"])]
                if hits:
                    found.add((book, v["ref"]))
                    assert (book, v["ref"]) in DIRECTIVE_OK, (theme, book, v["ref"], hits, v["text"])
    assert found == set(DIRECTIVE_OK)                  # an exception no longer needed is taken off the list


def test_a_message_naming_another_person_gets_only_comfort_topics(s):
    # Judge 7's four harmful chains: the gate now catches them, and even if it missed them none would be offered
    # anger, guilt or self-control.
    for q in ["he only gets rough when i make him angry, so i should try harder right?",
              "how do i be a better submissive wife so he stops getting angry and leaving bruises",
              "i said no but he didnt stop. was it my fault for drinking?",
              "something happened at the sleepover and i feel dirty and cant stop shaking"]:
        assert s.suggest(q)["category"] == "crisis", q
        assert s.comfort_cue(q) and not {t["id"] for t in as_if_missed(s, q)["topics"]} - COMFORT, q
    # The accepted cost: an everyday message about another person no longer gets Anger; with nothing left the
    # category is "none" and the site shows the full list.
    assert s.suggest("my brother annoys me, how do i stay calm") == {"category": "none", "note": None, "topics": []}
    assert [t["id"] for t in s.suggest("how do i stop losing my temper")["topics"]] == ["anger"]
    # Judge 7's two judgmental offers.
    dep = s.suggest("i'm depressed and can't get out of bed")
    assert dep["topics"] and {t["id"] for t in dep["topics"]} <= COMFORT
    god = s.suggest("i'm angry at god for letting this happen")
    assert god["topics"] and {t["id"] for t in god["topics"]} <= COMFORT


# ----- judge 7's messages (spent after its one held-out run) as regression tests for suggest() -----

@pytest.mark.parametrize("q", [p["text"] for p in JUDGE7["crisis"]])
def test_judge7_crisis_messages_get_the_crisis_note(s, leak, q):
    res = s.suggest(q)
    assert res["category"] == "crisis", q
    check_suggest(s, leak, q, res)


@pytest.mark.parametrize("q", JUDGE7["bereavement"])
def test_judge7_bereaved_messages_are_not_crisis(s, leak, q):
    res = s.suggest(q)
    assert res["category"] != "crisis", q
    check_suggest(s, leak, q, res)


@pytest.mark.parametrize("q", JUDGE7["innocent"] + JUDGE7["sensitive"]
                         + [x["query"] for x in CASES["judge7_blind_hand_judged"]["situations"]])
def test_judge7_innocent_sensitive_and_everyday_messages_are_not_crisis(s, leak, q):
    res = s.suggest(q)
    assert res["category"] != "crisis", q
    check_suggest(s, leak, q, res)


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


# Round 8 paraphrases of judge 7's misses ("finish my life" in any order and non-native wording, scriptural
# suicides with "me too / should i", ...), and near misses that must not fire.
PARAPHRASED_CRISIS_8 = [
    "i want to finish my life", "my life i want to finish it", "please i want finish life", "i want my life to end",
    "i decide to finish my life tonight", "how to finish my own life", "i want to stop my life",
    "my life finish now please", "my english is bad, i want to close my life", "i think to finish my life soon",
    "saul fell on his sword, should i do the same", "judas did it, maybe i should too",
    "samson took his own life, can god forgive me if i do", "is it ok if i do what judas did",
    "i'm as bad as judas, i should go the same way", "if a truck hit me tomorrow i would not mind",
    "i wouldnt even care if i got hit by a bus", "i'm not afraid of death, i'm afraid of another 40 years like this",
    "i can't do another years like this", "when am i allowed to stop, i've suffered for years",
    "the pain will end and i know only one way out", "i know which bridge already",
    "tonight is the night, i have everything i need", "whoever finds this, i'm sorry",
    "if anyone finds this, it's not your fault", "see you all on the other side",
    "my last night on earth, give me a verse before the end", "i can't hold on much longer",
    "i punish myself when i fail", "i don't deserve food so i stopped eating", "i starve myself because i'm fat",
    "my thighs are covered in marks and i hide them in summer", "he gets rough with me when i talk back",
    "he only gets physical when i make him mad, it's my fault", "she leaves bruises on my arms",
    "covering my bruises before work again", "my stepdad doesn't feed me when i'm bad",
    "they told me to say i fell down the stairs", "my dad uses his belt on me, spare the rod he says",
    "i told him no but he kept going", "i said stop but he did not stop",
    "something happened at the party and i feel dirty", "he bought me a drink and i woke up and can't remember",
    "a guy online says he'll buy me a gift if i send a pic", "he keeps asking what i'm wearing",
    "i want to make my ex bleed", "i'm scared of what i might do to my baby",
    "if she leaves me she will regret it, i'll kill her",
    "thinking of bringing a gun to work and making them all feel what i feel",
    "my boyfriend says if i loved him i'd cut off my friends",
    "mom gets drunk and i hide in my room until she passes out",
]
PARAPHRASED_SAFE_8 = [
    "i want to finish my degree so my life gets better", "when i finish school my life will change",
    "i want to finish my life's work before i retire", "we had to stop the life support for my dad",
    "who was samson in the bible", "why did judas betray jesus", "the story of saul and david",
    "if it rains tomorrow i would not mind", "another year like this at work and i'll quit",
    "am i allowed to stop work early on fridays", "there's only one way to find out",
    "which bridge is best to cross into the city", "tonight's the night! prom!", "if anyone finds my wallet please call me",
    "see you on the other side of the exams", "final night of our vacation in rome",
    "i can't hold on much longer, i need the bathroom lol", "my mom bruises easily since her surgery",
    "my dad got a black eye at football", "the sea gets rough in winter",
    "i need an extension cord for the christmas lights", "i said no to a second piece of cake",
    "something happened at work and i'm stressed", "i woke up with my clothes all wrinkled",
    "send me a pic of your new puppy", "what are you wearing to the wedding", "i made my knee bleed falling off my bike",
    "my baby won't stop crying and i'm exhausted", "my grandma will not live to see my wedding and that makes me sad",
    "bring something in tomorrow for the bake sale", "i was so hungry i could die", "the movie was so sad i wanted to die",
    "i hit my brother in a pillow fight and he cried lol", "my laptop crashed and i lost my essay, kill me now",
]


@pytest.mark.parametrize("q", PARAPHRASED_CRISIS_8)
def test_round_8_paraphrases_fire(s, q):
    assert s.suggest(q)["category"] == "crisis", q


@pytest.mark.parametrize("q", PARAPHRASED_SAFE_8)
def test_round_8_near_misses_do_not_fire(s, q):
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
    cues = [x for g in SAFETY["suggest"]["comfort_only_cues"].values() for x in g]
    for p in SAFETY["suggest"]["signals"] + cues:
        assert p == p.lower() and p.strip() == p and "  " not in p, p
        for slot in p.split():
            for alt in slot.split("|"):
                assert alt == "*" or words(alt.rstrip("*")) == [alt.rstrip("*")], (p, alt)
    assert NEVER == {"courage", "love_compassion", "friendship", "marriage", "family_parents", "forgiveness"}
