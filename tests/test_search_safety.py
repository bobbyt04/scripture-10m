"""Safety of the public advice search: crisis, sensitive topics, idioms, and the allowlist guarantee.

The search can only return pool verses; a crisis returns none; a sensitive
topic returns only verses of the hand-picked sensitive set. The fuzz test
checks these for thousands of random queries.
"""

import json
import math
import random

import pytest

from scripture.search import BOOKS, Searcher
from search import build


@pytest.fixture(scope="module")
def s():
    return Searcher.load()


@pytest.fixture(scope="module")
def pool_refs():
    return {b: {v["ref"] for v in build.load_pool()[b]} for b in BOOKS}


# Every crisis phrasing from the safety reviews (the second set was missed by the old lists).
CRISIS = [
    "I've been cutting myself", "I want to die", "my husband hits me", "should I hit my child",
    "I'm thinking about suicide", "I was sexually abused as a child", "I was raped", "I want to kill myself",
    "I want to end my life", "kms", "I just want to end it all", "my wife hits me", "my stepfather abuses me",
    "he beats me when he is drunk", "I hate my neighbor and want to hurt him", "should I spank my kids",
    "I think about hurting myself", "i feel suicidal", "sucide", "I wish I was dead", "I'm better off dead",
    "my boyfriend hit me last night", "my dad beats us", "I'm afraid to go home", "I was molested",
    "I want to hurt someone", "my husband won't let me work", "nobody would care if I took my own life",
    "I don't want to be alive anymore", "I self harm",
    "thinking of jumping off the roof", "planning how to end it", "I keep thinking about ending things",
    "I've been cutting again", "I wish I could go to sleep and not wake up", "I've been burning my arms with a lighter",
    "I don't see the point anymore", "life isn't worth living", "I want to disappear forever",
    "I took a bunch of pills last night on purpose", "I hate myself and want it all to stop",
    "no one would miss me if I was gone", "I keep imagining crashing my car on purpose", "i am so done with life",
    "he gets rough with me when he's drunk", "my stepdad comes into my room at night", "my dad hits my mom",
    "he forces me to do sexual things", "I was assaulted at a party",
    "my partner controls my money and won't let me see my friends", "I want to punch my coworker in the face",
    "I'm going to kill that guy", "i dont want to be here anymore", "I want to kil myself",
    # variants: typos, missing apostrophes, slang
    "i want to kill mysef", "im gonna kms", "i wanna die", "i want to unalive myself", "suicdie", "feeling suicidel",
    "i cant go on like this", "i cant do this anymore", "i want to end it all tonight", "thinking about overdosing",
    "i took too many pills", "i want to hang myself", "i want to jump off a bridge", "i wish i was never born",
    "everyone would be better off without me", "there's no reason to live", "nothing to live for anymore",
    "i keep cutting my wrists", "i burn myself to feel something", "my mom hits me", "my uncle touched me inappropriately",
    "my boyfriend choked me", "i'm scared of my husband", "he threatened to kill me", "my teacher molested me",
    "i'm going to shoot up the school", "i want to stab him", "i'm gonna kill my boss", "i want to kill everyone",
    "my girlfriend slapped me", "he won't let me leave the house", "she takes my phone and money",
]
# Grief, fear of death and everyday idioms that must not be read as crisis.
NOT_CRISIS = [
    "my mom died", "my dog was put down", "my mom passed away last month and I can't stop crying",
    "my dad has cancer and I'm scared he'll die", "I'm afraid of dying", "I don't want to die", "my grandpa is dying",
    "I'm killing it at work", "I could murder a pizza", "I'm dead tired", "I bombed my presentation",
    "my brother beat me at chess", "I need to cut back on sugar", "my feet are killing me", "this heat is killing me",
    "I'm dying to see the new movie", "I nearly died laughing", "that dress is to die for", "I burned myself out at work",
    "I cut myself a slice of cake", "I need to cut myself some slack", "I'm going to end it with my girlfriend",
    "we ended things with our old landlord", "my boss is going to kill me if I'm late", "I lost my temper and yelled at my kids",
    "I hate my job", "I'm so done with this job", "I'm sick of waiting", "I've been drinking way too much every night",
    "I want revenge on my ex", "I don't want to hurt my mom's feelings", "I hit the gym every day",
    "it hit me that I was wrong", "hit me up later", "my team beat us at football", "I'm cutting carbs",
    "I'm cutting ties with toxic friends", "I shot my shot and asked her out", "shoot me a text", "I'm drowning in debt",
    "I feel dead inside after the breakup", "my phone died", "my car broke down", "the deadline is killing me",
    "I'm going to crush this exam", "I want to beat my personal record", "I want to kill time before my flight",
    "I'm beat after work", "my sister is a knockout", "the comedian killed it", "I'm hanging out with friends",
    "I'm hanging the pictures", "I took my pills this morning", "I jumped at the chance", "I'm overdoing it at the gym",
    "I bombed the interview", "I want to get away for a weekend trip", "life is short so I'm travelling",
    "I lost my keys", "my dad hit a deer with his car",
]
SENSITIVE = [
    "I'm gay", "I'm gay and my parents rejected me", "I lost the pregnancy last week", "I'm the only brown kid in my class",
    "I had an abortion and feel guilty", "my friend is a Christian and I'm Muslim", "I'm Jewish", "I'm an atheist",
    "I'm blind", "I'm black and face racism", "I'm transgender", "I'm a lesbian", "I have an eating disorder",
    "I want to lose weight", "my daughter is marrying someone of another faith", "I'm pregnant and not married",
    "I'm disabled and use a wheelchair", "I had a miscarriage", "I'm autistic", "I face discrimination at work",
    "my son came out as gay", "people bully me because of my skin", "I feel like I was born in the wrong body",
    "I'm a guy and I'm into guys", "I like girls and I'm a girl", "my parents won't accept me because I'm from a different caste",
    "I'm thinking about leaving my religion", "I'm ending my pregnancy", "I hate my body", "I'm 16 and pregnant",
]

# Two held-out rounds written after the lists were frozen (recall 28/60, then 21/40); added afterwards.
HELD_OUT_CRISIS = [
    'honestly i just want everything to be over', "i've been researching ways to die", 'i bought rope today',
    "i'm going to swallow all my sleeping pills", 'my family would be happier if i was dead',
    "i'm not going to be around much longer", 'tonight is my last night',
    'i wrote letters to everyone saying goodbye', 'i keep scratching my skin until it bleeds',
    'i punch walls until my knuckles bleed', 'i starve myself as punishment', "i've relapsed into self-harm",
    'i stood on the bridge last night', 'i want to stop breathing', 'i wish a car would hit me',
    "i can't keep living like this", 'nobody would notice if i vanished', 'i want to fall asleep forever',
    "what's the most painless way to go", "i'm a burden to everyone and they'd be better off",
    "i hope i don't wake up tomorrow", 'ending my life seems like the only option', "i'm planning my suicide",
    'i think about killing myself every day', "i have a gun and i'm thinking of using it on myself",
    'my husband strangled me last night', 'my dad locks me in my room for days',
    "my boyfriend says he'll kill me if i leave", 'my mom burns me with cigarettes',
    'my brother touches me when nobody is home', "my coach makes me do things i don't want to",
    'i was drugged and woke up somewhere', "my girlfriend hits me when she's angry", 'my parents beat me with a belt',
    "my husband won't let me have any money of my own", "he checks my phone and won't let me go out",
    'my partner threatens to hurt my dog', 'my stepmom slaps me every day',
    "i'm scared my dad will hurt me when he gets home", 'someone at school forced me to have sex',
    'i want to hurt my baby when he cries', "i'm going to beat my wife if she talks back",
    'i want to shoot my classmates', 'i fantasize about killing my neighbor', "i'm going to stab my roommate",
    'i want to burn his house down with him inside', "i'm so angry i could strangle my brother",
    'i want to run him over with my car', "i don't want to exist anymore", 'life is pointless and i want out',
    "i'm thinking about overdosing on my meds", 'i deserve to die', "i'm going to jump",
    "i've been hurting myself again", 'cut myself last night', 'i feel like killing myself', 'suicidal again',
    'i want to kms fr', "i'm done. goodbye everyone", 'everyone hates me and i want to die',
    'i want to go to sleep forever and never wake', "i'm thinking about taking my life", 'no point living anymore',
    "i'd rather be dead", "i've started cutting my legs", 'i hit myself when i mess up',
    'i want to end everything tonight', "i'm going to take all my pills tonight",
    'i looked up how many pills it takes', "i'm standing on the roof right now", 'it would be easier if i died',
    "i can't do life anymore", 'i want to be dead', 'i have nothing left to live for',
    "everyone would be better off if i wasn't here", 'i keep thinking about suicide', 'i feel like ending it',
    "i'm tired of being alive", 'my dad punches me', "my mom's boyfriend touches me", 'my husband raped me',
    'my ex stalks me and threatens me', "i'm being abused at home", 'my father hits me with his belt',
    'my partner chokes me during fights', "i'm afraid my husband will kill me", 'my teacher sends me sexual messages',
    'i want to kill my stepdad', "i'm going to hurt the kid who bullies me",
    'i want to beat up my brother until he bleeds', "i'm thinking about shooting people at work",
    'i keep imagining stabbing my boss', "i'm so done i want to die", 'kill me now i mean it i want to die',
    'i want to cease to exist', 'life has no point and i want to end it', "i overdosed last week and i'd do it again",
    "i'm saving my pills for when i'm ready", "i've been burning myself", 'i self-harmed again',
]
# Innocent sentences written with them (no false alarms in either round).
HELD_OUT_NOT_CRISIS = [
    "i'm dying of laughter", 'this commute is killing me', 'i killed it at the interview',
    "i'd kill for a coffee right now", 'my phone is dead', "i'm dead serious", 'that joke killed',
    "i'm going to crush it tomorrow", 'my plants keep dying', 'i shot a great photo',
    "i'm swamped and drowning in emails", 'she knocked it out of the park', 'i want to quit my job',
    'i need to cut down on caffeine', 'we cut the cake at the wedding', "i'm cutting my losses", 'i burned the toast',
    'i got burned by a bad investment', 'i hit the jackpot', 'my dad hit the roof when he saw my grades',
    'i want to end my subscription', "i'm ending my lease early", "i'm done with my exams",
    "i'm so over this weather", 'i could sleep for a week', "i'm starving, what's for dinner", "i'm hanging in there",
    'this traffic makes me want to scream', "i'm bored to death", "i'd die of embarrassment",
    "he's a killer salesman", 'my legs are dead after the run', 'my battery died', "i'm beat",
    'we beat the other team', "i'm taking pills for my allergies", 'i overdid it at dinner',
    "i'm jumping into a new project", 'i jumped out of my seat', 'i took a leap of faith',
    'i want to disappear into a good book', "i'm lost without my morning coffee",
    'my heart is broken over the breakup', 'my grandma passed away peacefully', 'my cat died last week',
    'my uncle was killed in a car crash', "i'm afraid of death", 'i think about death a lot lately',
    'i wish i could start over', 'i want to run away to the beach', 'i hate mondays',
    'i hate myself for eating so much cake', 'i feel like a failure', "i'm so tired of my job",
    "i can't take this job anymore", 'my parents fight a lot', 'my sister and i argue all the time',
    'my boss yelled at me', "i'm furious with my landlord", 'i want to punch a pillow',
    "i'm dying to try that restaurant", 'my laptop died', 'i killed the spider', 'i nailed my presentation',
    'this homework is murder', "i'm swamped at work", "i'm cutting back on screen time",
    "i burned out last year but i'm better", "i'm drowning in laundry", 'i fell asleep on the couch',
    'i want to end the call', "i'm done with this book", 'i bombed my driving test', "i'm going to kill this workout",
    'i laughed so hard i cried', 'the movie was a killer', "i'd kill to have her job",
    'i overslept and missed the bus', 'my boss is a slave driver', 'we got crushed in the finals',
    "i'm going to beat the traffic", 'my phone battery is dying', 'the plant died', 'my goldfish died',
    'my grandfather died last year', 'my mom has cancer', "i'm terrified of flying", "i'm scared of the dark",
    'i hate my commute', "i'm sick of my roommate", "i can't stand my boss", 'i lost the game',
    "i'm hurting after my breakup", 'i cut my finger cooking', 'i hurt my back lifting',
    'i hit my head on the cupboard', 'i was jumping for joy', "i'm thinking of ending my membership",
    "i'm ending the year strong", "i'm dead on my feet after work",
]


# Verses the full-corpus design showed for situations, which must never come back.
HARMFUL = {
    "quran": ["22:15", "17:31", "7:124", "4:16", "81:8", "5:51", "9:34", "2:88", "17:72", "2:18", "43:17"],
    "bible": ["Jeremiah 47:5", "Proverbs 31:6", "Jeremiah 16:7", "Mark 5:39", "Romans 7:2", "Exodus 34:7",
              "Genesis 19:36", "Titus 1:14", "Psalms 109:9", "Psalms 137:9", "Exodus 20:12", "Deuteronomy 5:16",
              "Proverbs 10:12", "1 Peter 4:8", "Colossians 3:13", "Matthew 6:12"],
}


def check(s, pool_refs, res):
    query_themes = {t["id"] for t in res["themes"]}
    for book in BOOKS:
        for h in res["results"][book]:
            assert h["ref"] in pool_refs[book] and h["ref"] in s.books[book]["index"], (book, h["ref"])
            assert math.isfinite(h["score"]) and h["score"] >= 0
            assert h["theme"] in query_themes and h["theme"] in h["themes"]          # themes choose verses
            if res["category"] == "sensitive":
                assert s.books[book]["verses"][s.books[book]["index"][h["ref"]]]["sensitive"], (book, h["ref"])
    if not res["themes"]:
        assert all(res["results"][b] == [] for b in BOOKS)                          # no word-only fallback
    if res["category"] == "crisis":
        assert all(res["results"][b] == [] for b in BOOKS) and res["themes"] == [] and res["notes"] == []
        assert res["message"] and not res["no_match"]


@pytest.mark.parametrize("q", CRISIS + HELD_OUT_CRISIS)
def test_crisis_returns_no_verses(s, q):
    res = s.search(q)
    assert res["category"] == "crisis", q
    assert all(res["results"][b] == [] for b in BOOKS)


@pytest.mark.parametrize("q", NOT_CRISIS + HELD_OUT_NOT_CRISIS)
def test_grief_and_idioms_are_not_crisis(s, q):
    assert s.classify(q)[0] != "crisis", q


@pytest.mark.parametrize("q", SENSITIVE)
def test_sensitive_topics_return_only_the_sensitive_set(s, pool_refs, q):
    res = s.search(q)
    assert res["category"] == "sensitive", q
    check(s, pool_refs, res)
    assert not res["no_match"], q


def test_sensitive_results_depend_on_the_query(s):
    shown = {q: tuple(h["ref"] for b in BOOKS for h in s.search(q)["results"][b]) for q in SENSITIVE}
    assert len(set(shown.values())) >= 8                         # not one canned list for every query
    gay = s.search("I'm gay and my parents rejected me")
    assert {"loneliness", "self_worth", "love_compassion"} <= {t["id"] for t in gay["themes"]}


def test_harmful_verses_never_come_back(s, pool_refs):
    for book, refs in HARMFUL.items():
        assert not set(refs) & pool_refs[book], book
    queries = CRISIS + NOT_CRISIS + SENSITIVE + ["I've been drinking way too much", "my mom passed away",
                                                  "I want to end my marriage", "my parents are strict"]
    for q in queries:
        res = s.search(q, k=10)
        for book, refs in HARMFUL.items():
            assert not set(refs) & {h["ref"] for h in res["results"][book]}, q


@pytest.mark.parametrize("q, book, banned", [
    ("I've been drinking way too much every night", "bible", "Proverbs 31:6"),
    ("my mom passed away last month and I can't stop crying", "bible", "Mark 5:39"),
    ("my mom passed away", "bible", "Jeremiah 16:7"),
])
def test_reported_worst_results_are_gone(s, q, book, banned):
    assert banned not in [h["ref"] for h in s.search(q, k=10)["results"][book]]


def test_brief_examples_get_fitting_themes(s):
    want = {
        "I've been drinking way too much every night": "temptation_self_control",
        "my mom passed away last month and I can't stop crying": "grief",
        "I'm starting to doubt whether God is even real": "doubt_faith",
        "I've been waiting forever to hear back and I'm so impatient": "patience",
        "working 70 hour weeks and I'm completely burned out": "overwork_rest",
        "just got diagnosed with something serious and I'm really scared": "illness",
        "I lose my temper way too easily": "anger",
        "my best friend just stopped talking to me": "friendship",
    }
    for q, theme in want.items():
        res = s.search(q)
        assert res["themes"][0]["id"] == theme, (q, res["themes"])
        assert all(theme in h["themes"] or h["matched"]["themes"] for b in BOOKS for h in res["results"][b])


def test_fuzz_allowlist_guarantee(s, pool_refs):
    """Thousands of random queries: only pool verses, crisis empty, sensitive only from the set, finite scores."""
    rnd = random.Random(0)
    themes = build.load_json(build.THEMES)
    theme_words = [p.replace("*", "").replace("|", " ") for k, v in themes.items() if not k.startswith("_") for p in v["phrases"]]
    bridge_keys = [k for k in build.load_json(build.BRIDGE) if not k.startswith("_")]
    safety = build.load_json(build.SAFETY)
    risky = [p.replace("*", "").split("|")[0] for ps in safety["crisis"].values() for p in ps] + \
        [p.replace("*", "") for d in safety["sensitive"].values() for p in d["phrases"]]
    corpus_words = [w for book in BOOKS for v in build.load_pool()[book][:80] for w in v["why"].split()]
    sources = [theme_words, bridge_keys, risky, corpus_words, ["", "!!!", "...", "12345", "’", "I'm", "can't"]]

    def typo(w):
        if len(w) < 3:
            return w
        i = rnd.randrange(len(w) - 1)
        return rnd.choice([w[:i] + w[i + 1:], w[:i] + w[i + 1] + w[i] + w[i + 2:], w[:i] + "x" + w[i + 1:]])

    seen = set()
    for _ in range(3000):
        parts = [rnd.choice(rnd.choice(sources)) for _ in range(rnd.randint(0, 5))]
        parts = [typo(p) if rnd.random() < 0.2 else p for p in parts]
        q = rnd.choice([" ", ", ", ". ", " and "]).join(parts)
        if rnd.random() < 0.2:
            q = q.upper()
        res = s.search(q, k=rnd.randint(1, 5))
        check(s, pool_refs, res)
        seen.add(res["category"])
    assert seen == {"crisis", "sensitive", "general"}


def test_safety_lists_are_well_formed():
    from scripture.search import words
    safety = build.load_json(build.SAFETY)
    phrases = safety["idioms"] + [p for ps in safety["crisis"].values() for p in ps] + \
        [p for d in safety["sensitive"].values() for p in d["phrases"]]
    themes = {k for k in build.load_json(build.THEMES) if not k.startswith("_")}
    for p in phrases:
        assert p == p.lower() and p.strip() == p and "  " not in p, p
        for slot in p.split():
            for alt in slot.split("|"):
                assert alt == "*" or words(alt.rstrip("*")) == [alt.rstrip("*")], (p, alt)
    for d in safety["sensitive"].values():
        assert set(d["themes"]) <= themes
    for w in safety["fuzzy"]:
        assert len(w) >= 6


def test_theme_phrases_are_well_formed():
    from scripture.search import words
    for k, v in build.load_json(build.THEMES).items():
        if k.startswith("_"):
            continue
        assert v["name"] and v["description"] and v["phrases"]
        assert len(v.get("note", "")) <= 200 and "helpline" not in v.get("note", "").lower()
        for p in v["phrases"]:
            for slot in p.split():
                for alt in slot.split("|"):
                    assert words(alt.rstrip("*")) == [alt.rstrip("*")], (k, p)
    assert sum(len(v["phrases"]) for k, v in build.load_json(build.THEMES).items() if not k.startswith("_")) >= 600


def test_ambiguous_single_words_are_not_theme_phrases():
    themes = build.load_json(build.THEMES)
    singles = {p for k, v in themes.items() if not k.startswith("_") for p in v["phrases"] if " " not in p}
    assert not singles & {"passed", "high", "moved", "fired", "down", "kid", "lost"}


# ----- round 5: the probe sets, the distress gate, bereavement, and harmless misses -----

PROBES = json.loads((build.ROOT / "evals" / "search_probes.json").read_text())
ACCEPTED_MISSES = set(PROBES["accepted"]["missed_crisis"])
ACCEPTED_ALARMS = set(PROBES["accepted"]["false_alarms"])
FRESH = ("fresh_1", "fresh_2", "fresh_3")
PROBE_CRISIS = [p["text"] for split in ("blind_1",) + FRESH for p in PROBES[split]["crisis"]]
PROBE_NOT_CRISIS = PROBES["blind_1"]["innocent"] + [q for f in FRESH for g in ("innocent", "bereavement") for q in PROBES[f][g]]


@pytest.mark.parametrize("q", PROBE_CRISIS)
def test_probe_crisis_messages_get_no_verses(s, pool_refs, q):
    res = s.search(q)
    if q in ACCEPTED_MISSES:
        assert res["category"] != "crisis", "an accepted miss now fires: take it out of 'accepted'"
        return
    assert res["category"] == "crisis", q
    check(s, pool_refs, res)


@pytest.mark.parametrize("q", PROBE_NOT_CRISIS)
def test_probe_innocent_and_bereaved_messages_are_not_crisis(s, q):
    want_crisis = q in ACCEPTED_ALARMS                   # known over-firing: when in doubt the gate fires
    assert (s.classify(q)[0] == "crisis") == want_crisis, q


@pytest.mark.parametrize("q", PROBES["blind_1"]["sensitive"])
def test_probe_sensitive_messages_route_to_the_sensitive_set(s, pool_refs, q):
    res = s.search(q)
    assert res["category"] == "sensitive", q
    check(s, pool_refs, res)


@pytest.mark.parametrize("q", [q for f in FRESH for q in PROBES[f]["bereavement"]] + [
    "my dad passed away last month", "lost my friend to cancer", "my husband died by suicide two years ago",
    "my friend's dad died of an overdose and i dont know what to say", "lost my brother to suicide last spring",
    "my son overdosed and died in march", "my uncle was killed in a car crash", "my cat died this morning",
    "my best friend's dad passed away from an overdose and i dont know what to say",
])
def test_a_third_persons_death_goes_to_grief_not_the_gate(s, q):
    if q in ACCEPTED_ALARMS:
        return
    res = s.search(q)
    assert res["category"] != "crisis", q


@pytest.mark.parametrize("q", [
    "my dad passed away last month", "lost my friend to cancer", "my husband died by suicide two years ago",
    "my friend's dad died of an overdose and i dont know what to say", "my cat died this morning",
])
def test_bereavement_gets_only_grief_verses_and_the_grief_note(s, pool_refs, q):
    res = s.search(q)
    assert [t["id"] for t in res["themes"]] == ["grief"], q
    assert [n["theme"] for n in res["notes"]] == ["grief"]
    assert res["results"]["bible"] and all(h["theme"] == "grief" for b in BOOKS for h in res["results"][b])
    check(s, pool_refs, res)


@pytest.mark.parametrize("q", [
    "my husband died by suicide and i want to join him", "my dad passed away and i want to die too",
    "since my brother overdosed i keep thinking about taking all my pills", "my mom died and i want to kill myself",
])
def test_bereavement_with_the_writers_own_danger_still_fires(s, q):
    assert s.classify(q)[0] == "crisis", q


@pytest.mark.parametrize("q", [
    "this traffic is killing me", "i could kill for a pizza", "dying of laughter", "i'd kill for a cup of coffee",
    "my feet are killing me", "the deadline is going to kill me", "my phone died", "my brother beat me at chess",
])
def test_plain_idioms_do_not_fire(s, q):
    assert s.classify(q)[0] != "crisis", q


@pytest.mark.parametrize("q", [
    "im so done, ive got the pills right here", "pills in front of me", "have the meds lined up, how many is enough",
    "im a burden, theyd be better off w/o me", "every1 would be better off without me",
    "standing on the bridge rn", "said my goodbyes", "giving away all my stuff", "i wont be here much longer",
    "he grabs me by the throat", "she keeps hitting me", "my bf choked me last night", "he forced himself on me",
    "coach sends me pics and says keep it secret", "my ex is stalking me", "he threatens to kill me if i leave",
    "my parents starve me", "im going to bring a knife to school", "i want to poison my roommate",
])
def test_distress_gate_handles_typos_slang_and_word_order(s, q):
    assert s.classify(q)[0] == "crisis", q


def test_crisis_result_shape(s):
    res = s.search("i have the pills in front of me right now")
    assert res["category"] == "crisis" and res["topic"] in ("suicide", "distress")
    assert res["message"] == build.load_json(build.SAFETY)["crisis_message"]
    assert res["themes"] == res["notes"] == [] and not res["no_match"]
    assert not any(ch.isdigit() for ch in res["message"])                # no helpline numbers (the owner's decision)


def test_every_removed_verse_stays_out_of_the_pool(pool_refs):
    for name in ("review-5-final", "review-6-sweep", "review-7-flags"):
        review = build.load_json(build.POOL / f"{name}.json")
        for book in BOOKS:
            for item in review[book]:
                if item["action"] == "remove":
                    assert item["ref"] not in pool_refs[book], (name, book, item["ref"])
    sens = build.load_json(build.SENSITIVE_SET)
    assert not {"6:9", "12:12", "16:2"} & {x["ref"] for x in sens["gita"]}
    assert not {"93:4", "87:8", "28:22"} & {x["ref"] for x in sens["quran"]}


DEATH_WORDING = ["give you rest", "give thee rest", "see you again", "ease thy way", "extinction of all pains",
                 "goeth to peace", "abode of peace", "rest unto your souls", "shadow of death", "house of the lord for ever",
                 "better for thee than the former", "kingdom of heaven", "paradise", "hereafter", "we are returning",
                 "no more death", "end of that man is peace", "lay me down in peace", "an expected end"]


def test_no_general_verse_reads_as_an_invitation_to_die(s):
    for book in BOOKS:
        for v in s.books[book]["verses"]:
            text = (v["text"] + " " + v["why"]).lower()
            assert not [w for w in DEATH_WORDING if w in text], (book, v["ref"])


def test_forgiveness_and_patience_verses_are_not_reached_by_someone_being_harmed(s):
    for q in ["my husband hurts me but he always apologizes", "my boyfriend gets angry and yells at me",
              "someone at work is spreading rumours about me", "my dad is always angry"]:
        res = s.search(q)
        assert "forgiveness" not in {t["id"] for t in res["themes"]}, q
        for book in BOOKS:
            for h in res["results"][book]:
                assert not any(w in h["text"].lower() for w in ("forgive", "pardon", "bear with", "longsuffering")), (q, h["ref"])
    themes = build.load_json(build.THEMES)
    assert all(not p.startswith("forgive") or p in ("forgive me", "forgiven me") for p in themes["forgiveness"]["phrases"])


def test_safety_notes_are_gone_and_notes_only_follow_their_theme(s):
    themes = build.load_json(build.THEMES)
    for t in ("family_parents", "marriage", "anger", "suffering_endurance", "forgiveness"):
        assert "note" not in themes[t], t
    for q in ["just feeling really grateful today for my family and health", "my dad passed away last month",
              "my coworker is always angry at everyone"]:
        notes = {n["theme"] for n in s.search(q)["notes"]}
        assert notes <= {"grief", "self_worth", "patience", "betrayal"}, (q, notes)
    assert [t["id"] for t in s.search("just feeling really grateful today for my family and health")["themes"]] == ["gratitude"]


@pytest.mark.parametrize("q, theme", [
    ("i'm so lonley these days", "loneliness"),
    ("i keep putting off everything important and wasting days on my phone", "work_effort"),
    ("had to say goodbye to my dog of 14 years on saturday", "grief"),
    ("my 15 year old son wont talk to me, slams doors, i feel like ive lost him", "parenting"),
    ("i did something really embarassing at the party and i cant stop replaying it", "fear_anxiety"),
    ("she broke up with me after 3 years and i cant stop checking her instagram", "loneliness"),
    ("i want to be more generous but i always feel like i dont have enough to give", "generosity"),
    ("just feeling really grateful today for my family and health", "gratitude"),
])
def test_round_5_theme_detection_fixes(s, q, theme):
    res = s.search(q)
    assert res["themes"] and res["themes"][0]["id"] == theme, (q, res["themes"])
    if q.startswith("my 15"):
        assert "grief" not in {t["id"] for t in res["themes"]}
    if "embarassing" in q:
        assert "temptation_self_control" not in {t["id"] for t in res["themes"]}


def test_bible_marriage_speaks_to_either_spouse(s):
    marriage = {v["ref"] for v in s.books["bible"]["verses"] if "marriage" in v["themes"]}
    assert not marriage & {"Ephesians 5:25", "Colossians 3:19", "Proverbs 18:22", "Proverbs 31:28"}
    assert len(marriage) >= 3
