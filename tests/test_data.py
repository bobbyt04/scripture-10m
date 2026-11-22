"""Checks on the committed files in data/processed/ (no network needed)."""

import json
import string
from pathlib import Path

import pytest

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
ALLOWED = set(string.ascii_letters + string.digits + " .,;:!?'\"()-")
BOOKS = ("gita", "quran", "bible")


def load(book):
    with open(PROCESSED / f"{book}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module", params=BOOKS)
def verses(request):
    return load(request.param)


def by_ref(book):
    return {v["ref"]: v for v in load(book)}


def test_bible_counts():
    verses = load("bible")
    assert len(verses) == 31102
    assert len({v["section"] for v in verses}) == 66
    assert len({(v["section"], v["chapter"]) for v in verses}) == 1189


def test_quran_counts():
    verses = load("quran")
    assert len(verses) == 6236
    assert {v["chapter"] for v in verses} == set(range(1, 115))


def test_gita_counts():
    # Besant numbers Arjuna's opening question in chapter 13 as its own verse,
    # so her edition has 701 verses instead of the usual 700.
    verses = load("gita")
    assert len(verses) == 701
    assert {v["chapter"] for v in verses} == set(range(1, 19))
    assert sum(v["chapter"] == 13 for v in verses) == 35


def test_fields_and_refs(verses):
    refs = [v["ref"] for v in verses]
    assert len(refs) == len(set(refs)), "duplicate refs"
    for v in verses:
        assert set(v) == {"book", "ref", "chapter", "verse", "section", "text"}
        assert v["text"] and v["section"]
        prefix = "" if v["book"] != "bible" else v["section"] + " "
        assert v["ref"] == f"{prefix}{v['chapter']}:{v['verse']}"


def test_verses_are_numbered_without_gaps(verses):
    expected = {}
    for v in verses:
        key = (v["section"], v["chapter"])
        expected[key] = expected.get(key, 0) + 1
        assert v["verse"] == expected[key], v["ref"]


def test_charset(verses):
    for v in verses:
        bad = set(v["text"]) - ALLOWED
        assert not bad, (v["ref"], bad)
        assert v["text"] == " ".join(v["text"].split()), v["ref"]


@pytest.mark.parametrize("junk", [
    "Gutenberg", "http", "www.", "Layout 2", "Wikisource", "***",
    "Total Verses", "[1]", "said: said:",
])
def test_no_boilerplate(verses, junk):
    hits = [v["ref"] for v in verses if junk in v["text"]]
    assert not hits, hits[:5]


def test_quran_has_only_pickthall():
    # The source interleaves three translations as "Y: ...", "P: ...", "S: ...".
    hits = [v["ref"] for v in load("quran") if "Y: " in v["text"] or "S: " in v["text"]]
    assert not hits, hits[:5]


@pytest.mark.parametrize("book,ref,start", [
    ("bible", "Genesis 1:1", "In the beginning God created the heaven"),
    ("bible", "Psalms 23:1", "The LORD is my shepherd; I shall not want."),
    ("bible", "John 3:16", "For God so loved the world"),
    ("bible", "John 11:35", "Jesus wept."),
    ("bible", "1 Corinthians 13:4", "Charity suffereth long"),
    ("bible", "Revelation 22:21", "The grace of our Lord Jesus Christ"),
    ("quran", "1:1", "In the name of Allah, the Beneficent, the Merciful."),
    ("quran", "2:255", "Allah! There is no deity save Him, the Alive, the Eternal."),
    ("quran", "2:286", "Allah tasketh not a soul beyond its scope."),
    ("quran", "17:33", "And slay not the life which Allah hath forbidden"),
    ("quran", "112:1", "Say: He is Allah, the One!"),
    ("quran", "114:6", "Of the jinn and of mankind."),
    ("gita", "1:1", "Dhritarashtra said: On the holy plain, on the field of Kuru"),
    ("gita", "2:47", "Thy business is with the action only, never with its fruits"),
    ("gita", "10:31", "Of purifiers I am the wind"),
    ("gita", "13:1", "Arjuna said: Matter and Spirit"),
    ("gita", "18:66", "Abandoning all duties come unto Me alone for shelter"),
    ("gita", "18:78", "Wherever is Krishna, Yoga's Lord"),
])
def test_famous_verses(book, ref, start):
    assert by_ref(book)[ref]["text"].startswith(start)


def test_stats_match_files():
    stats = json.loads((PROCESSED / "stats.json").read_text())
    for book in BOOKS:
        verses = load(book)
        assert stats[book]["verses"] == len(verses)
        assert stats[book]["characters"] == sum(len(v["text"]) for v in verses)
    assert set(stats["charset"]) <= ALLOWED
