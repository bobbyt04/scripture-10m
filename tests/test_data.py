"""Checks on the committed files in data/processed/ (no network needed)."""

import hashlib
import json
import string
from collections import Counter
from pathlib import Path

import pytest

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
ALLOWED = set(string.ascii_letters + " .,;:!?'\"()-")
CHARSET = " !\"'(),-.:;?ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz"
BOOKS = ("gita", "quran", "bible")

# SHA-256 of each processed file. To change the data on purpose: rerun
# `python data/prepare.py`, review `git diff data/processed`, then paste the
# output of `shasum -a 256 data/processed/*` here.
SHA256 = {
    "bible": "599d7dea5a104c354ae2d0e88ade423c14ee82cad679c3a77afc05fb87b5a864",
    "gita": "74e047047eb4d0fcd0d6ed90dd3332659d2802aa1ab5ca24a8903968ff83ab2b",
    "quran": "a7bdc7deab1fdce97e3e157d0b76c853554e946833c052cfd1cefa924634e0a2",
}

# Verses per chapter, from sources independent of data/processed: the Bible
# table from two other KJV copies (thiagobodruk/bible and
# scrollmapper/bible_databases, which agree with each other and with Project
# Gutenberg #10); the Quran table from Tanzil's metadata (quran-data.js); the
# Gita table is the standard one, with Besant's extra verse in chapter 13.
GITA_CHAPTER_VERSES = [47, 72, 43, 42, 29, 47, 30, 28, 34, 42, 55, 20, 35, 27, 20, 24, 28, 78]
BIBLE_CHAPTER_VERSES = {
    "Genesis": [31, 25, 24, 26, 32, 22, 24, 22, 29, 32, 32, 20, 18, 24, 21, 16, 27, 33, 38, 18, 34, 24, 20, 67, 34, 35, 46, 22, 35, 43, 55, 32, 20, 31, 29, 43, 36, 30, 23, 23, 57, 38, 34, 34, 28, 34, 31, 22, 33, 26],
    "Exodus": [22, 25, 22, 31, 23, 30, 25, 32, 35, 29, 10, 51, 22, 31, 27, 36, 16, 27, 25, 26, 36, 31, 33, 18, 40, 37, 21, 43, 46, 38, 18, 35, 23, 35, 35, 38, 29, 31, 43, 38],
    "Leviticus": [17, 16, 17, 35, 19, 30, 38, 36, 24, 20, 47, 8, 59, 57, 33, 34, 16, 30, 37, 27, 24, 33, 44, 23, 55, 46, 34],
    "Numbers": [54, 34, 51, 49, 31, 27, 89, 26, 23, 36, 35, 16, 33, 45, 41, 50, 13, 32, 22, 29, 35, 41, 30, 25, 18, 65, 23, 31, 40, 16, 54, 42, 56, 29, 34, 13],
    "Deuteronomy": [46, 37, 29, 49, 33, 25, 26, 20, 29, 22, 32, 32, 18, 29, 23, 22, 20, 22, 21, 20, 23, 30, 25, 22, 19, 19, 26, 68, 29, 20, 30, 52, 29, 12],
    "Joshua": [18, 24, 17, 24, 15, 27, 26, 35, 27, 43, 23, 24, 33, 15, 63, 10, 18, 28, 51, 9, 45, 34, 16, 33],
    "Judges": [36, 23, 31, 24, 31, 40, 25, 35, 57, 18, 40, 15, 25, 20, 20, 31, 13, 31, 30, 48, 25],
    "Ruth": [22, 23, 18, 22],
    "1 Samuel": [28, 36, 21, 22, 12, 21, 17, 22, 27, 27, 15, 25, 23, 52, 35, 23, 58, 30, 24, 42, 15, 23, 29, 22, 44, 25, 12, 25, 11, 31, 13],
    "2 Samuel": [27, 32, 39, 12, 25, 23, 29, 18, 13, 19, 27, 31, 39, 33, 37, 23, 29, 33, 43, 26, 22, 51, 39, 25],
    "1 Kings": [53, 46, 28, 34, 18, 38, 51, 66, 28, 29, 43, 33, 34, 31, 34, 34, 24, 46, 21, 43, 29, 53],
    "2 Kings": [18, 25, 27, 44, 27, 33, 20, 29, 37, 36, 21, 21, 25, 29, 38, 20, 41, 37, 37, 21, 26, 20, 37, 20, 30],
    "1 Chronicles": [54, 55, 24, 43, 26, 81, 40, 40, 44, 14, 47, 40, 14, 17, 29, 43, 27, 17, 19, 8, 30, 19, 32, 31, 31, 32, 34, 21, 30],
    "2 Chronicles": [17, 18, 17, 22, 14, 42, 22, 18, 31, 19, 23, 16, 22, 15, 19, 14, 19, 34, 11, 37, 20, 12, 21, 27, 28, 23, 9, 27, 36, 27, 21, 33, 25, 33, 27, 23],
    "Ezra": [11, 70, 13, 24, 17, 22, 28, 36, 15, 44],
    "Nehemiah": [11, 20, 32, 23, 19, 19, 73, 18, 38, 39, 36, 47, 31],
    "Esther": [22, 23, 15, 17, 14, 14, 10, 17, 32, 3],
    "Job": [22, 13, 26, 21, 27, 30, 21, 22, 35, 22, 20, 25, 28, 22, 35, 22, 16, 21, 29, 29, 34, 30, 17, 25, 6, 14, 23, 28, 25, 31, 40, 22, 33, 37, 16, 33, 24, 41, 30, 24, 34, 17],
    "Psalms": [6, 12, 8, 8, 12, 10, 17, 9, 20, 18, 7, 8, 6, 7, 5, 11, 15, 50, 14, 9, 13, 31, 6, 10, 22, 12, 14, 9, 11, 12, 24, 11, 22, 22, 28, 12, 40, 22, 13, 17, 13, 11, 5, 26, 17, 11, 9, 14, 20, 23, 19, 9, 6, 7, 23, 13, 11, 11, 17, 12, 8, 12, 11, 10, 13, 20, 7, 35, 36, 5, 24, 20, 28, 23, 10, 12, 20, 72, 13, 19, 16, 8, 18, 12, 13, 17, 7, 18, 52, 17, 16, 15, 5, 23, 11, 13, 12, 9, 9, 5, 8, 28, 22, 35, 45, 48, 43, 13, 31, 7, 10, 10, 9, 8, 18, 19, 2, 29, 176, 7, 8, 9, 4, 8, 5, 6, 5, 6, 8, 8, 3, 18, 3, 3, 21, 26, 9, 8, 24, 13, 10, 7, 12, 15, 21, 10, 20, 14, 9, 6],
    "Proverbs": [33, 22, 35, 27, 23, 35, 27, 36, 18, 32, 31, 28, 25, 35, 33, 33, 28, 24, 29, 30, 31, 29, 35, 34, 28, 28, 27, 28, 27, 33, 31],
    "Ecclesiastes": [18, 26, 22, 16, 20, 12, 29, 17, 18, 20, 10, 14],
    "Song of Solomon": [17, 17, 11, 16, 16, 13, 13, 14],
    "Isaiah": [31, 22, 26, 6, 30, 13, 25, 22, 21, 34, 16, 6, 22, 32, 9, 14, 14, 7, 25, 6, 17, 25, 18, 23, 12, 21, 13, 29, 24, 33, 9, 20, 24, 17, 10, 22, 38, 22, 8, 31, 29, 25, 28, 28, 25, 13, 15, 22, 26, 11, 23, 15, 12, 17, 13, 12, 21, 14, 21, 22, 11, 12, 19, 12, 25, 24],
    "Jeremiah": [19, 37, 25, 31, 31, 30, 34, 22, 26, 25, 23, 17, 27, 22, 21, 21, 27, 23, 15, 18, 14, 30, 40, 10, 38, 24, 22, 17, 32, 24, 40, 44, 26, 22, 19, 32, 21, 28, 18, 16, 18, 22, 13, 30, 5, 28, 7, 47, 39, 46, 64, 34],
    "Lamentations": [22, 22, 66, 22, 22],
    "Ezekiel": [28, 10, 27, 17, 17, 14, 27, 18, 11, 22, 25, 28, 23, 23, 8, 63, 24, 32, 14, 49, 32, 31, 49, 27, 17, 21, 36, 26, 21, 26, 18, 32, 33, 31, 15, 38, 28, 23, 29, 49, 26, 20, 27, 31, 25, 24, 23, 35],
    "Daniel": [21, 49, 30, 37, 31, 28, 28, 27, 27, 21, 45, 13],
    "Hosea": [11, 23, 5, 19, 15, 11, 16, 14, 17, 15, 12, 14, 16, 9],
    "Joel": [20, 32, 21],
    "Amos": [15, 16, 15, 13, 27, 14, 17, 14, 15],
    "Obadiah": [21],
    "Jonah": [17, 10, 10, 11],
    "Micah": [16, 13, 12, 13, 15, 16, 20],
    "Nahum": [15, 13, 19],
    "Habakkuk": [17, 20, 19],
    "Zephaniah": [18, 15, 20],
    "Haggai": [15, 23],
    "Zechariah": [21, 13, 10, 14, 11, 15, 14, 23, 17, 12, 17, 14, 9, 21],
    "Malachi": [14, 17, 18, 6],
    "Matthew": [25, 23, 17, 25, 48, 34, 29, 34, 38, 42, 30, 50, 58, 36, 39, 28, 27, 35, 30, 34, 46, 46, 39, 51, 46, 75, 66, 20],
    "Mark": [45, 28, 35, 41, 43, 56, 37, 38, 50, 52, 33, 44, 37, 72, 47, 20],
    "Luke": [80, 52, 38, 44, 39, 49, 50, 56, 62, 42, 54, 59, 35, 35, 32, 31, 37, 43, 48, 47, 38, 71, 56, 53],
    "John": [51, 25, 36, 54, 47, 71, 53, 59, 41, 42, 57, 50, 38, 31, 27, 33, 26, 40, 42, 31, 25],
    "Acts": [26, 47, 26, 37, 42, 15, 60, 40, 43, 48, 30, 25, 52, 28, 41, 40, 34, 28, 41, 38, 40, 30, 35, 27, 27, 32, 44, 31],
    "Romans": [32, 29, 31, 25, 21, 23, 25, 39, 33, 21, 36, 21, 14, 23, 33, 27],
    "1 Corinthians": [31, 16, 23, 21, 13, 20, 40, 13, 27, 33, 34, 31, 13, 40, 58, 24],
    "2 Corinthians": [24, 17, 18, 18, 21, 18, 16, 24, 15, 18, 33, 21, 14],
    "Galatians": [24, 21, 29, 31, 26, 18],
    "Ephesians": [23, 22, 21, 32, 33, 24],
    "Philippians": [30, 30, 21, 23],
    "Colossians": [29, 23, 25, 18],
    "1 Thessalonians": [10, 20, 13, 18, 28],
    "2 Thessalonians": [12, 17, 18],
    "1 Timothy": [20, 15, 16, 16, 25, 21],
    "2 Timothy": [18, 26, 17, 22],
    "Titus": [16, 15, 15],
    "Philemon": [25],
    "Hebrews": [14, 18, 19, 16, 14, 20, 28, 13, 28, 39, 40, 29, 25],
    "James": [27, 26, 18, 17, 20],
    "1 Peter": [25, 25, 22, 19, 14],
    "2 Peter": [21, 22, 18],
    "1 John": [10, 29, 24, 21, 21],
    "2 John": [13],
    "3 John": [14],
    "Jude": [25],
    "Revelation": [20, 29, 22, 11, 14, 17, 17, 13, 21, 11, 19, 17, 18, 20, 8, 21, 18, 24, 21, 15, 27, 21],
}
QURAN_SURAH_VERSES = [7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128, 111, 110, 98, 135, 112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73, 54, 45, 83, 182, 88, 75, 85, 54, 53, 89, 59, 37, 35, 38, 29, 18, 45, 60, 49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11, 11, 18, 12, 12, 30, 52, 52, 44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17, 19, 26, 30, 20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3, 6, 3, 5, 4, 5, 6]


def load(book):
    with open(PROCESSED / f"{book}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module", params=BOOKS)
def verses(request):
    return load(request.param)


def by_ref(book):
    return {v["ref"]: v for v in load(book)}


def chapter_counts(verses, book=None):
    """{chapter: number of verses}, for one Bible book or a whole book."""
    return Counter(v["chapter"] for v in verses if book is None or v["section"] == book)


@pytest.mark.parametrize("book", BOOKS)
def test_checksums(book):
    digest = hashlib.sha256((PROCESSED / f"{book}.jsonl").read_bytes()).hexdigest()
    assert digest == SHA256[book], f"{book}.jsonl changed; see the note above SHA256"


def test_bible_chapter_counts():
    bible = load("bible")
    assert list(dict.fromkeys(v["section"] for v in bible)) == list(BIBLE_CHAPTER_VERSES)
    for book, expected in BIBLE_CHAPTER_VERSES.items():
        assert sorted(chapter_counts(bible, book).items()) == list(enumerate(expected, 1)), book
    assert sum(map(sum, BIBLE_CHAPTER_VERSES.values())) == 31102


def test_quran_chapter_counts():
    counts = chapter_counts(load("quran"))
    assert sorted(counts.items()) == list(enumerate(QURAN_SURAH_VERSES, 1))
    assert sum(QURAN_SURAH_VERSES) == 6236


def test_gita_chapter_counts():
    # Besant numbers Arjuna's opening question in chapter 13 as its own verse,
    # so her edition has 701 verses instead of the usual 700.
    counts = chapter_counts(load("gita"))
    assert sorted(counts.items()) == list(enumerate(GITA_CHAPTER_VERSES, 1))
    assert sum(GITA_CHAPTER_VERSES) == 701


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


def test_one_section_per_chapter(verses):
    # Every chapter carries one section name, and no two Quran or Gita
    # chapters share one.
    sections = {}
    for v in verses:
        sections.setdefault(v["chapter"] if v["book"] != "bible" else v["section"],
                            set()).add(v["section"])
    assert all(len(s) == 1 for s in sections.values())
    names = [s.pop() for s in sections.values()]
    assert len(names) == len(set(names))


def test_gita_section_titles():
    gita = by_ref("gita")
    assert gita["2:1"]["section"] == "Yoga by the Sankhya"
    assert gita["8:1"]["section"] == "The Yoga of the Indestructible Supreme Eternal"


def test_charset(verses):
    for v in verses:
        bad = set(v["text"]) - ALLOWED
        assert not bad, (v["ref"], bad)
        assert v["text"] == " ".join(v["text"].split()), v["ref"]


@pytest.mark.parametrize("junk", [
    "Gutenberg", "http", "www.", "Layout 2", "Wikisource", "***",
    "Total Verses", "[1]", "said: said:", "Heb.",
])
def test_no_boilerplate(verses, junk):
    hits = [v["ref"] for v in verses if junk in v["text"]]
    assert not hits, hits[:5]


def test_quran_has_only_pickthall():
    # The source interleaves three translations as "Y: ...", "P: ...", "S: ...".
    hits = [v["ref"] for v in load("quran") if "Y: " in v["text"] or "S: " in v["text"]]
    assert not hits, hits[:5]


# Full texts, spread through each book. They include verses that Project
# Gutenberg's KJV gets wrong and verses next to the source numbering typos.
@pytest.mark.parametrize("book,ref,text", [
    ("bible", "Genesis 1:1", "In the beginning God created the heaven and the earth."),
    ("bible", "Genesis 47:4", "They said moreover unto Pharaoh, For to sojourn in the land are we come; for thy servants have no pasture for their flocks; for the famine is sore in the land of Canaan: now therefore, we pray thee, let thy servants dwell in the land of Goshen."),
    ("bible", "Exodus 20:13", "Thou shalt not kill."),
    ("bible", "1 Samuel 15:33", "And Samuel said, As thy sword hath made women childless, so shall thy mother be childless among women. And Samuel hewed Agag in pieces before the LORD in Gilgal."),
    ("bible", "Nehemiah 9:28", "But after they had rest, they did evil again before thee: therefore leftest thou them in the hand of their enemies, so that they had the dominion over them: yet when they returned, and cried unto thee, thou heardest them from heaven; and many times didst thou deliver them according to thy mercies;"),
    ("bible", "Job 3:14", "With kings and counsellors of the earth, which built desolate places for themselves;"),
    ("bible", "Psalms 23:1", "A Psalm of David. The LORD is my shepherd; I shall not want."),
    ("bible", "Psalms 52:2", "Thy tongue deviseth mischiefs; like a sharp razor, working deceitfully."),
    ("bible", "Psalms 119:105", "Thy word is a lamp unto my feet, and a light unto my path."),
    ("bible", "Proverbs 3:5", "Trust in the LORD with all thine heart; and lean not unto thine own understanding."),
    ("bible", "Isaiah 40:31", "But they that wait upon the LORD shall renew their strength; they shall mount up with wings as eagles; they shall run, and not be weary; and they shall walk, and not faint."),
    ("bible", "Ezekiel 22:21", "Yea, I will gather you, and blow upon you in the fire of my wrath, and ye shall be melted in the midst thereof."),
    ("bible", "Daniel 8:11", "Yea, he magnified himself even to the prince of the host, and by him the daily sacrifice was taken away, and the place of his sanctuary was cast down."),
    ("bible", "Jonah 1:15", "So they took up Jonah, and cast him forth into the sea: and the sea ceased from her raging."),
    ("bible", "Matthew 5:9", "Blessed are the peacemakers: for they shall be called the children of God."),
    ("bible", "Mark 15:2", "And Pilate asked him, Art thou the King of the Jews? And he answering said unto him, Thou sayest it."),
    ("bible", "John 3:16", "For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life."),
    ("bible", "John 11:35", "Jesus wept."),
    ("bible", "Acts 2:20", "The sun shall be turned into darkness, and the moon into blood, before that great and notable day of the Lord come:"),
    ("bible", "Acts 4:20", "For we cannot but speak the things which we have seen and heard."),
    ("bible", "1 Corinthians 13:4", "Charity suffereth long, and is kind; charity envieth not; charity vaunteth not itself, is not puffed up,"),
    ("bible", "Hebrews 9:5", "And over it the cherubims of glory shadowing the mercyseat; of which we cannot now speak particularly."),
    ("bible", "Revelation 22:21", "The grace of our Lord Jesus Christ be with you all. Amen."),
    ("quran", "1:1", "In the name of Allah, the Beneficent, the Merciful."),
    ("quran", "1:7", "The path of those whom Thou hast favoured; Not the (path) of those who earn Thine anger nor of those who go astray."),
    ("quran", "2:2", "This is the Scripture whereof there is no doubt, a guidance unto those who ward off (evil)."),
    ("quran", "2:124", "And (remember) when his Lord tried Abraham with (His) commands, and he fulfilled them, He said: Lo! I have appointed thee a leader for mankind. (Abraham) said: And of my offspring (will there be leaders)? He said: My covenant includeth not wrong-doers."),
    ("quran", "2:255", "Allah! There is no deity save Him, the Alive, the Eternal. Neither slumber nor sleep overtaketh Him. Unto Him belongeth whatsoever is in the heavens and whatsoever is in the earth. Who is he that intercedeth with Him save by His leave? He knoweth that which is in front of them and that which is behind them, while they encompass nothing of His knowledge save what He will. His throne includeth the heavens and the earth, and He is never weary of preserving them. He is the Sublime, the Tremendous."),
    ("quran", "17:32", "And come not near unto adultery. Lo! it is an abomination and an evil way."),
    ("quran", "17:33", "And slay not the life which Allah hath forbidden save with right. Whoso is slain wrongfully, We have given power unto his heir, but let him not commit excess in slaying. Lo! he will be helped."),
    ("quran", "17:34", "Come not near the wealth of the orphan save with that which is better till he come to strength; and keep the covenant. Lo! of the covenant it will be asked."),
    ("quran", "39:46", "Say: O Allah! Creator of the heavens and the earth! Knower of the Invisible and the Visible! Thou wilt judge between Thy slaves concerning that wherein they used to differ."),
    ("quran", "39:47", "And though those who do wrong possess all that is in the earth, and therewith as much again, they verily will seek to ransom themselves therewith on the Day of Resurrection from the awful doom; and there will appear unto them, from their Lord, that wherewith they never reckoned."),
    ("quran", "45:31", "And as for those who disbelieved (it will be said unto them): Were not Our revelations recited unto you? But ye were scornful and became a guilty folk."),
    ("quran", "45:32", "And when it was said: Lo! Allah's promise is the truth, and there is no doubt of the Hour's coming, ye said: We know not what the Hour is. We deem it naught but a conjecture, and we are by no means convinced."),
    ("quran", "45:33", "And the evils of what they did will appear unto them, and that which they used to deride will befall them."),
    ("quran", "56:26", "(Naught) but the saying: Peace, (and again) Peace."),
    ("quran", "56:27", "And those on the right hand; what of those on the right hand?"),
    ("quran", "88:21", "Remind them, for thou art but an admonisher,"),
    ("quran", "112:1", "Say: He is Allah, the One!"),
    ("quran", "114:6", "Of the jinn and of mankind."),
    ("gita", "1:1", "Dhritarashtra said: On the holy plain, on the field of Kuru, gathered together, eager for battle, what did they, O Sanjaya, my people and the Pandavas?"),
    ("gita", "1:28", "Deeply moved to pity, thus uttered in sadness: Arjuna said: Seeing these my kinsmen, O Krishna, arrayed, eager to fight,"),
    ("gita", "2:47", "Thy business is with the action only, never with its fruits; so let not the fruit of action be thy motive, nor be thou to inaction attached."),
    ("gita", "3:42", "It is said that the senses are great; greater than the senses is the mind: greater than the mind is the Reason; but what is greater than the Reason, is He."),
    ("gita", "3:43", "Thus understanding Him as greater than the Reason, restraining the self by the Self, slay thou, O mighty-armed, the enemy in the form of desire, difficult to overcome."),
    ("gita", "10:31", "Of purifiers I am the wind; Rama of warriors I; and I am Makara of fishes; of streams the Ganga am I."),
    ("gita", "10:32", "Of creations the beginning and the ending, and also the middle am I, O Arjuna. Of sciences the science concerning the Self; the speech of orators I."),
    ("gita", "11:8", "But verily thou art not able to behold Me with these thine eyes; the divine eye I give unto thee. Behold My sovereign Yoga."),
    ("gita", "11:9", "Sanjaya said: Having thus spoken, O King, the great Lord of Yoga, Hari, showed to Partha His supreme Form as Lord."),
    ("gita", "13:1", "Arjuna said: Matter and Spirit, even the Field and the Knower of the Field, wisdom and that which ought to be known, these I fain would learn, O Keshava."),
    ("gita", "13:2", "The Blessed Lord said: This body, son of Kunti, is called the Field; that which knoweth it is called the Knower of the Field by the Sages."),
    ("gita", "17:19", "That austerity done under a deluded understanding, with self-torture, or with the object of destroying another, that is declared of darkness."),
    ("gita", "17:20", "That alms given to one who does nothing in return, believing that a gift ought to be made in a fit place and time to a worthy person, that alms is accounted pure."),
    ("gita", "18:14", "The body, the actor, the various organs, the diverse kinds of energies, and the presiding deities also, the fifth."),
    ("gita", "18:15", "Whatever action a man performeth by his body, speech and mind, whether right or the reverse, these five are the cause thereof."),
    ("gita", "18:32", "That which, enwrapped in darkness, thinketh wrong to be right and seeth all things subverted, that Reason, O Partha, is of darkness."),
    ("gita", "18:33", "The unwavering firmness by which, through Yoga, one restraineth the activity of the mind, of the life breaths and of the sense-organs, that firmness, O Partha, is pure."),
    ("gita", "18:66", "Abandoning all duties come unto Me alone for shelter; sorrow not, I will liberate thee from all sins."),
    ("gita", "18:78", "Wherever is Krishna, Yoga's Lord, wherever is Partha, the archer, assured are there prosperity, victory and happiness. So I think."),
])
def test_exact_verses(book, ref, text):
    assert by_ref(book)[ref]["text"] == text


def test_stats_match_files():
    stats = json.loads((PROCESSED / "stats.json").read_text())
    assert stats["charset"] == CHARSET
    chars = set()
    for book in BOOKS:
        verses = load(book)
        text = "".join(v["text"] for v in verses)
        chars |= set(text)
        assert stats[book]["verses"] == len(verses)
        assert stats[book]["characters"] == len(text)
        assert stats[book]["charset_size"] == len(set(text))
        assert stats[book]["chapters"] == len({(v["section"], v["chapter"]) for v in verses})
    assert "".join(sorted(chars)) == CHARSET
