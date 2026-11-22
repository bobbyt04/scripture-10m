"""Download the three scriptures and write one clean verse per line.

Usage: python data/prepare.py

(If HTTPS downloads fail with CERTIFICATE_VERIFY_FAILED on macOS, run with
SSL_CERT_FILE=/etc/ssl/cert.pem.)

Standard library only. Raw downloads are cached in data/raw/ (gitignored);
output goes to data/processed/. See data/SOURCES.md for the sources and why
each one is public domain.
"""

import json
import re
import string
import time
import unicodedata
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "processed"
USER_AGENT = "scripture-3m-data-prep/1.0"

# Project Gutenberg's main site, then a PGLAF mirror of the same files.
GUTENBERG = [
    "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt",
    "https://gutenberg.pglaf.org/cache/epub/{n}/pg{n}.txt",
]
WIKISOURCE_API = "https://en.wikisource.org/w/api.php"
GITA_PAGE = "Bhagavad-Gita (Besant 4th)/Discourse {n}"

# The only characters allowed in verse text.
ALLOWED = set(string.ascii_letters + string.digits + " .,;:!?'\"()-")

REPLACE = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2014": " - ", "\u2013": "-", "\u00e6": "ae", "\u00c6": "AE",
    "\u0153": "oe", "\u0152": "OE", "\u00a0": " ", "\u200b": "",
}


def clean(text):
    """Collapse whitespace and reduce to the ALLOWED character set."""
    for a, b in REPLACE.items():
        text = text.replace(a, b)
    # Drop diacritics: "Madhusûdana" -> "Madhusudana".
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = " ".join(text.split())
    bad = set(text) - ALLOWED
    if bad:
        raise ValueError(f"unexpected characters {sorted(bad)} in: {text!r}")
    return text


def fetch(urls, path):
    """Download the first URL that works into path, unless already cached."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        time.sleep(2)  # be polite; Wikisource rate-limits rapid requests
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                text = r.read().decode("utf-8")
        except OSError as e:
            print(f"  failed {url}: {e}")
            continue
        if "*** END OF THE PROJECT GUTENBERG" in text or url.startswith(WIKISOURCE_API):
            path.write_text(text, encoding="utf-8")
            return text
        print(f"  incomplete download from {url}")
    raise RuntimeError(f"could not download {path.name}")


def gutenberg(n):
    """Return the body of Gutenberg ebook n, without the licence header/footer."""
    text = fetch([u.format(n=n) for u in GUTENBERG], RAW / f"pg{n}.txt")
    text = text.replace("\r\n", "\n")
    start = text.index("\n", text.index("*** START OF THE PROJECT GUTENBERG"))
    return text[start:text.index("*** END OF THE PROJECT GUTENBERG")]


# --- Bible: King James Version, Project Gutenberg #10 -----------------------

BIBLE_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua",
    "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
    "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah",
    "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
    "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
    "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation",
]
TESTAMENTS = (
    "The Old Testament of the King James Version of the Bible",
    "The New Testament of the King James Bible",
)


def bible():
    lines = gutenberg(10).split("\n")
    # The file opens with a contents list: the 66 full book titles in order.
    first = lines.index(TESTAMENTS[0])
    titles = [l for l in lines[first + 1:first + 70] if l and l not in TESTAMENTS]
    assert len(titles) == 66, len(titles)
    body = lines[lines.index(TESTAMENTS[0], first + 1) + 1:]

    books, current = [], None
    for line in body:
        if len(books) < 66 and line == titles[len(books)]:
            current = []
            books.append(current)
        elif current is not None and line not in TESTAMENTS + ("***",):
            current.append(line)

    verses = []
    for name, book in zip(BIBLE_BOOKS, books):
        # Each verse starts "chapter:verse "; split on those markers.
        parts = re.split(r"(?:^|\s)(\d+):(\d+) ", " ".join(book))
        # Before 1:1 there may be a subtitle ("Otherwise Called: ..."), never a verse.
        assert not re.search(r"\d", parts[0]), (name, parts[0][:80])
        for c, v, text in zip(parts[1::3], parts[2::3], parts[3::3]):
            c, v = int(c), int(v)
            verses.append(dict(book="bible", ref=f"{name} {c}:{v}", chapter=c,
                               verse=v, section=name, text=clean(text)))
    return verses


# --- Quran: Pickthall, from Project Gutenberg #16955 ------------------------

def surah_name(heading):
    # "AL-BAQARA (THE COW)" -> "Al-Baqara"
    name = heading.split(" (")[0].strip()
    return re.sub(r"[A-Z][A-Z']*", lambda m: m.group(0).capitalize(), name)


def quran():
    text = gutenberg(16955)
    verses = []
    # Each surah opens " Chapter N:", its name and "Total Verses: N".
    surahs = re.findall(r"\n Chapter (\d+):\s*\n\s*(.+?)\s*\n Total Verses: (\d+)"
                        r".*?\n(.*?)(?=\n Chapter \d+:|\Z)", text, re.S)
    assert [int(s[0]) for s in surahs] == list(range(1, 115))
    for c, heading, total, body in surahs:
        # Each verse is a label like "002.255" and then three translations,
        # "Y:", "P:" and "S:"; we keep only Pickthall's ("P:"). Four labels
        # have typos (e.g. "0.033"), so verses are numbered by position and
        # checked against the surah's stated total.
        labels = re.findall(r"^\d*\.\d*$", body, re.M)
        picks = re.findall(r"^P: (.*?)\n\nS: ", body, re.S | re.M)
        c = int(c)
        assert len(picks) == len(labels) == int(total), (c, len(picks), total)
        for v, pick in enumerate(picks, 1):
            verses.append(dict(book="quran", ref=f"{c}:{v}", chapter=c, verse=v,
                               section=surah_name(heading), text=clean(pick)))
    return verses


# --- Gita: Annie Besant's translation, from English Wikisource -------------

class Paragraphs(HTMLParser):
    """Collect the visible text of each <p>, minus footnotes and page numbers."""

    def __init__(self):
        super().__init__()
        self.paras, self.buf, self.skip_tag, self.depth = [], None, None, 0

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if self.skip_tag:
            self.depth += tag == self.skip_tag
        elif tag in ("style", "ol") or "reference" in cls or "pagenum" in cls:
            self.skip_tag, self.depth = tag, 1
        elif tag == "p":
            self.buf = []

    def handle_endtag(self, tag):
        if self.skip_tag:
            self.depth -= tag == self.skip_tag
            if self.depth == 0:
                self.skip_tag = None
        elif tag == "p" and self.buf is not None:
            self.paras.append(" ".join("".join(self.buf).split()))
            self.buf = None

    def handle_data(self, data):
        if self.buf is not None and not self.skip_tag:
            self.buf.append(data)


SMALL_WORDS = {"of", "the", "and", "by", "from", "between"}
# Verses per chapter in Besant's edition: the usual 700, plus an opening
# question by Arjuna that she numbers as 13:1 (many editions omit it).
GITA_COUNTS = [47, 72, 43, 42, 29, 47, 30, 28, 34, 42, 55, 20, 35, 27, 20, 24, 28, 78]


def gita():
    verses = []
    for c in range(1, 19):
        query = urllib.parse.urlencode({
            "action": "parse", "page": GITA_PAGE.format(n=c),
            "prop": "text", "format": "json", "formatversion": "2",
        })
        raw = fetch([f"{WIKISOURCE_API}?{query}"], RAW / "gita" / f"discourse_{c:02d}.json")
        parser = Paragraphs()
        parser.feed(json.loads(raw)["parse"]["text"])

        title, pending, chapter = None, [], []
        for i, p in enumerate(parser.paras):
            if not p or p == "Layout 2" or p.isupper() or re.search("[\u0900-\u097f]", p):
                continue  # empty, layout marker, heading or Sanskrit
            if p.endswith("entitled:"):  # closing colophon names the chapter
                title = next(q for q in parser.paras[i + 1:] if q)
                break
            # A verse ends with its number, "(12)". The transcription has a few
            # typos ("(12", "12)", a wrong or missing number), so verses are
            # numbered by position and the printed number is only a hint.
            m = re.search(r"\(\d+\)?$|\d+\)$", p)
            if not m:
                pending.append(p)  # a speaker ("Arjuna said:") or part of a verse
                continue
            if (int(m.group(0).strip("()")) == len(chapter) + 2
                    and any(not q.endswith("said:") for q in pending)):
                # The previous verse lost its number: it is the pending text up
                # to the last paragraph that is not a speaker.
                k = 1 + max(j for j, q in enumerate(pending) if not q.endswith("said:"))
                chapter.append(pending[:k])
                pending = pending[k:]
            chapter.append(pending + [p[:m.start()]])
            pending = []
        assert len(chapter) == GITA_COUNTS[c - 1], (c, len(chapter))
        # "THE YOGA OF ACTION." -> "The Yoga of Action"
        words = title.rstrip(".").lower().split()
        section = " ".join(w if i and w in SMALL_WORDS else w.capitalize()
                           for i, w in enumerate(words))
        for v, parts in enumerate(chapter, 1):
            verses.append(dict(book="gita", ref=f"{c}:{v}", chapter=c, verse=v,
                               section=section, text=clean(" ".join(parts))))
    return verses


def main():
    OUT.mkdir(exist_ok=True)
    stats, all_chars = {}, set()
    for name, build in (("gita", gita), ("quran", quran), ("bible", bible)):
        print(f"{name}...")
        verses = build()
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for v in verses:
                f.write(json.dumps(v, ensure_ascii=True) + "\n")
        chars = "".join(v["text"] for v in verses)
        all_chars |= set(chars)
        stats[name] = dict(
            verses=len(verses), characters=len(chars),
            chapters=len({(v["section"], v["chapter"]) for v in verses}),
            charset_size=len(set(chars)),
        )
        print(f"  {stats[name]}")
    stats["charset"] = "".join(sorted(all_chars))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"charset ({len(all_chars)}): {stats['charset']}")


if __name__ == "__main__":
    main()
