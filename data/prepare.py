"""Download the three scriptures and write one clean verse per line.

Usage: python data/prepare.py

(If HTTPS downloads fail with CERTIFICATE_VERIFY_FAILED on macOS, run with
SSL_CERT_FILE=/etc/ssl/cert.pem.)

Standard library only. Raw downloads are cached in data/raw/ (gitignored);
output goes to data/processed/. See data/SOURCES.md for the sources and why
each one is public domain. Anything unexpected in a source raises an error
rather than being merged into a neighbouring verse.
"""

import io
import json
import re
import string
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "processed"
USER_AGENT = "scripture-10m-data-prep/1.0"

EBIBLE_KJV = "https://ebible.org/Scriptures/eng-kjv_vpl.zip"
# Project Gutenberg's main site, then a PGLAF mirror of the same files.
GUTENBERG = [
    "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt",
    "https://gutenberg.pglaf.org/cache/epub/{n}/pg{n}.txt",
]
WIKISOURCE_API = "https://en.wikisource.org/w/api.php"
GITA_PAGE = "Bhagavad-Gita (Besant 4th)/Discourse {n}"

# The only characters allowed in verse text.
ALLOWED = set(string.ascii_letters + " .,;:!?'\"()-")

REPLACE = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "—": " - ", "–": "-", "æ": "ae", "Æ": "Ae",
    "œ": "oe", "Œ": "Oe", " ": " ", "​": "",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def clean(text):
    """Collapse whitespace and reduce to the ALLOWED character set."""
    for a, b in REPLACE.items():
        text = text.replace(a, b)
    # Drop diacritics: "Madhusûdana" -> "Madhusudana".
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = " ".join(text.split())
    bad = set(text) - ALLOWED
    require(text and not bad, f"unexpected characters {sorted(bad)} in: {text!r}")
    return text


def fetch(urls, path, looks_right):
    """Return the bytes of path, downloading it from the first URL that works.

    Only a response that passes looks_right() is cached. 429 and 503 answers
    are retried with backoff.
    """
    if path.exists():
        return path.read_bytes()
    for url in urls:
        for attempt in range(4):
            time.sleep(2 * 2 ** attempt)  # 2, 4, 8, 16 s: polite and backs off
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
            except urllib.error.HTTPError as e:
                print(f"  {url}: HTTP {e.code}")
                if e.code in (429, 503):
                    continue
                break
            except OSError as e:
                print(f"  {url}: {e}")
                break
            if not looks_right(data):
                print(f"  {url}: unexpected content")
                break
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return data
    raise RuntimeError(f"could not download {path.name}")


def check_order(verses):
    """Chapters and verses must run 1, 2, 3... (within each Bible book)."""
    prev = None
    for v in verses:
        key = (v["section"] if v["book"] == "bible" else "", v["chapter"], v["verse"])
        if prev and key[0] == prev[0]:
            ok = key[1:] in ((prev[1], prev[2] + 1), (prev[1] + 1, 1))
        else:
            ok = key[1:] == (1, 1)
        require(ok, f"verse out of order: {v['ref']} after {prev}")
        prev = key


# --- Bible: King James Version, eBible.org eng-kjv --------------------------

BIBLE_CODES = (
    "GEN EXO LEV NUM DEU JOS JDG RUT 1SA 2SA 1KI 2KI 1CH 2CH EZR NEH EST JOB "
    "PSA PRO ECC SOL ISA JER LAM EZE DAN HOS JOE AMO OBA JON MIC NAH HAB ZEP "
    "HAG ZEC MAL MAT MAR LUK JOH ACT ROM 1CO 2CO GAL EPH PHI COL 1TH 2TH 1TI "
    "2TI TIT PHM HEB JAM 1PE 2PE 1JO 2JO 3JO JUD REV"
).split()
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
APOCRYPHA = set("TOB JDT ESG WIS SIR BAR PRA SUS BEL 1MA 2MA 1ES PRM 4ES".split())
# Words the eBible text splits in two where three independent KJV copies
# (and the printed 1769 text) have one word. See data/SOURCES.md.
BIBLE_FIXES = {
    "Acts 4:16": ("can not", "cannot"), "Acts 4:20": ("can not", "cannot"),
    "Acts 5:39": ("can not", "cannot"), "Colossians 3:17": ("what soever", "whatsoever"),
    "Colossians 3:23": ("what soever", "whatsoever"), "1 Timothy 2:7": ("Where unto", "Whereunto"),
    "1 Timothy 5:25": ("can not", "cannot"), "1 Timothy 6:12": ("where unto", "whereunto"),
    "2 Timothy 1:11": ("Where unto", "Whereunto"), "2 Timothy 2:9": ("Where in", "Wherein"),
    "2 Timothy 2:13": ("can not", "cannot"), "Hebrews 6:17": ("Where in", "Wherein"),
    "Hebrews 9:5": ("can not", "cannot"),
}


def bible():
    data = fetch([EBIBLE_KJV], RAW / "eng-kjv_vpl.zip",
                 lambda d: zipfile.is_zipfile(io.BytesIO(d)))
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        lines = z.read("eng-kjv_vpl.txt").decode("utf-8-sig").splitlines()
    names = dict(zip(BIBLE_CODES, BIBLE_BOOKS))
    verses = []
    for line in lines:
        # Every line is "JOH 3:16 For God so loved...".
        m = re.fullmatch(r"(\w+) (\d+):(\d+) (.+)", line)
        require(m, f"unparseable line: {line!r}")
        code, c, v, text = m[1], int(m[2]), int(m[3]), m[4]
        if code in APOCRYPHA:
            continue
        require(code in names, f"unknown book code: {line!r}")
        # Square brackets mark the translators' italics; a pilcrow marks a
        # paragraph. Keep the words, drop the marks.
        text = re.sub(r"[\[\]¶]", "", text)
        ref = f"{names[code]} {c}:{v}"
        if ref in BIBLE_FIXES:
            wrong, right = BIBLE_FIXES[ref]
            require(text.count(wrong) == 1, f"fix for {ref} no longer applies")
            text = text.replace(wrong, right)
        verses.append(dict(book="bible", ref=ref, chapter=c, verse=v,
                           section=names[code], text=clean(text)))
    require([s for s in dict.fromkeys(v["section"] for v in verses)] == BIBLE_BOOKS,
            "books missing or out of order")
    return verses


# --- Quran: Pickthall, from Project Gutenberg #16955 ------------------------

def gutenberg(n):
    """Return the body of Gutenberg ebook n, without the licence header/footer."""
    end = b"*** END OF THE PROJECT GUTENBERG"
    text = fetch([u.format(n=n) for u in GUTENBERG], RAW / f"pg{n}.txt",
                 lambda d: end in d).decode("utf-8")
    text = text.replace("\r\n", "\n")
    start = text.index("\n", text.index("*** START OF THE PROJECT GUTENBERG"))
    return text[start:text.index(end.decode())]


BISMILLAH = "In the name of Allah, the Most Beneficent, the Most Merciful."


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
    require([int(s[0]) for s in surahs] == list(range(1, 115)), "surahs missing")
    for c, heading, total, body in surahs:
        c = int(c)
        # Each verse is a label like "002.255" and then three translations,
        # "Y:", "P:" and "S:"; we keep only Pickthall's ("P:"). Four labels
        # have typos (e.g. "0.033"), so verses are numbered by position; every
        # block between labels must hold exactly one translation of each kind.
        body = re.sub(r"^-{20,}$", "", body, flags=re.M)  # separator rules
        blocks = re.split(r"^\d*\.\d+$", body, flags=re.M)
        # Before verse 1 the file may print an unnumbered Bismillah (in Yusuf
        # Ali's wording), which we leave out; anything else is an error.
        require(blocks[0].strip() in ("", BISMILLAH), f"text before verse 1 of surah {c}")
        blocks = blocks[1:]
        require(len(blocks) == int(total), f"surah {c}: {len(blocks)} verses, expected {total}")
        for v, block in enumerate(blocks, 1):
            m = re.fullmatch(r"\s*Y: (.+?)\nP: (.+?)\nS: (.+?)\s*", block, re.S)
            require(m and len(re.findall(r"^[A-Z]: ", block, re.M)) == 3,
                    f"unparseable verse {c}:{v}: {block[:200]!r}")
            verses.append(dict(book="quran", ref=f"{c}:{v}", chapter=c, verse=v,
                               section=surah_name(heading), text=clean(m[2])))
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
# Verses whose printed number in the transcription is wrong (None: missing).
GITA_NUMBER_TYPOS = {(10, 31): None, (11, 8): None, (17, 19): 20, (18, 14): 15}
# Spelling corrections to the chapter titles in the transcription.
GITA_TITLE_FIXES = {"INDESCTRUCTIBLE": "INDESTRUCTIBLE"}


def gita():
    verses = []
    for c in range(1, 19):
        query = urllib.parse.urlencode({
            "action": "parse", "page": GITA_PAGE.format(n=c),
            "prop": "text", "format": "json", "formatversion": "2",
        })
        raw = fetch([f"{WIKISOURCE_API}?{query}"], RAW / "gita" / f"discourse_{c:02d}.json",
                    lambda d: "parse" in json.loads(d))
        parser = Paragraphs()
        parser.feed(json.loads(raw)["parse"]["text"])

        title, pending, chapter = None, [], []  # chapter: (printed number, parts)
        for i, p in enumerate(parser.paras):
            if not p or p == "Layout 2" or p.isupper() or re.search("[ऀ-ॿ]", p):
                continue  # empty, layout marker, heading or Sanskrit
            if p.endswith("entitled:"):  # closing colophon names the chapter
                title = next(q for q in parser.paras[i + 1:] if q)
                break
            # A verse ends with its number, "(12)"; two are printed "(12" and "12)".
            m = re.search(r"\(\d+\)?$|\d+\)$", p)
            body = p[:m.start()] if m else p
            require(not re.search(r"\d", body), f"stray number in Gita {c}: {p!r}")
            if not m:
                pending.append(p)  # a speaker ("Arjuna said:") or part of a verse
                continue
            printed = int(m.group(0).strip("()"))
            if (c, len(chapter) + 1) in GITA_NUMBER_TYPOS and \
                    GITA_NUMBER_TYPOS[(c, len(chapter) + 1)] is None:
                # This verse lost its number: it is the pending text up to the
                # last paragraph that is not a speaker.
                k = 1 + max(j for j, q in enumerate(pending) if not q.endswith("said:"))
                chapter.append((None, pending[:k]))
                pending = pending[k:]
            chapter.append((printed, pending + [body]))
            pending = []
        require(len(chapter) == GITA_COUNTS[c - 1], f"Gita {c}: {len(chapter)} verses")
        for v, (printed, _) in enumerate(chapter, 1):
            expected = GITA_NUMBER_TYPOS.get((c, v), v)
            require(printed == expected, f"Gita {c}:{v} is printed as {printed}")
        for wrong, right in GITA_TITLE_FIXES.items():
            title = title.replace(wrong, right)
        # "THE YOGA OF ACTION." -> "The Yoga of Action"
        words = title.rstrip(".").lower().split()
        section = " ".join(w if i and w in SMALL_WORDS else w.capitalize()
                           for i, w in enumerate(words))
        for v, (_, parts) in enumerate(chapter, 1):
            verses.append(dict(book="gita", ref=f"{c}:{v}", chapter=c, verse=v,
                               section=section, text=clean(" ".join(parts))))
    return verses


def main():
    OUT.mkdir(exist_ok=True)
    stats, all_chars = {}, set()
    for name, build in (("gita", gita), ("quran", quran), ("bible", bible)):
        print(f"{name}...")
        verses = build()
        check_order(verses)
        refs = [v["ref"] for v in verses]
        require(len(refs) == len(set(refs)), f"duplicate refs in {name}")
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
