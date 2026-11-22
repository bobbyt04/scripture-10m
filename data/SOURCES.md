# Data sources

All three texts are public-domain English translations. `data/prepare.py`
downloads them into `data/raw/` (not committed) and writes the cleaned verses
to `data/processed/` (committed). Run it with `python data/prepare.py`. It uses
only the Python standard library, and on the same raw files it always gives the
same output.

| Book | Translation | Verses | Chapters | Characters |
|---|---|---|---|---|
| Bible | King James Version | 31,102 | 1,189 in 66 books | 4,106,612 |
| Quran | Marmaduke Pickthall, 1930 | 6,236 | 114 surahs | 822,126 |
| Gita | Annie Besant, 4th edition, 1922 | 701 | 18 | 98,403 |

## Bible: King James Version

- **Translation:** the Authorized (King James) Version, 1611, in the standard
  1769 Oxford text.
- **Source:** Project Gutenberg ebook #10, *The King James Version of the Bible*
  (last updated 29 October 2024).
  <https://www.gutenberg.org/cache/epub/10/pg10.txt>, with the PGLAF mirror
  <https://gutenberg.pglaf.org/cache/epub/10/pg10.txt> as a fallback.
- **Why it is public domain:** it is more than 400 years old and has no
  copyright anywhere. The one caveat is the United Kingdom, where the Crown
  controls printing of the KJV under letters patent (a royal right, not
  copyright). That does not affect this repository, which is hosted in the US.
- **Source terms:** the Project Gutenberg licence covers only Gutenberg's
  header, footer and trademark. We remove all of these. The public-domain text
  itself has no restrictions.
- **Cleaning:** removed the Gutenberg header and footer, the table of contents,
  the testament headings, book titles and subtitles ("Otherwise Called: ...")
  and a `***` separator. Each verse is split on its `chapter:verse` marker.
  Full book titles are mapped to short names, so "The Gospel According to Saint
  John" becomes "John". Refs look like `John 3:16`.

## Quran: Pickthall, *The Meaning of the Glorious Koran* (1930)

- **Translation:** Mohammed Marmaduke Pickthall, *The Meaning of the Glorious
  Koran: An Explanatory Translation*, first published in 1930.
- **Source:** Project Gutenberg ebook #16955, *Three translations of the Koran
  (Al-Qur'an) side by side* (last updated 12 December 2020).
  <https://www.gutenberg.org/cache/epub/16955/pg16955.txt>, with the PGLAF
  mirror <https://gutenberg.pglaf.org/cache/epub/16955/pg16955.txt> as a
  fallback. The file gives three translations of each verse: Yusuf Ali (`Y:`),
  Pickthall (`P:`) and Shakir (`S:`). Gutenberg's note says it was checked
  against printed copies. **We keep only Pickthall's lines.** The other two
  translations are discarded.
- **Why it is public domain:** in the US, works published in 1930 entered the
  public domain on 1 January 2026, when the 95-year term ran out (Gutenberg
  already listed this file as public domain in the US). Pickthall died in 1936,
  so the translation is also public domain in countries whose term is the
  author's life plus 70 years (or less).
- **Source terms:** the Gutenberg licence, as for the Bible. We remove the
  Gutenberg material.
- **Sources we did not use:** Tanzil (`en.pickthall`) has a clean,
  verse-numbered file, but its terms say the translations are "for
  non-commercial purposes only", and they limit redistribution. Those terms
  clash with a public repo, so we do not use Tanzil. English Wikisource's
  transcription of the 1930 edition is not finished: only about 80 of the 114
  surahs are there.
- **Numbering:** verses follow the standard Kufan numbering used by modern
  Qurans (6,236 verses). Refs look like `2:255`, and `section` holds the surah's
  transliterated name as the file gives it (for example "Al-Baqara").
  Pickthall's printed edition numbers a few surahs a little differently, but
  this file has already been put into the standard numbering.
- **Cleaning:** removed the Gutenberg header and footer, the surah headings and
  the Yusuf Ali and Shakir translations. Four verse labels in the file have
  typos (for example "0.033" for 17:33). So verses are numbered by their
  position in each surah, and the script checks every surah against the verse
  total printed in its heading.

## Gita: Annie Besant's translation (4th edition, 1922)

- **Translation:** Annie Besant, *The Bhagavad-Gita, or the Lord's Song*, 4th
  edition, G. A. Natesan & Co., Madras, November 1922.
- **Source:** the proofread English Wikisource transcription of that edition.
  The script fetches its 18 chapter pages, `Bhagavad-Gita_(Besant_4th)/Discourse_1`
  to `Discourse_18`, through the MediaWiki API:
  `https://en.wikisource.org/w/api.php?action=parse&page=Bhagavad-Gita (Besant 4th)/Discourse 1&prop=text&format=json&formatversion=2`
  (the index page is <https://en.wikisource.org/wiki/Bhagavad-Gita_(Besant_4th)>).
- **Why it is public domain:** it was published in 1922, before 1931, so it is
  public domain in the US. Besant died in 1933, so it is also public domain in
  countries with a term of life plus 70 years. Wikisource tags it
  `{{PD/US|1933}}`. A faithful transcription of a public-domain text adds no
  new copyright. We credit Wikisource here anyway.
- **Why this translation, not Edwin Arnold's:** we were first told to use
  Arnold's *The Song Celestial* (1885). But it is free verse with no shloka
  numbers, so the verse search could not cite real `chapter:verse` refs.
  Besant's prose translation numbers every verse and maps one-to-one onto the
  Sanskrit shlokas. We also looked at two other translations. K. T. Telang's
  1882 version (Sacred Books of the East, vol. 8) carries heavy footnotes, and
  its main host blocks automated downloads. We found no clean digital text of
  Swami Swarupananda's 1909 version.
- **Numbering:** Besant's own verse numbers. Her edition has **701** verses,
  not the usual 700, because it counts Arjuna's opening question in chapter 13
  ("Matter and Spirit, even the Field and the Knower of the Field...") as verse
  13:1. Chapter 13 therefore has 35 verses, and its later refs are one higher
  than in editions that leave that verse out. Refs look like `2:47`. `section`
  is the chapter title from Besant's closing colophon (for example "Yoga by the
  Sankhya"). Chapter 8's title, "The Yoga of the Indesctructible Supreme
  Eternal", is spelled as in the transcription.
- **Cleaning:** removed the Sanskrit (Devanagari) text, footnote markers and
  footnotes, page numbers, headings, and the colophon that closes each chapter.
  Speaker lines such as "Arjuna said:" are kept at the start of the verse they
  introduce. The transcription has six numbering typos. 3:42 is printed "(42"
  and 18:32 "32)". 10:31 and 11:8 have no number. 17:19 is printed "(20)" and
  18:14 "(15)". The script therefore numbers verses by position and checks each
  chapter against the known verse count.

## Character normalisation (all books)

The model works on characters, so every verse is cut down to a small fixed set:

- Curly quotes become straight quotes. An em dash becomes ` - ` and an en dash
  becomes `-`. `ae` and `oe` ligatures are written out as two letters.
- Diacritics are dropped through Unicode NFKD decomposition, so "Madhusûdana"
  becomes "Madhusudana" and "Kuntî" becomes "Kunti". Besant already spells
  names in plain Latin letters ("Krishna", "Arjuna"), so the result is
  consistent.
- Whitespace is collapsed, so each verse is a single line.
- Anything outside letters, digits, space and `.,;:!?'"()-` makes the script
  fail rather than slip through. Verse text never contains `<`, `>` or a
  newline.

The final combined character set has **63 characters**: space, `!"'(),-.:;?`,
the capital letters A to Z except X, and the lowercase letters a to z. Digits
are allowed but never occur.

## Output format

Each `data/processed/<book>.jsonl` file holds one verse per line:

```json
{"book": "bible", "ref": "John 3:16", "chapter": 3, "verse": 16, "section": "John", "text": "For God so loved the world, ..."}
```

`stats.json` gives the verse, character and chapter counts and the charset size
for each book, plus the combined charset. Training will render each verse as
`<book-tag>text\n` and sample the three books equally. That is why `<`, `>` and
newlines are kept out of verse text.
