# Data sources

All three texts are public-domain English translations. `data/prepare.py`
downloads them into `data/raw/` (not committed) and writes the cleaned verses
to `data/processed/` (committed). Run it with `python data/prepare.py`. It uses
only the Python standard library, and on the same raw files it always gives the
same output. If anything in a source looks unexpected, the script stops with an
error. Examples are a verse line it cannot parse, a verse out of order, a
missing or extra translation block, a character outside the allowed set, or a
correction that no longer applies. It never merges such text into a
neighbouring verse.

| Book | Translation | Verses | Chapters | Characters |
|---|---|---|---|---|
| Bible | King James Version (eBible.org text) | 31,102 | 1,189 in 66 books | 4,112,881 |
| Quran | Marmaduke Pickthall, 1930 | 6,236 | 114 surahs | 822,126 |
| Gita | Annie Besant, 4th edition, 1922 | 701 | 18 | 98,403 |

`tests/test_data.py` pins a SHA-256 checksum for each processed file. Any
change to the data, even a single character, fails the tests until the
checksum is updated on purpose. It also checks every chapter's verse count
against tables taken from independent sources, and checks the exact text of
about 20 verses per book.

## Bible: King James Version

- **Translation:** the Authorized (King James) Version, in the standardised
  1769 text.
- **Source:** eBible.org's `eng-kjv` edition, the "King James Version +
  Apocrypha" text supplied by the CrossWire Bible Society and eBible.org. We use
  the verse-per-line file, <https://ebible.org/Scriptures/eng-kjv_vpl.zip>
  (file `eng-kjv_vpl.txt`; the copy used is dated 2026-11-15). Edition page:
  <https://ebible.org/Scriptures/details.php?id=eng-kjv>. We keep only the 66
  books of the Protestant canon and drop the Apocrypha.
- **Licence:** eBible.org marks the text "Public Domain". In the UK, the Crown
  controls printing of the KJV through letters patent, so printing it there or
  importing printed copies needs permission. eBible.org notes that this "has no
  effect outside of the UK". The repository is hosted in the US. eBible.org
  attaches no other terms.
- **Why not Project Gutenberg #10:** an earlier version of this pipeline used
  it. A review found real wording errors in that file, for example:
  - "they look up Jonah" for "took up" (Jonah 1:15)
  - "morever" (Genesis 47:4)
  - "The tongue" for "Thy tongue" (Psalms 52:2)
  - "build" for "built" (Job 3:14)
  - "land of their enemies" for "hand" (Nehemiah 9:28)

  Gutenberg does not say which printing its file follows, so we no longer claim
  it is the 1769 Oxford text. All of the errors the review listed are correct
  in the current data, and the tests check several of them word for word.
- **Validation:** we compared all 31,102 verses word by word against three
  independent KJV copies. Before comparing, case, punctuation, hyphens and the
  `ae` ligature were normalised. The three copies are
  `thiagobodruk/bible` (`en_kjv.json`), `scrollmapper/bible_databases`
  (`KJV.json`) and Project Gutenberg #10. eBible's other KJV, `eng-kjv2006`,
  matched word for word, but it comes from the same CrossWire lineage, so it
  does not count as independent. The comparison copies were used only for
  checking; none of their text is in this repo. Results:
  - **Before corrections**, eBible was the odd one out in only 13 verses, where
    the other copies agreed with each other. In all 13 it splits a compound
    into two words: "can not" (6 verses), "where unto" (3), "where in" (2) and
    "what soever" (2), in Acts, Colossians, 1 and 2 Timothy and Hebrews. All
    three other copies, and the printed text, have one word. `prepare.py`
    corrects these 13 verses (`BIBLE_FIXES`).
  - **After corrections**, eBible is never the odd one out on wording. The
    remaining differences are known and deliberate:
    - **thiagobodruk: 157 verses differ.** 116 are Psalm titles, which eBible
      prints at the start of verse 1 ("A Psalm of David. The LORD is my
      shepherd...") and that copy leaves out. The rest are marginal notes that
      copy adds, such as "(or, Jehovah)".
    - **scrollmapper: 149 verses differ.** All are spelling: that copy writes
      "Judea" and "Cesar" where eBible has "Judaea" and "Caesar", adds the
      Hebrew letter headings in Psalm 119, and has a few variant spellings of
      its own. It includes the Psalm titles, as eBible does.
    - **Gutenberg #10: 282 verses differ.** These are the same 116 Psalm
      titles, which Gutenberg leaves out, plus Gutenberg's own errors and
      spellings ("morever", "first born", "cloak" for "cloke" and so on).
- **Verse counts:** the per-chapter verse counts of thiagobodruk, scrollmapper
  and Gutenberg #10 agree for all 1,189 chapters. That table is committed in
  the tests.
- **Cleaning:** dropped the Apocrypha, the square brackets that mark the
  translators' italic words (the words are kept), and the pilcrows (¶). Book
  codes are mapped to short names, so "JOH" becomes "John". Psalm titles stay
  at the start of verse 1, as eBible prints them. Refs look like `John 3:16`.

## Quran: Pickthall, *The Meaning of the Glorious Koran* (1930)

- **Translation:** Mohammed Marmaduke Pickthall, *The Meaning of the Glorious
  Koran: An Explanatory Translation*, first published in 1930.
- **Source:** Project Gutenberg ebook #16955, *Three translations of the Koran
  (Al-Qur'an) side by side* (last updated 12 December 2020).
  <https://www.gutenberg.org/cache/epub/16955/pg16955.txt>, with the PGLAF
  mirror <https://gutenberg.pglaf.org/cache/epub/16955/pg16955.txt> as a
  fallback. The file gives three translations of each verse: Yusuf Ali (`Y:`),
  Pickthall (`P:`) and Shakir (`S:`). **We keep only Pickthall's lines.**
- **Provenance (please note):** Gutenberg's own note says the text "originates
  from a file whose origins we don't know, found in many places on the
  Internet". It says it was then "re-proofed and corrected for Project
  Gutenberg against paper copies of the translations". It does not say which
  printing of Pickthall was used, so **the exact edition is unknown**.
- **Edition variants:** we compared all 6,236 verses with another digital
  Pickthall, Tanzil's `en.pickthall`, used only for checking. 73 verses differ.
  - 43 are "God" against "Allah": in 41 our copy has "God" where Tanzil has
    "Allah", and in 2 the reverse.
  - Most of the rest are spelling: "enroll"/"enrol", "subtle"/"subtile",
    "razed"/"rased" and the like.
  - A handful are real wording variants: 88:21 "an admonisher" (Tanzil: "a
    remembrancer"), 14:31 "bargaining" ("traffick"), 59:2 "reckoned"
    ("recked") and 23:111.

  We keep the Gutenberg wording as it is. Some of these readings may come from
  a later, lightly modernised printing.
- **Why it is public domain:** in the US, works published in 1930 entered the
  public domain on 1 January 2026, when the 95-year term ran out (Gutenberg
  already listed this file as public domain in the US). Pickthall died in 1936,
  so the translation is also public domain in countries whose term is the
  author's life plus 70 years (or less).
- **Source terms:** the Project Gutenberg licence covers only Gutenberg's
  header, footer and trademark. We remove all of these.
- **Sources we did not use:** Tanzil (`en.pickthall`) has a clean,
  verse-numbered file, but its terms say the translations are "for
  non-commercial purposes only", and they limit redistribution. Those terms
  clash with a public repo. English Wikisource's transcription of the 1930
  edition is not finished: only about 80 of the 114 surahs are there.
- **Numbering:** verses follow the standard Kufan numbering (6,236 verses). The
  per-surah counts match both the totals printed in the file and Tanzil's
  metadata. Refs look like `2:255`, and `section` holds the surah's
  transliterated name as the file gives it (for example "Al-Baqara").
- **Cleaning:** removed the Gutenberg header and footer, the surah headings and
  the Yusuf Ali and Shakir translations. Also removed the unnumbered Bismillah
  the file prints before most surahs, which is in Yusuf Ali's wording. Four
  verse labels in the file have typos (for example "0.033" for 17:33). So
  verses are numbered by position in each surah, and every block between labels
  must hold exactly one `Y:`, one `P:` and one `S:` line.

## Gita: Annie Besant's translation (4th edition, 1922)

- **Translation:** Annie Besant, *The Bhagavad-Gita, or the Lord's Song*, 4th
  edition, G. A. Natesan & Co., Madras, November 1922.
- **Source:** the English Wikisource transcription of that edition. The script
  fetches its 18 chapter pages, `Bhagavad-Gita (Besant 4th)/Discourse 1` to
  `Discourse 18`, through the MediaWiki API:
  `https://en.wikisource.org/w/api.php?action=parse&page=Bhagavad-Gita (Besant 4th)/Discourse 1&prop=text&format=json&formatversion=2`
  (the index page is <https://en.wikisource.org/wiki/Bhagavad-Gita_(Besant_4th)>).
- **Status:** Wikisource marks the pages "proofread", meaning one volunteer has
  checked them against the scan. That is not the same as "validated", which
  would need a second check. The download is not pinned to a revision: the
  chapter pages pull their text from separate page-by-page transcriptions, so a
  later run could pick up edits. The committed `gita.jsonl`, with its pinned
  checksum, is the reference copy.
- **Why it is public domain:** it was published in 1922, before 1931, so it is
  public domain in the US. Besant died in 1933, so it is also public domain in
  countries with a term of life plus 70 years. Wikisource tags it
  `{{PD/US|1933}}`. A faithful transcription of a public-domain text adds no
  new copyright. We credit Wikisource here anyway.
- **Why this translation, not Edwin Arnold's:** we were first told to use
  Arnold's *The Song Celestial* (1885). But it is free verse with no shloka
  numbers, so the verse search could not cite real `chapter:verse` refs.
  Besant's prose translation numbers every verse and maps one-to-one onto the
  Sanskrit shlokas printed beside it. We also looked at two other translations.
  K. T. Telang's 1882 version (Sacred Books of the East, vol. 8) carries heavy
  footnotes, and its main host blocks automated downloads. We found no clean
  digital text of Swami Swarupananda's 1909 version.
- **Numbering: chapter 13 runs one ahead of most modern editions.** Besant
  counts Arjuna's opening question ("Matter and Spirit, even the Field and the
  Knower of the Field...") as verse 13:1. Her edition therefore has **701**
  verses, not the usual 700, and chapter 13 has 35. **Besant's 13:2 to 13:35
  are 13:1 to 13:34 in editions without that verse.** We keep her numbering,
  because it is the real reference in this translation. Every other chapter
  matches the standard numbering. Refs look like `2:47`.
- **Section titles** come from Besant's closing colophon for each chapter (for
  example "Yoga by the Sankhya"). One correction: the transcription spells
  chapter 8's title "INDESCTRUCTIBLE", so `prepare.py` writes "The Yoga of the
  Indestructible Supreme Eternal" (`GITA_TITLE_FIXES`).
- **Cleaning:** removed the Sanskrit (Devanagari) text, footnote markers and
  footnotes, page numbers, headings, and the colophon that closes each chapter.
  Speaker lines such as "Arjuna said:" are kept at the start of the verse they
  introduce. The transcription has six numbering typos:
  - 3:42 is printed "(42" and 18:32 is printed "32)". The script reads both.
  - 10:31 and 11:8 have no number.
  - 17:19 is printed "(20)" and 18:14 is printed "(15)".

  The last four are listed in `GITA_NUMBER_TYPOS`. Any other mismatch between
  a printed number and a verse's position stops the script. Each chapter is
  also checked against the known verse count.

## Character normalisation (all books)

The model works on characters, so every verse is cut down to a small fixed set:

- Curly quotes become straight quotes. An em dash becomes ` - ` and an en dash
  becomes `-`. `ae` and `oe` ligatures are written out ("Cæsar" becomes
  "Caesar").
- Diacritics are dropped through Unicode NFKD decomposition, so "Madhusûdana"
  becomes "Madhusudana" and "Kuntî" becomes "Kunti". Besant already spells
  names in plain Latin letters ("Krishna", "Arjuna"), so the result is
  consistent.
- Whitespace is collapsed, so each verse is a single line.
- Anything outside letters, space and `.,;:!?'"()-` makes the script fail. That
  includes digits, which would mean a stray verse or footnote number. Verse
  text never contains `<`, `>` or a newline.

The final combined character set has **63 characters**: space, `!"'(),-.:;?`,
the capital letters A to Z except X, and the lowercase letters a to z. The
tests check this exact set.

## Output format

Each `data/processed/<book>.jsonl` file holds one verse per line:

```json
{"book": "bible", "ref": "John 3:16", "chapter": 3, "verse": 16, "section": "John", "text": "For God so loved the world, ..."}
```

`stats.json` gives the verse, character and chapter counts and the charset size
for each book, plus the combined charset. Training will render each verse as
`<book-tag>text\n` and sample the three books equally. That is why `<`, `>` and
newlines are kept out of verse text.

## Known typos in the source texts

`data/processed` keeps the texts exactly as the sources give them. Four typos
were found while the advice pool was reviewed. They are corrected only in the
display text of the pool (`search/pool/*.json`, field `fixes`), after checking
each against the verse:

| Verse | Source text | Shown as |
|---|---|---|
| Gita 3:20 | "the welfare of he world" | "the welfare of the world" |
| Gita 11:44 | "I worship as it fitting" | "I worship as is fitting" |
| Quran 2:274 | "and their shall no fear come upon them" | "and there shall no fear come upon them" |
| Quran 28:26 | "the best (man) that thou canst hire in the strong" | "... hire is the strong" |
