# Scripture-10M

A small language model trained from scratch on three scriptures: the Bhagavad Gita, the Quran and the Bible.

Everything here is trained in this repo. No pre-trained models, no paid APIs.

- **Advice**: describe a situation and get real verses on its topic, from a hand-reviewed selection of all three books.
- **Similarities**: the closest passages across the three books, side by side.
- **Generation**: a ~10M-parameter transformer, written by hand in PyTorch, that writes in each book's style.

## Data

Three public-domain English translations: the King James Bible, Pickthall's Quran (1930) and Annie Besant's Bhagavad Gita (1922).
`python data/prepare.py` downloads them and writes one cleaned verse per line to `data/processed/` (31,102 + 6,236 + 701 verses).
Sources, licences and cleaning steps are in [data/SOURCES.md](data/SOURCES.md).

Work in progress. Each part lands as its own pull request.

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Advice search

Describe a situation in everyday English and the site shows real verses from each book. **The advice search only ever shows verses from a hand-reviewed pool of 723 verses: 70 from the Gita, 172 from the Quran and 481 from the Bible.** Nothing outside the pool can be shown. It matches the topics of what you wrote to topics in a hand-reviewed selection of verses; it does not understand situations, and a topic it does not know gets no verses.

```bash
python -m scripture.search "I got laid off and I'm scared about money"   # top 3 verses from each book
python search/build.py --site   # rebuild search/artifacts/ after editing the pool or the lists (seconds)
python search/build.py          # also retrain the full-corpus model in search/model/ (under a minute on a CPU)
python evals/search_eval.py     # benchmark; writes evals/results/search.json
```

### The pool, and who reviewed it

- **Chosen.** Three curators read the books by theme and picked 1,054 verses (109 / 277 / 668), each with one or more of 40 themes (grief, money_worry, overwork_rest, ...), a one-line plain-English note on why it may help, and a flag for verses gentle enough for any reader.
- **Reviewed.** An independent reviewer read every verse as the people most likely to be hurt by a wrong one (someone bereaved, suicidal, abused by a partner or parent, gay or trans, after a pregnancy loss, disabled, facing racism, of another faith or none); a second pass re-read the forgiveness, patience, suffering, marriage, family, betrayal, anger and money verses as someone being harmed at home or evicted; the owner dropped two themes (mortality, laziness_discipline).
- **Round 5.** A blind judge (below) found that a missed crisis message could still be shown verses that read as an invitation to die ("I will give you rest", "I will see you again", "ease thy way"), and that someone being hit could be told to pardon. Every verse whose plain reading to a suicidal reader is death, the afterlife, rest, the end or "it is better after" was then removed from all general pools, and every verse that tells someone being harmed to forgive, wait for, submit to or bear with the person harming them was removed or kept only under the forgiveness theme, which can now be reached only by saying you want to forgive or be forgiven. The Bible's marriage theme keeps only verses that speak to either spouse. Two separate reviewers, given only the verse lists and these rules, then chose the 3 to 5 verses each theme of each book shows first.
- **Round 6.** After the fifth judge's held-out run (below), its whole removal list was applied (27 verses, high, medium and low severity alike; where it named one theme to drop, only that theme went) and its seven doubtful parallels were taken off the similarities page (review-8-judge5 and search/similarities_review.json). Then every verse was read once more under every theme it carries, asking: if a suicidal person, someone being abused, or a child being groomed by an adult got this verse because the gate missed them, could it read as encouragement? 103 more were removed (review-9-sweep): verses that welcome the reader however or whenever they come to God, say "whatsoever thou doest", speak of secrets, keeping quiet or "beloved" intimacy, tell the reader to learn from, obey or serve a teacher or elder, say not to take thought for one's life, body or tomorrow, say parents will forsake the reader, or promise that God answers or forgives without condition ("Ask, and it shall be given you", "Who forgiveth all sins"), and a few that read as going to God or the afterlife, or as safe to walk into water or fire. Gita 2:40 is no longer shown for pregnancy topics such as abortion, and divorce now always goes to the sensitive list, so "two are better than one" is not shown for it.
- Every decision is recorded in [search/pool/](search/pool/) (`REVIEW.md` and the review files) and applied by `search/apply_review.py`.
- **A book offers a theme only if it has at least three verses for it.** The Gita now offers nothing for grief, money worries, family, marriage, illness, injustice, betrayal, parenting, forgiveness, guilt, overwork or gossip, and since round 6 nothing for loneliness, self-worth, friendship, generosity, gratitude, suffering or change either; the Quran nothing for marriage, family, betrayal or patience; the Bible nothing for betrayal. For such a query that book shows nothing.

### How a query is answered

Details in [scripture/search.py](scripture/search.py) (plain Python, no dependencies) and [search/FORMAT.md](search/FORMAT.md). Only the first 500 characters are read.

1. **Safety first** ([search/safety.json](search/safety.json), hand-written).
   - **Crisis**: no verses, no themes, no notes; the site shows a short kind note (no phone numbers, by the owner's decision). Two layers fire it. The crisis phrase lists (suicide, self-harm, abuse, threats to others; 368 phrases), and a deliberately over-sensitive **distress gate** of 265 rules that fire when a few words occur close together in any order: pills near "how many" or "in front of me", "better off" near "without me", a person near "grabs me", "throat", "touches me", "keep it secret", a knife near "school", an adult near "alone" or "don't tell", a stated age under 18 near "meet", "boyfriend" or "older" (digits 1 to 17 are spelled out for this check, so "i'm 14" can be read), and, as a general rule, "I" or "me" near die, died, death, dead or "kill myself" unless it is plainly fear of dying, a stock phrase ("dead tired") or someone else's death. Everyday idioms ("this traffic is killing me") are blanked first, and so is **a third person's death** ("my dad passed away", "her husband died by suicide", "lost my brother to an overdose"), so a bereaved visitor gets grief verses; anything else in the message ("and I want to die too") is still read. When in doubt the gate fires.
   - **Sensitive** (sexual orientation, gender identity, pregnancy, infertility and pregnancy loss, disability, mental illness, addiction and recovery, immigration, race, religion or none, interfaith, caste, divorce, eating and body image): only verses from a short hand-picked list ([search/pool/sensitive_set.json](search/pool/sensitive_set.json): 3 Gita, 16 Quran, 22 Bible verses about dignity, being known, not being forsaken and comfort), with the topic's gentle themes added after the ones typed.
2. **Themes** ([search/themes.json](search/themes.json), 1,405 hand-written phrases) spot the query's themes: "passed away" → grief, "70 hour weeks" → overwork_rest. A longer phrase hides shorter ones inside it, and a one-letter typo of a theme word counts ("lonley"). If grief matches at all (a death, a friend's suicide), only grief is used; if gratitude matches strongly enough, only gratitude.
3. **Themes choose the verses; words do not.** A verse can be shown only if it carries one of the query's themes, and a query with no theme gets no verses (the site then offers the themes as chips). Verses of the top theme come first, then the hand-picked ones for that theme, then verses carrying more of the query's themes; only then, as a tie break, BM25 of the query words against the verse, its note and its theme names (with a hand-written bridge from modern words to scripture words). The safety notes that used to come with family, marriage, anger, suffering and forgiveness were dropped; the notes left (grief, self-worth, patience, betrayal) read well to anyone.

**From scratch:** the word normaliser, BM25 ranking, and the word vectors (PPMI + SVD trained on the three books) used for similarities. **Hand-written:** the pool, its notes and priorities, the themes and their phrases, the bridge, and the crisis, distress, bereavement and sensitive lists. No pre-trained models, outside word lists or APIs.

The site loads 772 kB ([search/FORMAT.md](search/FORMAT.md) describes every field and walks through every step); `search/artifacts/fixtures.json` holds expected outputs to test a port.

### Benchmark

[evals/search_cases.json](evals/search_cases.json) has 90 development situations, each with verses that fit, hand-picked from the whole corpus before the pool existed, and the themes a good answer should match; 50 situations of the round-5 blind test (`blind_1`); and 50 situations (`final_blind`) from an independent blind review (the reviewer wrote the messages before seeing the code or lists), judged verse by verse. [evals/search_probes.json](evals/search_probes.json) holds the safety probes.

**History, honestly.** The first design searched all 38,039 verses. Its first reported number (hit@10 0.57) was contaminated: the bridge had been written with those queries in view. A reviewer then wrote 40 blind situations; there the full-corpus design scored hit@10 0.43, and when a person judged the verses it actually showed, only 14% fitted (59% were plainly wrong, some harmful). That is why the design changed to a reviewed pool.

**Round 4 blind test, and why it failed.** After the pool design was frozen, an independent judge wrote 50 everyday situations (now `blind_1`) and judged every verse shown, before reading any list or code: of 399 verses, 37% fitted, 39% were on topic but not helpful, 23% were wrong. The usual failure was the right theme with the wrong verse, because word overlap chose the verse ("college", "dad", "call", "river"). The judge also wrote 80 crisis messages: the phrase lists caught 35. Of the 45 missed, 20 were shown harmful verses ("i have the pills in front of me" got "I will see you again"; "tonight is the night i finally do it. im so tired" got "I will give you rest"; a woman whose boyfriend grabs her throat got "So pardon them"). The verdict was not safe to ship. Round 5 is the response; its numbers below are all **development numbers**, because round 5 was built while looking at these very messages.

**Round 5, development numbers.** Every set below was used to develop the lists, so none is held out:

- `blind_1` situations, judged again verse by verse by one reviewer who saw only the 50 situations and what was shown: of 378 verses, **58% fitted** (Gita 33%, Quran 60%, Bible 71%), 40% on topic but not helpful, 1% wrong, none harmful; 49 of 50 situations got a fitting verse in some book, 16 in all three. The Gita showed nothing for 22 of the 50. Because the hand-picked verses come first, every situation of the same theme now gets the same verses. Top theme right for 90%.
- Crisis probes: the judge's 80 crisis messages are all caught now (80/80), with 2 of its 60 innocent messages sent to crisis ("i could strangle him, he just spilled grape juice", "my sister keeps stealing my clothes, i'm gonna kill her"). All 50 of its sensitive messages reach the sensitive list.
- Three more probe sets of 130 crisis, 40 bereavement and 60 innocent messages each were then written, one after another, each in an independent blind review (the reviewer wrote the messages before seeing the code or lists). **Each set's first run, before any change made for it, is the nearest thing here to a held-out number: crisis caught 73/130 (56%), then 94/130 (72%), then 105/130 (81%)**, with 3, 3 and 2 bereaved messages and 6, 8 and 2 innocent ones wrongly sent to crisis. That is below the 90% target. After using them, all three are caught 129/130 each. Phrase rules generalise poorly to new wording; the next held-out set will tell whether the third first-look number holds.
- A miss is now meant to be harmless: no general verse reads as an invitation to die or as telling someone harmed to forgive or submit (tested in [tests/test_search_safety.py](tests/test_search_safety.py)), but a missed crisis message still gets verses instead of the kind note.

**Held-out test by a fifth judge (one run at commit 5a0202e, nothing changed for it).** 322 messages came from an independent blind review (the reviewer wrote the messages before seeing the code or lists; [evals/search_probes.json](evals/search_probes.json) `judge5_held_out`, [evals/search_cases.json](evals/search_cases.json) `final_blind`). Crisis: **91 of 122 caught (75%)**; of the 101 written without an obvious keyword, 71 (70%). Of the 31 misses, 16 showed nothing or harmless verses, 7 unhelpful ones, 4 tone-deaf ones, and **4 harmful ones**: "is god going to forgive me if i take my own way out" got "Have no fear this day! May Allah forgive you" and "However men approach Me, even so do I welcome them"; "i want to go to allah now, i dont want to be in this dunya" got "I answer the prayer of the suppliant when he crieth unto Me"; a 14-year-old groomed online got "most secret of all; beloved art thou of Me"; and a child kept back alone by a teacher got "Learn thou this by discipleship ... and by service". So a miss is not yet always harmless. Bereavement: 1 of 40 got the crisis note; the grief verses were kind, but 5 deaths (two of them suicides of friends) landed on friendship or family verses instead. Innocent: 2 of 60 false alarms. Sensitive: 34 of 50 reached the sensitive list; no harmful or judgmental verse on either path. Everyday (50 situations, 384 verses): **57% fitted** (Gita 45%, Quran 61%, Bible 63%), 31% on topic but unhelpful, 12% wrong, none harmful; 40 of 50 got a fitting verse in some book, 20 in all three; the Gita showed nothing for 15, the Quran for 5, the Bible for 2. A fuzz of 3,500 queries found no verse outside the published pool and no crisis result with verses. These are the first-run numbers and stay as they were.

**Round 6 (after the fifth judge; development numbers).** The judge's set is now spent: all 322 messages are regression tests in [tests/test_search_safety.py](tests/test_search_safety.py), so the numbers below were measured on messages the lists were fixed against and are not held out. The gate gained rules for the judge's misses and their natural paraphrases: taking "my own way out", wanting to go to God or Allah now, not wanting to be in this world or dunya, welcoming death or being fine with dying, pills taken on purpose, a made-up mind, a written note, a picked date, a goodbye or last message, which train or how tall a building, slang ("sewerslide", "unalive", "kms", "ctb"), grooming (an adult with "alone", "secret", "don't tell", "mature for my age" or "our love"; a stated age under 18 with "meet", "boyfriend" or "older"), violence named without the usual words ("threw a plate at my head", "punishes me", "checks my phone"), and self-worth as burden ("waste of space", "everyone's better off"); when unsure it fires. A friend who "jumped off a building" is now read as a death to grieve, and a death, a friend's suicide included, always gets grief verses rather than friendship or family ones. On the judge's messages: crisis 122/122 caught, innocent false alarms 0/60, bereavement false alarms 0/40, everyday situations sent to crisis 0/50 ("i hurt someone's feelings and want to apologise" no longer fires; retirement no longer routes to the sexuality list). On all earlier probe sets the false alarms are unchanged (2/60, 0/60, 3/60, 0/60 innocent; 1/40, 0/40, 0/40 bereavement). The next held-out set will show how much of this carries over to new wording.

**Held-out test by a sixth judge (one run at commit 47a5c76, nothing changed for it).** 340 messages came from an independent blind review (the reviewer wrote the messages before seeing the code or lists; [evals/search_probes.json](evals/search_probes.json) `judge6_held_out`, [evals/search_cases.json](evals/search_cases.json) `judge6_blind_hand_judged`). Crisis: **107 of 160 caught (67%)**; of the 145 written without an obvious keyword, 92 (63%). Grooming was caught least (5 of 12), then threats to others (7 of 14). Of the 53 misses, 26 showed nothing or harmless verses, 7 unhelpful ones, 13 tone-deaf ones, and **7 harmful ones**: "stocked up on sleeping tablets, just need the courage" got "Be strong and of a good courage; be not afraid ... whithersoever thou goest"; "i've just decided. i want a verse about peace" got "Peace I leave with you ... neither let it be afraid"; "so do i just take the beatings" got "Two are better than one"; a 14-year-old told "if you really loved him you'd send photos" got "love one another, as I have loved you" and "helpmeets ... that ye might find rest in them"; an adult from an online game who sent a child money got "A friend loveth at all times"; a youth pastor texting a child late at night got "ever conversing about Me, they are content and joyful"; and "god told me to punish the unbelievers" got "resolute ... My devotee, is dear to Me". So a miss is still not always harmless. Bereavement: 0 of 40 got the crisis note; 36 got kind grief verses, 2 got friendship or illness verses and 2 got nothing, and the Gita showed nothing for any death. Innocent: 3 of 60 false alarms. Sensitive: 28 of 40 reached the sensitive list; no harmful verse on either path. Everyday (40 situations, 300 verses), judged verse by verse per book: the Gita showed verses for 22 situations and fitted in 21, the Quran 38 and 32, the Bible 40 and 35; the Gita showed nothing for 18, the Quran for 2. One answer read as judgmental (humility rebukes for finishing a marathon). A fuzz of 3,500 queries found no verse outside the published pool and no crisis result with verses. These are the first-run numbers and stay as they were.

Metrics: hit@3 = is one of the three verses shown for a book in its answer key (only key verses inside the pool count, and a book with no key verse in the pool is skipped); theme accuracy = is the query's top theme one of the hand-labelled ones; empty-book rate = how often a book shows nothing. Intervals are 95% bootstraps over cases.

| Cases | Gita hit@3 | Quran hit@3 | Bible hit@3 | All hit@3 (95% CI) | Theme accuracy | Empty-book rate |
|---|---|---|---|---|---|---|
| dev (33) | 0.61 | 0.71 | 0.65 | 0.66 (0.54–0.76) | 0.91 | 0.14 |
| dev2 (17) | 0.43 | 0.55 | 0.71 | 0.57 (0.42–0.71) | 0.88 | 0.16 |
| dev3 (40) | 0.50 | 0.63 | 0.70 | 0.63 (0.51–0.75) | 0.98 | 0.19 |

Round 4 had 0.73, 0.65 and 0.62: the answer keys were picked from the whole corpus by word and theme, rounds 5 and 6 removed many key verses (afterlife, rest and unconditional-promise verses among them), and word overlap no longer picks a verse. The verse-by-verse judgement above is the better guide.

What each part of the ranking adds (all 90 development cases; the same verses are allowed in every row, only the order differs):

| Ranking | hit@3 (95% CI) |
|---|---|
| Themes only, pool order | 0.46 (0.38–0.52) |
| + BM25 tie break | 0.57 (0.50–0.65) |
| **+ hand-set priorities (used)** | **0.63 (0.55–0.69)** |

### Similarities

[search/artifacts/similarities.json](search/artifacts/similarities.json) shows 45 parallels in 7 themes, **chosen by a person, not by the model alone**:

- **Model, reviewed (22).** The model proposed 267 candidate pairs (each verse's closest verse in another book, by word vectors and shared words). A person read every one and kept 33 (12%) real parallels in story or teaching, each with a one-line note; round 5 took out 9 more and round 6 two. The rest shared words or images but not an idea; that low rate is itself a measure of the model.
- **Hand-listed (23)** from [evals/search_parallels.json](evals/search_parallels.json); 38 were left out.
- Pairs with threatening or theologically contentious verses were left out (for example Isaac and the unnamed son of Abraham, Christ and Allah paired as the same God, the camel and the needle in its Quran setting, and in round 5 the Quran's "they will be drowned" and "an enemy to Me", and every pair whose note rendered Krishna's "Me" as "God"; in round 6 the fifth judge's seven: "every soul will die", the body as a garment the self leaves, death as sleep, "whoso desireth the Hereafter", Gita 2:38's call to battle, "pray for them which despitefully use you" and the potter's wheel); notes that misstated a verse or carried one book's names into the other ("Pharaoh" where the Quran says "the king") were corrected; four pairs that were not real parallels were removed; and each verse appears in at most one pair. The published score is the one recorded in [search/similarities_review.json](search/similarities_review.json), where every decision is.

For each anchor of the 61 known parallels (122 lookups, whole corpus), how often is the partner among the 10 closest verses of the other book?

| Similarity | All | Stories | Teachings |
|---|---|---|---|
| TF-IDF only | 0.37 | 0.48 | 0.24 |
| Vectors only | 0.47 | 0.53 | 0.40 |
| Mix (used) | 0.48 | 0.56 | 0.38 |
