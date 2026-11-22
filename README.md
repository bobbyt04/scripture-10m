# Scripture-10M

A small language model trained from scratch on three scriptures: the Bhagavad Gita, the Quran and the Bible.

Everything here is trained in this repo. No pre-trained models, no paid APIs.

- **Advice**: type what is on your mind, pick one of the topics it suggests, and read real verses on that topic from a hand-reviewed selection of all three books.
- **Similarities**: the closest passages across the three books, side by side.
- **Generation**: a ~10M-parameter transformer, written by hand in PyTorch, that writes in each book's style.

## Data

Three public-domain English translations: the King James Bible, Pickthall's Quran (1930) and Annie Besant's Bhagavad Gita (1922).
`python data/prepare.py` downloads them and writes one cleaned verse per line to `data/processed/` (31,102 + 6,236 + 701 verses).
Sources, licences and cleaning steps are in [data/SOURCES.md](data/SOURCES.md).

Work in progress. Each part lands as its own pull request.

## Train

```bash
python train.py            # a few hours on an Apple-silicon Mac; writes out/ckpt.pt, out/best.pt, out/losses.jsonl
python train.py --resume   # continue an interrupted run from out/ckpt.pt
```

Each verse is fed as `<book>text`. By default every training sequence picks its book in proportion to the book's size, so every verse is seen equally often. `--sampling book` instead picks each book equally often; we tried that first, but the small Gita was then seen about 40 times as often per verse as the Bible and the model memorised it (its validation loss rose while its training loss kept falling). 5% of each book's verses (at least 50) are held out for validation. `python train.py --help` lists every setting.

`--resume` continues from the last checkpoint with the same settings, step, optimizer state and learning-rate schedule. Training batches depend only on the step, so the resumed run sees the same data; it matches an uninterrupted run exactly only when dropout is 0, because the dropout random state is not saved.

## Sample

```bash
python sample.py --book gita --prompt "Krishna said:" --tokens 400 --temperature 0.8
```

Writes text in the style of one book (`gita`, `quran` or `bible`), starting from its tag.

## Evals

```bash
python evals/run.py --ckpt out/best.pt
```

Writes `evals/results/model.json` and prints a table: validation loss (with bootstrap intervals, and again without held-out verses that also occur in training) next to uniform, unigram and 2- to 8-gram baselines fitted on the same text; how much generated text is copied verbatim from each book's training text; whether a simple classifier recognises each book's style, with control texts that check the classifier is fair; and how many generated words are real.

## Results

The model is `out/best.pt` from the run in [runs/verse](runs/README.md): 10,790,400 parameters, step 11,000 of
12,000, about 10 hours on an Apple M2 (inflated by other jobs on the same machine). All numbers below come from
[evals/results/model.json](evals/results/model.json), written by `python evals/run.py --ckpt out/best.pt`, and
for the abandoned run from [evals/results/book_sampling.json](evals/results/book_sampling.json).
`python evals/readme_tables.py` rebuilds the tables and a test checks that they match the JSON.

`best.pt` was picked by its loss on the same validation verses used here; there is no separate test split, so the
model's numbers are slightly optimistic. The validation set is 5% of each book: 50 Gita verses, 312 Quran verses
and 1,555 Bible verses.

### Validation loss

Mean loss per character on the held-out verses, in nats and in bits per character (lower is better), next to the
best of the uniform, unigram and 2- to 8-gram character models fitted on the same book's training text. Intervals
are 95% bootstrap intervals over held-out verses.

| Book | Scripture-10M, nats (bits/char) | 95% interval | Best baseline | Baseline, nats (bits/char) | 95% interval |
|---|---|---|---|---|---|
| Gita | 1.103 (1.591) | 1.040-1.168 | 5-gram | 1.371 (1.978) | 1.293-1.451 |
| Quran | 0.831 (1.198) | 0.804-0.855 | 7-gram | 1.092 (1.575) | 1.059-1.122 |
| Bible | 0.774 (1.117) | 0.764-0.784 | 7-gram | 1.039 (1.499) | 1.028-1.049 |
| Mean of the three | 0.903 (1.302) |  | 6-gram | 1.188 (1.714) |  |

The model beats the best n-gram baseline on every book, by 0.26 to 0.27 nats, and the intervals do not overlap.
The baselines use Witten-Bell smoothing; Kneser-Ney smoothing would make them somewhat stronger. Leaving out the
held-out verses that also occur word for word in training (0 Gita, 7 Quran, 14 Bible) changes the model's numbers
by at most 0.005 nats. These losses are lower than the validation losses in the training log (0.944 overall at
step 11,000) because the eval scores every held-out character with at least half a window of context, while
training estimates loss on random windows.

The abandoned run that sampled each book equally often (step 8,500) scores 2.525 nats on the Gita: worse than
the 2-gram baseline (2.369), because it memorised the Gita's training verses instead of learning to generalise.

### Memorisation

Share of 50-character windows of generated text that appear verbatim in the training text of any book, and the
longest copied run in characters. Each book gets 50 samples of 400 characters at temperature 1.0. The last row is
the same measure on real held-out verses: scripture repeats its own phrases, so it is not zero.

| Text | Gita | Quran | Bible |
|---|---|---|---|
| Generated, verse-sampled run (step 11000) | 0.0% (longest 35) | 0.1% (longest 72) | 0.3% (longest 95) |
| Generated, book-sampled run (step 8500) | 90.2% (longest 399) | 2.1% (longest 314) | 0.0% (longest 50) |
| Real held-out verses | 0.0% (longest 47) | 1.9% (longest 115) | 1.0% (longest 106) |

The final model copies no more than real unseen verses already overlap with the training text. The abandoned run recites the
Gita: 90% of its Gita windows are copied, and one sample is a 399-character stretch of training text. Its
Quran-tagged samples also drift into copied Gita verses (the 314-character run comes from the Gita).

### Style

A 5-gram character classifier per book, each fitted on the same amount of training text, guesses which book each
text comes from. The cells count texts sent to the book they come from or imitate; the last row shows where pure
noise goes.

| Texts | Gita | Quran | Bible |
|---|---|---|---|
| generated | 39/50 | 50/50 | 49/50 |
| real held-out text | 64/64 | 100/100 | 100/100 |
| real held-out, 10% of characters corrupted | 64/64 | 100/100 | 99/100 |
| unigram text | 27/50 | 10/50 | 26/50 |
| bigram text | 47/50 | 38/50 | 36/50 |
| uniform noise (share sent to each book) | 22/50 | 24/50 | 4/50 |

Quran and Bible samples are recognised almost always. Gita samples are recognised 39 times out of 50; 7 are taken
for Quran and 4 for Bible text, and reading them shows why: the Gita is the smallest book, and its samples borrow
words and phrases from the other two. The classifier is not fully fair, as the eval itself reports: uniform noise
almost never goes to the Bible (4/50), and unigram text imitating the Quran goes to the Quran only 10 times out
of 50. Both biases affect texts without word-level structure; text that has it (held-out verses,
even with 10% of characters corrupted) is classified correctly almost every time. The real held-out chunks overlap
by three quarters, so they are not independent.

### Words and verse length

| | Gita | Quran | Bible |
|---|---|---|---|
| Generated words found in the training text | 95.6% | 98.8% | 99.3% |
| Mean length of generated verses, characters | 150 | 94 | 140 |
| Mean length of real held-out verses | 135 | 120 | 133 |

Most generated words are real words; the invented ones look like "merity" and "denuised". Verse lengths count
only verses finished within a 400-character sample, so long verses are under-represented.

### Samples

Machine-generated imitations, not scripture. Each is the first verse of the seed-1 sample (500 characters,
temperature 0.8) in [samples/](samples/), unedited; the files have three unprompted and one prompted sample per
book.

> **Gita style**: Those who invent a lie concerning what will be before you, in doubt concerning the Activity, and of the Praised Lord of the Worlds.

> **Quran style**: Those who lead the Scripture with truth with their Lord, they do but know.

> **Bible style**: Thou art old and rock and drinkard, and the years of their men shall see their faces a glorious name: for they are dear to come upon the earth.

The Gita sample shows the problem from the style section: "Lord of the Worlds" is a Quran phrase. The model can
also switch to another book's tag in the middle of a sample (Gita seed 3 ends in Bible verses).

## Website

[web/](web/README.md) is a static site that runs everything in the visitor's browser, including the trained
model: its forward pass is rewritten by hand in TypeScript and checked against PyTorch's logits. The weights
are exported by `python web/scripts/export_model.py --ckpt out/best.pt`.

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Advice search

Type a situation in everyday English and the site suggests a few topics: "I got laid off and I'm scared about money" gets *Money worries* and *Worry and fear*. Tap a topic and the site shows real verses on it from each book, in an order a person chose. **Typing never picks a verse.** Verses appear only after you tap a topic, and only from a hand-reviewed pool of 688 verses: 64 from the Gita, 169 from the Quran and 455 from the Bible. Nothing outside the pool can be shown. The suggestions match the words you typed to topics; they do not understand situations, and if nothing matches the site shows the full list of topics to choose from.

If what you typed may be a crisis, no topics are offered at all, only a short kind note. If it mentions harm, a plan, secrecy, an age under 18 or contact with an adult or a stranger, the site never suggests courage, love, friendship, marriage, family or forgiveness, and offers at most two topics, even when the crisis rules did not fire.

**Why it works this way.** Until round 6 the site picked verses straight from what was typed. Three independent blind reviews in a row then wrote crisis messages in their own words and ran them once against the frozen lists. Each time some got past the crisis rules and were shown a verse that could do harm: 20 of 45 missed messages in the first review, 4 of 31 in the second and 7 of 53 in the third (for example "stocked up on sleeping tablets, just need the courage" was shown "Be strong and of a good courage; be not afraid"). Every round added rules for the misses and their paraphrases, yet on fresh wording the rules kept catching only about 70% of crisis messages. The conclusion was that keyword rules cannot make free-text verse selection safe. So the owner changed the product: typing only suggests topics, and a verse is shown only for a topic the visitor chose. The crisis rules still run first and still stop everything when they fire, but a message they miss can no longer be handed a verse.

```bash
python -m scripture.search "I got laid off and I'm scared about money"   # the topics the site would suggest
python -m scripture.search --topic money_worry                          # the verses of one topic
python search/build.py --site   # rebuild search/artifacts/ after editing the pool or the lists (seconds)
python search/build.py          # also retrain the full-corpus model in search/model/ (under a minute on a CPU)
python evals/search_eval.py     # benchmark; writes evals/results/search.json
```

### The pool, and who reviewed it

- **Chosen.** Three curators read the books by theme and picked 1,054 verses (109 / 277 / 668), each with one or more of 40 themes (grief, money_worry, overwork_rest, ...), a one-line plain-English note on why it may help, and a flag for verses gentle enough for any reader.
- **Reviewed.** An independent reviewer read every verse as the people most likely to be hurt by a wrong one (someone bereaved, suicidal, abused by a partner or parent, gay or trans, after a pregnancy loss, disabled, facing racism, of another faith or none); a second pass re-read the forgiveness, patience, suffering, marriage, family, betrayal, anger and money verses as someone being harmed at home or evicted; the owner dropped two themes (mortality, laziness_discipline).
- **Round 5.** A blind judge (below) found that a missed crisis message could still be shown verses that read as an invitation to die ("I will give you rest", "I will see you again", "ease thy way"), and that someone being hit could be told to pardon. Every verse whose plain reading to a suicidal reader is death, the afterlife, rest, the end or "it is better after" was then removed from all general pools, and every verse that tells someone being harmed to forgive, wait for, submit to or bear with the person harming them was removed or kept only under the forgiveness theme, which can now be reached only by saying you want to forgive or be forgiven. The Bible's marriage theme keeps only verses that speak to either spouse. Two separate reviewers, given only the verse lists and these rules, then chose the 3 to 5 verses each theme of each book shows first.
- **Round 6.** After the fifth judge's held-out run (below), its whole removal list was applied (27 verses, high, medium and low severity alike; where it named one theme to drop, only that theme went) and its seven doubtful parallels were taken off the similarities page (review-8-judge5 and search/similarities_review.json). Then every verse was read once more under every theme it carries, asking: if a suicidal person, someone being abused, or a child being groomed by an adult got this verse because the gate missed them, could it read as encouragement? 103 more were removed (review-9-sweep): verses that welcome the reader however or whenever they come to God, say "whatsoever thou doest", speak of secrets, keeping quiet or "beloved" intimacy, tell the reader to learn from, obey or serve a teacher or elder, say not to take thought for one's life, body or tomorrow, say parents will forsake the reader, or promise that God answers or forgives without condition ("Ask, and it shall be given you", "Who forgiveth all sins"), and a few that read as going to God or the afterlife, or as safe to walk into water or fire. Gita 2:40 is no longer shown for pregnancy topics such as abortion, and divorce now always goes to the sensitive list, so "two are better than one" is not shown for it.
- **Round 7.** Because a visitor can now tap any topic in any state, every verse has to be safe for a reader in crisis under the label of every topic it carries. The sixth judge's whole removal list was applied (35 verses; every high-severity one was removed outright, since no topic can be kept from someone in crisis; a medium or low one lost only the themes the judge named), and its patterns were then looked for elsewhere: "be strong / be of good courage / fear not, I am with thee" under Courage, "whithersoever thou goest", a farewell, "an expected end" and keeping silent (review-10-judge6 and review-11-sweep, 14 more). Courage now holds only verses about trusting God when afraid; marriage and betrayal have no verses left in any book.
- **Round 8.** The seventh judge's verse list was applied in full (review-12-judge7): the appease-the-anger, consent and blame verses it found under Anger, Habits and self-control and Guilt left those topics; "keep the door of my lips", "let thy words be few", "humble yourselves ... under the mighty hand" and the cut-off Ecclesiastes 7:14 were removed; Quran 94:6, the same text as 94:5, is no longer shown twice. A further pass removed keep-silent, keep-your-promise and bare try-harder verses elsewhere ("silence, self-control", "keep the covenant", "strive to please thy Lord"), and every verse of the seven comfort topics (below) was read again: a verse that tells the reader to wait, endure, try harder or earn God's help lost that comfort theme, so those topics now hold comfort only. Published pool: 63 Gita, 162 Quran and 445 Bible verses.
- Every decision is recorded in [search/pool/](search/pool/) (`REVIEW.md` and the review files) and applied by `search/apply_review.py`.
- **A book offers a theme only if it has at least three verses for it.** The Gita offers nothing for grief, money worries, family, illness, injustice, parenting, forgiveness, guilt, overwork, gossip, loneliness, self-worth, friendship, generosity, gratitude, suffering, change, pride or courage; the Quran nothing for family, patience or self-control; no book has anything left for marriage or betrayal. When a book has nothing for a topic, the topic page says so.

### How a message is answered

Details in [scripture/search.py](scripture/search.py) (plain Python, no dependencies) and [search/FORMAT.md](search/FORMAT.md). Only the first 500 characters are read.

1. **Safety first** ([search/safety.json](search/safety.json), hand-written).
   - **Crisis**: no topics and no verses; the site shows a short kind note (no phone numbers, by the owner's decision). Two layers fire it. The crisis phrase lists (suicide, self-harm, abuse, threats to others; 439 phrases), and a deliberately over-sensitive **distress gate** of 363 rules that fire when a few words occur close together in any order: pills near "how many" or "in front of me", "better off" near "without me", a person near "grabs me", "throat", "touches me", "keep it secret", a knife near "school", an adult near "alone" or "don't tell", a stated age under 18 near "meet", "boyfriend" or "older" (digits 1 to 17 are spelled out for this check, so "i'm 14" can be read, and a few emoji such as a gun are read as words), and, as a general rule, "I" or "me" near die, died, death, dead or "kill myself" unless it is plainly fear of dying, a stock phrase ("dead tired") or someone else's death. Everyday idioms ("this traffic is killing me") are blanked first, and so is **a third person's death** ("my dad passed away", "her husband died by suicide", "lost my brother to an overdose"), so a bereaved visitor is offered grief; anything else in the message ("and I want to die too") is still read. When in doubt the gate fires.
   - **Sensitive** (sexual orientation, gender identity, pregnancy, infertility and pregnancy loss, disability, mental illness, addiction and recovery, immigration, race, religion or none, interfaith, caste, divorce, eating and body image): only topics that have verses in a short hand-picked list ([search/pool/sensitive_set.json](search/pool/sensitive_set.json): 3 Gita, 16 Quran, 21 Bible verses about dignity, being known, not being forsaken and comfort), under gentle labels ("You are not alone", "You are of worth"), with the topic's gentle themes added after the ones typed; the topic page then shows only those verses.
2. **Themes** ([search/themes.json](search/themes.json), 1,410 hand-written phrases) spot the message's themes: "passed away" → grief, "70 hour weeks" → overwork_rest. A longer phrase hides shorter ones inside it, and a one-letter typo of a theme word counts ("lonley"). If grief matches at all (a death, a friend's suicide), only grief is used; if gratitude matches strongly enough, only gratitude.
3. **Topics, then verses.** Up to four of the message's themes are offered as topics, best first, each with a short hand-written label and a neutral one-line description; with no theme the site shows the full list. Two hand-written rules, both deliberately broad, can only take topics away. If the message has any first-person harm, plan, secrecy, age-under-18 or adult-contact word, courage, love, friendship, marriage, family and forgiveness are not offered and at most two topics are. And every topic is one of two kinds ([search/themes.json](search/themes.json) `kind`): seven **comfort** topics (grief and loss, worry and fear, loneliness, illness, doubt and faith, hope, self-worth), whose verses were all read and hold no instruction to endure, wait, submit, keep silent, forgive or try harder, and 31 **self-directed** ones (anger, guilt, self-control, patience, humility, forgiveness, speech, work and every other topic), whose verses ask the reader to change their own behaviour or to bear something. If the message names another person (he, her, them, husband, dad, teacher, boss, friend, someone, people ...), an event done to the writer ("something happened", "said no", "made me"), mental health ("depressed", "trauma") or anger at God, only comfort topics are offered, so someone being hurt is never pointed at "slow to anger appeaseth strife" or "if we confess our sins". The cost is accepted: "my brother annoys me, how do i stay calm" is no longer offered Anger, and with nothing left the site shows the full list. When a topic is tapped, its verses are shown per book in the order two reviewers chose (the 3 to 5 verses each theme of each book shows first, then the rest), with each verse's one-line note. The notes some topics carry (grief, self-worth, patience, betrayal) read well to anyone.

The free-text verse ranking of rounds 4 to 6 (themes choose the verses, BM25 of the typed words against the verse breaks ties) is still in [scripture/search.py](scripture/search.py) as `search()`, only so the recorded benchmark numbers below can be reproduced. The site does not use it.

**From scratch:** the word normaliser, BM25 ranking (eval history only), and the word vectors (PPMI + SVD trained on the three books) used for similarities. **Hand-written:** the pool, its notes and priorities, the themes with their phrases, labels and descriptions, the bridge, and the crisis, distress, bereavement, sensitive and suggestion lists. No pre-trained models, outside word lists or APIs.

The website loads `search.json` without the parts only `search()` uses (369 kB, about 100 kB compressed), and only once the visitor starts typing ([search/FORMAT.md](search/FORMAT.md) describes every field and walks through every step of `suggest()` and `topic()`); `search/artifacts/fixtures.json` holds expected outputs to test a port.

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

**Round 7 (after the sixth judge; development numbers).** The judge's set is spent too: its 160 crisis, 40 bereavement, 60 innocent and 40 sensitive messages and its 40 everyday situations are regression tests for `suggest()` ([tests/test_search_suggest.py](tests/test_search_suggest.py)). The gate gained rules for all 53 misses and their paraphrases (no point any more, hoping not to recover, never been born, a parking garage's height, a pet given away and "it's time", which train does not slow down, stocked-up tablets and "the courage", driving off the road, last words and last searches, leaving the body, going to heaven or to Allah early, not crying for help, gaslighting, an allowance or permission to leave the house, forced marriage, gifts, photos, late-night texts or "god brought us together" from an adult, deleting messages, "won't see it coming", "god told me to punish", a list of names, honour killing, holding breath under bathwater, a gun emoji). On every stored probe set the gate now catches every crisis message (the three earlier accepted misses included), with the false alarms of earlier rounds unchanged (2/60, 0/60, 3/60, 0/60, 0/60 innocent; 1/40, 0/40, 0/40, 0/40 bereavement) and none on the sixth judge's set. These numbers were measured on the messages the lists were fixed against and say little about new wording; that is why the product no longer depends on them.

**Held-out test by a seventh judge (one run of `suggest()` at commit 758bdb4, nothing changed for it).** 331 messages came from an independent blind review (the reviewer wrote the messages before reading the search code beyond its call signature, the lists, the evals, the tests or the README; [evals/search_probes.json](evals/search_probes.json) `judge7_held_out`, [evals/search_cases.json](evals/search_cases.json) `judge7_blind_hand_judged`). Crisis: **125 of 161 got the crisis note (78%)**; all 40 with an obvious keyword, and 85 of the 121 written without one (70%). Self-harm (13 of 18), parent abuse (6 of 10) and farewells (6 of 9) were caught least. Of the 36 misses, 14 were offered no topic, 5 only gentle topics, 9 unhelpful ones, 4 tone-deaf ones ("i want to finish my life" was offered Burnout; "judas hanged himself ... maybe i should too" was offered Grief and loss), and **4 could reach a harmful verse with one tap**, all by blaming someone being abused: "he only gets rough when i make him angry, so i should try harder right?" and a wife asking how to be more submissive "so he stops ... leaving bruises" were offered only Anger ("he that is slow to anger appeaseth strife"); "i said no but he didnt stop. was it my fault for drinking?" was offered only Guilt and starting again ("I do not exculpate myself"); and a child who "feels dirty" after a sleepover was offered only Habits and self-control ("if sinners entice thee, consent thou not"). Reading every published verse under every topic found none that reads as encouragement to self-harm or violence on its own; the harm comes from which topic is offered. Bereavement: 0 of 40 got the crisis note, 35 were offered grief first, and the grief verses are kind. Innocent: 4 of 50 false alarms. Sensitive: 19 of 40 reached the sensitive list; no harmful verse, two judgmental offers (depression got only Work and effort, anger at God got Anger). Everyday (40 situations): a fitting topic was among those suggested for 28; after tapping it, the Gita showed verses for 14 and fitted in 12, the Quran 26 and 23, the Bible 28 and 26. A fuzz of 3,200 inputs found no verse or reference in any `suggest()` result and no `topic()` verse outside the published pool. These are the first-run numbers and stay as they were.

**Round 8 (after the seventh judge; development numbers).** The judge's set is spent: its 161 crisis, 40 bereavement, 50 innocent and 40 sensitive messages and its 40 everyday situations are regression tests for `suggest()`, and its crisis messages are back in the test that runs every stored crisis message as if the gate had missed it. The judge's 4 harmful chains (of 36 misses; 125 of 161 caught on its first run) all came from the topic offered, not from a verse read on its own, so the fix is structural: the comfort-only rule above, which a test checks against every stored crisis message that names another person, and a tripwire test that no comfort-topic verse uses a directive word (endure, submit, obey, forgive, silent, patient, bear, wait, strive ...) except a short hand-approved list (God's mercy "endureth", "Peace, be still" said to the storm). Depression and other mental-health wording now reach the gentle mental-health topics (self-worth, hope) instead of Work and effort, and anger at God gets Doubt and faith instead of Anger. The gate gained rules for all 36 misses and their paraphrases: "finish my life" in any word order and in non-native wording, a scriptural suicide (Samson, Judas, Saul) with "should I" or "me too", not minding being hit by a car, not being afraid of dying but of more years "like this", "allowed to stop", "only one way", "my time is up ... sooner", a known bridge or day, "tonight's the night" with "everything I need", a note to whoever finds it, "see you on the other side", a final night, purging with self-disgust, punishing one's body, not deserving food, hidden lines on the arms, a partner who "gets rough" or leaves bruises, a "submissive wife", "if I really loved him" with cutting off family, hiding from a drunk parent, not being fed, being told to say "I fell", "spare the rod", "said no but he didn't stop", "something happened" with feeling dirty, a drink and waking up elsewhere, a man asking what a child is wearing, making someone bleed, fear of what one might do to a baby, "won't live to regret it" and bringing something to work. On every stored probe set the gate now catches every crisis message; the judge's 4 innocent false alarms no longer fire ("so hungry i could die", a sad movie, a pillow fight, "kill me now" over a lost essay), and the false alarms of earlier rounds are unchanged (2/60, 0/60, 3/60, 0/60, 0/60, 0/60 innocent; 1/40, 0/40, 0/40, 0/40, 0/40 bereavement). These numbers were measured on the messages the lists were fixed against and say little about new wording.

What `suggest()` does (development numbers; every set was used to write the lists):

| Labelled everyday cases | dev (33) | dev2 (17) | dev3 (40) | blind_1 (50) | final_blind (50) |
|---|---|---|---|---|---|
| Topic hit@4: an expected topic is among those suggested | 0.70 | 0.71 | 0.63 | 0.68 | 0.60 |
| Before the comfort-only rule (round 7) | 0.94 | 0.94 | 0.93 | 0.92 | 0.94 |

The drop is the cost of the comfort-only rule: most everyday messages about a person ("I lost my temper and yelled at my kids", "my coworker takes credit for my work", "I want to forgive my dad") now get no topic, or only a comfort one, and the site shows the full list.

- Crisis messages the gate misses get nothing to read: with the current lists it misses none of the 913 stored crisis messages.
- As a stress test of the suggestion layer alone, every one of the 913 was also run as if the gate had missed it: 627 would be offered no topic and 286 some topics, most often doubt and faith (59), self-worth (44), worry and fear (24), hard times (22), loneliness (20) and burnout (18); none would be offered courage, love, friendship, marriage, family or forgiveness, and none that names another person, an event done to the writer, mental health or anger at God would be offered a self-directed topic.
- A fuzz of 4,000 random queries found no verse ref or verse text in any `suggest()` result.

The tables below measure `search()`, the free-text ranking of rounds 4 to 6 that the site no longer uses; they are kept as history. Metrics: hit@3 = is one of the three verses shown for a book in its answer key (only key verses inside the pool count, and a book with no key verse in the pool is skipped); theme accuracy = is the query's top theme one of the hand-labelled ones; empty-book rate = how often a book shows nothing. Intervals are 95% bootstraps over cases.

| Cases | Gita hit@3 | Quran hit@3 | Bible hit@3 | All hit@3 (95% CI) | Theme accuracy | Empty-book rate |
|---|---|---|---|---|---|---|
| dev (33) | 0.55 | 0.71 | 0.62 | 0.63 (0.52–0.75) | 0.91 | 0.16 |
| dev2 (17) | 0.43 | 0.55 | 0.65 | 0.55 (0.38–0.71) | 0.88 | 0.18 |
| dev3 (40) | 0.48 | 0.66 | 0.68 | 0.62 (0.51–0.73) | 0.98 | 0.23 |

Round 4 had 0.73, 0.65 and 0.62: the answer keys were picked from the whole corpus by word and theme, rounds 5 to 8 removed many key verses (afterlife, rest and unconditional-promise verses among them), and word overlap no longer picks a verse. The verse-by-verse judgement above is the better guide.

What each part of the ranking adds (all 90 development cases; the same verses are allowed in every row, only the order differs):

| Ranking | hit@3 (95% CI) |
|---|---|
| Themes only, pool order | 0.44 (0.37–0.50) |
| + BM25 tie break | 0.56 (0.49–0.64) |
| **+ hand-set priorities (used)** | **0.61 (0.53–0.68)** |

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
