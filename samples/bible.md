# Generated samples: Bible style

**Machine-generated imitations, not scripture.** This text was written by Scripture-10M, a 10.8M-parameter character-level model trained in this repo. It imitates the style of the translation it was trained on. It is not a quotation, a translation or a teaching of the Bible, and much of it is ungrammatical or makes no sense.

Checkpoint `out/best.pt` (step 11,000). Each sample is 500 generated tokens (characters) at temperature 0.8, no top-k, on the CPU. Seeds 1, 2 and 3 start from the book tag alone; seed 4 starts from the tag and a short prompt. These are the only samples generated for this file: nothing was picked or edited. The `<gita>`, `<quran>` and `<bible>` tags are the model's own verse markers; the model can switch to another book's tag on its own.

## Seed 1, no prompt

```bash
python sample.py --book bible --tokens 500 --temperature 0.8 --seed 1 --device cpu
```

```text
<bible>Thou art old and rock and drinkard, and the years of their men shall see their faces a glorious name: for they are dear to come upon the earth.
<bible>And Jonathan the son of Shelemiah returned, and died.
<bible>And Jonathan and Abiram came unto his house, and said, Return unto me, and let us go to Jerusalem: and we took him by the hand of the priests and the Levites, and the priests, and all the people, and all the people, and all the service of the king's house, and all the people, and all the service of 
```

## Seed 2, no prompt

```bash
python sample.py --book bible --tokens 500 --temperature 0.8 --seed 2 --device cpu
```

```text
<bible>They have taken away the captivity of their captivity, and they have done it, and they have no doubt unto this evil thing: they have done themselves, and they have done it.
<bible>They have no believing in my heart, when I delighted in the land, when I came up out of the land of the living, saith the LORD.
<bible>I will be thine eyes upon thee, and thy righteousness upon thee, and thy right hand shall be above thee.
<bible>Thou shalt not eat the flesh in the midst of thee, nor the flesh of thy foot, nor the flesh
```

## Seed 3, no prompt

```bash
python sample.py --book bible --tokens 500 --temperature 0.8 --seed 3 --device cpu
```

```text
<bible>And there was a man between two cherubims of gold, and twenty cubits on the two right side; and the breadth of it was of silver, and the breadth of an homer of gold.
<bible>And this was the portion of the gate before the assembly of the tabernacle, that the work of the hoofs were of gold, and silver, and of gold, and of precious oil, and of raiment, and of gold, and of silver, and of iron, and of brass, and of oil, and of fine flour mingled with oil, and of fine flour mingled with oil, and of an hook 
```

## Seed 4, prompt "And the LORD said unto Moses,"

```bash
python sample.py --book bible --tokens 500 --temperature 0.8 --seed 4 --device cpu --prompt "And the LORD said unto Moses,"
```

```text
<bible>And the LORD said unto Moses, Aaron, what hast thou done unto thy land? seek out the midst of the fire? seeing thou hast brought in the captivity of Elam, and hast hid it in the midst of thee. So the LORD went out from the land of Egypt, and came into the land of Egypt.
<bible>And the LORD said unto Moses, Eliezer the son of Aaron thy brother, which smote thee among the children of Israel, shall be consumed that day and night for the children of Israel:
<bible>But the LORD hath chosen thee to be a prey, and thou shalt be no sin upon th
```
