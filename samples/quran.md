# Generated samples: Quran style

**Machine-generated imitations, not scripture.** This text was written by Scripture-10M, a 10.8M-parameter character-level model trained in this repo. It imitates the style of the translation it was trained on. It is not a quotation, a translation or a teaching of the Quran, and much of it is ungrammatical or makes no sense.

Checkpoint `out/best.pt` (step 11,000). Each sample is 500 generated tokens (characters) at temperature 0.8, no top-k, on the CPU. Seeds 1, 2 and 3 start from the book tag alone; seed 4 starts from the tag and a short prompt. These are the only samples generated for this file: nothing was picked or edited. The `<gita>`, `<quran>` and `<bible>` tags are the model's own verse markers; the model can switch to another book's tag on its own.

## Seed 1, no prompt

```bash
python sample.py --book quran --tokens 500 --temperature 0.8 --seed 1 --device cpu
```

```text
<quran>Those who lead the Scripture with truth with their Lord, they do but know.
<quran>And make manifest unto make mention of the Scripture after them, for had they say: We lead their tokens unto us, but we deny them in truth and in truth error.
<quran>And they were but We drowned, gardens underneath which rivers flow, and their shields flowed down with seasons of the sea; and they said: Our Lord! Then verily we are comforted, for us we will not believe.
<quran>He said: I am comforted to be our fathers to do question 
```

## Seed 2, no prompt

```bash
python sample.py --book quran --tokens 500 --temperature 0.8 --seed 2 --device cpu
```

```text
<quran>Lo! those who are with the wrong-doers in the Hereafter they have all been with the lie and with their hearts. And Allah is the High Wise, the Absolute.
<quran>And those who are slaves of the men of old in whom ye are slaves There, and even they are as choice belonging with it, and make them to slay the birds thereof. Allah is Knower, Wise.
<quran>And how expulse is it, if ye do it not which ye have spent of Him the tokens of those who do wrong it; then know what can ye know?
<quran>And when the angels answered h
```

## Seed 3, no prompt

```bash
python sample.py --book quran --tokens 500 --temperature 0.8 --seed 3 --device cpu
```

```text
<quran>And they are dry for wrongdoing folk to help the doom of hell.
<quran>And they will feel from the race of that which is revealed, and they will understand.
<quran>They ask thee of the guidance, that they may give successive thanks.
<quran>And We (brought them) the race of gushing who was pleasure.
<quran>And We brought forth the command of the Children of Israel and the captains, We brought them forth from their own land and from their own land, and from their children. Lo! He is the Mighty, the Wise.
<quran>And the angels s
```

## Seed 4, prompt "O ye who believe!"

```bash
python sample.py --book quran --tokens 500 --temperature 0.8 --seed 4 --device cpu --prompt "O ye who believe!"
```

```text
<quran>O ye who believe! Allah is ever Mighty, Wise.
<quran>O ye who believe! Surely give me of death and believe in Allah and would have been given you good tidings of a right path.
<quran>O ye who believe! Be not afraid of your duty to Allah, and be not afraid of your partners!
<quran>They say: By Allah. Lo! the reed is the son of Allah. He putteth away his way in faith. Lo! those who disbelieve are wrong-doers.
<quran>Allah it is Who sendeth down for you and stretcheth down from you the mountains and the mountains and the children who were
```
