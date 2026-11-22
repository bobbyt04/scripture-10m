# Generated samples: Bhagavad Gita style

**Machine-generated imitations, not scripture.** This text was written by Scripture-10M, a 10.8M-parameter character-level model trained in this repo. It imitates the style of the translation it was trained on. It is not a quotation, a translation or a teaching of the Bhagavad Gita, and much of it is ungrammatical or makes no sense.

Checkpoint `out/best.pt` (step 11,000). Each sample is 500 generated tokens (characters) at temperature 0.8, no top-k, on the CPU. Seeds 1, 2 and 3 start from the book tag alone; seed 4 starts from the tag and a short prompt. These are the only samples generated for this file: nothing was picked or edited. The `<gita>`, `<quran>` and `<bible>` tags are the model's own verse markers; the model can switch to another book's tag on its own.

## Seed 1, no prompt

```bash
python sample.py --book gita --tokens 500 --temperature 0.8 --seed 1 --device cpu
```

```text
<gita>Those who invent a lie concerning what will be before you, in doubt concerning the Activity, and of the Praised Lord of the Worlds.
<gita>The from the multitude of merity, the divine producing, the divine painful, the self also the self, the self, the denuised restraint, the divine power, the self is dear to the deluded.
<gita>Self, world, passionate, the objects of the self; self-relinquishing, verily and angels make refrain to the deluded.
<gita>Unfavouring Me, is no piety, equal in Me, is called to the dest
```

## Seed 2, no prompt

```bash
python sample.py --book gita --tokens 500 --temperature 0.8 --seed 2 --device cpu
```

```text
<gita>Of every state thereof; nor in the senses of many beasts I destroy the world, and in the Reality of all beings;
<gita>Fallen by the world and the Self; the threefold is destroyed by Me; the righteous self controlled, the self-controlled, the same to be contrary and high, the beginning dark, the Self-controlled, and the Visible, the Self-knower.
<gita>The Blessed Lord said: This, by me he also hast made all known to be brought of unto me, O Moses; the supreme Pure Reason hath made understanding in the caus
```

## Seed 3, no prompt

```bash
python sample.py --book gita --tokens 500 --temperature 0.8 --seed 3 --device cpu
```

```text
<gita>Abandoning all yoga, wrought with the other and intercession of the Eternal.
<gita>The Indestructible, he, and also the indestructible, Having understanding, the self all come, give thanks and also to Me, My substance, insomuch that to the Egyptians, the highest Eternal, with the unmercy and subdued; the mighty Spirit is great, the supreme Spirit.
<bible>The seed of a wise man speaketh by a faithful other thing. I cannot see Me; and henceforth he seeketh from death.
<bible>The Lord GOD heareth the voice of his n
```

## Seed 4, prompt "Arjuna said:"

```bash
python sample.py --book gita --tokens 500 --temperature 0.8 --seed 4 --device cpu --prompt "Arjuna said:"
```

```text
<gita>Arjuna said: All are declared to indeed a supperme Self, though he greed to be brought down, sacrifice to the Edomites, after that he had great stiff. And when the spirit of his sons had been plucked up from the hand of all dead, then the Majesty of the Eternal.
<gita>They act from the pure Reason even to the destroyed day, at the resurrection of the mind and the self-comfort of desire, that desires reach not to the relinquishment of battle for the relinquishment of the essence of perfection.
<gita>The Blessed Lord s
```
