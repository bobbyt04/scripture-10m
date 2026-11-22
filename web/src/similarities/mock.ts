// MOCK parallels, a stand-in until the reviewed pairs from PR 4 land. The verse texts are verbatim from
// data/processed (a test checks this); the pairing and the notes are placeholders written for the UI.

import type { SimilarityTheme } from "./api.ts";

export const MOCK_SIMILARITIES: SimilarityTheme[] = [
  {
    name: "Stories told in two books",
    pairs: [
      {
        idea: "A king's dream of seven fat and seven lean cattle, and seven ears of corn, which Joseph explains.",
        source: "placeholder",
        verses: [
          { book: "bible", ref: "Genesis 41:26", text: "The seven good kine are seven years; and the seven good ears are seven years: the dream is one." },
          { book: "quran", ref: "12:43", text: "And the king said: Lo! I saw in a dream seven fat kine which seven lean were eating, and seven green ears of corn and other (seven) dry. O notables! Expound for me my vision, if ye can interpret dreams." },
        ],
      },
    ],
  },
  {
    name: "Anger and gentleness",
    pairs: [
      {
        idea: "Mastering one's own anger is praised above outward strength.",
        source: "placeholder",
        verses: [
          { book: "bible", ref: "Proverbs 16:32", text: "He that is slow to anger is better than the mighty; and he that ruleth his spirit than he that taketh a city." },
          { book: "quran", ref: "3:134", text: "Those who spend (of that which Allah hath given them) in ease and in adversity, those who control their wrath and are forgiving toward mankind; Allah loveth the good;" },
        ],
      },
      {
        idea: "A gentle word is worth more than a harsh one, or a gift given with a sting.",
        source: "placeholder",
        verses: [
          { book: "bible", ref: "Proverbs 15:1", text: "A soft answer turneth away wrath: but grievous words stir up anger." },
          { book: "quran", ref: "2:263", text: "A kind word with forgiveness is better than almsgiving followed by injury. Allah is Absolute, Clement." },
        ],
      },
      {
        idea: "Anger is the start of a slide; slowness to anger is the remedy.",
        source: "placeholder",
        verses: [
          { book: "gita", ref: "2:63", text: "From anger proceedeth delusion; from delusion confused memory; from confused memory the destruction of Reason; from destruction of Reason he perishes." },
          { book: "bible", ref: "James 1:19", text: "Wherefore, my beloved brethren, let every man be swift to hear, slow to speak, slow to wrath:" },
        ],
      },
    ],
  },
  {
    name: "Effort and endurance",
    pairs: [
      {
        idea: "What is yours is the effort itself.",
        source: "placeholder",
        verses: [
          { book: "gita", ref: "2:47", text: "Thy business is with the action only, never with its fruits; so let not the fruit of action be thy motive, nor be thou to inaction attached." },
          { book: "quran", ref: "53:39", text: "And that man hath only that for which he maketh effort," },
        ],
      },
      {
        idea: "Hardship is borne steadily, and it does not last unchanged.",
        source: "placeholder",
        verses: [
          { book: "gita", ref: "2:14", text: "The contacts of matter, O son of Kunti, giving cold and heat, pleasure and pain, they come and go, impermanent; endure them bravely, O Bharata." },
          { book: "bible", ref: "Romans 5:3", text: "And not only so, but we glory in tribulations also: knowing that tribulation worketh patience;" },
        ],
      },
    ],
  },
];
