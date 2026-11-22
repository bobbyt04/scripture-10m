// MOCK advice search, a stand-in until the real search from PR 4 is ported.
//
// The verses are real (copied verbatim from data/processed, and a test checks that), but the topics, the
// keyword matching, the crisis check and the "why" notes here are placeholders written for the UI. They are
// NOT the reviewed pool, themes or distress gate of the real search, and must not ship as if they were.

import type { AdviceApi, Suggestion, TopicSummary, TopicVerses } from "./api.ts";

/** Placeholder crisis wording, from the draft in PR 4 (search/safety.json crisis_message). */
const CRISIS_NOTE =
  "We are not going to show verses for this one. What you wrote sounds heavy, and you deserve a real person " +
  "with you right now: please reach out to someone you trust today, or to local emergency services if you are " +
  "in danger.";

/** Placeholder only. The real distress gate has hundreds of reviewed rules. */
const MOCK_CRISIS = /\b(suicid\w*|kill (my ?self|me)|end (my|it all)|self[- ]?harm|hurt(ing)? myself|want to die|unalive)\b/i;

interface MockTopic extends TopicVerses {
  keywords: RegExp;
}

export const MOCK_TOPICS: MockTopic[] = [
  {
    id: "worry",
    label: "Worry and fear",
    blurb: "When the mind keeps running ahead to what might go wrong.",
    keywords: /\b(worr\w*|anxi\w*|afraid|scared|fear\w*|nervous|panic\w*|stress\w*|overwhelm\w*)\b/i,
    books: {
      gita: [
        { ref: "2:56", text: "He whose mind is free from anxiety amid pains, indifferent amid pleasures, loosed from passion, fear and anger, is called a sage of stable mind.", why: "Steadiness described as a mind that pain and worry do not shake." },
        { ref: "2:40", text: "In this there is no loss of effort, nor is there transgression. Even a little of this knowledge protects from great fear.", why: "Even a little understanding is said to guard against great fear." },
      ],
      quran: [
        { ref: "13:28", text: "Who have believed and whose hearts have rest in the remembrance of Allah. Verily in the remembrance of Allah do hearts find rest!", why: "Rest for the heart, found in remembrance." },
      ],
      bible: [
        { ref: "John 14:27", text: "Peace I leave with you, my peace I give unto you: not as the world giveth, give I unto you. Let not your heart be troubled, neither let it be afraid.", why: "A parting gift of peace to the troubled and afraid." },
        { ref: "Philippians 4:6", text: "Be careful for nothing; but in every thing by prayer and supplication with thanksgiving let your requests be made known unto God.", why: "\"Careful\" here means anxious: bring worries into prayer." },
      ],
    },
  },
  {
    id: "work",
    label: "Work and effort",
    blurb: "Doing the work well without being ruled by the outcome.",
    keywords: /\b(work\w*|job|career|boss|effort|exam\w*|study\w*|deadline\w*|project|fail\w*)\b/i,
    books: {
      gita: [
        { ref: "2:47", text: "Thy business is with the action only, never with its fruits; so let not the fruit of action be thy motive, nor be thou to inaction attached.", why: "Attend to the action itself, not to its rewards." },
        { ref: "3:8", text: "Perform thou right action, for, action is superior to inaction, and, inactive, even the maintenance of thy body would not be possible.", why: "Doing one's work is better than holding back from it." },
      ],
      quran: [
        { ref: "53:39", text: "And that man hath only that for which he maketh effort,", why: "What a person has is what they strive for." },
      ],
      bible: [
        { ref: "Proverbs 14:23", text: "In all labour there is profit: but the talk of the lips tendeth only to penury.", why: "Work bears fruit; talk alone does not." },
        { ref: "Galatians 6:9", text: "And let us not be weary in well doing: for in due season we shall reap, if we faint not.", why: "Keep going; the harvest comes in its season." },
      ],
    },
  },
  {
    id: "anger",
    label: "Anger",
    blurb: "When something or someone makes your blood boil.",
    keywords: /\b(ang(er|ry)\w*|furious|rage|mad at|annoy\w*|irritat\w*|resent\w*|argu\w*|fight\w*)\b/i,
    books: {
      gita: [
        { ref: "2:63", text: "From anger proceedeth delusion; from delusion confused memory; from confused memory the destruction of Reason; from destruction of Reason he perishes.", why: "How anger clouds judgement, step by step." },
      ],
      quran: [
        { ref: "3:134", text: "Those who spend (of that which Allah hath given them) in ease and in adversity, those who control their wrath and are forgiving toward mankind; Allah loveth the good;", why: "Holding back wrath is counted among good deeds." },
      ],
      bible: [
        { ref: "Proverbs 15:1", text: "A soft answer turneth away wrath: but grievous words stir up anger.", why: "A gentle reply calms; harsh words inflame." },
        { ref: "James 1:19", text: "Wherefore, my beloved brethren, let every man be swift to hear, slow to speak, slow to wrath:", why: "Listen first, speak later, anger last." },
      ],
    },
  },
  {
    id: "gratitude",
    label: "Gratitude",
    blurb: "Noticing and giving thanks for what is good.",
    keywords: /\b(grat\w*|thank\w*|bless\w*|appreciat\w*)\b/i,
    books: {
      gita: [],
      quran: [
        { ref: "14:7", text: "And when your Lord proclaimed: If ye give thanks, I will give you more; but if ye are thankless, lo! My punishment is dire.", why: "Thankfulness is met with more." },
      ],
      bible: [
        { ref: "1 Thessalonians 5:18", text: "In every thing give thanks: for this is the will of God in Christ Jesus concerning you.", why: "Thanks in every circumstance." },
      ],
    },
  },
  {
    id: "patience",
    label: "Patience in hard times",
    blurb: "Getting through a stretch that feels long and heavy.",
    keywords: /\b(patien\w*|hard time\w*|struggl\w*|difficult\w*|tough|endur\w*|wait\w*|hardship\w*)\b/i,
    books: {
      gita: [
        { ref: "2:14", text: "The contacts of matter, O son of Kunti, giving cold and heat, pleasure and pain, they come and go, impermanent; endure them bravely, O Bharata.", why: "Pleasure and pain come and go; bear them steadily." },
      ],
      quran: [
        { ref: "94:5", text: "But lo! with hardship goeth ease,", why: "Ease comes alongside hardship." },
        { ref: "2:153", text: "O ye who believe! Seek help in steadfastness and prayer. Lo! Allah is with the steadfast.", why: "Steadfastness and prayer as help." },
      ],
      bible: [
        { ref: "Romans 5:3", text: "And not only so, but we glory in tribulations also: knowing that tribulation worketh patience;", why: "Trouble, borne well, grows patience." },
      ],
    },
  },
  {
    id: "forgiving",
    label: "Forgiving someone",
    blurb: "Wanting to let go of a grudge.",
    keywords: /\b(forgiv\w*|grudge\w*|let go|betray\w*|hurt me|wronged)\b/i,
    books: {
      gita: [
        { ref: "16:3", text: "Vigour, forgiveness, fortitude, purity, absence of envy and pride - these are his who is born with the divine properties, O Bharata.", why: "Forgiveness listed among the divine qualities." },
      ],
      quran: [
        { ref: "42:43", text: "And verily whoso is patient and forgiveth - lo! that, verily, is (of) the steadfast heart of things.", why: "Patience and forgiveness as true resolve." },
      ],
      bible: [
        { ref: "Colossians 3:13", text: "Forbearing one another, and forgiving one another, if any man have a quarrel against any: even as Christ forgave you, so also do ye.", why: "Bear with one another and forgive." },
      ],
    },
  },
  {
    id: "decisions",
    label: "A hard decision",
    blurb: "Not knowing which way to go.",
    keywords: /\b(decid\w*|decision\w*|choice\w*|choos\w*|confus\w*|unsure|don'?t know what to do|torn)\b/i,
    books: {
      gita: [
        { ref: "2:7", text: "My heart is weighed down with the vice of faintness; my mind is confused as to duty. I ask thee which may be the better - that tell me decisively. I am thy disciple, suppliant to Thee; teach me.", why: "Arjuna, unsure of his duty, asks for clear counsel." },
      ],
      quran: [
        { ref: "3:159", text: "It was by the mercy of Allah that thou wast lenient with them (O Muhammad), for if thou hadst been stern and fierce of heart they would have dispersed from round about thee. So pardon them and ask forgiveness for them and consult with them upon the conduct of affairs. And when thou art resolved, then put thy trust in Allah. Lo! Allah loveth those who put their trust (in Him).", why: "Consult others, then act with trust once resolved." },
      ],
      bible: [
        { ref: "Proverbs 3:5", text: "Trust in the LORD with all thine heart; and lean not unto thine own understanding.", why: "Not leaning on one's own understanding alone." },
      ],
    },
  },
];

const summary = ({ id, label, blurb }: MockTopic): TopicSummary => ({ id, label, blurb });
const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

export const mockAdviceApi: AdviceApi = {
  isMock: true,
  async suggest(text: string): Promise<Suggestion> {
    await delay(120);
    if (MOCK_CRISIS.test(text)) return { category: "crisis", note: CRISIS_NOTE };
    const topics = MOCK_TOPICS.filter((t) => t.keywords.test(text)).slice(0, 4).map(summary);
    return topics.length ? { category: "topics", topics } : { category: "none" };
  },
  async topic(id: string): Promise<TopicVerses> {
    await delay(120);
    const t = MOCK_TOPICS.find((x) => x.id === id);
    if (!t) throw new Error(`unknown topic ${id}`);
    return { ...summary(t), books: t.books };
  },
  async allTopics(): Promise<TopicSummary[]> {
    return MOCK_TOPICS.map(summary);
  },
};
