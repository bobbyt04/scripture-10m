// The verse-topic search in the browser: a port of the site contract of scripture/search.py, as written down
// step by step in search/FORMAT.md (steps 0 to 5, 8 and 9). suggest() reads what a visitor typed and returns
// topics only, never verses; topic() returns one topic's verses in the hand-set order; topics() lists them all.
//
// Plain TypeScript, no dependencies, no DOM. It runs on the data that scripts/extract.ts emits from
// search/artifacts/search.json. test/search.test.ts checks it against fixtures.json and against the Python
// Searcher on thousands of messages.

export type BookId = "gita" | "quran" | "bible";
export const BOOKS: readonly BookId[] = ["gita", "quran", "bible"];
export const BOOK_NAMES: Record<BookId, string> = { gita: "Bhagavad Gita", quran: "Quran", bible: "Bible" };

// ------------------------------------------------------------------ the data (search.json, FORMAT.md)

interface SuffixTry { only_if_ends?: string[]; strip: number; add?: string; undouble?: boolean }
interface SuffixRule { ends: string[]; not_ends?: string[]; min_len: number; try: SuffixTry[] }

export interface ThemeData {
  name: string;
  description: string;
  label?: string;
  blurb?: string;
  kind?: "comfort" | "self_directed";
  auto_show?: boolean;
  gentle_label?: string;
  gentle_blurb?: string;
  note?: string;
  phrases: string[];
}

export interface PoolVerse {
  ref: string;
  text: string;
  themes: string[];
  why: string;
  sensitive: boolean;
  priority?: Record<string, number>;
}

/** The parts of search.json that suggest(), topic() and topics() read. */
export interface SearchData {
  params: { min_theme_score: number; exclusive_themes: string[]; exclusive_any_score: string[];
    theme_fuzzy_min_len: number; max_query_chars: number; sensitive_theme_score: number };
  lexicon: { stopwords: string[]; irregular: Record<string, string>; suffix_rules: SuffixRule[]; not_inflected: string[] };
  vocabulary: Record<string, string>;
  themes: Record<string, ThemeData>;
  safety: {
    fuzzy: string[];
    crisis_message: string;
    emoji: Record<string, string>;
    idioms: string[];
    bereavement: string[];
    crisis: Record<string, string[]>;
    sensitive: Record<string, { themes: string[]; phrases: string[]; exclude?: Partial<Record<BookId, string[]>> }>;
    distress: { window: number; rules: string[] };
    /** Rules read on the message's own words, before idioms and bereavement are blanked (round 13). */
    unblanked?: { window: number; rules: string[] };
    suggest: { max_topics: number; max_topics_after_signal: number; min_sensitive_verses: number;
      never_after_signal: string[]; comfort_only_cues: Record<string, string[]>; signals: string[];
      safeguard_exempt: string[] };
  };
  pool: Record<BookId, PoolVerse[]>;
}

// ------------------------------------------------------------------ what it returns (FORMAT.md steps 8 and 9)

/** auto_show: the site may open this topic without a tap (themes.json "auto_show"; for a sensitive message, comfort topics only). */
export interface TopicLabel { id: string; label: string; blurb: string; auto_show: boolean }

export interface Suggestion {
  category: "crisis" | "sensitive" | "general" | "none";
  note: string | null;
  topics: TopicLabel[];
}

export interface TopicVerse { ref: string; text: string; why: string; themes: string[]; priority: number | null }

export interface TopicResult extends TopicLabel {
  note: string | null;
  sensitive: boolean;
  books: Record<BookId, { verses: TopicVerse[]; nothing: string | null }>;
}

export interface TopicListing extends TopicLabel {
  verses: Record<BookId, number>;
  sensitive_verses: Record<BookId, number>;
}

// ------------------------------------------------------------------ step 0: only the first 500 characters

/** query[:n] in Python: counted in code points, not UTF-16 units. */
export function cap(query: string, n: number): string {
  let i = 0;
  let count = 0;
  while (i < query.length && count < n) {
    const c = query.codePointAt(i)!;
    i += c > 0xffff ? 2 : 1;
    count++;
  }
  return query.slice(0, i);
}

// ------------------------------------------------------------------ step 1: words

// Python's \b counts letters, digits and the underscore of every script as word characters; JavaScript's \b
// only knows ASCII, so the boundaries are written out with Unicode property escapes.
const NOT_AFTER_WORD = "(?<![\\p{L}\\p{N}_])";
const NOT_BEFORE_WORD = "(?![\\p{L}\\p{N}_])";
const SPELLED: Record<string, string> = { "won't": "will not", "can't": "can not", "ain't": "is not" };
const RE_SPELLED = new RegExp(`${NOT_AFTER_WORD}(won't|can't|ain't)`, "gu");
const RE_NT = new RegExp(`n't${NOT_BEFORE_WORD}`, "gu");
const RE_SUFFIX = new RegExp(`'(m|re|s|ve|ll|d)${NOT_BEFORE_WORD}`, "gu");

// Contractions typed without an apostrophe. "wont" and "ill" are left alone: both are real words.
const NO_APOSTROPHE = new Map<string, string[]>(Object.entries({
  cant: "can not", dont: "do not", didnt: "did not", doesnt: "does not", isnt: "is not",
  wasnt: "was not", arent: "are not", werent: "were not", havent: "have not", hasnt: "has not",
  hadnt: "had not", couldnt: "could not", wouldnt: "would not", shouldnt: "should not",
  im: "i", ive: "i", youre: "you", theyre: "they", thats: "that", whats: "what", hes: "he", shes: "she",
}).map(([k, v]) => [k, v.split(" ")]));

export function words(text: string): string[] {
  let t = text.toLowerCase().replaceAll("’", "'");
  t = t.replace(RE_SPELLED, (m) => SPELLED[m]);
  t = t.replace(RE_NT, " not");
  t = t.replace(RE_SUFFIX, " ");
  const out: string[] = [];
  for (const w of t.match(/[a-z]+/g) ?? []) {
    const e = NO_APOSTROPHE.get(w);
    if (e) out.push(...e);
    else out.push(w);
  }
  return out;
}

const NUMBER_WORDS = new Map(
  "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen"
    .split(" ").map((w, i) => [String(i + 1), w]));

/** Each run of ASCII digits becomes " <word> ": its English word if it is exactly 1 to 17 (no leading zero), else "num". */
function spellNumbers(text: string): string {
  return text.replace(/[0-9]+/g, (m) => ` ${NUMBER_WORDS.get(m) ?? "num"} `);
}

const LEET: Record<string, string> = { "1": "i", "3": "e", "0": "o" };

/**
 * For the safety check only, before spellNumbers: the text in lower case, with a calorie count of 800 or less
 * ("300 calories") written " fewcalories ", and a 1, 3 or 0 right after a letter a-z read as i, e or o ("d1e").
 * A digit that starts a run of digits is left alone, so "took 30 tylenol" reads as before.
 */
function unmask(text: string): string {
  const t = text.toLowerCase().replace(/(?<![a-z0-9_])([0-9]{1,3}) *(?:calories|calorie|cals|cal|kcal)(?![a-z0-9_])/g,
    (m, n: string) => (Number(n) <= 800 ? " fewcalories " : m));
  return t.replace(/(?<=[a-z])[130]/g, (c) => LEET[c]);
}

// ------------------------------------------------------------------ step 2: phrases

interface Slot { exact: Set<string>; prefixes: string[] }
type Phrase = Slot[];

function compilePhrase(p: string): Phrase {
  return p.split(/\s+/).filter(Boolean).map((slot) => {
    const alts = slot.split("|");
    return { exact: new Set(alts.filter((a) => !a.endsWith("*"))), prefixes: alts.filter((a) => a.endsWith("*")).map((a) => a.slice(0, -1)) };
  });
}

const slotMatches = (tok: string, s: Slot) => tok !== "" && (s.exact.has(tok) || s.prefixes.some((p) => tok.startsWith(p)));

/** Positions where a phrase matches consecutive tokens. The empty word "" (a blanked word) never matches. */
function phraseStarts(tokens: string[], phrase: Phrase, tokset: Set<string>): number[] {
  // Only a speed-up: a slot of exact words none of which occurs in the tokens cannot match.
  for (const s of phrase) {
    if (s.prefixes.length === 0 && ![...s.exact].some((a) => tokset.has(a))) return [];
  }
  const out: number[] = [];
  for (let i = 0; i + phrase.length <= tokens.length; i++) {
    if (phrase.every((s, j) => slotMatches(tokens[i + j], s))) out.push(i);
  }
  return out;
}

const anyMatch = (tokens: string[], phrases: Phrase[], tokset: Set<string>) =>
  phrases.some((p) => phraseStarts(tokens, p, tokset).length > 0);

interface Rule { groups: Phrase[][]; vetoes: Phrase[]; window: number | null }

/** "~10 he|she + hits me, hit me ! hit me up": groups split by " + ", alternatives by ", ", vetoes after " ! ". */
function compileRule(r: string): Rule {
  let window: number | null = null;
  if (r.startsWith("~")) {
    const sp = r.indexOf(" ");
    window = parseInt(r.slice(1, sp), 10);
    r = r.slice(sp + 1);
  }
  const parts = r.split(" ! ");
  return {
    groups: parts[0].split(" + ").map((g) => g.split(", ").map(compilePhrase)),
    vetoes: parts.slice(1).map(compilePhrase),
    window,
  };
}

/** True if one match per group fits within the window and no veto lies within the span widened by the window. */
function ruleFires(tokens: string[], rule: Rule, defaultWindow: number, tokset: Set<string>): boolean {
  const window = rule.window ?? defaultWindow;
  const hits: [number, number][][] = [];
  for (const g of rule.groups) {
    const h: [number, number][] = [];
    for (const ph of g) for (const i of phraseStarts(tokens, ph, tokset)) h.push([i, i + ph.length]);
    if (h.length === 0) return false;
    hits.push(h);
  }
  const vetoHits: [number, number][] = [];
  for (const v of rule.vetoes) for (const i of phraseStarts(tokens, v, tokset)) vetoHits.push([i, i + v.length]);
  const walk = (k: number, lo: number, hi: number): boolean => {
    if (k === hits.length) return hi - lo <= window && !vetoHits.some(([s, e]) => lo - window <= s && e <= hi + window);
    return hits[k].some(([s, e]) => walk(k + 1, Math.min(lo, s), Math.max(hi, e)));
  };
  return walk(0, tokens.length, 0);
}

/** True if b is a with one letter added, removed or changed, or two neighbouring letters swapped. */
export function oneEditApart(a: string, b: string): boolean {
  if (a === b || Math.abs(a.length - b.length) > 1) return false;
  if (a.length === b.length) {
    const diff: number[] = [];
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff.push(i);
    return diff.length === 1 || (diff.length === 2 && diff[1] === diff[0] + 1 && a[diff[0]] === b[diff[1]] && a[diff[1]] === b[diff[0]]);
  }
  const [short, long] = a.length < b.length ? [a, b] : [b, a];
  for (let i = 0; i < long.length; i++) if (long.slice(0, i) + long.slice(i + 1) === short) return true;
  return false;
}

/** Python's round4: halves away from zero. */
const round4 = (x: number) => (Math.sign(x) * Math.floor(Math.abs(x) * 1e4 + 0.5)) / 1e4;

// ------------------------------------------------------------------ the searcher

export class Searcher {
  private readonly d: SearchData;
  private readonly stopwords: Set<string>;
  private readonly notInflected: Set<string>;
  private readonly irregular: Map<string, string>;
  private readonly vocabulary: Map<string, string>;
  private readonly themePhrases: [string, Phrase[]][];
  private readonly themeWords: string[];
  private readonly knownWords: Set<string>;
  private readonly emoji: [string, string][];
  private readonly blanks: Phrase[];
  private readonly distress: Rule[];
  private readonly unblanked: Rule[];
  private readonly crisis: [string, Phrase[]][];
  private readonly sensitive: [string, Phrase[]][];
  private readonly sensitiveExcludeAny: Record<BookId, Set<string>>;
  private readonly signals: Phrase[];
  private readonly safeguardSignals: Phrase[];
  private readonly never: Set<string>;
  private readonly comfortCues: Phrase[];
  private readonly comfortTopics: Set<string>;
  private readonly counts = new Map<string, Record<BookId, number>>();

  constructor(data: SearchData) {
    this.d = data;
    const lex = data.lexicon;
    this.stopwords = new Set(lex.stopwords);
    this.notInflected = new Set(lex.not_inflected);
    this.irregular = new Map(Object.entries(lex.irregular));
    this.vocabulary = new Map(Object.entries(data.vocabulary));
    this.themePhrases = Object.entries(data.themes).map(([t, d]) => [t, d.phrases.map(compilePhrase)]);
    const single = new Set<string>();
    this.knownWords = new Set(this.stopwords);
    for (const [, phrases] of this.themePhrases) {
      for (const ph of phrases) {
        if (ph.length === 1) for (const a of ph[0].exact) single.add(a);
        for (const slot of ph) for (const a of slot.exact) this.knownWords.add(a);
      }
    }
    this.themeWords = [...single].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    const s = data.safety;
    this.emoji = Object.entries(s.emoji);
    this.blanks = [...s.idioms, ...s.bereavement].map(compilePhrase);
    this.distress = s.distress.rules.map(compileRule);
    this.unblanked = (s.unblanked?.rules ?? []).map(compileRule);
    this.crisis = Object.entries(s.crisis).map(([t, ps]) => [t, ps.map(compilePhrase)]);
    this.sensitive = Object.entries(s.sensitive).map(([t, d]) => [t, d.phrases.map(compilePhrase)]);
    // topic(sensitive=true) does not know which sensitive topic led there, so it leaves out every topic's excludes.
    this.sensitiveExcludeAny = { gita: new Set(), quran: new Set(), bible: new Set() };
    for (const d of Object.values(s.sensitive)) for (const b of BOOKS) for (const r of d.exclude?.[b] ?? []) this.sensitiveExcludeAny[b].add(r);
    this.signals = s.suggest.signals.map(compilePhrase);
    // The signals that also count for the no-topic safeguard: all but those that occur in everyday idioms.
    const exempt = new Set(s.suggest.safeguard_exempt);
    this.safeguardSignals = s.suggest.signals.filter((x) => !exempt.has(x)).map(compilePhrase);
    this.never = new Set(s.suggest.never_after_signal);
    this.comfortCues = Object.values(s.suggest.comfort_only_cues).flat().map(compilePhrase);
    this.comfortTopics = new Set(Object.entries(data.themes).filter(([, d]) => d.kind === "comfort").map(([t]) => t));
  }

  get crisisMessage(): string {
    return this.d.safety.crisis_message;
  }

  private cap(query: string): string {
    return cap(query, this.d.params.max_query_chars);
  }

  // ----- step 3: safety

  /** C of FORMAT.md step 3: emoji and numbers in digits read as words, idioms and a third person's death blanked, crisis typos fixed. */
  crisisWords(query: string): string[] {
    let q = this.cap(query);
    for (const [ch, word] of this.emoji) q = q.replaceAll(ch, ` ${word} `);
    const toks = words(spellNumbers(unmask(q)));
    const out = [...toks];
    const tokset = new Set(toks);
    for (const ph of this.blanks) {
      for (const i of phraseStarts(toks, ph, tokset)) out.fill("", i, i + ph.length);
    }
    const fuzzy = this.d.safety.fuzzy;
    return out.map((t) => (t.length >= 5 ? fuzzy.find((f) => oneEditApart(t, f)) : undefined) ?? t);
  }

  classify(query: string): [category: "crisis" | "sensitive" | "general", topic: string | null] {
    const q = this.cap(query);
    const c = this.crisisWords(q);
    const cset = new Set(c);
    for (const [topic, phrases] of this.crisis) if (anyMatch(c, phrases, cset)) return ["crisis", topic];
    if (this.distress.some((r) => ruleFires(c, r, this.d.safety.distress.window, cset))) return ["crisis", "distress"];
    const w = words(q);
    const wset = new Set(w);
    // Read on the message's own words, before idioms and bereavement are blanked ("since my wife passed i just
    // want to be with her", "die for god").
    if (this.unblanked.some((r) => ruleFires(w, r, this.d.safety.unblanked!.window, wset))) return ["crisis", "distress"];
    for (const [topic, phrases] of this.sensitive) if (anyMatch(w, phrases, wset)) return ["sensitive", topic];
    return ["general", null];
  }

  // ----- step 4: themes

  private themeTokens(query: string): string[] {
    const n = this.d.params.theme_fuzzy_min_len;
    return words(this.cap(query)).map((t) =>
      t.length < n || this.knownWords.has(t) || this.termOf(t) !== null ? t : (this.themeWords.find((w) => oneEditApart(t, w)) ?? t));
  }

  /** [theme, score] best first; the top theme scores 1. Scores are not rounded. */
  queryThemes(query: string): [string, number][] {
    const toks = this.themeTokens(query);
    const tset = new Set(toks);
    const found = new Map<string, [number, number, string]>();
    for (const [t, phrases] of this.themePhrases) {
      for (const p of phrases) for (const i of phraseStarts(toks, p, tset)) found.set(`${i} ${i + p.length} ${t}`, [i, i + p.length, t]);
    }
    const all = [...found.values()];
    const raw = new Map<string, number>();
    for (const [s, e, t] of all) {
      if (!all.some(([s2, e2]) => s2 <= s && e <= e2 && e2 - s2 > e - s)) raw.set(t, (raw.get(t) ?? 0) + (e - s));
    }
    const top = Math.max(0, ...raw.values());
    return [...raw.keys()]
      .sort((a, b) => raw.get(b)! - raw.get(a)! || (a < b ? -1 : a > b ? 1 : 0))
      .map((t) => [t, raw.get(t)! / top]);
  }

  /** The themes that choose topics, in order: a Map keeps insertion order like a Python dict. */
  selectionThemes(query: string, category: string, topic: string | null): Map<string, number> {
    const p = this.d.params;
    const found = new Map(this.queryThemes(query));
    let themes = new Map([...found].filter(([, s]) => s >= p.min_theme_score));
    const exclusive = p.exclusive_themes.find((t) => themes.has(t) || (found.has(t) && p.exclusive_any_score.includes(t)));
    if (exclusive !== undefined) themes = new Map([[exclusive, 1.0]]);
    if (category === "sensitive" && topic !== null) {
      for (const t of this.d.safety.sensitive[topic].themes) if (!themes.has(t)) themes.set(t, p.sensitive_theme_score);
    }
    return themes;
  }

  // ----- step 5: a word's pool term

  private candidates(w: string): string[] {
    if (this.notInflected.has(w)) return [];
    for (const rule of this.d.lexicon.suffix_rules) {
      if (w.length < rule.min_len || !rule.ends.some((e) => w.endsWith(e)) || (rule.not_ends ?? []).some((e) => w.endsWith(e))) continue;
      const out: string[] = [];
      for (const c of rule.try) {
        if (c.only_if_ends && !c.only_if_ends.some((e) => w.endsWith(e))) continue;
        let stem = w.slice(0, -c.strip); // as w[:-strip] in Python
        if (c.undouble) {
          if (stem.length < 2 || stem[stem.length - 1] !== stem[stem.length - 2]) continue;
          stem = stem.slice(0, -1);
        }
        stem += c.add ?? "";
        if (stem.length >= 3) out.push(stem);
      }
      return out;
    }
    return [];
  }

  private normalise(word: string): string {
    let w = this.irregular.get(word) ?? word;
    for (let step = 0; step < 2; step++) {
      const next = this.candidates(w).find((c) => this.vocabulary.has(c));
      if (next === undefined) break;
      w = next;
      if (this.stopwords.has(w)) break;
    }
    return w;
  }

  termOf(word: string): string | null {
    const t = this.vocabulary.get(word) ?? this.vocabulary.get(this.normalise(word));
    return t ? t : null;
  }

  // ----- step 8: suggest()

  signal(query: string): boolean {
    const c = this.crisisWords(query);
    return anyMatch(c, this.signals, new Set(c));
  }

  /** FORMAT.md step 8.8: no theme phrase matches at all, and a signal that is not exempt does. */
  safeguard(query: string): boolean {
    if (this.queryThemes(query).length > 0) return false;
    const c = this.crisisWords(query);
    return anyMatch(c, this.safeguardSignals, new Set(c));
  }

  comfortCue(query: string): boolean {
    const w = words(this.cap(query));
    return anyMatch(w, this.comfortCues, new Set(w));
  }

  /** Topics to offer for a typed message. Never a verse, a ref or verse text. */
  suggest(query: string): Suggestion {
    const q = this.cap(query);
    const [category, topic] = this.classify(q);
    if (category === "crisis") return { category: "crisis", note: this.d.safety.crisis_message, topics: [] };
    const sensitive = category === "sensitive";
    const sug = this.d.safety.suggest;
    const need = sensitive ? sug.min_sensitive_verses : 1;
    let ids = [...this.selectionThemes(q, category, topic).keys()].filter((t) => this.topicCount(t, sensitive) >= need);
    let max = sug.max_topics;
    if (this.signal(q)) {
      ids = ids.filter((t) => !this.never.has(t));
      max = sug.max_topics_after_signal;
    }
    if (this.comfortCue(q)) ids = ids.filter((t) => this.comfortTopics.has(t));
    const topics = ids.slice(0, max).map((t) => this.label(t, sensitive));
    if (topics.length === 0 && !sensitive) {
      if (this.safeguard(q)) return { category: "crisis", note: this.d.safety.crisis_message, topics: [] };
      return { category: "none", note: null, topics };
    }
    return { category, note: null, topics };
  }

  // ----- step 9: topic() and topics()

  topic(id: string, opts: { k?: number | null; sensitive?: boolean } = {}): TopicResult {
    const sensitive = opts.sensitive ?? false;
    const rows = this.rows(id, opts.k ?? null, sensitive);
    const books = {} as TopicResult["books"];
    for (const b of BOOKS) {
      const verses = rows[b].map(([p, , i]) => {
        const v = this.d.pool[b][i];
        return { ref: v.ref, text: v.text, why: v.why, themes: v.themes, priority: p === 1000 ? null : p };
      });
      books[b] = { verses, nothing: verses.length ? null : `The ${BOOK_NAMES[b]} has no verse for this topic in the reviewed selection.` };
    }
    return { ...this.label(id, sensitive), note: this.d.themes[id].note ?? null, sensitive, books };
  }

  topics(): TopicListing[] {
    return Object.keys(this.d.themes).filter((t) => this.topicCount(t, false) > 0).map((t) => ({
      ...this.label(t, false),
      verses: this.bookCounts(t, false),
      sensitive_verses: this.bookCounts(t, true),
    }));
  }

  private bookCounts(theme: string, sensitive: boolean): Record<BookId, number> {
    const key = `${sensitive ? "s" : "g"} ${theme}`;
    let c = this.counts.get(key);
    if (!c) {
      const rows = this.rows(theme, null, sensitive);
      c = { gita: rows.gita.length, quran: rows.quran.length, bible: rows.bible.length };
      this.counts.set(key, c);
    }
    return c;
  }

  private topicCount(theme: string, sensitive: boolean): number {
    const c = this.bookCounts(theme, sensitive);
    return c.gita + c.quran + c.bible;
  }

  private label(theme: string, sensitive: boolean): TopicLabel {
    const d = this.d.themes[theme];
    const auto_show = Boolean(d.auto_show) && (!sensitive || this.comfortTopics.has(theme));
    if (sensitive && d.gentle_label !== undefined) return { id: theme, label: d.gentle_label, blurb: d.gentle_blurb!, auto_show };
    return { id: theme, label: d.label ?? d.name, blurb: d.blurb ?? d.description, auto_show };
  }

  /** Per book: [priority or 1000, score, pool index], sorted, the first k (all if k is null). */
  private rows(theme: string, k: number | null, sensitive: boolean): Record<BookId, [number, number, number][]> {
    if (!Object.hasOwn(this.d.themes, theme)) throw new Error(`unknown theme ${JSON.stringify(theme)}`);
    if (k !== null && !(Number.isInteger(k) && k >= 1)) throw new Error(`k must be a positive integer, got ${k}`);
    const out = {} as Record<BookId, [number, number, number][]>;
    for (const b of BOOKS) {
      const rows: [number, number, number][] = [];
      this.d.pool[b].forEach((v, i) => {
        const pos = v.themes.indexOf(theme);
        if (pos < 0 || (sensitive && (!v.sensitive || this.sensitiveExcludeAny[b].has(v.ref)))) return;
        rows.push([v.priority?.[theme] ?? 1000, round4(1 / (pos + 1)), i]);
      });
      rows.sort((x, y) => x[0] - y[0] || y[1] - x[1] || x[2] - y[2]);
      out[b] = k === null ? rows : rows.slice(0, k);
    }
    return out;
  }
}
