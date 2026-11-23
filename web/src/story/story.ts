// "How it works": short sections drawn from real artefacts in the repository. Every number here comes from
// src/generated/repo.json (written at build time by scripts/extract.ts from the eval JSON, training logs,
// data stats and source files) or from the small JSON files beside this one.

import type { RepoData, Segment } from "../../scripts/repo-data.ts";
import { BOOK_NAMES, type BookId } from "../advice/api.ts";
import { fmtInt, h } from "../dom.ts";
import manifest from "../model/manifest.json";
import repoJson from "../generated/repo.json";
import adviceHistory from "./advice-history.json";
import { NO_TOPIC_EXAMPLE } from "./examples.ts";
import { intervalChart, legend, lineChart, missChart, responsive } from "./charts.ts";
import { modelSizeMB, toLines } from "../generate/ui.ts";
import testCounts from "./test-counts.json";

const repo = repoJson as unknown as RepoData;
const BOOKS: BookId[] = ["gita", "quran", "bible"];
const COLOR: Record<BookId, string> = { gita: "var(--gita)", quran: "var(--quran)", bible: "var(--bible)" };
const REPO_URL = "https://github.com/bobbyt04/scripture-10m";
const blob = (path: string) => `${REPO_URL}/blob/main/${path}`;

const pct = (x: number, digits = 1) => `${(100 * x).toFixed(digits)}%`;
const nats = (x: number) => x.toFixed(3);

function section(id: string, title: string, ...body: (Node | null)[]): HTMLElement {
  return h("section", { class: "section", id }, h("h2", {}, title), ...body);
}

/** A recorded sample from samples/bible.md (seed 1), and the link that runs the model live in the browser. */
function recordedSample(): HTMLElement[] {
  const sm = repo.samples.bible;
  return [
    p("A recorded sample, Bible style (", src(sm.file), `, seed ${sm.seed}, temperature ${sm.temperature}, step ${fmtInt(sm.step)}). Machine-generated imitation, not scripture:`),
    h("div", { class: "sample", "data-sample": "" }, toLines(sm.text, "bible").map((l) => h("p", {}, l.text))),
    p(h("button", { type: "button", class: "textlink", "data-gen-run": "" }, `Run it live (${modelSizeMB} MB)`),
      h("span", { class: "muted" }, " downloads the model once and writes in your browser, under the box at the top.")),
  ];
}

const p = (...c: (Node | string)[]) => h("p", {}, ...c);
const src = (path: string, label = path) => h("a", { href: blob(path) }, h("code", {}, label));
const figure = (caption: (Node | string)[], ...content: Node[]) =>
  h("figure", {}, ...content, h("figcaption", {}, ...caption));
const sub = (text: string) => h("h3", {}, text);
const table = (label: string, head: (string | Node)[], rows: (string | Node)[][], cls = "") =>
  h("div", { class: "table-wrap", tabindex: "0", role: "region", "aria-label": label },
    h("table", { class: cls },
      h("thead", {}, h("tr", {}, head.map((c, i) => h("th", { scope: "col", class: i ? "num" : "" }, c)))),
      h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c, i) => (i ? h("td", { class: "num" }, c) : h("th", { scope: "row" }, c))))))));

// ------------------------------------------------------------------ model

function codeBlock(): HTMLElement {
  const c = repo.code;
  const notesByLine = new Map(c.notes.map((n, i) => [n.line, i + 1]));
  const seg = ([cls, text]: Segment) => (cls ? h("span", { class: `tok-${cls}` }, text) : document.createTextNode(text));
  return h("figure", {},
    h("pre", { class: "code", tabindex: "0", "aria-label": `Excerpt of ${c.file}` },
      h("code", {}, c.lines.map((segs, i) => {
        const line = c.firstLine + i;
        const n = notesByLine.get(line);
        return h("span", { class: "code__line" },
          h("span", { class: "code__ln", "aria-hidden": "true" }, n ? `[${n}]` : String(line)),
          ...segs.map(seg), "\n");
      }))),
    h("figcaption", {}, src(c.file, `${c.file}, lines ${c.firstLine}–${c.firstLine + c.lines.length - 1}`), "."),
    h("ol", { class: "code-notes" }, c.notes.map((n) => h("li", {}, n.note))));
}

function modelSection(): HTMLElement {
  const cfg = repo.params.config;
  const total = repo.params.total;
  const show = (t: string) => (t === "\n" ? "↵" : t === " " ? "␣" : t);
  const verses = BOOKS.reduce((a, b) => a + repo.stats[b].verses, 0);
  return section("built-model", "Model",
    p(`A decoder-only transformer with ${cfg.n_layer} layers, ${cfg.n_head} heads, ${cfg.n_embd}-dimensional embeddings and a ${cfg.block_size}-character context: ${fmtInt(total)} parameters. `,
      "Model, tokenizer, training loop and evals are implemented from scratch in PyTorch and are in the repo. The architecture follows Andrej Karpathy’s “Let’s build GPT” lecture. No pretrained weights, no external model APIs. ",
      `The browser runs the same forward pass, reimplemented in TypeScript, on int8 weights (${(manifest.bytes / 1e6).toFixed(1)} MB). `,
      `Trained on ${fmtInt(verses)} verses.`),
    h("p", { id: "why" }, "It predicts one character at a time, so it can continue text in the style of these books; it cannot answer questions. ",
      "The Verses search is a separate topic lookup, not the model."),
    ...recordedSample(),
    p("Attention is written out step by step rather than calling PyTorch’s built-in attention:"),
    codeBlock(),
    table("Where the parameters are", ["Part", "Parameters", "Share"],
      [...repo.params.parts.map((part) => [h("span", {}, part.label, h("span", { class: "muted" }, `, ${part.detail}`)), fmtInt(part.count), pct(part.count / total)]),
        [h("strong", {}, "Total"), h("strong", {}, fmtInt(total)), ""]]),
    p(`The vocabulary is ${repo.vocab.length} tokens: 63 characters, newline (↵) and one tag per book.`),
    h("p", { class: "vocab", "aria-label": "The vocabulary" },
      repo.vocab.map((t, i) => h("span", {}, h("code", {}, show(t)), h("sub", {}, String(i))))),
  );
}

// ------------------------------------------------------------------ data

function dataSection(): HTMLElement {
  const st = repo.stats;
  const translations: Record<BookId, string> = { gita: "Besant, 1922", quran: "Pickthall, 1930", bible: "King James Version, eBible.org" };
  return section("built-data", "Data",
    p("Three public-domain English translations. A script downloads them, strips headers, footnotes and Sanskrit, reduces the text to 63 characters and writes one verse per line. It stops on anything it does not expect instead of guessing."),
    table("Verses per book", ["Book", "Verses", "Chapters", "Characters"],
      BOOKS.map((b) => [h("span", {}, BOOK_NAMES[b], h("span", { class: "muted" }, `, ${translations[b]}`)), fmtInt(st[b].verses), fmtInt(st[b].chapters), fmtInt(st[b].characters)])),
    p(`My first Bible text, Project Gutenberg #10, had wording errors (${repo.kjvErrors.map((e) => e.text).join("; ")}). `,
      "I replaced it with eBible.org’s text and checked every verse against three independent copies; a word split in two (“can not”) in 13 verses is corrected. Details: ",
      src("data/SOURCES.md"), "."),
  );
}

// ------------------------------------------------------------------ training

function trainingSection(): HTMLElement {
  const t = repo.training;
  const a = t.trainArgs as Record<string, number>;
  const last = t.verse[t.verse.length - 1];
  const hours = Math.floor(last.elapsed_s / 3600);
  const mins = Math.round((last.elapsed_s % 3600) / 60);
  const ev = repo.evals;
  const box = h("div", { class: "chart-box" });
  responsive(box, (w) => lineChart({
    title: "Validation loss per book during training, with the best n-gram baseline for each book",
    series: BOOKS.map((b) => ({ label: BOOK_NAMES[b], color: COLOR[b], points: t.verse.filter((r) => r.step > 0).map((r) => [r.step, r.val[b]] as [number, number]) })),
    hlines: BOOKS.map((b) => ({ y: ev.loss[b].baseline.nats, color: COLOR[b], below: b === "bible", label: w < 520 ? "" : `${BOOK_NAMES[b]}: best n-gram (${ev.loss[b].baseline.name}), ${nats(ev.loss[b].baseline.nats)}` })),
    x: { domain: [0, a.max_iters], ticks: [0, 3000, 6000, 9000, 12000], format: (v) => fmtInt(v), label: "training step" },
    y: { domain: [0.6, 2.0], ticks: [0.6, 1.0, 1.4, 1.8], format: (v) => v.toFixed(1), label: "loss (nats)" },
  }, w));
  return section("built-training", "Training",
    p(`${fmtInt(a.max_iters)} steps of ${a.batch_size} windows of ${a.block_size} characters. Each window comes from one book, chosen in proportion to the book’s size. `,
      `AdamW, learning rate warmed up to ${a.lr} and decayed to ${a.min_lr}, dropout ${a.dropout}. It took ${hours} h ${mins} min on an Apple M2 that was running other jobs. I kept step ${fmtInt(t.bestStep)}, the lowest validation loss.`),
    figure([
      "Solid: validation loss per book from ", src("runs/verse/losses.jsonl"), ", estimated on random windows. Dashed: the best n-gram model for each book, fitted on the same training text. ",
      "Scored on every held-out character, the model gets ", BOOKS.map((b) => `${BOOK_NAMES[b]} ${nats(ev.loss[b].model.nats)}`).join(", "), ".",
    ], legend([...BOOKS.map((b) => ({ label: BOOK_NAMES[b], color: COLOR[b] })), { label: "best n-gram", color: "var(--muted)", dashed: true }]), box),
  );
}

// ------------------------------------------------------------------ results

function resultsSection(): HTMLElement {
  const ev = repo.evals;
  const st = ev.style;
  const box = h("div", { class: "chart-box" });
  responsive(box, (w) => intervalChart(BOOKS.map((b) => ({
    label: BOOK_NAMES[b], color: COLOR[b],
    model: { value: ev.loss[b].model.nats, lo: ev.loss[b].model.ci95_nats![0], hi: ev.loss[b].model.ci95_nats![1] },
    baseline: { name: `best n-gram (${ev.loss[b].baseline.name})`, value: ev.loss[b].baseline.nats, lo: ev.loss[b].baseline.ci95_nats![0], hi: ev.loss[b].baseline.ci95_nats![1] },
  })), w, [0.6, 2.0], "Validation loss of the model and the best n-gram baseline per book, with 95% intervals"));
  const gitaStyle = st.rows[0].cells.gita;
  const head = ["", ...BOOKS.map((b) => BOOK_NAMES[b])];
  return section("built-results", "Results",
    p("Every number is next to a baseline: n-gram models fitted on the same training text. All of them are read from ", src("evals/results/model.json"), ` (commit ${repo.evalCommit}).`),
    figure([`Held-out verses: ${BOOKS.map((b) => `${ev.loss[b].valVerses} ${BOOK_NAMES[b]}`).join(", ")}. Lines are 95% bootstrap intervals; none overlap. `,
      "I picked the checkpoint on these same verses and there is no separate test split, so the model’s numbers are slightly optimistic."], box),
    sub("Memorisation"),
    p(`The share of 50-character windows in ${ev.samplesPerBook} samples per book that appear verbatim in the training text. Real held-out verses are measured the same way, because scripture repeats its own phrases.`),
    table("Memorisation", head, [
      ["generated", ...BOOKS.map((b) => `${pct(ev.memorisation[b].generated.copied_from_any)} (longest ${ev.memorisation[b].generated.longest_run})`)],
      ["real held-out verses", ...BOOKS.map((b) => `${pct(ev.memorisation[b].heldout.copied_from_any)} (longest ${ev.memorisation[b].heldout.longest_run})`)],
    ]),
    sub("Style classifier"),
    p("A 5-gram classifier per book guesses where a text comes from. Each cell counts the texts assigned to the book they come from or imitate. ",
      "The other rows test the classifier, which is not perfectly fair: pure noise almost never goes to the Bible. ",
      `The Gita samples are recognised ${gitaStyle.counts.gita} times out of ${gitaStyle.n}; the Gita is the smallest book and its samples borrow from the other two.`),
    table("Style classifier results", ["Texts", ...BOOKS.map((b) => BOOK_NAMES[b])], [
      ...st.rows.map((r, i) => [i === 0 ? "generated by the model" : r.label, ...BOOKS.map((b) => `${r.cells[b].counts[b]}/${r.cells[b].n}`)]),
      ["uniform noise (share sent to each book)", ...BOOKS.map((b) => `${st.noise.counts[b]}/${st.noise.n}`)],
    ]),
    sub("Real words"),
    table("Real-word rate", head, [["generated words found in the training text", ...BOOKS.map((b) => pct(ev.words[b].realWordFrac))]]),
  );
}

// ------------------------------------------------------------------ what failed

function memorisedRun(): Node[] {
  const rows = repo.training.book;
  const last = rows[rows.length - 1];
  const minVal = rows.reduce((m, r) => (r.val.gita < m.val.gita ? r : m));
  const br = repo.bookRun;
  const ratio = Math.round(repo.stats.bible.characters / repo.stats.gita.characters);
  const box = h("div", { class: "chart-box" });
  responsive(box, (w) => lineChart({
    title: "The abandoned run: Gita training loss falls towards zero while Gita validation loss rises",
    series: [
      { label: "Quran validation", color: "var(--quran)", points: rows.filter((r) => r.step > 0).map((r) => [r.step, r.val.quran]), width: 1.5 },
      { label: "Bible validation", color: "var(--bible)", points: rows.filter((r) => r.step > 0).map((r) => [r.step, r.val.bible]), width: 1.5 },
      { label: "Gita training", color: "var(--gita)", points: rows.filter((r) => r.step > 0).map((r) => [r.step, r.train.gita]), dashed: true, width: 2 },
      { label: "Gita validation", color: "var(--gita)", points: rows.filter((r) => r.step > 0).map((r) => [r.step, r.val.gita]), width: 2 },
    ],
    x: { domain: [0, last.step], ticks: [0, 2000, 4000, 6000, 8000].filter((v) => v <= last.step), format: (v) => fmtInt(v), label: "training step" },
    y: { domain: [0, 2.6], ticks: [0, 0.5, 1.0, 1.5, 2.0, 2.5], format: (v) => v.toFixed(1), label: "loss (nats)" },
    markers: [
      { x: last.step, y: last.val.gita, label: `validation ${last.val.gita.toFixed(2)}`, color: "var(--gita)" },
      { x: last.step, y: last.train.gita, label: `training ${last.train.gita.toFixed(2)}`, color: "var(--gita)" },
    ],
  }, w));
  return [
    sub("Sampling the books equally made the model memorise the Gita"),
    p(`My first run picked each book equally often. The Bible has about ${ratio} times as many characters as the Gita, so each Gita verse came round about 40 times as often. `,
      `Gita training loss fell to ${last.train.gita.toFixed(2)} by step ${fmtInt(last.step)} while validation loss rose from ${minVal.val.gita.toFixed(2)} (step ${fmtInt(minVal.step)}) to ${last.val.gita.toFixed(2)}. `,
      `Its samples copied ${pct(br.gitaCopied)} of their 50-character windows verbatim (longest run ${br.gitaLongestRun} characters), and on unseen Gita verses it scored ${nats(br.gitaLoss)} nats, worse than a bigram model (${nats(br.bigramGitaLoss)}). `,
      `Sampling books in proportion to their size fixed it: the final model copies ${pct(repo.evals.memorisation.gita.generated.copied_from_any)} of its Gita windows.`),
    figure(["From ", src("runs/book/losses.jsonl"), ". The gap between the two Gita lines is memorisation."],
      legend([
        { label: "Gita validation", color: "var(--gita)" },
        { label: "Gita training", color: "var(--gita)", dashed: true },
        { label: "Quran validation", color: "var(--quran)" },
        { label: "Bible validation", color: "var(--bible)" },
      ]), box),
  ];
}

// Plain names for the labelled everyday sets of evals/search_cases.json.
const SPLIT_LABEL: Record<string, string> = {
  dev: "Set 1", dev2: "Set 2", dev3: "Set 3", blind_1: "Round 4 judge", final_blind: "Fifth judge",
};

function adviceFailures(): Node[] {
  const a = repo.advice;
  const fc = adviceHistory.fullCorpus;
  const [first, second, third, last] = a.reviews;
  const j8 = a.judge8;
  const totalVerses = BOOKS.reduce((n, b) => n + repo.stats[b].verses, 0);
  const share = (x: number, n: number) => pct(x / n, 0);
  const cases = a.topicHit.reduce((n, t) => n + t.cases, 0);
  const mean = (k: "before" | "after") => a.topicHit.reduce((n, t) => n + t[k] * t.cases, 0) / cases;
  const looks = a.firstLook;
  const reviewRows = [
    { r: first, label: "Round 4 judge", sub: `${first.crisis} messages, verses from free text` },
    { r: second, label: "Fifth judge", sub: `${second.crisis} messages, commit ${second.commit}` },
    { r: third, label: "Sixth judge", sub: `${third.crisis} messages, commit ${third.commit}` },
    { r: last, label: "Seventh judge", sub: `${last.crisis} messages, type then pick a topic`, apart: true },
  ];
  const box = h("div", { class: "chart-box" });
  responsive(box, (w) => missChart(reviewRows.map(({ r, label, sub, apart }) => ({ label, sub, missed: r.missed, harmful: r.harmful, apart })), w,
    "Crisis messages each blind review's rules missed, and how many of those could reach a harmful verse"));
  const oblique = (r: typeof second) => share(r.obliqueCaught!, r.oblique!);
  return [
    sub("The first verse-search benchmark was contaminated"),
    p(`Version one searched all ${fmtInt(totalVerses)} verses for words like the ones typed. It reported hit@10 ${fc.contaminatedHit10.toFixed(2)}, but I had written its list of modern-to-scripture words with the test queries in view. `,
      `On ${fc.blindSituations} situations a reviewer wrote blind it scored ${fc.blindHit10.toFixed(2)} (Quran ${fc.blindQuranHit10.toFixed(2)}). `,
      `An independent reviewer judged the verses it actually showed: ${pct(fc.shownFitting, 0)} fitted and ${pct(fc.shownPlainlyWrong, 0)} were plainly wrong, some of them harmful. `,
      `Now only a reviewed pool can be shown: ${a.pool.gita} Gita, ${a.pool.quran} Quran and ${a.pool.bible} Bible verses, each with themes and a note on why it may help (`, src("search/pool/"), ")."),
    sub("Choosing verses from free text failed blind safety review"),
    p("Independent reviewers wrote crisis messages in their own words, without seeing the code, and ran them once against the frozen rules. ",
      `Missed messages that were shown a possibly harmful verse: ${first.harmful} of ${first.missed} (round 4), ${second.harmful} of ${second.missed} (fifth judge), ${third.harmful} of ${third.missed} (sixth). `,
      "For the sixth, “stocked up on sleeping tablets, just need the courage” got “Be strong and of a good courage; be not afraid”."),
    figure(["Grey: crisis messages missed on the one run. Dark: those that could reach a harmful verse (for the seventh judge, one tap away). ",
      "The last three are the ", h("code", {}, "first_run"), " records in ", src("evals/search_probes.json"), "; round 4 is in the README."], box),
    p("I patched each review’s misses, but new wording kept getting through. First-sight catch rates: ",
      `three round 5 sets of ${looks[0].crisis} messages, ${looks.map((l) => l.caught).join(", ")} caught; fifth judge ${share(second.caught, second.crisis)}, sixth ${share(third.caught, third.crisis)}; `,
      `${oblique(second)} and ${oblique(third)} of the messages with no obvious keyword. Keyword rules cannot make free-text verse selection safe.`),
    p("So typing now maps only to reviewed topics; no verse is picked from what was typed. ",
      "The crisis rules run first; when they fire the page shows a short note and nothing else (no verses, topics or phone numbers). ",
      `The seventh judge tested this the same way: ${last.caught} of ${last.crisis} got the note. Of the ${last.missed} misses, ${last.harmful} could reach a harmful verse in one tap, all the same way: `,
      "someone being abused was offered a topic that asks the reader to change (a wife with bruises got only Anger: “he that is slow to anger appeaseth strife”)."),
    p(`The fix was a rule: ${a.comfortTopics.length} comfort topics (${a.comfortTopics.join(", ").toLowerCase()}), whose verses never tell the reader to endure, wait, submit, keep silent, forgive or try harder. `,
      "If a message names another person, something done to the writer, mental health or anger at God, only comfort topics are offered."),
    p(`It costs accuracy: topic hit@4 fell from ${mean("before").toFixed(2)} to ${mean("after").toFixed(2)} over ${cases} labelled situations, and “${NO_TOPIC_EXAMPLE}” now gets no topic and sees the full list.`),
    table("Topic hit@4 before and after the comfort-only rule", ["Topic hit@4", ...a.topicHit.map((t) => `${SPLIT_LABEL[t.split] ?? t.split} (${t.cases})`)], [
      ["before the rule", ...a.topicHit.map((t) => t.before.toFixed(2))],
      ["with it (live)", ...a.topicHit.map((t) => t.after.toFixed(2))],
    ]),
    p(`When this page started opening the first suggested topic without a tap, an eighth judge reviewed that design blind: ${j8.caught} of ${j8.crisis} crisis messages got the note; `,
      `of the ${j8.crisis - j8.caught} misses, ${j8.shownNothing} got no topic and ${j8.shownComfort} were shown comfort verses, ${j8.harmful === 0 ? "none" : j8.harmful} harmful; ${j8.falseAlarms} of ${j8.controls} everyday messages got the note by mistake. `,
      `Its follow-up probes found guilt and family topics opening by themselves, so now only ${a.autoShow.length} topics whose every verse I read may open without a tap (${a.autoShow.join(", ").toLowerCase()}); any other is listed for you to pick.`),
    p(h("small", { class: "muted" }, "All of these are development numbers: every set was used while writing the lists. ",
      `The rules catch ${fmtInt(a.storedCrisis - a.storedMissed)} of the ${fmtInt(a.storedCrisis)} stored crisis messages`,
      a.storedMissed ? ` (the ${a.storedMissed} misses are not in English and get no topic)` : "",
      ", but those are the messages they were fixed against. Sources: ",
      src("evals/results/search.json"), ", ", src("evals/search_probes.json"), ", ", src("README.md", "README"), ".")),
  ];
}

function failedSection(): HTMLElement {
  return section("built-failed", "What failed", ...memorisedRun(), ...adviceFailures());
}

// ------------------------------------------------------------------ limits

function limitsSection(): HTMLElement {
  const ev = repo.evals;
  const invented = (b: BookId) => Math.round(1 / (1 - ev.words[b].realWordFrac));
  const chars = (BOOKS.reduce((a, b) => a + repo.stats[b].characters, 0) / 1e6).toFixed(1);
  return section("built-limits", "Limits",
    h("ul", {},
      h("li", {}, "It imitates style. It predicts the next character from the previous 256 and does not know what the words mean; its sentences often make no sense."),
      h("li", {}, `It is weakest on the Gita, the smallest book: Gita samples borrow Quran and Bible phrases (“Lord of the Worlds”), and only ${ev.style.rows[0].cells.gita.counts.gita} of ${ev.style.rows[0].cells.gita.n} are recognised as Gita.`),
      h("li", {}, `It invents words: about one Gita-style word in ${invented("gita")} and one Bible-style word in ${invented("bible")} is not in the training text.`),
      h("li", {}, "It sometimes writes another book’s tag mid-sample and switches style. The demo marks where."),
      h("li", {}, `It is small: ${fmtInt(repo.params.total)} parameters and about ${chars} million characters of text. Large language models are many thousands of times bigger.`),
      h("li", {}, "Its output is not scripture, a quotation, a translation or a teaching, and should never be quoted as one."),
    ),
    p(h("small", { class: "muted" }, `The browser uses int8 weights: mean validation loss ${manifest.val_loss_nats.float32.overall.toFixed(4)} → ${manifest.val_loss_nats.int8.overall.toFixed(4)} nats `,
      `(+${(manifest.val_loss_nats.int8.overall - manifest.val_loss_nats.float32.overall).toFixed(4)}). `,
      `Tests: ${fmtInt(testCounts.python.tests)} Python, ${fmtInt(testCounts.vitest.tests)} unit and ${fmtInt(testCounts.playwright.tests)} browser tests (`, src("web/scripts/count_tests.ts", "count_tests.ts"), ").")),
  );
}

export function renderStory(root: HTMLElement): void {
  root.replaceChildren(modelSection(), dataSection(), trainingSection(), resultsSection(), failedSection(), limitsSection());
}
