// "How it works": short chapters drawn from real artefacts in the repository. Every number here comes from
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
import testCounts from "./test-counts.json";

const repo = repoJson as unknown as RepoData;
const BOOKS: BookId[] = ["gita", "quran", "bible"];
const COLOR: Record<BookId, string> = { gita: "var(--gita)", quran: "var(--quran)", bible: "var(--bible)" };
const REPO_URL = "https://github.com/bobbyt04/scripture-10m";
const blob = (path: string) => `${REPO_URL}/blob/main/${path}`;

const pct = (x: number, digits = 1) => `${(100 * x).toFixed(digits)}%`;
const nats = (x: number) => x.toFixed(3);

function chapter(id: string, num: string, title: string, ...body: (Node | null)[]): HTMLElement {
  return h("li", { class: "chapter", id },
    h("div", { class: "chapter__meta reveal" }, h("span", { class: "chapter__num" }, num), h("h3", { class: "chapter__title" }, title)),
    h("div", { class: "chapter__body" }, body.map((b) => (b instanceof HTMLElement && !b.classList.contains("reveal") ? wrapReveal(b) : b))));
}

const wrapReveal = (el: HTMLElement) => { el.classList.add("reveal"); return el; };
const p = (...c: (Node | string)[]) => h("p", {}, ...c);
const src = (path: string, label = path) => h("a", { href: blob(path), class: "src" }, label);
const figure = (caption: (Node | string)[], ...content: Node[]) =>
  h("figure", { class: "artefact" }, ...content, h("figcaption", {}, ...caption));
const sub = (text: string) => h("h4", { class: "sub" }, text);
const callout = (num: string, text: string, cls = "") =>
  h("div", { class: `callout ${cls}` }, h("p", { class: "callout__num" }, num), h("p", {}, text));

// ------------------------------------------------------------------ 01 model

function codeBlock(): HTMLElement {
  const c = repo.code;
  const notesByLine = new Map(c.notes.map((n, i) => [n.line, i + 1]));
  const seg = ([cls, text]: Segment) => (cls ? h("span", { class: `tok-${cls}` }, text) : document.createTextNode(text));
  const pre = h("pre", { class: "code", tabindex: "0", "aria-label": `Excerpt of ${c.file}` },
    h("code", {}, c.lines.map((segs, i) => {
      const line = c.firstLine + i;
      const n = notesByLine.get(line);
      return h("span", { class: `code__line${n ? " code__line--note" : ""}` },
        h("span", { class: "code__ln", "aria-hidden": "true" }, String(line)),
        n ? h("span", { class: "code__mark", "aria-label": `note ${n}` }, String(n)) : null,
        ...segs.map(seg), "\n");
    })));
  return h("div", { class: "code-wrap" },
    h("div", { class: "code-head" }, h("span", {}, `${c.file}, lines ${c.firstLine}–${c.firstLine + c.lines.length - 1}`), src(c.file, "GitHub")),
    pre,
    h("ol", { class: "code-notes" }, c.notes.map((n) => h("li", {}, n.note))));
}

function modelChapter(): HTMLElement {
  const cfg = repo.params.config;
  const total = repo.params.total;
  const show = (t: string) => (t === "\n" ? "↵" : t === " " ? "␣" : t);
  return chapter("built-model", "01", "Model",
    p(`Decoder-only transformer: ${cfg.n_layer} layers, ${cfg.n_head} heads, ${cfg.n_embd} dims, ${cfg.block_size} characters of context, ${fmtInt(total)} parameters. `,
      "Attention written out by hand, after Andrej Karpathy’s from-scratch lecture. The browser runs the same forward pass, rewritten in TypeScript."),
    h("p", { id: "why", class: "why" }, h("strong", {}, "Why it doesn’t answer you. "),
      `It is a ${(total / 1e6).toFixed(1)}M-parameter character model: it predicts the next character and can only imitate the style of the three books. `,
      "So the box above returns real verses, found by topic, not model output."),
    codeBlock(),
    figure([`${fmtInt(total)} parameters. The MLPs hold two thirds.`],
      h("div", { class: "params" },
        h("div", { class: "params__bar", "aria-hidden": "true" },
          repo.params.parts.map((part) => h("span", { class: `params__seg params__seg--${part.id}`, style: `--w:${(100 * part.count) / total}%` }))),
        h("table", { class: "params__table" },
          h("tbody", {}, repo.params.parts.map((part) =>
            h("tr", {},
              h("th", { scope: "row" }, h("span", { class: `params__key params__seg--${part.id}` }), part.label, h("small", {}, part.detail)),
              h("td", {}, fmtInt(part.count)),
              h("td", { class: "muted" }, pct(part.count / total)))),
            h("tr", { class: "params__total" }, h("th", { scope: "row" }, "Total"), h("td", {}, fmtInt(total)), h("td", {}, "")))))),
    figure([`Vocabulary: ${repo.vocab.length} tokens. 63 characters, newline (↵), and one tag per book.`],
      h("ol", { class: "vocab", start: 0, "aria-label": "The vocabulary" },
        repo.vocab.map((t, i) => h("li", { class: t.length > 1 ? "vocab__tag" : "" }, h("span", {}, show(t)), h("small", {}, String(i)))))),
  );
}

// ------------------------------------------------------------------ 02 data

function dataChapter(): HTMLElement {
  const st = repo.stats;
  const translations: Record<BookId, string> = {
    gita: "Besant, 1922", quran: "Pickthall, 1930", bible: "KJV, eBible.org",
  };
  const totalChars = BOOKS.reduce((a, b) => a + st[b].characters, 0);
  return chapter("built-data", "02", "Data",
    p("Three public-domain English translations, one verse per line, cut to 63 characters. The pipeline stops on anything unexpected instead of guessing."),
    h("div", { class: "books-stats" },
      BOOKS.map((b) =>
        h("div", { class: `stat-card book--${b}` },
          h("p", { class: "stat-card__book" }, BOOK_NAMES[b], h("span", { class: "stat-card__tr" }, translations[b])),
          h("p", { class: "stat-card__num" }, fmtInt(st[b].verses), h("span", {}, " verses")),
          h("p", { class: "stat-card__sub" }, `${fmtInt(st[b].chapters)} chapters · ${fmtInt(st[b].characters)} chars`),
          h("div", { class: "stat-card__bar", style: `--w:${(100 * st[b].characters) / totalChars}%`, "aria-hidden": "true" })))),
    p(h("small", { class: "muted" }, `The first Bible text, Project Gutenberg #10, had wording errors (${repo.kjvErrors.map((e) => e.text).join("; ")}). `,
      "Replaced with eBible.org’s, checked verse by verse against three independent copies; a word split in two (“can not”) in 13 verses is corrected. ",
      src("data/SOURCES.md"), ".")),
  );
}

// ------------------------------------------------------------------ 03 training

function trainingChapter(): HTMLElement {
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
  return chapter("built-training", "03", "Training",
    p(`${fmtInt(a.max_iters)} steps × ${a.batch_size} windows of ${a.block_size} characters, each window from one book, picked in proportion to its size. `,
      `AdamW, lr warmed to ${a.lr}, decayed to ${a.min_lr}, dropout ${a.dropout}. Apple M2, ${hours} h ${mins} min (shared with other jobs). Checkpoint: step ${fmtInt(t.bestStep)}, lowest validation loss.`),
    figure([
      "Solid: validation loss from ", src("runs/verse/losses.jsonl"), " (random windows). Dashed: best n-gram per book, same training text. ",
      "Scored on every held-out character: ", BOOKS.map((b) => `${BOOK_NAMES[b]} ${nats(ev.loss[b].model.nats)}`).join(", "), ".",
    ], legend([...BOOKS.map((b) => ({ label: BOOK_NAMES[b], color: COLOR[b] })), { label: "best n-gram", color: "var(--muted)", dashed: true }]), box),
  );
}

// ------------------------------------------------------------------ 04 results

function styleTable(): HTMLElement {
  const st = repo.evals.style;
  return h("div", { class: "table-wrap", tabindex: "0", role: "region", "aria-label": "Style classifier results" },
    h("table", { class: "data" },
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Texts"), BOOKS.map((b) => h("th", { scope: "col" }, BOOK_NAMES[b])))),
      h("tbody", {},
        st.rows.map((r, i) => h("tr", { class: i === 0 ? "row--main" : "row--control" },
          h("th", { scope: "row" }, i === 0 ? "generated by the model" : r.label),
          BOOKS.map((b) => h("td", {}, `${r.cells[b].counts[b]}/${r.cells[b].n}`)))),
        h("tr", { class: "row--control" }, h("th", { scope: "row" }, "uniform noise (share sent to each book)"),
          BOOKS.map((b) => h("td", {}, `${st.noise.counts[b]}/${st.noise.n}`))))));
}

function resultsChapter(): HTMLElement {
  const ev = repo.evals;
  const box = h("div", { class: "chart-box" });
  responsive(box, (w) => intervalChart(BOOKS.map((b) => ({
    label: BOOK_NAMES[b], color: COLOR[b],
    model: { value: ev.loss[b].model.nats, lo: ev.loss[b].model.ci95_nats![0], hi: ev.loss[b].model.ci95_nats![1] },
    baseline: { name: `best n-gram (${ev.loss[b].baseline.name})`, value: ev.loss[b].baseline.nats, lo: ev.loss[b].baseline.ci95_nats![0], hi: ev.loss[b].baseline.ci95_nats![1] },
  })), w, [0.6, 2.0], "Validation loss of the model and the best n-gram baseline per book, with 95% intervals"));
  const gitaStyle = ev.style.rows[0].cells.gita;
  return chapter("built-results", "04", "Results",
    p("Every number sits next to a baseline: n-gram models fitted on the same training text. From ", src("evals/results/model.json"), `, commit ${repo.evalCommit}.`),
    figure([`Held-out verses: ${BOOKS.map((b) => `${ev.loss[b].valVerses} ${BOOK_NAMES[b]}`).join(", ")}. 95% bootstrap intervals; none overlap. `,
      "The checkpoint was picked on these verses (no separate test split), so the model’s numbers are slightly optimistic."], box),
    sub("Memorisation"),
    p(`Share of 50-character windows in ${ev.samplesPerBook} samples per book found verbatim in the training text. Real unseen verses for comparison: scripture repeats itself.`),
    h("div", { class: "table-wrap", tabindex: "0", role: "region", "aria-label": "Memorisation results" },
      h("table", { class: "data" },
        h("thead", {}, h("tr", {}, h("th", { scope: "col" }, ""), BOOKS.map((b) => h("th", { scope: "col" }, BOOK_NAMES[b])))),
        h("tbody", {},
          h("tr", { class: "row--main" }, h("th", { scope: "row" }, "generated"), BOOKS.map((b) => h("td", {}, pct(ev.memorisation[b].generated.copied_from_any), h("small", {}, `longest ${ev.memorisation[b].generated.longest_run}`)))),
          h("tr", {}, h("th", { scope: "row" }, "real held-out verses"), BOOKS.map((b) => h("td", {}, pct(ev.memorisation[b].heldout.copied_from_any), h("small", {}, `longest ${ev.memorisation[b].heldout.longest_run}`))))))),
    sub("Style classifier"),
    p("A 5-gram classifier per book guesses the source. Cells: texts assigned to the book they come from or imitate. ",
      "Control rows test the classifier; it is biased (noise almost never goes to the Bible). ",
      `Gita samples: ${gitaStyle.counts.gita}/${gitaStyle.n}. The Gita is the smallest book and its samples borrow from the other two.`),
    styleTable(),
    sub("Real-word rate"),
    h("div", { class: "callouts callouts--three" },
      BOOKS.map((b) => callout(pct(ev.words[b].realWordFrac), `of generated ${BOOK_NAMES[b]}-style words occur in the training text.`, `book--${b}`))),
  );
}

// ------------------------------------------------------------------ 05 what failed

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
      { label: "Gita training", color: "var(--gita)", points: rows.filter((r) => r.step > 0).map((r) => [r.step, r.train.gita]), dashed: true, width: 2.5 },
      { label: "Gita validation", color: "var(--gita)", points: rows.filter((r) => r.step > 0).map((r) => [r.step, r.val.gita]), width: 2.5 },
    ],
    x: { domain: [0, last.step], ticks: [0, 2000, 4000, 6000, 8000].filter((v) => v <= last.step), format: (v) => fmtInt(v), label: "training step" },
    y: { domain: [0, 2.6], ticks: [0, 0.5, 1.0, 1.5, 2.0, 2.5], format: (v) => v.toFixed(1), label: "loss (nats)" },
    markers: [
      { x: last.step, y: last.val.gita, label: `validation ${last.val.gita.toFixed(2)}`, color: "var(--gita)" },
      { x: last.step, y: last.train.gita, label: `training ${last.train.gita.toFixed(2)}`, color: "var(--gita)" },
    ],
  }, w));
  return [
    sub("1. Equal books: the model memorised the Gita"),
    p(`The first run sampled each book equally often. The Bible has ~${ratio}× the Gita’s characters, so each Gita verse came round ~40× as often. `,
      `Gita training loss fell to ${last.train.gita.toFixed(2)} by step ${fmtInt(last.step)}; validation rose from ${minVal.val.gita.toFixed(2)} (step ${fmtInt(minVal.step)}) to ${last.val.gita.toFixed(2)}. `,
      `Scored on every held-out character, its final Gita loss is ${nats(br.gitaLoss)}.`),
    figure(["From ", src("runs/book/losses.jsonl"), ". The widening gap between the two Gita lines is memorisation."],
      legend([
        { label: "Gita validation", color: "var(--gita)" },
        { label: "Gita training", color: "var(--gita)", dashed: true },
        { label: "Quran validation", color: "var(--quran)" },
        { label: "Bible validation", color: "var(--bible)" },
      ]), box),
    h("div", { class: "callouts" },
      callout(pct(br.gitaCopied), `of 50-char windows in its Gita samples copied verbatim. Longest: ${br.gitaLongestRun} chars.`),
      callout(nats(br.gitaLoss), `nats on unseen Gita verses, worse than a bigram model (${nats(br.bigramGitaLoss)}).`)),
    p(h("strong", {}, "Fix: "), "sample books in proportion to size, so every verse is seen about equally often. ",
      `The final model copies ${pct(repo.evals.memorisation.gita.generated.copied_from_any)} of its Gita windows.`),
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
    sub("2. The first verse-search benchmark was contaminated"),
    p(`Version one searched all ${fmtInt(totalVerses)} verses for words like the ones typed. Reported hit@10: ${fc.contaminatedHit10.toFixed(2)}, but I had written its modern-word → scripture-word list with the test queries in view. `,
      `On ${fc.blindSituations} situations a reviewer wrote blind: ${fc.blindHit10.toFixed(2)} (Quran ${fc.blindQuranHit10.toFixed(2)}). `,
      `Of the verses it actually showed, an independent reviewer judged ${pct(fc.shownFitting, 0)} fitting and ${pct(fc.shownPlainlyWrong, 0)} plainly wrong, some harmful.`),
    p(`Now only a reviewed pool can be shown: ${a.pool.gita} Gita, ${a.pool.quran} Quran, ${a.pool.bible} Bible verses, each with themes and a note on why it may help (`, src("search/pool/"), ")."),
    sub("3. Verses from free text failed blind safety review"),
    p("Independent reviewers wrote crisis messages in their own words, without seeing the code, and ran them once against the frozen rules. ",
      `Missed messages shown a possibly harmful verse: ${first.harmful} of ${first.missed} (round 4), ${second.harmful} of ${second.missed} (fifth), ${third.harmful} of ${third.missed} (sixth). `,
      "For the sixth, “stocked up on sleeping tablets, just need the courage” got “Be strong and of a good courage; be not afraid”."),
    figure(["Grey: crisis messages missed on the one run. Colour: of those, how many could reach a harmful verse (for the seventh judge: one tap away). ",
      "The last three are the ", h("code", {}, "first_run"), " records in ", src("evals/search_probes.json"), "; round 4 is in the README."], box),
    p("Each review’s misses were patched, but new wording kept getting through. First-sight catch rates: ",
      `three round 5 sets of ${looks[0].crisis} messages, ${looks.map((l) => l.caught).join(", ")} caught; fifth judge ${share(second.caught, second.crisis)}, sixth ${share(third.caught, third.crisis)}; `,
      `${oblique(second)} and ${oblique(third)} of messages with no obvious keyword. `,
      h("strong", {}, "Keyword rules cannot make free-text verse selection safe.")),
    p(h("strong", {}, "So typing now maps only to reviewed topics; no verse is picked from what was typed. "),
      "Crisis rules run first; when they fire, the page shows a short note and nothing else (no verses, topics or phone numbers). ",
      `Seventh judge, same method: ${last.caught} of ${last.crisis} got the note. Of ${last.missed} misses, ${last.harmful} could reach a harmful verse in one tap, all the same way: `,
      "someone being abused was offered a topic asking the reader to change (a wife with bruises got only Anger: “he that is slow to anger appeaseth strife”). ",
      "This page now opens the first suggested topic without a tap, so such a miss would show that topic’s verses directly."),
    p(`Fix: ${a.comfortTopics.length} comfort topics (${a.comfortTopics.join(", ").toLowerCase()}), whose reviewed verses never tell the reader to endure, wait, submit, keep silent, forgive or try harder. `,
      "If a message names another person, something done to the writer, mental health or anger at God, only comfort topics are offered."),
    p(`Cost: topic hit@4 fell from ${mean("before").toFixed(2)} to ${mean("after").toFixed(2)} over ${cases} labelled situations. “${NO_TOPIC_EXAMPLE}” now gets no topic and sees the full list. Accepted.`),
    h("div", { class: "table-wrap", tabindex: "0", role: "region", "aria-label": "Topic hit@4 before and after the comfort-only rule" },
      h("table", { class: "data hit-table" },
        h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Topic hit@4"), a.topicHit.map((t) => h("th", { scope: "col" }, SPLIT_LABEL[t.split] ?? t.split, h("small", {}, `${t.cases} cases`))))),
        h("tbody", {},
          h("tr", {}, h("th", { scope: "row" }, "before"), a.topicHit.map((t) => h("td", {}, t.before.toFixed(2)))),
          h("tr", { class: "row--main" }, h("th", { scope: "row" }, "comfort-only (live)"), a.topicHit.map((t) => h("td", {}, t.after.toFixed(2))))))),
    p(h("small", { class: "muted" }, "Development numbers: every set was used while writing the lists. ",
      `The rules catch all ${fmtInt(a.storedCrisis)} stored crisis messages, but those are the ones they were fixed against. `,
      "Safety rests on the design: a missed message gets reviewed topics and their reviewed verses, never a verse chosen from its words. Sources: ",
      src("evals/results/search.json"), ", ", src("evals/search_probes.json"), ", ", src("README.md", "README"), ".")),
  ];
}

function failedChapter(): HTMLElement {
  return chapter("built-failed", "05", "What failed", ...memorisedRun(), ...adviceFailures());
}

// ------------------------------------------------------------------ 06 limits

function limitsChapter(): HTMLElement {
  const ev = repo.evals;
  const invented = (b: BookId) => Math.round(1 / (1 - ev.words[b].realWordFrac));
  const chars = (BOOKS.reduce((a, b) => a + repo.stats[b].characters, 0) / 1e6).toFixed(1);
  return chapter("built-limits", "06", "Limits",
    h("ul", { class: "limits" },
      h("li", {}, h("strong", {}, "Style only. "), "Next character from the previous 256. No meaning, no answers; sentences often make no sense."),
      h("li", {}, h("strong", {}, "Weakest on the Gita. "), `Smallest book; its samples leak Quran and Bible phrases (“Lord of the Worlds”). ${ev.style.rows[0].cells.gita.counts.gita} of ${ev.style.rows[0].cells.gita.n} recognised as Gita.`),
      h("li", {}, h("strong", {}, "Invents words. "), `~1 in ${invented("gita")} Gita-style words and ~1 in ${invented("bible")} Bible-style words are not in the training text.`),
      h("li", {}, h("strong", {}, "Wanders. "), "Sometimes writes another book’s tag mid-sample and switches style. The demo marks it."),
      h("li", {}, h("strong", {}, "Tiny. "), `${fmtInt(repo.params.total)} parameters, ~${chars}M characters of text. Large language models are many thousands of times bigger.`),
      h("li", {}, h("strong", {}, "Not scripture. "), "Its output is not a quotation, translation or teaching. Never quote it as one."),
    ),
    p(h("small", { class: "muted" }, `The browser uses int8 weights: mean validation loss ${manifest.val_loss_nats.float32.overall.toFixed(4)} → ${manifest.val_loss_nats.int8.overall.toFixed(4)} nats `,
      `(+${(manifest.val_loss_nats.int8.overall - manifest.val_loss_nats.float32.overall).toFixed(4)}). `,
      `Tests: ${fmtInt(testCounts.python.tests)} Python, ${fmtInt(testCounts.vitest.tests)} unit, ${fmtInt(testCounts.playwright.tests)} browser (`, src("web/scripts/count_tests.ts", "counted"), ").")),
  );
}

export function renderStory(list: HTMLElement): void {
  list.replaceChildren(modelChapter(), dataChapter(), trainingChapter(), resultsChapter(), failedChapter(), limitsChapter());
}
