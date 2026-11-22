// "How it was built": chapters drawn from real artefacts in the repository. Every number here comes from
// src/generated/repo.json (written at build time by scripts/extract.ts from the eval JSON, training logs,
// data stats and source files) or from the two small JSON files beside this one.

import type { RepoData, Segment } from "../../scripts/repo-data.ts";
import { BOOK_NAMES, type BookId } from "../advice/api.ts";
import { fmtInt, h } from "../dom.ts";
import manifest from "../model/manifest.json";
import repoJson from "../generated/repo.json";
import adviceHistory from "./advice-history.json";
import { intervalChart, legend, lineChart, missChart, responsive } from "./charts.ts";
import process from "./process.json";
import testCounts from "./test-counts.json";

const repo = repoJson as unknown as RepoData;
const BOOKS: BookId[] = ["gita", "quran", "bible"];
const COLOR: Record<BookId, string> = { gita: "var(--gita)", quran: "var(--quran)", bible: "var(--bible)" };
const REPO_URL = process.repo;
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

// ------------------------------------------------------------------ 01 data

function dataChapter(): HTMLElement {
  const st = repo.stats;
  const translations: Record<BookId, string> = {
    gita: "Annie Besant, 4th edition, 1922",
    quran: "Marmaduke Pickthall, 1930",
    bible: "King James Version (eBible.org text)",
  };
  const totalChars = BOOKS.reduce((a, b) => a + st[b].characters, 0);
  const totalVerses = BOOKS.reduce((a, b) => a + st[b].verses, 0);
  return chapter("built-data", "01", "The data",
    p(`Three English translations that are free to copy, ${fmtInt(totalVerses)} verses in all. A script downloads them, `,
      "strips headers, footnotes and the Sanskrit, cuts the text down to 63 characters, and writes one verse per line. ",
      "It stops with an error, rather than guessing, if anything in a source looks wrong."),
    h("div", { class: "books-stats" },
      BOOKS.map((b) =>
        h("div", { class: `stat-card book--${b}` },
          h("p", { class: "stat-card__book" }, BOOK_NAMES[b]),
          h("p", { class: "stat-card__tr" }, translations[b]),
          h("p", { class: "stat-card__num" }, fmtInt(st[b].verses), h("span", {}, " verses")),
          h("p", { class: "stat-card__sub" }, `${fmtInt(st[b].chapters)} chapters · ${fmtInt(st[b].characters)} characters`),
          h("div", { class: "stat-card__bar", style: `--w:${(100 * st[b].characters) / totalChars}%`, "aria-hidden": "true" }))),
    ),
    h("div", { class: "aside" },
      p(h("strong", {}, "The Bible source I replaced. "),
        "My first Bible text was Project Gutenberg #10. Comparing it word by word with three independent copies turned up wording errors, for example:"),
      h("ul", { class: "errata" }, repo.kjvErrors.map((e) => h("li", {}, e.text))),
      p("So I switched to eBible.org’s King James text and checked all 31,102 verses against the same three copies. ",
        "Where it disagreed with all of them it had split a word in two (“can not”), in 13 verses, which the pipeline corrects. ",
        "Sources, licences and every correction: ", src("data/SOURCES.md"), ".")),
  );
}

// ------------------------------------------------------------------ 02 tokenizer

function tokenizerChapter(): HTMLElement {
  const show = (t: string) => (t === "\n" ? "↵" : t === " " ? "␣" : t);
  const ex = repo.example;
  return chapter("built-tokenizer", "02", "The tokenizer",
    p(`The model reads characters, not words. Each character is one token: the 63 characters of the text, the newline that ends a verse, `,
      `and three tags, `, h("code", {}, "<gita>"), ", ", h("code", {}, "<quran>"), " and ", h("code", {}, "<bible>"),
      `, that say which book a verse comes from. That is the whole vocabulary: ${repo.vocab.length} tokens.`),
    figure([`The ${repo.vocab.length} tokens, numbered as the model numbers them. ↵ is the newline and ␣ the space.`],
      h("ol", { class: "vocab", start: 0, "aria-label": "The vocabulary" },
        repo.vocab.map((t, i) => h("li", { class: t.length > 1 ? "vocab__tag" : "" }, h("span", {}, show(t)), h("small", {}, String(i)))))),
    figure([`${BOOK_NAMES[ex.book]} ${ex.ref} as the model sees it: ${ex.tokens.length} token ids, starting with the book tag and ending with a newline.`],
      h("ol", { class: "tokens", "aria-label": `${BOOK_NAMES[ex.book]} ${ex.ref} as token ids` },
        ex.tokens.map((t, i) => h("li", { style: `--i:${i}` }, h("span", { class: "tokens__ch" }, show(t.token)), h("span", { class: "tokens__id" }, String(t.id)))))),
  );
}

// ------------------------------------------------------------------ 03 model

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
    h("div", { class: "code-head" }, h("span", {}, `${c.file}, lines ${c.firstLine}–${c.firstLine + c.lines.length - 1}`), src(c.file, "View on GitHub")),
    pre,
    h("ol", { class: "code-notes" }, c.notes.map((n) => h("li", {}, n.note))));
}

function modelChapter(): HTMLElement {
  const cfg = repo.params.config;
  const total = repo.params.total;
  return chapter("built-model", "03", "The model",
    p(`A decoder-only transformer with ${cfg.n_layer} layers, ${cfg.n_head} attention heads, ${cfg.n_embd} numbers per position and ${cfg.block_size} characters of context, `,
      "following Andrej Karpathy’s from-scratch lecture. I wrote the attention out step by step instead of using PyTorch’s built-in attention. ",
      "This is the real code, taken from the repository when this page was built:"),
    codeBlock(),
    figure([`Where the ${fmtInt(total)} parameters are. The feed-forward layers hold two thirds of them.`],
      h("div", { class: "params" },
        h("div", { class: "params__bar", "aria-hidden": "true" },
          repo.params.parts.map((part) => h("span", { class: `params__seg params__seg--${part.id}`, style: `--w:${(100 * part.count) / total}%` }))),
        h("table", { class: "params__table" },
          h("tbody", {}, repo.params.parts.map((part) =>
            h("tr", {},
              h("th", { scope: "row" }, h("span", { class: `params__key params__seg--${part.id}` }), part.label, h("small", {}, part.detail)),
              h("td", {}, fmtInt(part.count)),
              h("td", { class: "muted" }, pct(part.count / total)))),
            h("tr", { class: "params__total" }, h("th", { scope: "row" }, "Total"), h("td", {}, fmtInt(total)), h("td", {}, ""))))))
  );
}

// ------------------------------------------------------------------ 04 training

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
  return chapter("built-training", "04", "Training",
    p(`Each training example is a random window of ${a.block_size} characters from one book. ${fmtInt(a.max_iters)} steps of ${a.batch_size} windows, `,
      `AdamW with the learning rate warmed up to ${a.lr} and decayed to ${a.min_lr}, dropout ${a.dropout}. `,
      `It ran on an Apple M2 laptop in ${hours} h ${mins} min, slowed by other jobs on the same machine. The checkpoint used here is step ${fmtInt(t.bestStep)}, the lowest validation loss.`),
    figure([
      "Solid: validation loss per book from the training log (", src("runs/verse/losses.jsonl"), "). Dashed: the best n-gram model per book, fitted on the same training text. ",
      `The log estimates loss on random windows; the eval scores every character with at least half a window of context, and there the model scores `,
      BOOKS.map((b) => `${BOOK_NAMES[b]} ${nats(ev.loss[b].model.nats)}`).join(", "), ": below every baseline.",
    ], legend([...BOOKS.map((b) => ({ label: BOOK_NAMES[b], color: COLOR[b] })), { label: "best n-gram baseline", color: "var(--muted)", dashed: true }]), box),
  );
}

// ------------------------------------------------------------------ 05 mistake

function mistakeChapter(): HTMLElement {
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
  return chapter("built-mistake", "05", "The mistake",
    p(`My first run picked each book equally often. The Bible has about ${ratio} times as many characters as the Gita, so each Gita verse came round about 40 times as often as each Bible verse. `,
      `The model learned the Gita by heart. Its Gita training loss kept falling, to ${last.train.gita.toFixed(2)} by step ${fmtInt(last.step)}, `,
      `while its Gita validation loss climbed from ${minVal.val.gita.toFixed(2)} at step ${fmtInt(minVal.step)} to ${last.val.gita.toFixed(2)}.`),
    figure(["The abandoned run, from ", src("runs/book/losses.jsonl"), ". The widening gap between the two orange lines is memorisation."],
      legend([
        { label: "Gita validation", color: "var(--gita)" },
        { label: "Gita training", color: "var(--gita)", dashed: true },
        { label: "Quran validation", color: "var(--quran)" },
        { label: "Bible validation", color: "var(--bible)" },
      ]), box),
    h("div", { class: "callouts" },
      h("div", { class: "callout" }, h("p", { class: "callout__num" }, pct(br.gitaCopied)),
        h("p", {}, `of 50-character windows in its Gita samples were copied word for word from training text. One sample was a ${br.gitaLongestRun}-character stretch of it.`)),
      h("div", { class: "callout" }, h("p", { class: "callout__num" }, nats(br.gitaLoss)),
        h("p", {}, `nats on unseen Gita verses: worse than a two-character n-gram model (${nats(br.bigramGitaLoss)}).`))),
    p(h("strong", {}, "The fix. "), "Pick each training window’s book in proportion to the book’s size, so every verse is seen about equally often. ",
      `The final model copies ${pct(repo.evals.memorisation.gita.generated.copied_from_any)} of its Gita windows.`),
  );
}

// ------------------------------------------------------------------ 06 evals

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

function evalsChapter(): HTMLElement {
  const ev = repo.evals;
  const box = h("div", { class: "chart-box" });
  responsive(box, (w) => intervalChart(BOOKS.map((b) => ({
    label: BOOK_NAMES[b], color: COLOR[b],
    model: { value: ev.loss[b].model.nats, lo: ev.loss[b].model.ci95_nats![0], hi: ev.loss[b].model.ci95_nats![1] },
    baseline: { name: `best n-gram (${ev.loss[b].baseline.name})`, value: ev.loss[b].baseline.nats, lo: ev.loss[b].baseline.ci95_nats![0], hi: ev.loss[b].baseline.ci95_nats![1] },
  })), w, [0.6, 2.0], "Validation loss of the model and the best n-gram baseline per book, with 95% intervals"));
  const gitaStyle = ev.style.rows[0].cells.gita;
  return chapter("built-evals", "06", "Evals",
    p("A loss number means little on its own, so every number here sits next to a simple baseline: n-gram models that count which character follows which, ",
      "fitted on exactly the same training text. All of it is read from ", src("evals/results/model.json"), ` (evals run at commit ${repo.evalCommit}).`),
    figure([`Loss on held-out verses (${BOOKS.map((b) => `${ev.loss[b].valVerses} ${BOOK_NAMES[b]}`).join(", ")}). Lines are 95% bootstrap intervals; they do not overlap. `,
      "The checkpoint was chosen on these same verses and there is no separate test split, so the model’s numbers are slightly optimistic."], box),
    h("h4", { class: "sub" }, "Does it copy?"),
    p(`Share of 50-character windows in ${ev.samplesPerBook} generated samples per book that appear word for word in the training text. Real unseen verses are measured the same way, because scripture repeats its own phrases.`),
    h("div", { class: "table-wrap", tabindex: "0", role: "region", "aria-label": "Memorisation results" },
      h("table", { class: "data" },
        h("thead", {}, h("tr", {}, h("th", { scope: "col" }, ""), BOOKS.map((b) => h("th", { scope: "col" }, BOOK_NAMES[b])))),
        h("tbody", {},
          h("tr", { class: "row--main" }, h("th", { scope: "row" }, "generated"), BOOKS.map((b) => h("td", {}, pct(ev.memorisation[b].generated.copied_from_any), h("small", {}, `longest ${ev.memorisation[b].generated.longest_run}`)))),
          h("tr", {}, h("th", { scope: "row" }, "real held-out verses"), BOOKS.map((b) => h("td", {}, pct(ev.memorisation[b].heldout.copied_from_any), h("small", {}, `longest ${ev.memorisation[b].heldout.longest_run}`))))))),
    h("h4", { class: "sub" }, "Does it sound like the right book?"),
    p("A simple 5-gram classifier per book guesses where each text comes from. The cells count texts sent to the book they come from or imitate. ",
      "The control rows test the classifier itself, and it is not perfectly fair: pure noise almost never goes to the Bible, and unigram text imitating the Quran rarely goes to the Quran. ",
      `Text with real word structure is classified correctly almost every time. The model’s Gita samples are recognised ${gitaStyle.counts.gita} times out of ${gitaStyle.n}: `,
      "the Gita is the smallest book, and its samples borrow phrases from the other two."),
    styleTable(),
    h("h4", { class: "sub" }, "Are the words real?"),
    h("div", { class: "callouts callouts--three" },
      BOOKS.map((b) => h("div", { class: `callout book--${b}` }, h("p", { class: "callout__num" }, pct(ev.words[b].realWordFrac)),
        h("p", {}, `of generated ${BOOK_NAMES[b]}-style words occur in the training text.`)))),
  );
}

// ------------------------------------------------------------------ 07 testing

function testingChapter(): HTMLElement {
  return chapter("built-testing", "07", "Testing",
    p("Each part landed as its own pull request, with tests written alongside it. Where I could, I checked the tests themselves: ",
      "I broke the code on purpose in many small ways (mutation testing) and made sure some test failed every time."),
    h("ol", { class: "timeline" },
      process.prs.map((pr) => h("li", {},
        h("p", { class: "timeline__head" }, h("a", { href: `${REPO_URL}/pull/${pr.number}` }, `PR ${pr.number}`), ` ${pr.title}`),
        pr.tests !== undefined ? h("p", { class: "timeline__tests" }, `${fmtInt(pr.tests)} tests passing`) : null,
        h("p", {}, pr.check)))),
    h("div", { class: "callouts callouts--three" },
      h("div", { class: "callout" }, h("p", { class: "callout__num" }, fmtInt(testCounts.python.tests)), h("p", {}, "Python tests today: the model, data, training, evals and the verse search.")),
      h("div", { class: "callout" }, h("p", { class: "callout__num" }, fmtInt(testCounts.vitest.tests)), h("p", {}, "unit tests for this website: the forward pass, the tokenizer, the browser search against Python, the UI.")),
      h("div", { class: "callout" }, h("p", { class: "callout__num" }, fmtInt(testCounts.playwright.tests)), h("p", {}, "browser tests of the built site, crisis note and topic flow included."))),
    p(process.ci, " ", h("small", { class: "muted" }, "Counted by ", src("web/scripts/count_tests.ts"), ".")),
  );
}

// ------------------------------------------------------------------ 08 advice search

function adviceChapter(): HTMLElement {
  const a = repo.advice;
  const fc = adviceHistory.fullCorpus;
  const [first, second, third, last] = a.reviews;
  const totalVerses = BOOKS.reduce((n, b) => n + repo.stats[b].verses, 0);
  const share = (x: number, n: number) => pct(x / n, 0);
  const cases = a.topicHit.reduce((n, t) => n + t.cases, 0);
  const mean = (k: "before" | "after") => a.topicHit.reduce((n, t) => n + t[k] * t.cases, 0) / cases;
  const reviewRows = [
    { r: first, label: "First review", sub: `${first.crisis} messages, verses from free text` },
    { r: second, label: "Second review", sub: `${second.crisis} messages, commit ${second.commit}` },
    { r: third, label: "Third review", sub: `${third.crisis} messages, commit ${third.commit}` },
    { r: last, label: "Final review", sub: `${last.crisis} messages, type then pick a topic`, apart: true },
  ];
  const box = h("div", { class: "chart-box" });
  responsive(box, (w) => missChart(reviewRows.map(({ r, label, sub, apart }) => ({ label, sub, missed: r.missed, harmful: r.harmful, apart })), w,
    "Crisis messages each blind review's rules missed, and how many of those could reach a harmful verse"));
  const oblique = (r: typeof second) => share(r.obliqueCaught!, r.oblique!);
  return chapter("built-advice", "08", "The verse search",
    p(`The first version searched all ${fmtInt(totalVerses)} verses for words like the ones typed. Its first reported number, hit@10 ${fc.contaminatedHit10.toFixed(2)}, `,
      "was contaminated: I had written the list that maps modern words to scripture words with those very test queries in view. ",
      `On ${fc.blindSituations} situations a reviewer wrote blind it scored ${fc.blindHit10.toFixed(2)} (the Quran ${fc.blindQuranHit10.toFixed(2)}), and when a person judged the verses it actually showed, `,
      `only ${pct(fc.shownFitting, 0)} fitted; ${pct(fc.shownPlainlyWrong, 0)} were plainly wrong, some of them harmful.`),
    p("So I stopped searching the whole corpus. The site shows verses only from an allowlist: a pool that people chose and read, ",
      `each verse with its themes and a one-line note on why it may help. It now holds ${a.pool.gita} Gita, ${a.pool.quran} Quran and ${a.pool.bible} Bible verses, `,
      "and nothing outside it can be shown. Every decision about it is recorded in ", src("search/pool/"), "."),
    h("h4", { class: "sub" }, "Three blind safety reviews"),
    p("What worried me most was a message from someone in crisis. Three times an independent reviewer wrote crisis messages in their own words, ",
      "before seeing my code or lists, and ran them once against the frozen rules. Each time some got past the crisis rules, and some of those were shown a verse that could do harm: ",
      `${first.harmful} of ${first.missed} missed messages in the first review, ${second.harmful} of ${second.missed} in the second and ${third.harmful} of ${third.missed} in the third. `,
      "In the third, “stocked up on sleeping tablets, just need the courage” was shown “Be strong and of a good courage; be not afraid”."),
    figure(["Crisis messages each blind review’s rules missed on its one run (grey), and how many of those could reach a harmful verse (colour). ",
      "The last three are the ", h("code", {}, "first_run"), " records in ", src("evals/search_probes.json"), "; the first review’s counts are in the README. ",
      "In the final review, harmful means one tap away."], box),
    p("After each review I added rules for its misses and their paraphrases, and on those messages the rules then caught everything. ",
      `But fresh wording kept getting past them: on first sight they caught ${share(second.caught, second.crisis)}, then ${share(third.caught, third.crisis)} of the crisis messages, `,
      `and ${oblique(second)} and ${oblique(third)} of those written without an obvious keyword. `,
      h("strong", {}, "My conclusion was that keyword rules cannot make free-text verse selection safe.")),
    h("h4", { class: "sub" }, "Type, then pick a topic"),
    p("So I changed the product. Typing now only suggests topics, and a verse appears only after the visitor taps one; the site never shows a verse chosen from free text. ",
      "The crisis rules still run first. When they fire, the page shows a short note and nothing else: no verses, no topics, no phone numbers. ",
      "A message they miss can no longer be handed a verse it did not ask for."),
    p(`A fourth reviewer then tested this design the same way. ${last.caught} of ${last.crisis} crisis messages got the note. Of the ${last.missed} it missed, `,
      `${last.harmful} could reach a harmful verse with one tap, and all ${last.harmful} in the same way: someone being abused was offered a topic whose verses ask the reader to change. `,
      "A wife whose husband leaves bruises was offered only Anger (“he that is slow to anger appeaseth strife”). That reads as blame."),
    p(`The fix was a rule, not more phrases. ${a.comfortTopics.length} topics are comfort topics (${a.comfortTopics.join(", ").toLowerCase()}): `,
      "I read every one of their verses, and none tells the reader to endure, wait, submit, keep silent, forgive or try harder. ",
      "The others ask the reader to change their own behaviour or to bear something. If a message names another person, an event done to the writer, mental health or anger at God, only comfort topics are offered."),
    p(`The rule has a measured cost. Topic hit@4, whether a topic I labelled as fitting is among those suggested, fell from ${mean("before").toFixed(2)} to ${mean("after").toFixed(2)} `,
      `over ${cases} labelled everyday situations. “I lost my temper and yelled at my kids” now gets no topic, and the site shows the full list instead. I accepted that.`),
    h("div", { class: "table-wrap", tabindex: "0", role: "region", "aria-label": "Topic hit@4 before and after the comfort-only rule" },
      h("table", { class: "data hit-table" },
        h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Topic hit@4"), a.topicHit.map((t) => h("th", { scope: "col" }, t.split.replace("_", " "), h("small", {}, `${t.cases} cases`))))),
        h("tbody", {},
          h("tr", {}, h("th", { scope: "row" }, "before the comfort-only rule"), a.topicHit.map((t) => h("td", {}, t.before.toFixed(2)))),
          h("tr", { class: "row--main" }, h("th", { scope: "row" }, "with it (published)"), a.topicHit.map((t) => h("td", {}, t.after.toFixed(2))))))),
    p(h("small", { class: "muted" }, `With the current lists the rules catch all ${fmtInt(a.storedCrisis)} stored crisis messages, but those are the messages they were fixed against, `,
      "so that says little about new wording. The site’s safety rests on the design: a missed message gets topics to choose from, never a verse. ",
      "Every number is in ", src("evals/results/search.json"), ", ", src("evals/search_probes.json"), " and the ", src("README.md", "README"), ".")),
  );
}

// ------------------------------------------------------------------ 09 limits

function limitsChapter(): HTMLElement {
  const ev = repo.evals;
  const invented = (b: BookId) => Math.round(1 / (1 - ev.words[b].realWordFrac));
  return chapter("built-limits", "09", "Honest limits",
    h("ul", { class: "limits" },
      h("li", {}, h("strong", {}, "It imitates style, nothing more. "), "It predicts the next character from the previous 256. It does not know what the words mean, cannot answer questions, and its sentences often make no sense."),
      h("li", {}, h("strong", {}, "The Gita is its weakest book. "), `It is the smallest of the three, and Gita-style samples leak phrases from the Quran and the Bible ("Lord of the Worlds" is a Quran phrase). Only ${ev.style.rows[0].cells.gita.counts.gita} of ${ev.style.rows[0].cells.gita.n} Gita samples are recognised as Gita.`),
      h("li", {}, h("strong", {}, "It invents words. "), `About one Gita-style word in ${invented("gita")} does not occur anywhere in the training text; for the Bible it is about one in ${invented("bible")}.`),
      h("li", {}, h("strong", {}, "It can wander. "), "Mid-sample it sometimes writes another book’s tag and carries on in that book’s style. The demo marks where that happens."),
      h("li", {}, h("strong", {}, "It is tiny. "), `${fmtInt(repo.params.total)} parameters, trained on ${(BOOKS.reduce((a, b) => a + repo.stats[b].characters, 0) / 1e6).toFixed(1)} million characters. Modern large language models are many thousands of times larger and are trained on vastly more text.`),
      h("li", {}, h("strong", {}, "Its output is not scripture. "), "Nothing it writes is a quotation, a translation or a teaching of any of these books, and it should never be quoted as one."),
    ),
    p(h("small", { class: "muted" }, `The browser runs the weights quantised to 8 bits. Measured with the eval’s own scorer, that changes the mean validation loss by `,
      `${(manifest.val_loss_nats.int8.overall - manifest.val_loss_nats.float32.overall).toFixed(4)} nats (${manifest.val_loss_nats.float32.overall.toFixed(4)} to ${manifest.val_loss_nats.int8.overall.toFixed(4)}), far inside the intervals above.`)),
  );
}

export function renderStory(list: HTMLElement): void {
  list.replaceChildren(dataChapter(), tokenizerChapter(), modelChapter(), trainingChapter(), mistakeChapter(),
    evalsChapter(), testingChapter(), adviceChapter(), limitsChapter());
}

export const heroFacts = () => ({
  params: repo.params.total,
  vocab: repo.vocab.length,
  verses: BOOKS.reduce((a, b) => a + repo.stats[b].verses, 0),
});
