import { BOOK_NAMES } from "../advice/api.ts";
import { fmtInt, h } from "../dom.ts";
import type { Pair, Similarities } from "./api.ts";

const PAGE = 6;

export const SOURCE_LABEL: Record<Pair["source"], string> = {
  "model, reviewed": "Proposed by word statistics built in this repo, then reviewed",
  "hand-listed": "From a hand-written list of known parallels, then reviewed",
};

function pairCard(p: Pair, theme: string): HTMLElement {
  return h("article", { class: "pair", "data-pair": "" },
    h("p", { class: "pair__theme" }, theme),
    h("div", { class: "pair__cols" },
      p.verses.map((v) =>
        h("figure", { class: `pair__verse book--${v.book}` },
          h("figcaption", {}, h("span", { class: "pair__book" }, BOOK_NAMES[v.book]), h("span", { class: "pair__ref" }, v.ref)),
          h("blockquote", {}, v.text)))),
    h("p", { class: "pair__idea" }, p.idea),
    h("p", { class: "pair__source", "data-source": p.source }, SOURCE_LABEL[p.source]));
}

export function renderSimilarities(root: HTMLElement, data: Similarities): void {
  const all = data.themes.flatMap((t) => t.pairs.map((p) => ({ p, theme: t.name })));
  const fromModel = all.filter((r) => r.p.source === "model, reviewed").length;
  let filter = "All";
  let shown = PAGE;
  const list = h("div", { class: "pairs" });
  const more = h("button", { type: "button", class: "button button--quiet" });
  const status = h("p", { class: "pairs__status", "aria-live": "polite" });
  const chips = h("div", { class: "chips", role: "group", "aria-label": "Filter by theme" });

  const draw = () => {
    const rows = all.filter((r) => filter === "All" || r.theme === filter);
    list.replaceChildren(...rows.slice(0, shown).map((r) => pairCard(r.p, r.theme)));
    const left = rows.length - Math.min(shown, rows.length);
    more.hidden = left === 0;
    more.textContent = `Show ${Math.min(PAGE, left)} more`;
    status.textContent = `Showing ${Math.min(shown, rows.length)} of ${rows.length} pairs${filter === "All" ? "" : ` in ${filter}`}.`;
    for (const c of chips.querySelectorAll("button")) c.setAttribute("aria-pressed", String(c.dataset.theme === filter));
  };
  const chip = (name: string, n: number) => h("button", {
    type: "button", class: "chip", "data-theme": name,
    onclick: () => { filter = name; shown = PAGE; draw(); },
  }, name, h("span", { class: "chip__n" }, String(n)));
  chips.append(chip("All", all.length), ...data.themes.map((t) => chip(t.name, t.pairs.length)));
  more.addEventListener("click", () => { shown += PAGE; draw(); });
  root.replaceChildren(
    h("p", { class: "pairs__about" },
      `${all.length} pairs. ${fromModel} were proposed by word statistics built in this repo, then reviewed: `,
      `of the ${fmtInt(data.candidates_proposed)} candidate pairs the statistics proposed, ${data.kept} (${Math.round(100 * data.kept_fraction)}%) `,
      `are kept as real parallels in story or teaching after several rounds of review. The other ${all.length - fromModel} come from a `,
      `hand-written list of known parallels, reviewed the same way (${data.hand_listed_left_out} were left out).`),
    chips, list, h("div", { class: "pairs__more" }, status, more));
  draw();
}
