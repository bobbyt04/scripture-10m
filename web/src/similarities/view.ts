import { BOOK_NAMES } from "../advice/api.ts";
import { h } from "../dom.ts";
import type { Pair, Similarities } from "./api.ts";

const PAGE = 4;


function pairCard(p: Pair, theme: string): HTMLElement {
  return h("article", { class: "pair", "data-pair": "" },
    h("p", { class: "pair__theme" }, theme),
    h("div", { class: "pair__cols" },
      p.verses.map((v) =>
        h("figure", { class: `pair__verse book--${v.book}` },
          h("figcaption", {}, h("span", { class: "pair__book" }, BOOK_NAMES[v.book]), h("span", { class: "pair__ref" }, v.ref)),
          h("blockquote", {}, v.text)))),
    h("p", { class: "pair__idea" }, p.idea));
}

export function renderSimilarities(root: HTMLElement, data: Similarities): void {
  const all = data.themes.flatMap((t) => t.pairs.map((p) => ({ p, theme: t.name })));
  let filter = "All";
  let shown = PAGE;
  const list = h("div", { class: "pairs", "aria-live": "polite" });
  const more = h("button", { type: "button", class: "button button--quiet" }, "Show more");
  const chips = h("div", { class: "chips", role: "group", "aria-label": "Filter by theme" });

  const draw = () => {
    const rows = all.filter((r) => filter === "All" || r.theme === filter);
    list.replaceChildren(...rows.slice(0, shown).map((r) => pairCard(r.p, r.theme)));
    more.hidden = rows.length <= shown;
    for (const c of chips.querySelectorAll("button")) c.setAttribute("aria-pressed", String(c.dataset.theme === filter));
  };
  for (const name of ["All", ...data.themes.map((t) => t.name)]) {
    chips.append(h("button", {
      type: "button", class: "chip", "data-theme": name,
      onclick: () => { filter = name; shown = PAGE; draw(); },
    }, name));
  }
  more.addEventListener("click", () => { shown += PAGE; draw(); });
  root.replaceChildren(
    ...(data.isMock ? [h("p", { class: "preview-note" }, h("strong", {}, "Preview data. "),
      "These few pairs are placeholders while the reviewed set is finished; the verse texts are real.")] : []),
    chips, list, h("div", { class: "pairs__more" }, more));
  draw();
}
