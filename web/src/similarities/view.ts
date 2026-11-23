import { BOOK_NAMES } from "../advice/api.ts";
import { fmtInt, h } from "../dom.ts";
import type { Pair, Similarities } from "./api.ts";

const PAGE = 6;

export const SOURCE_LABEL: Record<Pair["source"], string> = {
  "model, reviewed": "proposed by word statistics, then reviewed",
  "hand-listed": "from a manually compiled list, then reviewed",
};

function pairItem(p: Pair, theme: string): HTMLElement {
  return h("article", { class: "pair", "data-pair": "" },
    p.verses.map((v) =>
      h("figure", { class: "verse" },
        h("blockquote", {}, v.text),
        h("figcaption", {}, h("span", { class: "ref" }, `${BOOK_NAMES[v.book]} ${v.ref}`)))),
    h("p", { class: "muted" }, p.idea, " ", h("small", {}, `${theme}; `, h("span", { "data-source": p.source }, SOURCE_LABEL[p.source]), ".")));
}

export function renderSimilarities(root: HTMLElement, data: Similarities): void {
  const all = data.themes.flatMap((t) => t.pairs.map((p) => ({ p, theme: t.name })));
  const fromModel = all.filter((r) => r.p.source === "model, reviewed").length;
  let filter = "All";
  let shown = PAGE;
  const list = h("div", {});
  const more = h("button", { type: "button", class: "button" });
  const status = h("span", { class: "muted", "aria-live": "polite" });
  const select = h("select", { "aria-label": "Theme" },
    h("option", { value: "All" }, `All themes (${all.length})`),
    data.themes.map((t) => h("option", { value: t.name }, `${t.name} (${t.pairs.length})`)));

  const draw = () => {
    const rows = all.filter((r) => filter === "All" || r.theme === filter);
    list.replaceChildren(...rows.slice(0, shown).map((r) => pairItem(r.p, r.theme)));
    const left = rows.length - Math.min(shown, rows.length);
    more.hidden = left === 0;
    more.textContent = `Show ${Math.min(PAGE, left)} more`;
    status.textContent = `${Math.min(shown, rows.length)} of ${rows.length}`;
  };
  select.addEventListener("change", () => { filter = select.value; shown = PAGE; draw(); });
  more.addEventListener("click", () => { shown += PAGE; draw(); });
  root.replaceChildren(
    h("p", {}, `${all.length} pairs of passages that tell the same story or teach the same thing in two of the books.`),
    h("p", {}, h("label", {}, "Theme: ", select)),
    list,
    h("p", { class: "pairs__more" }, more, " ", status),
    h("p", { class: "note" },
      `${fromModel} pairs come from word statistics built in this repo: ${data.kept} of the ${fmtInt(data.candidates_proposed)} candidates it proposed `,
      `(${Math.round(100 * data.kept_fraction)}%) were kept after review. The other ${all.length - fromModel} come from a manually compiled list of known parallels, `,
      `reviewed the same way (${data.hand_listed_left_out} left out).`));
  draw();
}
