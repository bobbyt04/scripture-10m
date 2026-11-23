import "@fontsource-variable/inter/wght.css";
import "@fontsource-variable/source-serif-4/wght.css";
import "./styles.css";

import { searchApi } from "./advice/api.ts";
import { AdviceController } from "./advice/state.ts";
import { mountResults } from "./advice/view.ts";
import { mountGenerate } from "./generate/ui.ts";
import { mountPrompt } from "./prompt.ts";
import { loadSimilarities } from "./similarities/api.ts";
import { renderSimilarities } from "./similarities/view.ts";
import { h } from "./dom.ts";
import { renderStory } from "./story/story.ts";

document.documentElement.classList.add("js");

const ctl = new AdviceController(searchApi);
const results = document.querySelector<HTMLElement>("#ask-results")!;
const gen = document.querySelector<HTMLElement>("[data-generate]")!;
renderStory(document.querySelector<HTMLElement>("[data-story]")!); // before mountPrompt, which wires its "Run it live" link
mountResults(results, ctl);
mountPrompt(document.querySelector<HTMLFormElement>("#ask")!, results, gen, ctl, (onPhase, onRate) => mountGenerate(gen, onPhase, onRate),
  async (text) => {
    const s = await searchApi.suggest(text);
    return s.category === "crisis" ? s.note ?? "" : null;
  });


// A link to a section (/#built-failed) must land on it even though the chapters, the parallels and the fonts
// arrive after the browser's first scroll. Re-scroll as each arrives, until the reader scrolls themselves.
let readerScrolled = false;
for (const ev of ["wheel", "touchmove", "keydown", "mousedown"]) {
  addEventListener(ev, () => (readerScrolled = true), { once: true, passive: true });
}
const landOnHash = () => {
  if (readerScrolled || location.hash.length < 2) return;
  document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();
};
landOnHash();
void document.fonts?.ready.then(landOnHash);

const parallels = document.querySelector<HTMLElement>("[data-similarities]")!;
function showParallels(): void {
  loadSimilarities()
    .then((d) => renderSimilarities(parallels, d))
    .catch(() => parallels.replaceChildren(h("p", { class: "pairs__error", role: "alert" },
      "The parallels could not be loaded. ",
      h("button", { type: "button", class: "textlink", onclick: () => showParallels() }, "Try again"))))
    .finally(landOnHash);
}
showParallels();
