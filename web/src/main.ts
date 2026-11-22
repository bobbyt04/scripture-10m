import "@fontsource-variable/inter/wght.css";
import "@fontsource-variable/source-serif-4/wght.css";
import "./styles.css";

import { adviceApi } from "./advice/api.ts";
import { AdviceController } from "./advice/state.ts";
import { mountAdvice } from "./advice/view.ts";
import { mountGenerate } from "./generate/ui.ts";
import { loadSimilarities } from "./similarities/api.ts";
import { renderSimilarities } from "./similarities/view.ts";
import { renderStory } from "./story/story.ts";

document.documentElement.classList.add("js");

mountAdvice(document.querySelector<HTMLFormElement>("#ask")!, document.querySelector<HTMLElement>("#ask-results")!,
  new AdviceController(adviceApi));
mountGenerate(document.querySelector<HTMLElement>("[data-generate]")!);
void loadSimilarities().then((d) => renderSimilarities(document.querySelector<HTMLElement>("[data-similarities]")!, d));
renderStory(document.querySelector<HTMLElement>("[data-story]")!);

// Reveal chapters as they scroll into view. With reduced motion the CSS shows everything at once.
const io = typeof IntersectionObserver === "function"
  ? new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add("is-visible");
        io?.unobserve(e.target);
      }
    }
  }, { rootMargin: "0px 0px -12% 0px", threshold: 0.08 })
  : null;
document.querySelectorAll(".reveal").forEach((el) => (io ? io.observe(el) : el.classList.add("is-visible")));
