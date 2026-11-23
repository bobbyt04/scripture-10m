// Writes public/og.png (1200x630, the link preview) and public/apple-touch-icon.png (180x180): plain
// typographic images in the site's colours and fonts, rendered by Playwright's browser. Run it after changing
// the wording or the model size; both files are committed.
//
//   node scripts/social_images.ts        (from web/; uses the installed Google Chrome, or CHROMIUM=1 for Playwright's)

import { readFileSync } from "node:fs";
import { chromium } from "@playwright/test";

const here = (p: string) => new URL(p, import.meta.url);
const font = (p: string) => readFileSync(here(`../node_modules/@fontsource-variable/${p}`)).toString("base64");
const repo = JSON.parse(readFileSync(here("../src/generated/repo.json"), "utf8"));
const params = `${(repo.params.total / 1e6).toFixed(1)}M`;

const css = `
@font-face { font-family: "Serif"; font-weight: 200 900; src: url(data:font/woff2;base64,${font("source-serif-4/files/source-serif-4-latin-wght-normal.woff2")}) format("woff2"); }
@font-face { font-family: "Sans"; font-weight: 100 900; src: url(data:font/woff2;base64,${font("inter/files/inter-latin-wght-normal.woff2")}) format("woff2"); }
html, body { margin: 0; }
body { background: #0a0a0a; color: #ededed; font-family: "Sans"; }
`;

const og = `<style>${css}
.card { box-sizing: border-box; width: 1200px; height: 630px; padding: 84px 96px; display: flex; flex-direction: column; justify-content: space-between; }
.eyebrow { font-family: ui-monospace, Menlo, monospace; font-size: 24px; letter-spacing: 0.06em; text-transform: uppercase; color: #a1a1a1; margin: 0; }
h1 { font-weight: 650; font-size: 136px; letter-spacing: -0.055em; margin: 18px 0 0; }
.lede { font-size: 34px; line-height: 1.4; color: #a1a1a1; max-width: 960px; margin: 0; }
.books { display: flex; gap: 36px; font-size: 28px; color: #ededed; }
.books span::before { content: ""; display: inline-block; width: 16px; height: 16px; border-radius: 50%; margin-right: 12px; background: var(--c); }
</style>
<div class="card">
  <div><p class="eyebrow">${repo.params.total.toLocaleString("en-US")} params · ${repo.params.config.n_layer} layers · character-level</p><h1>Scripture-10M</h1></div>
  <p class="lede">A ${params}-parameter transformer, trained from scratch. It runs in your browser.</p>
  <div class="books"><span style="--c:#f59e0b">Bhagavad Gita</span><span style="--c:#34d399">Quran</span><span style="--c:#60a5fa">Bible</span></div>
</div>`;

const icon = `<style>${css}
.icon { width: 180px; height: 180px; background: #0a0a0a; color: #ffffff; display: grid; place-items: center; font-size: 116px; font-weight: 650; letter-spacing: -0.04em; line-height: 1; padding-bottom: 8px; box-sizing: border-box; }
</style><div class="icon">S</div>`;

const browser = await chromium.launch(process.env.CHROMIUM ? {} : { channel: "chrome" });
for (const [html, width, height, out] of [[og, 1200, 630, "og.png"], [icon, 180, 180, "apple-touch-icon.png"]] as const) {
  const page = await browser.newPage({ viewport: { width, height } });
  await page.setContent(html);
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: new URL(`../public/${out}`, import.meta.url).pathname });
  await page.close();
  console.log(`wrote public/${out}`);
}
await browser.close();
