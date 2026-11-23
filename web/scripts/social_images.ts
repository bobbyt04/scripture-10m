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
@font-face { font-family: "Sans"; font-weight: 100 900; src: url(data:font/woff2;base64,${font("inter/files/inter-latin-wght-normal.woff2")}) format("woff2"); }
html, body { margin: 0; }
body { background: #fff; color: #111; font-family: "Sans"; }
`;

const og = `<style>${css}
.card { box-sizing: border-box; width: 1200px; height: 630px; padding: 96px 104px; border-bottom: 12px solid #e2e2e2; }
h1 { font-weight: 600; font-size: 96px; letter-spacing: -0.02em; margin: 0 0 40px; }
p { font-size: 38px; line-height: 1.4; color: #333; max-width: 960px; margin: 0 0 28px; }
.mono { font-family: ui-monospace, Menlo, monospace; font-size: 28px; color: #595959; }
</style>
<div class="card">
  <h1>Scripture-10M</h1>
  <p>A ${params}-parameter character-level transformer trained from scratch on the Bhagavad Gita, the Quran and the Bible. It runs in your browser.</p>
  <p class="mono">${repo.params.total.toLocaleString("en-US")} parameters · ${repo.params.config.n_layer} layers · ${repo.params.config.block_size}-character context</p>
</div>`;

const icon = `<style>${css}
.icon { width: 180px; height: 180px; background: #fff; color: #111; display: grid; place-items: center; font-size: 120px; font-weight: 600; line-height: 1; padding-bottom: 8px; box-sizing: border-box; border: 6px solid #d0d0d0; }
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
