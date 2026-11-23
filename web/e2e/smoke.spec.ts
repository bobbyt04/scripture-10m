import { expect, test, type Page } from "@playwright/test";

// The first answer waits for the search data (104 kB gzipped) to load and parse; on a busy machine, with the
// model generating in another test, that has taken over 15 s.
const FIRST_ANSWER = 30_000;

test("page loads without errors and without the model", async ({ page }) => {
  const errors: string[] = [];
  const modelRequests: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("request", (r) => { if (r.url().includes("/model/")) modelRequests.push(r.url()); });
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: "Scripture-10M" })).toBeVisible();
  await expect(page.locator("#built-results")).toContainText("Scripture-10M");
  // Before anything runs, the live slot is hidden; the Model section shows a recorded sample, labelled as such.
  await expect(page.locator("[data-generate]")).toBeHidden();
  await expect(page.locator("#built-model")).toContainText("A recorded sample, Bible style (samples/bible.md, seed 1");
  expect(await page.locator("[data-sample] p").count()).toBeGreaterThan(0);
  await expect(page.locator("[data-gen-run]")).toHaveText(/^Run it live \(\d+\.\d MB\)$/);
  await page.waitForLoadState("networkidle");
  expect(modelRequests).toEqual([]); // the weights load only when the model is asked to write
  expect(errors).toEqual([]);
});

/** Everything typed must stay in the page: record every request URL and check none carries the text. */
function watchRequests(page: Page): string[] {
  const urls: string[] = [];
  page.on("request", (r) => urls.push(r.url()));
  return urls;
}

test.describe("find verses", () => {
  test("Enter shows the first topic's verses at once, with the other topics as chips", async ({ page }) => {
    const searchData: string[] = [];
    page.on("request", (r) => { if (/search[-.].*\.js$/.test(new URL(r.url()).pathname)) searchData.push(r.url()); });
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    expect(searchData).toEqual([]); // the search data loads on first use, not with the page
    const urls = watchRequests(page);
    await page.getByRole("button", { name: "Verses", exact: true }).click();
    await page.fill("#ask-input", "I got laid off and I'm scared about money");
    await page.keyboard.press("Enter");
    const results = page.locator("#ask-results");
    await expect(results.locator(".answer__title")).toHaveText("Money worries", { timeout: FIRST_ANSWER });
    await expect(results.locator("[data-verse]").first()).toBeInViewport(); // scrolled to, without a second click
    await expect(page.locator("#ask-input")).toBeFocused(); // a follow-up can be typed at once
    await expect(results.locator("[data-verse]").first()).toBeVisible();
    await expect(page.locator("[data-empty=gita]")).toHaveText("The Bhagavad Gita has no verse for this topic in the reviewed selection.");
    const chips = results.locator("[data-topic]");
    await expect(chips).toHaveCount(2);
    await expect(chips.first()).toHaveAttribute("aria-pressed", "true");

    await chips.nth(1).click();
    await expect(results.locator(".answer__title")).toHaveText("Worry and fear");
    await expect(chips.nth(1)).toHaveAttribute("aria-pressed", "true");
    await expect(results.locator("[data-verse]").first()).toBeVisible();

    await results.getByRole("button", { name: "All topics" }).click();
    expect(await results.locator("[data-topic]").count()).toBeGreaterThan(20);
    await expect(results.locator("[data-verse]")).toHaveCount(0);
    await results.getByRole("button", { name: "Back", exact: true }).click();
    await expect(results.locator(".answer__title")).toHaveText("Worry and fear");

    expect(new URL(page.url()).search + new URL(page.url()).hash).toBe("");
    expect(urls.filter((u) => /laid|scared|money/i.test(decodeURIComponent(u)))).toEqual([]);
  });

  test("a crisis message renders the note and zero verses or topics", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Verses", exact: true }).click();
    await page.fill("#ask-input", "I want to end my life");
    await page.keyboard.press("Enter");
    const note = page.locator("[data-crisis]");
    await expect(note).toBeVisible({ timeout: FIRST_ANSWER });
    await expect(note).toContainText("This page will not show verses for what you wrote.");
    await expect(page.locator("#ask-results [data-topic], #ask-results [data-verse], #ask-results button, #ask-results a")).toHaveCount(0);
    await expect(page.locator("[data-verse]")).toHaveCount(0);
    expect(new URL(page.url()).search).toBe("");
  });

  test("a message that matches nothing says so and lists every topic, with no verses", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Verses", exact: true }).click();
    await page.fill("#ask-input", "my cat knocked a plant off the shelf");
    await page.keyboard.press("Enter");
    await expect(page.locator("#ask-results")).toContainText("Nothing matched.", { timeout: FIRST_ANSWER });
    expect(await page.locator("#ask-results [data-topic]").count()).toBeGreaterThan(20);
    await expect(page.locator("[data-verse]")).toHaveCount(0);
    await page.locator("#ask-results [data-topic]").first().click();
    await expect(page.locator("#ask-results [data-verse]").first()).toBeVisible();
  });

  test("a first topic outside the auto-show allowlist is listed and waits for a tap", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Verses", exact: true }).click();
    await page.fill("#ask-input", "this homework is killing me");
    await page.keyboard.press("Enter");
    const results = page.locator("#ask-results");
    await expect(results).toContainText("Pick a topic:", { timeout: FIRST_ANSWER });
    await expect(page.locator("[data-verse]")).toHaveCount(0);
    await results.locator("[data-topic]").first().click();
    await expect(results.locator("[data-verse]").first()).toBeVisible();
  });

  test("Shift+Enter adds a new line instead of submitting", async ({ page }) => {
    await page.goto("/");
    await page.fill("#ask-input", "first line");
    await page.keyboard.press("Shift+Enter");
    await page.keyboard.type("second");
    await expect(page.locator("#ask-input")).toHaveValue("first line\nsecond");
    await expect(page.locator("#ask-results")).toBeEmpty();
    await expect(page.locator("[data-generate]")).toBeHidden();
  });
});

test("let the model write: downloads the model and streams labelled text", async ({ page }) => {
  await page.goto("/");
  // The model is the default mode; its placeholder types out example starts.
  await expect(page.getByRole("button", { name: "Model", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect.poll(() => page.locator("#ask-input").getAttribute("placeholder")).toMatch(/^(In the beginning|O Arjuna,|And when they came|Blessed are|Say: He is)/);
  const urls = watchRequests(page);
  await page.getByRole("button", { name: "Bible", exact: true }).click();
  await page.locator("[data-gen-temp]").fill("1.2");
  await expect(page.locator("[data-gen-temp-out]")).toHaveText("1.2");
  await page.fill("#ask-input", "And it came to pass");
  await page.keyboard.press("Enter");
  const gen = page.locator("[data-generate]");
  await expect(gen.locator("[data-gen-rate]")).toContainText("chars/s", { timeout: 110_000 }); // live rate while streaming
  await expect(page.locator("[data-live-rate]")).toHaveText(/^(\d+|–) chars\/s$/); // and under the box
  // Watch it think: the five most likely next characters with their probabilities, from the worker's softmax.
  const rows = gen.locator(".think__row");
  await expect(rows).toHaveCount(5);
  await expect(rows.first().locator(".think__p")).toHaveText(/^\d+\.\d%$/);
  await expect(gen.locator(".think__rest")).toContainText("others");
  // A full run is 400 characters; on a busy machine that can outlast the test, so it is stopped after 60.
  await expect.poll(async () => (await page.locator("[data-gen-output]").innerText()).length, { timeout: 100_000 }).toBeGreaterThan(60);
  await expect(page.locator("#ask button[type=submit]")).toHaveAttribute("aria-label", "Stop");
  await page.locator("#ask button[type=submit]").click();
  await expect(gen).toHaveAttribute("data-tokens-per-second", /\d/);
  await expect(gen.locator("[data-gen-status]")).toHaveText("Stopped.");
  const text = await page.locator("[data-gen-output]").innerText();
  expect(text).not.toMatch(/<(gita|quran|bible)>/);
  // The visitor's start comes first, marked apart, and the model's own text follows it.
  await expect(gen.locator("[data-gen-start]")).toHaveText("And it came to pass");
  expect(text.startsWith("And it came to pass")).toBe(true);
  expect(text.length).toBeGreaterThan("And it came to pass".length + 20);
  expect(urls.filter((u) => /came|pass/i.test(decodeURIComponent(u)))).toEqual([]);
  await expect(gen.locator(".meta")).toContainText("machine-generated imitation, not scripture");
  await expect(page.locator("#ask button[type=submit]")).toHaveAttribute("aria-label", "Continue again");
  await page.getByRole("button", { name: "Verses", exact: true }).click();
  await expect(page.locator("#ask-input")).toHaveValue("And it came to pass");
  await expect(page.locator("#ask-input")).toHaveAttribute("placeholder", "A worry or a topic: money, grief, a hard decision");
});

test("let the model write: crisis text gets the note and the model does not write", async ({ page }) => {
  const modelRequests: string[] = [];
  page.on("request", (r) => { if (r.url().includes("/model/")) modelRequests.push(r.url()); });
  await page.goto("/");
  await page.fill("#ask-input", "I want to end my life"); // the default mode is the model
  await page.keyboard.press("Enter");
  const note = page.locator("[data-generate] [data-crisis]");
  await expect(note).toBeVisible({ timeout: FIRST_ANSWER });
  await expect(note).toContainText("This page will not show verses for what you wrote.");
  await expect(page.locator("[data-gen-output] .gen__line")).toHaveCount(0);
  await expect(page.locator(".think__row")).toHaveCount(0);
  await expect(page.locator("#ask button[type=submit]")).toHaveAttribute("aria-label", "Continue");
  expect(modelRequests).toEqual([]);
  expect(new URL(page.url()).search).toBe("");
});

test.describe("without JavaScript", () => {
  test.use({ javaScriptEnabled: false });
  test("nothing typed can leave the page: no field name, no enabled submit, and a note", async ({ page }) => {
    const navigations: string[] = [];
    page.on("request", (r) => { if (r.isNavigationRequest()) navigations.push(r.url()); });
    await page.goto("/");
    await expect(page.locator(".ask__noscript")).toHaveText("This page needs JavaScript. Nothing you type is sent anywhere.");
    await expect(page.locator("#ask-input")).not.toHaveAttribute("name", /.*/);
    await expect(page.locator("#ask [name]")).toHaveCount(0);
    await expect(page.locator("#ask button[type=submit]")).toBeDisabled();
    await page.fill("#ask-input", "my private worry");
    await page.locator("#ask-input").press("Enter");
    await page.locator("#ask button[type=submit]").click({ force: true });
    await page.evaluate(() => document.querySelector<HTMLFormElement>("#ask")?.requestSubmit()).catch(() => {});
    await page.waitForTimeout(500);
    expect(navigations.filter((u) => new URL(u).search !== "")).toEqual([]);
    expect(new URL(page.url()).search).toBe("");
  });
});

test("parallels: every reviewed pair, a page at a time, with its source", async ({ page }) => {
  await page.goto("/");
  const section = page.locator("[data-similarities]");
  await expect(section.locator("[data-pair]")).toHaveCount(6);
  await expect(section.locator("select option").first()).toHaveText("All themes (45)");
  await expect(section.locator("[data-source]").first()).toHaveText(/then reviewed$/);
  await section.locator("select").selectOption("Prophets and their stories");
  await section.getByRole("button", { name: /^Show \d+ more$/ }).click();
  await expect(section.locator("[data-pair]")).toHaveCount(12);
});

test("parallels that fail to load offer a retry", async ({ page }) => {
  let fail = true;
  await page.route(/similarities[-.][^/]*\.js$/, (route) => (fail ? route.abort() : route.continue()));
  await page.goto("/");
  const section = page.locator("[data-similarities]");
  await expect(section).toContainText("The parallels could not be loaded.");
  fail = false;
  await section.getByRole("button", { name: "Try again" }).click();
  await expect(section.locator("[data-pair]")).toHaveCount(6);
});

test("a link to a chapter lands on it after the page has finished building", async ({ page }) => {
  await page.goto("/#built-failed");
  await page.waitForLoadState("networkidle");
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(800); // smooth scrolling
  const top = await page.locator("#built-failed").evaluate((el) => el.getBoundingClientRect().top);
  expect(top).toBeGreaterThanOrEqual(0);
  expect(top).toBeLessThan(200);
});

test.describe("small screens", () => {
  test.use({ viewport: { width: 375, height: 812 } });
  test("no horizontal scrolling, before and after an answer, and every section is linked", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("header nav a")).toHaveText(["How it works", "Parallels", "GitHub"]);
    const width = () => page.evaluate(() => document.documentElement.scrollWidth);
    expect(await width()).toBeLessThanOrEqual(375);
    await page.getByRole("button", { name: "Verses", exact: true }).click();
    await page.fill("#ask-input", "I keep worrying about money and whether I can pay rent");
    await page.keyboard.press("Enter");
    await expect(page.locator("#ask-results [data-verse]").first()).toBeVisible({ timeout: FIRST_ANSWER });
    await expect(page.locator("#ask-results [data-verse]").first()).toBeInViewport();
    expect(await width()).toBeLessThanOrEqual(375);
  });
});

test.describe("one plain theme", () => {
  test.use({ colorScheme: "dark" });
  test("the page stays white with dark text and nothing animates at rest", async ({ page }) => {
    await page.goto("/");
    const look = await page.evaluate(() => {
      const b = getComputedStyle(document.body);
      const moving = [...document.querySelectorAll("*")].filter((el) => {
        const c = getComputedStyle(el);
        return c.animationName !== "none";
      }).length;
      return { bg: b.backgroundColor, color: b.color, moving };
    });
    expect(look).toEqual({ bg: "rgb(255, 255, 255)", color: "rgb(13, 13, 13)", moving: 0 });
  });
});
