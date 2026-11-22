import { expect, test } from "@playwright/test";

test("page loads without errors and without the model", async ({ page }) => {
  const errors: string[] = [];
  const modelRequests: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("request", (r) => { if (r.url().includes("/model/")) modelRequests.push(r.url()); });
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: "Scripture-10M" })).toBeVisible();
  await expect(page.locator("#built-evals")).toContainText("Scripture-10M");
  await page.waitForLoadState("networkidle");
  expect(modelRequests).toEqual([]); // the weights load only when Generate is pressed
  expect(errors).toEqual([]);
});

test.describe("advice", () => {
  // Fail the test the moment a verse element exists while no topic has been tapped.
  async function watchVerses(page: import("@playwright/test").Page) {
    await page.evaluate(() => {
      const w = window as unknown as { tapped: boolean; versesBeforeTap: number };
      w.tapped = false;
      w.versesBeforeTap = 0;
      document.addEventListener("click", (e) => { if ((e.target as Element).closest("[data-topic]")) w.tapped = true; }, true);
      new MutationObserver(() => {
        if (!w.tapped && document.querySelector("[data-verse]")) w.versesBeforeTap++;
      }).observe(document.body, { childList: true, subtree: true });
    });
  }
  const versesBeforeTap = (page: import("@playwright/test").Page) =>
    page.evaluate(() => (window as unknown as { versesBeforeTap: number }).versesBeforeTap);

  test("a crisis message renders the note and zero verse elements", async ({ page }) => {
    await page.goto("/");
    await watchVerses(page);
    await page.fill("#ask-input", "I want to end my life");
    await page.keyboard.press("Enter");
    const note = page.locator("[data-crisis]");
    await expect(note).toBeVisible();
    await expect(note).toContainText("We are not going to show verses for this one.");
    await expect(page.locator("#ask-results [data-topic], #ask-results [data-verse], #ask-results button")).toHaveCount(0);
    await expect(page.locator("[data-verse]")).toHaveCount(0);
    expect(await versesBeforeTap(page)).toBe(0);
  });

  test("an everyday worry renders topic chips and no verses until a chip is tapped", async ({ page }) => {
    const searchData: string[] = [];
    page.on("request", (r) => { if (/search[-.].*\.js$/.test(new URL(r.url()).pathname)) searchData.push(r.url()); });
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    expect(searchData).toEqual([]); // the search data loads on first use, not with the page
    await watchVerses(page);
    await page.fill("#ask-input", "I got laid off and I'm scared about money");
    await page.click("#ask button[type=submit]");
    const chips = page.locator("#ask-results [data-topic]");
    await expect(chips).toHaveCount(2);
    await expect(chips.first()).toContainText("Money worries");
    await expect(page.locator("[data-verse]")).toHaveCount(0);

    await chips.first().click();
    await expect(page.locator("#ask-results [data-verse]").first()).toBeVisible();
    await expect(page.locator("[data-empty=gita]")).toHaveText("The Bhagavad Gita has no verse for this topic in the reviewed selection.");
    expect(await versesBeforeTap(page)).toBe(0);

    await page.getByRole("button", { name: "← Topics" }).click();
    await expect(chips).toHaveCount(2);
    await expect(page.locator("[data-verse]")).toHaveCount(0);
  });

  test("a message that matches nothing shows the full topic list, still with no verses", async ({ page }) => {
    await page.goto("/");
    await watchVerses(page);
    await page.fill("#ask-input", "my cat knocked a plant off the shelf");
    await page.keyboard.press("Enter");
    await expect(page.locator("#ask-results")).toContainText("I couldn’t match that to a topic");
    expect(await page.locator("#ask-results [data-topic]").count()).toBeGreaterThan(20);
    await expect(page.locator("[data-verse]")).toHaveCount(0);
    expect(await versesBeforeTap(page)).toBe(0);
  });
});

test("parallels: every reviewed pair, a page at a time, with its source", async ({ page }) => {
  await page.goto("/");
  const section = page.locator("[data-similarities]");
  await expect(section.locator("[data-pair]")).toHaveCount(6);
  await expect(section.locator(".chip").first()).toContainText("All45");
  await expect(section.locator("[data-source]").first()).toHaveText(/then reviewed$/);
  await section.getByRole("button", { name: "Prophets and their stories" }).click();
  await section.getByRole("button", { name: /^Show \d+ more$/ }).click();
  await expect(section.locator("[data-pair]")).toHaveCount(12);
});

test("generate produces labelled text in the browser", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("radio", { name: "Bible" }).check({ force: true });
  await page.click("[data-gen-start]");
  await expect(page.locator("[data-generate]")).toHaveAttribute("data-tokens-per-second", /\d/, { timeout: 110_000 });
  const text = await page.locator("[data-gen-output]").innerText();
  expect(text.length).toBeGreaterThan(200);
  expect(text).not.toMatch(/<(gita|quran|bible)>/);
  await expect(page.locator(".gen__label")).toContainText("Machine-generated imitation");
  await expect(page.locator("[data-gen-stop]")).toBeHidden();
});

test.describe("reduced motion", () => {
  test.use({ reducedMotion: "reduce" });
  test("story content is visible without scrolling animations", async ({ page }) => {
    await page.goto("/");
    const opacity = await page.locator("#built-limits .chapter__meta").evaluate((el) => getComputedStyle(el).opacity);
    expect(opacity).toBe("1"); // not yet scrolled into view, but shown anyway
  });
});
