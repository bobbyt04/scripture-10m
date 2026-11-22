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

test("advice: a crisis message shows the note and nothing else", async ({ page }) => {
  await page.goto("/");
  await page.fill("#ask-input", "I want to end my life");
  await page.keyboard.press("Enter");
  await expect(page.locator("[data-crisis]")).toBeVisible();
  await expect(page.locator("#ask-results [data-topic], #ask-results [data-verse]")).toHaveCount(0);
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
