import { defineConfig } from "@playwright/test";

// Smoke tests against the production build. Locally they use the installed Google Chrome; in CI, the
// Playwright Chromium that `npx playwright install chromium` downloads.
export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  // The search data loads on first use; on a busy machine the first answer can take a few seconds.
  expect: { timeout: 15_000 },
  use: {
    baseURL: "http://localhost:5301",
    channel: process.env.CI ? undefined : "chrome",
  },
  webServer: {
    command: "npx vite preview --port 5301 --strictPort",
    url: "http://localhost:5301",
    reuseExistingServer: !process.env.CI,
  },
});
