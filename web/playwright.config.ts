import { defineConfig } from "@playwright/test";

// PORT serves the build on another port when 5301 is taken.
const port = Number(process.env.PORT ?? 5301);

// Smoke tests against the production build. Locally they use the installed Google Chrome; in CI, the
// Playwright Chromium that `npx playwright install chromium` downloads.
export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  // The search data loads on first use; on a busy machine the first answer can take a few seconds.
  expect: { timeout: 15_000 },
  use: {
    baseURL: `http://localhost:${port}`,
    channel: process.env.CI ? undefined : "chrome",
  },
  webServer: {
    command: `npx vite preview --port ${port} --strictPort`,
    url: `http://localhost:${port}`,
    reuseExistingServer: !process.env.CI,
  },
});
