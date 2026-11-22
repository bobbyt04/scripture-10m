import { readFileSync } from "node:fs";
import { defineConfig, type Plugin } from "vitest/config";

/** Fill {{placeholders}} in index.html with numbers from the build-time data, so they are in the static HTML. */
function repoNumbers(): Plugin {
  return {
    name: "repo-numbers",
    transformIndexHtml(html) {
      const repo = JSON.parse(readFileSync(new URL("./src/generated/repo.json", import.meta.url), "utf8"));
      const manifest = JSON.parse(readFileSync(new URL("./src/model/manifest.json", import.meta.url), "utf8"));
      const books = ["gita", "quran", "bible"] as const;
      const values: Record<string, string> = {
        params: repo.params.total.toLocaleString("en-US"),
        paramsShort: `${(repo.params.total / 1e6).toFixed(1)}M`,
        vocab: String(repo.vocab.length),
        verses: books.reduce((a: number, b) => a + repo.stats[b].verses, 0).toLocaleString("en-US"),
        modelSize: `${(manifest.bytes / 1e6).toFixed(1)} MB`,
      };
      return html.replace(/\{\{(\w+)\}\}/g, (_, k: string) => {
        if (!(k in values)) throw new Error(`index.html: unknown placeholder ${k}`);
        return values[k];
      });
    },
  };
}

export default defineConfig({
  plugins: [repoNumbers()],
  build: {
    target: "es2022",
    assetsInlineLimit: 0,
  },
  worker: {
    format: "es",
  },
  test: {
    include: ["test/**/*.test.ts"],
    testTimeout: 60_000,
  },
});
