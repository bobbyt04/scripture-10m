import { readFileSync } from "node:fs";
import { defineConfig, type Plugin } from "vitest/config";

// The site's public address, without a trailing slash, e.g. SITE_URL=https://example.org npm run build.
const SITE_URL = (process.env.SITE_URL ?? "").replace(/\/+$/, "");

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
        layers: String(repo.params.config.n_layer),
        heads: String(repo.params.config.n_head),
        dims: String(repo.params.config.n_embd),
        context: String(repo.params.config.block_size),
        verses: books.reduce((a: number, b) => a + repo.stats[b].verses, 0).toLocaleString("en-US"),
        modelSize: `${(manifest.bytes / 1e6).toFixed(1)} MB`,
        siteUrl: SITE_URL,
      };
      // The canonical address is known only when SITE_URL is set at build time (for example on Vercel); without it
      // the tags are left out rather than guessed.
      const canonical = SITE_URL
        ? `<link rel="canonical" href="${SITE_URL}/" />\n    <meta property="og:url" content="${SITE_URL}/" />`
        : "";
      return html.replace("<!--canonical-->", canonical).replace(/\{\{(\w+)\}\}/g, (_, k: string) => {
        if (!(k in values)) throw new Error(`index.html: unknown placeholder ${k}`);
        return values[k];
      });
    },
  };
}

/** The SIL Open Font License of the two self-hosted fonts, shipped next to the font files. */
function fontLicences(): Plugin {
  const fonts = { "Inter-OFL.txt": "inter", "SourceSerif4-OFL.txt": "source-serif-4" };
  return {
    name: "font-licences",
    generateBundle() {
      for (const [name, pkg] of Object.entries(fonts)) {
        const source = readFileSync(new URL(`./node_modules/@fontsource-variable/${pkg}/LICENSE`, import.meta.url), "utf8");
        this.emitFile({ type: "asset", fileName: `fonts/${name}`, source });
      }
    },
  };
}

export default defineConfig({
  plugins: [repoNumbers(), fontLicences()],
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
