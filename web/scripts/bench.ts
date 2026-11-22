// Tokens per second of the hand-written forward pass in Node, for comparison with the browser.
//   node scripts/bench.ts
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseModel } from "../src/model/format.ts";
import { generate, seededRandom } from "../src/model/generate.ts";
import { Transformer } from "../src/model/transformer.ts";

const manifest = JSON.parse(readFileSync(fileURLToPath(new URL("../src/model/manifest.json", import.meta.url)), "utf8"));
const buf = readFileSync(fileURLToPath(new URL(`../public/${manifest.file}`, import.meta.url)));
const model = new Transformer(parseModel(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)));
const n = 400;
const t0 = performance.now();
let count = 0;
for (const _ of generate(model, [0], { maxNewTokens: n, temperature: 0.8, random: seededRandom(1) })) count++;
const s = (performance.now() - t0) / 1000;
console.log(`${count} tokens in ${s.toFixed(2)} s: ${(count / s).toFixed(1)} tokens/s`);
