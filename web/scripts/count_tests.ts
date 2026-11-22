// Count the repository's tests and write src/story/test-counts.json, which the Testing chapter reads.
// Collecting the Python tests needs PyTorch, which the site build does not have, so the counts are committed;
// rerun this after adding tests:
//
//   PYTHON=../.venv/bin/python node scripts/count_tests.ts      (from web/, after npm run extract)

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = join(dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = join(WEB, "..");
const run = (cmd: string, args: string[], cwd: string) => execFileSync(cmd, args, { cwd, encoding: "utf8", maxBuffer: 64 << 20 });

function lastNumber(out: string, re: RegExp, what: string): number {
  const m = out.match(re);
  if (!m) throw new Error(`could not count ${what}:\n${out.slice(-500)}`);
  return Number(m[1]);
}

// PYTHON may carry arguments, e.g. "arch -arm64 ../.venv/bin/python" when Node runs under Rosetta.
const [python, ...pyArgs] = (process.env.PYTHON ?? "python3").split(" ");
const pytest = lastNumber(run(python, [...pyArgs, "-m", "pytest", "--collect-only", "-q"], ROOT), /(\d+) tests? collected/, "Python tests");
// `vitest list` leaves out tests made in loops, so the suite is run and its report counted.
const reportFile = join(mkdtempSync(join(tmpdir(), "vitest-")), "report.json");
run("npx", ["vitest", "run", "--reporter=json", `--outputFile=${reportFile}`], WEB);
const vitestReport = JSON.parse(readFileSync(reportFile, "utf8")) as { numTotalTests: number; success: boolean };
if (!vitestReport.success) throw new Error("the unit tests fail; fix them before counting");
const vitest = vitestReport.numTotalTests;
const playwright = lastNumber(run("npx", ["playwright", "test", "--list"], WEB), /Total: (\d+) tests?/, "Playwright tests");

const counts = {
  about: "Written by web/scripts/count_tests.ts; each count is what the command lists.",
  python: { tests: pytest, command: "python -m pytest --collect-only -q" },
  vitest: { tests: vitest, command: "npx vitest run (in web/)" },
  playwright: { tests: playwright, command: "npx playwright test --list (in web/)" },
};
writeFileSync(join(WEB, "src/story/test-counts.json"), JSON.stringify(counts, null, 2) + "\n");
console.log(`python ${pytest}, vitest ${vitest}, playwright ${playwright}`);
