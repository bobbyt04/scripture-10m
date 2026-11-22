import { describe, expect, test } from "vitest";
import { errorMessage, toLines } from "../src/generate/ui.ts";
import { verifyDownload } from "../src/generate/verify.ts";

describe("display of generated text", () => {
  test("drops book tags and keeps one line per generated verse, with no verse numbers", () => {
    const lines = toLines("Krishna said: be steady.\n<gita>The wise grieve not.\n<gi", "gita");
    expect(lines).toEqual([
      { text: "Krishna said: be steady.", switchedTo: null },
      { text: "The wise grieve not.", switchedTo: null },
      { text: "<gi", switchedTo: null },
    ]);
    expect(lines.every((l) => !/\d/.test(l.text))).toBe(true);
  });

  test("flags a switch to another book's style", () => {
    expect(toLines("first\n<bible>And it came to pass", "gita")[1]).toEqual({ text: "And it came to pass", switchedTo: "bible" });
  });
});

describe("model download", () => {
  const data = new TextEncoder().encode("abc").buffer as ArrayBuffer;
  const sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"; // SHA-256 of "abc"

  test("a complete download with the right checksum passes", async () => {
    await expect(verifyDownload(data, 3, sha)).resolves.toBeUndefined();
  });

  test("a cut-off or altered download is rejected", async () => {
    await expect(verifyDownload(data, 4, sha)).rejects.toThrow("expected 4");
    await expect(verifyDownload(data, 3, "0".repeat(64))).rejects.toThrow("checksum");
  });

  test("errors read as plain advice, with a way to try again", () => {
    expect(errorMessage(false)).toMatch(/Try again\.$/);
    expect(errorMessage(true)).toMatch(/try again\.$/);
  });
});
