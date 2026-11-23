import { describe, expect, test } from "vitest";
import { errorMessage, looksLikeQuestion, showChar, toLines } from "../src/generate/ui.ts";
import { encodeStart, MAX_START_CHARS } from "../src/generate/start.ts";
import { verifyDownload } from "../src/generate/verify.ts";
import { CharTokenizer } from "../src/model/tokenizer.ts";
import vocab from "../src/model/vocab.json";

describe("the visitor's typed start", () => {
  const tok = new CharTokenizer(vocab);

  test("is the book tag followed by the typed characters", () => {
    const s = encodeStart(tok, "quran", "And lo, the");
    expect(s.ids[0]).toBe(tok.id("<quran>"));
    expect(tok.decode(s.ids)).toBe("<quran>And lo, the");
    expect(s).toMatchObject({ text: "And lo, the", dropped: 0, cut: 0 });
  });

  test("an empty box starts from the tag alone", () => {
    expect(encodeStart(tok, "gita", "   ")).toEqual({ ids: [tok.id("<gita>")], text: "", dropped: 0, cut: 0 });
  });

  test("drops characters the model has no token for, and counts them", () => {
    const s = encodeStart(tok, "bible", "Café 2024 🙂 <bible> ok");
    expect(s.text).toBe("Caf   bible ok");
    expect(s.dropped).toBe(8); // é, 2, 0, 2, 4, the emoji, < and >
    expect(s.ids.slice(1).every((id) => tok.tokens[id].length === 1)).toBe(true); // a typed tag never becomes a tag
  });

  test("turns line breaks into spaces and caps the start well inside the 256-character window", () => {
    expect(encodeStart(tok, "gita", "one\ntwo").text).toBe("one two");
    const long = encodeStart(tok, "gita", "a".repeat(MAX_START_CHARS + 30));
    expect(long.text).toHaveLength(MAX_START_CHARS);
    expect(long.cut).toBe(30);
    expect(long.ids.length).toBeLessThanOrEqual(128);
  });
});

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

  test("shows space and newline as visible symbols in the guess bars", () => {
    expect([" ", "\n", "a", "<bible>"].map(showChar)).toEqual(["␣", "↵", "a", "<bible>"]);
  });

  test("flags a switch to another book's style", () => {
    expect(toLines("first\n<bible>And it came to pass", "gita")[1]).toEqual({ text: "And it came to pass", switchedTo: "bible" });
  });

  test("detects questions typed into Model mode", () => {
    expect(looksLikeQuestion("Is god good")).toBe(true);
    expect(looksLikeQuestion("Why do bad things happen?")).toBe(true);
    expect(looksLikeQuestion("In the beginning")).toBe(false);
    expect(looksLikeQuestion("O Arjuna,")).toBe(false);
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
