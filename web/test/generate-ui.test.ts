import { describe, expect, test } from "vitest";
import { toLines } from "../src/generate/ui.ts";

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
