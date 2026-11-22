// @vitest-environment jsdom
import { describe, expect, test, vi } from "vitest";
import { MAX_INPUT_CHARS, type AdviceApi, type Suggestion, type TopicVerses } from "../src/advice/api.ts";
import { mockAdviceApi } from "../src/advice/mock.ts";
import { AdviceController, type AdviceState } from "../src/advice/state.ts";
import { renderResults, type ResultHandlers } from "../src/advice/view.ts";

const topics = [
  { id: "worry", label: "Worry and fear", blurb: "b1" },
  { id: "anger", label: "Anger", blurb: "b2" },
  { id: "gratitude", label: "Gratitude", blurb: "b3" },
];
const verses: TopicVerses = {
  ...topics[2],
  books: {
    gita: [],
    quran: [{ ref: "14:7", text: "If ye give thanks, I will give you more.", why: "w1" }],
    bible: [{ ref: "1 Thessalonians 5:18", text: "In every thing give thanks.", why: "w2" }],
  },
};

function fakeApi(suggestion: Suggestion, overrides: Partial<AdviceApi> = {}): AdviceApi {
  return {
    isMock: false,
    suggest: vi.fn(async () => suggestion),
    topic: vi.fn(async () => verses),
    allTopics: vi.fn(async () => topics),
    ...overrides,
  };
}

const noop: ResultHandlers = { openTopic: () => {}, browseAll: () => {}, back: () => {} };
function render(state: AdviceState, on = noop): HTMLElement {
  const root = document.createElement("div");
  root.append(...renderResults(state, on));
  return root;
}

describe("advice state machine", () => {
  test("crisis: only the note, no topics or verses, and nothing else is reachable", async () => {
    const api = fakeApi({ category: "crisis", note: "Please reach out to someone you trust." });
    const ctl = new AdviceController(api);
    await ctl.submit("anything");
    expect(ctl.state).toEqual({ kind: "crisis", note: "Please reach out to someone you trust." });

    const root = render(ctl.state);
    expect(root.textContent).toBe("Please reach out to someone you trust.");
    expect(root.querySelector("[data-crisis]")).not.toBeNull();
    expect(root.querySelectorAll("[data-topic], [data-verse], [data-browse], a, button")).toHaveLength(0);
    expect(root.textContent).not.toMatch(/\d{3}/); // no phone numbers

    await ctl.openTopic("worry"); // topics are not reachable from a crisis
    expect(ctl.state.kind).toBe("crisis");
    expect(api.topic).not.toHaveBeenCalled();
    expect(api.allTopics).not.toHaveBeenCalled();
  });

  test("topics: up to four suggestions, never verses, until one is picked", async () => {
    const many = [...topics, { id: "a", label: "A", blurb: "" }, { id: "b", label: "B", blurb: "" }];
    const ctl = new AdviceController(fakeApi({ category: "topics", topics: many }));
    await ctl.submit("worried and angry");
    expect(ctl.state.kind).toBe("suggestions");
    const root = render(ctl.state);
    expect(root.querySelectorAll("[data-topic]")).toHaveLength(4);
    expect(root.querySelectorAll("[data-verse]")).toHaveLength(0);
  });

  test("none: shows the full topic list", async () => {
    const api = fakeApi({ category: "none" });
    const ctl = new AdviceController(api);
    await ctl.submit("my cat knocked over a plant");
    expect(ctl.state).toEqual({ kind: "browse", reason: "no-match", topics });
    const root = render(ctl.state);
    expect(root.querySelectorAll("[data-topic]")).toHaveLength(topics.length);
    expect(root.textContent).toContain("couldn’t match");
  });

  test("topic: per-book verses, with an explicit empty state for a book with nothing", async () => {
    const ctl = new AdviceController(fakeApi({ category: "topics", topics }));
    await ctl.submit("thankful");
    const pending = ctl.openTopic("gratitude");
    expect(ctl.state.kind).toBe("loadingTopic");
    await pending;
    expect(ctl.state.kind).toBe("topic");
    const root = render(ctl.state);
    expect(root.querySelectorAll("[data-verse]")).toHaveLength(2);
    expect(root.querySelector("[data-empty=gita]")?.textContent).toBe("Nothing in the Bhagavad Gita for this topic in this selection.");
    expect(root.textContent).toContain("1 Thessalonians 5:18");
    expect(root.textContent).toContain("Quran 14:7");

    ctl.back();
    expect(ctl.state.kind).toBe("suggestions");
  });

  test("a slow topic reply never lands on top of a newer crisis note", async () => {
    let release!: () => void;
    const slowTopic = () => new Promise<TopicVerses>((r) => (release = () => r(verses)));
    const api = fakeApi({ category: "topics", topics }, { topic: vi.fn(slowTopic) });
    const ctl = new AdviceController(api);
    await ctl.submit("thankful");
    const late = ctl.openTopic("gratitude");
    (api.suggest as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ category: "crisis", note: "note" });
    await ctl.submit("something else");
    release();
    await late;
    expect(ctl.state).toEqual({ kind: "crisis", note: "note" });
  });

  test("empty input does nothing; long input is cut to the limit", async () => {
    const api = fakeApi({ category: "none" });
    const ctl = new AdviceController(api);
    await ctl.submit("   ");
    expect(api.suggest).not.toHaveBeenCalled();
    expect(ctl.state.kind).toBe("idle");
    await ctl.submit("x".repeat(MAX_INPUT_CHARS + 50));
    expect((api.suggest as ReturnType<typeof vi.fn>).mock.calls[0][0]).toHaveLength(MAX_INPUT_CHARS);
  });

  test("a failed request shows an error, not stale results", async () => {
    const ctl = new AdviceController(fakeApi({ category: "none" }, { suggest: vi.fn(async () => { throw new Error("boom"); }) }));
    await ctl.submit("hello");
    expect(ctl.state.kind).toBe("error");
    expect(render(ctl.state).querySelector("[role=alert]")).not.toBeNull();
  });

  test("browse all topics from idle, and pick one", async () => {
    const ctl = new AdviceController(fakeApi({ category: "none" }));
    await ctl.browseAll();
    expect(ctl.state).toEqual({ kind: "browse", reason: "asked", topics });
    await ctl.openTopic("gratitude");
    expect(ctl.state.kind).toBe("topic");
  });
});

describe("mock advice api (placeholder until PR 4)", () => {
  test("is marked as a mock", () => {
    expect(mockAdviceApi.isMock).toBe(true);
  });

  test("crisis wording gets the note only", async () => {
    const s = await mockAdviceApi.suggest("I want to end my life");
    expect(s.category).toBe("crisis");
  });

  test("suggests at most four topics, or none", async () => {
    const s = await mockAdviceApi.suggest("worried, angry, thankful, struggling, confused about work");
    expect(s.category).toBe("topics");
    if (s.category === "topics") expect(s.topics.length).toBeLessThanOrEqual(4);
    expect((await mockAdviceApi.suggest("the weather is fine")).category).toBe("none");
  });
});
