// @vitest-environment jsdom
import { describe, expect, test, vi } from "vitest";
import { MAX_INPUT_CHARS, searchApi, type AdviceApi, type Suggestion, type TopicListing, type TopicResult } from "../src/advice/api.ts";
import { AdviceController, type AdviceState } from "../src/advice/state.ts";
import { renderResults, type ResultHandlers } from "../src/advice/view.ts";

const counts = (gita: number, quran: number, bible: number) => ({ gita, quran, bible });
const topics: TopicListing[] = [
  { id: "fear_anxiety", label: "Worry and fear", blurb: "b1", verses: counts(3, 4, 5), sensitive_verses: counts(0, 1, 1) },
  { id: "anger", label: "Anger", blurb: "b2", verses: counts(3, 0, 4), sensitive_verses: counts(0, 0, 0) },
  { id: "gratitude", label: "Gratitude", blurb: "b3", verses: counts(0, 3, 3), sensitive_verses: counts(0, 0, 0) },
];
const verse = (ref: string, priority: number | null) => ({ ref, text: `text of ${ref}`, why: `why ${ref}`, themes: ["gratitude"], priority });
const result: TopicResult = {
  id: "gratitude", label: "Gratitude", blurb: "b3", note: null, sensitive: false,
  books: {
    gita: { verses: [], nothing: "The Bhagavad Gita has no verse for this topic in the reviewed selection." },
    quran: { verses: [verse("14:7", 1)], nothing: null },
    bible: { verses: [verse("1 Thessalonians 5:18", 1), verse("Psalms 100:4", 2), verse("Psalms 107:1", null), verse("Colossians 3:15", null)], nothing: null },
  },
};

function fakeApi(suggestion: Suggestion, overrides: Partial<AdviceApi> = {}): AdviceApi {
  return {
    suggest: vi.fn(async () => suggestion),
    topic: vi.fn(async () => result),
    topics: vi.fn(async () => topics),
    ...overrides,
  };
}

const noop: ResultHandlers = { openTopic: () => {}, browseAll: () => {}, back: () => {} };
function render(state: AdviceState, on = noop): HTMLElement {
  const root = document.createElement("div");
  root.append(...renderResults(state, on));
  return root;
}
const general = (t = topics): Suggestion => ({ category: "general", note: null, topics: t.map(({ id, label, blurb }) => ({ id, label, blurb })) });

describe("advice state machine", () => {
  test("crisis: only the note, no topics or verses, and nothing else is reachable", async () => {
    const api = fakeApi({ category: "crisis", note: "Please reach out to someone you trust.", topics: [] });
    const ctl = new AdviceController(api);
    await ctl.submit("anything");
    expect(ctl.state).toEqual({ kind: "crisis", note: "Please reach out to someone you trust." });

    const root = render(ctl.state);
    expect(root.textContent).toBe("Please reach out to someone you trust.");
    expect(root.querySelector("[data-crisis]")).not.toBeNull();
    expect(root.querySelectorAll("[data-topic], [data-verse], [data-browse], a, button")).toHaveLength(0);

    await ctl.openTopic("fear_anxiety"); // topics are not reachable from a crisis
    expect(ctl.state.kind).toBe("crisis");
    expect(api.topic).not.toHaveBeenCalled();
    expect(api.topics).not.toHaveBeenCalled();
  });

  test("topics: up to four suggestions, never verses, until one is picked", async () => {
    const many = [...topics, { ...topics[0], id: "a" }, { ...topics[0], id: "b" }];
    const ctl = new AdviceController(fakeApi(general(many)));
    await ctl.submit("worried and angry");
    expect(ctl.state.kind).toBe("suggestions");
    const root = render(ctl.state);
    expect(root.querySelectorAll("[data-topic]")).toHaveLength(4);
    expect(root.querySelectorAll("[data-verse]")).toHaveLength(0);
  });

  test("none: shows the full topic list, with the books that have nothing dimmed", async () => {
    const api = fakeApi({ category: "none", note: null, topics: [] });
    const ctl = new AdviceController(api);
    await ctl.submit("my cat knocked over a plant");
    expect(ctl.state).toEqual({ kind: "browse", reason: "no-match", topics });
    const root = render(ctl.state);
    expect(root.querySelectorAll("[data-topic]")).toHaveLength(topics.length);
    expect(root.querySelectorAll("[data-verse]")).toHaveLength(0);
    expect(root.textContent).toContain("couldn’t match");
    expect(root.querySelectorAll("[data-topic=gratitude] .topic__book.is-empty")).toHaveLength(1);
  });

  test("topic: the hand-picked verses per book, the rest behind a button, a quiet line for an empty book", async () => {
    const ctl = new AdviceController(fakeApi(general()));
    await ctl.submit("thankful");
    const pending = ctl.openTopic("gratitude");
    expect(ctl.state.kind).toBe("loadingTopic");
    await pending;
    expect(ctl.state.kind).toBe("topic");
    const root = render(ctl.state);
    expect(root.querySelectorAll("[data-verse]")).toHaveLength(5);
    expect(root.querySelectorAll("[data-book=bible] .book__more[hidden] [data-verse]")).toHaveLength(1);
    expect(root.querySelector("[data-book=gita]")).toBeNull();
    expect(root.querySelector("[data-empty=gita]")?.textContent).toBe("The Bhagavad Gita has no verse for this topic in the reviewed selection.");
    expect(root.textContent).toContain("Quran 14:7");
    root.querySelector<HTMLButtonElement>(".book__toggle")!.click();
    expect(root.querySelectorAll(".book__more[hidden]")).toHaveLength(0);

    ctl.back();
    expect(ctl.state.kind).toBe("suggestions");
  });

  test("a sensitive suggestion opens its topic with sensitive verses only", async () => {
    const api = fakeApi({ category: "sensitive", note: null, topics: [{ id: "self_worth", label: "You are of worth", blurb: "g" }] });
    const ctl = new AdviceController(api);
    await ctl.submit("anything");
    expect(ctl.state).toMatchObject({ kind: "suggestions", sensitive: true });
    await ctl.openTopic("self_worth");
    expect(api.topic).toHaveBeenCalledWith("self_worth", { sensitive: true });
  });

  test("a topic picked from the full list is never sensitive-only", async () => {
    const api = fakeApi(general());
    const ctl = new AdviceController(api);
    await ctl.browseAll();
    await ctl.openTopic("anger");
    expect(api.topic).toHaveBeenCalledWith("anger", { sensitive: false });
  });

  test("a slow topic reply never lands on top of a newer crisis note", async () => {
    let release!: () => void;
    const slowTopic = () => new Promise<TopicResult>((r) => (release = () => r(result)));
    const api = fakeApi(general(), { topic: vi.fn(slowTopic) });
    const ctl = new AdviceController(api);
    await ctl.submit("thankful");
    const late = ctl.openTopic("gratitude");
    (api.suggest as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ category: "crisis", note: "note", topics: [] });
    await ctl.submit("something else");
    release();
    await late;
    expect(ctl.state).toEqual({ kind: "crisis", note: "note" });
  });

  test("empty input does nothing; long input is cut to the limit", async () => {
    const api = fakeApi({ category: "none", note: null, topics: [] });
    const ctl = new AdviceController(api);
    await ctl.submit("   ");
    expect(api.suggest).not.toHaveBeenCalled();
    expect(ctl.state.kind).toBe("idle");
    await ctl.submit("x".repeat(MAX_INPUT_CHARS + 50));
    expect((api.suggest as ReturnType<typeof vi.fn>).mock.calls[0][0]).toHaveLength(MAX_INPUT_CHARS);
  });

  test("a failed request shows an error, not stale results", async () => {
    const ctl = new AdviceController(fakeApi(general(), { suggest: vi.fn(async () => { throw new Error("boom"); }) }));
    await ctl.submit("hello");
    expect(ctl.state.kind).toBe("error");
    expect(render(ctl.state).querySelector("[role=alert]")).not.toBeNull();
  });
});

describe("the real search behind the UI", () => {
  const run = async (text: string) => {
    const ctl = new AdviceController(searchApi);
    await ctl.submit(text);
    return { ctl, root: render(ctl.state) };
  };

  test("a crisis message renders the note from the data and nothing else", async () => {
    const { ctl, root } = await run("I want to die");
    expect(ctl.state.kind).toBe("crisis");
    expect(root.textContent).toMatch(/^We are not going to show verses for this one\./);
    expect(root.querySelectorAll("[data-topic], [data-verse], button, a")).toHaveLength(0);
    expect(root.textContent).not.toMatch(/\d{3}/); // no phone numbers
  });

  test("an everyday worry gets topic chips and no verses; verses come after a tap", async () => {
    const { ctl, root } = await run("I got laid off and I'm scared about money");
    expect([...root.querySelectorAll<HTMLElement>("[data-topic]")].map((b) => b.dataset.topic)).toEqual(["money_worry", "fear_anxiety"]);
    expect(root.querySelectorAll("[data-verse]")).toHaveLength(0);
    await ctl.openTopic("money_worry");
    const page = render(ctl.state);
    expect(page.querySelectorAll("[data-verse]").length).toBeGreaterThan(0);
    expect(page.querySelector("[data-empty=gita]")?.textContent).toBe("The Bhagavad Gita has no verse for this topic in the reviewed selection.");
  });

  test("a sensitive message gets gentle labels", async () => {
    const { ctl } = await run("I'm gay and my parents rejected me");
    expect(ctl.state).toMatchObject({ kind: "suggestions", sensitive: true });
    if (ctl.state.kind !== "suggestions") return;
    const labels = ctl.state.topics.map((t) => t.label);
    expect(labels).toContain("You are of worth");
    await ctl.openTopic(ctl.state.topics[0].id);
    expect(ctl.state).toMatchObject({ kind: "topic", result: { sensitive: true } });
  });

  test("nothing matched: every topic that has a verse somewhere", async () => {
    const { ctl } = await run("xyzzy");
    expect(ctl.state.kind).toBe("browse");
    if (ctl.state.kind !== "browse") return;
    expect(ctl.state.topics.length).toBeGreaterThan(20);
    expect(ctl.state.topics.map((t) => t.id)).not.toContain("marriage"); // no verses left in any book
  });
});
