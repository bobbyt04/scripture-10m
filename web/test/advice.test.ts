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
    await ctl.browseAll();
    expect(ctl.state.kind).toBe("crisis");
    expect(api.topic).not.toHaveBeenCalled();
    expect(api.topics).not.toHaveBeenCalled();
  });

  test("topics: the first suggestion's verses straight away, up to four topics as chips", async () => {
    const many = [...topics, { ...topics[0], id: "a" }, { ...topics[0], id: "b" }];
    const api = fakeApi(general(many));
    const ctl = new AdviceController(api);
    await ctl.submit("worried and angry");
    expect(ctl.state).toMatchObject({ kind: "answer", current: { id: "fear_anxiety" }, sensitive: false });
    expect(api.topic).toHaveBeenCalledWith("fear_anxiety", { sensitive: false });
    const root = render(ctl.state);
    expect(root.querySelectorAll("[data-topic]")).toHaveLength(4);
    expect(root.querySelector("[data-topic][aria-pressed=true]")?.getAttribute("data-topic")).toBe("fear_anxiety");
    expect(root.querySelectorAll("[data-verse]").length).toBeGreaterThan(0);
  });

  test("a chip switches the topic; the list of chips stays", async () => {
    const api = fakeApi(general());
    const ctl = new AdviceController(api);
    await ctl.submit("thankful");
    const pending = ctl.openTopic("gratitude");
    expect(ctl.state).toMatchObject({ kind: "answer", current: { id: "gratitude" }, result: null });
    expect(render(ctl.state).querySelectorAll("[data-verse]")).toHaveLength(0);
    await pending;
    expect(ctl.state).toMatchObject({ kind: "answer", current: { id: "gratitude" } });
    expect(render(ctl.state).querySelectorAll("[data-topic]")).toHaveLength(3);
    await ctl.openTopic("not_suggested"); // only suggested topics are reachable from the chips
    expect(api.topic).toHaveBeenCalledTimes(2);
  });

  test("none: one line and the full topic list as chips, no verses", async () => {
    const api = fakeApi({ category: "none", note: null, topics: [] });
    const ctl = new AdviceController(api);
    await ctl.submit("my cat knocked over a plant");
    expect(ctl.state).toEqual({ kind: "browse", reason: "no-match", topics, back: null });
    const root = render(ctl.state);
    expect(root.querySelectorAll("[data-topic]")).toHaveLength(topics.length);
    expect(root.querySelectorAll("[data-verse]")).toHaveLength(0);
    expect(root.textContent).toContain("Nothing matched");
    expect(api.topic).not.toHaveBeenCalled();
  });

  test("topic: the hand-picked verses per book, the rest behind a button, a quiet line for an empty book", async () => {
    const ctl = new AdviceController(fakeApi(general([topics[2]])));
    await ctl.submit("thankful");
    const root = render(ctl.state);
    expect(root.querySelectorAll("[data-verse]")).toHaveLength(5);
    expect(root.querySelectorAll("[data-book=bible] .book__more[hidden] [data-verse]")).toHaveLength(1);
    expect(root.querySelector("[data-book=gita]")).toBeNull();
    expect(root.querySelector("[data-empty=gita]")?.textContent).toBe("The Bhagavad Gita has no verse for this topic in the reviewed selection.");
    expect(root.textContent).toContain("Quran 14:7");
    expect(root.querySelectorAll("[data-topic]")).toHaveLength(0); // one topic: no chips to switch
    root.querySelector<HTMLButtonElement>(".book__toggle")!.click();
    expect(root.querySelectorAll(".book__more[hidden]")).toHaveLength(0);
  });

  test("all topics opens the full list and Back returns to the answer", async () => {
    const ctl = new AdviceController(fakeApi(general()));
    await ctl.submit("thankful");
    const answer = ctl.state;
    await ctl.browseAll();
    expect(ctl.state).toMatchObject({ kind: "browse", reason: "asked", back: answer });
    ctl.back();
    expect(ctl.state).toBe(answer);
  });

  test("a sensitive suggestion opens its topics with sensitive verses only", async () => {
    const api = fakeApi({ category: "sensitive", note: null, topics: [
      { id: "self_worth", label: "You are of worth", blurb: "g" }, { id: "loneliness", label: "Loneliness", blurb: "l" }] });
    const ctl = new AdviceController(api);
    await ctl.submit("anything");
    expect(ctl.state).toMatchObject({ kind: "answer", sensitive: true });
    expect(api.topic).toHaveBeenCalledWith("self_worth", { sensitive: true });
    await ctl.openTopic("loneliness");
    expect(api.topic).toHaveBeenLastCalledWith("loneliness", { sensitive: true });
  });

  test("a topic picked from the full list is never sensitive-only", async () => {
    const api = fakeApi({ category: "sensitive", note: null, topics: [{ id: "self_worth", label: "You are of worth", blurb: "g" }] });
    const ctl = new AdviceController(api);
    await ctl.submit("anything");
    await ctl.browseAll();
    await ctl.openTopic("anger");
    expect(api.topic).toHaveBeenLastCalledWith("anger", { sensitive: false });
    expect(ctl.state).toMatchObject({ kind: "answer", sensitive: false, topics: [] });
  });

  test("a slow topic reply never lands on top of a newer crisis note", async () => {
    let release!: () => void;
    const slowTopic = () => new Promise<TopicResult>((r) => (release = () => r(result)));
    const api = fakeApi(general(), { topic: vi.fn(slowTopic).mockResolvedValueOnce(result) });
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
    expect(root.textContent).toMatch(/^This page will not show verses for what you wrote\./);
    expect(root.querySelectorAll("[data-topic], [data-verse], button, a")).toHaveLength(0);
    expect(root.textContent).not.toMatch(/\d{3}/); // no phone numbers
  });

  test("an everyday worry gets the first topic's verses at once, the others as chips", async () => {
    const { ctl, root } = await run("I got laid off and I'm scared about money");
    expect([...root.querySelectorAll<HTMLElement>("[data-topic]")].map((b) => b.dataset.topic)).toEqual(["money_worry", "fear_anxiety"]);
    expect(ctl.state).toMatchObject({ kind: "answer", current: { id: "money_worry" } });
    expect(root.querySelectorAll("[data-verse]").length).toBeGreaterThan(0);
    expect(root.querySelector("[data-empty=gita]")?.textContent).toBe("The Bhagavad Gita has no verse for this topic in the reviewed selection.");
    await ctl.openTopic("fear_anxiety");
    expect(ctl.state).toMatchObject({ kind: "answer", current: { id: "fear_anxiety" } });
    expect(render(ctl.state).querySelectorAll("[data-verse]").length).toBeGreaterThan(0);
  });

  test("a sensitive message gets gentle labels and sensitive verses", async () => {
    const { ctl } = await run("I'm gay and my parents rejected me");
    expect(ctl.state).toMatchObject({ kind: "answer", sensitive: true, result: { sensitive: true } });
    if (ctl.state.kind !== "answer") return;
    expect(ctl.state.topics.map((t) => t.label)).toContain("You are of worth");
  });

  test("nothing matched: every topic that has a verse somewhere", async () => {
    const { ctl } = await run("xyzzy");
    expect(ctl.state.kind).toBe("browse");
    if (ctl.state.kind !== "browse") return;
    expect(ctl.state.topics.length).toBeGreaterThan(20);
    expect(ctl.state.topics.map((t) => t.id)).not.toContain("marriage"); // no verses left in any book
  });
});
