// The same character tokenizer as scripture/tokenizer.py: one id per character, plus whole-string
// special tokens (the book tags) matched longest first. Characters the model never saw are dropped.

export class CharTokenizer {
  readonly tokens: readonly string[];
  private readonly stoi: Map<string, number>;
  private readonly specials: string[];

  constructor(tokens: readonly string[]) {
    this.tokens = tokens;
    this.stoi = new Map(tokens.map((t, i) => [t, i]));
    // In the Python tokenizer the special tokens are the multi-character ones.
    this.specials = tokens.filter((t) => [...t].length > 1).sort((a, b) => b.length - a.length);
  }

  get vocabSize(): number {
    return this.tokens.length;
  }

  encode(s: string): number[] {
    const ids: number[] = [];
    let i = 0;
    while (i < s.length) {
      let tok = this.specials.find((t) => s.startsWith(t, i));
      if (tok === undefined) {
        const cp = s.codePointAt(i)!;
        tok = String.fromCodePoint(cp); // a Python character is a code point
      }
      const id = this.stoi.get(tok);
      if (id !== undefined) ids.push(id);
      i += tok.length;
    }
    return ids;
  }

  decode(ids: readonly number[]): string {
    return ids.map((i) => this.tokens[i]).join("");
  }

  id(token: string): number {
    const id = this.stoi.get(token);
    if (id === undefined) throw new Error(`unknown token ${JSON.stringify(token)}`);
    return id;
  }
}
