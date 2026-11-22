"""Character-level tokenizer: every unique character in the training text is one token.

Special tokens (such as the book tags "<gita>", "<quran>", "<bible>") are whole strings
that each get a single id of their own.
"""
import json


class CharTokenizer:
    def __init__(self, chars, special_tokens=()):
        self.special_tokens = list(special_tokens)
        self.chars = list(chars)
        self.tokens = self.special_tokens + self.chars  # id -> string
        self.stoi = {t: i for i, t in enumerate(self.tokens)}

    @classmethod
    def from_text(cls, text, special_tokens=()):
        for tok in special_tokens:
            text = text.replace(tok, "")  # special tokens are not split into characters
        return cls(sorted(set(text)), special_tokens)

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, s):
        ids, i = [], 0
        while i < len(s):
            tok = next((t for t in self.special_tokens if s.startswith(t, i)), s[i])
            if tok in self.stoi:  # characters never seen in training are dropped
                ids.append(self.stoi[tok])
            i += len(tok)
        return ids

    def decode(self, ids):
        return "".join(self.tokens[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars, "special_tokens": self.special_tokens}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["chars"], d["special_tokens"])
