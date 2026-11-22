from scripture.tokenizer import CharTokenizer

TAGS = ["<gita>", "<quran>", "<bible>"]
TEXT = "<gita>Arjuna said.\n<quran>In the name of God.\n<bible>In the beginning.\n"


def test_round_trip():
    tok = CharTokenizer.from_text(TEXT, TAGS)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_special_tokens_are_single_ids():
    tok = CharTokenizer.from_text(TEXT, TAGS)
    for tag in TAGS:
        ids = tok.encode(tag)
        assert len(ids) == 1
        assert tok.decode(ids) == tag
    assert tok.encode("<bible>In") == [tok.stoi["<bible>"], tok.stoi["I"], tok.stoi["n"]]
    assert tok.vocab_size == len(set(TEXT.replace("<gita>", "").replace("<quran>", "").replace("<bible>", ""))) + 3


def test_unknown_chars_dropped():
    tok = CharTokenizer.from_text("abc")
    assert tok.encode("aXbZc") == tok.encode("abc")
    assert tok.decode(tok.encode("aXbZc")) == "abc"


def test_save_load(tmp_path):
    tok = CharTokenizer.from_text(TEXT, TAGS)
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.tokens == tok.tokens
    assert loaded.encode(TEXT) == tok.encode(TEXT)
