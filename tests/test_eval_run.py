"""evals/run.py and sample.py: the loss benchmark, the reports, and both scripts end to end on a tiny checkpoint."""
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

import sample
import train
from evals import run
from evals.metrics import NGram, Source, memorisation, strip_tags
from scripture.data import BOOKS, load_splits
from tests.test_train import TINY

V, T = 7, 8


class Bigram(torch.nn.Module):
    """Logits depend only on the current token, so the exact loss of any text is easy to compute."""

    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(block_size=T)
        self.table = torch.nn.Embedding(V, V)

    def forward(self, idx):
        assert idx.shape[1] <= self.cfg.block_size
        return self.table(idx), None


class NeedsContext(torch.nn.Module):
    """Knows the next token (current + 1) only with at least T // 2 tokens of context in the window."""

    cfg = SimpleNamespace(block_size=T)

    def forward(self, idx):
        logits = 30.0 * F.one_hot((idx + 1) % V, V).float()
        logits[:, : T // 2] = 0.0  # too little context: a uniform guess
        return logits, None


@pytest.mark.parametrize("length", [5, 9, 50, 51])
def test_model_logprobs_scores_every_token_once(length):
    torch.manual_seed(0)
    model = Bigram()
    ids = torch.randint(0, V, (length,))
    lp = run.model_logprobs(model, ids, "cpu")
    assert lp.shape == (length - 1,)
    assert torch.allclose(lp, -F.cross_entropy(model.table(ids[:-1]), ids[1:], reduction="none"), atol=1e-6)


def test_model_logprobs_gives_half_a_window_of_context():
    ids = torch.arange(50) % V
    lp = run.model_logprobs(NeedsContext(), ids, "cpu")
    # only the first window's first T // 2 positions lack context
    assert -lp.sum().item() == pytest.approx(T // 2 * math.log(V), rel=1e-3)


def test_summarise_dedup_and_interval():
    tags = torch.tensor([0])
    ids = torch.tensor([0, 1, 1, 0, 2, 2, 0, 1, 1])  # three verses of three tokens, each starting with tag 0
    verse = run.verse_index(ids, tags)
    assert verse.tolist() == [0, 0, 1, 1, 1, 2, 2, 2]
    lp = torch.tensor([-1.0, -1.0, -4.0, -4.0, -4.0, -1.0, -1.0, -1.0])
    s = run.summarise(lp, verse, torch.tensor([False, True, False]))
    assert s["nats"] == pytest.approx(17 / 8)
    assert s["nats_dedup"] == pytest.approx(1.0)  # the duplicated middle verse left out
    assert s["bpc"] == pytest.approx(s["nats"] / math.log(2))
    assert s["bpc_dedup"] == pytest.approx(1.0 / math.log(2))
    lo, hi = s["ci95_nats"]
    assert 1.0 <= lo < s["nats"] < hi <= 4.0
    flat = run.summarise(torch.full((8,), -2.0), verse, torch.tensor([False, False, False]))
    assert flat["ci95_nats"] == pytest.approx([2.0, 2.0])


def test_loss_entry_overall_and_bits():
    verse = torch.zeros(4, dtype=torch.long)
    dup = torch.tensor([False])
    lps = {"gita": torch.full((4,), -1.0), "quran": torch.full((4,), -2.0), "bible": torch.full((4,), -3.0)}
    e = run.loss_entry("x", lps, {b: verse for b in BOOKS}, {b: dup for b in BOOKS})
    assert e["quran"]["bpc"] == pytest.approx(2 / math.log(2))
    assert e["overall"]["nats"] == pytest.approx(2.0)
    assert e["overall"]["bpc"] == pytest.approx(2 / math.log(2))


def test_chunks_spread_and_limit():
    text = "".join(chr(65 + i % 26) for i in range(1000))
    assert run.chunks(text, 100, 100, 50) == [text[i : i + 100] for i in range(0, 1000, 100)]
    picked = run.chunks(text, 100, 10, 5)
    assert len(picked) == 5 and picked[0] == text[:100] and picked[-1] != text[40:140]


def test_copy_report_finds_the_source_book():
    words = {"gita": "arjuna yoga self ", "quran": "allah mercy day ", "bible": "moses king unto "}
    texts = {b: (w * 40) for b, w in words.items()}
    sources = {b: Source(t) for b, t in texts.items()}
    half = texts["quran"][:60] + texts["bible"][:60]
    r = run.copy_report([texts["gita"][:120], half], sources, 20)
    assert r["windows"] == 2 * (120 - 20 + 1)
    assert r["copied_from"]["gita"] == pytest.approx(101 / 202)
    assert 0 < r["copied_from"]["quran"] < 0.5 and 0 < r["copied_from"]["bible"] < 0.5
    # the union counts the quran and the bible halves, but not most windows spanning the join
    assert (101 + 41 + 41) / 202 <= r["copied_from_any"] < 1
    assert r["copied_from_any"] == pytest.approx(sum(r["copied_from"].values()))  # the sources share no window
    assert (r["longest_run"], r["longest_run_source"]) == (120, "gita")


def test_style_verdict_flags_bias():
    def rows(own):
        return {name: {b: {"n": 10, "counts": {c: (own if c == b else 0) for c in BOOKS}, "own_book": own / 10}
                       for b in BOOKS}
                for name in ("unigram text", "bigram text", "real held-out, 10% of characters corrupted",
                             "real held-out text")}
    fair_noise = {"n": 30, "counts": {"gita": 10, "quran": 11, "bible": 9}}
    assert "No strong bias" in run.style_verdict(rows(9), fair_noise)
    biased = run.style_verdict(rows(9), {"n": 30, "counts": {"gita": 21, "quran": 5, "bible": 4}})
    assert "not fully fair" in biased and "noise goes to gita 21/30" in biased and "bible 4/30 times" in biased
    weak = run.style_verdict(rows(4), fair_noise)
    assert "not fully fair" in weak and "unigram text imitating quran is sent to quran only 4/10" in weak


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory, tiny_data_dir):
    out = tmp_path_factory.mktemp("ckpt")
    train.train(train.parse_args(TINY + ["--out_dir", str(out), "--data_dir", str(tiny_data_dir)]))
    return out / "best.pt"


def test_sample_script(tiny_ckpt, capsys):
    sample.main(["--ckpt", str(tiny_ckpt), "--book", "quran", "--prompt", "allah", "--tokens", "30",
                 "--device", "cpu"])
    text = capsys.readouterr().out
    assert text.startswith("<quran>allah") and len(text.rstrip("\n")) >= len("<quran>allah") + 25


def test_evals_end_to_end(tiny_ckpt, tiny_data_dir, tmp_path):
    out = tmp_path / "model.json"
    r = run.main(["--ckpt", str(tiny_ckpt), "--out", str(out), "--samples", "4", "--chars", "60",
                  "--device", "cpu", "--data_dir", str(tiny_data_dir), "--window", "20"])
    assert json.loads(out.read_text()) == json.loads(json.dumps(r))
    assert r["step"] > 0 and r["params"] > 0 and r["git"]["commit"]
    tok, texts, data = load_splits(tiny_data_dir)

    # baselines are fitted on training text and score the validation text
    loss = r["loss"]
    names = [m["name"] for m in loss["models"]]
    assert names == ["Scripture-10M", "uniform", "unigram"] + [f"{n}-gram" for n in range(2, 9)]
    sym = {s: {b: "".join(map(chr, data[s][b].tolist())) for b in BOOKS} for s in ("train", "val")}
    three = loss["models"][names.index("3-gram")]
    for b in BOOKS:
        expected = NGram(sym["train"][b], 3, tok.vocab_size).nll(sym["val"][b])
        assert three[b]["nats"] == pytest.approx(expected)
        assert three[b]["bpc"] == pytest.approx(expected / math.log(2))
        assert loss["val_tokens"][b] == len(data["val"][b])
        best = min(loss["models"][1:], key=lambda m: m[b]["nats"])["name"]
        assert loss["best_baseline"][b] == best
    assert loss["models"][1]["gita"]["nats"] == pytest.approx(math.log(tok.vocab_size))

    # memorisation is measured against the training text of each book
    memo = r["memorisation"]["quran"]["heldout_real"]
    real = run.chunks(strip_tags(texts["val"]["quran"]), 60, 60, 4)
    rs = [memorisation(t, strip_tags(texts["train"]["quran"]), 20) for t in real]
    assert memo["copied_from"]["quran"] == pytest.approx(sum(x["memorised"] for x in rs) / sum(x["windows"] for x in rs))
    assert memo["copied_from"]["quran"] < 1.0

    # the style classifier is fitted on equal amounts of training text
    style = r["style"]
    smallest = min(len(strip_tags(texts["train"][b]).rstrip("\n")) for b in BOOKS)
    assert style["classifier"]["train_chars_per_book"] == smallest
    for name, rows in style["rows"].items():
        for b in BOOKS:
            assert sum(rows[b]["counts"].values()) == rows[b]["n"] > 0, name
    assert style["rows"]["generated"]["gita"]["n"] == 4
    assert style["verdict"] in r["notes"]["style"]
    assert set(r["wellformed"]) == set(BOOKS)
