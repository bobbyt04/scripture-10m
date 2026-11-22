"""Benchmarks and evals for a trained checkpoint. Writes evals/results/model.json and prints a table.

    python evals/run.py --ckpt out/best.pt

1. Loss: validation loss of the model next to baselines counted from the same training text.
2. Memorisation: how much generated text is copied verbatim from the training text.
3. Style: which book a simple n-gram classifier thinks each generated sample comes from.
4. Well-formedness: real words, and verse length compared with real verses.
"""
import argparse
import datetime
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python evals/run.py` finds the scripture package

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from evals.metrics import NGram, classify, complete_verses, memorisation, strip_tags, words  # noqa: E402
from scripture.data import BOOKS, load_splits  # noqa: E402
from train import load_checkpoint, pick_device  # noqa: E402

MAX_N = 6


@torch.no_grad()
def model_nll(model, ids, device):
    """Mean loss in nats over ids[1:], each token predicted once, with at least half a window of context
    (except at the start). Windows of block_size tokens slide forward by half a window."""
    T = model.cfg.block_size
    stride = T // 2
    total, prev_end = 0.0, 0
    for begin in range(0, len(ids) - 1, stride):
        end = min(begin + T, len(ids) - 1)
        x = ids[begin:end].unsqueeze(0).to(device)
        y = ids[begin + 1 : end + 1].to(device)
        logits, _ = model(x)
        new = end - prev_end  # targets not scored by an earlier window
        total += F.cross_entropy(logits[0, -new:], y[-new:], reduction="sum").item()
        prev_end = end
        if end == len(ids) - 1:
            break
    return total / (len(ids) - 1)


@torch.no_grad()
def sample_batch(model, idx, n_tokens, temperature, top_k, gen):
    """Like ScriptureLM.generate, but for a batch of rows, sampling on the CPU so a seed gives the same text."""
    for _ in range(n_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size :])
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1).cpu(), 1, generator=gen)
        idx = torch.cat((idx, nxt.to(idx.device)), dim=1)
    return idx


def loss_entry(name, nats):
    entry = {b: {"nats": nats[b], "bpc": nats[b] / math.log(2)} for b in BOOKS}
    overall = sum(nats[b] for b in BOOKS) / len(BOOKS)
    entry["overall"] = {"nats": overall, "bpc": overall / math.log(2)}
    return {"name": name, **entry}


def chunks(text, size, limit):
    return [text[i : i + size] for i in range(0, len(text) - size + 1, size)][:limit]


def git_commit():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="out/best.pt")
    p.add_argument("--out", default=str(ROOT / "evals" / "results" / "model.json"))
    p.add_argument("--samples", type=int, default=50, help="generated samples per book")
    p.add_argument("--chars", type=int, default=400, help="tokens per sample (one token is one character)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--window", type=int, default=50, help="window length for the memorisation check")
    p.add_argument("--classifier_n", type=int, default=5, help="n-gram order of the style classifier")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--data_dir", default=None, help="default: the data_dir the checkpoint was trained on")
    args = p.parse_args(argv)
    t_start = time.time()

    device = pick_device(args.device)
    model, tok, ckpt = load_checkpoint(args.ckpt, device)
    data_tok, texts, data = load_splits(args.data_dir or ckpt["args"]["data_dir"])
    assert data_tok.tokens == tok.tokens, "the data has changed since this checkpoint was trained"
    sym = {s: {b: "".join(map(chr, data[s][b].tolist())) for b in BOOKS} for s in data}  # one char per token

    # 1. loss benchmark: every model scores the same validation tokens, after training on the same text
    print(f"loss benchmark... ({time.time() - t_start:.0f} s)")
    ngrams = {b: NGram(sym["train"][b], MAX_N, tok.vocab_size) for b in BOOKS}
    losses = [loss_entry("Scripture-10M", {b: model_nll(model, data["val"][b], device) for b in BOOKS}),
              loss_entry("uniform", {b: math.log(tok.vocab_size) for b in BOOKS})]
    for n in range(1, MAX_N + 1):
        name = "unigram" if n == 1 else f"{n}-gram"
        losses.append(loss_entry(name, {b: ngrams[b].nll(sym["val"][b], order=n) for b in BOOKS}))

    # generate samples, starting from each book's tag
    print(f"generating... ({time.time() - t_start:.0f} s)")
    samples = {}
    for i, book in enumerate(BOOKS):
        gen = torch.Generator().manual_seed(args.seed + i)
        idx = torch.full((args.samples, 1), tok.stoi[f"<{book}>"], device=device)
        out = sample_batch(model, idx, args.chars, args.temperature, args.top_k, gen)
        samples[book] = [tok.decode(row) for row in out.tolist()]

    # 2. memorisation, next to the same measure on real held-out text of the same length
    print(f"memorisation... ({time.time() - t_start:.0f} s)")
    def copied(texts_to_check, train_text):
        rs = [memorisation(t, train_text, args.window) for t in texts_to_check]
        best = max(rs, key=lambda r: r["longest"])
        return {"window_frac": sum(r["memorised"] for r in rs) / max(1, sum(r["windows"] for r in rs)),
                "longest_run": best["longest"], "median_longest_run": statistics.median(r["longest"] for r in rs),
                "longest_run_text": best["longest_text"]}
    memo = {b: {"generated": copied(samples[b], texts["train"][b]),
                "heldout_real": copied(chunks(texts["val"][b], args.chars, args.samples), texts["train"][b])}
            for b in BOOKS}

    # 3. style: n-gram classifiers trained on each book's training verses (tags removed)
    print(f"style... ({time.time() - t_start:.0f} s)")
    clf = {b: NGram(strip_tags(texts["train"][b]), args.classifier_n, tok.vocab_size) for b in BOOKS}
    confusion = {b: {c: 0 for c in BOOKS} for b in BOOKS}
    for b in BOOKS:
        for s in samples[b]:
            confusion[b][classify(strip_tags(s), clf)] += 1
    ceiling = {}
    for b in BOOKS:
        real = chunks(strip_tags(texts["val"][b]), args.chars, args.samples)
        ceiling[b] = {"accuracy": sum(classify(t, clf) == b for t in real) / len(real), "chunks": len(real)}
    style = {"confusion": confusion,
             "accuracy": {b: confusion[b][b] / args.samples for b in BOOKS},
             "classifier_ceiling": ceiling}

    # 4. well-formedness
    vocab = set(words("".join(texts["train"].values())))
    wellformed = {}
    for b in BOOKS:
        ws = [w for s in samples[b] for w in words(s)[:-1]]  # the last word may be cut off
        gen_lens = [len(v) for s in samples[b] for v in complete_verses(s)]
        real_lens = [len(v) for v in complete_verses(strip_tags(texts["val"][b]))]
        wellformed[b] = {"real_word_frac": sum(w in vocab for w in ws) / max(1, len(ws)), "words": len(ws),
                         "mean_verse_chars": statistics.mean(gen_lens) if gen_lens else None,
                         "complete_verses": len(gen_lens), "real_mean_verse_chars": statistics.mean(real_lens)}

    results = {
        "model": "Scripture-10M", "checkpoint": str(args.ckpt), "step": ckpt["step"],
        "config": ckpt["config"], "params": model.num_params(), "train_args": ckpt["args"],
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "git": git_commit(),
        "eval_args": vars(args),
        "notes": {
            "loss": "Mean loss per token on the held-out verses (5% per book). Tokens are characters, plus one tag "
                    "token per verse, so bpc = nats / ln 2. 'overall' is the mean of the three books, as in "
                    "training. Every baseline is fitted on the same training text, per book.",
            "memorisation": f"Share of {args.window}-character windows of generated text found verbatim in that "
                            "book's training text. 'heldout_real' is the same measure on real held-out verses, "
                            "for reference: scripture repeats itself, so it is not zero.",
            "style": f"{args.classifier_n}-gram classifiers, one per book, trained on training verses without tags; "
                     "highest likelihood wins. The ceiling is its accuracy on real held-out text of the same length.",
            "wellformed": "real_word_frac: generated words that appear in the training text of any book. Verse "
                          f"lengths count only verses finished within {args.chars} characters, so very long "
                          "verses are under-represented.",
        },
        "loss": {"val_tokens": {b: len(data["val"][b]) for b in BOOKS}, "models": losses},
        "memorisation": memo, "style": style, "wellformed": wellformed,
        "samples": {b: samples[b][:3] for b in BOOKS},
        "runtime_s": round(time.time() - t_start, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print_table(results)
    print(f"wrote {args.out} in {results['runtime_s']:.0f} s")


def print_table(r):
    cols = (*BOOKS, "overall")
    print(f"\nScripture-10M, step {r['step']}, {r['params']:,} parameters")
    print("\nValidation loss, nats per token (bits per character)")
    print(f"{'':15}" + "".join(f"{c:>17}" for c in cols))
    for m in r["loss"]["models"]:
        print(f"{m['name']:15}" + "".join(f"{m[c]['nats']:>9.3f} ({m[c]['bpc']:.3f})" for c in cols))
    print(f"\n{'':15}" + "".join(f"{b:>12}" for b in BOOKS))
    rows = [
        (f"copied {r['eval_args']['window']}-ch", lambda b: f"{r['memorisation'][b]['generated']['window_frac']:.1%}"),
        ("  real ref", lambda b: f"{r['memorisation'][b]['heldout_real']['window_frac']:.1%}"),
        ("longest copy", lambda b: str(r["memorisation"][b]["generated"]["longest_run"])),
        ("  real ref", lambda b: str(r["memorisation"][b]["heldout_real"]["longest_run"])),
        ("style acc", lambda b: f"{r['style']['accuracy'][b]:.0%}"),
        ("  ceiling", lambda b: f"{r['style']['classifier_ceiling'][b]['accuracy']:.0%}"),
        ("real words", lambda b: f"{r['wellformed'][b]['real_word_frac']:.1%}"),
        ("verse chars", lambda b: f"{r['wellformed'][b]['mean_verse_chars'] or 0:.0f}"),
        ("  real", lambda b: f"{r['wellformed'][b]['real_mean_verse_chars']:.0f}"),
    ]
    for name, f in rows:
        print(f"{name:15}" + "".join(f"{f(b):>12}" for b in BOOKS))
    print("\nStyle confusion (rows: tag generated under, columns: classifier's guess)")
    print(f"{'':15}" + "".join(f"{b:>12}" for b in BOOKS))
    for b in BOOKS:
        print(f"{b:15}" + "".join(f"{r['style']['confusion'][b][c]:>12}" for c in BOOKS))


if __name__ == "__main__":
    main()
