"""Benchmarks and evals for a trained checkpoint. Writes evals/results/model.json and prints a table.

    python evals/run.py --ckpt out/best.pt

1. Loss: validation loss of the model next to baselines counted from the same training text.
2. Memorisation: how much generated text is copied verbatim from the training text of each book.
3. Style: which book a simple n-gram classifier thinks each generated sample comes from, with
   control texts that show whether the classifier itself is fair.
4. Well-formedness: real words, and verse length compared with real verses.
"""
import argparse
import datetime
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python evals/run.py` finds the scripture package

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from evals.metrics import (NGram, Source, classify, complete_verses, corrupt, markov_samples,  # noqa: E402
                           memorisation, strip_tags, subsample, words)
from scripture.data import BOOKS, TAGS, load_splits  # noqa: E402
from train import load_checkpoint, pick_device  # noqa: E402

MAX_N = 8
BOOTSTRAP = 1000
LN2 = math.log(2)


# ---------------------------------------------------------------- 1. loss benchmark

@torch.no_grad()
def model_logprobs(model, ids, device):
    """Log-probability of each token of ids[1:], each predicted once. Windows of block_size tokens slide
    forward by half a window, so every token after the first window has at least half a window of context."""
    T = model.cfg.block_size
    stride = T // 2
    out, prev_end = [], 0
    for begin in range(0, len(ids) - 1, stride):
        end = min(begin + T, len(ids) - 1)
        logits, _ = model(ids[begin:end].unsqueeze(0).to(device))
        new = end - prev_end  # targets not scored by an earlier window
        targets = ids[end - new + 1 : end + 1].to(device)
        out.append(F.log_softmax(logits[0, -new:].float(), dim=-1).gather(1, targets[:, None])[:, 0].cpu())
        prev_end = end
        if end == len(ids) - 1:
            break
    return torch.cat(out)


def verse_index(ids, tag_ids):
    """For each token of ids[1:], the index of the verse it belongs to (a verse starts at its tag)."""
    return (torch.isin(ids, tag_ids).long().cumsum(0) - 1)[1:]


def summarise(logprobs, verse, dup):
    """Mean loss on all held-out text and without the verses that also occur in training (dup: one bool per
    verse), with a 95% bootstrap interval from resampling whole verses (the same resamples for every model)."""
    nll = -logprobs.double()
    sums = torch.bincount(verse, weights=nll, minlength=len(dup))
    counts = torch.bincount(verse, minlength=len(dup)).double()
    pick = torch.randint(len(dup), (BOOTSTRAP, len(dup)), generator=torch.Generator().manual_seed(0))
    boot = sums[pick].sum(1) / counts[pick].sum(1)
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=torch.double)).tolist()
    nats, dedup = nll.mean().item(), nll[~dup[verse]].mean().item()
    return {"nats": nats, "bpc": nats / LN2, "ci95_nats": [lo, hi], "nats_dedup": dedup, "bpc_dedup": dedup / LN2}


def loss_entry(name, logprobs, verse, dup):
    entry = {b: summarise(logprobs[b], verse[b], dup[b]) for b in BOOKS}
    nats = sum(entry[b]["nats"] for b in BOOKS) / len(BOOKS)
    dedup = sum(entry[b]["nats_dedup"] for b in BOOKS) / len(BOOKS)
    entry["overall"] = {"nats": nats, "bpc": nats / LN2, "nats_dedup": dedup, "bpc_dedup": dedup / LN2}
    return {"name": name, **entry}


def loss_benchmark(model, tok, texts, data, device):
    """Every model scores the same validation tokens; every baseline is fitted on that book's training text."""
    tag_ids = torch.tensor([tok.stoi[t] for t in TAGS])
    verse, dup, duplicates = {}, {}, {}
    for b in BOOKS:
        train_verses = set(strip_tags(texts["train"][b]).split("\n"))
        val_verses = strip_tags(texts["val"][b]).split("\n")[:-1]
        dup[b] = torch.tensor([v in train_verses for v in val_verses])
        verse[b] = verse_index(data["val"][b], tag_ids)
        duplicates[b] = {"val_verses": len(val_verses), "also_in_train": int(dup[b].sum()),
                         "val_tokens": len(data["val"][b]), "tokens_in_duplicates": int(dup[b][verse[b]].sum())}

    logprobs = {"Scripture-10M": {b: model_logprobs(model, data["val"][b], device) for b in BOOKS},
                "uniform": {b: torch.full((len(data["val"][b]) - 1,), -math.log(tok.vocab_size)) for b in BOOKS}}
    for b in BOOKS:
        train, val = ("".join(map(chr, data[s][b].tolist())) for s in ("train", "val"))  # one char per token
        for n, lp in enumerate(NGram(train, MAX_N, tok.vocab_size, only=val).logprobs(val), start=1):
            logprobs.setdefault("unigram" if n == 1 else f"{n}-gram", {})[b] = torch.tensor(lp)
    models = [loss_entry(name, lp, verse, dup) for name, lp in logprobs.items()]
    best = {c: min(models[1:], key=lambda m: m[c]["nats"])["name"] for c in (*BOOKS, "overall")}
    return {"val_tokens": {b: len(data["val"][b]) for b in BOOKS}, "duplicates": duplicates,
            "models": models, "best_baseline": best}


# ---------------------------------------------------------------- generation

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


def generate(model, tok, n, chars, temperature, top_k, seed, device):
    """n samples per book, each starting from the book's tag, all generated in one batch."""
    idx = torch.tensor([tok.stoi[f"<{b}>"] for b in BOOKS for _ in range(n)], device=device)[:, None]
    out = sample_batch(model, idx, chars, temperature, top_k, torch.Generator().manual_seed(seed))
    rows = [tok.decode(r) for r in out.tolist()]
    return {b: rows[i * n : (i + 1) * n] for i, b in enumerate(BOOKS)}


# ---------------------------------------------------------------- 2. memorisation

def chunks(text, size, step, limit):
    """Up to `limit` pieces of `size` characters, starting every `step` characters, spread over the text."""
    starts = range(0, len(text) - size + 1, step)
    keep = [starts[i * len(starts) // limit] for i in range(limit)] if len(starts) > limit else starts
    return [text[i : i + size] for i in keep]


def copy_report(texts_to_check, sources, window):
    """How much of the texts appears verbatim in each book's training text, and in any of them."""
    res = {s: [memorisation(t, sources[s], window) for t in texts_to_check] for s in BOOKS}
    windows = sum(r["windows"] for r in res[BOOKS[0]])
    union = sum(any(f) for k in range(len(texts_to_check)) for f in zip(*(res[s][k]["flags"] for s in BOOKS)))
    src, best = max(((s, r) for s in BOOKS for r in res[s]), key=lambda x: x[1]["longest"])
    return {"texts": len(texts_to_check), "windows": windows,
            "copied_from": {s: sum(r["memorised"] for r in res[s]) / max(1, windows) for s in BOOKS},
            "copied_from_any": union / max(1, windows),
            "longest_run": best["longest"], "longest_run_source": src, "longest_run_text": best["longest_text"],
            "median_longest_run": {s: statistics.median(r["longest"] for r in res[s]) for s in BOOKS}}


def memorisation_report(samples, texts, window, chars, n):
    sources = {b: Source(strip_tags(texts["train"][b])) for b in BOOKS}
    return {b: {"generated": copy_report([strip_tags(s) for s in samples[b]], sources, window),
                "heldout_real": copy_report(chunks(strip_tags(texts["val"][b]), chars, chars, n), sources, window)}
            for b in BOOKS}


# ---------------------------------------------------------------- 3. style

def style_report(samples, texts, vocab_size, chars, n, order, seed):
    """Classify generated samples, real held-out text and control texts with per-book n-gram models,
    each fitted on the same amount of training text (as much as the smallest book has)."""
    rng = random.Random(seed)
    train = {b: strip_tags(texts["train"][b]).split("\n")[:-1] for b in BOOKS}
    size = min(len("\n".join(v)) for v in train.values())
    clf = {b: NGram(subsample(train[b], size, seed), order, vocab_size) for b in BOOKS}
    charset = sorted(set("".join("".join(v) for v in train.values())))

    def row(ts, book=None):
        counts = Counter(classify(t, clf) for t in ts)
        r = {"n": len(ts), "counts": {c: counts[c] for c in BOOKS}}
        if book:
            r["own_book"] = counts[book] / len(ts)
        return r

    real = {b: chunks(strip_tags(texts["val"][b]), chars, chars // 4, 2 * n) for b in BOOKS}
    rows = {
        "generated": {b: row([strip_tags(s) for s in samples[b]], b) for b in BOOKS},
        "real held-out text": {b: row(real[b], b) for b in BOOKS},
        "real held-out, 10% of characters corrupted": {
            b: row([corrupt(t, 0.1, charset, rng) for t in real[b]], b) for b in BOOKS},
        "unigram text": {b: row(markov_samples("\n".join(train[b]), 1, chars, n, rng), b) for b in BOOKS},
        "bigram text": {b: row(markov_samples("\n".join(train[b]), 2, chars, n, rng), b) for b in BOOKS},
    }
    noise = row(["".join(rng.choice(charset) for _ in range(chars)) for _ in range(n)])
    return {"classifier": {"order": order, "train_chars_per_book": size}, "rows": rows, "uniform_noise": noise,
            "verdict": style_verdict(rows, noise)}


def style_verdict(rows, noise):
    """A plain-words reading of the controls."""
    spread = ", ".join(f"{b} {noise['counts'][b]}/{noise['n']}" for b in BOOKS)
    parts = [f"Uniform noise is classified as: {spread}."]
    for name in ("unigram text", "bigram text", "real held-out, 10% of characters corrupted", "real held-out text"):
        own = sum(rows[name][b]["counts"][b] for b in BOOKS)
        total = sum(rows[name][b]["n"] for b in BOOKS)
        parts.append(f"{name}: {own}/{total} sent to its own book.")
    problems = [f"uniform noise goes to {b} {noise['counts'][b]}/{noise['n']} times"
                for b in BOOKS if not noise["n"] / 6 <= noise["counts"][b] <= 2 * noise["n"] / 3]
    problems += [f"{name} imitating {b} is sent to {b} only {rows[name][b]['counts'][b]}/{rows[name][b]['n']} times"
                 for name in ("unigram text", "bigram text", "real held-out, 10% of characters corrupted")
                 for b in BOOKS if rows[name][b]["own_book"] < 0.5]
    if problems:
        parts.append("The classifier is not fully fair: " + "; ".join(problems) +
                     ". Read the generated row against these controls.")
    else:
        parts.append("No strong bias: noise is spread out and text with a book's statistics goes to that book.")
    return " ".join(parts)


# ---------------------------------------------------------------- 4. well-formedness

def wellformed_report(samples, texts):
    vocab = set(words("".join(texts["train"].values())))
    out = {}
    for b in BOOKS:
        ws = [w for s in samples[b] for w in words(s)[:-1]]  # the last word may be cut off
        gen_lens = [len(v) for s in samples[b] for v in complete_verses(s)]
        real_lens = [len(v) for v in complete_verses(strip_tags(texts["val"][b]))]
        out[b] = {"real_word_frac": sum(w in vocab for w in ws) / max(1, len(ws)), "words": len(ws),
                  "mean_verse_chars": statistics.mean(gen_lens) if gen_lens else None,
                  "complete_verses": len(gen_lens), "real_mean_verse_chars": statistics.mean(real_lens)}
    return out


# ---------------------------------------------------------------- main

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

    def stage(name):
        print(f"{name}... ({time.time() - t_start:.0f} s)", flush=True)

    device = pick_device(args.device)
    model, tok, ckpt = load_checkpoint(args.ckpt, device)
    data_tok, texts, data = load_splits(args.data_dir or ckpt["args"]["data_dir"])
    assert data_tok.tokens == tok.tokens, "the data has changed since this checkpoint was trained"

    stage("loss benchmark")
    loss = loss_benchmark(model, tok, texts, data, device)
    stage("generating")
    samples = generate(model, tok, args.samples, args.chars, args.temperature, args.top_k, args.seed, device)
    stage("memorisation")
    memo = memorisation_report(samples, texts, args.window, args.chars, args.samples)
    stage("style")
    style = style_report(samples, texts, tok.vocab_size, args.chars, args.samples, args.classifier_n, args.seed)

    results = {
        "model": "Scripture-10M", "checkpoint": str(args.ckpt), "step": ckpt["step"],
        "config": ckpt["config"], "params": model.num_params(), "train_args": ckpt["args"],
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "git": git_commit(),
        "eval_args": vars(args),
        "notes": {
            "loss": "Mean loss per token on the held-out verses (5% of each book, at least 50 verses). Tokens are "
                    "characters plus one tag token per verse, so bpc = nats / ln 2. 'overall' is the mean of the "
                    "three books, as in training. Every baseline is fitted on the same book's training text. "
                    "The n-gram baselines use interpolated Witten-Bell smoothing; a stronger smoothing such as "
                    "Kneser-Ney would lower them further. 'best_baseline' is the lowest of them per book. "
                    "best.pt was chosen by this same validation loss; there is no separate test split. "
                    "'ci95_nats' is a 95% bootstrap interval from resampling held-out verses. Some held-out "
                    "verses also occur word for word in training (see 'duplicates'); the '_dedup' numbers "
                    "leave those verses out.",
            "memorisation": f"Share of {args.window}-character windows of generated text (tags removed) found "
                            "verbatim in each book's training text, and in any of them. 'heldout_real' is the same "
                            "measure on real held-out verses, for reference: scripture repeats itself, so it is not "
                            "zero.",
            "style": f"{args.classifier_n}-gram classifiers, one per book, each fitted on "
                     f"{style['classifier']['train_chars_per_book']:,} characters of training verses (all of the "
                     "smallest book, a random subset of the others) without tags; highest likelihood wins. The "
                     "control rows test the classifier itself: uniform noise should spread over the books, and "
                     "text with one book's statistics should go to that book. Real held-out chunks overlap by "
                     "three quarters, so they are not independent. " + style["verdict"],
            "wellformed": "real_word_frac: generated words that appear in the training text of any book. Verse "
                          f"lengths count only verses finished within {args.chars} characters, so very long "
                          "verses are under-represented.",
        },
        "loss": loss, "memorisation": memo, "style": style, "wellformed": wellformed_report(samples, texts),
        "samples": {b: samples[b][:3] for b in BOOKS},
        "runtime_s": round(time.time() - t_start, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print_table(results)
    print(f"wrote {args.out} in {results['runtime_s']:.0f} s")
    return results


def print_table(r):
    cols = (*BOOKS, "overall")
    loss = r["loss"]
    print(f"\nScripture-10M, step {r['step']}, {r['params']:,} parameters")
    for key, title in (("nats", "all held-out text"), ("nats_dedup", "without verses also in training")):
        print(f"\nValidation loss, nats per token (bits per character), {title}. * = best baseline")
        print(f"{'':15}" + "".join(f"{c:>17}" for c in cols))
        for m in loss["models"]:
            print(f"{m['name']:15}" + "".join(
                f"{m[c][key]:>9.3f} ({m[c][key] / LN2:.3f})" + ("*" if loss["best_baseline"][c] == m["name"] else " ")
                for c in cols))
    model = loss["models"][0]
    print("Scripture-10M 95% interval: " + ", ".join(
        f"{b} {model[b]['ci95_nats'][0]:.3f}-{model[b]['ci95_nats'][1]:.3f}" for b in BOOKS))
    print("held-out verses also in training: " + ", ".join(
        f"{b} {d['also_in_train']}/{d['val_verses']}" for b, d in loss["duplicates"].items()))

    print(f"\nCopied verbatim ({r['eval_args']['window']}-character windows): generated under each tag, then real")
    print(f"{'':15}" + "".join(f"{'from ' + s:>12}" for s in BOOKS) + f"{'any':>8}   longest run")
    for b in BOOKS:
        for kind in ("generated", "heldout_real"):
            m = r["memorisation"][b][kind]
            label = b if kind == "generated" else "  real " + b
            print(f"{label:15}" + "".join(f"{m['copied_from'][s]:>12.1%}" for s in BOOKS) +
                  f"{m['copied_from_any']:>8.1%}   {m['longest_run']} ({m['longest_run_source']})")

    style = r["style"]
    print("\nStyle classifier: texts sent to their own book (columns: the book the text comes from or imitates)")
    print(f"{'':44}" + "".join(f"{b:>10}" for b in BOOKS))
    for name, row in style["rows"].items():
        print(f"{name:44}" + "".join(f"{row[b]['counts'][b]:>6}/{row[b]['n']:<3}" for b in BOOKS))
    noise = style["uniform_noise"]
    print(f"{'uniform noise (classified as)':44}" + "".join(f"{noise['counts'][b]:>6}/{noise['n']:<3}" for b in BOOKS))
    print("generated confusion (rows: tag, columns: guess): " + "; ".join(
        f"{b}: " + " ".join(f"{style['rows']['generated'][b]['counts'][c]}" for c in BOOKS) for b in BOOKS))
    print(style["verdict"])

    print(f"\n{'':15}" + "".join(f"{b:>12}" for b in BOOKS))
    wf = r["wellformed"]
    for name, f in (("real words", lambda b: f"{wf[b]['real_word_frac']:.1%}"),
                    ("verse chars", lambda b: f"{wf[b]['mean_verse_chars'] or 0:.0f}"),
                    ("  real", lambda b: f"{wf[b]['real_mean_verse_chars']:.0f}")):
        print(f"{name:15}" + "".join(f"{f(b):>12}" for b in BOOKS))


if __name__ == "__main__":
    main()
