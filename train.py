"""Train Scripture-10M from scratch.

    python train.py              # full run, writes out/ckpt.pt, out/best.pt and out/losses.jsonl
    python train.py --resume     # continue from out/ckpt.pt
"""
import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from scripture.data import BOOKS, DATA_DIR, get_batch, load_splits
from scripture.model import ModelConfig, ScriptureLM
from scripture.tokenizer import CharTokenizer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # model
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--block_size", type=int, default=256)
    # optimisation
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate, reached after warmup")
    p.add_argument("--min_lr", type=float, default=1e-4, help="learning rate at the end of the cosine decay")
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--max_iters", type=int, default=12000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # evaluation and logging
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=40, help="batches per book per split at each evaluation")
    p.add_argument("--log_interval", type=int, default=50)
    # run
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sampling", choices=["verse", "book"], default="verse",
                   help="verse: every verse seen equally often (books weighted by size); book: each book equally often")
    p.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--out_dir", default="out")
    p.add_argument("--resume", action="store_true", help="continue from out_dir/ckpt.pt with its saved settings")
    return p.parse_args(argv)


def pick_device(name="auto"):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_lr(step, args):
    """Linear warmup to args.lr, then cosine decay to args.min_lr at args.max_iters."""
    if step < args.warmup_iters:
        return args.lr * (step + 1) / args.warmup_iters
    progress = min(1.0, (step - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters))
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)


def make_optimizer(model, args):
    # weight decay only on matrices (linear weights, embeddings), not on biases or LayerNorm gains
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": args.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2))


@torch.no_grad()
def estimate_loss(model, data, args, device):
    """Mean loss over eval_iters batches per book per split. 'overall' is the mean of the three books.

    The batches come from a fixed seed, so every evaluation scores the same windows.
    """
    model.eval()
    out = {}
    for split, split_data in data.items():
        gen = torch.Generator().manual_seed(args.seed)
        losses = {}
        for book in BOOKS:
            total = 0.0
            for _ in range(args.eval_iters):
                x, y, _ = get_batch(split_data, args.batch_size, args.block_size, gen, book=book)
                _, loss = model(x.to(device), y.to(device))
                total += loss.item()
            losses[book] = total / args.eval_iters
        losses["overall"] = sum(losses[b] for b in BOOKS) / len(BOOKS)
        out[split] = losses
    model.train()
    return out


def load_checkpoint(path, device):
    """The model (in eval mode), tokenizer and raw checkpoint dict saved by train()."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ScriptureLM(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), CharTokenizer(**ckpt["tokenizer"]), ckpt


def save_atomic(obj, path):
    """Write to a temporary file first, so a run killed mid-save never leaves a broken checkpoint."""
    tmp = Path(path).with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = None
    if args.resume:
        ckpt = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
        device_choice, out = args.device, args.out_dir
        args = argparse.Namespace(**ckpt["args"])  # same settings, so the LR schedule carries on unchanged
        args.device, args.out_dir, args.resume = device_choice, out, True
        args.sampling = ckpt["args"].get("sampling", "book")  # checkpoints from before --sampling sampled by book
    device = pick_device(args.device)

    torch.manual_seed(args.seed)
    tok, _, data = load_splits(args.data_dir)
    if ckpt is not None:
        assert CharTokenizer(**ckpt["tokenizer"]).tokens == tok.tokens, "data changed since this checkpoint"
    cfg = ModelConfig(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout)
    model = ScriptureLM(cfg).to(device)
    optimizer = make_optimizer(model, args)
    start_step, best_val = 0, float("inf")
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"], ckpt["best_val"]
        print(f"resuming from step {start_step} (best val {best_val:.4f})")
    print(f"{model.num_params():,} parameters on {device}, {args.max_iters} steps")

    log_path = out_dir / "losses.jsonl"
    if args.resume:  # the log up to the checkpoint plus the checkpoint's own line, whatever was written after
        records = []
        for line in log_path.read_text().splitlines() if log_path.exists() else []:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut short when the process died while writing it
                pass
        kept = [r for r in records if r["step"] < start_step] + [ckpt["record"]]
        log_path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    else:
        log_path.write_text("")
    gen = torch.Generator()
    book_weights = None
    if args.sampling == "verse":
        book_weights = torch.tensor([float(len(data["train"][b])) for b in BOOKS])
    t_start = time.time() - (ckpt["record"]["elapsed_s"] if args.resume else 0)  # elapsed_s counts all runs
    t_last, steps_since, eval_s = time.time(), 0, 0.0  # for the ms/step and ETA printout
    for step in range(start_step, args.max_iters + 1):
        lr = get_lr(step, args)
        is_eval = step % args.eval_interval == 0 or step == args.max_iters
        if is_eval and not (args.resume and step == start_step):  # a resumed step was already evaluated
            t_eval = time.time()
            losses = estimate_loss(model, data, args, device)
            record = {"step": step, "lr": lr, "train": losses["train"], "val": losses["val"],
                      "elapsed_s": round(time.time() - t_start, 1)}
            print(f"step {step}: " + " | ".join(
                f"{s} " + " ".join(f"{k} {v:.4f}" for k, v in losses[s].items()) for s in ("train", "val")))
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": asdict(cfg),
                    "tokenizer": {"chars": tok.chars, "special_tokens": tok.special_tokens}, "step": step,
                    "best_val": min(best_val, losses["val"]["overall"]), "args": vars(args), "record": record}
            save_atomic(ckpt, out_dir / "ckpt.pt")
            if losses["val"]["overall"] < best_val:
                best_val = losses["val"]["overall"]
                save_atomic(ckpt, out_dir / "best.pt")
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            eval_s = time.time() - t_eval
            t_last, steps_since = time.time(), 0
        if step == args.max_iters:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr
        gen.manual_seed(args.seed + step)  # the batch depends only on the step, so a resumed run sees the same data
        x, y, _ = get_batch(data["train"], args.batch_size, args.block_size, gen, weights=book_weights)
        _, loss = model(x.to(device), y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        steps_since += 1

        if (step + 1) % args.log_interval == 0:
            loss_value = loss.item()  # waits for the device, so the timing below is real
            step_s = (time.time() - t_last) / steps_since
            left = args.max_iters - step - 1
            evals_left = left // args.eval_interval + 1
            eta = left * step_s + evals_left * eval_s
            print(f"step {step + 1}: loss {loss_value:.4f}, lr {lr:.2e}, {1000 * step_s:.0f} ms/step, "
                  f"ETA {eta / 60:.0f} min")
            t_last, steps_since = time.time(), 0
    return model


if __name__ == "__main__":
    train(parse_args())
