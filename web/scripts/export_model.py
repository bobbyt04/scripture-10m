"""Export a trained checkpoint for the website, and the fixtures the TypeScript tests check against.

    python web/scripts/export_model.py --ckpt out/best.pt

Writes, from the repository root:

- web/public/model/scripture-10m.<format>.<hash>.bin: the weights in one compact binary (format below).
- web/src/model/manifest.json: file name, size, SHA-256 and the measured validation loss of every format tried.
- web/src/model/vocab.json: the tokenizer's tokens, id order.
- web/test/fixtures/parity.json: PyTorch logits and greedy generations for fixed inputs, computed from exactly
  the weights stored in the binary (dequantised to float32), so the TypeScript forward pass must match them.
- web/test/fixtures/tokenizer.json: Python tokenizer ids for a set of test strings.

Formats tried: float16 (every tensor as IEEE half) and int8 (each weight matrix and embedding quantised
symmetrically with one float32 scale per tensor; LayerNorm parameters and biases stay float32). For each, the
weights are dequantised back into the PyTorch model and the validation loss is measured with the eval
harness's own scorer (evals/run.py model_logprobs) on the same held-out verses as evals/results/model.json.

Binary layout (little-endian):
    bytes 0-3   b"S10M"
    bytes 4-7   uint32 version (1)
    bytes 8-11  uint32 n = length of the JSON header in bytes
    bytes 12..  JSON header, padded with spaces so the data starts at a multiple of 16
    then        the tensors, each starting at a multiple of 16 bytes from the start of the data section

The header holds the model config, the format and, per tensor: name, shape, dtype (float32, float16 or int8),
byte offset within the data section and, for int8, the float32 scale (value = int8 * scale).

The checkpoint is only read, never written. Running the script twice on the same checkpoint gives identical files.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from evals.run import model_logprobs  # noqa: E402
from scripture.data import BOOKS, load_splits  # noqa: E402
from scripture.model import ModelConfig, ScriptureLM  # noqa: E402
from scripture.tokenizer import CharTokenizer  # noqa: E402

WEB = ROOT / "web"
FORMATS = ("float16", "int8")
ALIGN = 16
GREEDY_TOKENS = 160
PROMPTS = ["<gita>", "<quran>", "<bible>", "<gita>The Blessed Lord said:", "<bible>In the beginning God"]
TOKENIZER_CASES = [
    "", "a", "<gita>", "<quran>Lo! Allah", "<bible>In the beginning God created the heaven and the earth.\n",
    "<gita<quran>", "<bib", "x<bible>y", "Numbers 3:16 and X-rays", "Café naïve — “quotes”",
    "tab\there", "line one\nline two\n", "<gita><gita><bible>", "<>", "Krishna said: (O Arjuna!)",
]


def is_quantised(name, t):
    return t.dim() == 2  # weight matrices and embeddings; LayerNorm parameters and biases are 1-D


def encode_tensors(state, fmt):
    """[(name, shape, dtype, bytes, scale)] and the dequantised float32 state dict the browser will compute with."""
    out, deq = [], {}
    for name, t in state.items():
        if name.endswith("attn.mask"):
            continue  # a constant, rebuilt in the forward pass
        t = t.detach().to(torch.float32).cpu().contiguous()
        if fmt == "float16":
            h = t.to(torch.float16)
            out.append((name, list(t.shape), "float16", h.numpy().tobytes(), None))
            deq[name] = h.to(torch.float32)
        elif is_quantised(name, t):
            scale = torch.tensor(t.abs().max().item() / 127.0, dtype=torch.float32)
            q = torch.clamp(torch.round(t / scale), -127, 127).to(torch.int8)
            out.append((name, list(t.shape), "int8", q.numpy().tobytes(), float(scale)))
            deq[name] = q.to(torch.float32) * scale
        else:
            out.append((name, list(t.shape), "float32", t.numpy().tobytes(), None))
            deq[name] = t
    return out, deq


def pack(tensors, cfg, fmt):
    entries, offset = [], 0
    for name, shape, dtype, data, scale in tensors:
        offset = math.ceil(offset / ALIGN) * ALIGN
        e = {"name": name, "shape": shape, "dtype": dtype, "offset": offset}
        if scale is not None:
            e["scale"] = scale
        entries.append(e)
        offset += len(data)
    header = json.dumps({"format": fmt, "config": cfg, "tensors": entries}, separators=(",", ":")).encode()
    pad = (-(12 + len(header))) % ALIGN
    header += b" " * pad
    blob = bytearray(b"S10M" + struct.pack("<II", 1, len(header)) + header)
    base = len(blob)
    for e, (_, _, _, data, _) in zip(entries, tensors):
        blob += b"\0" * (base + e["offset"] - len(blob))
        blob += data
    return bytes(blob)


def model_from(state, cfg, device):
    m = ScriptureLM(ModelConfig(**cfg))
    m.load_state_dict(state, strict=False)  # the attention masks are buffers rebuilt by the constructor
    return m.to(device).eval()


@torch.no_grad()
def val_loss(model, data, device):
    per = {b: -model_logprobs(model, data["val"][b], device).double().mean().item() for b in BOOKS}
    per["overall"] = sum(per[b] for b in BOOKS) / len(BOOKS)
    return per


@torch.no_grad()
def logits_all(model, ids):
    return model(torch.tensor([ids]))[0][0]  # (T, vocab)


@torch.no_grad()
def greedy(model, ids, n):
    ids = list(ids)
    out = []
    for _ in range(n):
        nxt = int(logits_all(model, ids[-model.cfg.block_size:])[-1].argmax())
        ids.append(nxt)
        out.append(nxt)
    return out


def rounded(t, digits=6):
    return [round(float(x), digits) for x in t]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="out/best.pt")
    p.add_argument("--format", choices=FORMATS, default="int8", help="the format written to web/public/model")
    p.add_argument("--device", default="cpu", help="device for the loss measurement (cpu or mps)")
    p.add_argument("--threads", type=int, default=4, help="CPU threads, to share the machine politely")
    p.add_argument("--skip_loss", action="store_true", help="skip the slow loss measurement (keeps old numbers)")
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, state = ckpt["config"], ckpt["model"]
    tok = CharTokenizer(**ckpt["tokenizer"])
    manifest_path = WEB / "src/model/manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    data_tok, _, data = load_splits(ROOT / "data/processed")
    assert data_tok.tokens == tok.tokens, "data/processed does not match the checkpoint's tokenizer"

    losses = dict(old.get("val_loss_nats", {}))
    variants = {}
    for fmt in FORMATS:
        tensors, deq = encode_tensors(state, fmt)
        variants[fmt] = (tensors, deq)
    if not args.skip_loss:
        print("measuring validation loss (the eval harness's scorer, all held-out verses)...", flush=True)
        losses["float32"] = val_loss(model_from(state, cfg, args.device), data, args.device)
        print("  float32", losses["float32"], flush=True)
        for fmt in FORMATS:
            losses[fmt] = val_loss(model_from(variants[fmt][1], cfg, args.device), data, args.device)
            print(f"  {fmt}", losses[fmt], flush=True)

    tensors, deq = variants[args.format]
    blob = pack(tensors, cfg, args.format)
    sha = hashlib.sha256(blob).hexdigest()
    name = f"scripture-10m.{args.format}.{sha[:10]}.bin"
    out_dir = WEB / "public/model"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("scripture-10m.*.bin"):
        f.unlink()  # only one model file is ever published
    (out_dir / name).write_bytes(blob)

    sizes = {fmt: len(pack(variants[fmt][0], cfg, fmt)) for fmt in FORMATS}
    manifest = {
        "file": f"model/{name}", "format": args.format, "bytes": len(blob), "sha256": sha,
        "checkpoint_step": ckpt["step"], "params": sum(t.numel() for k, t in state.items() if not k.endswith("mask")),
        "config": cfg, "bytes_by_format": sizes, "val_loss_nats": losses,
        "note": "val_loss_nats: mean loss per token on all held-out verses, scored by evals/run.py model_logprobs "
                "with the weights of each format dequantised to float32. Written by web/scripts/export_model.py.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    (WEB / "src/model/vocab.json").write_text(json.dumps(tok.tokens, ensure_ascii=False) + "\n")

    # fixtures, from exactly the weights in the binary
    model = model_from(deq, cfg, "cpu")
    ref = model_from(state, cfg, "cpu")
    long_ids = data["val"]["quran"][:cfg["block_size"]].tolist()
    check_positions = [0, 1, 63, 127, 200, cfg["block_size"] - 1]
    cases = []
    for prompt in PROMPTS:
        ids = tok.encode(prompt)
        cases.append({"prompt": prompt, "ids": ids, "positions": list(range(len(ids))),
                      "logits": [rounded(r) for r in logits_all(model, ids)]})
    la, lr = logits_all(model, long_ids), logits_all(ref, long_ids)
    cases.append({"prompt": "first 256 tokens of the Quran validation text", "ids": long_ids,
                  "positions": check_positions, "logits": [rounded(la[i]) for i in check_positions]})
    greedy_cases = [{"prompt": b, "ids": tok.encode(b), "tokens": greedy(model, tok.encode(b), GREEDY_TOKENS)}
                    for b in ("<gita>", "<quran>", "<bible>")]
    for g in greedy_cases:
        g["text"] = tok.decode(g["tokens"])
    fixtures = {
        "about": "PyTorch (scripture/model.py) logits and greedy continuations computed from the weights stored in "
                 f"{name}, dequantised to float32. Written by web/scripts/export_model.py.",
        "model_file": name, "format": args.format, "vocab_size": tok.vocab_size,
        "max_abs_logit_change_from_float32_checkpoint": round(float((la - lr).abs().max()), 6),
        "logit_cases": cases, "greedy": greedy_cases,
    }
    fx = WEB / "test/fixtures"
    fx.mkdir(parents=True, exist_ok=True)
    (fx / "parity.json").write_text(json.dumps(fixtures, separators=(",", ":")) + "\n")
    (fx / "tokenizer.json").write_text(json.dumps(
        {"tokens": tok.tokens, "cases": [{"text": s, "ids": tok.encode(s)} for s in TOKENIZER_CASES]},
        ensure_ascii=False, indent=1) + "\n")

    print(f"wrote {out_dir / name} ({len(blob):,} bytes, sha256 {sha[:16]}...)")
    for fmt in FORMATS:
        line = f"  {fmt:8s} {sizes[fmt]:>12,} bytes"
        if fmt in losses and "float32" in losses:
            d = {b: losses[fmt][b] - losses["float32"][b] for b in (*BOOKS, "overall")}
            line += "  loss change: " + ", ".join(f"{b} {v:+.5f}" for b, v in d.items())
        print(line)


if __name__ == "__main__":
    main()
