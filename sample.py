"""Generate text in the style of one book.

    python sample.py --book gita --prompt "Krishna said:"
"""
import argparse

import torch

from scripture.data import BOOKS
from train import load_checkpoint, pick_device


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--book", choices=BOOKS, required=True)
    p.add_argument("--prompt", default="", help="text to continue after the book tag")
    p.add_argument("--tokens", type=int, default=500, help="how many tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default="out/best.pt")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    model, tok, _ = load_checkpoint(args.ckpt, device)
    torch.manual_seed(args.seed)
    idx = torch.tensor([tok.encode(f"<{args.book}>" + args.prompt)], device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
