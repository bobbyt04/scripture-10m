# Training runs

Two runs of the same 10,790,400-parameter model (6 layers, 6 heads, 384-wide, 256-token context, dropout 0.2,
batch 64, learning rate 1e-3 decaying to 1e-4 over 12,000 steps, seed 1337). They differ only in how each training
sequence picks its book. Each `losses.jsonl` has one line per evaluation (every 500 steps): the step, the learning
rate, the mean training and validation loss per book in nats per token (40 batches each), and `elapsed_s`.
Checkpoints are not in the repo.

## `verse/`: the real run

```bash
python train.py            # --sampling verse is the default
python evals/run.py --ckpt out/best.pt --data_dir data/processed
```

Every sequence picks its book in proportion to the book's size, so every verse is seen about equally often.

- Hardware: Apple M2, PyTorch on MPS.
- Steps: 12,000 of 12,000, in one run with no resume. `best.pt` is step 11,000, the lowest overall validation
  loss (0.944 nats); the final step 12,000 is 0.950. Validation loss was almost flat over the last 3,000 steps.
- Wall time: 35,649 s (9 h 54 min), including the evaluations. This is inflated: for roughly the first 6,500
  steps the abandoned run below and other jobs were training on the same machine, and the run took about 4 s per
  step. Once it had the machine mostly to itself it took about 1.5 s per step, so an uncontended run would take
  somewhere near 5 hours. That figure is an estimate, not a measurement.

## `book/`: the abandoned run

```bash
python train.py --sampling book   # at the time this was the default; it ran before --sampling existed
```

Every sequence picks one of the three books with equal probability. The Gita has 701 verses against the Bible's
31,102, so each Gita verse was seen about 40 times as often as each Bible verse. The model memorised the Gita: by
step 8,500 its Gita training loss was 0.06 nats and its Gita validation loss 2.39, up from a minimum of 1.22 at
step 1,000. Its overall validation loss was lowest at step 1,500 (1.153). The run was stopped after step 8,500 and
`losses.jsonl` covers steps 0 to 8,500. The checkpoint evaluated in `evals/results/book_sampling.json` is the
step-8,500 `ckpt.pt`, the memorising model, not its best one:

```bash
python evals/run.py --ckpt out_equal_books/ckpt.pt --data_dir data/processed --out evals/results/book_sampling.json
```

This run shared the machine with the real run and other jobs for its whole life, so its `elapsed_s` values and
step times say nothing about the cost of training. Do not read timings from it.
