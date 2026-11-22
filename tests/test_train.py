"""The learning-rate schedule and a short real training run with a checkpoint and a resume."""
import json
import math
from pathlib import Path

import pytest
import torch

import train
from scripture.data import BOOKS, load_splits
from scripture.model import ModelConfig, ScriptureLM

TINY = ["--n_layer", "1", "--n_head", "2", "--n_embd", "16", "--block_size", "16", "--batch_size", "8",
        "--dropout", "0", "--lr", "1e-2", "--min_lr", "1e-3", "--warmup_iters", "4", "--max_iters", "20",
        "--eval_interval", "10", "--eval_iters", "2", "--log_interval", "5", "--device", "cpu"]


def test_lr_schedule_endpoints():
    args = train.parse_args(["--lr", "1e-3", "--min_lr", "1e-4", "--warmup_iters", "200", "--max_iters", "12000"])
    assert train.get_lr(0, args) == pytest.approx(1e-3 / 200)
    assert train.get_lr(99, args) == pytest.approx(0.5e-3)
    assert train.get_lr(199, args) == pytest.approx(1e-3)
    assert train.get_lr(200, args) == pytest.approx(1e-3)
    assert train.get_lr(200 + (12000 - 200) // 2, args) == pytest.approx(0.55e-3)
    assert train.get_lr(12000, args) == pytest.approx(1e-4)
    assert train.get_lr(15000, args) == pytest.approx(1e-4)  # stays at the floor past the end
    lrs = [train.get_lr(s, args) for s in range(200, 12001, 100)]
    assert all(a > b for a, b in zip(lrs, lrs[1:]))  # cosine decay only goes down


def test_defaults():
    args = train.parse_args([])
    assert (args.batch_size, args.block_size, args.max_iters, args.warmup_iters) == (64, 256, 12000, 200)
    assert (args.lr, args.min_lr, args.eval_interval, args.eval_iters) == (1e-3, 1e-4, 500, 40)


def test_weight_decay_only_on_matrices():
    model = torch.nn.Sequential(torch.nn.Embedding(5, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 3))
    opt = train.make_optimizer(model, train.parse_args([]))
    decayed, plain = opt.param_groups
    assert decayed["weight_decay"] == 0.1 and plain["weight_decay"] == 0.0
    assert {p.dim() for p in decayed["params"]} == {2} and len(decayed["params"]) == 2
    assert {p.dim() for p in plain["params"]} == {1} and len(plain["params"]) == 3


def test_estimate_loss_is_repeatable_and_restores_training_mode(tiny_data_dir):
    _, _, data = load_splits(tiny_data_dir)
    torch.manual_seed(0)
    model = ScriptureLM(ModelConfig(vocab_size=50, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.5))
    args = train.parse_args(TINY)
    a = train.estimate_loss(model, data, args, "cpu")
    b = train.estimate_loss(model, data, args, "cpu")
    assert a == b  # same windows every time, and no dropout while evaluating
    assert model.training
    assert a["val"]["gita"] != a["val"]["bible"]


def test_save_atomic_never_leaves_a_broken_file(tmp_path, monkeypatch):
    path = tmp_path / "ckpt.pt"
    train.save_atomic({"step": 1}, path)
    real_save = torch.save

    def dies_half_way(obj, f):
        Path(f).write_bytes(b"half a checkpoint")
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", dies_half_way)
    with pytest.raises(KeyboardInterrupt):
        train.save_atomic({"step": 2}, path)
    monkeypatch.setattr(torch, "save", real_save)
    assert torch.load(path, weights_only=False) == {"step": 1}


def read_log(out_dir):
    return [json.loads(line) for line in (out_dir / "losses.jsonl").read_text().splitlines()]


def run(out_dir, data_dir, *extra):
    return train.train(train.parse_args(TINY + ["--out_dir", str(out_dir), "--data_dir", str(data_dir), *extra]))


def test_training_run_logs_and_checkpoints(tmp_path, tiny_data_dir):
    run(tmp_path, tiny_data_dir)
    log = read_log(tmp_path)
    assert [r["step"] for r in log] == [0, 10, 20]
    for r in log:
        assert set(r["train"]) == set(r["val"]) == {*BOOKS, "overall"}
        assert r["val"]["overall"] == pytest.approx(sum(r["val"][b] for b in BOOKS) / 3)
    model, tok, _ = train.load_checkpoint(tmp_path / "ckpt.pt", "cpu")
    assert model.cfg.n_embd == 16 and tok.vocab_size == model.cfg.vocab_size
    first, last = log[0], log[-1]
    assert first["val"]["overall"] == pytest.approx(math.log(tok.vocab_size), abs=0.1)  # a uniform guess at first
    for book in BOOKS:  # every book improves
        assert last["val"][book] < first["val"][book] - 0.5
    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert ckpt["step"] == 20
    assert ckpt["best_val"] == pytest.approx(min(r["val"]["overall"] for r in log))
    assert best["step"] == min(log, key=lambda r: r["val"]["overall"])["step"]
    assert set(ckpt) >= {"model", "optimizer", "config", "tokenizer", "step", "best_val", "args"}
    args = train.parse_args(TINY)
    for group in ckpt["optimizer"]["param_groups"]:  # the schedule really reaches the optimizer
        assert group["lr"] == pytest.approx(train.get_lr(19, args))


def test_best_checkpoint_keeps_lowest_val_loss(tmp_path, tiny_data_dir, monkeypatch):
    scripted = iter([3.0, 1.0, 1.0, 2.0])  # val loss at steps 0, 10, 20, 25: best (first) at step 10

    def fake_estimate(*_):
        v = next(scripted)
        losses = {**{b: v for b in BOOKS}, "overall": v}
        return {"train": losses, "val": losses}

    monkeypatch.setattr(train, "estimate_loss", fake_estimate)
    run(tmp_path, tiny_data_dir, "--max_iters", "25")
    assert [r["step"] for r in read_log(tmp_path)] == [0, 10, 20, 25]  # the last step is always evaluated
    assert torch.load(tmp_path / "best.pt", weights_only=False)["step"] == 10
    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert ckpt["step"] == 25 and ckpt["best_val"] == 1.0


def test_gradients_are_clipped(tmp_path, tiny_data_dir):
    # Adam rescales gradients, but clipped to almost nothing they fall below its epsilon and nothing is learned
    run(tmp_path, tiny_data_dir, "--grad_clip", "1e-12", "--weight_decay", "0")
    log = read_log(tmp_path)
    assert log[-1]["val"]["overall"] == pytest.approx(log[0]["val"]["overall"], abs=0.02)


class Interrupted(Exception):
    pass


def spy_on_training_batches(monkeypatch):
    """Record the `weights` of every training batch (evaluation batches pass `book` instead)."""
    seen, real = [], train.get_batch

    def spy(*a, **k):
        if "book" not in k:
            seen.append(k.get("weights"))
        return real(*a, **k)

    monkeypatch.setattr(train, "get_batch", spy)
    return seen


@pytest.mark.parametrize("mode", ["verse", "book"])
def test_sampling_modes(tmp_path, tiny_data_dir, monkeypatch, mode):
    seen = spy_on_training_batches(monkeypatch)
    run(tmp_path, tiny_data_dir, "--sampling", mode)
    assert len(seen) == 20
    if mode == "book":
        assert all(w is None for w in seen)
    else:  # books weighted by their number of training tokens
        _, _, data = load_splits(tiny_data_dir)
        sizes = torch.tensor([float(len(data["train"][b])) for b in BOOKS])
        assert all(torch.equal(w, sizes) for w in seen)
    assert torch.load(tmp_path / "ckpt.pt", weights_only=False)["args"]["sampling"] == mode


def test_old_checkpoint_without_sampling_resumes_by_book(tmp_path, tiny_data_dir, monkeypatch):
    real_save = train.save_atomic

    def save_then_crash(obj, path):
        real_save(obj, path)
        if obj["step"] == 10 and path.name == "ckpt.pt":
            raise Interrupted

    monkeypatch.setattr(train, "save_atomic", save_then_crash)
    with pytest.raises(Interrupted):
        run(tmp_path, tiny_data_dir)
    monkeypatch.setattr(train, "save_atomic", real_save)
    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    del ckpt["args"]["sampling"]  # as saved before the option existed
    torch.save(ckpt, tmp_path / "ckpt.pt")
    seen = spy_on_training_batches(monkeypatch)
    train.train(train.parse_args(["--resume", "--out_dir", str(tmp_path), "--device", "cpu"]))
    assert len(seen) == 10 and all(w is None for w in seen)
    assert torch.load(tmp_path / "ckpt.pt", weights_only=False)["args"]["sampling"] == "book"


def test_resume_matches_an_uninterrupted_run_without_dropout(tmp_path, tiny_data_dir, monkeypatch):
    straight, resumed = tmp_path / "straight", tmp_path / "resumed"
    final = run(straight, tiny_data_dir)

    real_save = train.save_atomic

    def save_then_crash(obj, path):  # stop the second run right after its step-10 checkpoint
        real_save(obj, path)
        if obj["step"] == 10 and path.name == "ckpt.pt":
            raise Interrupted

    monkeypatch.setattr(train, "save_atomic", save_then_crash)
    with pytest.raises(Interrupted):
        run(resumed, tiny_data_dir)
    monkeypatch.setattr(train, "save_atomic", real_save)
    assert [r["step"] for r in read_log(resumed)] == [0]  # crashed before logging step 10
    with open(resumed / "losses.jsonl", "a") as f:
        f.write(json.dumps({"step": 10}) + "\n")  # as if step 10 had been logged too: must not appear twice
        f.write(json.dumps({"step": 15}) + "\n")  # from after the checkpoint: must not survive
        f.write('{"step": 16, "tr')  # cut short when the process died

    again = train.train(train.parse_args(["--resume", "--out_dir", str(resumed), "--device", "cpu"]))
    a, b = read_log(straight), read_log(resumed)
    assert [r["step"] for r in b] == [0, 10, 20]  # carries on after step 10, without repeating it
    assert [r["lr"] for r in b] == [r["lr"] for r in a]
    assert b[-1]["val"] == pytest.approx(a[-1]["val"])
    for p, q in zip(final.parameters(), again.parameters()):  # same weights as never stopping
        assert torch.allclose(p, q, atol=1e-6)
    assert torch.load(resumed / "ckpt.pt", weights_only=False)["step"] == 20
