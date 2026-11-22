import math

import pytest
import torch

from scripture.model import ModelConfig, ScriptureLM

SMALL = ModelConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def make_model(cfg=SMALL):
    torch.manual_seed(0)
    return ScriptureLM(cfg).eval()


def test_output_shapes():
    model = make_model()
    idx = torch.randint(0, SMALL.vocab_size, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, SMALL.vocab_size)
    assert loss is None
    _, loss = model(idx, idx)
    assert loss.shape == ()


def test_causality():
    model = make_model()
    idx = torch.randint(0, SMALL.vocab_size, (1, 12))
    changed = idx.clone()
    changed[0, 8] = (idx[0, 8] + 1) % SMALL.vocab_size
    a, _ = model(idx)
    b, _ = model(changed)
    assert torch.allclose(a[:, :8], b[:, :8])
    assert not torch.allclose(a[:, 8:], b[:, 8:])


def test_initial_loss_is_uniform_guess():
    model = make_model()
    idx = torch.randint(0, SMALL.vocab_size, (8, 16))
    targets = torch.randint(0, SMALL.vocab_size, (8, 16))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(SMALL.vocab_size)) < 0.1


def test_overfits_one_batch():
    model = make_model().train()
    torch.manual_seed(1)
    data = torch.randint(0, SMALL.vocab_size, (4, 17))
    x, y = data[:, :-1], data[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(50):
        _, loss = model(x, y)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first


def test_default_param_count():
    model = ScriptureLM(ModelConfig())
    assert model.num_params() == 3_240_448  # ~3.2M with the default vocab_size of 100
    assert 3.0e6 < model.num_params() < 3.5e6


def test_generate_length_and_stream():
    model = make_model()
    idx = torch.tensor([[1, 2, 3]])
    out = model.generate(idx, 5)
    assert out.shape == (1, 8)
    assert torch.equal(out[:, :3], idx)
    ids = list(model.generate_stream(idx, 4))
    assert len(ids) == 4 and all(isinstance(i, int) for i in ids)


def test_top_k_one_is_greedy():
    model = make_model()
    idx = torch.tensor([[1, 2, 3]])
    a = model.generate(idx, 10, top_k=1)
    b = model.generate(idx, 10, top_k=1)
    assert torch.equal(a, b)
    logits, _ = model(idx)
    assert a[0, 3].item() == logits[0, -1].argmax().item()


def test_top_k_restricts_choices():
    model = make_model()
    idx = torch.tensor([[1, 2, 3]])
    logits, _ = model(idx)
    allowed = set(torch.topk(logits[0, -1], 3).indices.tolist())
    torch.manual_seed(0)
    for _ in range(30):
        assert next(model.generate_stream(idx, 1, temperature=2.0, top_k=3)) in allowed


def test_generation_crops_to_block_size():
    model = make_model()
    idx = torch.randint(0, SMALL.vocab_size, (1, SMALL.block_size))
    out = model.generate(idx, 10)  # context grows past block_size without error
    assert out.shape == (1, SMALL.block_size + 10)


def test_forward_rejects_too_long_input():
    model = make_model()
    idx = torch.zeros((1, SMALL.block_size + 1), dtype=torch.long)
    with pytest.raises(AssertionError):
        model(idx)
