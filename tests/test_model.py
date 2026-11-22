import math

import pytest
import torch
from torch.nn import functional as F

from scripture.model import ModelConfig, ScriptureLM

SMALL = ModelConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def make_model(cfg=SMALL):
    torch.manual_seed(0)
    return ScriptureLM(cfg).eval()


def random_weights_model(cfg=SMALL):
    """A float64 model with large random weights, so that small maths mistakes show up in the logits."""
    model = make_model(cfg).double()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.3)
    return model


def reference_logits(model, idx):
    """The same forward pass, built from the model's weights with PyTorch's own attention."""
    cfg = model.cfg
    B, T = idx.shape
    C, nh = cfg.n_embd, cfg.n_head

    def ln(x, m):
        return F.layer_norm(x, (C,), m.weight, m.bias)

    x = model.tok_emb.weight[idx] + model.pos_emb.weight[:T]
    for blk in model.blocks:
        q, k, v = F.linear(ln(x, blk.ln1), blk.attn.qkv.weight).split(C, dim=2)
        q, k, v = (t.view(B, T, nh, C // nh).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        x = x + F.linear(y, blk.attn.proj.weight, blk.attn.proj.bias)
        fc1, fc2 = blk.mlp.net[0], blk.mlp.net[2]
        x = x + F.linear(F.gelu(F.linear(ln(x, blk.ln2), fc1.weight, fc1.bias)), fc2.weight, fc2.bias)
    return F.linear(ln(x, model.ln_f), model.lm_head.weight)


def capture_probs(monkeypatch):
    """Replace sampling with greedy picking and record the probabilities generate_stream samples from."""
    seen = []

    def fake_multinomial(probs, num_samples):
        seen.append(probs.clone())
        return probs.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    return seen


def test_output_shapes():
    model = make_model()
    idx = torch.randint(0, SMALL.vocab_size, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, SMALL.vocab_size)
    assert loss is None
    _, loss = model(idx, idx)
    assert loss.shape == ()


@pytest.mark.parametrize("T", [1, 7, SMALL.block_size])
def test_matches_reference_forward(T):
    model = random_weights_model()
    idx = torch.randint(0, SMALL.vocab_size, (2, T))
    logits, _ = model(idx)
    assert torch.allclose(logits, reference_logits(model, idx), atol=1e-10)


@pytest.mark.parametrize("pos", [0, 8, SMALL.block_size - 1])
def test_causality(pos):
    model = random_weights_model()
    idx = torch.randint(0, SMALL.vocab_size, (1, SMALL.block_size))
    changed = idx.clone()
    changed[0, pos] = (idx[0, pos] + 1) % SMALL.vocab_size
    a, _ = model(idx)
    b, _ = model(changed)
    assert torch.allclose(a[:, :pos], b[:, :pos])
    assert not torch.allclose(a[:, pos], b[:, pos])


@pytest.mark.parametrize("T", [1, 5, SMALL.block_size])
def test_prefix_logits_unchanged_by_later_tokens(T):
    model = random_weights_model()
    idx = torch.randint(0, SMALL.vocab_size, (1, SMALL.block_size))
    full, _ = model(idx)
    prefix, _ = model(idx[:, :T])
    assert torch.allclose(prefix, full[:, :T])


def test_position_matters():
    model = random_weights_model()
    same = torch.full((1, SMALL.block_size), 3)
    logits, _ = model(same)
    # without position embeddings every position would see identical tokens and give identical logits
    assert not torch.allclose(logits[0, 0], logits[0, -1])


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
    assert model.num_params() == 10_815_744  # with the default vocab_size of 100
    assert 10.5e6 < model.num_params() < 11.0e6


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


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_temperature_scales_logits(monkeypatch, temperature):
    model = random_weights_model()
    idx = torch.tensor([[1, 2, 3]])
    seen = capture_probs(monkeypatch)
    next(model.generate_stream(idx, 1, temperature=temperature))
    logits, _ = model(idx)
    assert torch.allclose(seen[0][0], F.softmax(logits[0, -1] / temperature, dim=-1))


def test_top_k_keeps_exactly_k_renormalised(monkeypatch):
    model = random_weights_model()
    idx = torch.tensor([[1, 2, 3]])
    seen = capture_probs(monkeypatch)
    next(model.generate_stream(idx, 1, top_k=3))
    logits, _ = model(idx)
    top = torch.topk(logits[0, -1], 3).indices
    probs = seen[0][0]
    assert set(probs.nonzero().flatten().tolist()) == set(top.tolist())
    assert torch.allclose(probs[top], F.softmax(logits[0, -1, top], dim=-1))


def test_generation_crops_to_newest_tokens(monkeypatch):
    model = random_weights_model()
    torch.manual_seed(0)
    idx = torch.randint(0, SMALL.vocab_size, (1, SMALL.block_size + 5))
    seen = capture_probs(monkeypatch)
    out = model.generate(idx, 3)  # context grows past block_size without error
    assert out.shape == (1, SMALL.block_size + 8)
    for step, probs in enumerate(seen):
        context = out[:, : SMALL.block_size + 5 + step][:, -SMALL.block_size:]
        logits, _ = model(context)
        assert torch.allclose(probs[0], F.softmax(logits[0, -1], dim=-1))


def test_generation_ignores_dropout_and_restores_mode(monkeypatch):
    cfg = ModelConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.5)
    model = random_weights_model(cfg).train()
    idx = torch.tensor([[1, 2, 3]])
    seen = capture_probs(monkeypatch)
    list(model.generate_stream(idx, 1))
    assert model.training  # previous mode restored
    model.eval()
    logits, _ = model(idx)
    assert torch.allclose(seen[0][0], F.softmax(logits[0, -1], dim=-1))


def test_forward_rejects_too_long_input():
    model = make_model()
    idx = torch.zeros((1, SMALL.block_size + 1), dtype=torch.long)
    with pytest.raises(AssertionError):
        model(idx)
