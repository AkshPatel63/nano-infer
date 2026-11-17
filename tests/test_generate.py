"""Tests for the generation loop, using a tiny random GPT-2."""

import pytest
import torch

from nanoinfer.generate import generate
from nanoinfer.models.gpt2 import GPT2Config, GPT2Model


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=100, context_length=32, emb_dim=32, n_heads=4, n_layers=2)
    return GPT2Model(cfg).eval()


def test_adds_new_tokens_and_keeps_prompt(tiny_model):
    idx = torch.tensor([[1, 2, 3]])
    out = generate(tiny_model, idx, max_new_tokens=5)
    assert out.shape == (1, 8)                       # 3 prompt tokens + 5 new
    assert torch.equal(out[:, :3], idx)              # prompt unchanged at the front


def test_greedy_is_deterministic(tiny_model):
    idx = torch.tensor([[1, 2, 3]])
    out1 = generate(tiny_model, idx, max_new_tokens=5)
    out2 = generate(tiny_model, idx, max_new_tokens=5)
    assert torch.equal(out1, out2)                   # no randomness in greedy


def test_stops_at_eos(tiny_model):
    idx = torch.tensor([[1, 2, 3]])
    first = generate(tiny_model, idx, max_new_tokens=1)[0, -1].item()
    out = generate(tiny_model, idx, max_new_tokens=10, eos_id=first)
    assert out.shape == (1, 4)                       # stopped right after the "EOS" token


def test_handles_prompt_longer_than_context(tiny_model):
    idx = torch.randint(0, 100, (1, 40))             # 40 tokens > context of 32
    out = generate(tiny_model, idx, max_new_tokens=3)
    assert out.shape == (1, 43)                      # cropping kept it running