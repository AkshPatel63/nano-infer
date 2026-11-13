"""GPT-2, written from scratch."""

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GPT2Config:
    """Hyperparameters of a GPT-2 model."""

    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    qkv_bias: bool = True
    ln_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.emb_dim % self.n_heads != 0:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Size of each attention head (768 / 12 = 64 for GPT-2 small)."""
        return self.emb_dim // self.n_heads

    @classmethod
    def gpt2_small(cls) -> "GPT2Config":
        """GPT-2 small, the 124M-parameter model."""
        return cls()


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension, as used in GPT-2."""

    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)  # divide by N, not N-1
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class GELU(nn.Module):
    """GELU with the tanh approximation."""

    def forward(self, x):
        return (
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        )


class MLP(nn.Module):
    """Feed Forward network, applied to each token separately"""

    def __init__(self, emb_dim):
        super().__init__()
        self.c_fc = nn.Linear(emb_dim, 4 * emb_dim)  # expand 768 -> 3072
        self.gelu = GELU()
        self.c_proj = nn.Linear(4 * emb_dim, emb_dim)  # project back: 3072 -> 768

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class CausalSelfAttention(nn.Module):
    """Multi-head causal Self-attention"""

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.emb_dim = cfg.emb_dim
        self.c_attn = nn.Linear(
            cfg.emb_dim, 3 * cfg.emb_dim, bias=cfg.qkv_bias
        )  # q, k, v together
        self.c_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        mask = torch.triu(
            torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer(
            "mask", mask, persistent=False
        )  # Ture = Futrue position, hide it

    def forward(self, x):  # (B, T, 768)
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.emb_dim, dim=-1)  # each (B, T, 768)

        # split 768 into 12 heads of 64, then move heads next to batch

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, 12, T, 64)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, 12, T, T)
        scores = scores.masked_fill(
            self.mask[:T, :T], float("-inf")
        )  # hide the future tokens with -inf
        weights = torch.softmax(scores, dim=-1)

        out = weights @ v  # (B, 12, T, 64)
        out = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )  # merge heads : (B, T, 768)
        return self.c_proj(out)  # (B, T, 768)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.emb_dim, cfg.ln_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.emb_dim, cfg.ln_eps)
        self.mlp = MLP(cfg.emb_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # communicate
        x = x + self.mlp(self.ln_2(x))  # compute
        return x
