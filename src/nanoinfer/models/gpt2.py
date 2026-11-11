"""GPT-2, written from scratch."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
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
    def gpt2_small(cls) -> "GPTConfig":
        """GPT-2 small, the 124M-parameter model."""
        return cls()
