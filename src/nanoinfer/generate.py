"""Text generation: repeatedly predict the next token and append it."""

import torch

from nanoinfer.sampling import sample_next_token


@torch.no_grad()
def generate(
    model, idx, max_new_tokens, eos_id=None,
    temperature=0.0, top_k=None, top_p=None, seed=None,
):
    """Generate tokens after idx (B, T). Returns (B, T + new tokens).

    temperature=0 means greedy (always the top token).
    temperature>0 means sampling, optionally narrowed with top_k and/or top_p.
    A seed makes sampling reproducible.
    """
    context_length = model.cfg.context_length
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]                   # crop to what the model can see
        next_logits = model(idx_cond)[:, -1, :]               # (B, vocab): last position only
        if temperature == 0:
            next_id = next_logits.argmax(dim=-1, keepdim=True)          # greedy
        else:
            next_id = sample_next_token(next_logits, temperature, top_k, top_p, generator)
        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None and (next_id == eos_id).all():
            break
    return idx