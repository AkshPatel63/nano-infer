"""Text generation: repeatedly predict the next token and append it."""

import torch 

@torch.no_grad()
def generate(model, idx, max_new_tokens, eos_id=None):
    """Greedy generation: always pick the most likely next token.

    idx: (B, T) token IDs. Returns (B, T + new tokens).
    """

    context_length = model.cfg.context_length
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]                   # crop to what the model can see
        logits = model(idx_cond)                              # (B, T, vocab)
        next_logits = logits[:, -1, :]                        # (B, vocab): last position only
        next_id = next_logits.argmax(dim=-1, keepdim=True)    # (B, 1): greedy pick
        idx = torch.cat([idx, next_id], dim=1)                # (B, T + 1)
        if eos_id is not None and (next_id == eos_id).all():
            break                                             # every sequence said "done"
    return idx
