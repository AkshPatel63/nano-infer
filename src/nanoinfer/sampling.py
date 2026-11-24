"""Sampling: Turn next-token logits into a choide, with controlled randomness"""

import torch 

def apply_temperature(logits, temperature):
    """Divide logits by the temperature. Below 1 = more focused, above 1 = more random.  """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use greedy decoding for 0)")
    return logits / temperature


def top_k_filter(logits, k):
    """Keep the k highest logits in eaach row, set the rest to -inf."""
    if k is None or k >= logits.size(-1):
        return logits 
    kth_best = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth_best, float("-inf"))

def top_p_filter(logits, p):
    """Keep the smallest set of tokens whose probabilities atleast add up to p """
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    before = probs.cumsum(dim=-1) - probs          # total probability of the tokens ahead of each one
    sorted_logits = sorted_logits.masked_fill(before >= p, float("-inf"))
    # put the filtered scores back in their original vocabulary order
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """Pick one token per row from (B, vocab) logits. Returns (B, 1) token ids."""
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    # roll the die on the CPU, so a seed gives the same result on CPU, MPS and CUDA
    next_id = torch.multinomial(probs.cpu(), num_samples=1, generator=generator)
    return next_id.to(logits.device)