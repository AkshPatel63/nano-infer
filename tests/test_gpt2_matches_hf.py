"""Check that our GPT-2 computes exactly what Hugging Face's GPT-2 computes.

Every test puts the same weights into our model (nano) and Hugging Face's
reference model (hf), feeds both the same tokens, and compares the results.

- The tiny random test needs no download and runs in CI on every push.
- The pretrained tests use OpenAI's trained weights on several kinds of text.
  They're marked slow. Run them, with details printed, using:
  uv run pytest -m slow tests/test_gpt2_matches_hf.py -v -s
"""

import pytest
import torch
from transformers import GPT2Config as HFConfig
from transformers import GPT2LMHeadModel

from nanoinfer.generate import generate
from nanoinfer.models.gpt2 import GPT2Config, GPT2Model, load_hf_weights
from nanoinfer.tokenizer import GPT2Tokenizer

PROMPTS = {
    "history": "In 1905, Albert Einstein published four papers that",
    "story": "Once upon a time, in a small village by the sea, there lived an old fisherman who",
    "question": "Q: What is the capital of France?\nA:",
    "recipe": "The following is a recipe for chocolate chip cookies.\n\nIngredients:\n-",
    "code": 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
}


# --- Shared models: each is loaded once, and only if a test needs it ---

@pytest.fixture(scope="module")
def tok():
    return GPT2Tokenizer()


@pytest.fixture(scope="module")
def hf_gpt2():
    return GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", attn_implementation="eager"
    ).eval()


@pytest.fixture(scope="module")
def nano_gpt2():
    return GPT2Model.from_pretrained()


# --- Fast test (runs in CI) ---

def test_tiny_random_model_matches_hf():
    """A tiny GPT-2 with random weights: checks our loader and math, fast."""
    torch.manual_seed(0)
    hf_cfg = HFConfig(
        vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        attn_implementation="eager",
    )
    hf = GPT2LMHeadModel(hf_cfg).eval()
    nano = GPT2Model(
        GPT2Config(vocab_size=100, context_length=32, emb_dim=32, n_heads=4, n_layers=2)
    ).eval()
    load_hf_weights(nano, hf.state_dict())

    idx = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        torch.testing.assert_close(nano(idx), hf(idx).logits, atol=1e-5, rtol=1e-5)


# --- Slow tests (real GPT-2) ---

def first_difference(a, b):
    """Index of the first position where two token lists differ."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


@pytest.mark.slow
@pytest.mark.parametrize("prompt", PROMPTS.values(), ids=PROMPTS.keys())
def test_pretrained_logits_match_hf(prompt, tok, hf_gpt2, nano_gpt2):
    """Same text in, same scores out, for every position."""
    idx = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        nano_logits = nano_gpt2(idx)
        hf_logits = hf_gpt2(idx).logits

    gap = (nano_logits - hf_logits).abs().max().item()
    print(f"\n\n  PROMPT  {prompt!r}")
    print(f"  TOKENS  {idx.shape[1]}")
    print(f"  GAP     {gap:.2e}  (largest difference between nano and hf scores)")
    torch.testing.assert_close(nano_logits, hf_logits, atol=1e-4, rtol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("prompt", PROMPTS.values(), ids=PROMPTS.keys())
def test_greedy_generation_matches_hf(prompt, tok, hf_gpt2, nano_gpt2):
    """Greedy generation: our loop must produce exactly the same tokens as HF's generate()."""
    idx = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        expected = hf_gpt2.generate(
            idx,
            attention_mask=torch.ones_like(idx),
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tok.eot_id,
        )
    actual = generate(nano_gpt2, idx, max_new_tokens=50, eos_id=tok.eot_id)

    n_prompt = idx.shape[1]
    nano_new = actual[0, n_prompt:].tolist()       # only the generated tokens
    hf_new = expected[0, n_prompt:].tolist()
    same = nano_new == hf_new

    print(f"\n\n  PROMPT  {prompt!r}")
    print(f"  NANO    {tok.decode(nano_new)!r}")
    print(f"  HF      {tok.decode(hf_new)!r}")
    if same:
        print(f"  MATCH   yes, all {len(nano_new)} new tokens identical")
    else:
        i = first_difference(nano_new, hf_new)
        print(f"  MATCH   NO, first difference at new token #{i}")
    assert same, "nano and hf generated different tokens (see the printout above)"