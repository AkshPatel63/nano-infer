"""Check that our GPT-2 computes exactly what Hugging Face's GPT-2 computes.

Both tests follow the same recipe: put the same weights into our model (nano)
and Hugging Face's reference model (hf), feed both the same tokens, and check
that the logits match within a small tolerance.

- The tiny random test needs no download and runs in CI on every push.
- The real GPT-2 test uses OpenAI's trained weights. It's marked slow, so run
  it with: uv run pytest -m slow tests/test_gpt2_matches_hf.py
"""



import pytest
import torch 
from transformers import GPT2Config as HFConfig 
from transformers import GPT2LMHeadModel

from nanoinfer.models.gpt2 import GPT2Config, GPT2Model, load_hf_weights
from nanoinfer.tokenizer import GPT2Tokenizer


def test_tiny_random_model_matches_hf():
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
    
@pytest.mark.slow
def test_pretrained_gpt2_matches_hf():
    """The real GPT-2 (124M) with OpenAI's trained weights, on real text."""
    hf = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", attn_implementation="eager"
    ).eval()
    nano = GPT2Model.from_pretrained()

    ids = GPT2Tokenizer().encode("The cat sat on the mat because it was tired")
    idx = torch.tensor([ids])
    with torch.no_grad():
        torch.testing.assert_close(nano(idx), hf(idx).logits, atol=1e-4, rtol=1e-4)

