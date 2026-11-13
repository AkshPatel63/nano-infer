import torch
import torch.nn.functional as F

from nanoinfer.models.gpt2 import MLP, CausalSelfAttention, GPT2Config


def test_mlp_output_shape():
    x = torch.randn(2, 5, 768)
    assert MLP(768)(x).shape == (2, 5, 768)


def test_mlp_parameter_count():
    # c_fc: 768*3072 + 3072, c_proj: 3072*768 + 768
    n = sum(p.numel() for p in MLP(768).parameters())
    assert n == 4_722_432


def test_mlp_is_per_token():
    torch.manual_seed(0)
    mlp = MLP(768)
    x = torch.randn(1, 10, 768)
    x2 = x.clone()
    x2[0, 7] = torch.randn(768)  # change only token 7
    out, out2 = mlp(x), mlp(x2)
    torch.testing.assert_close(out[0, :7], out2[0, :7])  # tokens 0-6 unchanged
    assert not torch.allclose(out[0, 7], out2[0, 7])  # token 7 changed


def test_attention_output_shape():
    attn = CausalSelfAttention(GPT2Config())
    x = torch.randn(2, 5, 768)
    assert attn(x).shape == (2, 5, 768)


def test_attention_is_causal():
    # Changing token 3 must not affect tokens 0-2, which can't see it.
    torch.manual_seed(0)
    attn = CausalSelfAttention(GPT2Config())
    x = torch.randn(1, 6, 768)
    x2 = x.clone()
    x2[0, 3] = torch.randn(768)
    out, out2 = attn(x), attn(x2)
    torch.testing.assert_close(out[0, :3], out2[0, :3])  # the past is unchanged
    assert not torch.allclose(
        out[0, 3:], out2[0, 3:]
    )  # token 3 onward can see the change


def test_attention_matches_torch_sdpa():
    # PyTorch's built-in attention does scores, scale, mask, softmax and the
    # weighted sum in one call. Ours must give the same answer.
    torch.manual_seed(0)
    cfg = GPT2Config()
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 5, 768)
    B, T, C = x.shape
    q, k, v = attn.c_attn(x).split(C, dim=-1)
    q, k, v = (
        t.view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2) for t in (q, k, v)
    )
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.c_proj(ref.transpose(1, 2).reshape(B, T, C))
    torch.testing.assert_close(attn(x), ref, atol=1e-5, rtol=1e-5)


def test_attention_parameter_count():
    # c_attn: 768*2304 + 2304, c_proj: 768*768 + 768
    n = sum(p.numel() for p in CausalSelfAttention(GPT2Config()).parameters())
    assert n == 2_362_368


def test_attention_mask_not_in_state_dict():
    # The mask is fixed, not learned, so it shouldn't be saved or loaded.
    assert "mask" not in CausalSelfAttention(GPT2Config()).state_dict()
