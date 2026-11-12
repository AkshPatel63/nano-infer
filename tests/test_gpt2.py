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
