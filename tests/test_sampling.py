"""Tests for sampling, using the small example from the theory:
floor, bed, couch, ground, mat, banana."""

import pytest
import torch

from nanoinfer.sampling import apply_temperature, sample_next_token, top_k_filter, top_p_filter

LOGITS = torch.tensor([[3.0, 2.8, 2.6, 2.5, 1.5, -2.0]])


def kept(logits):
    """Which tokens survived filtering (weren't set to -inf)."""
    return torch.isfinite(logits[0]).tolist()


def test_temperature_sharpens_and_flattens():
    p_normal = torch.softmax(LOGITS, dim=-1)[0, 0]
    p_cold = torch.softmax(apply_temperature(LOGITS, 0.5), dim=-1)[0, 0]
    p_hot = torch.softmax(apply_temperature(LOGITS, 2.0), dim=-1)[0, 0]
    assert p_cold > p_normal > p_hot          # 0.39 > 0.30 > 0.25 for "floor"


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        apply_temperature(LOGITS, 0)


def test_top_k_keeps_exactly_k():
    assert kept(top_k_filter(LOGITS, 3)) == [True, True, True, False, False, False]


def test_top_k_none_keeps_everything():
    assert torch.equal(top_k_filter(LOGITS, None), LOGITS)


def test_top_p_keeps_smallest_set_reaching_p():
    # running totals 0.30, 0.55, 0.75, 0.93: "ground" crosses 0.8, so it's the last one kept
    assert kept(top_p_filter(LOGITS, 0.8)) == [True, True, True, True, False, False]


def test_top_p_keeps_only_the_top_token_when_confident():
    confident = torch.tensor([[10.0, 0.0, 0.0]])   # top token has about 0.9999
    assert kept(top_p_filter(confident, 0.9)) == [True, False, False]


def test_top_p_works_on_unsorted_input():
    perm = torch.tensor([3, 0, 5, 1, 4, 2])        # shuffle the vocabulary order
    shuffled = top_p_filter(LOGITS[:, perm], 0.8)
    expected = top_p_filter(LOGITS, 0.8)[:, perm]
    assert torch.equal(torch.isfinite(shuffled), torch.isfinite(expected))


def test_sampling_never_picks_filtered_tokens():
    g = torch.Generator().manual_seed(0)
    picks = sample_next_token(LOGITS.repeat(500, 1), top_k=3, generator=g)
    assert set(picks.flatten().tolist()) <= {0, 1, 2}


def test_sampling_frequencies_match_probabilities():
    g = torch.Generator().manual_seed(0)
    picks = sample_next_token(LOGITS.repeat(5000, 1), generator=g)
    floor_share = (picks == 0).float().mean().item()
    assert abs(floor_share - 0.30) < 0.03      # "floor" has probability 0.30


def test_same_seed_same_picks_different_seed_different_picks():
    rows = LOGITS.repeat(100, 1)
    a = sample_next_token(rows, generator=torch.Generator().manual_seed(1))
    b = sample_next_token(rows, generator=torch.Generator().manual_seed(1))
    c = sample_next_token(rows, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)