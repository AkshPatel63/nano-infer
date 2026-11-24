"""Tests for device selection. monkeypatch pretends to be machines with or without GPUs."""

import pytest
import torch

from nanoinfer.utils import get_device


def fake_hardware(monkeypatch, cuda, mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_auto_prefers_cuda(monkeypatch):
    fake_hardware(monkeypatch, cuda=True, mps=True)
    assert get_device().type == "cuda"


def test_auto_uses_mps_without_cuda(monkeypatch):
    fake_hardware(monkeypatch, cuda=False, mps=True)
    assert get_device().type == "mps"


def test_auto_falls_back_to_cpu(monkeypatch):
    fake_hardware(monkeypatch, cuda=False, mps=False)
    assert get_device().type == "cpu"


def test_explicit_cpu_always_works():
    assert get_device("cpu").type == "cpu"


def test_unavailable_device_raises(monkeypatch):
    fake_hardware(monkeypatch, cuda=False, mps=False)
    with pytest.raises(ValueError):
        get_device("cuda")


def test_unknown_device_name_raises():
    with pytest.raises(ValueError):
        get_device("gpu")