"""Small helpers shared across the engine."""

import torch

_DEVICES = ("cuda", "mps", "cpu")


def get_device(name="auto"):
    """Return the torch.device to run on.

    "auto" picks the best available: CUDA (NVIDIA GPU), then MPS (Apple GPU), then CPU.
    Or pass "cuda", "mps" or "cpu" to choose explicitly.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name not in _DEVICES:
        raise ValueError(f"unknown device {name!r}; choose from {_DEVICES} or 'auto'")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda was requested, but no NVIDIA GPU is available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("mps was requested, but the Apple GPU is not available")
    return torch.device(name)