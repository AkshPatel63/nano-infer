import torch

import nanoinfer


def test_package_imports():
    assert nanoinfer.__version__


def test_torch_runs_on_cpu():
    x = torch.ones(2, 3)
    assert (x @ x.T).sum().item() == 12
