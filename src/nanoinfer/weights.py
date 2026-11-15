"""Download and read model checkpoints from the Hugging Face Hub."""

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def load_hf_checkpoint(repo_id, filename="model.safetensors"):
    """Download a safetensors checkpoint (cached after the first time) and return name -> tensor."""
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_file(path)