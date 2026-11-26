# nano-infer

![tests](https://github.com/AkshPatel63/nano-infer/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

An LLM inference engine built from scratch in PyTorch, focused on **speed and correctness**: every step is verified against Hugging Face `transformers`.

**Status: GPT-2 (124M) runs end to end, matching Hugging Face. Next: benchmarks and a KV cache, then Qwen3 0.6B.**

## Results

OpenAI's GPT-2 weights in nano-infer vs. Hugging Face, on five kinds of text (CPU, float32):

| | |
|---|---|
| Max logit difference | ≤ 2.5e-4 (about one part in a million) |
| Greedy generation, 50 tokens | Identical, token for token, on all 5 prompts |

## Quickstart

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AkshPatel63/nano-infer.git
cd nano-infer
uv sync

uv run nano-infer "The cat sat on the"                                # greedy
uv run nano-infer "Once upon a time," -n 60 -t 0.8 --top-p 0.9 -s 1   # sampling
uv run nano-infer --help                                              # all options
```

The first run downloads GPT-2's weights (about 550 MB) and caches them.

## Tests

```bash
uv run pytest            # fast tests, run in CI on every push
uv run pytest -m slow    # full comparisons against Hugging Face's GPT-2
```

## Roadmap

## Roadmap

- ✅ **GPT-2 from scratch, verified against Hugging Face** (v0.1.0)
  - ✅ Model, weight loading, logits match
  - ✅ Greedy generation matching Hugging Face, sampling (temperature, top-k, top-p), seeds
  - ✅ CLI and automatic device selection
- ⬜ Benchmarks + KV cache on GPT-2 (time to first token, per-token latency, tokens/sec, memory)
- ⬜ Qwen3 0.6B support (RoPE, RMSNorm, grouped-query attention, SwiGLU, chat template)
- ⬜ Single-request speed: bf16, `torch.compile`, correctness harness (KL divergence, top-1 agreement), GPU benchmarks
- ⬜ Continuous batching
- ⬜ Paged KV cache + prefix caching
- ⬜ OpenAI-compatible server
- ⬜ Custom Triton kernels

## Acknowledgements

Started from Sebastian Raschka's *[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)*. Also inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## License

[MIT](LICENSE)
