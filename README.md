# nano-infer

[![tests](https://github.com/AkshPatel63/nano-infer/actions/workflows/tests.yml/badge.svg)](https://github.com/AkshPatel63/nano-infer/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An LLM inference engine built from scratch in PyTorch, focused on **speed and correctness**: every step is verified against Hugging Face `transformers`.

**Status:** GPT-2 (124M) runs end to end, matching Hugging Face. Next: Qwen3 0.6B, then making it fast.

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

- ✅ **GPT-2 from scratch, verified against Hugging Face**
  - ✅ Model, weight loading, logits match
  - ✅ Greedy generation matching Hugging Face, sampling (temperature, top-k, top-p), seeds
  - ✅ CLI and automatic device selection
- ⬜ Qwen3 0.6B support (RoPE, RMSNorm, grouped-query attention, SwiGLU, chat template)
- ⬜ Benchmarks + correctness harness (time to first token, latency percentiles, tokens/sec, % of hardware roofline; KL divergence and top-1 agreement across fp32/bf16/fp16)
- ⬜ KV cache, bf16, `torch.compile`
- ⬜ Continuous batching
- ⬜ Paged KV cache + prefix caching
- ⬜ OpenAI-compatible server
- ⬜ Custom Triton kernels

## Acknowledgements

Started from Sebastian Raschka's [*Build a Large Language Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch). Also inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## License

[MIT](LICENSE)
