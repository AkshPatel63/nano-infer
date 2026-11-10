# nano-infer

An LLM inference engine built from scratch in PyTorch, focused on
speed *and* correctness. Work in progress.

## Roadmap

- [ ] GPT-2 from scratch, verified against Hugging Face
- [ ] Qwen3 0.6B support
- [ ] Benchmark + correctness harness
- [ ] KV cache, bf16, torch.compile
- [ ] Continuous batching
- [ ] Paged KV cache + prefix caching
- [ ] OpenAI-compatible server
- [ ] Custom Triton kernels

## Development

```bash
uv sync          # create .venv and install dependencies
uv run pytest    # run the tests
```

## Acknowledgements

Started from Sebastian Raschka's *Build a Large Language Model (From Scratch)*.
