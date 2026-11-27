"""Benchmark nano-infer generation: TTFT, total time and decode tok/s."""

import argparse
import statistics
import time

import torch

from nanoinfer.generate import generate
from nanoinfer.models.gpt2 import GPT2Model
from nanoinfer.tokenizer import GPT2Tokenizer
from nanoinfer.utils import get_device

PROMPT = "The quick brown fox jumps over the lazy dog. Once upon a time"


def sync(device):
    """Wait until the GPU has finished all queued work. Does nothing on CPU."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def measure(fn, device, warmup=2, repeats=5):
    """Run fn() `warmup` times untimed, then `repeats` times timed.

    Returns a list with the time of each timed run, in seconds.
    """
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        sync(device)  # kitchen empty before we start
        start = time.perf_counter()
        fn()
        sync(device)  # wait for the cooking to finish
        times.append(time.perf_counter() - start)
    return times


def main():
    parser = argparse.ArgumentParser(description="Time nano-infer's generate().")
    parser.add_argument(
        "-d", "--device", default="cpu", choices=["auto", "cpu", "mps", "cuda"]
    )
    parser.add_argument("-n", "--new-tokens", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    device = get_device(args.device)
    tok = GPT2Tokenizer()
    model = GPT2Model.from_pretrained().to(device)
    idx = torch.tensor([tok.encode(PROMPT)], device=device)  # (1, T)
    n = args.new_tokens

    print(
        f"device={device.type}  dtype=float32  batch=1  "
        f"prompt_len={idx.shape[1]}  new_tokens={n}"
    )

    ttft = measure(
        lambda: generate(model, idx, 1, eos_id=None), device, args.warmup, args.repeats
    )
    total = measure(
        lambda: generate(model, idx, n, eos_id=None), device, args.warmup, args.repeats
    )

    ttft_med = statistics.median(ttft)
    total_med = statistics.median(total)
    decode_tps = (n - 1) / (total_med - ttft_med)

    print("TTFT runs (ms): ", [round(t * 1000, 1) for t in ttft])
    print("total runs (s): ", [round(t, 3) for t in total])
    print(f"median TTFT:    {ttft_med * 1000:.1f} ms")
    print(f"median total:   {total_med:.3f} s")
    print(f"decode:         {decode_tps:.1f} tok/s")


if __name__ == "__main__":
    main()
