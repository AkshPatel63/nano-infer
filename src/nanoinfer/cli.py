"""Command-line interface: nano-infer "your prompt" [options]."""

import argparse

import torch

from nanoinfer.generate import generate
from nanoinfer.models.gpt2 import GPT2Model
from nanoinfer.tokenizer import GPT2Tokenizer
from nanoinfer.utils import get_device


def parse_args(argv=None):
    """Read the command-line options. argv=None means 'use the real command line'."""
    parser = argparse.ArgumentParser(
        prog="nano-infer",
        description="Generate text with GPT-2, running on nano-infer.",
    )
    parser.add_argument("prompt", help="the text to continue")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=50,
                        help="how many tokens to generate (default: 50)")
    parser.add_argument("-t", "--temperature", type=float, default=0.0,
                        help="0 = greedy; above 0 = sampling (default: 0)")
    parser.add_argument("-k", "--top-k", type=int, default=None,
                        help="sample only from the k most likely tokens")
    parser.add_argument("--top-p", type=float, default=None,
                        help="sample only from the smallest set of tokens reaching probability p")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="random seed, for reproducible sampling")
    parser.add_argument("-d", "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="where to run (default: auto)")
    args = parser.parse_args(argv)

    if args.temperature == 0 and (args.top_k is not None or args.top_p is not None):
        parser.error("--top-k and --top-p only apply when sampling; set --temperature above 0")
    return args


def main(argv=None):
    """Load GPT-2, generate from the prompt, print the result."""
    args = parse_args(argv)
    device = get_device(args.device)

    tok = GPT2Tokenizer()
    model = GPT2Model.from_pretrained().to(device)
    idx = torch.tensor([tok.encode(args.prompt)], device=device)

    out = generate(
        model, idx, args.max_new_tokens, eos_id=tok.eot_id,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()