import tiktoken


class GPT2Tokenizer:
    """GPT-2's byte-level BPE tokenizer, backed by tiktoken."""

    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """convert the text into list of token IDs"""
        if allow_special:
            return self._enc.encode(text, allowed_special={"<|endoftext|>"})
        return self._enc.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to text."""
        return self._enc.decode(ids)

    @property
    def eot_id(self) -> int:
        """ID of the end-of-text token."""
        return self._enc.eot_token

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return self._enc.n_vocab


if __name__ == "__main__":
    tok = GPT2Tokenizer()
    ids = tok.encode("Hello, world")
    print(ids, repr(tok.decode(ids)))
