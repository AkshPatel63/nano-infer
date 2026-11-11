import pytest

from nanoinfer.tokenizer import GPT2Tokenizer


@pytest.fixture(scope="module")
def tok():
    return GPT2Tokenizer()


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world",
        "The cat sat on the mat.",
        "  leading spaces and\nnew lines\n",
        "emoji 🚀 and accents: café",
        "",
    ],
)
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_known_ids(tok):
    assert tok.encode("Hello, world") == [15496, 11, 995]


def test_constants(tok):
    assert tok.eot_id == 50256
    assert tok.vocab_size == 50257


def test_leading_space_matters(tok):
    assert tok.encode("world") != tok.encode(" world")


def test_special_token_handling(tok):
    text = "<|endoftext|>"
    assert tok.encode(text, allow_special=True) == [50256]
    assert tok.encode(text) != [50256]  # treated as plain text by default
