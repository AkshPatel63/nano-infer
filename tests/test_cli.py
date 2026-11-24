"""Tests for the command-line interface."""

import pytest

from nanoinfer.cli import main, parse_args


def test_defaults():
    args = parse_args(["hello"])
    assert args.prompt == "hello"
    assert args.max_new_tokens == 50
    assert args.temperature == 0.0
    assert args.top_k is None and args.top_p is None and args.seed is None
    assert args.device == "auto"


def test_short_options():
    args = parse_args(["hi", "-n", "10", "-t", "0.8", "-k", "40", "--top-p", "0.9", "-s", "1", "-d", "cpu"])
    assert (args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.seed, args.device) == (
        10, 0.8, 40, 0.9, 1, "cpu"
    )


def test_top_p_without_temperature_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["hi", "--top-p", "0.9"])


def test_prompt_is_required():
    with pytest.raises(SystemExit):
        parse_args([])


@pytest.mark.slow
def test_cli_generates_text(capsys):
    main(["The cat sat on the", "-n", "5", "-d", "cpu"])
    printed = capsys.readouterr().out
    assert printed.startswith("The cat sat on the")