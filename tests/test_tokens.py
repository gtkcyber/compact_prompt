"""Tests for token counting utilities."""

from compactprompt.tokens import count_tokens, simple_word_tokens


def test_count_empty():
    assert count_tokens("") == 0


def test_count_nonempty_positive():
    assert count_tokens("hello world") >= 1


def test_count_monotonic_with_length():
    short = count_tokens("the cat")
    long = count_tokens("the cat sat on the mat in the sun")
    assert long > short


def test_simple_word_tokens_splits_punctuation():
    toks = simple_word_tokens("Hello, world!")
    assert "Hello" in toks
    assert "," in toks
    assert "world" in toks
    assert "!" in toks
