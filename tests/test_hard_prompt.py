"""Tests for Hard Prompt Compression.

These run with no dynamic scorer and no spaCy (static-only, word-unit fallback),
so they exercise the always-available path. A fake scorer test verifies the
dynamic plumbing without needing torch.
"""

from compactprompt import HardPromptCompressor
from compactprompt.hard_prompt import HardPromptResult


VERBOSE = (
    "Please could you very kindly go ahead and provide a concise summary "
    "of the quarterly financial report for the board of directors today."
)


def test_compress_reduces_tokens():
    res = HardPromptCompressor(use_phrases=False).compress(VERBOSE, ratio=0.5)
    assert isinstance(res, HardPromptResult)
    assert res.tokens_after < res.tokens_before


def test_compress_respects_budget():
    res = HardPromptCompressor(use_phrases=False).compress(VERBOSE, budget=8)
    assert res.tokens_after <= 8 + 2  # allow small tokenizer slack


def test_ratio_zero_keeps_everything():
    res = HardPromptCompressor(use_phrases=False).compress(VERBOSE, ratio=0.0)
    assert res.tokens_after == res.tokens_before


def test_savings_and_ratio_consistent():
    res = HardPromptCompressor(use_phrases=False).compress(VERBOSE, ratio=0.5)
    assert 0.0 <= res.savings <= 1.0
    assert res.ratio >= 1.0


def test_empty_input_safe():
    res = HardPromptCompressor(use_phrases=False).compress("   ")
    assert res.compressed.strip() == ""


def test_str_returns_compressed():
    res = HardPromptCompressor(use_phrases=False).compress(VERBOSE, ratio=0.4)
    assert str(res) == res.compressed


def test_dynamic_scorer_is_used():
    """A fake scorer that makes the word 'secret' extremely informative should
    cause 'secret' to survive aggressive pruning."""
    text = "the the the the secret the the the the the the the the the"

    def fake_scorer(t):
        out = []
        for m_start, word in _iter_words(t):
            bits = 50.0 if word == "secret" else 0.01
            out.append((word, m_start, m_start + len(word), bits))
        return out

    res = HardPromptCompressor(scorer=fake_scorer, use_phrases=False).compress(
        text, ratio=0.7
    )
    assert "secret" in res.compressed
    assert res.tokens_after < res.tokens_before


def _iter_words(text):
    import re

    for m in re.finditer(r"\w+", text):
        yield m.start(), m.group()
