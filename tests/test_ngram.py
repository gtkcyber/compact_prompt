"""Tests for Textual N-gram Abbreviation (lossless, pure-Python)."""

import pytest

from compactprompt import NgramAbbreviator, abbreviate, restore


REPETITIVE = (
    "net income increased. net income grew. operating cash flow rose. "
    "operating cash flow improved. net income is key. operating cash flow matters."
)


def test_round_trip_is_lossless():
    abbr = NgramAbbreviator(n=2, top_k=50).compress(REPETITIVE)
    assert abbr.restore() == REPETITIVE


def test_round_trip_helper_functions():
    a = abbreviate(REPETITIVE, n=2, top_k=50)
    assert restore(a.text, a.dictionary) == REPETITIVE


def test_frequent_pattern_is_abbreviated():
    # With the token-savings guard off, frequency alone drives selection, so the
    # most frequent bigram ("net income", 3x) must be captured.
    abbr = NgramAbbreviator(n=2, top_k=50, require_savings=False).compress(REPETITIVE)
    assert "net income" in abbr.dictionary.values()
    assert abbr.restore() == REPETITIVE


def test_savings_guard_skips_non_beneficial():
    # "net income" is the same token cost as its placeholder under cl100k, so
    # with the guard on it is skipped rather than abbreviated to no benefit.
    abbr = NgramAbbreviator(n=2, top_k=50, require_savings=True).compress(REPETITIVE)
    assert abbr.restore() == REPETITIVE
    # Never longer than the input.
    from compactprompt import count_tokens

    assert count_tokens(abbr.text) <= count_tokens(REPETITIVE)


def test_abbreviation_shortens_text():
    abbr = NgramAbbreviator(n=2, top_k=50).compress(REPETITIVE)
    assert len(abbr.text) < len(REPETITIVE)


def test_min_count_threshold_respected():
    # With min_count high, nothing qualifies and text is unchanged.
    abbr = NgramAbbreviator(n=2, top_k=50, min_count=100).compress(REPETITIVE)
    assert abbr.text == REPETITIVE
    assert abbr.dictionary == {}


def test_no_patterns_when_unique():
    text = "every single word here is completely distinct from all others around"
    abbr = NgramAbbreviator(n=2, top_k=50).compress(text)
    assert abbr.text == text
    assert abbr.dictionary == {}


def test_top_k_limits_dictionary_size():
    abbr = NgramAbbreviator(n=2, top_k=1).compress(REPETITIVE)
    assert len(abbr.dictionary) <= 1


def test_placeholders_absent_from_original():
    abbr = NgramAbbreviator(n=2, top_k=50).compress(REPETITIVE)
    for ph in abbr.dictionary:
        assert ph not in REPETITIVE


def test_empty_input():
    abbr = NgramAbbreviator().compress("")
    assert abbr.text == ""
    assert abbr.restore() == ""


def test_trigrams():
    text = "the quick brown fox. the quick brown dog. the quick brown cat."
    abbr = NgramAbbreviator(n=3, top_k=10).compress(text)
    assert "the quick brown" in abbr.dictionary.values()
    assert abbr.restore() == text


def test_invalid_n_raises():
    with pytest.raises(ValueError):
        NgramAbbreviator(n=0)


def test_unicode_preserved_round_trip():
    text = "café déjà vu. café déjà vu again. café déjà vu thrice."
    abbr = NgramAbbreviator(n=2, top_k=10).compress(text)
    assert abbr.restore() == text
