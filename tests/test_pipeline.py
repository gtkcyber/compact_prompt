"""Tests for the unified CompactPrompt pipeline / headline API."""

from compactprompt import CompactPrompt, compact
from compactprompt.pipeline import CompactResult


PROMPT = (
    "Please could you kindly summarize the quarterly report. "
    "The quarterly report covers revenue. The quarterly report covers costs. "
    "The quarterly report covers profit and the quarterly report covers risk."
)


def test_compact_classmethod_returns_result():
    res = CompactPrompt.compact(PROMPT)
    assert isinstance(res, CompactResult)
    assert res.text
    assert res.original == PROMPT


def test_compact_reduces_tokens():
    res = CompactPrompt.compact(PROMPT, ratio=0.5)
    assert res.tokens_after < res.tokens_before
    assert res.ratio > 1.0


def test_top_level_compact_helper():
    res = compact(PROMPT, ratio=0.3)
    assert isinstance(res, CompactResult)


def test_restore_reverses_abbreviation_when_pruning_off():
    # With pruning disabled, only the lossless n-gram step runs, so restore()
    # must reproduce the input exactly.
    res = CompactPrompt.compact(PROMPT, prune=False, abbreviate=True)
    assert res.restore() == PROMPT


def test_abbreviation_produces_dictionary():
    res = CompactPrompt.compact(PROMPT, prune=False, abbreviate=True, ngram=3)
    # A repeated trigram about the quarterly report should be captured.
    assert any("quarterly report" in v for v in res.dictionary.values())
    # And it must shrink (or at worst not grow) the token count.
    assert res.tokens_after <= res.tokens_before


def test_disable_both_is_noop():
    res = CompactPrompt.compact(PROMPT, prune=False, abbreviate=False)
    assert res.text == PROMPT
    assert res.tokens_after == res.tokens_before


def test_steps_recorded():
    res = CompactPrompt.compact(PROMPT, prune=True, abbreviate=True)
    assert "hard_prompt" in res.steps
    assert "ngram_abbreviation" in res.steps


def test_instance_reuse():
    cp = CompactPrompt(top_k=20)
    r1 = cp.run(PROMPT)
    r2 = cp.run("another simpler prompt with no repetition at all here")
    assert isinstance(r1, CompactResult)
    assert isinstance(r2, CompactResult)


def test_str_dunder():
    res = CompactPrompt.compact(PROMPT)
    assert str(res) == res.text


def test_savings_in_range():
    res = CompactPrompt.compact(PROMPT)
    assert 0.0 <= res.savings <= 1.0


def test_budget_overrides_ratio():
    res = CompactPrompt.compact(PROMPT, budget=10, abbreviate=False)
    assert res.tokens_after <= 12  # small tokenizer slack
