"""Tests for the LLMLingua pruning-engine integration.

These avoid downloading any LLMLingua model: the pipeline-injection path is
tested with a fake pruner, and the ratio/budget -> LLMLingua-param mapping is
tested by stubbing the underlying compressor.
"""

# Tests deliberately set the private ``_compressor`` to stub out model loading.
# pylint: disable=protected-access

import sys

from compactprompt import CompactPrompt, LLMLinguaCompressor
from compactprompt.hard_prompt import HardPromptResult


class FakePruner:
    """Minimal pruning engine: keeps the first half of the words."""

    def __init__(self):
        self.called_with = None

    def compress(self, text, ratio=0.5, budget=None):
        self.called_with = {"ratio": ratio, "budget": budget}
        words = text.split()
        kept = " ".join(words[: max(1, len(words) // 2)])
        return HardPromptResult(text, kept, len(words), len(kept.split()))


def test_pipeline_uses_injected_pruner():
    fake = FakePruner()
    res = CompactPrompt(pruner=fake).run("alpha beta gamma delta epsilon zeta", ratio=0.5)
    assert fake.called_with == {"ratio": 0.5, "budget": None}
    assert res.text == "alpha beta gamma"
    assert "hard_prompt" in res.steps


def test_compact_classmethod_accepts_pruner():
    fake = FakePruner()
    res = CompactPrompt.compact("one two three four", pruner=fake, abbreviate=False)
    assert res.text == "one two"


def test_invalid_engine_raises():
    import pytest

    with pytest.raises(ValueError):
        CompactPrompt.compact("hello world", engine="bogus")


def test_construction_is_lazy():
    # Building the adapter must not import llmlingua (cheap, import-safe).
    sys.modules.pop("llmlingua", None)
    LLMLinguaCompressor(model_name="dummy")
    assert "llmlingua" not in sys.modules


class _StubCompressor:
    """Stand-in for llmlingua.PromptCompressor that records call kwargs."""

    def __init__(self):
        self.kwargs = None
        self.context = None

    def compress_prompt(self, context, **kwargs):
        self.context = context
        self.kwargs = kwargs
        return {
            "compressed_prompt": "compressed text here",
            "origin_tokens": 100,
            "compressed_tokens": 3,
        }


def test_ratio_maps_to_keep_rate():
    comp = LLMLinguaCompressor()
    comp._compressor = _StubCompressor()  # bypass model load
    comp.compress("some long prompt text", ratio=0.4)
    # ratio=0.4 removed -> keep rate 0.6
    assert comp._compressor.kwargs["rate"] == 0.6
    assert "target_token" not in comp._compressor.kwargs
    assert comp._compressor.context == ["some long prompt text"]


def test_budget_maps_to_target_token():
    comp = LLMLinguaCompressor()
    comp._compressor = _StubCompressor()
    comp.compress("some long prompt text", budget=25)
    assert comp._compressor.kwargs["target_token"] == 25
    assert "rate" not in comp._compressor.kwargs


def test_returns_hard_prompt_result():
    comp = LLMLinguaCompressor()
    comp._compressor = _StubCompressor()
    result = comp.compress("some long prompt text", ratio=0.5)
    assert isinstance(result, HardPromptResult)
    assert result.compressed == "compressed text here"
    assert result.tokens_after < result.tokens_before


def test_pipeline_with_stubbed_llmlingua_then_abbreviation():
    comp = LLMLinguaCompressor()
    comp._compressor = _StubCompressor()
    res = CompactPrompt(pruner=comp).run("anything at all", ratio=0.5, abbreviate=True)
    assert res.text  # ran end to end
    assert "hard_prompt" in res.steps and "ngram_abbreviation" in res.steps
