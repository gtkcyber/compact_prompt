"""Tests for the Caveman LLM-based compression engine.

The LLM is injected as a fake callable, so these are deterministic and make no
API calls or model downloads.
"""

# Fake LLM callables intentionally ignore their prompt argument.
# pylint: disable=unused-argument

import sys

import pytest

from compactprompt import CavemanCompressor, CompactPrompt
from compactprompt.caveman import validate_structure
from compactprompt.hard_prompt import HardPromptResult

DOC = (
    "# Title\n\n"
    "The reason that your component keeps re-rendering on every single update "
    "is that a brand new object reference is created each render.\n\n"
    "See https://example.com/docs for details. Run `npm install` first.\n\n"
    "```python\nx = 1\n```\n"
)

# A terse caveman rewrite that preserves heading, URL, inline code, code block.
CAVEMAN = (
    "# Title\n\n"
    "New object ref each render. Wrap in useMemo.\n\n"
    "See https://example.com/docs for details. Run `npm install` first.\n\n"
    "```python\nx = 1\n```\n"
)


def fake_llm(prompt):
    return CAVEMAN


def test_compress_returns_hard_prompt_result():
    res = CavemanCompressor(llm=fake_llm).compress(DOC)
    assert isinstance(res, HardPromptResult)
    assert res.tokens_after < res.tokens_before
    assert "useMemo" in res.compressed


def test_structure_preserved_passes_validation():
    assert not validate_structure(DOC, CAVEMAN)


def test_validation_flags_dropped_url():
    broken = CAVEMAN.replace("See https://example.com/docs for details. ", "")
    errors = validate_structure(DOC, broken)
    assert any("URL" in e for e in errors)


def test_validation_flags_changed_code_block():
    broken = CAVEMAN.replace("x = 1", "x = 2")
    errors = validate_structure(DOC, broken)
    assert any("Code blocks" in e for e in errors)


def test_validation_flags_lost_inline_code():
    broken = CAVEMAN.replace("`npm install`", "npm install")
    errors = validate_structure(DOC, broken)
    assert any("Inline code" in e for e in errors)


def test_fix_retry_recovers():
    """First call drops a URL; the fix call returns a valid version."""
    calls = {"n": 0}

    def flaky_llm(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return CAVEMAN.replace("See https://example.com/docs for details. ", "")
        return CAVEMAN

    res = CavemanCompressor(llm=flaky_llm, max_retries=2).compress(DOC)
    assert calls["n"] == 2  # one compress + one fix
    assert "https://example.com/docs" in res.compressed


def test_raises_if_structure_unrecoverable():
    def bad_llm(prompt):
        return "# Title\n\nlost everything"  # drops URL + code permanently

    with pytest.raises(ValueError):
        CavemanCompressor(llm=bad_llm, max_retries=1).compress(DOC)


def test_strips_outer_fence_wrapper():
    def wrapping_llm(prompt):
        return "```markdown\n" + CAVEMAN + "\n```"

    res = CavemanCompressor(llm=wrapping_llm).compress(DOC)
    assert not res.compressed.lstrip().startswith("```markdown")


def test_frontmatter_preserved_verbatim():
    doc = "---\ntitle: x\n---\n# H\n\nlots of filler words here to compress\n"

    def llm(prompt):
        return "# H\n\nfiller words\n"

    res = CavemanCompressor(llm=llm).compress(doc)
    assert res.compressed.startswith("---\ntitle: x\n---\n")


def test_ratio_and_budget_ignored():
    # Accepted for interface compatibility; result is the same regardless.
    a = CavemanCompressor(llm=fake_llm).compress(DOC, ratio=0.1)
    b = CavemanCompressor(llm=fake_llm).compress(DOC, budget=5)
    assert a.compressed == b.compressed


def test_pipeline_injection():
    res = CompactPrompt(pruner=CavemanCompressor(llm=fake_llm)).run(DOC, abbreviate=False)
    assert "hard_prompt" in res.steps
    assert "useMemo" in res.text


def test_empty_input_safe():
    res = CavemanCompressor(llm=fake_llm).compress("   ")
    assert res.compressed.strip() == ""


def test_construction_is_lazy():
    # Building the engine must not import anthropic.
    sys.modules.pop("anthropic", None)
    CavemanCompressor()
    assert "anthropic" not in sys.modules
