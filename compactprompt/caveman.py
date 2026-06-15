"""Caveman-compress backend for prompt compression.

An LLM-based compression engine that rewrites natural-language text into terse
"caveman speak" while preserving structure exactly — code blocks, inline code,
URLs, headings and file paths are kept verbatim, and the result is validated
(with a fix-retry loop) so nothing structural is lost.

This is a port of the ``caveman-compress`` skill from the **Caveman** project
by Julius Brussee (MIT License, Copyright (c) 2026 Julius Brussee):
https://github.com/JuliusBrussee/caveman — adapted to operate on strings and to
take a *pluggable* LLM, so it slots into compactprompt's swappable pruning-engine
interface (``compress(text, ratio=, budget=) -> HardPromptResult``).

Unlike the built-in / LLMLingua engines (which drop tokens to a target ratio),
caveman rewrites prose, so ``ratio``/``budget`` are accepted for interface
compatibility but not used — caveman compresses to its own degree.

Usage::

    from compactprompt import CompactPrompt
    from compactprompt.caveman import CavemanCompressor

    # Pluggable LLM: any callable str(prompt) -> str(response)
    engine = CavemanCompressor(llm=my_llm)
    result = CompactPrompt.compact(prompt, pruner=engine)

    # Or the shortcut (defaults to Anthropic SDK / `claude` CLI):
    result = CompactPrompt.compact(prompt, engine="caveman")
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable, List, Optional

from .hard_prompt import HardPromptResult
from .markdown import split_frontmatter, strip_outer_fence, validate_structure

# A pluggable LLM is any callable: prompt -> completion text.
LLM = Callable[[str], str]

# Re-exported so existing imports (compactprompt.caveman.validate_structure) work.
__all__ = ["CavemanCompressor", "default_anthropic_llm", "validate_structure", "LLM"]


# --- prompts (adapted from caveman's compress.py) ----------------------------
def _compress_prompt(text: str) -> str:
    return f"""Compress this markdown into caveman format: drop filler words, \
keep substance, use terse fragments.

STRICT RULES:
- Do NOT modify anything inside ``` code blocks
- Do NOT modify anything inside inline backticks
- Preserve ALL URLs exactly
- Preserve ALL headings exactly
- Preserve file paths and commands
- Only compress natural language
- Return ONLY the compressed markdown body — do not wrap it in an outer fence.

TEXT:
{text}
"""


def _fix_prompt(original: str, compressed: str, errors: List[str]) -> str:
    errors_str = "\n".join(f"- {e}" for e in errors)
    return f"""You are fixing a caveman-compressed markdown file. Fix ONLY the \
listed validation errors; leave everything else exactly as-is. Preserve caveman \
style in untouched sections.

ERRORS TO FIX:
{errors_str}

ORIGINAL (reference only):
{original}

COMPRESSED (fix this):
{compressed}

Return ONLY the fixed compressed file. No explanation.
"""


def default_anthropic_llm(model: Optional[str] = None) -> LLM:
    """Build the default LLM caller: Anthropic SDK if a key is set, else `claude` CLI.

    Mirrors Caveman's behavior. Prefer passing your own ``llm`` callable; this is
    a convenience default for the ``engine="caveman"`` shortcut.
    """
    model = model or os.environ.get("CAVEMAN_MODEL", "claude-sonnet-4-6")

    def call(prompt: str) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=model,
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}],
                )
                return strip_outer_fence(msg.content[0].text.strip())
            except ImportError:
                pass  # fall back to CLI
        claude_bin = shutil.which("claude")
        if not claude_bin:
            raise ImportError(
                "Caveman's default LLM needs either ANTHROPIC_API_KEY + the "
                "'anthropic' package (pip install 'compactprompt[caveman]'), or "
                "the `claude` CLI on PATH. Or pass your own llm= callable."
            )
        result = subprocess.run(
            [claude_bin, "--print"],
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
        return strip_outer_fence(result.stdout.strip())

    return call


class CavemanCompressor:
    """Caveman LLM-based compression as a compactprompt pruning engine.

    Args:
        llm: Pluggable callable ``prompt -> completion``. Defaults to
            :func:`default_anthropic_llm` (Anthropic SDK or ``claude`` CLI).
        model: Model name for the default LLM caller (ignored if ``llm`` given).
        max_retries: Validation fix-retry attempts (caveman default 2).
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        model: Optional[str] = None,
        max_retries: int = 2,
    ):
        self._llm = llm
        self._model = model
        self.max_retries = max_retries

    def _get_llm(self) -> LLM:
        if self._llm is None:
            self._llm = default_anthropic_llm(self._model)
        return self._llm

    def compress(
        self,
        text: str,
        ratio: float = 0.5,  # accepted for interface compatibility; unused
        budget: Optional[int] = None,  # accepted for interface compatibility; unused
    ) -> HardPromptResult:
        """Compress ``text`` into caveman style, preserving structure.

        ``ratio``/``budget`` are ignored — caveman rewrites prose to its own
        degree rather than to a token target.

        Returns:
            A :class:`HardPromptResult`. Raises ``ValueError`` if the LLM cannot
            produce a structure-valid compression within ``max_retries``.
        """
        del ratio, budget  # not used by caveman
        if not text.strip():
            return HardPromptResult.from_texts(text, text)

        llm = self._get_llm()
        frontmatter, body = split_frontmatter(text)

        compressed_body = strip_outer_fence(llm(_compress_prompt(body)).strip())
        errors = validate_structure(body, compressed_body)
        for _ in range(self.max_retries):
            if not errors:
                break
            compressed_body = strip_outer_fence(
                llm(_fix_prompt(body, compressed_body, errors)).strip()
            )
            errors = validate_structure(body, compressed_body)
        if errors:
            raise ValueError(
                "Caveman compression could not preserve structure after "
                f"{self.max_retries} fix attempts: {errors}"
            )

        compressed = frontmatter + compressed_body
        return HardPromptResult.from_texts(text, compressed)
