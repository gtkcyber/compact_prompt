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
import re
import shutil
import subprocess
from collections import Counter
from typing import Callable, List, Optional, Tuple

from .hard_prompt import HardPromptResult

# A pluggable LLM is any callable: prompt -> completion text.
LLM = Callable[[str], str]

# --- structure regexes (ported from caveman's validate.py / compress.py) -----
_URL_RE = re.compile(r"https?://[^\s)]+")
_FENCE_OPEN_RE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})(.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
_OUTER_FENCE_RE = re.compile(r"\A\s*(`{3,}|~{3,})[^\n]*\n(.*)\n\1\s*\Z", re.DOTALL)
_FRONTMATTER_RE = re.compile(r"\A(---\r?\n.*?\r?\n---\r?\n)(.*)", re.DOTALL)


# --- extractors --------------------------------------------------------------
def _extract_headings(text: str) -> List[Tuple[str, str]]:
    return [(lvl, title.strip()) for lvl, title in _HEADING_RE.findall(text)]


def _extract_urls(text: str) -> set:
    return set(_URL_RE.findall(text))


def _extract_inline_codes(text: str) -> List[str]:
    without = re.sub(r"^```[\s\S]*?^```", "", text, flags=re.MULTILINE)
    without = re.sub(r"^~~~[\s\S]*?^~~~", "", without, flags=re.MULTILINE)
    return re.findall(r"`([^`]+)`", without)


def _extract_code_blocks(text: str) -> List[str]:
    """Line-based fenced code-block extractor (handles ``` and ~~~, any length)."""
    blocks: List[str] = []
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        m = _FENCE_OPEN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        fence_char, fence_len = m.group(2)[0], len(m.group(2))
        block_lines = [lines[i]]
        i += 1
        closed = False
        while i < n:
            close = _FENCE_OPEN_RE.match(lines[i])
            if (
                close
                and close.group(2)[0] == fence_char
                and len(close.group(2)) >= fence_len
                and close.group(3).strip() == ""
            ):
                block_lines.append(lines[i])
                closed = True
                i += 1
                break
            block_lines.append(lines[i])
            i += 1
        if closed:
            blocks.append("\n".join(block_lines))
    return blocks


def validate_structure(original: str, compressed: str) -> List[str]:
    """Return a list of structure-preservation errors (empty == valid).

    Mirrors caveman's hard checks: headings, fenced code blocks, URLs and
    inline code must be preserved exactly. Paths/bullets are warnings in the
    original and are not enforced here.
    """
    errors: List[str] = []
    h1, h2 = _extract_headings(original), _extract_headings(compressed)
    if len(h1) != len(h2):
        errors.append(f"Heading count mismatch: {len(h1)} vs {len(h2)}")
    if _extract_code_blocks(original) != _extract_code_blocks(compressed):
        errors.append("Code blocks not preserved exactly")
    u1, u2 = _extract_urls(original), _extract_urls(compressed)
    if u1 != u2:
        errors.append(f"URL mismatch: lost={u1 - u2}, added={u2 - u1}")
    c1, c2 = Counter(_extract_inline_codes(original)), Counter(_extract_inline_codes(compressed))
    if c1 != c2:
        lost = {k for k in c1 if c2.get(k, 0) < c1[k]}
        if lost:
            errors.append(f"Inline code lost: {lost}")
    return errors


def _split_frontmatter(text: str) -> Tuple[str, str]:
    m = _FRONTMATTER_RE.match(text)
    return (m.group(1), m.group(2)) if m else ("", text)


def _strip_outer_fence(text: str) -> str:
    m = _OUTER_FENCE_RE.match(text)
    return m.group(2) if m else text


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
                return _strip_outer_fence(msg.content[0].text.strip())
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
        return _strip_outer_fence(result.stdout.strip())

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
        frontmatter, body = _split_frontmatter(text)

        compressed_body = _strip_outer_fence(llm(_compress_prompt(body)).strip())
        errors = validate_structure(body, compressed_body)
        for _ in range(self.max_retries):
            if not errors:
                break
            compressed_body = _strip_outer_fence(
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
