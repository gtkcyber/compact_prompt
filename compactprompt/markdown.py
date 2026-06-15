"""Shared markdown structure utilities.

These helpers identify and preserve the structural parts of markdown — fenced
code blocks, inline code, URLs, headings, and YAML frontmatter — so that
compaction never silently corrupts a document. They are used both by the
:mod:`compactprompt.caveman` engine and by the file layer
(:mod:`compactprompt.files`).

The structure-extraction and validation logic is adapted from the Caveman
project by Julius Brussee (MIT); see ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

_URL_RE = re.compile(r"https?://[^\s)]+")
_FENCE_OPEN_RE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})(.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
_OUTER_FENCE_RE = re.compile(r"\A\s*(`{3,}|~{3,})[^\n]*\n(.*)\n\1\s*\Z", re.DOTALL)
_FRONTMATTER_RE = re.compile(r"\A(---\r?\n.*?\r?\n---\r?\n)(.*)", re.DOTALL)


# --- extractors --------------------------------------------------------------
def extract_headings(text: str) -> List[Tuple[str, str]]:
    """Return ``(level, title)`` pairs for every ATX heading."""
    return [(lvl, title.strip()) for lvl, title in _HEADING_RE.findall(text)]


def extract_urls(text: str) -> set:
    """Return the set of ``http(s)`` URLs in ``text``."""
    return set(_URL_RE.findall(text))


def extract_inline_codes(text: str) -> List[str]:
    """Return inline-code spans, ignoring anything inside fenced code blocks."""
    without = re.sub(r"^```[\s\S]*?^```", "", text, flags=re.MULTILINE)
    without = re.sub(r"^~~~[\s\S]*?^~~~", "", without, flags=re.MULTILINE)
    return re.findall(r"`([^`]+)`", without)


def extract_code_blocks(text: str) -> List[str]:
    """Return fenced code blocks (handles ``` and ~~~ fences of any length)."""
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


def prose_segments(body: str) -> List[Tuple[str, str]]:
    """Split ``body`` into ordered ``("prose"|"code", text)`` segments.

    Fenced code blocks become ``"code"`` segments; everything else is
    ``"prose"``. Joining all segment texts with ``"\\n"`` reproduces ``body``
    exactly, so a caller can compact only the prose segments and reassemble
    losslessly. Unclosed fences are treated as prose (malformed markdown).
    """
    lines = body.split("\n")
    segments: List[Tuple[str, str]] = []
    buf: List[str] = []
    i, n = 0, len(lines)

    def flush_prose() -> None:
        if buf:
            segments.append(("prose", "\n".join(buf)))
            buf.clear()

    while i < n:
        m = _FENCE_OPEN_RE.match(lines[i])
        if not m:
            buf.append(lines[i])
            i += 1
            continue
        fence_char, fence_len = m.group(2)[0], len(m.group(2))
        block = [lines[i]]
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
                block.append(lines[i])
                i += 1
                closed = True
                break
            block.append(lines[i])
            i += 1
        if closed:
            flush_prose()
            segments.append(("code", "\n".join(block)))
        else:
            buf.extend(block)  # unclosed fence -> treat as prose
    flush_prose()
    return segments


# --- frontmatter / fences ----------------------------------------------------
def split_frontmatter(text: str) -> Tuple[str, str]:
    """Split a leading YAML frontmatter block from the body.

    Returns ``(frontmatter, body)``; ``frontmatter`` is empty when absent. The
    frontmatter delimiters and trailing newline are kept with the block so that
    ``frontmatter + body == text``.
    """
    m = _FRONTMATTER_RE.match(text)
    return (m.group(1), m.group(2)) if m else ("", text)


def strip_outer_fence(text: str) -> str:
    """Remove an outer ```` ``` ```` / ``~~~`` fence that wraps the whole text."""
    m = _OUTER_FENCE_RE.match(text)
    return m.group(2) if m else text


# --- validation --------------------------------------------------------------
def validate_structure(original: str, compressed: str) -> List[str]:
    """Return structure-preservation errors (empty list == valid).

    Hard checks that compaction must never violate: heading count, fenced code
    blocks (exact), the set of URLs, and inline-code spans.
    """
    errors: List[str] = []
    h1, h2 = extract_headings(original), extract_headings(compressed)
    if len(h1) != len(h2):
        errors.append(f"Heading count mismatch: {len(h1)} vs {len(h2)}")
    if extract_code_blocks(original) != extract_code_blocks(compressed):
        errors.append("Code blocks not preserved exactly")
    u1, u2 = extract_urls(original), extract_urls(compressed)
    if u1 != u2:
        errors.append(f"URL mismatch: lost={u1 - u2}, added={u2 - u1}")
    c1, c2 = Counter(extract_inline_codes(original)), Counter(extract_inline_codes(compressed))
    if c1 != c2:
        lost = {k for k in c1 if c2.get(k, 0) < c1[k]}
        if lost:
            errors.append(f"Inline code lost: {lost}")
    return errors
