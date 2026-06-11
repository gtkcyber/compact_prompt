"""Textual N-gram Abbreviation (CompactPrompt Sec. 3.2).

Replace frequent multi-word patterns with short, unique, **reversible**
placeholders — a lossless, LZW-style compression that shrinks repetitive
documents while guaranteeing exact round-trip reconstruction.

This module is pure Python with no dependencies.

Steps (matching the paper):
1. Extract every word n-gram and rank by frequency.
2. Take the top-K patterns and assign each a unique placeholder.
3. Replace occurrences (longest-first to handle overlaps) and store the
   reversible mapping.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .tokens import count_tokens

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class Abbreviation:
    """Result of :meth:`NgramAbbreviator.compress`.

    Attributes:
        text: The abbreviated text.
        dictionary: Reversible mapping ``placeholder -> original phrase``.
        original_tokens: Word count before abbreviation (for reporting).
    """

    text: str
    dictionary: Dict[str, str] = field(default_factory=dict)
    original_tokens: int = 0

    def restore(self) -> str:
        """Reconstruct the original text exactly (lossless)."""
        return NgramAbbreviator.decompress(self.text, self.dictionary)

    def __str__(self) -> str:
        return self.text


class NgramAbbreviator:
    """Reversibly abbreviate frequent n-grams.

    Args:
        n: N-gram length in words. The paper finds ``n=2`` best for QA accuracy.
        top_k: Number of top-frequency patterns to abbreviate. The paper's
            extraction step selects ~100-150; its best-accuracy ablation uses 3.
        min_count: Only abbreviate patterns occurring at least this many times.
        marker: Prefix for placeholders (default ``"@"``). Placeholders are kept
            token-cheap (e.g. ``@0``) so abbreviation actually reduces tokens.
        placeholder: Optional function ``index -> placeholder string`` to fully
            override the placeholder scheme.
        require_savings: When ``True`` (default), only abbreviate a pattern if
            its placeholder costs strictly fewer tokens than the phrase, so the
            output is never longer than the input.
    """

    def __init__(
        self,
        n: int = 2,
        top_k: int = 100,
        min_count: int = 2,
        marker: str = "@",
        placeholder=None,
        require_savings: bool = True,
    ):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.top_k = top_k
        self.min_count = min_count
        self.require_savings = require_savings
        self._placeholder = placeholder or (lambda i: f"{marker}{i}")

    def _candidates(self, text: str) -> Counter:
        """Count exact surface n-grams (words joined by single spaces)."""
        words = list(_WORD_RE.finditer(text))
        counts: Counter = Counter()
        pat = re.compile(r"\w+(?: \w+){%d}$" % (self.n - 1))
        for i in range(len(words) - self.n + 1):
            s = words[i].start()
            e = words[i + self.n - 1].end()
            surface = text[s:e]
            if pat.match(surface):
                counts[surface] += 1
        return counts

    def compress(self, text: str) -> Abbreviation:
        """Abbreviate ``text`` and return a reversible :class:`Abbreviation`."""
        original_tokens = len(_WORD_RE.findall(text))
        if not text:
            return Abbreviation(text, {}, 0)

        counts = self._candidates(text)
        ranked = [
            (gram, c) for gram, c in counts.most_common() if c >= self.min_count
        ]
        if not ranked:
            return Abbreviation(text, {}, original_tokens)

        # Assign each kept phrase a token-cheap placeholder guaranteed absent
        # from the text. Skip any phrase whose placeholder would not actually
        # save tokens, and stop once we have ``top_k`` beneficial patterns.
        dictionary: Dict[str, str] = {}
        phrase_to_ph: Dict[str, str] = {}
        idx = 0
        for gram, _c in ranked:
            if len(dictionary) >= self.top_k:
                break
            ph = self._placeholder(idx)
            while ph in text or ph in dictionary:
                idx += 1
                ph = self._placeholder(idx)
            idx += 1
            if self.require_savings and count_tokens(ph) >= count_tokens(gram):
                continue
            dictionary[ph] = gram
            phrase_to_ph[gram] = ph

        if not phrase_to_ph:
            return Abbreviation(text, {}, original_tokens)

        # Replace longest phrases first so overlaps resolve deterministically.
        phrases = sorted(phrase_to_ph, key=len, reverse=True)
        alt = re.compile("|".join(re.escape(p) for p in phrases))
        compressed = alt.sub(lambda m: phrase_to_ph[m.group(0)], text)

        # Drop any placeholder that ended up unused (e.g. fully overlapped).
        used = {ph for ph in dictionary if ph in compressed}
        dictionary = {ph: g for ph, g in dictionary.items() if ph in used}
        return Abbreviation(compressed, dictionary, original_tokens)

    @staticmethod
    def decompress(text: str, dictionary: Dict[str, str]) -> str:
        """Reverse abbreviation. Placeholders are restored longest-first."""
        for ph in sorted(dictionary, key=len, reverse=True):
            text = text.replace(ph, dictionary[ph])
        return text
