"""Token counting utilities.

The paper measures compression as a token-count ratio. We count tokens with
``tiktoken`` when it is available (so numbers line up with what the OpenAI /
Anthropic style tokenizers actually charge), and otherwise fall back to a
fast, dependency-free whitespace/word tokenizer. The fallback is deterministic
and good enough for relative comparisons.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=8)
def _tiktoken_encoder(name: str):
    """Return a cached tiktoken encoder, or ``None`` if tiktoken is missing."""
    try:
        import tiktoken
    except Exception:
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return None


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count the number of tokens in ``text``.

    Uses ``tiktoken`` (encoding ``cl100k_base`` by default, the GPT-4 family
    tokenizer) when installed; otherwise falls back to a regex word/punctuation
    tokenizer.

    Args:
        text: The string to measure.
        encoding: Name of the tiktoken encoding to use when available.

    Returns:
        The number of tokens.
    """
    if not text:
        return 0
    enc = _tiktoken_encoder(encoding)
    if enc is not None:
        return len(enc.encode(text))
    return len(_WORD_RE.findall(text))


def simple_word_tokens(text: str) -> List[str]:
    """Split ``text`` into word and punctuation tokens (dependency-free)."""
    return _WORD_RE.findall(text)
