"""CompactPrompt — easy prompt & data compression for LLM workflows.

A faithful, dependency-light implementation of the four strategies in
*"CompactPrompt: A Unified Pipeline for Prompt and Data Compression in LLM
Workflows"* (Choi et al., 2025, arXiv:2510.18043):

1. Hard Prompt Compression  — :class:`HardPromptCompressor`
2. Textual N-gram Abbreviation — :class:`NgramAbbreviator`
3. Numerical Quantization — :func:`quantize`, :func:`quantize_dataframe`
4. Representative Example Selection — :func:`select_examples`

Quick start::

    from compactprompt import CompactPrompt

    result = CompactPrompt.compact("Please kindly go ahead and summarize ...")
    print(result.text)          # compressed prompt
    print(f"{result.ratio:.2f}x smaller")
    print(result.restore())     # reverse the reversible (n-gram) step

Heavy libraries (torch, spaCy, scikit-learn, sentence-transformers) are imported
lazily, only when a strategy that needs them is used. The headline
``CompactPrompt.compact`` works on a bare Python install.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .embedding import get_embedder
from .examples import SelectionResult, select_examples
from .fidelity import FidelityResult, cosine_fidelity
from .caveman import CavemanCompressor
from .hard_prompt import HardPromptCompressor, HardPromptResult
from .llmlingua import LLMLinguaCompressor
from .ngram import Abbreviation, NgramAbbreviator
from .pipeline import CompactPrompt, CompactResult
from .quantize import (
    QuantizedColumn,
    quantize,
    quantize_dataframe,
    quantize_kmeans,
    quantize_uniform,
)
from .scoring import (
    LocalLMScorer,
    StaticSelfInformation,
    combine,
    default_static,
)
from .tokens import count_tokens

try:
    __version__ = version("compactprompt")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.0.0+unknown"

# --- convenience top-level functions ---------------------------------------


def compact(prompt: str, **kwargs) -> CompactResult:
    """Shorthand for :meth:`CompactPrompt.compact`."""
    return CompactPrompt.compact(prompt, **kwargs)


def abbreviate(text: str, n: int = 2, top_k: int = 100) -> Abbreviation:
    """Shorthand for reversible n-gram abbreviation."""
    return NgramAbbreviator(n=n, top_k=top_k).compress(text)


def restore(text: str, dictionary) -> str:
    """Reverse n-gram abbreviation given its dictionary."""
    return NgramAbbreviator.decompress(text, dictionary)


__all__ = [
    "__version__",
    # headline API
    "CompactPrompt",
    "CompactResult",
    "compact",
    # hard prompt
    "HardPromptCompressor",
    "HardPromptResult",
    "LLMLinguaCompressor",
    "CavemanCompressor",
    # n-gram
    "NgramAbbreviator",
    "Abbreviation",
    "abbreviate",
    "restore",
    # quantization
    "quantize",
    "quantize_uniform",
    "quantize_kmeans",
    "quantize_dataframe",
    "QuantizedColumn",
    # exemplar selection
    "select_examples",
    "SelectionResult",
    # fidelity
    "cosine_fidelity",
    "FidelityResult",
    "get_embedder",
    # scoring internals
    "StaticSelfInformation",
    "LocalLMScorer",
    "combine",
    "default_static",
    "count_tokens",
]
