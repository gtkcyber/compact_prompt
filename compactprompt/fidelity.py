"""Semantic fidelity metric (CompactPrompt Sec. 4).

Measure how well compressed text preserves meaning by embedding the original
and compressed strings with ``all-mpnet-base-v2`` and computing cosine
similarity. We report the mean and the 5th-percentile (worst-case) scores, as
in the paper.

Needs the ``embeddings`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

from .embedding import get_embedder


@dataclass
class FidelityResult:
    """Cosine-similarity fidelity statistics.

    Attributes:
        scores: Per-pair cosine similarities.
        mean: Mean cosine similarity.
        p5: 5th-percentile cosine similarity (worst-case fidelity).
    """

    scores: List[float]
    mean: float
    p5: float


def cosine_fidelity(
    original: Union[str, Sequence[str]],
    compressed: Union[str, Sequence[str]],
    embedder: Optional[Callable[[List[str]], "object"]] = None,
) -> FidelityResult:
    """Compute cosine-similarity fidelity between original and compressed text.

    Accepts either single strings or equal-length sequences of strings.

    Returns:
        A :class:`FidelityResult` with per-pair scores plus mean / 5th pct.
    """
    try:
        import numpy as np
    except Exception as exc:
        raise ImportError("cosine_fidelity needs numpy.") from exc

    if isinstance(original, str):
        original = [original]
    if isinstance(compressed, str):
        compressed = [compressed]
    if len(original) != len(compressed):
        raise ValueError("original and compressed must have equal length")

    embed = embedder or get_embedder()
    a = np.asarray(embed(list(original)), dtype=float)
    b = np.asarray(embed(list(compressed)), dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)

    # Embeddings are L2-normalized, but normalize defensively for robustness.
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    scores = np.sum(a * b, axis=1).tolist()
    return FidelityResult(
        scores=scores,
        mean=float(np.mean(scores)),
        p5=float(np.percentile(scores, 5)),
    )
