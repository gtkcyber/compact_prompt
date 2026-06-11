"""Representative Example Selection for few-shot prompts (CompactPrompt Sec. 3.4).

Pick a small, diverse set of exemplars instead of random ones:

1. Embed candidate texts with ``all-mpnet-base-v2`` and standardize any numeric
   features (zero mean, unit variance).
2. Run k-means for ``k`` in ``[5, 50]`` and pick ``k*`` maximizing the average
   silhouette score.
3. In each of the ``k*`` clusters, keep the point closest to the centroid.

Needs the ``embeddings`` and ``ml`` extras (sentence-transformers, scikit-learn).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .embedding import get_embedder


@dataclass
class SelectionResult:
    """Result of :func:`select_examples`.

    Attributes:
        indices: Indices (into the input list) of the chosen exemplars.
        examples: The chosen exemplar texts.
        k_star: The selected number of clusters.
        silhouette: Silhouette score at ``k_star``.
        silhouette_by_k: ``{k: score}`` for every evaluated ``k``.
    """

    indices: List[int]
    examples: List[str]
    k_star: int
    silhouette: float
    silhouette_by_k: dict = field(default_factory=dict)


def select_examples(
    texts: Sequence[str],
    k_range: Tuple[int, int] = (5, 50),
    numeric_features: Optional[Sequence[Sequence[float]]] = None,
    embedder: Optional[Callable[[List[str]], "object"]] = None,
    random_state: int = 42,
) -> SelectionResult:
    """Select representative few-shot exemplars via clustering.

    Args:
        texts: Candidate exemplar texts.
        k_range: Inclusive ``(min_k, max_k)`` search range for cluster count.
        numeric_features: Optional per-text numeric features; standardized and
            concatenated to the embeddings (as in the paper).
        embedder: Callable ``List[str] -> array`` of embeddings. Defaults to
            ``all-mpnet-base-v2`` via sentence-transformers.
        random_state: Seed for k-means.

    Returns:
        A :class:`SelectionResult`.
    """
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise ImportError(
            "select_examples needs numpy + scikit-learn. Install with: "
            "pip install 'compactprompt[ml]'"
        ) from exc

    n = len(texts)
    if n == 0:
        return SelectionResult([], [], 0, float("nan"), {})

    embed = embedder or get_embedder()
    X = np.asarray(embed(list(texts)), dtype=float)
    if X.ndim == 1:
        X = X.reshape(n, -1)

    if numeric_features is not None:
        feats = StandardScaler().fit_transform(np.asarray(numeric_features, dtype=float))
        X = np.hstack([X, feats])

    lo, hi = k_range
    # Silhouette is undefined for k=1 or k>=n; clamp the search range.
    hi = min(hi, n - 1)
    lo = max(2, lo)
    if hi < lo:
        # Too few points to cluster meaningfully: return all of them.
        return SelectionResult(list(range(n)), list(texts), n, float("nan"), {})

    best_k, best_score, best_labels, best_centers = lo, -1.0, None, None
    sil_by_k = {}
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X, labels))
        sil_by_k[k] = score
        if score > best_score:
            best_k, best_score, best_labels, best_centers = k, score, labels, km.cluster_centers_

    if best_labels is None:  # degenerate; fall back to one-per-point
        return SelectionResult(list(range(n)), list(texts), n, float("nan"), sil_by_k)

    indices: List[int] = []
    for c in range(best_k):
        members = np.where(best_labels == c)[0]
        if members.size == 0:
            continue
        dists = np.linalg.norm(X[members] - best_centers[c], axis=1)
        indices.append(int(members[int(np.argmin(dists))]))
    indices.sort()
    return SelectionResult(
        indices=indices,
        examples=[texts[i] for i in indices],
        k_star=best_k,
        silhouette=best_score,
        silhouette_by_k=sil_by_k,
    )
