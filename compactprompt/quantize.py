"""Numerical Quantization (CompactPrompt Sec. 3.3).

Shrink the token footprint of numeric data by lowering precision within a
bounded error.

* **Uniform integer quantization** (Eq. 4-5): map values to ``L = 2**b``
  integer levels with a guaranteed max absolute error ``(max-min)/(L-1)``.
* **K-means quantization**: map values to ``k`` learned centroids, minimizing
  average squared reconstruction error.

Uniform quantization is pure Python (works on lists or numpy arrays). K-means
quantization uses scikit-learn (install the ``ml`` extra).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence


@dataclass
class QuantizedColumn:
    """A quantized numeric column plus the metadata needed to reconstruct it.

    Attributes:
        codes: Integer codes (uniform) or centroid indices (k-means).
        method: ``"uniform"`` or ``"kmeans"``.
        metadata: Reconstruction metadata (e.g. ``min``, ``max``, ``bits`` or
            ``centroids``).
    """

    codes: List[int]
    method: str
    metadata: dict = field(default_factory=dict)

    def reconstruct(self) -> List[float]:
        """Reconstruct approximate original values from codes + metadata."""
        if self.method == "uniform":
            lo = self.metadata["min"]
            hi = self.metadata["max"]
            levels = self.metadata["levels"]
            if levels <= 1 or hi == lo:
                return [lo for _ in self.codes]
            step = (hi - lo) / (levels - 1)
            return [lo + q * step for q in self.codes]
        if self.method == "kmeans":
            centroids = self.metadata["centroids"]
            return [centroids[c] for c in self.codes]
        raise ValueError(f"Unknown quantization method: {self.method}")

    @property
    def max_error(self) -> float:
        """The bound ``epsilon_max`` on absolute reconstruction error."""
        if self.method == "uniform":
            lo, hi, levels = self.metadata["min"], self.metadata["max"], self.metadata["levels"]
            if levels <= 1 or hi == lo:
                return 0.0
            return (hi - lo) / (levels - 1)
        if self.method == "kmeans":
            return self.metadata.get("max_error", float("nan"))
        return float("nan")


def quantize_uniform(values: Sequence[float], bits: int = 8) -> QuantizedColumn:
    """Uniform integer quantization (CompactPrompt Eq. 4-5).

    Args:
        values: The numeric column.
        bits: Bit-width ``b``; yields ``L = 2**b`` levels.

    Returns:
        A :class:`QuantizedColumn` with ``method="uniform"``.
    """
    vals = [float(v) for v in values]
    if not vals:
        return QuantizedColumn([], "uniform", {"min": 0.0, "max": 0.0, "levels": 1, "bits": bits})
    lo, hi = min(vals), max(vals)
    levels = 2 ** bits
    if hi == lo or levels <= 1:
        codes = [0 for _ in vals]
    else:
        span = hi - lo
        codes = [int(round((v - lo) / span * (levels - 1))) for v in vals]
    return QuantizedColumn(
        codes=codes,
        method="uniform",
        metadata={"min": lo, "max": hi, "levels": levels, "bits": bits},
    )


def quantize_kmeans(
    values: Sequence[float], k: int = 16, random_state: int = 42
) -> QuantizedColumn:
    """K-means quantization: map values to ``k`` centroids.

    Requires scikit-learn (``pip install 'compactprompt[ml]'``).

    Args:
        values: The numeric column.
        k: Number of centroids. Clamped to the number of distinct values.

    Returns:
        A :class:`QuantizedColumn` with ``method="kmeans"``.
    """
    try:
        import numpy as np
        from sklearn.cluster import KMeans
    except Exception as exc:
        raise ImportError(
            "quantize_kmeans needs numpy + scikit-learn. Install with: "
            "pip install 'compactprompt[ml]'"
        ) from exc

    vals = np.asarray([float(v) for v in values], dtype=float)
    if vals.size == 0:
        return QuantizedColumn([], "kmeans", {"centroids": [], "max_error": 0.0})
    distinct = np.unique(vals)
    k_eff = int(min(k, distinct.size))
    if k_eff <= 1:
        c = float(vals.mean())
        return QuantizedColumn(
            [0] * vals.size,
            "kmeans",
            {"centroids": [c], "max_error": float(np.max(np.abs(vals - c)))},
        )
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=random_state)
    labels = km.fit_predict(vals.reshape(-1, 1))
    centroids = km.cluster_centers_.reshape(-1).tolist()
    recon = np.asarray([centroids[c] for c in labels])
    return QuantizedColumn(
        codes=[int(c) for c in labels],
        method="kmeans",
        metadata={"centroids": centroids, "max_error": float(np.max(np.abs(vals - recon)))},
    )


def quantize(
    values: Sequence[float],
    method: str = "uniform",
    bits: int = 8,
    k: int = 16,
) -> QuantizedColumn:
    """Quantize a numeric column by ``method`` (``"uniform"`` or ``"kmeans"``)."""
    if method == "uniform":
        return quantize_uniform(values, bits=bits)
    if method == "kmeans":
        return quantize_kmeans(values, k=k)
    raise ValueError("method must be 'uniform' or 'kmeans'")


def quantize_dataframe(
    df: "object",
    columns: Optional[Iterable[str]] = None,
    method: str = "uniform",
    bits: int = 8,
    k: int = 16,
) -> tuple:
    """Quantize numeric columns of a pandas DataFrame in place-safe fashion.

    Args:
        df: A pandas DataFrame.
        columns: Columns to quantize; defaults to all numeric columns.
        method: Quantization method, passed to :func:`quantize`.
        bits: Bit-width for uniform quantization.
        k: Number of clusters for k-means quantization.

    Returns:
        A ``(new_df, results)`` tuple where ``new_df`` has reconstructed
        (quantized) values and ``results`` maps column name to
        :class:`QuantizedColumn`.
    """
    try:
        import pandas  # noqa: F401  # pylint: disable=unused-import
    except Exception as exc:
        raise ImportError("quantize_dataframe needs pandas.") from exc

    if columns is None:
        columns = [c for c in df.columns if df[c].dtype.kind in "fiu"]
    new_df = df.copy()
    results = {}
    for col in columns:
        qc = quantize(df[col].tolist(), method=method, bits=bits, k=k)
        results[col] = qc
        new_df[col] = qc.reconstruct()
    return new_df, results
