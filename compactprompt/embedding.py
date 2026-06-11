"""Shared sentence-embedding loader (``all-mpnet-base-v2``).

Used by exemplar selection and the semantic-fidelity metric. The model is
loaded lazily and cached, so importing the package stays cheap.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List

DEFAULT_MODEL = "all-mpnet-base-v2"


@lru_cache(maxsize=4)
def _load_model(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise ImportError(
            "Semantic embeddings need sentence-transformers. Install with: "
            "pip install 'compactprompt[embeddings]'"
        ) from exc
    return SentenceTransformer(name)


def get_embedder(model_name: str = DEFAULT_MODEL) -> Callable[[List[str]], "object"]:
    """Return a callable ``List[str] -> ndarray`` of normalized embeddings."""

    def embed(texts: List[str]):
        model = _load_model(model_name)
        return model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )

    return embed
