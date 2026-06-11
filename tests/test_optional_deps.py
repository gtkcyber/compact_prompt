"""Tests for the embedding-based strategies (skipped without optional deps).

These cover Representative Example Selection and the semantic-fidelity metric.
They require sentence-transformers (+ scikit-learn) and will download a model on
first run, so they are skipped automatically when the deps are absent.
"""

import pytest


def _have(mod):
    import importlib

    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_have("sentence_transformers") and _have("sklearn")),
    reason="needs sentence-transformers + scikit-learn",
)


def test_select_examples_returns_subset():
    from compactprompt import select_examples

    texts = [f"example about topic {i % 4}: lorem ipsum number {i}" for i in range(40)]
    res = select_examples(texts, k_range=(3, 8))
    assert 0 < len(res.indices) <= len(texts)
    assert res.examples == [texts[i] for i in res.indices]


def test_cosine_fidelity_high_for_similar():
    from compactprompt import cosine_fidelity

    r = cosine_fidelity("the cat sat on the mat", "a cat is sitting on the mat")
    assert r.mean > 0.5
    assert len(r.scores) == 1


def test_cosine_fidelity_batch():
    from compactprompt import cosine_fidelity

    r = cosine_fidelity(
        ["hello world", "financial report"],
        ["hello there world", "the financial report"],
    )
    assert len(r.scores) == 2
    assert r.p5 <= r.mean
