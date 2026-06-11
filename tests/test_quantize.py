"""Tests for Numerical Quantization."""

import pytest

from compactprompt import QuantizedColumn, quantize, quantize_uniform


sk = pytest.importorskip  # for optional k-means / pandas tests


VALUES = [1.0, 2.5, 3.3, 4.8, 5.1, 7.9, 9.2, 10.0]


def test_uniform_endpoints_exact():
    q = quantize_uniform(VALUES, bits=8)
    recon = q.reconstruct()
    assert recon[0] == pytest.approx(min(VALUES))
    assert recon[-1] == pytest.approx(max(VALUES))


def test_uniform_error_within_bound():
    q = quantize_uniform(VALUES, bits=4)
    recon = q.reconstruct()
    errors = [abs(a - b) for a, b in zip(VALUES, recon)]
    assert max(errors) <= q.max_error + 1e-9


def test_more_bits_less_error():
    e_low = max(abs(a - b) for a, b in zip(VALUES, quantize_uniform(VALUES, 3).reconstruct()))
    e_high = max(abs(a - b) for a, b in zip(VALUES, quantize_uniform(VALUES, 8).reconstruct()))
    assert e_high <= e_low


def test_uniform_codes_are_integers():
    q = quantize_uniform(VALUES, bits=6)
    assert all(isinstance(c, int) for c in q.codes)


def test_constant_column():
    q = quantize_uniform([5.0, 5.0, 5.0], bits=8)
    assert q.reconstruct() == [5.0, 5.0, 5.0]
    assert q.max_error == 0.0


def test_empty_column():
    q = quantize_uniform([], bits=8)
    assert q.reconstruct() == []


def test_quantize_dispatch_uniform():
    q = quantize(VALUES, method="uniform", bits=8)
    assert q.method == "uniform"


def test_quantize_invalid_method():
    with pytest.raises(ValueError):
        quantize(VALUES, method="nonsense")


def test_max_error_formula():
    q = quantize_uniform(VALUES, bits=4)
    expected = (max(VALUES) - min(VALUES)) / (2 ** 4 - 1)
    assert q.max_error == pytest.approx(expected)


# --- optional: k-means (needs scikit-learn) --------------------------------
def test_kmeans_quantization():
    pytest.importorskip("sklearn")
    from compactprompt import quantize_kmeans

    q = quantize_kmeans(VALUES, k=4)
    assert q.method == "kmeans"
    assert len(q.codes) == len(VALUES)
    recon = q.reconstruct()
    assert max(abs(a - b) for a, b in zip(VALUES, recon)) <= q.max_error + 1e-9


def test_kmeans_more_clusters_less_error():
    pytest.importorskip("sklearn")
    from compactprompt import quantize_kmeans

    e2 = quantize_kmeans(VALUES, k=2).max_error
    e6 = quantize_kmeans(VALUES, k=6).max_error
    assert e6 <= e2 + 1e-9


# --- optional: pandas dataframe --------------------------------------------
def test_quantize_dataframe():
    pd = pytest.importorskip("pandas")
    from compactprompt import quantize_dataframe

    df = pd.DataFrame({"a": VALUES, "label": ["x"] * len(VALUES)})
    new_df, results = quantize_dataframe(df, bits=8)
    assert "a" in results
    assert list(new_df["label"]) == ["x"] * len(VALUES)
    assert isinstance(results["a"], QuantizedColumn)
