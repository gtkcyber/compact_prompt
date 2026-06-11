"""Tests for self-information scoring and fusion."""

import math

import pytest

from compactprompt.scoring import (
    StaticSelfInformation,
    aggregate_word_surprisals,
    combine,
    default_static,
)


# --- static self-information ----------------------------------------------
def test_static_from_corpus_rare_more_informative():
    corpus = ["the the the the cat", "the the dog the the"]
    s = StaticSelfInformation.from_corpus(corpus)
    # "cat" is rarer than "the" -> higher self-information.
    assert s.score("cat") > s.score("the")


def test_static_score_is_nonnegative():
    s = StaticSelfInformation.from_text("alpha beta gamma alpha beta")
    assert s.score("alpha") >= 0


def test_static_floor_prevents_infinite():
    s = StaticSelfInformation(lambda _t: 0.0)
    assert math.isfinite(s.score("anything"))


def test_default_static_always_works():
    s = default_static("some bootstrap text here")
    assert math.isfinite(s.score("text"))


# --- fusion rule (Eq. 2-3) -------------------------------------------------
def test_combine_uses_mean_when_close():
    # delta = |10.5 - 10| / 10 = 0.05 <= 0.1 -> mean = 10.25
    assert combine(10.0, 10.5, delta_threshold=0.1) == pytest.approx(10.25)


def test_combine_prefers_dynamic_when_divergent():
    # delta = |20 - 10| / 10 = 1.0 > 0.1 -> use dynamic (20)
    assert combine(10.0, 20.0, delta_threshold=0.1) == 20.0


def test_combine_handles_zero_static():
    assert combine(0.0, 7.0) == 7.0


def test_combine_exact_threshold_uses_mean():
    # delta exactly 0.1 -> mean branch (<=)
    assert combine(10.0, 11.0, delta_threshold=0.1) == pytest.approx(10.5)


# --- aggregation -----------------------------------------------------------
def test_aggregate_sums_subwords_into_spans():
    # Two words at spans (0,5) and (6,11); three subwords.
    surprisals = [
        ("hel", 0, 3, 1.0),
        ("lo", 3, 5, 2.0),
        ("world", 6, 11, 4.0),
    ]
    spans = [(0, 5), (6, 11)]
    assert aggregate_word_surprisals(surprisals, spans) == [3.0, 4.0]


def test_aggregate_empty_span_is_zero():
    spans = [(0, 5), (100, 110)]
    surprisals = [("x", 0, 5, 5.0)]
    assert aggregate_word_surprisals(surprisals, spans) == [5.0, 0.0]
