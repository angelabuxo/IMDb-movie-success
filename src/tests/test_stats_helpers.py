"""The shared statistics behind the screenplay-text pipeline: genre and decade bins, empirical p-values,
Benjamini-Hochberg, permutations and Scattertext scores.

Notebooks: notebooks/scripts_text/* (via analysis/scripts_text/textstats.py and s01_groups.py).
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import false_discovery_control

from analysis.scripts_text import s01_groups, textstats


# ---------------------------------------------------------------- genre and decade bins

@pytest.mark.parametrize("genres, expected", [
    ("Drama,Horror", "Horror"),          # priority order wins, not listed order
    ("Action,Sci-Fi", "Sci-Fi"),
    ("Drama", "Drama"),
    ("Romance,Musical", "Other"),
    ("", "Other"),
    (None, "Other"),
])
def test_primary_genre_uses_priority_order(genres, expected):
    assert s01_groups.primary_genre(genres) == expected


def test_primary_genre_does_not_match_substrings():
    assert s01_groups.primary_genre("Sci-Fi Horror Comedy") == "Other"  # one unsplit token, not three genres


@pytest.mark.parametrize("year, expected", [
    (1950, "pre-1980"), (1979, "pre-1980"), (1980, "1980s"), (1999, "1990s"),
    (2000, "2000s"), (2019, "2010s"), (2025, "2020s"), (2026, "other"),
])
def test_decade_bin_edges(year, expected):
    assert s01_groups.decade_bin(year) == expected


# ---------------------------------------------------------------- power and p-values

@pytest.mark.parametrize("fn", [textstats.empirical_p])
def test_empirical_p_is_never_zero_and_at_most_one(fn):
    assert float(fn(0, 999)) == pytest.approx(0.001)
    assert float(fn(999, 999)) == 1.0
    assert float(fn(4, 99)) == pytest.approx(0.05)


def test_bh_matches_scipy():
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(0, 0.01, 10), rng.uniform(0, 1, 90)])
    assert np.allclose(textstats.bh(p), false_discovery_control(p))
    assert np.allclose(textstats.bh([0.01, 0.04, 0.03, 0.2]), false_discovery_control([0.01, 0.04, 0.03, 0.2]))


def test_jaccard():
    assert textstats.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert textstats.jaccard(set(), set()) == 1.0
    assert textstats.jaccard({1}, {2}) == 0.0


# ---------------------------------------------------------------- permutations

def test_stratified_permutation_only_moves_labels_within_strata():
    rng = np.random.default_rng(2)
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1])
    strata = np.array(["a", "a", "a", "b", "b", "b", "b", "c", "d"])
    seen = set()
    for _ in range(50):
        yp = textstats.stratified_permutation(y, strata, rng)
        for s in np.unique(strata):
            assert sorted(yp[strata == s]) == sorted(y[strata == s])
        assert yp[-1] == y[-1] and yp[-2] == y[-2]               # singleton strata never move
        seen.add(tuple(yp))
    assert len(seen) > 1                                         # and it does shuffle
    assert y.tolist() == [1, 0, 1, 0, 1, 1, 0, 0, 1]             # input not modified


def test_quartile_labels_use_the_basis_quartiles_inclusively():
    basis = pd.Series(np.arange(1, 101, dtype=float))           # q1 = 25.75, q3 = 75.25
    v = pd.Series([25.75, 25.8, 50.0, 75.25, 75.2, 1.0, 100.0])
    lab = s01_groups.quartile_labels(v, basis)
    # The middle comes back missing (None on older pandas, NaN on pandas 3); callers only use isin(low, high).
    assert lab.fillna("mid").tolist() == ["low", "mid", "mid", "high", "mid", "low", "high"]


# ---------------------------------------------------------------- Scattertext scores and shuffle null

def _toy_dtm(seed=3, n=40, vocab=30):
    """n scripts, half high. Term 0 is used only by high scripts, term 1 only by low ones."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(n, vocab))
    y = np.repeat([0, 1], n // 2).astype(np.int8)
    X[:, 0] = np.where(y == 1, 5, 0)
    X[:, 1] = np.where(y == 0, 5, 0)
    X[:, 2] = 0
    X[3, 2] = 1                                                  # a term in one script only
    return sp.csr_matrix(X), X.sum(1), y


@pytest.mark.parametrize("mode", ["repeated", "presence"])
def test_scores_sign_and_min_df(mode):
    X, lengths, y = _toy_dtm()
    r = textstats.scores(X, lengths, y, min_df=2, mode=mode)
    assert r["score"][0] > textstats.cfg()["scattertext"]["threshold"]
    assert r["score"][1] < -textstats.cfg()["scattertext"]["threshold"]
    assert not r["ok"][2] and r["ok"][0]                         # below min_df is flagged, not dropped
    flipped = textstats.scores(X, lengths, 1 - y, min_df=2, mode=mode)
    assert np.allclose(flipped["score"], -r["score"])            # swapping groups flips every score


def test_shuffle_null_separates_planted_terms_from_noise():
    X, lengths, y = _toy_dtm()
    res = textstats.shuffle_null(X, lengths, y, None, min_df=2, shuffles=200, seed=0, progress=False)
    p = textstats.empirical_p(res["extreme"], res["shuffles"])
    assert p[0] < 0.01 and p[1] < 0.01                           # the two planted terms (ok-index 0, 1)
    assert np.median(p[2:]) > 0.2                                # Poisson noise terms mostly aren't
    assert (res["std"] >= 0).all() and len(res["max_abs"]) == 200


def test_shuffle_null_is_reproducible_from_its_seed():
    X, lengths, y = _toy_dtm()
    a = textstats.shuffle_null(X, lengths, y, None, 2, 30, seed=5, progress=False)
    b = textstats.shuffle_null(X, lengths, y, None, 2, 30, seed=5, progress=False)
    assert np.array_equal(a["extreme"], b["extreme"]) and np.array_equal(a["max_abs"], b["max_abs"])
