"""Q1 genre shuffle (analysis/genre/q1_genre.py): the shuffle keeps each stratum's returns, the Holm step is
monotone, and the budget bins are equal-sized. No data files needed."""
import numpy as np
import pandas as pd

from analysis.genre import q1_genre as q


def toy():
    rng = np.random.default_rng(0)
    n = 200
    films = pd.DataFrame({"imdb_id": [f"tt{i}" for i in range(n)], "usable_money": True,
                          "ln_gross_over_budget": rng.normal(size=n), "budget": np.exp(rng.normal(16, 1, n)),
                          "era_bin": rng.choice(["1990s", "2000s"], n)})
    genres = pd.DataFrame({"imdb_id": [f"tt{i}" for i in range(n) for _ in range(2)],
                           "genre": [g for i in range(n) for g in (("A", "B") if i % 2 else ("A", "C"))]})
    return films, genres


def test_budget_bins_are_equal_sized():
    films, genres = toy()
    f, keep, M = q.prepare(films, genres, "usable_money", keep=["A", "B", "C"])
    assert f["budget_bin"].value_counts().nunique() == 1
    assert M.shape == (200, 3) and M[:, 0].all()


def test_shuffle_keeps_each_stratum():
    films, genres = toy()
    f, keep, M = q.prepare(films, genres, "usable_money", keep=["A", "B", "C"])
    y = f["ln_gross_over_budget"].to_numpy()
    groups = q.strata(f, "budget_era")
    nl = q.permuted_means(y, M, groups, 50, np.random.default_rng(1))
    # genre A holds every film, so its mean can't move under any shuffle
    assert np.allclose(nl[:, 0], y.mean())
    # a stratum that only shuffles inside itself keeps its own mean for a genre covering exactly it
    one = (groups == groups[0]).astype(float)[:, None]
    nl1 = q.permuted_means(y, one, groups, 20, np.random.default_rng(2))
    assert np.allclose(nl1, y[groups == groups[0]].mean())


def test_holm():
    adj = q.holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])
    assert (q.holm([0.2, 0.9]) <= 1).all()


def test_curveball_keeps_both_margins():
    from analysis.genre.exploratory.q1_network import cooc, curveball
    rng = np.random.default_rng(3)
    M = (rng.random((60, 6)) < .35).astype(int)
    rows = [set(np.flatnonzero(m)) for m in M]
    before_rows, before_cols = [len(r) for r in rows], np.bincount([g for r in rows for g in r], minlength=6)
    rows = curveball([set(r) for r in rows], 500, rng)
    assert [len(r) for r in rows] == before_rows
    assert (np.bincount([g for r in rows for g in r], minlength=6) == before_cols).all()
    assert np.allclose(np.diag(cooc(rows, 6)), before_cols)


def test_assortativity_signs():
    from analysis.genre.exploratory.q1_network import assortativity
    edges = np.array([[0, 1], [2, 3], [4, 5], [0, 2]])
    assert assortativity(edges[:3], np.array([1, 1, 0, 0, 1, 1])) > .99
    assert assortativity(np.array([[0, 1], [2, 3]]), np.array([1, 0, 1, 0])) < -.99


def test_log_odds_symmetric():
    from analysis.genre.exploratory.q1_words import log_odds_z
    a = np.array([5., 5., 5.])
    z = log_odds_z(np.array([10., 2., 4.]), np.array([2., 10., 4.]), a)
    assert z[0] > 0 > z[1] and abs(z[0] + z[1]) < 1e-9


def test_shares_sum_to_one_and_keep_order():
    from analysis.genre.exploratory.q1_share_fixes import to_shares
    R = np.array([[0.9, 0.3, 0.1], [0.2, 0.2, 0.8]])
    for p in (0.5, 1, 1.5, 2, 4):
        S = to_shares(R, p)
        assert np.allclose(S.sum(1), 1)
        assert (np.argsort(S, 1) == np.argsort(R, 1)).all()
    assert to_shares(R, 4)[0, 0] > to_shares(R, 1)[0, 0]


def test_genre_coefs_drops_one_share_column():
    from analysis.genre.exploratory.q1_share_fixes import genre_coefs
    rng = np.random.default_rng(5)
    n = 400
    f = pd.DataFrame({"budget_bin": rng.integers(0, 3, n), "era_bin": rng.choice(["1990s", "2000s"], n)})
    S = rng.dirichlet([1, 1, 1], n)
    y = 2 * S[:, 0] + rng.normal(0, .1, n)
    c = genre_coefs(y, f, S, ["A", "B", "C"]).set_index("genre")
    assert c.loc["A", "coef_centred"] > c.loc["B", "coef_centred"] and abs(c["coef_centred"].sum()) < 1e-6


def test_outliers_remove_both_ends(monkeypatch):
    films, genres = toy()
    films["title"], films["year"], films["genres"], films["gross"] = "t", 2000, "A", 1.0
    f, keep, M = q.prepare(films, genres, "usable_money", keep=["A", "B", "C"])
    from analysis.genre import q1_checks as ck
    monkeypatch.setattr(ck, "OUTLIER_CUTS", (("10 highest and 10 lowest films", 10), ("top and bottom 5%", 0.05)))
    rows, gone = ck.outliers(f, keep, M, 20)
    d = pd.DataFrame(rows)
    assert d.groupby("check")["films_removed"].first().to_dict() == {"10 highest and 10 lowest films": 20, "top and bottom 5%": 20}
    assert len(gone) == 20 and set(gone["side"]) == {"highest", "lowest"}
    assert d.loc[d["genre"] == "A", "n_films"].eq(180).all()
