"""Profit attribution (archive/analysis/attribution): the model on made-up films, where the right answer is known.

Nothing here reads data/raw. The explorer page is tested on the committed outputs.
"""

from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.attribution import model as M

ROOT = Path(__file__).resolve().parents[2]


def planted(n_films=1500, n_dir=60, n_act=150, seed=0, sd_dir=0.5, sd_act=0.3, noise=0.5):
    """Films whose return is a planted director effect plus three planted actor effects plus noise."""
    rng = np.random.default_rng(seed)
    d_eff, a_eff = rng.normal(0, sd_dir, n_dir), rng.normal(0, sd_act, n_act)
    films = [f"tt{i:05d}" for i in range(n_films)]
    director = rng.integers(0, n_dir, n_films)
    cast = np.array([rng.choice(n_act, 3, replace=False) for _ in range(n_films)])
    y = 0.4 + d_eff[director] + a_eff[cast].sum(axis=1) + rng.normal(0, noise, n_films)
    table = pd.DataFrame({"log_budget": rng.normal(17, 1, n_films), "year": rng.integers(1980, 2020, n_films),
                          "genres": rng.choice(["Drama", "Comedy", "Action,Drama"], n_films),
                          "franchise": 0}, index=films)
    credits = pd.concat([
        pd.DataFrame({"film": films, "entity": [f"d{k}" for k in director], "role": "director"}),
        pd.DataFrame({"film": np.repeat(films, 3), "entity": [f"a{k}" for k in cast.ravel()], "role": "cast"})])
    return table, credits, y, d_eff, a_eff


ROLES2 = ["director", "cast"]


def test_shapley_product_matches_brute_force_and_adds_up():
    rng = np.random.default_rng(1)
    terms = rng.normal(0, 0.6, 6)
    base = 3e7
    phi = M.shapley_product(base, terms)
    brute = np.zeros(6)
    orders = list(permutations(range(6)))
    for order in orders:
        level = 0.0
        for i in order:
            brute[i] += base * (np.exp(level + terms[i]) - np.exp(level))
            level += terms[i]
    brute /= len(orders)
    assert np.allclose(phi, brute, rtol=1e-10)
    assert np.isclose(phi.sum(), base * (np.exp(terms.sum()) - 1))


def test_shapley_product_edge_cases():
    assert np.allclose(M.shapley_product(100.0, np.zeros(4)), 0)
    assert np.isclose(M.shapley_product(100.0, np.array([np.log(2)]))[0], 100.0)  # one term: doubling adds the base
    phi = M.shapley_product(1.0, np.array([0.3, 0.3, -0.5]))
    assert np.isclose(phi[0], phi[1]) and phi[2] < 0 < phi[0]                      # equal terms, equal dollars
    many = np.random.default_rng(2).normal(0, 0.2, 60)
    assert np.isclose(M.shapley_product(5.0, many).sum(), 5.0 * np.expm1(many.sum()))
    with pytest.raises(ValueError):
        M.shapley_product(1.0, np.zeros(200))


def test_shapley_blocks_on_an_additive_game():
    worth = {"a": 0.1, "b": 0.05, "c": 0.0}
    shap, v = M.shapley_blocks(lambda s: sum(worth[p] for p in s), list(worth))
    assert shap == pytest.approx(worth) and len(v) == 8


def test_design_gives_columns_only_to_repeat_entities():
    films = pd.DataFrame({"log_budget": [1.0, 2, 3, 4], "year": [1990, 1991, 2001, 2002],
                          "genres": ["Drama"] * 4, "franchise": [0, 1, 0, 0]}, index=list("wxyz"))
    credits = pd.DataFrame({"film": ["w", "x", "y", "z", "z", "q"], "entity": ["A", "A", "B", "A", "A", "A"],
                            "role": ["director"] * 6})
    d = M.build_design(films, credits, roles=["director"])
    assert list(d.key[d.cols("director")]) == ["A"]                     # B has one film; the film "q" isn't in the sample
    assert d.X[:, d.cols("director")].toarray().ravel().tolist() == [1, 1, 0, 1]   # the duplicate credit counts once
    assert len(d.cols("franchise")) == 1 and len(d.cols("budget")) == 2


def test_planted_effects_are_recovered():
    table, credits, y, d_eff, a_eff = planted()
    d = M.build_design(table, credits, roles=ROLES2)
    _, beta = M.fit(d, y, {"director": 1.0, "cast": 1.0})
    got_d = pd.Series(beta[d.cols("director")], index=d.key[d.cols("director")])
    got_a = pd.Series(beta[d.cols("cast")], index=d.key[d.cols("cast")])
    assert np.corrcoef(got_d.values, [d_eff[int(k[1:])] for k in got_d.index])[0, 1] > 0.95
    assert np.corrcoef(got_a.values, [a_eff[int(k[1:])] for k in got_a.index])[0, 1] > 0.9


def test_heavier_penalty_shrinks_and_infinite_switches_off():
    table, credits, y, *_ = planted(n_films=400)
    d = M.build_design(table, credits, roles=ROLES2)
    light = M.fit(d, y, {"director": 1.0, "cast": 1.0})[1]
    heavy = M.fit(d, y, {"director": 1000.0, "cast": 1.0})[1]
    off = M.fit(d, y, {"director": np.inf, "cast": 1.0})[1]
    c = d.cols("director")
    assert np.abs(heavy[c]).mean() < 0.2 * np.abs(light[c]).mean()
    assert np.all(off[c] == 0) and np.any(off[d.cols("cast")] != 0)
    only = M.fit(d, y, {"director": 1.0, "cast": 1.0}, blocks=["budget", "director"])[1]
    assert np.all(only[d.cols("cast")] == 0) and np.all(only[d.cols("genre")] == 0)


def test_out_of_fold_ignores_the_films_own_return():
    """Changing one film's return must not change that film's own split, only other films'."""
    table, credits, y, *_ = planted(n_films=300)
    d = M.build_design(table, credits, roles=ROLES2)
    pen = {"director": 5.0, "cast": 5.0}
    folds = M.grouped_folds(np.arange(len(y)), 5, 3)
    p1, _, C1 = M.out_of_fold(d, y, pen, folds)
    y2 = y.copy()
    y2[7] += 25.0
    p2, _, C2 = M.out_of_fold(d, y2, pen, folds)
    same_fold = folds == folds[7]
    assert np.allclose(p1[same_fold], p2[same_fold], atol=1e-4)
    assert np.allclose(C1[7].toarray(), C2[7].toarray(), atol=1e-4)
    assert not np.allclose(p1[~same_fold], p2[~same_fold], atol=1e-4)


def test_out_of_fold_parts_add_up_to_the_prediction():
    table, credits, y, *_ = planted(n_films=300)
    d = M.build_design(table, credits, roles=ROLES2)
    folds = M.grouped_folds(np.arange(len(y)), 4, 0)
    pred, icpt, C = M.out_of_fold(d, y, {"director": 5.0, "cast": 5.0}, folds)
    parts = M.block_contributions(d, C, M.CONTROLS + ROLES2)
    assert np.allclose(icpt + sum(parts.values()), pred)


def test_groups_stay_in_one_fold():
    groups = np.array(["s1", "s1", "s1", "x", "y", "s2", "s2", "z"] * 5, dtype=object)
    folds = M.grouped_folds(groups, 3, 1)
    for g in set(groups):
        assert len(set(folds[groups == g])) == 1
    assert set(folds) == {0, 1, 2}


def test_forward_uses_earlier_years_only():
    table, credits, y, *_ = planted(n_films=400)
    d = M.build_design(table, credits, roles=ROLES2)
    year = table.year.to_numpy()
    pen = {"director": 5.0, "cast": 5.0}
    p1, _, C1 = M.forward(d, y, pen, year, 1990)
    assert np.isnan(p1[year < 1990]).all() and np.isfinite(p1[year >= 1990]).all()
    y2 = y.copy()
    y2[year >= 2005] += 10.0                      # the future changes
    p2, _, C2 = M.forward(d, y2, pen, year, 1990)
    past = (year >= 1990) & (year <= 2005)        # 2005 itself is fitted on years before 2005
    assert np.allclose(p1[past], p2[past], atol=1e-4)
    assert np.allclose(C1[past].toarray(), C2[past].toarray(), atol=1e-4)


def test_shuffle_keeps_each_years_returns():
    rng = np.random.default_rng(0)
    y, year = rng.normal(size=200), rng.integers(2000, 2005, 200)
    s = M.shuffle_within(y, year, rng)
    assert not np.array_equal(s, y)
    for v in set(year):
        assert sorted(s[year == v]) == sorted(y[year == v])


def test_shuffled_returns_predict_nothing_and_real_ones_do():
    table, credits, y, *_ = planted(n_films=800)
    d = M.build_design(table, credits, roles=ROLES2)
    pen = {"director": 3.0, "cast": 3.0}
    folds = M.grouped_folds(np.arange(len(y)), 5, 0)
    assert M.cv_r2(d, y, pen, folds) > 0.3
    ys = M.shuffle_within(y, table.year.to_numpy(), np.random.default_rng(5))
    assert M.cv_r2(d, ys, pen, folds) < 0.03


def test_split_half_tells_signal_from_noise():
    table, credits, y, *_ = planted(n_films=2000, n_dir=40)
    d = M.build_design(table, credits, roles=ROLES2)
    pen = {"director": 3.0, "cast": 3.0}
    groups = np.arange(len(y))
    real = M.split_half(d, y, pen, groups, ROLES2, reps=3, seed=0)
    noise = M.split_half(d, np.random.default_rng(1).normal(size=len(y)), pen, groups, ROLES2, reps=3, seed=0)
    assert real["director"]["mean_r"] > 0.8 and real["director"]["entities"] > 10
    assert abs(noise["director"]["mean_r"]) < 0.4


def test_tune_prefers_a_light_penalty_for_real_effects_and_a_heavy_one_for_noise():
    table, credits, y, *_ = planted(n_films=1200, sd_act=0.0)   # directors matter, the cast doesn't
    d = M.build_design(table, credits, roles=ROLES2)
    folds = M.grouped_folds(np.arange(len(y)), 4, 0)
    pen, path = M.tune(d, y, folds, ROLES2, [1.0, 30.0, np.inf], start=30.0, passes=1)
    assert pen["director"] == 1.0 and pen["cast"] != 1.0 and len(path) == 4


def test_franchise_groups_join_linked_films():
    from analysis.attribution.run import franchise_groups

    fr = pd.DataFrame({"tconst": ["t1", "t2", "t3", "t4", "t5"], "qid": ["Q1", "Q2", "Q3", "Q4", None],
                       "follows": [None, "Q1", None, None, None], "followed_by": ["Q2", None, None, None, None],
                       "series": [None, "Q9", "Q9", None, None]})
    g = franchise_groups(fr, ["t1", "t2", "t3", "t4", "t5", "t6"])
    assert g[0] == g[1] == g[2] and len({g[0], g[3], g[4], g[5]}) == 4


def test_cpi_annual_means_and_deflator(tmp_path):
    from analysis.attribution.cpi import annual, deflator

    months = pd.DataFrame({"observation_date": [f"{y}-{m:02d}-01" for y in (2000, 2001, 2002) for m in range(1, 13)],
                           "CPIAUCNS": [100.0] * 12 + [200.0] * 11 + [None] + [300.0] * 5 + [None] * 7})
    a = annual(months)
    assert a.year.tolist() == [2000, 2001] and a.cpi.tolist() == [100.0, 200.0] and a.months.tolist() == [12, 11]
    a.to_csv(tmp_path / "cpi.csv", index=False)
    k, base = deflator(path=tmp_path / "cpi.csv")
    assert base == 2001 and k[2000] == 2.0 and k[2001] == 1.0
    assert deflator(2000, tmp_path / "cpi.csv")[0][2001] == 0.5


def test_committed_cpi_covers_the_sample_years():
    c = pd.read_csv(ROOT / "data" / "cpi_us_annual.csv").set_index("year")
    assert set(range(1975, 2026)) <= set(c.index)
    assert (c.cpi.loc[1975:2025] > 0).all() and c.cpi[2025] > c.cpi[1975] and (c.months >= 11).all()


# ---------------------------------------------------------------- saved outputs and the explorer page

OUT = ROOT / "archive" / "analysis" / "attribution" / "outputs"
saved = pytest.mark.skipif(not (OUT / "summary.json").exists(), reason="attribution outputs not built")


@saved
def test_saved_dollars_add_up_for_every_film():
    f = pd.read_csv(OUT / "films.csv")
    usd = f[[c for c in f.columns if c.startswith("usd_")]].sum(axis=1)
    assert np.allclose(usd, f.gross_usd - f.baseline_gross_usd, atol=2.0)
    ln = f[[c for c in f.columns if c.startswith("ln_")]].sum(axis=1) + f.unexplained
    assert np.allclose(ln + np.log(f.baseline_gross_usd / f.budget_usd), f.log_mult, atol=1e-3)


@saved
def test_saved_entity_dollars_match_the_film_split():
    f = pd.read_csv(OUT / "films.csv")
    fe = pd.read_csv(OUT / "film_entities.csv.gz")
    by_role = fe.groupby("role").usd.sum()
    for r in M.ROLES:
        assert np.isclose(by_role.get(r, 0.0), f[f"usd_{r}"].sum(), rtol=1e-6, atol=10)
    e = pd.read_csv(OUT / "entities.csv")
    assert (e.lo90 <= e.hi90).all() and not e.duplicated(["role", "id"]).any() and (e.films >= 2).all()


@saved
def test_attribution_page_runs():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from explorer import attribution_page\nattribution_page.attribution()\n", default_timeout=120).run()
    assert not at.exception, at.exception
    for view in ("By role", "People and companies", "One film", "Profit measures", "Checks"):
        at = AppTest.from_string("from explorer import attribution_page\nattribution_page.attribution()\n", default_timeout=120)
        at.session_state["attr_view"] = view
        at.run()
        assert not at.exception, (view, at.exception)
        at.session_state["attr_real"] = False          # nominal dollars
        at.run()
        assert not at.exception, (view, "nominal", at.exception)
