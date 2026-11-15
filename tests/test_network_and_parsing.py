"""Network scripts behind notebooks/movie_graph, network_reach and what_predicts_profit, the screenplay
features behind scripts_and_reviews, and the Wikipedia budget/gross parser that every profit figure
comes from. Synthetic inputs only.
"""

import math

import numpy as np
import pytest
import scipy.sparse as sp

import pair_returns as pr
import roi_similarity as rs
import screenplay_features as sf
import team_history as th


# ---------------------------------------------------------------- budget and gross (scripts/imdb/parse_infobox.py)

@pytest.mark.parametrize("raw, value, cur", [
    ("$10 million", 10e6, "USD"),
    ("US$25.5 million", 25.5e6, "USD"),
    ("$1,200,000", 1.2e6, "USD"),
    ("$150–200 million", 175e6, "USD"),                       # a range gives its midpoint
    ("$150-200 million", 175e6, "USD"),
    ("$800,000–1.2 million", 1e6, "USD"),                     # low end in units, high end in millions
    ("{{US$|40 million}}", 40e6, "USD"),
    ("£3 million ($4.5 million)", 4.5e6, "USD"),              # the USD figure wins over the first one
    ("€12 million", 12e6, "EUR"),
    ("₹50 crore", 5e8, "INR"),
    ("$2.1 billion", 2.1e9, "USD"),
    ("$30 million<ref>Variety, 2001</ref>", 30e6, "USD"),     # citations stripped
    ("$30&nbsp;million", 30e6, "USD"),
])
def test_money(infobox, raw, value, cur):
    v, c, _ = infobox["money"](raw)
    assert c == cur and v == pytest.approx(value)


@pytest.mark.parametrize("raw", ["", None, "Unknown", "not disclosed"])
def test_money_without_a_figure(infobox, raw):
    assert infobox["money"](raw)[:2] == (None, None)


@pytest.mark.parametrize("text, flags", [
    ("$50 million (domestic)", "regional"),
    ("$50 million (North America)", "regional"),
    ("$300 million (worldwide)", ""),
    ("$12 million (US rentals)", "rentals;regional"),
    ("$5 million (net)", "net_gross"),
])
def test_scope_flags_partial_grosses(infobox, text, flags):
    assert infobox["scope"](text) == flags


def test_names_reads_links_and_list_templates(infobox):
    assert infobox["names"]("[[Warner Bros.|Warner Bros. Pictures]]<br>[[Legendary Pictures]]") == [
        "Warner Bros.", "Legendary Pictures"]
    assert infobox["names"]("{{Plainlist|\n* [[Tom Hanks]]\n* [[Meg Ryan]]}}") == ["Tom Hanks", "Meg Ryan"]
    assert infobox["names"]("{{ubl|A24|Plan B}}") == ["A24", "Plan B"]
    assert infobox["names"]("Studio One\nStudio Two") == ["Studio One", "Studio Two"]


def test_names_splits_plain_text_on_br(infobox):
    """Regression (fixed 2026-11-15): strip_code() deletes <br> tags, so splitting after it merged unlinked
    names ('Studio OneStudio Two'): 48 infoboxes in the 100k+ tier, 831 in the 10k+ tier."""
    assert infobox["names"]("Studio One<br/>Studio Two") == ["Studio One", "Studio Two"]
    assert infobox["names"]("Jane Doe<br>John Roe<BR />Kim Lee") == ["Jane Doe", "John Roe", "Kim Lee"]
    assert infobox["names"]("''Jane Doe''<br>John Roe<ref>x</ref>") == ["Jane Doe", "John Roe"]
    # entities are decoded before splitting, so '&nbsp;' and '&ndash;' don't cut a name at their ';'
    assert infobox["names"]("Corinth Films (1979 U.S.&nbsp;release)<br>Acme") == ["Corinth Films (1979 U.S.\xa0release)", "Acme"]
    assert infobox["names"]("Hiller Productions &ndash; Layton") == ["Hiller Productions – Layton"]
    assert infobox["names"]("Jane Doe<br>(as associate producer)") == ["Jane Doe"]


# ---------------------------------------------------------------- film-film projection and distance rings

def _members():
    # Films f0..f4 in a path f0-f1-f2-f3, f4 isolated. p1 is on f0 twice (director and writer).
    return {("director", "p1"): {"f0", "f1"}, ("writer", "p1"): {"f0", "f1"}, ("cast", "p2"): {"f0", "f1"},
            ("cast", "p3"): {"f1", "f2"}, ("editor", "p4"): {"f2", "f3"}, ("studio", "s1"): {"f0", "f3", "f4"}}


def test_projection_counts_distinct_shared_people():
    ids = ["f0", "f1", "f2", "f3", "f4"]
    a = pr.projection(ids, _members(), pr.PEOPLE).toarray()
    assert a[0, 1] == 2                                         # p1 counted once despite two roles, plus p2
    assert a[1, 2] == 1 and a[2, 3] == 1 and a[0, 3] == 0      # the studio is not a person
    assert (a == a.T).all() and (np.diag(a) == 0).all() and a[4].sum() == 0
    c = pr.projection(ids, _members(), pr.COMPANIES).toarray()
    assert c[0, 3] == c[0, 4] == c[3, 4] == 1


def test_rings_are_shortest_path_distances():
    ids = ["f0", "f1", "f2", "f3", "f4"]
    a1 = pr.binary(pr.projection(ids, _members(), pr.PEOPLE))
    out, reached = pr.rings(a1, 4)
    pairs = lambda m: {tuple(sorted(p)) for p in zip(*m.nonzero())}
    assert pairs(out[1]) == {(0, 1), (1, 2), (2, 3)}
    assert pairs(out[2]) == {(0, 2), (1, 3)}
    assert pairs(out[3]) == {(0, 3)}
    assert out[4].nnz == 0
    assert reached[4, 4] == 1 and reached[0, 4] == 0            # f4 is unconnected: "far"


def test_pair_corr_and_far_corr_match_brute_force():
    rng = np.random.default_rng(0)
    n = 30
    m = sp.random(n, n, density=0.15, random_state=1, format="csr")
    m = pr.binary(m + m.T)
    m.setdiag(0)
    m.eliminate_zeros()
    z = pr.standardise(rng.normal(size=n))
    i, j = m.nonzero()
    assert pr.pair_corr(m, z[:, None])[0] == pytest.approx(np.mean(z[i] * z[j]))
    reached = pr.binary(m + sp.identity(n, format="csr"))
    far = [(a, b) for a in range(n) for b in range(n) if reached[a, b] == 0]
    assert pr.far_corr(reached, z[:, None])[0] == pytest.approx(np.mean([z[a] * z[b] for a, b in far]))


@pytest.mark.parametrize("fn", ["pr", "rs", "th"])
def test_within_stratum_shuffles_never_mix_strata(fn):
    rng = np.random.default_rng(1)
    strata = np.array([2, 0, 1, 0, 2, 1, 0, 2, 2, 1])
    y = np.arange(10, dtype=float)
    for _ in range(30):
        if fn == "pr":
            yp = pr.shuffles(y, strata, 1, rng)[:, 0]
        elif fn == "rs":
            yp = rs.shuffle(y, strata, rng)
        else:
            yp = y[th.within(strata, rng)]
        for s in np.unique(strata):
            assert sorted(yp[strata == s]) == sorted(y[strata == s])


def test_normal_scores_keep_order_and_are_centred():
    y = np.array([5.0, 1.0, 100.0, 3.0, 2.0])
    z = rs.normal_scores(y)
    assert np.array_equal(np.argsort(z), np.argsort(y))
    assert z.mean() == pytest.approx(0, abs=1e-12) and z.max() < 2   # the outlier no longer dominates


def test_residuals_remove_budget_decade_and_genre():
    rng = np.random.default_rng(2)
    movies = {f"m{i}": {"budget": float(rng.uniform(1e6, 1e8)), "year": int(rng.integers(1980, 2020)),
                        "genres": set(rng.choice(["Horror", "Drama", "Comedy"], 2, replace=False))}
              for i in range(200)}
    ids = sorted(movies)
    y = np.array([2 - 0.3 * math.log(movies[i]["budget"]) + ("Horror" in movies[i]["genres"]) for i in ids])
    assert np.allclose(rs.residuals(ids, movies, y), 0, atol=1e-8)


def test_role_similarity_is_high_for_homogeneous_groups():
    index = {f"f{i}": i for i in range(12)}
    groups = [{"f0", "f1", "f2"}, {"f3", "f4", "f5"}, {"f6", "f7", "f8"}, {"f9", "f10", "f11"}]
    role = rs.Role(groups, index)
    y_same = np.repeat([-2.0, -1.0, 1.0, 2.0], 3) + np.random.default_rng(0).normal(0, 0.05, 12)
    sim, loo = role.stats(y_same)
    assert sim > 0.9 and loo > 0.9
    sim, _ = role.stats(np.random.default_rng(1).normal(size=12))
    assert sim < 0.5


# ---------------------------------------------------------------- screenplay features (scripts_and_reviews)

@pytest.mark.parametrize("raw, name", [
    ("JOHN (V.O.)", "JOHN"), ("Mary (CONT'D)", "MARY"), ("CUT TO:", ""), ("THE END", ""),
    ("INT. HOUSE", ""), ("O'BRIEN", "O'BRIEN"), ("DR. JONES", "DR. JONES"),
])
def test_clean_name(raw, name):
    assert sf.clean_name(raw) == name


def test_parse_and_network():
    def scene(*lines):
        return "<scene>" + "".join(f"<character>{a}</character><dialogue>{t}</dialogue>" for a, t in lines) + "</scene>"
    script = (scene(("ANN", "hi there"), ("BOB", "hello"), ("CAL", "hey")) * 3
              + scene(("ANN", "one more"), ("DAN", "only line")))
    scenes, total = sf.parse(script)
    assert len(scenes) == 4 and scenes[0][0] == ("ANN", "hi there")
    g, lines = sf.network(scenes)
    assert set(g) == {"ANN", "BOB", "CAL"}                       # DAN has fewer than MIN_LINES lines
    assert g["ANN"]["BOB"]["weight"] == 3 and lines["ANN"] == 4


def test_gini():
    assert sf.gini([5, 5, 5, 5]) == pytest.approx(0)
    assert sf.gini([0, 0, 0, 10]) == pytest.approx(0.75)        # (n - 1) / n for one speaker with every line
    assert sf.gini([]) == 0.0


def test_arc_bins_and_descriptors():
    a, d = sf.arc(np.full(400, 0.2))
    assert len(a) == sf.BINS and d["sent_swing"] == pytest.approx(0) and d["sent_ending"] == pytest.approx(0)
    a, d = sf.arc(np.linspace(-1, 1, 400))
    assert d["sent_ending"] > 0 and d["sent_low_point"] == 0.0
