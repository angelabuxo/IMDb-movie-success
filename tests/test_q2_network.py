import numpy as np
import pandas as pd

import scripts.network.q2_network as q2
from explorer.graph_pages import _community_summary, _network_assortativity


def test_has_any_genre_supports_union():
    assert q2.has_any_genre("Horror,Thriller", ["Horror", "Supernatural"])
    assert not q2.has_any_genre("Drama,Comedy", ["Horror", "Supernatural"])


def test_empty_genre_sample_is_explicit():
    empty = pd.DataFrame(columns=["imdb_id"])
    result = q2.test_sample(empty, {}, 20, 2805)
    assert len(result) == 5
    assert result["combined"]["pairs"] == 0
    assert result["combined"]["assortativity"] is None


def test_shuffle_within_era_preserves_strata():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    eras = np.array(["a", "a", "b", "b"])
    rng = np.random.default_rng(2805)
    shuffled = q2.shuffle_within_era(y, eras, rng)
    assert set(shuffled[:2]) == {1.0, 2.0}
    assert set(shuffled[2:]) == {3.0, 4.0}


def test_make_graph_top_three_actors_and_combined():
    ids = {"f1", "f2", "f3"}
    members = {role: {"e": set(ids)} for role in q2.ROLES}
    pairs = q2.make_graph(ids, members, "actor")
    assert set(pairs) == {("f1", "f2"), ("f1", "f3"), ("f2", "f3")}
    combined = q2.combined_graph(ids, members)
    assert set(combined) == set(pairs)
    assert all(len(entities) == 4 for entities in combined.values())


def test_assortativity_positive_and_no_edges():
    index = {"f1": 0, "f2": 1, "f3": 2}
    y = np.array([1.0, 2.0, 3.0])
    value, pairs = q2.assortativity({("f1", "f2"): {"e"}}, index, y)
    assert pairs == 1
    assert np.isnan(value)
    assert q2.assortativity({}, index, y)[1] == 0


def test_community_summary_reports_return_statistics():
    ids = np.array(["f1", "f2", "f3"])
    by = pd.DataFrame(
        {"label": ["One", "Two", "Three"], "log_multiple": [1.0, 3.0, np.nan]},
        index=["f1", "f2", "f3"],
    )
    table = _community_summary(ids, by, np.array([0, 0, 1]))
    first = table.set_index("community").loc[1]
    second = table.set_index("community").loc[2]
    assert first["films"] == 2
    assert first["total_ln_multiple"] == 4.0
    assert first["average"] == 2.0
    assert first["median"] == 2.0
    assert first["minimum"] == 1.0
    assert first["maximum"] == 3.0
    assert first["range"] == 2.0
    assert second["films"] == 1


def test_network_assortativity_uses_all_retained_pairs():
    ids = np.array(["f1", "f2", "f3", "f4"])
    by = pd.DataFrame({
        "label": ["One", "Two", "Three", "Four"],
        "genres": ["Action,Drama", "Action", "Drama", "Comedy"],
        "year": [2000, 2001, 2002, 2003],
        "budget_usd": [10.0, 10.0, 10.0, 10.0],
        "log_multiple": [1.0, 1.2, -1.0, -1.2],
    }, index=ids)
    result = _network_assortativity(ids, by, np.array([0, 1, 2]), np.array([1, 2, 3]),
                                    shuffles=20, seed=2805)
    assert result["pairs"] == 3
    assert result["observed"] is not None
