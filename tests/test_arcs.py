"""The screenplay-arcs pipeline (notebooks/arcs/02, 05, 06, 07, 08) on synthetic scripts and films.

Covers the maths each notebook reports: sentence parsing (02), arc resampling, smoothing and the D1/D2
descriptors (05), the scene-order shuffle null (06), the Frisch-Waugh association with its within-cell
null and bootstrap (07), and out-of-fold prediction with its splits (08).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from analysis.arcs import s02_units, s05_arcs, s06_shuffle, s07_primary, s08_prediction
from analysis.arcs.config import cfg

# ---------------------------------------------------------------- 02: text units

SCRIPT = (
    "<scene><stage_direction>12 INT. KITCHEN - NIGHT</stage_direction>"
    "<scene_description>John enters. He does n't look happy!</scene_description>"
    "<character>JOHN</character><parenthetical>(quietly)</parenthetical>"
    "<dialogue>I ca n't do this. Really? Yes.</dialogue></scene>"
    "<scene><dialogue>The dog&#39;s bowl. It 's empty.</dialogue></scene>"
    "<scene><character>X</character></scene>"
)


def test_parse_splits_layers_sentences_and_scenes():
    rows, c = s02_units.parse(SCRIPT)
    got = [(r["scene"], r["layer"], r["pos"], r["text"]) for r in rows]
    assert got == [
        (0, "action", 0, "John enters."),
        (0, "action", 1, "He doesn't look happy!"),
        (0, "dialogue", 2, "I can't do this."),
        (0, "dialogue", 3, "Really?"),
        (0, "dialogue", 4, "Yes."),
        (1, "dialogue", 0, "The dog's bowl."),
        (1, "dialogue", 1, "It's empty."),
    ]
    assert c["n_scene_tags"] == 3                     # an empty scene still counts as a scene tag
    assert c["joins"] == 3 and c["lone_nt_after"] == 0  # does n't, ca n't, It 's all joined
    assert c["headings"] == 1 and c["numbered_headings"] == 1
    assert c["words_action"] == 6 and c["words_dialogue"] == 11


def test_parse_never_scores_headings_cues_or_parentheticals():
    rows, _ = s02_units.parse(SCRIPT)
    text = " ".join(r["text"] for r in rows)
    for absent in ("KITCHEN", "JOHN", "quietly"):
        assert absent not in text


def test_split_sentences_drops_wordless_fragments():
    # Known quirk: the split regex eats a closing quote or bracket after the end punctuation. Harmless:
    # words and sentiment are unchanged, but a test will notice if the regex changes.
    assert s02_units.split_sentences('He said "Go." Then left!  ... ok') == ['He said "Go.', "Then left!", "ok"]
    assert s02_units.split_sentences("... !!!") == []


# ---------------------------------------------------------------- 05: arc maths

@pytest.mark.parametrize("n, T", [(1, 100), (7, 100), (100, 100), (250, 100), (33, 50)])
def test_resample_matrix_is_an_overlap_weighted_mean(n, T):
    R = s05_arcs.resample_matrix(n, T)
    assert R.shape == (T, n)
    assert np.allclose(R.sum(1), 1) and (R >= 0).all()
    assert np.allclose(R.sum(0), T / n)               # every scene gets weight equal to its share of story time


def test_resample_matrix_is_the_identity_when_scenes_match_the_grid():
    assert np.allclose(s05_arcs.resample_matrix(100, 100), np.eye(100))


def test_smooth_matrix_is_a_truncated_centred_moving_average():
    S = s05_arcs.smooth_matrix(20, 5)
    assert np.allclose(S.sum(1), 1)
    assert np.count_nonzero(S[10]) == 5 and np.count_nonzero(S[0]) == 3   # truncated at the ends
    assert np.allclose(S[10, 8:13], 0.2)


def test_describe_on_flat_rising_and_falling_arcs():
    flat = s05_arcs.describe(np.full(40, 0.3))
    assert flat["D1"][0] == pytest.approx(0) and flat["D2"][0] == pytest.approx(0)
    assert flat["mean_level"][0] == pytest.approx(0.3)
    up = s05_arcs.describe(np.linspace(-1, 1, 40))
    down = s05_arcs.describe(np.linspace(1, -1, 40))
    assert up["D2"][0] > 0 and down["D2"][0] == pytest.approx(-up["D2"][0])
    assert up["D1"][0] == pytest.approx(down["D1"][0])
    assert down["min_position"][0] > 0.9 and up["min_position"][0] < 0.1
    assert up["sign_changes"][0] == 1


def test_describe_mean_level_is_the_scene_mean():
    x = np.random.default_rng(0).normal(size=37)
    assert s05_arcs.describe(x)["mean_level"][0] == pytest.approx(x.mean())


def test_describe_batch_matches_one_at_a_time():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 60))
    batch = s05_arcs.describe(X)
    for i in range(5):
        one = s05_arcs.describe(X[i])
        assert batch["D1"][i] == pytest.approx(one["D1"][0]) and batch["D2"][i] == pytest.approx(one["D2"][0])


# ---------------------------------------------------------------- 06: scene-order shuffle null

def test_null_for_uses_the_same_maths_as_describe():
    """Step 6's matrix shortcut must give Step 5's real D1 and D2 for the unshuffled script."""
    a = cfg()["arc"]
    x = np.random.default_rng(2).normal(size=45)
    res = s06_shuffle.null_for(x, np.random.default_rng(0), 50, a["time_points"], a["smoothing_window"], a["edge_share"])
    d = s05_arcs.describe(x)
    assert res["D1"] == pytest.approx(d["D1"][0]) and res["D2"] == pytest.approx(d["D2"][0])
    assert len(res["D1_null"]) == 50


def test_null_for_flags_an_ordered_story_and_not_a_random_one():
    a = cfg()["arc"]
    args = (400, a["time_points"], a["smoothing_window"], a["edge_share"])
    rng = np.random.default_rng(3)
    ordered = np.sin(np.linspace(0, 2 * np.pi, 80)) + rng.normal(0, 0.3, 80)
    r = s06_shuffle.null_for(ordered, np.random.default_rng(0), *args)
    assert (r["D1_null"] >= r["D1"]).mean() < 0.01             # far more structure than shuffled orders
    noise = rng.normal(size=80)
    r = s06_shuffle.null_for(noise, np.random.default_rng(0), *args)
    assert 0.05 < (r["D1_null"] >= r["D1"]).mean() < 0.95


# ---------------------------------------------------------------- 07: primary association

def test_rho_is_spearman():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=100), rng.normal(size=100)
    b[:5] = b[5]                                               # ties
    assert s07_primary._rho(a, b) == pytest.approx(spearmanr(a, b).statistic)


def _films(n, seed, planted):
    rng = np.random.default_rng(seed)
    cell = pd.Series(rng.choice([f"g{i} | d{j}" for i in range(4) for j in range(3)], n))
    eff = cell.map({c: rng.normal(0, 1) for c in cell.unique()}).to_numpy()
    budget = rng.normal(17, 1, n)
    d = eff + 0.5 * budget + rng.normal(0, 1, n)                 # descriptor confounded with the controls
    y = 2 * eff + budget + planted * (d - eff - 0.5 * budget) + rng.normal(0, 1, n)
    return pd.DataFrame({"cell": cell, "D": d, "y": y, "log_budget": budget})


def test_association_removes_confounding_and_finds_a_planted_effect():
    rng = np.random.default_rng(0)
    null = s07_primary.association(_films(400, 1, 0.0), "D", "y", ["log_budget"], 300, 100, rng)
    raw = spearmanr(_films(400, 1, 0.0).D, _films(400, 1, 0.0).y).statistic
    assert raw > 0.5                                           # the raw correlation is pure confounding
    assert abs(null["rho"]) < 0.12 and null["p"] > 0.05
    assert null["ci_low"] < 0 < null["ci_high"]
    hit = s07_primary.association(_films(400, 1, 0.5), "D", "y", ["log_budget"], 300, 100, rng)
    assert hit["rho"] > 0.25 and hit["p"] < 0.01 and hit["ci_low"] > 0
    assert hit["rho"] == pytest.approx(hit["spearman_check"])


class _RecordingRng:
    """Wraps a Generator and records every permutation the null draws."""

    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.permuted = []

    def permutation(self, x):
        self.permuted.append(np.array(x, copy=True))
        return self._rng.permutation(x)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def test_association_null_permutes_within_cells_only():
    """The pre-registered null shuffles the outcome only among films of the same genre x decade cell."""
    df = _films(200, 2, 0.0)
    rng = _RecordingRng(0)
    s07_primary.association(df, "D", "y", ["log_budget"], 20, 2, rng)
    cells = df.cell.to_numpy()
    assert len(rng.permuted) == 20 * df.cell.nunique()
    for idx in rng.permuted:
        assert idx.ndim == 1 and len(idx) > 0, "a permutation over all films, not a cell's index array"
        assert len(set(cells[idx])) == 1
    assert sorted(np.concatenate(rng.permuted[: df.cell.nunique()])) == list(range(len(df)))


def test_self_test_recovers_planted_and_rejects_noise():
    t = s07_primary.self_test().set_index("planted partial effect")
    assert t.loc[0.0, "p"] > 0.05 and t.loc[0.25, "p"] < 0.01
    assert t.loc[0.0, "ci_low"] < 0 < t.loc[0.0, "ci_high"]


# ---------------------------------------------------------------- 08: prediction

def test_make_splits_partitions_every_repeat_and_stratifies_by_cell():
    cells = pd.Series(np.repeat(["a", "b", "c"], [50, 30, 20]))
    splits = s08_prediction.make_splits(cells, folds=5, repeats=3, seed=0)
    assert len(splits) == 3 and all(len(r) == 5 for r in splits)
    for rep in splits:
        test = np.concatenate([te for _, te in rep])
        assert sorted(test) == list(range(100))                # each film tested exactly once per repeat
        for tr, te in rep:
            assert not set(tr) & set(te)
            assert (cells[te] == "a").sum() == 10              # 50 / 5 folds
    assert not np.array_equal(splits[0][0][1], splits[1][0][1])  # repeats differ


def test_oof_r2_is_near_one_for_a_linear_outcome_and_not_positive_for_noise():
    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(200), rng.normal(size=(200, 3))])
    splits = s08_prediction.make_splits(pd.Series(["x"] * 200), 5, 2, 0)
    y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(0, 0.01, 200)
    assert (s08_prediction.oof_r2(X, y, splits) > 0.999).all()
    assert (s08_prediction.oof_r2(X, rng.normal(size=200), splits) < 0.05).all()


@pytest.mark.slow
def test_prediction_self_test():
    t = s08_prediction.self_test().set_index("planted weight on D1")
    assert not t.loc[0.0, "adds"] and t.loc[0.3, "adds"]
    assert t.loc[0.3, "gain"] > 0.03
