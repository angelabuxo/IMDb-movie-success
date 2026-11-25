"""The screenplay-text pipeline (notebooks/scripts_text/03, 04, 09) on synthetic text and films.

Covers name removal and cleaning (03), budget matching and balance (04) and the TF-IDF logistic
regression's leakage guard and verdict (09).
"""

from collections import Counter

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from analysis.scripts_text import s03_clean, s04_controls, s09_logreg

# ---------------------------------------------------------------- 03: cleaning and names

def test_label_names_strips_extensions_titles_and_headers():
    cues = ["JOHN (V.O.)", "MARY (CONT'D)", "CUT TO:", "DR. SMITH #2", "THE END", "INT. HOUSE", "O'BRIEN"]
    assert s03_clean.label_names(cues) == {"john", "mary", "smith", "o'brien"}


def test_name_mask_removes_dictionary_names_only_where_capitalised_mid_sentence():
    dict_words = {"will", "grace", "the", "bowl"}
    names = {"will", "grace", "zed"}
    toks, mask = s03_clean.name_mask("Will went home. I will go. Then Will left.", names, dict_words)
    assert list(zip(toks, mask)) == [("will", False), ("went", False), ("home", False), ("i", False),
                                     ("will", False), ("go", False), ("then", False), ("will", True),
                                     ("left", False)]
    toks, mask = s03_clean.name_mask("The bowl, said Zed. Zed's turn.", names, dict_words)
    assert dict(zip(toks, mask))["zed"] and dict(zip(toks, mask))["zed's"]   # not a word: always a name


def test_clean_lines_joins_contractions_and_counts_every_removal():
    counts = Counter()
    out = s03_clean.clean_lines("He does n't know.\nINT. HOUSE\nCONTINUED:\n(V.O.) hi there", counts,
                                headings={"INT. HOUSE"})
    assert "doesn't" in out and "HOUSE" not in out and "CONTINUED" not in out and "hi there" in out
    assert counts["contraction_joins"] == 1
    assert counts["heading_tokens"] == 2
    assert counts["technical_tokens"] >= 3                     # CONTINUED, V, O


def _row(tokens, is_name, is_stop, lemmas=None):
    return {"tokens": " ".join(tokens), "is_name": is_name, "is_name_ner": is_name, "is_stop": is_stop,
            "lemmas": " ".join(lemmas) if lemmas else None}


def test_stream_variants():
    r = _row(["john", "ran", "to", "the", "a", "dogs"], "100000", "001110", ["john", "run", "to", "the", "a", "dog"])
    assert s03_clean.stream(r, names="kept") == ["john", "ran", "to", "the", "dogs"]        # "a" is too short
    assert s03_clean.stream(r, names="labels") == ["ran", "to", "the", "dogs"]
    assert s03_clean.stream(r, names="labels", stopwords="removed") == ["ran", "dogs"]
    assert s03_clean.stream(r, names="labels", lemmatize="on") == ["run", "to", "the", "dog"]


# ---------------------------------------------------------------- 04: matching

def _films(seed=0, n=120):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "tconst": [f"tt{i:04d}" for i in range(n)],
        "group_A": rng.choice(["low", "high", None], n, p=[0.3, 0.5, 0.2]),
        "primary_genre": rng.choice(["Horror", "Drama"], n),
        "decade_bin": rng.choice(["1990s", "2000s"], n),
        "log_budget": rng.normal(17, 1, n),
        "year": rng.integers(1990, 2010, n),
        "log_votes": rng.normal(11, 1, n),
    })


@pytest.mark.parametrize("order_seed", [None, 3])
def test_match_is_exact_on_cell_within_caliper_and_without_replacement(order_seed):
    sc = _films()
    m = s04_controls.match(sc, caliper_sd=0.2, order_seed=order_seed)
    s = sc.set_index("tconst")
    assert set(m.treated) == set(sc.tconst[sc.group_A == "low"])          # every treated film gets a row
    ok = m.dropna(subset=["control"])
    assert ok.control.is_unique                                            # no control used twice
    assert (s.loc[ok.control, "group_A"] == "high").all()
    for t, c in zip(ok.treated, ok.control):
        assert s.loc[t, "primary_genre"] == s.loc[c, "primary_genre"]
        assert s.loc[t, "decade_bin"] == s.loc[c, "decade_bin"]
        assert abs(s.loc[t, "log_budget"] - s.loc[c, "log_budget"]) <= 0.2 * sc.log_budget.std() + 1e-12
    lab = s04_controls.matched_labels(sc, m)
    assert (lab == "low").sum() == (lab == "high").sum() == len(ok)


def test_smd():
    a = pd.Series([1.0, 2, 3, 4])
    assert s04_controls.smd(a, a) == 0
    assert s04_controls.smd(a + 1, a) > 0 > s04_controls.smd(a, a + 1)
    assert s04_controls.smd(pd.Series([1.0, 1]), pd.Series([1.0, 1])) == 0.0   # zero variance: no crash


# ---------------------------------------------------------------- 09: logistic regression

def test_tfidf_uses_training_documents_only():
    """The idf and the kept vocabulary must come from the training fold, or test documents leak in."""
    rng = np.random.default_rng(0)
    Xtr = sp.csr_matrix(rng.poisson(1.0, (30, 12)))
    Xte = sp.csr_matrix(rng.poisson(1.0, (10, 12)))
    ltr, lte = np.asarray(Xtr.sum(1)).ravel(), np.asarray(Xte.sum(1)).ravel()
    a_tr, _, keep_a = s09_logreg.tfidf(Xtr, ltr, Xte, lte, min_df=3)
    Xte2 = sp.csr_matrix(rng.poisson(5.0, (10, 12)))              # completely different test fold
    b_tr, _, keep_b = s09_logreg.tfidf(Xtr, ltr, Xte2, np.asarray(Xte2.sum(1)).ravel(), min_df=3)
    assert np.array_equal(keep_a, keep_b)
    assert np.allclose(a_tr.toarray(), b_tr.toarray())


def test_tfidf_follows_the_course_definitions():
    X = sp.csr_matrix(np.array([[2, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1]]))
    L = np.array([3, 2, 2, 1])
    tr, _, keep = s09_logreg.tfidf(X, L, X, L, min_df=2)
    assert keep.tolist() == [0, 2]                                 # term 1 is in one document only
    assert tr[0, 0] == pytest.approx(2 / 3 * np.log(4 / 2))
    assert np.allclose(tr[:, 1].toarray(), 0)                      # in every document: idf = ln(1) = 0


@pytest.mark.parametrize("text, base, expected", [
    ([0.60, 0.65], [0.62, 0.70], True),     # overlap
    ([0.71, 0.75], [0.60, 0.70], False),    # text clearly above
    ([0.50, 0.55], [0.60, 0.70], False),    # text clearly below
    ([0.60, 0.70], [0.70, 0.80], True),     # touching counts as overlap
])
def test_no_clear_gain(text, base, expected):
    assert s09_logreg.no_clear_gain({"auc_text_repeat_range": text, "auc_baseline_repeat_range": base}) is expected


def test_s01_imdb_fields_fall_back_to_the_data_folder(tmp_path, monkeypatch):
    """Regression: without the private clean table, Step 1's load() had no 'year', and its tables and figures
    failed in the explorer. The IMDb fields now come from movies_features.csv, the file Step 1 reads them from."""
    import pandas as pd
    from analysis.scripts_text import s01_groups
    pd.DataFrame({"tconst": ["tt1", "tt2", "tt2"], "primaryTitle": ["A", "B", "B"], "y": [1999, 2005, 2005],
                  "genres": ["Drama", "Horror,Comedy", "Horror,Comedy"], "b": [1, 2, 2]}).to_csv(
        tmp_path / "movies_features.csv", index=False)
    monkeypatch.setattr(s01_groups, "FEATURES", tmp_path)
    f = s01_groups._imdb_fields(None)
    assert f.set_index("tconst").year.to_dict() == {"tt1": 1999, "tt2": 2005}          # one row per film
    assert list(f.columns) == ["tconst", "title_imdb", "year", "genres"]
    private = pd.DataFrame({"tconst": ["tt9"], "title_imdb": ["Z"], "year": [1980], "genres": ["Drama"], "x": [0]})
    assert s01_groups._imdb_fields(private).tconst.tolist() == ["tt9"]                  # the private table wins
    monkeypatch.setattr(s01_groups, "FEATURES", tmp_path / "missing")
    assert s01_groups._imdb_fields(None) is None
