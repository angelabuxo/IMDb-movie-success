"""The Streamlit explorer: every page runs without an exception, on a small synthetic data folder.

Pages are driven with Streamlit's AppTest, headless. The data folder is built here (a few dozen films, people
and studios, two screenplays, a very wide JSON) and handed to the explorer through IMDB_DATA_DIR, so nothing
reads data/raw. Pages that show the committed notebook outputs (text, network results, checks) read
those as they are in the repo. Clicks on charts can't be sent headlessly; their state hand-overs are tested
by setting the session state the click would set.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

TIMEOUT = 120


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """A data folder shaped like data/raw."""
    rng = np.random.default_rng(2805)
    root = tmp_path_factory.mktemp("explorer_data")
    feat, graph = root / "features", root / "features" / "graph"
    graph.mkdir(parents=True)

    films = [f"tt{1000000 + i}" for i in range(40)]
    directors = [f"nm{100 + i}" for i in range(6)]
    writers = [f"nm{200 + i}" for i in range(6)]
    actors = [f"nm{300 + i}" for i in range(18)]
    studios = [f"company:Q{900 + i}" for i in range(6)]
    distributors = ["company:name:Dist A", "company:name:Dist B"]
    genres = ["Drama", "Comedy", "Horror", "Action"]

    budget = rng.uniform(1e6, 2e8, len(films)).round()
    gross = (budget * np.exp(rng.normal(0.5, 1.2, len(films)))).round()
    gross[3] = np.nan                                   # a film with no gross
    movies = pd.DataFrame({
        "id": films, "type": "movie", "label": [f"Film {i}" for i in range(len(films))],
        "year": rng.integers(1995, 2021, len(films)), "votes": rng.integers(10_000, 900_000, len(films)),
        "rating": rng.uniform(4, 9, len(films)).round(1), "budget_usd": budget, "gross_usd": gross,
        "multiple": (gross / budget).round(4), "rt": rng.integers(10, 100, len(films)),
        "mc": rng.integers(20, 95, len(films)),
        "genres": [",".join(rng.choice(genres, 2, replace=False)) for _ in films],
    })
    movies["tier"] = np.where(movies.votes >= 100_000, "100k+", "10k-100k")
    movies["clean"] = movies.gross_usd.notna().astype(int)
    movies.loc[0, ["id", "label"]] = ["tt0119094", "Face/Off"]   # matches the synthetic screenplay
    films[0] = "tt0119094"
    people = pd.DataFrame({"id": directors + writers + actors, "type": "person",
                           "label": [f"Person {p}" for p in directors + writers + actors],
                           "birth_year": 1960, "death_year": np.nan, "roles": "director"})
    companies = pd.DataFrame({"id": studios + distributors, "type": "company",
                              "label": [f"Studio {i}" for i in range(6)] + ["Dist A", "Dist B"],
                              "qid": [s.split(":")[-1] for s in studios] + [None, None]})
    nodes = pd.concat([movies, people, companies], ignore_index=True)
    nodes.to_csv(graph / "nodes.csv", index=False)

    rows = []
    for f in films:
        rows.append((rng.choice(directors), f, "director", 1, "imdb"))
        rows.append((rng.choice(writers), f, "writer", 1, "imdb"))
        for k, a in enumerate(rng.choice(actors, 3, replace=False), 1):
            rows.append((a, f, "actor", k, "imdb"))
        for s in rng.choice(studios, 2, replace=False):
            rows.append((s, f, "studio", 1, "wikipedia"))
        rows.append((rng.choice(distributors), f, "distributor", 1, "wikipedia"))
    pd.DataFrame(rows, columns=["source", "target", "role", "ordering", "origin"]).to_csv(graph / "edges.csv", index=False)
    pd.DataFrame({"qid": [s.split(":")[-1] for s in studios], "name": [f"Studio {i}" for i in range(6)],
                  "method": "wikidata"}).to_csv(graph / "companies.csv", index=False)

    track = rng.normal(0, 1, len(films))
    pd.DataFrame({"tconst": films, "year": movies.year, "cast_track": track,
                  "director_track": np.where(np.arange(len(films)) % 3, track, np.nan),
                  "director_track_n": np.where(np.arange(len(films)) % 3, 2, 0)}).to_csv(
        feat / "film_network_features.csv", index=False)
    # one row per film with 600 word columns: wider than the inspector sends to the browser
    words = {f: {"counts": {f"w{j}": int(j % 7) for j in range(600)}} for f in films[:5]}
    (feat / "wide_words.json").write_text(json.dumps(words))

    # release months for all but the last two films (scripts/imdb/parse_infobox.py), split over the two files
    months = pd.DataFrame({"tconst": films[:38], "title": [f"Film {i}" for i in range(38)], "studios": "Studio 0|Studio 1",
                           "distributors": "Dist A", "producers": "Person P", "starring": "Person A|Person B",
                           "release_month": rng.integers(1, 13, 38)})
    months.iloc[:20].to_csv(feat / "wiki_infobox.csv", index=False)
    months.iloc[15:].to_csv(feat / "wiki_infobox_10k.csv", index=False)

    # tag relevance for the first 30 films over 12 tags, 4 of them genre tags (archive/scripts/reviews/tag_shares.py)
    tag_names = ["drama", "comedy", "horror", "action", "time travel", "heist", "original", "boring",
                 "space", "true", "none", "nan"]
    rel = rng.uniform(0, 1, (30, len(tag_names))).astype(np.float32)
    rel[:, 6] = 1.0                                     # a tag that is strong on every film
    np.save(feat / "tag_relevance.npy", rel)
    pd.DataFrame({"tconst": films[:30]}).to_csv(feat / "tag_relevance_ids.csv", index=False)
    pd.DataFrame({"tag": tag_names}).to_csv(feat / "tag_relevance_tags.csv", index=False)
    gs = pd.DataFrame({"tconst": films[:30], "primaryTitle": movies.label[:30], "y": movies.year[:30],
                       "genres": movies.genres[:30], "tier": movies.tier[:30]})
    for j, g in enumerate(["Drama", "Comedy", "Horror", "Action"]):
        gs["rel_" + g] = rel[:, j]
    gs.iloc[::-1].to_csv(feat / "genre_shares.csv", index=False)   # another row order than the matrix
    # plot-predicted tags for every film, including the ten the Tag Genome doesn't cover (archive/scripts/reviews/plot_tags.py)
    prel = rng.uniform(0, 0.7, (40, len(tag_names))).astype(np.float32)
    np.save(feat / "plot_tag_relevance.npy", prel)
    pd.DataFrame({"tconst": films, "out_of_fold": [1] * 30 + [0] * 10}).to_csv(feat / "plot_tag_relevance_ids.csv", index=False)
    ps = pd.DataFrame({"tconst": films, "primaryTitle": movies.label, "y": movies.year, "genres": movies.genres,
                       "has_tag_genome": [1] * 30 + [0] * 10})
    for j, g in enumerate(["Drama", "Comedy", "Horror", "Action"]):
        ps["rel_" + g] = prel[:, j]
    ps.to_csv(feat / "plot_genre_shares.csv", index=False)

    ms = root / "scripts" / "moviesum"
    ms.mkdir(parents=True)

    def script(names):
        scenes = []
        for i in range(30):
            a, b = names[i % len(names)], names[(i + 1) % len(names)]
            scenes.append(f"<scene><stage_direction>INT. ROOM {i} - DAY</stage_direction>"
                          f"<scene_description>Something happens in scene {i}.</scene_description>"
                          f"<character>{a}</character><dialogue>Line {i} from {a}.</dialogue>"
                          f"<character>{b} (CONT'D)</character><dialogue>Reply {i}.</dialogue></scene>")
        return "".join(scenes)

    lines = [{"movie_name": "Face_Off_1997", "imdb_id": "tt0119094", "script": script(["SEAN", "CASTOR", "EVE"]),
              "summary": "Two men swap faces."},
             {"movie_name": "Film_Two_2005", "imdb_id": "tt1000001", "script": script(["ANNA", "BEN"]), "summary": ""}]
    (ms / "train.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")

    # embedding maps: two spaces, four methods, one far outlier per map, a film with no multiple
    maps = []
    for space in ["Screenplays", "Genre shares"]:
        for method in ["PCA", "Isomap", "t-SNE", "UMAP"]:
            xy = rng.normal(0, 1, (len(films), 2))
            xy[5] = [80, -80]
            maps.append(pd.DataFrame({"space": space, "method": method, "tconst": films, "x": xy[:, 0], "y": xy[:, 1],
                                      "title": movies.label, "year": movies.year,
                                      "primary_genre": movies.genres.str.split(",").str[0],
                                      "log_mult": np.log(movies.multiple), "log_mult_net": rng.normal(0, 1, len(films))}))
    pd.concat(maps).to_csv(feat / "embedding_maps.csv", index=False)
    pd.DataFrame([{"space": s, "method": m, "films": len(films), "neighbours kept": .3, "distance rank": .5,
                   "profit smoothness": .1, "full-space smoothness": .08, "net smoothness": .05,
                   "full-space net smoothness": .04} for s in ["Screenplays", "Genre shares"]
                  for m in ["PCA", "Isomap", "t-SNE", "UMAP"]]).to_csv(feat / "embedding_map_scores.csv", index=False)

    return root


@pytest.fixture(scope="module")
def explorer(data_dir):
    """Point the explorer at the synthetic folders and import it fresh (its paths are read at import)."""
    old = {k: os.environ.get(k) for k in ("IMDB_DATA_DIR",)}
    os.environ["IMDB_DATA_DIR"] = str(data_dir)
    for m in [m for m in sys.modules if m == "explorer" or m.startswith("explorer.")]:
        del sys.modules[m]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import explorer.common
    assert explorer.common.DATA == data_dir
    yield data_dir
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    for m in [m for m in sys.modules if m == "explorer" or m.startswith("explorer.")]:
        del sys.modules[m]


def page(module: str, fn: str, state: dict | None = None) -> AppTest:
    at = AppTest.from_string(f"from explorer import {module}\n{module}.{fn}()\n", default_timeout=TIMEOUT)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    return at.run()


def ok(at: AppTest):
    assert not at.exception, [e.value for e in at.exception]


def widget(at: AppTest, kind: str, label: str):
    hits = [w for w in getattr(at, kind) if w.label.startswith(label)]
    assert hits, f"no {kind} labelled {label!r}: {[w.label for w in getattr(at, kind)]}"
    return hits[0]


PAGES = [("pages", "home"), ("pages", "text"), ("pages", "network"), ("pages", "profit"),
         ("pages", "calendar"), ("pages", "tags"), ("pages", "tag_followups"), ("pages", "inspector"), ("pages", "checks"),
         ("pages", "tmdb"), ("graph_pages", "whole_graph"), ("graph_pages", "node"), ("graph_pages", "screenplays"),
         ("embedding_page", "embeddings"), ("pipeline_page", "pipeline"), ("map_pages", "maps"), ("map_pages", "checks"),
         ("genre_page", "genre"), ("graph_pages", "movie_network")]


@pytest.mark.parametrize("module,fn", PAGES, ids=[f for _, f in PAGES])
def test_every_page_runs(explorer, module, fn):
    ok(page(module, fn))


def test_app_registers_every_page(explorer):
    """app.py builds the navigation with every page."""
    at = AppTest.from_file(str(ROOT / "explorer" / "app.py"), default_timeout=TIMEOUT).run()
    ok(at)
    from explorer import nav
    assert {"home", "pipeline", "text", "embeddings", "network", "graph", "node", "scripts", "profit", "attribution",
            "calendar", "tags", "followups", "data", "checks", "maps", "mapchecks", "tmdb", "genre", "movie_network"} <= set(nav.PAGES)


@pytest.mark.parametrize("step", range(12))
@pytest.mark.parametrize("fn", ["text"])
def test_every_notebook_step_renders(explorer, fn, step):
    at = page("pages", fn)
    radio = widget(at, "radio", "Step")
    if step >= len(radio.options):
        pytest.skip("fewer steps")
    radio.set_value(radio.options[step]).run()
    ok(at)


@pytest.mark.parametrize("name", ["roi_similarity", "pair_returns", "team_history", "movers"])
def test_network_result_files(explorer, name):
    at = page("pages", "network")
    widget(at, "selectbox", "Result file").set_value(name).run()
    ok(at)


@pytest.mark.parametrize("query", ["", "no such film", "Film", "tt0119094"])
def test_network_film_lookup(explorer, query):
    at = page("pages", "network")
    widget(at, "text_input", "Film title or IMDb ID").set_value(query).run()
    ok(at)


def test_profit_never_plots_a_column_against_itself(explorer):
    """Regression: x == y made plotly's trendline select the column twice (narwhals DuplicateError)."""
    at = page("pages", "profit")
    for x in widget(at, "selectbox", "x").options:
        widget(at, "selectbox", "x").set_value(x).run()
        ok(at)
        assert x not in widget(at, "selectbox", "y").options


@pytest.mark.parametrize("colour", ["first_genre", "tier", "clean", "none"])
def test_profit_colours_and_empty_sample(explorer, colour):
    at = page("pages", "profit")
    widget(at, "selectbox", "colour").set_value(colour).run()
    ok(at)
    widget(at, "multiselect", "Vote tier").set_value([]).run()
    ok(at)


def test_inspector_caps_very_wide_files(explorer):
    """Regression: a JSON flattening to 160,000+ columns could not be sent to the browser."""
    at = page("pages", "inspector")
    widget(at, "selectbox", "Folder").set_value("data/raw/features").run()
    files = widget(at, "selectbox", "File")
    files.select_index(next(i for i, o in enumerate(files.options) if "wide_words.json" in str(o))).run()
    ok(at)
    assert any("showing the first 500" in w.value for w in at.warning)


def test_inspector_opens_every_graph_file(explorer):
    at = page("pages", "inspector")
    widget(at, "selectbox", "Folder").set_value("data/raw/features/graph").run()
    for i in range(len(widget(at, "selectbox", "File").options)):
        widget(at, "selectbox", "File").select_index(i).run()
        ok(at)


@pytest.mark.parametrize("mode", ["Films sharing people", "Films sharing a company", "People who worked together",
                                  "Companies that co-produce"])
def test_whole_graph_modes_and_colours(explorer, mode):
    at = page("graph_pages", "whole_graph", {"graph_mode": mode})
    ok(at)
    widget(at, "slider", "Minimum shared").set_value(1).run()
    ok(at)
    for c in widget(at, "selectbox", "Colour by").options:
        widget(at, "selectbox", "Colour by").set_value(c).run()
        ok(at)


def test_whole_graph_selection_of_a_company(explorer):
    """Selecting a node (here with the finder) shows its row; companies get film counts, not film fields."""
    at = page("graph_pages", "whole_graph", {"graph_mode": "Companies that co-produce"})
    widget(at, "slider", "Minimum shared").set_value(1).run()
    widget(at, "selectbox", "Or find a node").select_index(0).run()
    ok(at)
    sel = at.dataframe[0].value
    assert {"films_in_selection", "median_log_multiple"} <= set(sel.columns)


def test_whole_graph_with_nothing_to_draw(explorer):
    at = page("graph_pages", "whole_graph")
    widget(at, "multiselect", "Vote tier").set_value([]).run()
    ok(at)
    assert at.info


def test_co_production_uses_studios_only(explorer):
    """The co-production network links studios; distributors are left out."""
    from explorer import graph_pages as gp
    stamp = gp.mtime(gp.NODES) + gp.mtime(gp.EDGES)
    v = gp.build_view(stamp, "Companies that co-produce", ("100k+", "10k-100k"), False, (1900, 2026), "any",
                      1, 2500, 8000, 1)
    assert v["ids"] and all(i.startswith("company:Q") for i in v["ids"])
    assert np.isfinite(v["node_lm"]).any() and (np.asarray(v["node_n"]) >= 1).all()


@pytest.mark.parametrize("kind", ["movie", "person", "company"])
@pytest.mark.parametrize("hops", [1, 2])
def test_node_explorer_on_each_node_type(explorer, kind, hops):
    nid = {"movie": "tt0119094", "person": "nm100", "company": "company:Q900"}[kind]
    at = page("graph_pages", "node", {"node_id": nid, "ego_hops": hops})
    ok(at)
    assert "Studio 0" in at.subheader[0].value or kind != "company"


def test_node_explorer_search(explorer):
    at = page("graph_pages", "node")
    widget(at, "text_input", "Name, title or ID").set_value("Studio").run()
    ok(at)
    assert widget(at, "radio", "Matches").options


def test_screenplay_titles(explorer):
    """Regression: 'Face_Off_1997' showed as 'Face (Off (1997)'."""
    from explorer.graph_pages import _script_label
    assert _script_label("Face_Off_1997") == "Face Off (1997)"
    assert _script_label("13_13_13_2013") == "13 13 13 (2013)"
    assert _script_label("Inception_2010") == "Inception (2010)"
    assert _script_label("odd") == "odd"
    at = page("graph_pages", "screenplays")
    ok(at)
    assert at.subheader[0].value == "Face Off (1997)"


def test_screenplay_character_click_reaches_read_tab(explorer):
    """Regression: a character picked in the Characters tab didn't filter the Read tab, and the user can
    still change the filter afterwards."""
    at = page("graph_pages", "screenplays")
    at.session_state["char_focus"] = "CASTOR"
    at.run()
    ok(at)
    assert widget(at, "selectbox", "Only scenes with").value == "CASTOR"
    widget(at, "selectbox", "Only scenes with").set_value("anyone").run()
    ok(at)
    assert widget(at, "selectbox", "Only scenes with").value == "anyone"


def test_screenplay_scene_focus_opens_its_page(explorer):
    at = page("graph_pages", "screenplays")
    widget(at, "number_input", "Scenes per page").set_value(5).run()
    at.session_state["scene_focus"] = 12
    at.run()
    ok(at)
    assert any(str(c.value).startswith("Scenes 10 to 14") for c in at.caption)


def test_screenplay_switch_and_search(explorer):
    at = page("graph_pages", "screenplays")
    at.session_state["char_focus"] = "CASTOR"
    at.run()
    widget(at, "selectbox", "Screenplay").set_value("tt1000001").run()   # CASTOR isn't in this one
    ok(at)
    assert widget(at, "selectbox", "Only scenes with").value == "anyone"
    widget(at, "text_input", "Search in the script").set_value("Reply 3").run()
    ok(at)
    widget(at, "text_input", "Search in the script").set_value("nowhere at all").run()
    ok(at)


def test_cached_loaders_see_a_rebuilt_file(explorer, tmp_path):
    """Regression: the file time was passed as '_mt', which Streamlit leaves out of the cache key, so a
    rebuilt file kept being served from the cache until the app restarted."""
    from explorer.common import table
    p = tmp_path / "t.csv"
    p.write_text("a\n1\n")
    assert table(p).a.tolist() == [1]
    p.write_text("a\n2\n")
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 10))
    assert table(p).a.tolist() == [2]


def test_private_file_gap_is_a_quiet_note(explorer):
    """A figure that needs a file in a private/ folder says which file, without a warning box."""
    at = AppTest.from_string(
        "from explorer.common import unavailable, REPO\n"
        "unavailable('Figure', 'layers', FileNotFoundError(2, 'No such file', str(REPO / 'analysis/scripts_text/private/x.parquet')))\n"
        "unavailable('Table', 'sizes', KeyError('year'))\n", default_timeout=TIMEOUT).run()
    ok(at)
    assert any("analysis/scripts_text/private/x.parquet" in c.value for c in at.caption)
    assert len(at.warning) == 1 and "year" in at.warning[0].value


def test_pages_without_the_screenplays(explorer, monkeypatch, tmp_path):
    """Regression: with no MovieSum files the script index raised a KeyError, so the Screenplays page and every
    film in the node explorer crashed instead of saying the screenplays are missing."""
    monkeypatch.setattr("explorer.graph_pages.MOVIESUM", [tmp_path / "none.jsonl"])
    at = page("graph_pages", "screenplays")
    ok(at)
    assert any("No MovieSum files" in w.value for w in at.warning)
    ok(page("graph_pages", "node", {"node_id": "tt0119094"}))


@pytest.mark.parametrize("method", ["PCA", "t-SNE", "All four"])
@pytest.mark.parametrize("colour", ["ln(multiple)", "net multiple", "primary genre", "release year"])
def test_embedding_maps_methods_and_colours(explorer, method, colour):
    at = page("map_pages", "maps")
    widget(at, "radio", "Method").set_value(method).run()
    widget(at, "selectbox", "Colour by").set_value(colour).run()
    ok(at)
    assert at.get("plotly_chart")
    widget(at, "selectbox", "Space").set_value("Genre shares").run()
    ok(at)


def test_embedding_maps_show_scores(explorer):
    at = page("map_pages", "maps")
    ok(at)
    sc = next(df.value for df in at.dataframe if "neighbours kept" in df.value.columns)
    assert len(sc) == 4 and "space" not in sc.columns
    assert any("neighbours kept" in str(c.value) for c in at.caption)


@pytest.mark.parametrize("method", ["UMAP", "All four"])
def test_embedding_maps_film_search(explorer, method):
    """Picking a film marks its nearest films and lists them; in "All four" there is one list per method."""
    at = page("map_pages", "maps")
    widget(at, "radio", "Method").set_value(method).run()
    widget(at, "selectbox", "Find a film").set_value("tt0119094").run()
    ok(at)
    nb = [df.value for df in at.dataframe if "ln(multiple)" in df.value.columns]
    assert len(nb) == (4 if method == "All four" else 1) and all(len(x) == 10 for x in nb)
    assert "Face/Off" not in set(nb[0].title)
    assert any(b.label.startswith("Open this film") for b in at.button)


def test_embedding_maps_nearest():
    from explorer.map_pages import nearest
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"tconst": [f"tt{i}" for i in range(50)], "x": rng.normal(size=50), "y": rng.normal(size=50)})
    d = pd.concat([d, d.iloc[[3]]])                       # a duplicated row must not count twice
    nb = nearest(d, "tt0")
    assert len(nb) == 10 and nb.tconst.nunique() == 10 and "tt0" not in set(nb.tconst)
    xy = d.drop_duplicates("tconst").set_index("tconst")[["x", "y"]]
    dist = np.hypot(*(xy - xy.loc["tt0"]).to_numpy().T)
    assert set(nb.tconst) == set(xy.index[np.argsort(dist)[1:11]])
    assert nearest(d, "tt_missing").empty
    assert len(nearest(d.head(4), "tt0")) == 3            # fewer films than k


def test_embedding_maps_without_the_file(explorer, monkeypatch, tmp_path):
    monkeypatch.setattr("explorer.map_pages.MAPS", tmp_path / "none.csv")
    at = page("map_pages", "maps")
    ok(at)
    assert any("archive/notebooks/embedding_maps.ipynb" in i.value for i in at.info)
    assert not at.get("plotly_chart")


def test_embedding_maps_without_scores(explorer, monkeypatch, tmp_path):
    monkeypatch.setattr("explorer.map_pages.SCORES", tmp_path / "none.csv")
    monkeypatch.setattr("explorer.map_pages.SCORES_COMMITTED", tmp_path / "none2.csv")
    at = page("map_pages", "maps")
    ok(at)
    assert at.get("plotly_chart") and any("No scores yet" in i.value for i in at.info)
@pytest.mark.parametrize("corpus", ["reviews", "screenplays", "summaries"])
def test_embedding_page_widgets(explorer, corpus):
    """The text embeddings page with each corpus in the word list, the film map and the neighbour lookup.
    (Streamlit runs every tab's code on each run, so this covers all tabs.)"""
    at = page("embedding_page", "embeddings", {"emb_words_corpus": corpus, "emb_map_corpus": corpus,
                                                "emb_map_color": "year", "emb_nb_corpus": corpus, "emb_leak_corpus": corpus,
                                                "emb_nb_variant": "raw"})
    ok(at)


def test_tag_shares_sum_to_one_and_keep_the_order(explorer):
    from explorer import pages
    rel = np.array([[0.9, 0.3, 0.1], [0.0, 0.0, 0.0], [0.2, 0.2, 0.6]])
    for power in (1, 2, 4):
        sh = pages.tag_shares(rel, power)
        assert np.allclose(sh.sum(axis=1), [1, 0, 1])          # a film with no relevance gets no shares
        assert (sh[0].argsort() == rel[0].argsort()).all()
    assert pages.tag_shares(rel, 2)[0, 0] > pages.tag_shares(rel, 1)[0, 0]


def test_tags_page_reads_names_literally_and_aligns_rows(explorer):
    """Tags called "nan" or "none" stay tags, and the genre table follows the matrix's row order."""
    from explorer import pages
    rel, names, films = pages.tag_data(str(pages.FEATURES), 0.0)
    assert names[-2:] == ["none", "nan"] and len(names) == rel.shape[1]
    ids = pd.read_csv(pages.FEATURES / "tag_relevance_ids.csv")["tconst"].tolist()
    assert films["tconst"].tolist() == ids
    assert np.allclose(films["rel_Drama"], rel[:, 0], atol=1e-6)


@pytest.mark.parametrize("power", [1.0, 4.0])
def test_tags_page_controls(explorer, power):
    at = page("pages", "tags")
    widget(at, "slider", "Power").set_value(power)
    widget(at, "radio", "Rank tags by").set_value("highest relevance")
    ok(at.run())
    at = page("pages", "tags", {"tag_view": "One tag"})
    widget(at, "multiselect", "Tags").set_value(["heist", "time travel"])
    ok(at.run())
    widget(at, "multiselect", "Tags").set_value([])
    ok(at.run())


@pytest.mark.parametrize("view", ["One film", "One tag", "Genre mix", "Profit", "Checks"])
def test_tags_page_views(explorer, view):
    ok(page("pages", "tags", {"tag_view": view}))


def test_tags_page_opens_on_a_requested_film(explorer):
    """What nav.open_tags sets: the page lands on that film's view, with its overview and links out."""
    at = page("pages", "tags", {"tag_film_pending": "tt1000007", "tag_view_pending": "One film", "tag_view": "Checks"})
    ok(at)
    assert widget(at, "selectbox", "Film").value == 7
    assert any("Film 7" in m.value for m in at.markdown)
    assert "tag_film_pending" not in at.session_state and "tag_view_pending" not in at.session_state
    ok(page("pages", "tags", {"tag_film_pending": "tt9999999"}))     # a film without scores is ignored


@pytest.mark.parametrize("view,key", [("One tag", "sel_tag_top_films"), ("Genre mix", "sel_tag_mix_films")])
def test_tags_page_row_click_shows_the_film(explorer, view, key):
    plain = page("pages", "tags", {"tag_view": view})
    at = page("pages", "tags", {"tag_view": view, key: {"selection": {"rows": [0], "columns": [], "cells": []}}})
    ok(at)
    assert len([b for b in at.button if b.label == "Its tags"]) == 1
    assert not [b for b in plain.button if b.label == "Its tags"]


def test_external_links(explorer):
    from explorer import graph_pages as gp
    film = gp.external_links("tt0004972")                       # first row of data/movie_links.csv
    assert film["IMDb"] == "https://www.imdb.com/title/tt0004972/"
    assert film["Wikipedia"].startswith("https://en.wikipedia.org/wiki/") and "wikidata.org" in film["Wikidata"]
    assert gp.external_links("tt9999999") == {"IMDb": "https://www.imdb.com/title/tt9999999/"}
    assert gp.external_links("nm0000123") == {"IMDb": "https://www.imdb.com/name/nm0000123/"}
    assert gp.external_links("company:Q900") == {"Wikidata": "https://www.wikidata.org/wiki/Q900"}
    assert gp.external_links("company:name:Dist A") == {}


def test_show_df_returns_the_clicked_row(explorer):
    at = AppTest.from_string(
        "import pandas as pd, streamlit as st\nfrom explorer.common import show_df\n"
        "r = show_df(pd.DataFrame({'a': range(12)}), key='t', select=True)\n"
        "st.write('picked', None if r is None else int(r['a']))\n", default_timeout=TIMEOUT)
    at.session_state["sel_t"] = {"selection": {"rows": [3], "columns": [], "cells": []}}
    ok(at.run())
    assert any("3" in str(m.value) for m in at.markdown)


def test_tags_page_without_the_files(explorer, monkeypatch):
    from explorer import pages
    monkeypatch.setattr(pages, "TAG_FILES", ("not_built.npy",))
    at = page("pages", "tags")
    ok(at)
    assert any("archive/scripts/reviews/tag_shares.py" in w.value for w in at.warning)


def test_tags_profit_view_controls(explorer):
    """The Profit view on the synthetic films: every outcome and tag class, another tag, another pair of genres."""
    at = page("pages", "tags", {"tag_view": "Profit"})
    ok(at)
    assert not at.info, [i.value for i in at.info]            # the saved results and the graph were found
    for outcome in ("gross", "adjusted_votes"):
        widget(at, "selectbox", "Outcome").set_value(outcome)
        widget(at, "selectbox", "Tags").set_value("all")
        ok(at.run())
    at.selectbox(key="tp_tag").set_value(4)
    widget(at, "selectbox", "Genre A").set_value("Horror")
    widget(at, "slider", "Counts as the genre").set_value(0.9)   # empty cells must not break the table
    ok(at.run())


def test_tags_profit_view_without_saved_results(explorer, monkeypatch, tmp_path):
    from explorer import pages
    monkeypatch.setattr(pages, "TAG_PROFIT", tmp_path)
    at = page("pages", "tags", {"tag_view": "Profit"})
    ok(at)
    assert any("tag_profit.py" in i.value for i in at.info)


@pytest.mark.parametrize("view", ["Tags from plots", "Usual genre mix", "Seen and liked", "Critics and audience", "Franchise and month"])
def test_tag_followups_views(explorer, view):
    """The follow-ups page draws every view from the committed results and needs no data folder."""
    at = page("pages", "tag_followups", {"fu_view": view})
    ok(at)
    assert not at.info, [i.value for i in at.info]


def test_tag_followups_controls_and_missing_results(explorer, monkeypatch, tmp_path):
    at = page("pages", "tag_followups", {"fu_view": "Seen and liked"})
    widget(at, "selectbox", "Quadrant").set_value("not liked, earned")
    ok(at.run())
    at = page("pages", "tag_followups", {"fu_view": "Critics and audience"})
    widget(at, "radio", "Critic score").set_value("rotten_tomatoes")
    ok(at.run())
    from explorer import pages
    monkeypatch.setattr(pages, "TAG_PROFIT", tmp_path)
    at = page("pages", "tag_followups")
    ok(at)
    assert any("plot_tags.py" in i.value for i in at.info)


def test_calendar_films_have_a_month_and_an_adjusted_return(explorer):
    from explorer import pages
    m = pages.calendar_films()
    assert len(m) == 38 and m["tconst"].is_unique and m["month"].between(1, 12).all()
    fit = m[m["clean"] & (m["year"] >= 1975) & m["multiple"].notna()]
    assert fit["adjusted"].notna().all() and abs(fit["adjusted"].mean()) < 1e-9      # residuals of the fit
    assert m.loc[~m["clean"], "adjusted"].isna().all()


@pytest.mark.parametrize("metric", ["Median gross ÷ budget", "Share grossing under budget", "Total gross ($M)", "Number of films"])
def test_calendar_measures_and_layouts(explorer, metric):
    at = page("pages", "calendar")
    ok(at)
    widget(at, "selectbox", "Measure").set_value(metric)
    widget(at, "toggle", "Years across").set_value(False)
    widget(at, "number_input", "Min films").set_value(1)
    ok(at.run())


def test_calendar_filters_down_to_nothing_and_back(explorer):
    at = page("pages", "calendar")
    widget(at, "toggle", "Clean films only").set_value(False)
    widget(at, "multiselect", "Must include genre").set_value(["Drama", "Comedy", "Horror"])   # films have two genres
    ok(at.run())
    assert any("No films match" in i.value for i in at.info)
    widget(at, "multiselect", "Must include genre").set_value(["Drama"])
    ok(at.run())


def test_calendar_square_hand_over_lists_its_films(explorer):
    """A click on a square can't be sent headlessly; it sets these two keys, which pick the films listed below."""
    from explorer import pages
    m = pages.calendar_films()
    r = m[m["clean"] & m["adjusted"].notna()].iloc[0]
    at = page("pages", "calendar", {"cal_years": (int(m["year"].min()), int(m["year"].max())),
                                    "cal_pick_year": int(r["year"]), "cal_pick_month": int(r["month"])})
    ok(at)
    assert widget(at, "selectbox", "Year").value == int(r["year"]) and widget(at, "selectbox", "Month").value == int(r["month"])
    assert any(r["title"] in str(df.value.to_dict()) for df in at.dataframe)
    # a month or year that the filters removed falls back to one that is shown
    ok(page("pages", "calendar", {"cal_pick_year": 1066, "cal_pick_month": 13}))


def test_calendar_without_release_months(explorer, monkeypatch, tmp_path):
    from explorer import pages
    monkeypatch.setattr(pages, "INFOBOX", [tmp_path / "none.csv"])
    at = page("pages", "calendar")
    ok(at)
    assert any("parsed infoboxes" in w.value for w in at.warning)


@pytest.mark.parametrize("view", ["One film", "One tag", "Genre mix", "Profit", "Checks"])
def test_tags_page_on_plot_scores(explorer, view):
    """Every view with the plot-based scores; Profit and Checks fall back to the Tag Genome."""
    at = page("pages", "tags", {"tag_source": "Plot summaries (all years)", "tag_view": view})
    ok(at)
    assert widget(at, "radio", "Scores from").value == "Plot summaries (all years)"
    on_plot = any("plot summary" in c.value for c in at.caption)
    assert on_plot == (view not in ("Profit", "Checks"))


def test_tags_page_plot_scores_cover_films_the_tag_genome_does_not(explorer):
    """A film after the Tag Genome (row 35 has plot scores only) opens on the plot-based scores, and its tags show."""
    from explorer import pages
    at = page("pages", "tags", {"tag_film_pending": "tt1000035", "tag_view_pending": "One film"})
    ok(at)
    assert widget(at, "radio", "Scores from").value == pages.PLOT
    assert widget(at, "selectbox", "Film").value == 35
    rel, names, films = pages.tag_data(str(pages.FEATURES), 0.0, pages.PLOT)
    assert rel.shape == (40, len(names)) and films["tconst"].tolist()[35] == "tt1000035"


def test_tags_page_hides_tags_a_plot_does_not_recover(explorer):
    at = page("pages", "tags", {"tag_source": "Plot summaries (all years)", "tag_view": "One tag"})
    ok(at)
    options = widget(at, "multiselect", "Tags").options
    assert options and not any(o.startswith("nan") or o.startswith("none") for o in options)   # not in plot_tags_tags.csv
    widget(at, "slider", "Only tags a plot recovers").set_value(0.0)
    ok(at.run())
    assert len(widget(at, "multiselect", "Tags").options) == 12


def test_tags_page_without_plot_scores(explorer, monkeypatch):
    from explorer import pages
    monkeypatch.setitem(pages.SOURCE_FILES, pages.PLOT, ("missing.npy", "missing.csv", "missing.csv"))
    at = page("pages", "tags", {"tag_source": "Plot summaries (all years)"})
    ok(at)
    assert widget(at, "radio", "Scores from").options == [pages.GENOME]


def test_embedding_maps_fall_back_to_committed_scores(explorer, monkeypatch, tmp_path):
    """Without the data folder's maps, the page still shows the committed scores and the notebook's takeaway."""
    monkeypatch.setattr("explorer.map_pages.MAPS", tmp_path / "none.csv")
    monkeypatch.setattr("explorer.map_pages.SCORES", tmp_path / "none2.csv")
    at = page("map_pages", "maps")
    ok(at)
    assert not at.get("plotly_chart")
    assert any("neighbours kept" in df.value.columns for df in at.dataframe)
    assert any("**Takeaway.**" in m.value or "Takeaway." in m.value for m in at.markdown)


def test_embedding_map_checks_page(explorer):
    """Every tab of the checks page renders from the committed numbers in archive/analysis/embedding_maps/."""
    at = page("map_pages", "checks")
    ok(at)
    assert len(at.tabs) == 6 and len(at.get("plotly_chart")) == 4
    assert any("Takeaway." in m.value for m in at.markdown)
    assert any("What the checks show" not in m.value and "Stability." in m.value for m in at.markdown)
    widget(at, "selectbox", "Clusters of one space").set_value("Cast and crew").run()
    ok(at)


def test_embedding_map_checks_without_results(explorer, monkeypatch, tmp_path):
    monkeypatch.setattr("explorer.map_pages.COMMITTED", tmp_path)
    at = page("map_pages", "checks")
    ok(at)
    assert any("embedding_map_checks.ipynb" in i.value for i in at.info)


def test_notebook_markdown_reads_the_notebooks():
    from explorer.map_pages import NB_CHECKS, NB_MAPS, notebook_markdown
    assert notebook_markdown(NB_MAPS).startswith("**Takeaway.**")
    assert notebook_markdown(NB_CHECKS, "## 6. What the checks show").startswith("## 6.")
    assert notebook_markdown(NB_MAPS, "## 10. What the maps show").startswith("## 10.")
    assert notebook_markdown(NB_MAPS.parent / "missing.ipynb") == ""


@pytest.mark.parametrize("view", ["Main test", "Robustness", "Budget tiers", "Second source (TMDB)", "Finer categories", "Before release", "Exploratory: genre network",
                                  "Exploratory: genre shares", "Exploratory: Horror words"])
def test_genre_page_views(explorer, view):
    """Every view of the Q1 genre page runs on the committed outputs."""
    ok(page("genre_page", "genre", {"q1_view": view}))
