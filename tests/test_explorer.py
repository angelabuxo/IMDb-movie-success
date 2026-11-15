"""The Streamlit explorer: every page runs without an exception, on a small synthetic data folder.

Pages are driven with Streamlit's AppTest, headless. The data folder is built here (a few dozen films, people
and studios, two screenplays, a very wide JSON) and handed to the explorer through IMDB_DATA_DIR, so nothing
reads data/raw. Pages that show the committed notebook outputs (arcs, text, network results, checks) read
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
    """A data folder shaped like data/raw, and an ideas folder for the Topic history page."""
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

    ideas = root / "ideas"
    ideas.mkdir()
    (ideas / "README.md").write_text(
        "# Idea files\n\n| File | Made | What it holds |\n| --- | --- | --- |\n"
        "| [week8-inspired-ideas.md](week8-inspired-ideas.md) | 28 Oct | Ideas 76-100 |\n"
        "| [missing.md](missing.md) | 28 Oct | Listed but not on disk |\n")
    (ideas / "week8-inspired-ideas.md").write_text(
        "# Ideas\n\n96. **Record deals.**\n    - one\n97. **Who co-finances films.**\n"
        "    - **Question:** do co-financing networks decide what gets made? It costs $5 to ask.\n"
        "98. **Novels.**\n")
    return root


@pytest.fixture(scope="module")
def explorer(data_dir):
    """Point the explorer at the synthetic folders and import it fresh (its paths are read at import)."""
    old = {k: os.environ.get(k) for k in ("IMDB_DATA_DIR", "IDEAS_DIR")}
    os.environ["IMDB_DATA_DIR"] = str(data_dir)
    os.environ["IDEAS_DIR"] = str(data_dir / "ideas")
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


PAGES = [("pages", "home"), ("pages", "arcs"), ("pages", "text"), ("pages", "network"), ("pages", "profit"),
         ("pages", "inspector"), ("pages", "checks"), ("pages", "topic_history"), ("graph_pages", "whole_graph"),
         ("graph_pages", "node"), ("graph_pages", "screenplays")]


@pytest.mark.parametrize("module,fn", PAGES, ids=[f for _, f in PAGES])
def test_every_page_runs(explorer, module, fn):
    ok(page(module, fn))


def test_app_registers_every_page(explorer):
    """app.py builds the navigation; with an ideas folder the Topic history page is in it."""
    at = AppTest.from_file(str(ROOT / "explorer" / "app.py"), default_timeout=TIMEOUT).run()
    ok(at)
    from explorer import nav
    assert {"home", "arcs", "text", "network", "graph", "node", "scripts", "profit", "data", "checks",
            "topic"} <= set(nav.PAGES)


@pytest.mark.parametrize("step", range(12))
@pytest.mark.parametrize("fn", ["arcs", "text"])
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


def test_topic_history_reads_the_ideas_folder(explorer):
    at = page("pages", "topic_history")
    ok(at)
    assert "Who co-finances films" in at.info[0].value and "98." not in at.info[0].value
    assert "\\$5" in at.info[0].value                      # money, not LaTeX
    assert len(widget(at, "selectbox", "File").options) == 2      # README plus the one listed file on disk
    widget(at, "selectbox", "File").select_index(1).run()
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


@pytest.fixture
def arcs_scores(explorer, tmp_path, monkeypatch):
    """Synthetic arcs Step 3 scene scores for the Face/Off screenplay; scene 7 has none (no scored sentence)."""
    from explorer import common
    d = tmp_path / "arcs"
    d.mkdir()
    scenes = [i for i in range(30) if i != 7]
    base = pd.DataFrame({"movie_name": "Face_Off_1997", "scene": scenes})
    base.assign(sst2_both=np.linspace(-0.9, 0.9, len(scenes)), sst2_dialogue=0.25, sst2_action=-0.5, n_both=3).to_csv(
        d / "s03_scene_sst2.csv.gz", index=False)
    base.assign(vader_both=0.1, vader_dialogue=0.2, vader_action=np.nan, n_both=3).to_csv(
        d / "s03_scene_vader.csv.gz", index=False)
    base.assign(sst2_scene=0.75, sst2_scene_neutral=np.nan, sst2_pieces=1).to_csv(
        d / "s03_scene_context_sst2.csv.gz", index=False)
    tt = np.arange(100)
    arc = np.sin(tt / 100 * 2 * np.pi) * 0.4
    pd.DataFrame({"movie_name": "Face_Off_1997", "t": tt, "raw": arc, "smooth": arc}).to_csv(
        d / "s05_arcs_sst2.csv.gz", index=False)
    pd.DataFrame([{"movie_name": "Face_Off_1997", "in_primary": True, "primary_genre": "Action", "sentences_both": 90,
                   "mean_sentences_per_scene": 3.0, "n_scene_tags": 30, "possible_transcript": False,
                   "scenes": len(scenes), "D1": 0.28, "D2": -0.05, "mean_level": 0.0, "min_position": 0.755,
                   "sign_changes": 1}]).to_csv(d / "s05_descriptors_sst2.csv", index=False)
    monkeypatch.setitem(common.OUTPUT_DIRS, "arcs", d)
    return d


def _script_html(at: AppTest) -> str:
    return next(m.value for m in at.markdown if 'class="sp"' in m.value)


def test_read_tab_shows_scene_sentiment(arcs_scores):
    at = page("graph_pages", "screenplays")
    ok(at)
    assert widget(at, "multiselect", "Sentiment beside each scene").value == ["sst2"]   # the primary instrument
    html = _script_html(at)
    assert "SST-2" in html and "VADER" not in html
    assert "both -0.90" in html and "dialogue +0.25" in html and "action -0.50" in html and "whole scene +0.75" in html
    assert html.count("no scored sentences") == 1                                       # scene 7
    assert at.get("plotly_chart"), "the reading arc is drawn"


def test_read_tab_adds_and_removes_instruments(arcs_scores):
    at = page("graph_pages", "screenplays")
    widget(at, "multiselect", "Sentiment beside each scene").set_value(["sst2", "vader"]).run()
    ok(at)
    html = _script_html(at)
    assert "VADER" in html and "both +0.10" in html and "action +nan" not in html   # a missing layer is skipped
    widget(at, "multiselect", "Sentiment beside each scene").set_value([]).run()
    ok(at)
    assert 'class="b"' not in _script_html(at)


def test_read_tab_without_saved_sentiment(arcs_scores):
    """A script outside the arcs sample says so, and coming back keeps the instrument choice."""
    at = page("graph_pages", "screenplays")
    widget(at, "multiselect", "Sentiment beside each scene").set_value(["vader"]).run()
    widget(at, "selectbox", "Screenplay").set_value("tt1000001").run()
    ok(at)
    assert any("isn't in the arcs sample" in str(c.value) for c in at.caption)
    assert 'class="b"' not in _script_html(at)
    widget(at, "selectbox", "Screenplay").set_value("tt0119094").run()
    ok(at)
    assert widget(at, "multiselect", "Sentiment beside each scene").value == ["vader"]


def test_sentiment_badge_colours():
    from explorer.graph_pages import _badge
    assert "0,114,178" in _badge(0.4, "both") and "+0.40" in _badge(0.4, "both")
    assert "213,94,0" in _badge(-0.4, "both") and "-0.40" in _badge(-0.4, "both")
    assert "0.75)" in _badge(3.0, "both")          # opacity capped


def test_node_explorer_shows_a_films_arc(arcs_scores):
    at = page("graph_pages", "node", {"node_id": "tt0119094"})
    ok(at)
    assert [t.label for t in at.tabs][-1] == "Sentiment arc"
    m = {x.label: x.value for x in at.metric}
    assert m["Swing (D1)"] == "0.280" and m["Ending lift (D2)"] == "-0.050" and m["Low point"] == "76% into the story"
    assert any("In the primary sample" in str(c.value) for c in at.caption)


@pytest.mark.parametrize("nid", ["tt1000001", "nm100", "company:Q900"])
def test_no_arc_tab_without_an_arc(arcs_scores, nid):
    """A film with a screenplay but no saved arc, a person and a company get no arc tab."""
    at = page("graph_pages", "node", {"node_id": nid})
    ok(at)
    assert "Sentiment arc" not in [t.label for t in at.tabs]


def test_arc_point_maps_to_a_scored_scene():
    from explorer.graph_pages import scene_at
    scored = [0, 1, 2, 4, 5, 9]                    # scenes 3, 6, 7 and 8 have no scored sentence
    assert scene_at(0, scored) == 0 and scene_at(99, scored) == 9
    assert scene_at(50, scored) == 4               # point 50 (centre 0.505) is the 4th of 6 scored scenes
    assert {scene_at(t, scored) for t in range(100)} == set(scored)


def test_private_file_gap_is_a_quiet_note(explorer):
    """A figure that needs a file in a private/ folder says which file, without a warning box."""
    at = AppTest.from_string(
        "from explorer.common import unavailable, REPO\n"
        "unavailable('Figure', 'layers', FileNotFoundError(2, 'No such file', str(REPO / 'analysis/arcs/private/x.parquet')))\n"
        "unavailable('Table', 'sizes', KeyError('year'))\n", default_timeout=TIMEOUT).run()
    ok(at)
    assert any("analysis/arcs/private/x.parquet" in c.value for c in at.caption)
    assert len(at.warning) == 1 and "year" in at.warning[0].value


@pytest.fixture
def profit_arcs(explorer, tmp_path, monkeypatch):
    """Saved arcs for every synthetic film, keyed by script name through a script table, as on a machine
    without the screenplays. Films with a higher return get a rising arc."""
    from explorer import common
    nodes = pd.read_csv(explorer / "features" / "graph" / "nodes.csv")
    films = nodes[nodes.type == "movie"].reset_index(drop=True)
    names = [f"Film_{i}_2000" for i in range(len(films))]
    arcs_dir, text_dir = tmp_path / "arcs", tmp_path / "text"
    arcs_dir.mkdir()
    text_dir.mkdir()
    pd.DataFrame({"tconst": films.id, "movie_name": names}).to_csv(text_dir / "script_table.csv", index=False)
    lift = np.log(films.multiple.fillna(1).clip(lower=.01)).to_numpy()
    tt = np.arange(100)
    pd.concat([pd.DataFrame({"movie_name": n, "t": tt, "raw": l * (tt / 99 - .5), "smooth": l * (tt / 99 - .5)})
               for n, l in zip(names, lift)]).to_csv(arcs_dir / "s05_arcs_sst2.csv.gz", index=False)
    pd.DataFrame({"movie_name": names, "in_primary": np.arange(len(names)) % 5 != 0, "scenes": 30, "D1": np.abs(lift) / 3,
                  "D2": lift, "mean_level": 0.0, "min_position": np.where(lift > 0, .005, .995),
                  "sign_changes": 1}).to_csv(arcs_dir / "s05_descriptors_sst2.csv", index=False)
    monkeypatch.setitem(common.OUTPUT_DIRS, "arcs", arcs_dir)
    monkeypatch.setitem(common.OUTPUT_DIRS, "scripts_text", text_dir)
    monkeypatch.setattr("explorer.graph_pages.MOVIESUM", [tmp_path / "none.jsonl"])   # names from the table only
    return arcs_dir


def test_profit_playground_arcs(profit_arcs):
    at = page("pages", "profit")
    ok(at)
    assert {"arc_swing", "arc_ending_lift", "arc_low_point"} <= set(widget(at, "selectbox", "x").options)
    assert at.get("plotly_chart")
    rho = next(df.value for df in at.dataframe if "descriptor" in df.value.columns).set_index("descriptor")
    assert rho.loc["ending lift (D2)", "Spearman rho with log(gross / budget)"] > 0.99   # built that way
    n_primary = rho.films.iloc[0]
    widget(at, "toggle", "Primary arcs sample only").set_value(False).run()
    ok(at)
    rho = next(df.value for df in at.dataframe if "descriptor" in df.value.columns)
    assert rho.films.iloc[0] > n_primary
    for split in widget(at, "radio", "Compare").options:
        widget(at, "radio", "Compare").set_value(split).run()
        ok(at)
    widget(at, "selectbox", "x").set_value("arc_ending_lift").run()
    ok(at)


def test_profit_playground_without_enough_arcs(arcs_scores):
    """With one film's arc only, the section says so instead of drawing."""
    at = page("pages", "profit")
    ok(at)
    assert any("Fewer than 20 films" in i.value for i in at.info)


def test_pages_without_the_screenplays(explorer, monkeypatch, tmp_path):
    """Regression: with no MovieSum files the script index raised a KeyError, so the Screenplays page and every
    film in the node explorer crashed instead of saying the screenplays are missing."""
    monkeypatch.setattr("explorer.graph_pages.MOVIESUM", [tmp_path / "none.jsonl"])
    at = page("graph_pages", "screenplays")
    ok(at)
    assert any("No MovieSum files" in w.value for w in at.warning)
    ok(page("graph_pages", "node", {"node_id": "tt0119094"}))
