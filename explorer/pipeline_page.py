"""Data pipeline: what each cleaning and processing step does, high level, with the counts it leaves.

The sample funnel is recomputed from the movie table when the data folder is there. Every other number is read from
the committed step summaries (analysis/*/outputs), so the page never states a number a script didn't write.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .common import FEATURES, REPO, mtime, table

GRAPH = FEATURES / "graph"
MOVIE_TABLE = FEATURES / "movies_features_10k.csv"
OUT = {"arcs": REPO / "analysis" / "arcs" / "outputs",
       "text": REPO / "analysis" / "scripts_text" / "outputs",
       "emb": REPO / "analysis" / "embeddings" / "outputs",
       "attr": REPO / "analysis" / "attribution" / "outputs"}
BLUE, ORANGE = "#0072B2", "#D55E00"


@st.cache_data(show_spinner=False)
def _json(path: str, stamp: float) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def summary(group: str, name: str) -> dict:
    p = OUT[group] / name
    return _json(str(p), mtime(p)) or {}


def get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fmt(x) -> str:
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    if isinstance(x, float) and not x.is_integer():
        return f"{x:,.2f}" if abs(x) >= 1 else f"{x:.1%}"
    return f"{int(x):,}"


def funnel(rows: list[tuple[str, int | None]], key: str, height: int | None = None, unit: str = "films"):
    """A left-to-right drop funnel: one bar per step, with how many the step removed."""
    rows = [(label, n) for label, n in rows if n is not None]
    if not rows:
        st.info("The summary file for this step isn't in the repo.")
        return
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    text = []
    for i, (label, n) in enumerate(rows):
        drop = rows[i - 1][1] - n if i else 0
        text.append(f"{n:,}" + (f"   (−{drop:,})" if drop > 0 else ""))
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", text=text[::-1], textposition="auto",
                           marker_color=[ORANGE] + [BLUE] * (len(rows) - 1)))  # bars run bottom-up: the last step is orange
    fig.update_layout(template="simple_white", height=height or 70 + 46 * len(rows),
                      margin=dict(l=10, r=10, t=10, b=10), xaxis_title=unit, yaxis=dict(automargin=True))
    st.plotly_chart(fig, width="stretch", key=key)


def card(does: str, drops: str, misses: str):
    c = st.columns(3)
    c[0].markdown(f"**What it does**  \n{does}")
    c[1].markdown(f"**What it drops or changes**  \n{drops}")
    c[2].markdown(f"**What it can miss**  \n{misses}")


def kv(rows: list[tuple[str, object]]):
    df = pd.DataFrame([(k, fmt(v)) for k, v in rows if v is not None], columns=["measure", "value"])
    if len(df):
        st.dataframe(df, hide_index=True, width="stretch")


# ---------------------------------------------------------------- the overview diagram

FLOW = """
digraph {
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.45;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fillcolor="#eef4fa", color="#0072B2"];
  edge [color="#888888", arrowsize=0.7];
  subgraph cluster_src { label="1 · Sources"; fontname="Helvetica"; color="#cccccc";
    imdb [label="IMDb files\\nfilms, people, credits"]; wd [label="Wikidata\\nIDs, series, companies"];
    wp [label="Wikipedia\\narticle wikitext"]; ms [label="MovieSum\\nscreenplays"]; tg [label="Tag Genome\\naudience reviews"]; }
  parse [label="2 · Parse infoboxes\\nbudget, gross, currency,\\nstudios, scores"];
  rule  [label="3 · Clean theatrical rule\\nUSD, ≥ $1,000, no rentals,\\nno streaming, 1975–2025", fillcolor="#fdebd9", color="#D55E00"];
  mgraph [label="4 · Movie graph\\nnodes + edges"];
  arcs  [label="Screenplay arcs\\nparse scenes → sentiment"]; text [label="Screenplay text\\ntokens, names out"];
  emb   [label="Text embeddings\\nreviews, scripts, summaries"]; attr [label="Profit attribution\\nwho and what on a film"];
  net   [label="Network results"];
  imdb -> mgraph; wd -> parse; wp -> parse; parse -> rule; imdb -> rule; rule -> mgraph;
  mgraph -> net; mgraph -> attr; rule -> arcs; rule -> text; rule -> emb; ms -> arcs; ms -> text; ms -> emb; tg -> emb;
  subgraph cluster_an { label="5 · Analyses (each cleans its own text)"; fontname="Helvetica"; color="#cccccc"; arcs; text; emb; attr; net; }
}
"""


# ---------------------------------------------------------------- the sample funnel

STREAMING = re.compile("netflix|amazon|apple|hulu", re.I)


def clean_steps(m: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """The clean rule of scripts/network/build_graph.py, split into cumulative steps."""
    b = pd.to_numeric(m["b"], errors="coerce").fillna(0)
    g = pd.to_numeric(m["g"], errors="coerce").fillna(0)
    y = pd.to_numeric(m["y"], errors="coerce").fillna(0)
    flag = m["gross_flag"].fillna("").astype(str)
    dist = m["distributors"].fillna("").astype(str)
    return [
        ("Budget and gross both in USD", (m["budget_cur"] == "USD") & (m["gross_cur"] == "USD")),
        ("Both at least $1,000", (b >= 1000) & (g >= 1000)),
        ("Gross isn't rentals, regional or direct-to-video",
         ~flag.str.contains("rentals") & ~flag.str.contains("regional") & ~flag.str.contains("direct_to_video")),
        ("Not a streaming release", ~dist.str.contains(STREAMING)),
        ("Released 1975–2025", (y >= 1975) & (y < 2026)),
    ]


def sample_funnel():
    st.subheader("3 · From every film to the clean sample")
    card("Keeps films whose budget and gross can be compared as theatrical money: both in US dollars, both "
         "plausible, gross from cinemas, and old enough to have finished its run.",
         "Non-USD figures, budgets or grosses under $1,000, grosses cited as rentals, regional only or direct-to-video, "
         "Netflix/Amazon/Apple/Hulu releases, films before 1975 (mostly rentals) and 2026 releases.",
         "Gross minus budget still isn't profit: marketing isn't in the budget and cinemas keep part of the gross. "
         "The 100k+ tier favours hits. Wikipedia and Wikidata disagree by more than 50% on some budgets.")
    if not MOVIE_TABLE.exists():
        st.info(f"Needs `{MOVIE_TABLE.relative_to(FEATURES.parent)}` in the data folder (run `./setup.sh`). "
                "The analysis sections below still work from the committed outputs.")
        return
    m = table(MOVIE_TABLE)
    tiers = sorted(m["tier"].dropna().unique()) if "tier" in m else []
    tier = st.radio("Vote tier", ["All (10k+ votes)"] + tiers, horizontal=True, key="pipe_tier")
    if tier != "All (10k+ votes)":
        m = m[m["tier"] == tier]
    keep = pd.Series(True, index=m.index)
    rows = [("Films in the table", len(m))]
    for label, mask in clean_steps(m):
        keep &= mask
        rows.append((label, int(keep.sum())))
    funnel(rows, key="pipe_sample")
    # Check: the split rule must give the same films as the clean flag the graph build wrote.
    nodes = GRAPH / "nodes.csv"
    if nodes.exists():
        n = table(nodes)
        flag = n[n["type"] == "movie"].set_index("id")["clean"].astype(str).isin(["1", "1.0", "True"])
        ours = set(m.loc[keep, "tconst"])
        theirs = set(flag[flag].index) & set(m["tconst"])
        if ours == theirs:
            st.success(f"Check: these steps give exactly the {len(ours):,} films flagged clean in `nodes.csv`.")
        else:
            st.warning(f"Check: these steps give {len(ours):,} films, `nodes.csv` flags {len(theirs):,} "
                       f"({len(ours - theirs):,} only here, {len(theirs - ours):,} only there). "
                       "The bundled graph may predate the latest rule.")


# ---------------------------------------------------------------- per-analysis sections

def samples_compared():
    st.markdown("**Which clean sample each analysis started from**")
    a1, t1 = summary("arcs", "s01_summary.json"), summary("text", "s01_summary.json")
    e1, at = summary("emb", "s01_cleaning.json"), summary("attr", "summary.json")
    rows = [("Screenplay arcs", get(a1, "checks", "clean_movies", "got"), "100k+ votes, rule of 4 Nov"),
            ("Screenplay text", get(t1, "checks", "clean_movies", "got"), "100k+ votes, rule of 4 Nov"),
            ("Text embeddings", get(e1, "sample", "clean_films"), "10k+ votes"),
            ("Profit attribution", get(at, "sample", "films"), "10k+ votes, 1975–2025")]
    df = pd.DataFrame([(a, fmt(n), note) for a, n, note in rows if n is not None],
                      columns=["analysis", "clean films", "sample"])
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption("The analyses were run at different times, so they don't share one sample. Numbers from "
               "`analysis/*/outputs`; the sample notes come from the README's history of the clean rule.")


def arcs_section():
    s1, s2 = summary("arcs", "s01_summary.json"), summary("arcs", "s02_summary.json")
    card("Matches clean films to MovieSum screenplays, splits each script into scenes and sentences, and keeps "
         "scripts long enough to draw a sentiment arc.",
         "Duplicate MovieSum IDs, 8 scripts whose dialogue markup is broken, likely franchise films (for the primary "
         "test), and scripts with too few scenes. Rejoins split contractions (\"do n't\" → \"don't\").",
         "Franchise flag is rough (Wikidata 'follows' or a shared name). Some scripts may be transcripts or drafts, "
         "not the shooting script.")
    funnel([("Clean films", get(s1, "checks", "clean_movies", "got")),
            ("With a usable screenplay", get(s1, "checks", "clean_with_usable_script", "got")),
            ("Dialogue markup usable", get(s1, "checks", "after_dialogue_exclusion", "got")),
            ("Not likely franchise (primary)", get(s1, "checks", "not_likely_franchise", "got")),
            ("Enough scenes for an arc", get(s2, "arc_ready_primary"))], key="pipe_arcs")
    kv([("Scenes parsed", get(s2, "scenes_total")), ("Sentences (dialogue + action)", get(s2, "sentences", "both")),
        ("Parse errors", get(s2, "parse_errors")), ("Contractions rejoined", get(s2, "contraction_joins")),
        ("Lone \"n't\" before → after", None if get(s2, "lone_nt_before") is None
         else f"{get(s2, 'lone_nt_before'):,} → {get(s2, 'lone_nt_after'):,}"),
        ("Possible transcripts flagged", get(s2, "possible_transcripts")),
        ("Primary films dropped for too few scenes", get(s2, "excluded_primary", "under_min_scenes"))])


def text_section():
    s1, s2, s3 = (summary("text", f"s0{i}_summary.json") for i in (1, 2, 3))
    card("Splits each screenplay into dialogue and action text, tokenises it, and strips what is formatting or "
         "a name rather than what the characters say.",
         "Scripts with almost no dialogue, technical tokens (CUT TO, V.O.), scanning artefacts, scene headings, and "
         "character names (speaker labels, plus spaCy names in the stricter variant).",
         "Name removal misses names that are also dictionary words, and removes some ordinary words that are names. "
         "Short scripts are only flagged, not dropped.")
    excl = get(s2, "excluded_little_dialogue")
    usable = get(s1, "checks", "clean_with_usable_script", "got")
    funnel([("Clean films", get(s1, "checks", "clean_movies", "got")),
            ("With a usable screenplay", usable),
            ("Enough dialogue to analyse", usable - excl if usable is not None and excl is not None else None)],
           key="pipe_text")
    removed = {"Speaker names (labels)": get(s3, "dialogue_name_tokens_removed"),
               "Scene headings (full text)": get(s3, "heading_tokens_removed_full"),
               "Technical tokens": get(s3, "technical_tokens_removed"),
               "Scanning artefacts": get(s3, "artefact_tokens_removed")}
    removed = {k: v for k, v in removed.items() if v is not None}
    if removed:
        fig = go.Figure(go.Bar(x=list(removed.values()), y=list(removed), orientation="h", marker_color=BLUE,
                               text=[f"{v:,}" for v in removed.values()], textposition="auto"))
        fig.update_layout(template="simple_white", height=230, margin=dict(l=10, r=10, t=30, b=10),
                          title="Tokens removed in cleaning", xaxis_title="tokens")
        st.plotly_chart(fig, width="stretch", key="pipe_text_tokens")
    kv([("Dialogue tokens left (primary)", get(s3, "dialogue_tokens_primary")),
        ("Contractions rejoined", get(s3, "contraction_joins")),
        ("Median dialogue share of words", get(s2, "dialogue_share_of_words_median"))])


def embeddings_section():
    s = summary("emb", "s01_cleaning.json")
    card("Cleans three corpora before turning them into vectors: audience reviews, screenplays and MovieSum plot "
         "summaries.",
         "Markup and URLs; review sentences about money (they leak the outcome); the film's title and the names of "
         "cast and crew (masked, so a vector can't just recognise the film); screenplay headings and labels; "
         "empty, non-English and duplicate reviews.",
         "Money words are a regular expression, so some outcome talk slips through. Reviews stop at 2017. Name "
         "masking agrees with spaCy on only part of the words (see below).")
    r, sp, su = get(s, "reviews") or {}, get(s, "screenplays") or {}, get(s, "summaries") or {}
    tabs = st.tabs(["Reviews", "Screenplays", "Plot summaries"])
    with tabs[0]:
        read = r.get("reviews_read_for_clean_films")
        left = None if read is None else read - sum(r.get(k, 0) for k in
                                                  ("dropped_empty_or_short", "dropped_not_english", "dropped_duplicate"))
        funnel([("Reviews read for clean films", read),
                ("Not empty or too short", None if read is None else read - r.get("dropped_empty_or_short", 0)),
                ("English", None if read is None else read - r.get("dropped_empty_or_short", 0)
                 - r.get("dropped_not_english", 0)),
                ("Not a duplicate", left)], key="pipe_rev", unit="reviews")
        st.caption("Then a sample of reviews per film is embedded:")
        kv([("Films with any valid review", r.get("clean_films_with_any_valid_review")),
            ("Films dropped for too few reviews", r.get("films_dropped_fewer_than_min_reviews")),
            ("Films kept", r.get("films")), ("Reviews sampled", r.get("reviews_sampled")),
            ("Median reviews per film", r.get("median_reviews_per_film")),
            ("Share of sentences dropped for money talk", r.get("share_of_sentences_dropped_for_money")),
            ("Title mentions masked", r.get("title_mentions_masked")),
            ("Cast and director mentions masked", r.get("cast_and_director_mentions_masked")),
            ("Latest film year", r.get("latest_film_year"))])
    with tabs[1]:
        kv([("Clean films with a MovieSum script", sp.get("clean_films_with_a_moviesum_script")),
            ("Films kept", sp.get("films")), ("Median words before → after",
             None if sp.get("median_words_raw") is None else f"{sp['median_words_raw']:,.0f} → {sp['median_words_clean']:,.0f}"),
            ("Share of raw words removed", sp.get("share_of_raw_words_removed")),
            ("Share of clean words that are name placeholders", sp.get("share_of_clean_words_that_are_name_placeholders"))])
        v = sp.get("name_rule_vs_spacy") or {}
        if v:
            st.caption(f"Check against spaCy on {v.get('scripts')} scripts: spaCy also tags "
                       f"{v.get('share_of_rule_words_spacy_also_tags', 0):.0%} of the words the label rule masks, and "
                       f"the rule masks {v.get('share_of_spacy_person_words_the_rule_masks', 0):.0%} of the words "
                       "spaCy calls a person.")
    with tabs[2]:
        rm = su.get("removed") or {}
        kv([("Clean films with a MovieSum summary", su.get("clean_films_with_a_moviesum_summary")),
            ("Dropped as too short", su.get("dropped_too_short")), ("Films kept", su.get("films")),
            ("Bracketed notes removed", rm.get("brackets")), ("Actor names in parentheses removed", rm.get("actor_parens")),
            ("Titles masked", rm.get("titles")), ("People masked", rm.get("people"))])


def attribution_section():
    s = summary("attr", "summary.json")
    card("Builds one row per clean film with its budget, year, genre, franchise group and every person and company "
         "on it, then splits the return between them.",
         f"People and companies with fewer than {get(s, 'sample', 'min_films_for_an_effect', default='a few')} films "
         f"get no effect of their own; only the top {get(s, 'sample', 'cast_top', default='few')} billed actors count.",
         "No film has a real profit figure, so return is a stand-in built from gross and budget. Effects are shrunk "
         "towards zero, so people with few films look average. Credits are IMDb's principal list only.")
    films = get(s, "sample", "films")
    yrs = get(s, "sample", "years")
    st.caption(f"{fmt(films)} films" + (f", {yrs[0]}–{yrs[1]}" if yrs else "") + ".")
    have = get(s, "sample", "films_with_at_least_one") or {}
    eff = get(s, "sample", "entities_with_an_effect") or {}
    if have:
        df = pd.DataFrame({"role": list(have), "films with at least one": list(have.values()),
                           "people/companies with an effect": [eff.get(k) for k in have]})
        st.dataframe(df, hide_index=True, width="stretch")


# ---------------------------------------------------------------- the page

def pipeline():
    st.title("Data pipeline")
    st.caption("What happens to the data at each step, what each step drops, and how many films are left. The "
               "sample funnel is recomputed from the data folder; the analysis numbers come from the committed "
               "step summaries.")
    st.graphviz_chart(FLOW, width="stretch")

    st.subheader("1 · Sources")
    st.markdown("""
| Source | What we take | Key |
|---|---|---|
| IMDb datasets | titles, ratings, votes, genres, principal credits | `tconst`, `nconst` |
| Wikidata | IMDb → Wikipedia link, series/follows, company IDs | `tconst` → QID |
| English Wikipedia | article wikitext: infobox, reception section | page linked through Wikidata |
| MovieSum | screenplays with scene and dialogue markup, plot summaries | `tconst` |
| MovieLens Tag Genome | audience reviews (to 2017) and tag scores | MovieLens ID → `tconst` |
""")
    st.caption("Scope: non-adult films with at least 10,000 IMDb votes; the older analyses use 100,000+.")

    st.subheader("2 · Parse the infoboxes")
    card("`parse_infobox.py` reads each article's infobox for budget, gross and their currencies, studios, "
         "distributors, producers, Rotten Tomatoes and Metacritic; `extract_reception.py` pulls the reception text.",
         "A range becomes its midpoint, footnotes and markup go, and a US-dollar figure is preferred when several "
         "currencies are given. A gross is flagged when its text says rentals or a single region, and a film when "
         "the article is in a direct-to-video category.",
         "Every parser bug so far was found by reading raw rows. Currency can be misread "
         "(a $700M budget for White God). Studio names aren't merged unless Wikidata gives them one ID.")

    sample_funnel()

    st.subheader("4 · Build the movie graph")
    card("`build_graph.py` makes one node per film, person and company and one edge per credit, with the clean flag "
         "on every film.",
         "Company names are merged only when they share a Wikidata ID; producers missing from IMDb are added from the "
         "infobox.",
         "Credits are IMDb's principal list, so people further down the credits are missing.")
    nodes, edges = GRAPH / "nodes.csv", GRAPH / "edges.csv"
    if nodes.exists():
        n = table(nodes)
        c = st.columns(4)
        for col, kind in zip(c, ["movie", "person", "company"]):
            col.metric(f"{kind.title()} nodes", f"{int((n['type'] == kind).sum()):,}")
        if edges.exists():
            c[3].metric("Edges (credits)", f"{len(table(edges)):,}")
    else:
        st.info("Needs `features/graph/nodes.csv` in the data folder.")

    st.subheader("5 · Each analysis cleans its own text")
    samples_compared()
    for title, fn in [("Screenplay arcs", arcs_section), ("Screenplay text", text_section),
                      ("Text embeddings", embeddings_section), ("Profit attribution", attribution_section)]:
        with st.expander(title, expanded=False):
            fn()
