"""Text embeddings page: reviews, screenplays and summaries against profit and revenue.

Every table, figure and reading comes from analysis/embeddings/report.py, the same calls the notebook
notebooks/embeddings.ipynb makes. The film map and the neighbour lookup need the local vectors in data/raw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from analysis.embeddings import report as R
from analysis.embeddings.paths import OUTPUTS, PRIVATE, VECTORS

from .common import md, mtime, show_df


@st.cache_data(show_spinner="Reading results…")
def _load(stamp: float) -> dict:
    return R.load()


@st.cache_data(show_spinner="Projecting films…", max_entries=6)
def _map(corpus: str, variant: str, stamp: float) -> pd.DataFrame | None:
    return R.film_map(corpus, variant)


@st.cache_data(show_spinner="Reading vectors…", max_entries=3)
def _vectors(corpus: str, variant: str, stamp: float) -> tuple[np.ndarray, list[str]]:
    V = np.load(VECTORS / f"{corpus}_{variant}.npy")
    ids = pd.read_csv(VECTORS / f"{corpus}_ids.csv", dtype=str).tconst.tolist()
    return V, ids


def embeddings():
    st.title("Text embeddings and profit")
    st.caption("analysis/embeddings (s01_corpora, s02_embed, s03_correlate) and notebooks/embeddings.ipynb. "
               "Profitability is ln(gross / budget), revenue ln(gross); neither is real profit.")
    if not (OUTPUTS / "s03_results.csv").exists():
        st.info("No results yet. Run `python -m analysis.embeddings.s03_correlate` (after s01 and s02).")
        return
    res = _load(mtime(OUTPUTS / "s03_results.csv"))
    r, say = res["results"], R.reading(res)
    t = st.tabs(["Result", "Checks", "Cleaning", "Network", "Words", "Film map", "Neighbours", "All measurements"])

    with t[0]:
        st.plotly_chart(R.fig_net(res), width="stretch")
        st.markdown(md(say["net"]))
        show_df(R.main_table(r), key="emb_main")
        st.plotly_chart(R.fig_alone(r), width="stretch")
        st.markdown(md(say["alone"]))

    with t[1]:
        st.plotly_chart(R.fig_methods(r), width="stretch")
        st.markdown(md(say["checks"]))
        show_df(R.checks_table(r), key="emb_checks")
        st.subheader("OCR-damaged screenplays")
        st.markdown(md(say.get("ocr", "")))
        show_df(R.ocr_table(r), key="emb_ocr")
        st.subheader("Other samples")
        st.plotly_chart(R.fig_samples(r), width="stretch")
        st.markdown(md(say["samples"]))
        show_df(R.samples_table(r), key="emb_samples")
        st.caption("R² gain by number of principal components kept")
        show_df(R.components_table(r), key="emb_k")

    with t[2]:
        show_df(R.cleaning_table(res["cleaning"]), key="emb_clean")
        chk = res["cleaning"]["screenplays"]["name_rule_vs_spacy"]
        st.markdown(f"Character-name rule against spaCy PERSON tags on {chk['scripts']} scripts: spaCy also tags "
                    f"{chk['share_of_rule_words_spacy_also_tags']:.0%} of the words the rule masks, and the rule masks "
                    f"{chk['share_of_spacy_person_words_the_rule_masks']:.0%} of the words spaCy tags.")
        st.json(res["cleaning"], expanded=False)
        st.json(res["embedding"], expanded=False)

    with t[3]:
        st.plotly_chart(R.fig_network(r), width="stretch")
        st.markdown(md(say["network"]))
        st.markdown(md(say["agreement"]))

    with t[4]:
        c = st.radio("Corpus", R.CORPORA, format_func=R.LABEL.get, horizontal=True, key="emb_words_corpus")
        st.caption("Words used by more films at one end of the out-of-fold 'more profit than the controls predict' "
                   "score than at the other (top against bottom fifth). They describe the direction, not causes.")
        show_df(R.words_table(res["words"], c), key="emb_words")

    with t[5]:
        c1, c2, c3 = st.columns(3)
        corpus = c1.selectbox("Corpus", R.CORPORA, format_func=R.LABEL.get, key="emb_map_corpus")
        variant = c2.radio("Text", ["clean", "raw"], horizontal=True, key="emb_map_variant")
        color = c3.selectbox("Colour", ["log_mult", "log_gross", "log_budget", "year"], key="emb_map_color")
        stamp = mtime(VECTORS / f"{corpus}_{variant}.npy")
        d = _map(corpus, variant, stamp) if stamp else None
        if d is None:
            st.info("The map needs the local vectors: run s01_corpora and s02_embed on the data.")
        else:
            st.plotly_chart(R.fig_map(d, corpus, color), width="stretch")

    with t[6]:
        neighbours()

    with t[7]:
        show_df(r, key="emb_all", caption="analysis/embeddings/outputs/s03_results.csv")


def neighbours():
    """A film's nearest films in one corpus's vector space, with their outcomes."""
    films_path = PRIVATE / "films.parquet"
    if not films_path.exists():
        st.info("Needs the local film table and vectors: run s01_corpora and s02_embed.")
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    corpus = c2.selectbox("Corpus", R.CORPORA, format_func=R.LABEL.get, key="emb_nb_corpus")
    variant = c3.radio("Text", ["clean", "raw"], horizontal=True, key="emb_nb_variant")
    if not (VECTORS / f"{corpus}_{variant}.npy").exists():
        st.info("No vectors for this corpus yet.")
        return
    V, ids = _vectors(corpus, variant, mtime(VECTORS / f"{corpus}_{variant}.npy"))
    f = pd.read_parquet(films_path).set_index("tconst").loc[ids].reset_index()
    label = f.primaryTitle + " (" + f.year.astype(str) + ")"
    order = np.argsort(-f.votes.to_numpy())
    pick = c1.selectbox("Film", order, format_func=lambda i: label.iloc[i], key="emb_nb_film")
    sim = V @ V[pick]
    near = np.argsort(-sim)[1:16]
    out = f.iloc[near][["primaryTitle", "year", "genres", "log_mult", "log_gross", "franchise"]].assign(
        cosine=sim[near].round(3), multiple=np.exp(f.log_mult.iloc[near]).round(2))
    me = f.iloc[pick]
    st.markdown(f"**{label.iloc[pick]}**: gross / budget {np.exp(me.log_mult):.2f}×. Its 15 nearest films have a "
                f"median of {out.multiple.median():.2f}×, against {np.exp(f.log_mult).median():.2f}× for all "
                f"{len(f):,} films in this corpus.")
    show_df(out.round(3), key="emb_nb_table")
