"""Q1 genre page: do some genres earn more than others at the same budget? It reads only the committed outputs in
analysis/genre/outputs (written by the analysis.genre scripts), so it works without data/raw. The method and its
limits are in analysis/genre/README.md, CHECKS.md and COURSE.md; notebooks/q1_genre.ipynb walks through the same
numbers.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .common import REPO, md, show_df, table

OUT = REPO / "analysis" / "genre" / "outputs"
ABOVE, BELOW, INSIDE = "#2f6db5", "#c0504d", "#b8b8b8"
VIEWS = ["Main test", "Robustness", "Budget tiers", "Genre network", "People and genre", "Genre shares",
         "Before release", "Horror words"]


def _json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _colour(p, e):
    return INSIDE if p >= .05 else (ABOVE if e > 0 else BELOW)


def _main_test(sh: pd.DataFrame, summ: dict):
    st.markdown(md("For each genre, the mean ln(gross / budget) of its films against the same mean after shuffling "
                   "returns only between films in the same budget decile (10,000 shuffles). A bar outside the grey band "
                   "(Holm p < .05) returns more, or less, than its budgets predict. Gross over budget isn't profit."))
    sets = {"usable_money": "All films with usable money", "in_common_set": "Common set (films with a script)"}
    s = st.radio("Films", list(sets), format_func=sets.get, horizontal=True, key="q1_set")
    d = sh[(sh["set"] == s) & (sh["outcome"] == "ln_ratio") & (sh["null"] == "budget")].sort_values("mean")
    info = summ.get("sets", {}).get(s, {})
    st.caption(md(f"{info.get('films', 0):,} films, {len(d)} genres with at least {summ.get('min_films', 100)} films, "
                  f"{info.get('mean_genres_per_film', '?')} genres per film. Genres overlap, so the tests aren't independent."))
    fig = go.Figure()
    fig.add_bar(y=d["genre"], x=d["mean"], orientation="h", marker_color=[_colour(p, e) for p, e in zip(d["p_holm"], d["excess"])],
                customdata=np.c_[d["n_films"], d["excess"], np.exp(d["excess"]), d["p_holm"]],
                hovertemplate="%{y}: mean %{x:.3f}<br>%{customdata[0]} films<br>excess %{customdata[1]:+.3f} "
                              "(x%{customdata[2]:.2f} gross/budget)<br>Holm p %{customdata[3]:.3f}<extra></extra>", name="observed")
    fig.add_scatter(y=d["genre"], x=(d["null_lo95"] + d["null_hi95"]) / 2, mode="markers", name="95% of shuffles",
                    marker=dict(symbol="line-ns-open", size=14, color="#222"),
                    error_x=dict(type="data", symmetric=False, array=d["null_hi95"] - (d["null_lo95"] + d["null_hi95"]) / 2,
                                 arrayminus=(d["null_lo95"] + d["null_hi95"]) / 2 - d["null_lo95"], color="#555", thickness=6, width=0),
                    hoverinfo="skip")
    fig.update_layout(height=40 + 26 * len(d), xaxis_title="mean ln(gross / budget)", showlegend=False, margin=dict(l=10, r=10, t=10, b=40))
    st.plotly_chart(fig, width="stretch")
    t = d.sort_values("excess", ascending=False)[["genre", "n_films", "mean", "null_mean", "excess", "z", "p_holm"]].copy()
    t["x gross/budget"] = np.exp(t["excess"]).round(2)
    show_df(t.rename(columns={"null_mean": "expected"}), "q1_main", "Excess = the genre's mean minus what its budgets predict.")


def _robustness(sh: pd.DataFrame, summ: dict):
    st.markdown(md("The same test with other nulls, the regression as a second method, and five checks added after "
                   "the main result. A genre counts as robust when it passes every shuffle."))
    u = sh[sh["set"] == "usable_money"]
    names = {("ln_ratio", "none"): "no matching", ("ln_ratio", "budget"): "budget deciles", ("ln_ratio", "budget_era"): "budget x era",
             ("ln_ratio", "budget_fine"): "50 budget bins", ("rank", "budget"): "ranks"}
    rows = {}
    for (o, n), lab in names.items():
        v = u[(u["outcome"] == o) & (u["null"] == n)].set_index("genre")
        if len(v):
            rows[f"{lab}: excess"] = v["excess"]
            rows[f"{lab}: Holm p"] = v["p_holm"]
    t = pd.DataFrame(rows)
    ols = table(OUT / "genre_ols.csv")
    ols = ols[ols["set"] == "usable_money"].set_index("genre")
    t["OLS centred coef"] = ols["coef_centred"]
    chk = OUT / "checks"
    for f_, col, lab in (("smooth.csv", "excess", "smooth curve"), ("matched.csv", "excess", "matched pairs")):
        if (chk / f_).exists():
            t[lab] = table(chk / f_).set_index("genre")[col]
    if (chk / "trim.csv").exists():
        tr = table(chk / "trim.csv")
        t["without top 10"] = tr[tr["check"] == "trim_top10"].set_index("genre")["excess"]
    show_df(t.sort_values("budget deciles: excess", ascending=False).reset_index(names="genre"), "q1_robust")
    us = summ.get("sets", {}).get("usable_money", {})
    if us:
        st.caption(md(f"Genre adds {us['r2_added_by_genre']:.3f} to R² on top of budget decile and era ({us['r2_budget_era']:.3f}). "
                      f"The regression ranks genres like the shuffle: Spearman {us['spearman_ols_vs_shuffle_excess']}."))
    if (chk / "shares.csv").exists():
        st.subheader("How a genre gets its result: more hits, or fewer flops?")
        sh_ = table(chk / "shares.csv")
        p = sh_.pivot(index="genre", columns="check", values="excess_pp").rename(
            columns={"hit_share": "hit share (pp)", "below_budget_share": "share below budget (pp)"})
        show_df(p.sort_values("hit share (pp)", ascending=False).reset_index(), "q1_shares_pp",
                "Percentage points above the budget-matched shuffle. A hit is the top quarter of returns within an era.")


def _tiers():
    chk = OUT / "checks"
    if not (chk / "tiers.csv").exists():
        st.info("Run `python -m analysis.genre.q1_checks` for the tier view.")
        return
    d = table(chk / "tiers.csv")
    st.markdown(md("Inside each fixed budget tier, every genre against a shuffle within that tier's budget deciles. "
                   "Empty cells have fewer than 20 films; * marks Holm p < .05 over all filled cells."))
    order = d.groupby("genre")["excess"].mean().sort_values(ascending=False).index
    tiers = ["under $5M", "$5-30M", "$30-100M", "$100M+"]
    E = d.pivot(index="genre", columns="tier", values="excess").reindex(index=order, columns=tiers)
    P = d.pivot(index="genre", columns="tier", values="p_holm").reindex(index=order, columns=tiers)
    N = d.pivot(index="genre", columns="tier", values="n_films").reindex(index=order, columns=tiers)
    txt = [[("" if np.isnan(E.iat[i, k]) else f"{E.iat[i, k]:+.2f}{' *' if P.iat[i, k] < .05 else ''}") + f"<br>n={int(N.iat[i, k])}"
            for k in range(len(tiers))] for i in range(len(order))]
    lim = float(np.nanmax(np.abs(E.to_numpy())))
    fig = go.Figure(go.Heatmap(z=E.to_numpy(), x=tiers, y=list(order), text=txt, texttemplate="%{text}", colorscale="RdBu",
                               zmin=-lim, zmax=lim, colorbar=dict(title="excess"), hoverongaps=False))
    fig.update_layout(height=60 + 30 * len(order), yaxis_autorange="reversed", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(md("Genre matters most for $5-100M films; above $100M only Drama differs."))


def _network():
    net = OUT / "network"
    s = _json(net / "summary.json")
    if not s:
        st.info("Run `python -m analysis.genre.q1_network` for this view.")
        return
    g = s["genre_network"]
    st.markdown(md("Genres linked by the films they share, against a curveball null that keeps every film's number of "
                   "genres and every genre's size (course week 2). Links that beat it form the backbone; Louvain splits it "
                   "into families (week 4); colour is the genre's budget-matched return."))
    c = st.columns(4)
    c[0].metric("Backbone links", f"{g['backbone_links']} of {g['pairs']}")
    c[1].metric("Also in disparity filter", g["both"])
    c[2].metric("Louvain modularity", f"{g['modularity']:.2f}")
    c[3].metric("Rewired-null p", f"{g['modularity_p']:.2f}")
    img = net / "figures" / "q1_genre_network.png"
    if img.exists():
        st.image(str(img), width="stretch")
    show_df(table(net / "genre_families.csv"), "q1_families",
            "Each film split across families by its share of genres. No family differs from the shuffle, so the return "
            "effect belongs to single genres, not to the families they travel with.")
    pairs = table(net / "genre_pairs.csv")
    show_df(pairs.sort_values("z", ascending=False), "q1_pairs", "Every genre pair: films in both, the null's expectation, z.")


def _people():
    net, snw = OUT / "network", OUT / "scripts_network"
    s = _json(net / "summary.json").get("film_network", {})
    if s:
        st.markdown(md(f"Films linked when they share a director, writer, producer or one of the first three billed actors "
                       f"({s['links']:,} links between {s['films']:,} films). For each genre, the assortativity of its 0/1 "
                       f"indicator (course week 3), against genre sets shuffled within era. {s['share_links_sharing_a_genre']:.0%} "
                       f"of links join two films sharing a genre, against {s['share_null_mean']:.0%} after shuffling."))
        a = table(net / "film_assortativity.csv").sort_values("assortativity")
        fig = px.bar(a, x="assortativity", y="genre", orientation="h", color_discrete_sequence=[ABOVE])
        fig.update_layout(height=40 + 24 * len(a), margin=dict(l=10, r=10, t=10, b=40))
        st.plotly_chart(fig, width="stretch")
    ss = _json(snw / "summary.json").get("specialists", {})
    if ss:
        st.subheader("Genre specialists")
        st.markdown(md(f"A film's team and their films from *earlier* years only: how much of the team's experience is in the "
                       f"film's genres. Outcome: the return net of budget and era, with genre and a sequel flag held fixed "
                       f"({ss['films']:,} films; {ss['sequels']} sequels)."))
        q = table(snw / "specialists_quintiles.csv")
        fig = px.bar(q, x="genre_experience_range", y="mean_net_return", color_discrete_sequence=[ABOVE],
                     labels={"genre_experience_range": "team's earlier share of credits in the film's genres", "mean_net_return": "return net of budget and era"})
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=40))
        st.plotly_chart(fig, width="stretch")
        show_df(table(snw / "specialists_ols.csv"), "q1_spec", "Coefficients per standard deviation, HC3 intervals.")


def _shares():
    sf, fx = OUT / "share_followups", OUT / "share_fixes"
    st.markdown(md("The Tag Genome's genre shares (how much of a film each genre is) instead of IMDb's 0/1 tags. They come "
                   "from users' tags and reviews after release and stop in 2014, so they describe films rather than predict "
                   "them. The fixes choose the tag per genre and the power from agreement with IMDb's labels, never from returns."))
    s = _json(fx / "summary.json")
    if s:
        c = st.columns(3)
        c[0].metric("Power chosen", s["power_chosen"])
        c[1].metric("Genres remapped", len(s["mapping_changed"]))
        c[2].metric("Films with scores", f"{s['films_with_tag_genome']:,}")
        cr = table(fx / "cv_r2.csv")
        show_df(cr, "q1_cvr2", "Cross-validated R² added over each base model. With rating and votes in the base, the shares "
                               "are judged on what they add beyond reception and audience size.")
        show_df(table(fx / "spread.csv"), "q1_spread", "Genre mix per SD: alone it looks good; holding fame fixed, more mixed films earn less.")
        show_df(table(fx / "coverage.csv"), "q1_cov", "Budget-netted genre means: full set, Tag Genome subset, and the subset reweighted.")
    if (sf / "assortativity.csv").exists():
        a = table(sf / "assortativity.csv").pivot(index="genre", columns="representation", values="assortativity")
        show_df(a.sort_values(a.columns[-1], ascending=False).reset_index(), "q1_sh_assort", "Assortativity with tags vs with shares.")
    hc = fx / "hand_check_sheet.csv"
    if hc.exists():
        h = table(hc).fillna("")
        done = (h["which_is_right"].astype(str) != "").sum()
        st.subheader("Hand check (for the group)")
        st.caption(md(f"{len(h)} films where the dominant share is none of the film's IMDb genres; {done} labelled so far. "
                      "Fill in `which_is_right` in analysis/genre/outputs/share_fixes/hand_check_sheet.csv."))
        show_df(h, "q1_hand")


def _before_release():
    fx, snw = OUT / "share_fixes", OUT / "scripts_network"
    st.markdown(md("Genre measured from what exists before release: the screenplay and the people."))
    if (fx / "script_shares_validation.csv").exists():
        st.subheader("Genre shares predicted from the screenplay")
        show_df(table(fx / "script_shares_validation.csv"), "q1_scriptval", "Is it a valid genre measure? AUC against IMDb labels, "
                "agreement with the Tag Genome (out-of-fold), and how often its dominant genre is an IMDb genre.")
        show_df(table(fx / "script_shares_cv_r2.csv"), "q1_scriptr2", "On the common set: script shares against IMDb tags.")
    s = _json(snw / "summary.json").get("typicality", {})
    if s:
        st.subheader("Script typicality")
        st.markdown(md("How close a script's embedding is to the usual script of its genres. Typical scripts beat their budgets."))
        q = table(snw / "typicality_quintiles.csv")
        fig = px.bar(q, x="quintile", y="mean_net_return", color_discrete_sequence=[ABOVE],
                     labels={"quintile": "script typicality (quintile)", "mean_net_return": "return net of budget and era"})
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=40))
        st.plotly_chart(fig, width="stretch")
        show_df(table(snw / "typicality_ols.csv"), "q1_typ")
        show_df(table(snw / "typicality_per_genre.csv"), "q1_typg", "Correlation per genre.")


def _horror():
    w = OUT / "words"
    s = _json(w / "horror_summary.json")
    if not s:
        st.info("Run `python -m analysis.genre.q1_words` for this view.")
        return
    st.markdown(md(f"Wikipedia plots of {s['films_with_plot']} Horror films: top quarter against bottom quarter of return net "
                   f"of budget and era, compared word by word (log-odds with an informative prior, course week 6), labels "
                   f"shuffled within budget deciles. Only words and counts are stored; no plot text. Family-wise p for the "
                   f"strongest word: {s['familywise_p_max_abs_calibrated_z']}."))
    top = table(w / "horror_top.csv")
    c = st.columns(2)
    for col, side, lab in ((c[0], "hit", "Beat the budget"), (c[1], "flop", "Fell short")):
        col.markdown(f"**{lab}**")
        col.dataframe(top[top["side"] == side][["word", "calibrated_z", "hit_films", "flop_films"]], hide_index=True, width="stretch")
    if (w / "horror_supernatural_by_period.csv").exists():
        show_df(table(w / "horror_supernatural_by_period.csv"), "q1_sup", "Post hoc: plots with any supernatural word, by period.")
    hs = _json(OUT / "scripts_network" / "summary.json").get("horror_scripts", {})
    if hs:
        sp = hs["supernatural_in_script"]
        st.caption(md(f"In the screenplays (written before release) it doesn't replicate: {sp['share']:.0%} of Horror scripts use a "
                      f"supernatural word, and they differ by {sp['difference']:+.3f} (p {sp['p']})."))


def genre():
    st.title("Q1: genre and returns")
    sh_path = OUT / "genre_shuffle.csv"
    if not sh_path.exists():
        st.warning("No saved outputs yet. Run `python -m analysis.genre.q1_genre` from the repo root (about five seconds).")
        return
    st.caption(md("Do some genres earn more than others once we compare films with similar budgets? Every number here comes "
                  "from analysis/genre/outputs; the method and its limits are in analysis/genre/README.md, CHECKS.md and COURSE.md."))
    sh, summ = table(sh_path), _json(OUT / "summary.json")
    view = st.segmented_control("View", VIEWS, default=VIEWS[0], key="q1_view") or VIEWS[0]
    {"Main test": lambda: _main_test(sh, summ), "Robustness": lambda: _robustness(sh, summ), "Budget tiers": _tiers,
     "Genre network": _network, "People and genre": _people, "Genre shares": _shares, "Before release": _before_release,
     "Horror words": _horror}[view]()
