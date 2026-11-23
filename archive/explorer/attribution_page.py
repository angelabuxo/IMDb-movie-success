"""Profit attribution page: who and what goes with a film's return, from archive/analysis/attribution/outputs.

It reads only the committed outputs, so it works without data/raw. Every dollar on the page is a split of
gross, scaled by the share of gross you assume the studio keeps; none of it is audited profit.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr

from . import nav
from .common import REPO, md, mtime, show_df, table

OUT = REPO / "archive" / "analysis" / "attribution" / "outputs"
CPI = REPO / "data" / "cpi_us_annual.csv"
HIGH, LOW, NEUTRAL = "#0072B2", "#D55E00", "#999999"
CONTROLS = ["budget", "year", "genre", "franchise"]
ROLES = ["director", "cast", "writer", "producer", "composer", "cinematographer", "editor", "studio", "distributor"]
VIEWS = ["By role", "People and companies", "One film", "Profit measures", "Checks"]
LABEL = {"budget": "budget size", "year": "release year", "genre": "genre", "franchise": "franchise",
         "unexplained": "unexplained (the film itself)"}


@st.cache_data(show_spinner=False)
def _summary(stamp: float) -> dict:
    return json.loads((OUT / "summary.json").read_text())


@st.cache_data(show_spinner=False)
def _real_factor(stamp: float) -> tuple[pd.Series, int]:
    from analysis.attribution.cpi import deflator
    return deflator()


def _films(real: bool) -> tuple[pd.DataFrame, pd.Series]:
    """The film table and a multiplier per film: 1 for nominal dollars, the CPI factor of its year for real ones."""
    f = table(OUT / "films.csv")
    k = f.year.map(_real_factor(mtime(CPI))[0]).fillna(1.0) if real else pd.Series(1.0, index=f.index)
    return f, k


def _usd(v: float) -> str:
    a = abs(v)
    s = f"${a / 1e9:,.2f}B" if a >= 1e9 else f"${a / 1e6:,.1f}M" if a >= 1e6 else f"${a:,.0f}"
    return ("-" if v < 0 else "") + s


def attribution():
    st.title("Profit attribution")
    if not (OUT / "summary.json").exists():
        st.warning("No saved outputs yet. Run `python -m analysis.attribution.run` from the repo root (about ten minutes).")
        return
    s = _summary(mtime(OUT / "summary.json"))
    st.caption(md(
        f"{s['sample']['films']:,} clean films, {s['sample']['years'][0]} to {s['sample']['years'][1]}. A film's "
        "ln(gross / budget) is split into budget, year, genre, franchise, one effect for each person and company on "
        "it, and what's left. Each effect is fitted with everyone else's at once and shrunk towards zero when there "
        "are few films behind it. This describes who goes with high returns. It doesn't show who caused them, and "
        "gross over budget isn't profit. Method and limits: `archive/docs/profit-attribution.md`."))
    with st.sidebar:
        st.subheader("Dollars")
        share = st.slider("Share of gross the studio keeps", 0.3, 0.7, 0.5, 0.05, key="attr_share",
                          help="An assumption, not a measurement: cinemas keep part of the gross and the split isn't "
                               "published per film. 0.5 is the rule De Vany and Walls (1999) use. Cinemark's 2013 annual "
                               "report puts film rentals and advertising at 55.8% of its US admissions and 48.9% abroad. "
                               "Every dollar figure on this page is a gross figure times this share. Marketing costs are "
                               "still missing.")
        real = st.toggle("Adjust for inflation", CPI.exists(), key="attr_real", disabled=not CPI.exists(),
                         help="Turns every year's dollars into dollars of the latest year in data/cpi_us_annual.csv, "
                              "with the US consumer price index (BLS CPI-U) of the film's release year. Real data, with "
                              "limits: budgets are spent before release, and a worldwide gross isn't all earned in the US.")
        if real:
            st.caption(f"Dollars are {_real_factor(mtime(CPI))[1]} US dollars.")
    view = st.radio("View", VIEWS, horizontal=True, key="attr_view", label_visibility="collapsed")
    {"By role": _by_role, "People and companies": _entities, "One film": _film, "Profit measures": _measures,
     "Checks": _checks}[view](s, share, real)


# ---------------------------------------------------------------- by role

def _by_role(s: dict, share: float, real: bool):
    sh, nl = s["shapley"], s["null"]
    c = st.columns(4)
    c[0].metric("R², budget, year, genre", f"{s['cv']['r2_base']:.3f}")
    c[1].metric("R², adding who made it", f"{s['cv']['r2_full']:.3f}")
    c[2].metric("R², earlier years only", f"{s['forward']['r2_full']:.3f}",
                help=f"Films from {s['forward']['from_year']} on, effects fitted on films of earlier years only. "
                     f"Budget, year and genre alone: {s['forward']['r2_base']:.3f}.")
    c[3].metric("Gain if shuffled", f"{nl['gain_over_base_null_max']:+.3f} at most",
                help=f"{nl['shuffles']} shuffles of returns among films of the same year. Real gain: "
                     f"{nl['gain_over_base_observed']:+.3f}.")
    st.caption("R² is cross-validated: every film is predicted from a model fitted without it and without the other "
               "films of its franchise.")

    d = pd.DataFrame({"block": list(sh["share"]), "Shapley share": list(sh["share"].values()),
                      "alone on the base": [sh["alone_on_base"][b] for b in sh["share"]],
                      "lost when dropped": [sh["lost_when_dropped"][b] for b in sh["share"]],
                      "alone, shuffled (max)": [nl["alone_on_base_null_max"][b] for b in sh["share"]]})
    d = d.sort_values("Shapley share")
    fig = go.Figure()
    fig.add_bar(y=d.block, x=d["Shapley share"], orientation="h", marker_color=HIGH, name="Shapley share of R²")
    fig.add_scatter(y=d.block, x=d["alone on the base"], mode="markers", marker=dict(color="#222", symbol="line-ns-open", size=14),
                    name="alone on the base")
    fig.add_scatter(y=d.block, x=d["alone, shuffled (max)"], mode="markers", marker=dict(color=LOW, symbol="x", size=8),
                    name="alone, returns shuffled (largest)")
    fig.update_layout(template="simple_white", height=420, xaxis_title="cross-validated R² added to budget, year and genre",
                      legend=dict(orientation="h", y=1.12), title="How much of the variation in return goes with each block")
    st.plotly_chart(fig, width="stretch", key="attr_shapley")
    st.caption(md(
        "Bar: the block's Shapley share, its gain in R² averaged over every order in which the ten blocks can be added, "
        "so shared information is split and the bars sum to the total gain "
        f"({sh['r2_all'] - sh['r2_base']:.3f}). Tick: the gain when the block is added on its own. Cross: the largest "
        "gain on its own when returns are shuffled within year."))
    rel = pd.DataFrame(s["split_half"]["by_role"]).T.rename_axis("block").reset_index()
    t = d.merge(rel[["block", "mean_r", "entities"]], on="block", how="left").sort_values("Shapley share", ascending=False)
    t["penalty"] = t.block.map(s["penalties"])
    t["with an effect"] = t.block.map(s["sample"]["entities_with_an_effect"])
    show_df(t.rename(columns={"mean_r": "split-half r", "entities": "entities in the split-half test"}).round(4), key="attr_roles",
            caption="Split-half r: effects fitted on two random halves of the films, correlated across people or companies "
                    "with at least three films in each half. Near 0 means the individual effects are mostly noise even "
                    "when the block as a whole predicts.")

    st.subheader("In dollars")
    f, k = _films(real)
    u = f[[c for c in f.columns if c.startswith("usd_")]].mul(k * share, axis=0)
    dd = pd.DataFrame({"block": [c[4:] for c in u.columns], "net": u.sum().to_numpy(), "moved": u.abs().sum().to_numpy()})
    dd["block"] = dd.block.map(lambda b: LABEL.get(b, b))
    dd = dd.sort_values("moved")
    fig = go.Figure()
    fig.add_bar(y=dd.block, x=dd.moved, orientation="h", marker_color="#cfd8dc", name="moved (sum of absolute amounts)")
    fig.add_bar(y=dd.block, x=dd.net, orientation="h", marker_color=[HIGH if v >= 0 else LOW for v in dd.net], name="net")
    fig.update_layout(template="simple_white", height=460, barmode="overlay",
                      xaxis_title=f"USD, {'inflation-adjusted' if real else 'nominal'}, all films summed",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch", key="attr_usd_blocks")
    st.caption(md(
        f"Each film's gross minus its baseline (its budget times the average multiple) is split between its blocks so "
        f"the parts add up exactly, then multiplied by the {share:.0%} share. "
        + ("Each film's dollars are converted with the consumer price index of its release year." if real
           else "Nominal dollars, so recent films weigh more.")))


# ---------------------------------------------------------------- people and companies

def _entities(s: dict, share: float, real: bool):
    e = table(OUT / "entities.csv").copy()
    c = st.columns([2, 1, 1, 2])
    roles = c[0].multiselect("Role", ROLES, default=["director"], key="attr_roles_pick")
    most = int(e.films.max())
    k = c[1].number_input("At least this many films", 2, most, 5, key="attr_min_films")
    sure = c[2].toggle("Interval off zero only", False, key="attr_sure",
                       help="Keep only effects whose 90% bootstrap interval excludes zero.")
    order = c[3].selectbox("Sort by", ["effect", "total dollars", "dollars per film", "films"], key="attr_sort")
    d = e[e.role.isin(roles or ROLES) & (e.films >= k)]
    if sure:
        d = d[(d.lo90 > 0) | (d.hi90 < 0)]
    f, kf = _films(real)
    fe = table(OUT / "film_entities.csv.gz")
    tot = (fe.usd * fe.tconst.map(pd.Series(kf.to_numpy(), index=f.tconst)) * share).groupby([fe.role, fe.id]).agg(["sum", "mean"])
    d = d.drop(columns=["usd_total", "usd_per_film"]).join(tot.rename(columns={"sum": "usd_total", "mean": "usd_per_film"}),
                                                           on=["role", "id"])
    col = {"effect": "effect", "total dollars": "usd_total", "dollars per film": "usd_per_film", "films": "films"}[order]
    d = d.sort_values(col, ascending=False)
    st.caption(f"{len(d):,} people or companies. Effect is in ln(gross / budget): +0.10 means films gross about 10.5% more "
               "of their budget with them than with an average person in that seat. Someone with several roles has one "
               "row per role.")
    if d.empty:
        return
    top = pd.concat([d.head(15), d.tail(15)]).drop_duplicates(["role", "id"]) if len(d) > 30 else d
    top = top.sort_values(col)
    lab = top.name.fillna(top.id) + " (" + top.films.astype(str) + ")"
    fig = go.Figure()
    xcol = col if col != "films" else "effect"
    if xcol == "effect":
        fig.add_scatter(x=top.effect, y=lab, mode="markers", marker_color=[HIGH if v >= 0 else LOW for v in top.effect],
                        error_x=dict(type="data", symmetric=False, array=top.hi90 - top.effect, arrayminus=top.effect - top.lo90,
                                     color="#bbb"), hovertemplate="%{y}: %{x:+.3f}<extra></extra>")
        fig.add_vline(x=0, line_dash="dot", line_color="#888")
        title = "Effect with 90% bootstrap interval (films in brackets)"
    else:
        fig.add_bar(x=top[xcol], y=lab, orientation="h", marker_color=[HIGH if v >= 0 else LOW for v in top[xcol]])
        title = "Dollars at the assumed share (films in brackets)"
    fig.update_layout(template="simple_white", height=max(360, 22 * len(top)), title=title, margin=dict(l=10))
    st.plotly_chart(fig, width="stretch", key="attr_entities_fig")
    cols = ["role", "name", "films", "first_year", "last_year", "effect", "lo90", "hi90", "factor", "one_at_a_time",
            "usd_total", "usd_per_film", "id"]
    show_df(d[cols].round({"effect": 3, "lo90": 3, "hi90": 3, "factor": 3, "one_at_a_time": 3, "usd_total": 0, "usd_per_film": 0}),
            key="attr_entities",
            caption="one_at_a_time: the shrunk mean of their films' returns after budget, year, genre and franchise only, "
                    "ignoring who else was on the film. The gap to effect is what their collaborators explain.")
    pick = st.selectbox("Open in the node explorer", ["(pick one)"] + (d.name.fillna(d.id) + "  ·  " + d.id).head(200).tolist(),
                        key="attr_open")
    if pick != "(pick one)":
        nav.open_node(pick.split("  ·  ")[-1])


# ---------------------------------------------------------------- one film

def _film(s: dict, share: float, real: bool):
    f, kf = _films(real)
    fe = table(OUT / "film_entities.csv.gz")
    f = f.assign(k=kf, label=f.title + " (" + f.year.astype(str) + ")")
    labels = f.sort_values("gross_usd", ascending=False).label.tolist()
    want = st.session_state.get("attr_film")
    pick = st.selectbox("Film", labels, index=labels.index(want) if want in labels else 0, key="attr_film_pick")
    r = f[f.label == pick].iloc[0].copy()
    for col in [c for c in f.columns if c.startswith("usd_") or c in ("budget_usd", "gross_usd", "baseline_gross_usd")]:
        r[col] = r[col] * r.k
    c = st.columns(5)
    c[0].metric("Budget", _usd(r.budget_usd))
    c[1].metric("Gross", _usd(r.gross_usd))
    c[2].metric("Gross ÷ budget", f"{r.gross_usd / r.budget_usd:.2f}")
    c[3].metric(f"{share:.0%} of gross − budget", _usd(share * r.gross_usd - r.budget_usd))
    c[4].metric("Predicted ÷ actual gross", f"{np.exp(r.predicted - r.log_mult):.2f}",
                help="From a model fitted without this film or its franchise.")

    detail = st.toggle("Split roles into people and companies", True, key="attr_detail")
    parts = [(LABEL[b], r[f"usd_{b}"]) for b in CONTROLS]
    mine = fe[fe.tconst == r.tconst]
    mine = mine.assign(usd=mine.usd * r.k)
    if detail:
        parts += [(f"{x.name if isinstance(x.name, str) else x.id} · {x.role}", x.usd) for x in mine.itertuples(index=False)]
    else:
        parts += [(b, r[f"usd_{b}"]) for b in ROLES if abs(r[f"usd_{b}"]) > 0]
    parts.append((LABEL["unexplained"], r.usd_unexplained))
    names = ["baseline"] + [p[0] for p in parts] + ["studio's share of gross"]
    vals = [share * r.baseline_gross_usd] + [share * p[1] for p in parts] + [0]
    fig = go.Figure(go.Waterfall(orientation="h", y=names, x=vals, measure=["absolute"] + ["relative"] * len(parts) + ["total"],
                                 increasing=dict(marker_color=HIGH), decreasing=dict(marker_color=LOW),
                                 totals=dict(marker_color="#444"), connector=dict(line=dict(color="#ddd"))))
    fig.add_vline(x=r.budget_usd, line_dash="dash", line_color="#888", annotation_text="budget")
    fig.update_layout(template="simple_white", height=max(420, 24 * len(names)), yaxis=dict(autorange="reversed"),
                      xaxis_title=f"USD ({share:.0%} of gross)")
    st.plotly_chart(fig, width="stretch", key="attr_waterfall")
    st.caption(md(
        "Baseline: what a film of this budget grosses at the average multiple. Each bar moves it by that block's "
        "Shapley share, so the bars end exactly at the actual figure. Effects come from each person's other films; "
        "anyone with no other film in the sample has no bar, and their part sits in *unexplained*. Two people who "
        "always work together can't be told apart and share their effect equally."))

    a, b = st.columns(2)
    with a:
        if len(mine):
            show_df(mine.assign(usd=mine.usd * share, factor=np.exp(mine.effect))[["role", "name", "effect", "factor", "usd", "id"]]
                    .sort_values("usd", ascending=False).round({"effect": 3, "factor": 3, "usd": 0}), key="attr_film_entities",
                    caption="This film's people and companies")
        else:
            st.info("Nobody on this film has another film in the sample, so the split has controls and the unexplained part only.")
    with b:
        t = pd.DataFrame({"block": CONTROLS + ROLES,
                          "out of fold (ln)": [r[f"ln_{x}"] for x in CONTROLS + ROLES],
                          "earlier years only (ln)": [r.get(f"fwd_{x}", np.nan) for x in CONTROLS + ROLES]})
        show_df(t.round(3), key="attr_film_modes",
                caption="The same split fitted two ways. Out of fold uses the other films of every year. Earlier years "
                        f"only is what could be known before release (blank before {s['forward']['from_year']}). The "
                        "control rows are centred in the first column and not in the second.")


# ---------------------------------------------------------------- profit measures

def _measures(s: dict, share: float, real: bool):
    f, kf = _films(real)
    f = f.assign(budget_usd=f.budget_usd * kf, gross_usd=f.gross_usd * kf)
    est = share * f.gross_usd - f.budget_usd
    lm = f.log_mult
    c = st.columns(4)
    c[0].metric("Break-even gross ÷ budget", f"{1 / share:.2f}")
    c[1].metric("Films above it", f"{(est > 0).mean():.1%}", help=f"Grossed more than budget: {(lm > 0).mean():.1%}.")
    c[2].metric("Sum over all films", _usd(est.sum()))
    gains = est[est > 0].sort_values(ascending=False)
    c[3].metric("Gains from the top 10% of films", f"{gains.head(len(f) // 10).sum() / gains.sum():.0%}")
    st.caption(md(
        f"Estimated profit here is {share:.0%} of gross minus budget. The share is an assumption you set in the sidebar: "
        "it stands for the cinemas' cut. Marketing, home video and TV income are all missing, so read it as a scenario."))
    a, b = st.columns(2)
    clip = est.clip(*np.percentile(est, [1, 99])) / 1e6
    a.plotly_chart(px.histogram(clip, nbins=80, template="simple_white", height=360, color_discrete_sequence=[HIGH],
                                title="Estimated profit per film, USD millions (clipped at the 1st and 99th percentile)")
                   .add_vline(x=0, line_dash="dash").update_layout(showlegend=False, xaxis_title=None), width="stretch")
    d = f.assign(est_musd=est / 1e6, multiple=f.gross_usd / f.budget_usd, budget_musd=f.budget_usd / 1e6)
    b.plotly_chart(px.scatter(d, x="multiple", y="est_musd", log_x=True, hover_name="title", hover_data=["year", "budget_musd"],
                              opacity=.35, template="simple_white", height=360, color_discrete_sequence=[NEUTRAL],
                              title=f"The two measures rank films differently (Spearman {spearmanr(lm, est).statistic:+.2f})")
                   .add_vline(x=1 / share, line_dash="dash").add_hline(y=0, line_dash="dash"), width="stretch")
    k = len(f) // 10
    top_m, top_e = set(lm.nlargest(k).index), set(est.nlargest(k).index)
    st.caption(f"Of the top 10% of films by gross ÷ budget, {len(top_m & top_e) / k:.0%} are also in the top 10% by estimated "
               f"dollars. The ratio favours cheap films; dollars favour big ones. Median budget of the two top groups: "
               f"{_usd(f.budget_usd[list(top_m)].median())} against {_usd(f.budget_usd[list(top_e)].median())}.")
    sc = pd.DataFrame(s["measures"]["scenarios"]).T.rename_axis("share kept").reset_index()
    show_df(sc.round(3), key="attr_scenarios", caption="The same numbers at three fixed shares, as saved by the run (nominal dollars)")
    l, r = st.columns(2)
    cols = ["title", "year", "budget_usd", "gross_usd", "multiple", "est_musd"]
    with l:
        show_df(d.nlargest(25, "est_musd")[cols].round(2), key="attr_top_est", caption="Highest estimated profit")
    with r:
        show_df(d.nsmallest(25, "est_musd")[cols].round(2), key="attr_low_est", caption="Lowest estimated profit")


# ---------------------------------------------------------------- checks

def _checks(s: dict, share: float, real: bool):
    nl, sh = s["null"], s["shapley"]
    st.markdown(md(
        f"**Shuffle null.** Returns shuffled among films of the same release year, {nl['shuffles']} times. The gain in "
        f"cross-validated R² over budget, year and genre is {nl['gain_over_base_observed']:+.3f} on the real returns "
        f"and between {nl['gain_over_base_null_mean']:+.3f} (mean) and {nl['gain_over_base_null_max']:+.3f} (largest) "
        "on shuffled ones. Per block, added alone:"))
    t = pd.DataFrame({"block": list(sh["alone_on_base"]), "observed": list(sh["alone_on_base"].values()),
                      "shuffled mean": [nl["alone_on_base_null_mean"][b] for b in sh["alone_on_base"]],
                      "shuffled max": [nl["alone_on_base_null_max"][b] for b in sh["alone_on_base"]],
                      "shuffles at or above observed": [nl["alone_on_base_shuffles_at_or_above_observed"][b] for b in sh["alone_on_base"]]})
    show_df(t.round(4), key="attr_null")
    st.markdown("**Split halves, a second method and the bootstrap.**")
    rel = pd.DataFrame(s["split_half"]["by_role"]).T
    sec = pd.DataFrame(s["second_method"]["by_role"]).T
    boot = pd.Series(s["bootstrap"]["share_of_effects_with_interval_off_zero"], name="share with 90% interval off zero")
    t = rel[["mean_r", "sd_r", "entities"]].rename(columns={"mean_r": "split-half r", "sd_r": "sd over splits",
                                                            "entities": "entities compared"}).join(
        sec.rename(columns={"spearman": "rank agreement with one-at-a-time", "sd_joint": "sd of effects",
                            "sd_one_at_a_time": "sd one-at-a-time"})).join(boot).rename_axis("role").reset_index()
    show_df(t.astype({c: float for c in t.columns if c != "role"}).round(3), key="attr_reliability", caption=(
        "Split-half r: are a role's individual effects stable across two disjoint sets of films? One-at-a-time: the "
        "shrunk mean residual of each entity's films, ignoring collaborators. By chance, 10% of 90% intervals would "
        "exclude zero."))
    st.markdown(md(
        f"**Earlier years only.** For films from {s['forward']['from_year']} on ({s['forward']['films']:,}), with effects "
        f"fitted on earlier years: R² {s['forward']['r2_full']:.3f} against {s['forward']['r2_base']:.3f} for budget, "
        f"year and genre. Rank correlation within release year: {s['forward']['spearman_within_year_full']:+.3f} "
        f"against {s['forward']['spearman_within_year_base']:+.3f}."))
    f = table(OUT / "films.csv")
    gap = (f[[c for c in f.columns if c.startswith("usd_")]].sum(axis=1) - (f.gross_usd - f.baseline_gross_usd)).abs().max()
    st.markdown(md(f"**Dollars add up.** Largest gap between a film's parts and its gross minus baseline: ${gap:,.2f}."))
    with st.expander("Every number in summary.json"):
        from .common import flatten_json
        show_df(flatten_json({k: v for k, v in s.items() if k != "tuning_path"}), key="attr_json")
