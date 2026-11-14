"""Step 5: arcs. Each script's scene scores (in order) resampled to a fixed grid of story time, smoothed,
and summarised by the pre-registered descriptors. Numbers only; no outcome is used in this step.

The arc maths is written as matrices so Step 6 (1,000 shuffles per script) and Step 9 (other windows and
grid sizes) reuse it: arc = R @ x (resampled), smooth = M @ x with M = S @ R."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import spearmanr

from analysis.scripts_text.report import HIGH, LOW, NEUTRAL, Fig, Md, Reading, Run, Table, style

from . import s02_units, s03_instruments
from .common import INSTRUMENT_NAMES, primary_instrument
from .config import cfg, log_run
from .paths import OUTPUTS

OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def _paths(inst: str):
    return OUTPUTS / f"s05_arcs_{inst}.csv.gz", OUTPUTS / f"s05_descriptors_{inst}.csv", OUTPUTS / f"s05_summary_{inst}.json"


# ---------------------------------------------------------------- arc maths

@lru_cache(maxsize=4096)
def resample_matrix(n: int, T: int) -> np.ndarray:
    """T x n. Scene i covers [i/n, (i+1)/n); grid point k covers [k/T, (k+1)/T). Row k holds the overlap of
    point k with each scene, normalised to sum to 1 (overlap-weighted mean)."""
    a = np.arange(n + 1) / n
    b = np.arange(T + 1) / T
    W = np.clip(np.minimum(b[1:, None], a[None, 1:]) - np.maximum(b[:-1, None], a[None, :-1]), 0, None)
    return W / W.sum(1, keepdims=True)


@lru_cache(maxsize=64)
def smooth_matrix(T: int, w: int) -> np.ndarray:
    """T x T centred moving average of width w (odd), truncated at the ends."""
    h = w // 2
    k = np.arange(T)
    S = (np.abs(k[:, None] - k[None, :]) <= h).astype(float)
    return S / S.sum(1, keepdims=True)


def describe(x: np.ndarray, T: int | None = None, w: int | None = None) -> dict:
    """Descriptors for scene scores x (1-D, story order) or a batch (rows = orderings of the same scenes)."""
    a = cfg()["arc"]
    T = T or a["time_points"]
    w = w or a["smoothing_window"]
    x = np.atleast_2d(np.asarray(x, float))
    R = resample_matrix(x.shape[1], T)
    raw = x @ R.T
    sm = raw @ smooth_matrix(T, w).T
    e = int(round(a["edge_share"] * T))
    dev = np.sign(sm - sm.mean(1, keepdims=True))
    return {"D1": sm.std(1), "D2": raw[:, -e:].mean(1) - raw[:, :e].mean(1), "mean_level": raw.mean(1),
            "min_position": (sm.argmin(1) + 0.5) / T,
            "sign_changes": np.array([int((np.diff(r[r != 0]) != 0).sum()) for r in dev]), "raw": raw, "smooth": sm}


def scene_series(inst: str | None = None) -> dict[str, np.ndarray]:
    inst = inst or primary_instrument()
    """Scene scores ('both' layer) per script in story order; only scenes on the timeline."""
    sc = s03_instruments.load_scores(inst, "scene").sort_values(["movie_name", "scene"])
    return {m: g[f"{inst}_both"].to_numpy(float) for m, g in sc.groupby("movie_name", sort=False)}


# ---------------------------------------------------------------- run

def run(inst: str | None = None) -> dict:
    inst = inst or primary_instrument()
    a = cfg()["arc"]
    T = a["time_points"]
    series = scene_series(inst)
    rows, arcs = [], []
    for m, x in series.items():
        d = describe(x)
        odd, even = describe(x[0::2]), describe(x[1::2])
        rows.append({"movie_name": m, "scenes": len(x), **{k: float(d[k][0]) for k in
                     ("D1", "D2", "mean_level", "min_position", "sign_changes")},
                     **{f"{k}_odd": float(odd[k][0]) for k in ("D1", "D2", "mean_level")},
                     **{f"{k}_even": float(even[k][0]) for k in ("D1", "D2", "mean_level")}})
        arcs.append(pd.DataFrame({"movie_name": m, "t": np.arange(T), "raw": d["raw"][0].round(5),
                                  "smooth": d["smooth"][0].round(5)}))
    desc = pd.DataFrame(rows)
    tab = s02_units.load()["table"][["movie_name", "in_primary", "primary_genre", "sentences_both",
                                     "mean_sentences_per_scene", "n_scene_tags", "possible_transcript"]]
    desc = tab.merge(desc, on="movie_name", how="inner", validate="one_to_one")
    arc_p, desc_p, sum_p = _paths(inst)
    pd.concat(arcs).to_csv(arc_p, index=False, compression="gzip")
    desc.to_csv(desc_p, index=False)

    p = desc[desc.in_primary]
    reli = []
    for k in ("D1", "D2", "mean_level"):
        r = float(spearmanr(p[f"{k}_odd"], p[f"{k}_even"]).statistic)
        reli.append({"descriptor": k, "split-half Spearman": r, "Spearman-Brown": 2 * r / (1 + r)})
    proxy = []
    for k in ("D1", "D2", "mean_level"):
        for z, lab in (("scenes", "scenes on the timeline"), ("sentences_both", "sentences"),
                       ("mean_sentences_per_scene", "mean sentences per scene")):
            proxy.append({"descriptor": k, "against": lab, "Spearman": float(spearmanr(p[k], p[z]).statistic)})
    summary = {"instrument": inst, "scripts": int(len(desc)), "primary": int(p.shape[0]),
               "params": {k: a[k] for k in ("time_points", "smoothing_window", "edge_share")},
               "reliability": reli, "proxies": proxy,
               "D1_D2_spearman": float(spearmanr(p.D1, p.D2).statistic),
               "examples": sorted(p.movie_name.sample(6, random_state=cfg()["seed"]).tolist())}
    sum_p.write_text(json.dumps(summary, indent=1) + "\n")
    log_run(f"s05_arcs[{inst}]", [arc_p.name, desc_p.name, sum_p.name], {"arc": a, "instrument": inst})
    return load(inst)


def load(inst: str | None = None) -> dict:
    inst = inst or primary_instrument()
    arc_p, desc_p, sum_p = _paths(inst)
    return {"arcs": pd.read_csv(arc_p, dtype={"movie_name": str}), "desc": pd.read_csv(desc_p, dtype={"movie_name": str}),
            "summary": json.loads(sum_p.read_text())}


# ---------------------------------------------------------------- tables

def tables(res: dict) -> dict:
    s, d = res["summary"], res["desc"]
    p = d[d.in_primary]
    cols = ["D1", "D2", "mean_level", "min_position", "sign_changes"]
    dist = p[cols].describe(percentiles=[.1, .25, .5, .75, .9]).T.round(4).reset_index(names="descriptor (primary 834)")
    rel = pd.DataFrame(s["reliability"]).round(3)
    prox = pd.DataFrame(s["proxies"]).pivot(index="descriptor", columns="against", values="Spearman").round(3).reset_index()
    gen = p.groupby("primary_genre")[["D1", "D2", "mean_level"]].median().round(4)
    gen.insert(0, "films", p.groupby("primary_genre").size())
    ex = p[p.movie_name.isin(s["examples"])][["movie_name", "primary_genre", "scenes", "D1", "D2", "mean_level"]].round(4)
    return {"distribution": dist, "reliability": rel, "proxies": prox, "genres": gen.reset_index(), "examples": ex}


# ---------------------------------------------------------------- figures

def _band(fig, a: pd.DataFrame, colour: str, name: str, col: str = "smooth"):
    g = a.groupby("t")[col]
    m, se = g.mean(), g.std() / np.sqrt(g.size())
    x = (m.index + 0.5) / len(m)
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[m + 1.96 * se, (m - 1.96 * se)[::-1]], fill="toself",
                             fillcolor=colour, opacity=0.2, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=m, line=dict(color=colour, width=2), name=name))


def figures(res: dict) -> dict:
    s, d, a = res["summary"], res["desc"], res["arcs"]
    p = d[d.in_primary]
    ap = a[a.movie_name.isin(p.movie_name)]
    mean = go.Figure()
    _band(mean, ap, NEUTRAL, f"smoothed (n={len(p)})")
    _band(mean, ap, HIGH, "resampled, unsmoothed", "raw")
    mean.update_layout(xaxis_title="story time (0 = first scene, 1 = last)", yaxis_title=f"mean scene score ({INSTRUMENT_NAMES[s['instrument']]})")
    style(mean, "The average arc of the primary sample, with 95% bands", height=380)

    gen = go.Figure()
    for i, (g, gp) in enumerate(p.groupby("primary_genre")):
        if len(gp) < 20:
            continue
        m = ap[ap.movie_name.isin(gp.movie_name)].groupby("t").smooth.mean()
        gen.add_trace(go.Scatter(x=(m.index + 0.5) / len(m), y=m, name=f"{g} ({len(gp)})", line=dict(color=OKABE[i % 8])))
    gen.update_layout(xaxis_title="story time", yaxis_title="mean smoothed arc")
    style(gen, "Mean arc per genre bin (bins with at least 20 films)", height=420)

    ex = go.Figure()
    for i, m in enumerate(s["examples"]):
        e = a[a.movie_name == m]
        x = (e.t + 0.5) / len(e)
        ex.add_trace(go.Scatter(x=x, y=e.raw, mode="lines", line=dict(color=OKABE[i], width=1), opacity=0.35,
                                legendgroup=m, showlegend=False))
        ex.add_trace(go.Scatter(x=x, y=e.smooth, line=dict(color=OKABE[i], width=2.5), name=m, legendgroup=m))
    ex.update_layout(xaxis_title="story time", yaxis_title="scene score (thin: resampled; thick: smoothed)")
    style(ex, "Six arcs drawn at random from the primary sample", height=440)

    hist = go.Figure()
    for k, colour in (("D1", HIGH), ("D2", LOW)):
        n, e = np.histogram(p[k], bins=50)
        hist.add_trace(go.Bar(x=(e[:-1] + e[1:]) / 2, y=n, width=np.diff(e), name=k, marker_color=colour, opacity=0.7))
    hist.update_layout(barmode="overlay", xaxis_title="descriptor value", yaxis_title="films")
    style(hist, "D1 (volatility) and D2 (ending minus start) across the primary sample", height=360)

    noise = go.Figure(go.Scattergl(x=p.mean_sentences_per_scene, y=p.D1, mode="markers",
                                   marker=dict(color=NEUTRAL, size=5, opacity=0.6)))
    noise.update_xaxes(type="log", title="mean sentences per scene (log)")
    noise.update_yaxes(title="D1, volatility of the smoothed arc")
    style(noise, "Volatility against mean scene length", height=380)

    half = go.Figure(go.Scattergl(x=p.D1_odd, y=p.D1_even, mode="markers", marker=dict(color=HIGH, size=5, opacity=0.6)))
    half.update_layout(xaxis_title="D1 from odd scenes", yaxis_title="D1 from even scenes")
    style(half, "Split-half: volatility measured on two disjoint halves of each script", height=380)
    return {"mean": mean, "genres": gen, "examples": ex, "descriptors": hist, "noise": noise, "split_half": half}


# ---------------------------------------------------------------- reading

def reading(res: dict) -> str:
    s, d = res["summary"], res["desc"]
    p = d[d.in_primary]
    rel = {r["descriptor"]: r for r in s["reliability"]}
    m = res["arcs"][res["arcs"].movie_name.isin(p.movie_name)].groupby("t").smooth.mean()
    e = int(round(s["params"]["edge_share"] * len(m)))

    def strongest(k: str) -> str:
        r = max((x for x in s["proxies"] if x["descriptor"] == k), key=lambda x: abs(x["Spearman"]))
        return f"{r['against']} (ρ = {r['Spearman']:+.2f})"
    return (
        f"Every one of the {s['scripts']:,} scripts has an arc ({s['primary']} in the primary sample). The median "
        f"primary script has {p.scenes.median():.0f} scenes on its timeline, about "
        f"{p.scenes.median() / s['params']['time_points']:.1f} scenes per grid point. "
        f"The average smoothed arc stays close to zero: {m.iloc[:e].mean():+.3f} over the first 20%, "
        f"{m.iloc[-e:].mean():+.3f} over the last 20%, and between {m.min():+.3f} and {m.max():+.3f} overall. "
        f"Individual arcs swing much more than the average, which mostly cancels out (median D1 "
        f"{p.D1.median():.3f}, median D2 {p.D2.median():+.3f}, {100 * (p.D2 < 0).mean():.0f}% of films end lower "
        f"than they start). The scores are {INSTRUMENT_NAMES[s['instrument']]} and NOT YET VALIDATED. "
        f"**Proxies.** The strongest size correlate of D1 is {strongest('D1')}; of D2, {strongest('D2')}; of mean "
        f"level, {strongest('mean_level')}. "
        + ("All are weak (|ρ| < 0.3), so the descriptors are not script size in disguise. The size controls of "
           "Step 7 still apply. " if max(abs(x["Spearman"]) for x in s["proxies"]) < 0.3 else
           "At least one is not weak (|ρ| ≥ 0.3): that descriptor partly measures script size, which the Step 7 "
           "controls must absorb. ") +
        f"**Reliability.** Split-half (odd vs even scenes, Spearman-Brown corrected): D1 "
        f"{rel['D1']['Spearman-Brown']:.2f}, D2 {rel['D2']['Spearman-Brown']:.2f}, mean level "
        f"{rel['mean_level']['Spearman-Brown']:.2f}. A reliability of r means roughly a share 1 − r of the "
        f"between-film variance is scene-level noise: {1 - rel['D1']['Spearman-Brown']:.0%} for D1 and "
        f"{1 - rel['D2']['Spearman-Brown']:.0%} for D2, against {1 - rel['mean_level']['Spearman-Brown']:.0%} "
        f"for the mean level. That noise caps how strongly a descriptor can correlate with anything, profit "
        f"included. Step 6 checks whether the order of scenes carries any structure at all."
    )


SECTION = [
    Md("""## Step 5 · Arcs

**What we do.** For each script we take its scene scores in story order: the "both" layer, one score per
scene on the timeline. Then:
1. We put them on a common clock of **100 points of story time**. Scene *i* of *n* covers
   [*i*/*n*, (*i*+1)/*n*) of the story. Each grid point takes the mean of the scenes it overlaps, weighted by
   the overlap. This works the same for a script with 40 scenes as for one with 400; nothing is dropped or
   interpolated.
2. We **smooth** with a centred moving average over 11 points (about a tenth of the story), cut short at the
   two ends.
3. We summarise each arc with the two pre-registered **primary descriptors**:
   - **D1, volatility**: the standard deviation of the smoothed arc. How much the tone swings.
   - **D2, ending minus start**: the mean of the last 20% minus the mean of the first 20% of the unsmoothed
     arc. Does the story end lighter or darker than it began.

   Three exploratory descriptors come alongside (mean level, position of the low point, number of swings).

**Two checks.**
- **Proxies.** Are the descriptors just script size in disguise? A scene made of two sentences gives a noisy
  mean, so scripts of short scenes could look volatile for no narrative reason.
- **Reliability.** Each script is split into odd and even scenes, and every descriptor is computed on each
  half. If the halves disagree, scene noise dominates.

The instrument is the one the pre-registered fallback rule chose in Step 3 (named in every figure and reading);
the other instruments appear in Step 9. No outcome is used in this step."""),
    Run("s05_arcs"),
    Fig("s05_arcs", "mean", "Mean arc over the primary sample. Grey: smoothed. Blue: resampled, unsmoothed. Bands: "
        "±1.96 standard errors across films."),
    Fig("s05_arcs", "genres", "Mean smoothed arc per genre bin."),
    Table("s05_arcs", "genres", "Median descriptors per genre bin (primary sample)."),
    Fig("s05_arcs", "examples", "Six arcs drawn with the config seed: numbers only, no script text."),
    Table("s05_arcs", "examples", "The six example scripts and their descriptors."),
    Fig("s05_arcs", "descriptors", "Distributions of D1 and D2 in the primary sample."),
    Table("s05_arcs", "distribution", "Descriptor quantiles in the primary sample."),
    Table("s05_arcs", "proxies", "Spearman correlation of each descriptor with script size (primary sample)."),
    Fig("s05_arcs", "noise", "D1 against mean scene length: the scene-noise effect that the scene-length control "
        "addresses (the Spearman correlation is in the proxies table)."),
    Table("s05_arcs", "reliability", "Split-half reliability: odd against even scenes, Spearman correlation across "
          "films and its Spearman-Brown correction to full length."),
    Fig("s05_arcs", "split_half", "D1 computed on odd scenes against D1 on even scenes."),
    Reading("s05_arcs"),
]
