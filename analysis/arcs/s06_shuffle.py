"""Step 6: is there ORDER structure at all? Each script's scenes are shuffled 1,000 times; the real D1 and
D2 are compared with that script's own shuffle distribution. No outcome is used.

D1 one-sided (real > shuffled), D2 two-sided (distance from the null mean), as pre-registered. Empirical
p = (1 + #null as extreme) / (1 + shuffles). Note: shuffling never changes a script's mean level, and this
null compares each script with itself; it does not remove differences between scripts (Step 7 does that)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import binomtest, norm

from analysis.scripts_text.report import HIGH, LOW, NEUTRAL, Fig, Md, Reading, Run, Table, style

from . import s05_arcs
from .common import INSTRUMENT_NAMES, empirical_p, primary_instrument
from .config import cfg, log_run
from .paths import OUTPUTS


def _paths(inst: str):
    return OUTPUTS / f"s06_shuffle_{inst}.csv", OUTPUTS / f"s06_summary_{inst}.json"


def null_for(x: np.ndarray, rng: np.random.Generator, B: int, T: int, w: int, edge: float) -> dict:
    """Real and shuffled D1, D2 for one script (matrix form; same maths as s05_arcs.describe)."""
    R = s05_arcs.resample_matrix(len(x), T)
    S = s05_arcs.smooth_matrix(T, w)
    e = int(round(edge * T))
    X = np.vstack([x, rng.permuted(np.tile(x, (B, 1)), axis=1)])
    raw = X @ R.T
    sm = raw @ S.T
    d1 = sm.std(1)
    d2 = raw[:, -e:].mean(1) - raw[:, :e].mean(1)
    return {"D1": d1[0], "D1_null": d1[1:], "D2": d2[0], "D2_null": d2[1:]}


def run(inst: str | None = None) -> dict:
    inst = inst or primary_instrument()
    c = cfg()
    a, sh = c["arc"], c["scene_shuffle"]
    B = sh["shuffles"]
    series = s05_arcs.scene_series(inst)
    desc = s05_arcs.load(inst)["desc"].set_index("movie_name")
    rng = np.random.default_rng(c["seed"])
    rows, example = [], None
    ex_name = s05_arcs.load(inst)["summary"]["examples"][0]
    for m in sorted(series):
        r = null_for(series[m], rng, B, a["time_points"], a["smoothing_window"], a["edge_share"])
        assert abs(r["D1"] - desc.at[m, "D1"]) < 1e-9 and abs(r["D2"] - desc.at[m, "D2"]) < 1e-9, m
        row = {"movie_name": m, "scenes": len(series[m])}
        for k in ("D1", "D2"):
            nu = r[f"{k}_null"]
            mu, sd = nu.mean(), nu.std(ddof=1)
            row |= {k: r[k], f"{k}_null_mean": mu, f"{k}_null_sd": sd, f"{k}_z": (r[k] - mu) / sd}
        row["D1_p"] = empirical_p((r["D1_null"] >= r["D1"]).sum(), B)
        dev = np.abs(r["D2_null"] - row["D2_null_mean"])
        row["D2_p"] = empirical_p((dev >= abs(r["D2"] - row["D2_null_mean"])).sum(), B)
        rows.append(row)
        if m == ex_name:
            example = {"movie_name": m, "D1": float(r["D1"]), "D1_null": r["D1_null"].round(6).tolist(),
                       "D2": float(r["D2"]), "D2_null": r["D2_null"].round(6).tolist()}
    out = pd.DataFrame(rows).merge(desc[["in_primary", "primary_genre", "possible_transcript"]].reset_index(),
                                   on="movie_name", validate="one_to_one")
    csv_p, sum_p = _paths(inst)
    out.to_csv(csv_p, index=False)

    alpha = sh["alpha_per_script"]
    agg = []
    for sample, q in (("primary 834", out[out.in_primary]), ("all 1,087", out)):
        for k in ("D1", "D2"):
            n_sig = int((q[f"{k}_p"] < alpha).sum())
            agg.append({"sample": sample, "descriptor": k, "scripts": int(len(q)), "beyond null (p < 0.05)": n_sig,
                        "share": n_sig / len(q), "expected share under no order": alpha,
                        "binomial p (share vs 5%)": float(binomtest(n_sig, len(q), alpha, alternative="greater").pvalue),
                        "median z": float(q[f"{k}_z"].median()), "share z > 0": float((q[f"{k}_z"] > 0).mean()),
                        "real / null mean (median)": float((q[k] / q[f"{k}_null_mean"]).median()) if k == "D1" else None})
    summary = {"instrument": inst, "shuffles": B, "alpha_per_script": alpha, "aggregate": agg, "example": example,
               "seed": c["seed"]}
    sum_p.write_text(json.dumps(summary) + "\n")
    log_run(f"s06_shuffle[{inst}]", [csv_p.name, sum_p.name], {"scene_shuffle": sh, "arc": a, "instrument": inst})
    return load(inst)


def load(inst: str | None = None) -> dict:
    inst = inst or primary_instrument()
    csv_p, sum_p = _paths(inst)
    return {"table": pd.read_csv(csv_p, dtype={"movie_name": str}), "summary": json.loads(sum_p.read_text())}


def tables(res: dict) -> dict:
    agg = pd.DataFrame(res["summary"]["aggregate"])
    for k in ("share", "expected share under no order", "share z > 0"):
        agg[k] = (100 * agg[k]).round(1).astype(str) + "%"
    agg["binomial p (share vs 5%)"] = agg["binomial p (share vs 5%)"].map(lambda v: f"{v:.2g}")
    t = res["table"]
    p = t[t.in_primary]
    gen = p.groupby("primary_genre").agg(films=("D1_z", "size"), median_D1_z=("D1_z", "median"),
                                         D1_beyond=("D1_p", lambda v: f"{100 * (v < 0.05).mean():.0f}%"),
                                         median_D2_z=("D2_z", "median")).round(2).reset_index()
    tr = t.groupby("possible_transcript").agg(scripts=("D1_z", "size"), median_D1_z=("D1_z", "median"),
                                              D1_beyond=("D1_p", lambda v: f"{100 * (v < 0.05).mean():.0f}%")).round(2)
    return {"aggregate": agg.round(3), "genres": gen, "transcripts": tr.reset_index()}


def figures(res: dict) -> dict:
    t = res["table"]
    p = t[t.in_primary]
    zs = go.Figure()
    grid = np.linspace(-4, 12, 200)
    for k, colour in (("D1", HIGH), ("D2", LOW)):
        n, e = np.histogram(p[f"{k}_z"].clip(-4, 12), bins=np.arange(-4, 12.25, 0.25))
        zs.add_trace(go.Bar(x=(e[:-1] + e[1:]) / 2, y=n, width=0.25, name=f"{k}: real vs own shuffles",
                            marker_color=colour, opacity=0.6))
    zs.add_trace(go.Scatter(x=grid, y=norm.pdf(grid) * len(p) * 0.25, name="no order structure (N(0,1))",
                            line=dict(color="black", dash="dash")))
    zs.update_layout(barmode="overlay", xaxis_title="z of the real descriptor within its own shuffle distribution "
                     "(clipped to [-4, 12])", yaxis_title="scripts")
    style(zs, "Where each script's real arc falls among 1,000 shuffles of its own scenes", height=420)

    pv = go.Figure()
    for k, colour in (("D1", HIGH), ("D2", LOW)):
        n, e = np.histogram(p[f"{k}_p"], bins=np.linspace(0, 1, 21))
        pv.add_trace(go.Bar(x=(e[:-1] + e[1:]) / 2, y=n, width=0.05, name=k, marker_color=colour, opacity=0.6))
    pv.add_hline(line_width=1.5, y=len(p) / 20, line_dash="dash", annotation_text="flat if scene order carried nothing")
    pv.update_layout(barmode="overlay", xaxis_title="per-script empirical p", yaxis_title="scripts")
    style(pv, "Per-script p-values", height=360)

    ex = res["summary"]["example"]
    exf = go.Figure()
    n, e = np.histogram(ex["D1_null"], bins=40)
    exf.add_trace(go.Bar(x=(e[:-1] + e[1:]) / 2, y=n, width=np.diff(e), marker_color=NEUTRAL, name="1,000 shuffles"))
    exf.add_vline(x=ex["D1"], line_color=HIGH, line_width=3, annotation_text="real order")
    exf.update_layout(xaxis_title="D1 (volatility of the smoothed arc)", yaxis_title="shuffles", showlegend=False)
    style(exf, f"One script against its own shuffles: {ex['movie_name']}", height=340)
    return {"z": zs, "p": pv, "example": exf}


def reading(res: dict) -> str:
    a = {(r["sample"], r["descriptor"]): r for r in res["summary"]["aggregate"]}
    d1, d2 = a[("primary 834", "D1")], a[("primary 834", "D2")]
    order = d1["share"] > 0.05 and d1["binomial p (share vs 5%)"] < 0.001
    txt = (f"We shuffled each script's scenes 1,000 times. The real volatility (D1) lies beyond its own shuffle "
           f"null (one-sided p < 0.05) for {d1['beyond null (p < 0.05)']} of {d1['scripts']} primary scripts. "
           f"That's {100 * d1['share']:.0f}%, against the 5% we'd expect if order carried nothing (binomial p "
           f"{d1['binomial p (share vs 5%)']:.2g}). The median z is {d1['median z']:+.2f}, {100 * d1['share z > 0']:.0f}% "
           f"of scripts have z > 0, and the real D1 is typically {d1['real / null mean (median)']:.2f} times the "
           f"shuffled mean. ")
    txt += ("So the order of scenes does carry structure: neighbouring scenes share tone more than they would in a "
            "random order, and the smoothed arc keeps that. " if order else
            f"So, in {INSTRUMENT_NAMES[res['summary']['instrument']]} scores, the scenes carry a level but little arc "
            "structure. Most real arcs look like shuffled ones. ")
    d2_excess = d2["share"] > 0.05 and d2["binomial p (share vs 5%)"] < 0.001
    txt += (f"For D2 (two-sided), {d2['beyond null (p < 0.05)']} scripts ({100 * d2['share']:.0f}%) lie beyond "
            f"the null (binomial p {d2['binomial p (share vs 5%)']:.2g}). The median z is {d2['median z']:+.2f}, and "
            f"only {100 * d2['share z > 0']:.0f}% of scripts have z > 0. ")
    if d2_excess and d2["median z"] < 0:
        txt += ("So for more scripts than chance allows, endings are darker than a random placement of the same "
                "scenes would make them. For most single films, though, the difference is within the shuffle range. ")
    elif d2_excess:
        txt += ("So endings differ from a random placement of the same scenes more often than chance allows. ")
    else:
        txt += "Endings can't be told apart from a random placement of the same scenes. "
    txt += ("This is a within-script test. It doesn't yet say anything about differences between films, or about "
            f"profit, and it rests on {INSTRUMENT_NAMES[res['summary']['instrument']]} scores that are NOT YET "
            "VALIDATED.")
    return txt


def takeaway(res: dict) -> str:
    a = {(r["sample"], r["descriptor"]): r for r in res["summary"]["aggregate"]}
    d1, d2 = a[("primary 834", "D1")], a[("primary 834", "D2")]
    order = d1["share"] > 0.05 and d1["binomial p (share vs 5%)"] < 0.001
    d2_excess = d2["share"] > 0.05 and d2["binomial p (share vs 5%)"] < 0.001
    first = (f"Scene order carries structure: {100 * d1['share']:.0f}% of scripts swing more than shuffled versions "
             f"of themselves, against 5% by chance. " if order else
             f"Scene order carries little arc structure: {100 * d1['share']:.0f}% of scripts swing more than "
             f"shuffled versions of themselves, against 5% by chance. ")
    second = (f"Endings differ from a random placement more often than chance allows ({100 * d2['share']:.0f}% of "
              f"scripts)." if d2_excess else
              f"Endings can't be told apart from a random placement ({100 * d2['share']:.0f}% of scripts).")
    return first + second


SECTION = [
    Md("""## Step 6 · Compared to what (1): is there order structure at all?

Does the order of a script's scenes matter for its tone, or would the same scenes in random order give the
same arc?

An arc claims that the *order* of scenes matters: the story builds, turns and resolves. If that were false, a
script's scenes put in random order would give arcs just like the real one. So for every script we shuffle the
order of its scenes 1,000 times and recompute D1 and D2 for each shuffle. Those 1,000 values are the script's
own **shuffle null**, which shows what D1 and D2 look like when order carries nothing. Then we ask where the real value
falls in that distribution. We report a z score (how many standard deviations the real value sits from the
shuffled mean) and an empirical p (the share of shuffles at least as extreme):
- **D1, one-sided.** In a real story neighbouring scenes tend to share a mood, so smoothing should keep more
  swing than in a random order, where neighbours cancel out.
- **D2, two-sided.** Ending minus start could go either way.

Then we count how many scripts beat their own null. With no order structure about 5% would, by chance.

### What this null does and doesn't test
It compares each script with itself. The scenes and the mean tone stay the same, and only the order differs. It does
**not** remove differences *between* scripts, such as genre, length or budget. That's Step 7's job. Shuffling
never changes a script's mean level, so the mean level can't be tested here."""),
    Run("s06_shuffle"),
    Table("s06_shuffle", "aggregate", "How many scripts beat their own shuffle null, primary sample first. The binomial "
          "p tests that share against the 5% expected by chance."),
    Fig("s06_shuffle", "z", "z of each primary script's real D1 and D2 within its own 1,000 shuffles. Dashed: what "
        "the z values would look like if scene order carried nothing."),
    Fig("s06_shuffle", "p", "Per-script p-values. Without order structure they'd be flat (dashed line)."),
    Fig("s06_shuffle", "example", "The first of the Step 5 example scripts: its 1,000 shuffled D1 values and the "
        "real one."),
    Table("s06_shuffle", "genres", "Order structure by genre bin (primary sample)."),
    Table("s06_shuffle", "transcripts", "Scripts flagged as possible transcripts against the rest (all scripts). A "
          "transcript has little description, so its arc rests on dialogue alone."),
    Reading("s06_shuffle"),
]
