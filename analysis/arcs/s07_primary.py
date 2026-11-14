"""Step 7: PRIMARY ANALYSIS. Two descriptors (D1, D2) against log(gross/budget) in the primary sample.

Frisch-Waugh: outcome and descriptor are each regressed on the controls (genre x decade cells, log budget,
mean sentences per scene); Spearman rho of the residuals. Null: residualised outcome permuted WITHIN cells.
Bootstrap: films resampled, residualisation refitted. Everything is fixed in config.yaml `primary`.

This step refuses to run until Step 3 has written outputs/s03_instrument_decision.json (fallback rule) and
Step 5 has built the arcs with the instrument it names. It is run once; the result is never tuned."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import spearmanr

from analysis.scripts_text.report import HIGH, LOW, NEUTRAL, Fig, Md, Reading, Run, Table, style

from . import s02_units, s05_arcs
from .common import design, empirical_p, pooled_cells, r_min, residualise
from .config import cfg, config_hash, log_run
from .paths import OUTPUTS

DECISION = OUTPUTS / "s03_instrument_decision.json"
RESULTS = OUTPUTS / "s07_primary.json"
DESCRIPTORS = {"D1": "D1 volatility", "D2": "D2 ending minus start"}


def _rank(v: np.ndarray) -> np.ndarray:
    return pd.Series(v).rank().to_numpy()


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def frisch_waugh(df: pd.DataFrame, desc: str, outcome: str, covariates: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = design(df["cell"], df[covariates])
    return residualise(df[desc].to_numpy(float), X), residualise(df[outcome].to_numpy(float), X)


def association(df: pd.DataFrame, desc: str, outcome: str, covariates: list[str], n_perm: int, n_boot: int,
                rng: np.random.Generator, keep_null: bool = False) -> dict:
    """The pre-registered statistic, its within-cell permutation null and a refitting bootstrap. Reused by
    Steps 9 and 10 with their own (smaller) numbers of permutations and resamples. `df` needs a `cell` column."""
    rd, ry = frisch_waugh(df, desc, outcome, covariates)
    rho = _rho(rd, ry)
    rank_d = _rank(rd)
    groups = [np.flatnonzero(df["cell"].to_numpy() == c) for c in df["cell"].unique()]
    null = np.empty(n_perm)
    for b in range(n_perm):
        yp = ry.copy()
        for g in groups:
            yp[g] = ry[rng.permutation(g)]
        null[b] = np.corrcoef(rank_d, _rank(yp))[0, 1]
    boot = np.empty(n_boot)
    n = len(df)
    for b in range(n_boot):
        s = df.iloc[rng.integers(0, n, n)]
        bd, by = frisch_waugh(s, desc, outcome, covariates)
        boot[b] = _rho(bd, by)
    out = {"rho": rho, "n": int(n), "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
           "z": float((rho - null.mean()) / null.std(ddof=1)),
           "p": empirical_p((np.abs(null) >= abs(rho)).sum(), n_perm),
           "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
           "spearman_check": float(spearmanr(rd, ry).statistic)}
    if keep_null:
        out["null"] = null.round(5).tolist()
        out["residuals"] = {"descriptor": rd.round(6).tolist(), "outcome": ry.round(6).tolist()}
    return out


def sample(inst: str, which: str = "primary") -> pd.DataFrame:
    """Analysis frame: films x (descriptors, outcome, controls, cell). Cells pooled on THIS sample."""
    tab = s02_units.load()["table"]
    d = s05_arcs.load(inst)["desc"][["movie_name", "D1", "D2", "mean_level", "scenes"]]
    df = tab.merge(d, on="movie_name", how="inner", validate="one_to_one")
    if which == "primary":
        df = df[df.in_primary]
    df = df.reset_index(drop=True)
    df["cell"] = pooled_cells(df.primary_genre, df.decade_bin)
    return df


def run() -> dict:
    if not DECISION.exists():
        raise RuntimeError("Step 7 needs outputs/s03_instrument_decision.json: the fallback rule has not been "
                           "applied yet (contextual job not ingested). Not running.")
    c = cfg()
    pc, ctl = c["primary"], c["controls"]
    decision = json.loads(DECISION.read_text())
    inst = decision["primary_instrument"]
    df = sample(inst)
    if len(df) != c["sample"]["expected"]["not_likely_franchise_arc_ready"]:
        raise RuntimeError(f"primary sample has {len(df)} films, expected 834")
    rng = np.random.default_rng(c["seed"])
    res = {"instrument": inst, "decision_file": decision, "config": config_hash(), "n": int(len(df)),
           "outcome": pc["outcome"], "covariates": ctl["covariates"], "cells": int(df.cell.nunique()),
           "pooled_films": int(df.cell.str.contains("pooled").sum()),
           "permutations": pc["permutations"], "bootstrap": pc["bootstrap"], "alpha": pc["alpha_bonferroni"],
           "r_min": {"alpha_0.05": r_min(len(df), pc["alpha_family"], pc["power"]),
                     "alpha_0.025": r_min(len(df), pc["alpha_bonferroni"], pc["power"])},
           "results": {}}
    for k in DESCRIPTORS:
        r = association(df, k, pc["outcome"], ctl["covariates"], pc["permutations"], pc["bootstrap"], rng, keep_null=True)
        r["significant"] = bool(r["p"] < pc["alpha_bonferroni"])
        res["results"][k] = r
    RESULTS.write_text(json.dumps(res) + "\n")
    log_run("s07_primary", [RESULTS.name], {"primary": pc, "controls": ctl, "instrument": inst, "seed": c["seed"]})
    return load()


def load() -> dict:
    return json.loads(RESULTS.read_text())


# ---------------------------------------------------------------- self-test (synthetic data only)

def self_test(seed: int = 7) -> pd.DataFrame:
    """Checks the machinery on synthetic films, never on the real outcome: a planted association must be
    recovered with a small p and a CI covering it; a pure-noise outcome must give a p that is not small."""
    rng = np.random.default_rng(seed)
    n = 834
    cell = pd.Series(rng.choice([f"g{i} | d{j}" for i in range(7) for j in range(5)], n))
    eff = cell.map({c: rng.normal(0, 1) for c in cell.unique()}).to_numpy()
    budget = rng.normal(17, 1, n)
    spsc = rng.lognormal(3, 0.3, n)
    d = 0.5 * eff + 0.3 * budget + rng.normal(0, 1, n)          # descriptor confounded with the controls
    rows = []
    for planted in (0.0, 0.25):
        y = eff + 0.5 * budget + planted * (d - 0.5 * eff - 0.3 * budget) / 1.0 * 2 + rng.normal(0, 1, n)
        df = pd.DataFrame({"cell": cell, "D": d, "y": y, "log_budget": budget, "spsc": spsc})
        r = association(df, "D", "y", ["log_budget", "spsc"], 500, 300, rng)
        raw = float(spearmanr(df.D, df.y).statistic)
        rows.append({"planted partial effect": planted, "raw Spearman (confounded)": raw, **{k: r[k] for k in
                     ("rho", "spearman_check", "ci_low", "ci_high", "z", "p")}})
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------- report

def tables(res: dict) -> dict:
    rows = []
    for k, label in DESCRIPTORS.items():
        r = res["results"][k]
        rows.append({"descriptor": label, "N": r["n"], "Spearman ρ (residualised)": round(r["rho"], 3),
                     "bootstrap 95% CI": f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
                     "null mean ± sd": f"{r['null_mean']:+.3f} ± {r['null_sd']:.3f}", "z": round(r["z"], 2),
                     "empirical p": round(r["p"], 4), "Bonferroni α": res["alpha"],
                     "decision": "association" if r["significant"] else "no evidence of an association"})
    setup = pd.DataFrame([("instrument (fallback rule)", res["instrument"]), ("films", res["n"]),
                          ("outcome", "log(worldwide gross / budget)"),
                          ("controls", "genre × decade cells + " + ", ".join(res["covariates"])),
                          ("cells (pooled below 5 films)", res["cells"]),
                          ("permutations within cells / bootstrap resamples", f"{res['permutations']} / {res['bootstrap']}"),
                          ("minimum detectable |ρ| at 80% power, α = 0.05", round(res["r_min"]["alpha_0.05"], 3)),
                          ("… at α = 0.025 (Bonferroni)", round(res["r_min"]["alpha_0.025"], 3)),
                          ("config hash", res["config"])], columns=["item", "value"])
    return {"results": pd.DataFrame(rows), "setup": setup}


def figures(res: dict) -> dict:
    out = {}
    for k, label in DESCRIPTORS.items():
        r = res["results"][k]
        n, e = np.histogram(r["null"], bins=40)
        f = go.Figure(go.Bar(x=(e[:-1] + e[1:]) / 2, y=n, width=np.diff(e), marker_color=NEUTRAL,
                             name="1,000 within-cell permutations"))
        f.add_vline(x=r["rho"], line_color=HIGH, line_width=3, annotation_text=f"real ρ = {r['rho']:+.3f}")
        for s in (-1, 1):   # both lines labelled: an annotation with no text renders as plotly's "new text"
            f.add_vline(x=s * res["r_min"]["alpha_0.025"], line_dash="dot", line_color=LOW, line_width=2,
                        annotation_text="−r_min" if s < 0 else "+r_min",
                        annotation_position="top left" if s < 0 else "top right")
        f.update_layout(xaxis_title=f"Spearman ρ, residualised {label} vs residualised outcome",
                        yaxis_title="permutations", showlegend=False)
        out[f"null_{k}"] = style(f, f"{label}: the real correlation against its permutation null", height=360)
        rr = r["residuals"]
        sc = go.Figure(go.Scattergl(x=_rank(np.array(rr["descriptor"])), y=_rank(np.array(rr["outcome"])),
                                    mode="markers", marker=dict(color=NEUTRAL, size=5, opacity=0.5)))
        sc.update_layout(xaxis_title=f"rank of residualised {label}", yaxis_title="rank of residualised log(gross/budget)")
        out[f"ranks_{k}"] = style(sc, f"{label}: ranks of the two residuals (what ρ summarises)", height=380)
    return out


def reading(res: dict) -> str:
    rm = res["r_min"]["alpha_0.025"]
    parts = []
    for k, label in DESCRIPTORS.items():
        r = res["results"][k]
        if r["significant"]:
            parts.append(f"{label}: ρ = {r['rho']:+.3f} (95% CI {r['ci_low']:+.3f} to {r['ci_high']:+.3f}), "
                         f"p = {r['p']:.3f} < {res['alpha']}: an association, of this size, once genre, decade, "
                         f"budget and scene length are held fixed. It is an association, not an effect of the "
                         f"script on profit.")
        else:
            near = r["ci_low"] > 0 or r["ci_high"] < 0
            parts.append(f"{label}: ρ = {r['rho']:+.3f} (95% CI {r['ci_low']:+.3f} to {r['ci_high']:+.3f}), "
                         f"p = {r['p']:.3f}, z = {r['z']:+.2f}: no evidence of an association at the pre-registered "
                         f"α = {res['alpha']}. "
                         + (f"It is a near miss: the bootstrap interval excludes zero and p is below 0.05 but above "
                            f"{res['alpha']}. Under the rule fixed before the analysis this is not a finding; at most it "
                            f"suggests an association no larger than about {max(abs(r['ci_low']), abs(r['ci_high'])):.2f}, "
                            f"to be checked for stability in Steps 9 and 10. "
                            if near else
                            f"The interval runs from {r['ci_low']:+.2f} to {r['ci_high']:+.2f}, so associations "
                            f"larger than that in either direction are unlikely; with N = {r['n']}, |ρ| ≥ {rm:.3f} "
                            f"would be detected with 80% power. "))
    return (f"Primary analysis on {res['n']} films, instrument {res['instrument']}. " + " ".join(parts) +
            " Two caveats carry over: the sentiment scores are NOT YET VALIDATED against human readers (Step 4), "
            "and the split-half reliability of D1 and D2 (Step 5) attenuates any true association towards zero.")


SECTION = [
    Md("""## Step 7 · PRIMARY ANALYSIS. Compared to what (2): is the arc associated with profit?

**The one pre-registered test.** In the primary sample (not likely franchise, arc-ready, N = 834), we relate
each of the two primary descriptors (D1 volatility, D2 ending minus start) to the outcome,
log(worldwide gross / budget).

**Controls on both sides (Frisch-Waugh).** The outcome and the descriptor are each regressed on:
- genre × decade cells (cells with fewer than 5 films pooled);
- log budget;
- mean sentences per scene.

We then take the Spearman correlation of the two residuals. That is the association left once the controls
are held fixed.

**Compared to what.**
- We permute the residualised outcome 1,000 times **within** genre × decade cells, which keeps the cell
  structure, and recompute ρ each time.
- The two-sided empirical p is compared with the Bonferroni α = 0.025, because two descriptors are tested.
- A bootstrap of 2,000 film resamples gives the 95% CI; the residualisation is refitted in every resample.
- We also report the minimum detectable correlation. A null result then reads "we can rule out associations
  larger than about 0.1", not "there is none".

Reviews are not used here: they come after release. Everything above was fixed in `config.yaml` before any
association was computed, and this step runs once."""),
    Run("s07_primary"),
    Table("s07_primary", "setup", "Sample, controls, numbers of permutations and resamples, and power."),
    Table("s07_primary", "results", "The primary result: one row per descriptor."),
    Fig("s07_primary", "null_D1", "D1: the 1,000 within-cell permutation correlations, the real one (blue) and ±r_min "
        "(dotted)."),
    Fig("s07_primary", "null_D2", "D2: the same for ending minus start."),
    Fig("s07_primary", "ranks_D1", "D1: rank of the residualised descriptor against rank of the residualised outcome, "
        "one point per film (no names)."),
    Fig("s07_primary", "ranks_D2", "D2: the same."),
    Reading("s07_primary"),
]
