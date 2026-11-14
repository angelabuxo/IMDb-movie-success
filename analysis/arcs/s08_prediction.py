"""Step 8: second method, incremental prediction. Does adding the arc (D1, D2) to a baseline model
(genre x decade cells, log budget, scene length) improve the out-of-fold R2 of log(gross/budget)?

As pre-registered (config `prediction`): 5 folds x 10 repeats stratified by cell, OLS, out-of-fold R2 per
repeat, repeat interval = min-max of the 10 repeat values, null = outcome permuted 200 times.
Decision (from the brief, logged before any result): "adds predictive value" only if the repeat interval of
the gain excludes 0 AND the permutation p < 0.05. Exploratory second null: D1 and D2 permuted across films
(keeps the baseline's own signal intact).

Refuses to run until the instrument decision exists (same guard as Step 7)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import RepeatedStratifiedKFold

from analysis.scripts_text.report import HIGH, LOW, NEUTRAL, Fig, Md, Reading, Run, Table, style

from . import s07_primary
from .common import design, empirical_p
from .config import cfg, config_hash, log_run
from .paths import OUTPUTS

RESULTS = OUTPUTS / "s08_prediction.json"
PLUS = ["D1", "D2"]


def _matrices(df: pd.DataFrame, covariates: list[str]) -> tuple[np.ndarray, np.ndarray]:
    base = design(df["cell"], df[covariates])
    return base, np.column_stack([base, df[PLUS].to_numpy(float)])


def oof_r2(X: np.ndarray, y: np.ndarray, splits: list[list[tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
    """Out-of-fold R2 of OLS, one value per repeat (predictions pooled over that repeat's folds)."""
    out = np.empty(len(splits))
    sst = ((y - y.mean()) ** 2).sum()
    for r, folds in enumerate(splits):
        pred = np.empty_like(y)
        for tr, te in folds:
            beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            pred[te] = X[te] @ beta
        out[r] = 1 - ((y - pred) ** 2).sum() / sst
    return out


def make_splits(cells: pd.Series, folds: int, repeats: int, seed: int) -> list:
    rs = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    allf = list(rs.split(np.zeros(len(cells)), cells.to_numpy()))
    return [allf[r * folds:(r + 1) * folds] for r in range(repeats)]


def evaluate(df: pd.DataFrame, outcome: str, covariates: list[str], folds: int, repeats: int, n_perm: int,
             seed: int) -> dict:
    rng = np.random.default_rng(seed)
    splits = make_splits(df["cell"], folds, repeats, seed)
    Xb, Xp = _matrices(df, covariates)
    y = df[outcome].to_numpy(float)
    rb, rp = oof_r2(Xb, y, splits), oof_r2(Xp, y, splits)
    gain = rp - rb
    null_y, null_d = np.empty(n_perm), np.empty(n_perm)
    k = Xb.shape[1]
    for b in range(n_perm):
        yp = rng.permutation(y)                                   # pre-registered: outcome permuted
        null_y[b] = (oof_r2(Xp, yp, splits) - oof_r2(Xb, yp, splits)).mean()
        Xd = Xp.copy()
        Xd[:, k:] = Xp[rng.permutation(len(y))][:, k:]              # exploratory: D1, D2 permuted together
        null_d[b] = (oof_r2(Xd, y, splits) - rb).mean()
    g = float(gain.mean())
    res = {"n": int(len(df)), "baseline_r2": float(rb.mean()), "baseline_r2_range": [float(rb.min()), float(rb.max())],
           "plus_r2": float(rp.mean()), "plus_r2_range": [float(rp.min()), float(rp.max())],
           "gain": g, "gain_range": [float(gain.min()), float(gain.max())], "gain_per_repeat": gain.round(6).tolist(),
           "p_outcome_permuted": empirical_p((null_y >= g).sum(), n_perm),
           "p_descriptors_permuted": empirical_p((null_d >= g).sum(), n_perm),
           "null_outcome_permuted": null_y.round(6).tolist(), "null_descriptors_permuted": null_d.round(6).tolist()}
    res["adds_predictive_value"] = bool(gain.min() > 0 and res["p_outcome_permuted"] < 0.05)
    return res


def run() -> dict:
    if not s07_primary.DECISION.exists():
        raise RuntimeError("Step 8 needs outputs/s03_instrument_decision.json (fallback rule not applied yet).")
    c = cfg()
    pr = c["prediction"]
    inst = json.loads(s07_primary.DECISION.read_text())["primary_instrument"]
    df = s07_primary.sample(inst)
    res = evaluate(df, c["primary"]["outcome"], c["controls"]["covariates"], pr["folds"], pr["repeats"],
                   pr["permutations"], c["seed"])
    res |= {"instrument": inst, "config": config_hash(), "folds": pr["folds"], "repeats": pr["repeats"],
            "permutations": pr["permutations"]}
    RESULTS.write_text(json.dumps(res) + "\n")
    log_run("s08_prediction", [RESULTS.name], {"prediction": pr, "instrument": inst, "seed": c["seed"]})
    return load()


def load() -> dict:
    return json.loads(RESULTS.read_text())


def self_test(seed: int = 11) -> pd.DataFrame:
    """Synthetic films only: the arc adds nothing (gain ~ 0, not 'adds') vs. the arc adds a real signal."""
    rng = np.random.default_rng(seed)
    n = 834
    cell = pd.Series(rng.choice([f"g{i} | d{j}" for i in range(7) for j in range(5)], n))
    eff = cell.map({c: rng.normal(0, 0.5) for c in cell.unique()}).to_numpy()
    df = pd.DataFrame({"cell": cell, "log_budget": rng.normal(17, 1, n), "spsc": rng.lognormal(3, .3, n),
                       "D1": rng.normal(0, 1, n), "D2": rng.normal(0, 1, n)})
    rows = []
    for w in (0.0, 0.3):
        df["y"] = eff - 0.4 * (df.log_budget - 17) + w * df.D1 + rng.normal(0, 1, n)
        r = evaluate(df, "y", ["log_budget", "spsc"], 5, 10, 100, seed)
        rows.append({"planted weight on D1": w, "baseline R2": r["baseline_r2"], "plus R2": r["plus_r2"], "gain": r["gain"],
                     "gain range": r["gain_range"], "p (outcome permuted)": r["p_outcome_permuted"],
                     "p (descriptors permuted)": r["p_descriptors_permuted"], "adds": r["adds_predictive_value"]})
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------- report

def tables(res: dict) -> dict:
    f = lambda v, rg: f"{v:.4f} [{rg[0]:.4f}, {rg[1]:.4f}]"
    t = pd.DataFrame([
        ("baseline: cells + log budget + scene length", f(res["baseline_r2"], res["baseline_r2_range"])),
        ("baseline + D1 + D2", f(res["plus_r2"], res["plus_r2_range"])),
        ("gain from the arc", f(res["gain"], res["gain_range"])),
        ("permutation p, outcome permuted (pre-registered)", round(res["p_outcome_permuted"], 3)),
        ("permutation p, D1 and D2 permuted (exploratory)", round(res["p_descriptors_permuted"], 3)),
        ("decision", "adds predictive value" if res["adds_predictive_value"] else
         "does not add predictive value beyond the baseline")],
        columns=[f"out-of-fold R² of log(gross/budget), mean over {res['repeats']} repeats [min, max]", "value"])
    return {"results": t}


def figures(res: dict) -> dict:
    f = go.Figure()
    for k, colour, name in (("null_outcome_permuted", NEUTRAL, "outcome permuted (pre-registered)"),
                            ("null_descriptors_permuted", LOW, "D1, D2 permuted (exploratory)")):
        n, e = np.histogram(res[k], bins=30)
        f.add_trace(go.Bar(x=(e[:-1] + e[1:]) / 2, y=n, width=np.diff(e), name=name, marker_color=colour, opacity=0.6))
    f.add_vline(x=res["gain"], line_color=HIGH, line_width=3, annotation_text="real gain")
    f.add_vrect(x0=res["gain_range"][0], x1=res["gain_range"][1], fillcolor=HIGH, opacity=0.12, line_width=0)
    f.update_layout(barmode="overlay", xaxis_title="gain in out-of-fold R² from adding D1 and D2",
                    yaxis_title="permutations")
    style(f, "The arc's gain in prediction against two permutation nulls (shaded: range over repeats)", height=380)
    return {"gain": f}


def reading(res: dict) -> str:
    g, rg = res["gain"], res["gain_range"]
    txt = (f"Out of sample, the baseline explains R² = {res['baseline_r2']:.3f} of log(gross/budget) "
           f"(range over repeats {res['baseline_r2_range'][0]:.3f} to {res['baseline_r2_range'][1]:.3f}). Adding D1 "
           f"and D2 gives {res['plus_r2']:.3f}, a gain of {g:+.4f} (range {rg[0]:+.4f} to {rg[1]:+.4f}; permutation "
           f"p = {res['p_outcome_permuted']:.3f} with the outcome permuted, {res['p_descriptors_permuted']:.3f} "
           f"with the descriptors permuted). ")
    if res["adds_predictive_value"]:
        txt += ("The gain is positive in every repeat and beyond the permutation null: the arc adds a little "
                "predictive information, of the size shown. That is prediction, not a cause.")
    else:
        txt += ("The arc does not add predictive value beyond the baseline: the gain is "
                + ("not positive in every repeat" if rg[0] <= 0 else "within the permutation null") +
                ". We do not write that it \"beats\" the baseline.")
    return txt


SECTION = [
    Md("""## Step 8 · Second method: does the arc help predict profit?

Step 7 asked whether the arc is *associated* with profit once the controls are held fixed. This step asks a
more practical question: if we already know a film's genre, decade, budget and scene length, does also
knowing its arc help **predict** the profit of films the model has not seen?

**How.**
1. We split the films into 5 folds, stratified by genre × decade cell. We fit an ordinary least-squares model
   on 4 folds and predict the 5th, until every film has an out-of-fold prediction.
2. We do that 10 times with different splits.
3. We compare two models: the **baseline** (cells, log budget, scene length) and **baseline + D1 + D2**. The
   gain is the difference in out-of-fold R².
4. Compared to what:
   - the range of the gain over the 10 repeats;
   - a permutation null of 200 runs with the outcome shuffled (pre-registered);
   - an exploratory null that shuffles only D1 and D2 across films, keeping the baseline's own signal.

We say the arc "adds predictive value" only if the gain is positive in every repeat **and** beyond the
permutation null. A gap that sits inside the repeat range is reported as no gain."""),
    Run("s08_prediction"),
    Table("s08_prediction", "results", "Out-of-fold R² for both models, the gain, and the permutation p-values."),
    Fig("s08_prediction", "gain", "The real gain (blue line; blue band = range over repeats) against the two "
        "permutation nulls."),
    Reading("s08_prediction"),
]
