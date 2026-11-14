"""Step 9: stability and robustness (EXPLORATORY). The primary statistic of Step 7, re-run across variants:
instruments, layers, smoothing windows, grid sizes, samples (franchise cuts, arc-ready filters), outcome,
genre rule, and permutation seeds. One table of rho + CI per variant and a forest plot.

Each variant: same Frisch-Waugh Spearman as Step 7 (s07_primary.association), 500 within-cell permutations,
100 bootstrap resamples (percentile CI and sign stability), as in config `robustness`. Nothing here can
change the primary result; it shows how much the answer depends on the choices made."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.scripts_text.report import HIGH, LOW, NEUTRAL, Fig, Md, Reading, Run, Table, style

from . import s02_units, s03_instruments, s05_arcs, s07_primary
from .common import pooled_cells, r_min
from .config import cfg, config_hash, log_run
from .paths import OUTPUTS

RESULTS = OUTPUTS / "s09_robustness.csv"
META = OUTPUTS / "s09_meta.json"


# ---------------------------------------------------------------- descriptors under a variant

def descriptors(inst: str = "vader", layer: str = "both", T: int | None = None, w: int | None = None,
                context: bool = False, min_scenes: int = 20) -> pd.DataFrame:
    """D1, D2 per script for one arc configuration. layer 'dialogue'/'action' uses only scenes that have
    sentences in that layer; scripts with fewer than `min_scenes` such scenes are dropped (counted).
    context=True uses the scene-in-context scores from the HPC job (Step 3 amendment (a))."""
    if context:
        sc = pd.read_csv(OUTPUTS / f"s03_scene_context_{inst}.csv.gz", dtype={"movie_name": str})
        col = f"{inst}_scene"
    else:
        sc = s03_instruments.load_scores(inst, "scene")
        col = f"{inst}_{layer}"
    sc = sc.dropna(subset=[col]).sort_values(["movie_name", "scene"])
    rows = []
    for m, g in sc.groupby("movie_name", sort=False):
        x = g[col].to_numpy(float)
        if layer != "both" and len(x) < min_scenes:
            continue
        d = s05_arcs.describe(x, T, w)
        rows.append({"movie_name": m, "D1": float(d["D1"][0]), "D2": float(d["D2"][0])})
    return pd.DataFrame(rows)


def frame(desc: pd.DataFrame, mask: str, genre_col: str = "primary_genre") -> pd.DataFrame:
    """Join descriptors to the film table, apply a sample mask, pool cells on that sample."""
    tab = s02_units.load()["table"]
    masks = {"primary": tab.in_primary,
             "all scripts (franchise included)": tab.arc_ready,
             "exclude 'follows' only": tab.arc_ready & ~tab.sequel_follows,
             "exclude named franchise only": tab.arc_ready & ~tab.named_franchise,
             "strict arc-ready filter": ~tab.likely_franchise & tab.arc_ready_strict,
             "loose arc-ready filter": ~tab.likely_franchise & tab.arc_ready_loose,
             "primary without possible transcripts": tab.in_primary & ~tab.possible_transcript}
    df = tab[masks[mask]].merge(desc, on="movie_name", how="inner", validate="one_to_one").reset_index(drop=True)
    df["cell"] = pooled_cells(df[genre_col], df.decade_bin)
    return df


def variants(primary_inst: str) -> list[dict]:
    """(group, label, descriptor builder kwargs, sample mask, extra) for every variant, primary first."""
    a = cfg()["arc"]
    r = cfg()["robustness"]
    out = [{"group": "primary", "label": f"primary ({primary_inst}, both layers, window {a['smoothing_window']}, "
            f"{a['time_points']} points)", "desc": {"inst": primary_inst}, "mask": "primary"}]
    for inst in ("vader", "sst2", "roberta3"):
        if inst != primary_inst and (OUTPUTS / f"s03_scene_{inst}.csv.gz").exists():
            out.append({"group": "instrument", "label": f"instrument: {inst}", "desc": {"inst": inst}, "mask": "primary"})
    for inst in ("sst2", "roberta3"):
        if (OUTPUTS / f"s03_scene_context_{inst}.csv.gz").exists():
            out.append({"group": "instrument", "label": f"instrument: {inst}, scene in context",
                        "desc": {"inst": inst, "context": True}, "mask": "primary"})
    for layer in ("dialogue", "action"):
        out.append({"group": "layer", "label": f"layer: {layer} only", "desc": {"inst": primary_inst, "layer": layer},
                    "mask": "primary"})
    for w in r["smoothing_windows"]:
        if w != a["smoothing_window"]:
            out.append({"group": "arc", "label": f"smoothing window {w}", "desc": {"inst": primary_inst, "w": w},
                        "mask": "primary"})
    for T in r["time_points"]:
        if T != a["time_points"]:
            out.append({"group": "arc", "label": f"{T} time points", "desc": {"inst": primary_inst, "T": T},
                        "mask": "primary"})
    for m in ("all scripts (franchise included)", "exclude 'follows' only", "exclude named franchise only",
              "strict arc-ready filter", "loose arc-ready filter", "primary without possible transcripts"):
        out.append({"group": "sample", "label": f"sample: {m}", "desc": {"inst": primary_inst}, "mask": m})
    for o in cfg()["outcome"]["sensitivity"]:
        out.append({"group": "outcome", "label": f"outcome: {o}", "desc": {"inst": primary_inst}, "mask": "primary",
                    "outcome": o})
    out.append({"group": "genre rule (exploratory)", "label": "genre: first listed instead of priority rule",
                "desc": {"inst": primary_inst}, "mask": "primary", "genre_col": "primary_genre_first_listed"})
    for s in cfg()["stability_seeds"][1:]:
        out.append({"group": "seed", "label": f"permutation / bootstrap seed {s}", "desc": {"inst": primary_inst},
                    "mask": "primary", "seed": s})
    return out


def run_variants(primary_inst: str, outcome_override: str | None = None, table_hook=None) -> pd.DataFrame:
    """All variants. `outcome_override`/`table_hook` exist only for the synthetic dry run."""
    c = cfg()
    rb = c["robustness"]
    cov = c["controls"]["covariates"]
    cache, rows = {}, []
    for v in variants(primary_inst):
        key = json.dumps(v["desc"], sort_keys=True)
        if key not in cache:
            cache[key] = descriptors(**v["desc"])
        df = frame(cache[key], v["mask"], v.get("genre_col", "primary_genre"))
        if table_hook is not None:
            df = table_hook(df)
        outcome = outcome_override or v.get("outcome", c["primary"]["outcome"])
        df = df.dropna(subset=[outcome] + cov)
        rng = np.random.default_rng(v.get("seed", c["seed"]))
        for k in ("D1", "D2"):
            r = s07_primary.association(df, k, outcome, cov, rb["permutations_per_variant"], rb["bootstrap_resamples"],
                                        rng, keep_null=False)
            rows.append({"group": v["group"], "variant": v["label"], "descriptor": k, "N": r["n"],
                         "cells": int(df.cell.nunique()), "rho": r["rho"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                         "z": r["z"], "p": r["p"], "r_min_0.05": r_min(r["n"], 0.05)})
    return pd.DataFrame(rows)


def run() -> dict:
    if not s07_primary.DECISION.exists():
        raise RuntimeError("Step 9 needs outputs/s03_instrument_decision.json (fallback rule not applied yet).")
    c = cfg()
    inst = json.loads(s07_primary.DECISION.read_text())["primary_instrument"]
    t = run_variants(inst)
    t.to_csv(RESULTS, index=False)
    META.write_text(json.dumps({"instrument": inst, "config": config_hash(), "robustness": c["robustness"]}) + "\n")
    log_run("s09_robustness", [RESULTS.name, META.name], {"robustness": c["robustness"], "instrument": inst,
                                                          "seed": c["seed"]})
    return load()


def load() -> dict:
    return {"table": pd.read_csv(RESULTS), "meta": json.loads(META.read_text())}


# ---------------------------------------------------------------- report

def tables(res: dict) -> dict:
    t = res["table"].copy()
    t["95% CI"] = [f"[{a:+.3f}, {b:+.3f}]" for a, b in zip(t.ci_low, t.ci_high)]
    t["p < 0.05"] = np.where(t.p < 0.05, "yes", "")
    out = t[["group", "variant", "descriptor", "N", "cells", "rho", "95% CI", "z", "p", "p < 0.05", "r_min_0.05"]]
    return {"variants": out.round(3)}


def figures(res: dict) -> dict:
    t = res["table"]
    labels = list(dict.fromkeys(t.variant))[::-1]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("D1 volatility", "D2 ending minus start"),
                        horizontal_spacing=0.04)
    for j, k in enumerate(("D1", "D2"), start=1):
        q = t[t.descriptor == k].set_index("variant").loc[[l for l in labels if l in set(t.variant)]]
        colour = [HIGH if g == "primary" else (LOW if p < 0.05 else NEUTRAL) for g, p in zip(q.group, q.p)]
        fig.add_trace(go.Scatter(x=q.rho, y=q.index, mode="markers", marker=dict(color=colour, size=9),
                                 error_x=dict(type="data", symmetric=False, array=q.ci_high - q.rho,
                                              arrayminus=q.rho - q.ci_low, color="#888"), showlegend=False), 1, j)
        fig.add_vline(x=0, line_color="black", line_width=1, row=1, col=j)
        rm = float(q.loc[q.group == "primary", "r_min_0.05"].iloc[0])
        for s in (-1, 1):
            fig.add_vline(x=s * rm, line_dash="dot", line_color=LOW, row=1, col=j)
    fig.update_xaxes(title_text="Spearman ρ (residualised), 95% bootstrap CI")
    style(fig, "Robustness: the primary statistic across variants (dotted: ±r_min of the primary sample)",
          height=max(500, 26 * len(labels)))
    return {"forest": fig}


def reading(res: dict) -> str:
    t = res["table"]
    out = []
    for k, label in (("D1", "volatility (D1)"), ("D2", "ending minus start (D2)")):
        q = t[t.descriptor == k]
        p0 = q[q.group == "primary"].iloc[0]
        rest = q[q.group != "primary"]
        same = int((np.sign(rest.rho) == np.sign(p0.rho)).sum())
        sig = rest[rest.p < 0.05]
        out.append(f"For {label}, the primary ρ is {p0.rho:+.3f}. Across {len(rest)} variants ρ ranges from "
                   f"{rest.rho.min():+.3f} to {rest.rho.max():+.3f}; {same} of {len(rest)} have the same sign, and "
                   f"{len(sig)} {'reaches' if len(sig) == 1 else 'reach'} p < 0.05"
                   + (f" ({'; '.join(sig.variant)})" if len(sig) else "") + ".")
    return (" ".join(out) + " These rows are exploratory: about 1 in 20 would pass p < 0.05 by chance, so a lone "
            "significant variant is not a finding. What matters is whether the sign and size hold across "
            "choices. The primary decision stays the one in Step 7.")


SECTION = [
    Md("""## Step 9 · Stability and robustness (exploratory)

One run is not *the* answer. We repeat the Step 7 statistic, with the same controls and the same
within-cell null, while changing one choice at a time:
- **Instrument:** VADER, DistilBERT SST-2, the three-class RoBERTa, and both contextual models reading whole
  scenes in context. Each runs only if it is available.
- **Layer:** dialogue only, and action lines only.
- **Arc:** smoothing windows 5 and 21, and grids of 50 and 200 points.
- **Sample:** all scripts with franchise films included, each single-source franchise cut, the strict and
  loose arc-ready filters, and the primary sample without possible transcripts.
- **Outcome:** log gross instead of log(gross/budget).
- **Genre rule:** "first listed" instead of the frozen priority rule. The two agree on only 49% of films.
- **Seed:** the permutations and the bootstrap rerun with four other seeds.

Each variant uses 500 permutations and 100 bootstrap resamples. All rows are **exploratory**. With this many
rows, one or two will pass p < 0.05 by chance; the question is whether the sign and the size are stable."""),
    Run("s09_robustness"),
    Fig("s09_robustness", "forest", "One row per variant: ρ with its 95% bootstrap CI. Blue: the primary analysis. "
        "Orange: a variant with p < 0.05. Dotted lines: ±r_min."),
    Table("s09_robustness", "variants", "Every variant, both descriptors."),
    Reading("s09_robustness"),
]
