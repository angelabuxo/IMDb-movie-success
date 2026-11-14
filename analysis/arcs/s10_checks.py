"""Step 10: franchise, transcript and leakage checks. The Step 7 statistic on
  (a) all arc-ready scripts (franchise included), (b) the primary sample (likely franchise excluded),
  (c) the primary sample without possible transcripts, and
  (d) all scripts minus as many RANDOM non-franchise films as there are franchise films, 50 times:
      if (a) differs from (b) only because (b) has fewer films, (d) looks like (b); if franchise films
      carry the difference, (d) looks like (a).
(a)-(c): 500 permutations and 100 bootstrap resamples (as Step 9). (d): 200 permutations per drop (config
`franchise_check`), no bootstrap. Exploratory except (b), which repeats the primary sample."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analysis.scripts_text.report import HIGH, LOW, NEUTRAL, Fig, Md, Reading, Run, Table, style

from . import s07_primary, s09_robustness
from .common import pooled_cells
from .config import cfg, config_hash, log_run
from .paths import OUTPUTS

CHECKS = OUTPUTS / "s10_checks.csv"
DROPS = OUTPUTS / "s10_random_drops.csv"
META = OUTPUTS / "s10_meta.json"
SAMPLES = {"(a) all scripts, franchise included": "all scripts (franchise included)",
           "(b) primary: likely franchise excluded": "primary",
           "(c) primary without possible transcripts": "primary without possible transcripts"}


def compute(inst: str, outcome: str | None = None, table_hook=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    c = cfg()
    rb, fc = c["robustness"], c["franchise_check"]
    cov = c["controls"]["covariates"]
    outcome = outcome or c["primary"]["outcome"]
    desc = s09_robustness.descriptors(inst)
    rng = np.random.default_rng(c["seed"])
    rows = []
    for label, mask in SAMPLES.items():
        df = s09_robustness.frame(desc, mask)
        if table_hook is not None:
            df = table_hook(df)
        for k in ("D1", "D2"):
            r = s07_primary.association(df, k, outcome, cov, rb["permutations_per_variant"], rb["bootstrap_resamples"], rng)
            rows.append({"sample": label, "descriptor": k, "N": r["n"], "franchise films": int(df.likely_franchise.sum()),
                         "rho": r["rho"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "z": r["z"], "p": r["p"]})
    full = s09_robustness.frame(desc, "all scripts (franchise included)")
    if table_hook is not None:
        full = table_hook(full)
    n_fr = int(full.likely_franchise.sum())
    nonfr = np.flatnonzero(~full.likely_franchise.to_numpy())
    drops = []
    for b in range(fc["random_drops"]):
        keep = np.setdiff1d(np.arange(len(full)), rng.choice(nonfr, n_fr, replace=False))
        df = full.iloc[keep].reset_index(drop=True)
        df["cell"] = pooled_cells(df.primary_genre, df.decade_bin)
        for k in ("D1", "D2"):
            r = s07_primary.association(df, k, outcome, cov, fc["permutations_per_drop"], 1, rng)
            drops.append({"drop": b, "descriptor": k, "N": r["n"], "rho": r["rho"], "p": r["p"]})
    meta = {"instrument": inst, "config": config_hash(), "franchise_in_all": n_fr, "random_drops": fc["random_drops"],
            "permutations_per_drop": fc["permutations_per_drop"]}
    return pd.DataFrame(rows), pd.DataFrame(drops), meta


def run() -> dict:
    if not s07_primary.DECISION.exists():
        raise RuntimeError("Step 10 needs outputs/s03_instrument_decision.json (fallback rule not applied yet).")
    inst = json.loads(s07_primary.DECISION.read_text())["primary_instrument"]
    checks, drops, meta = compute(inst)
    checks.to_csv(CHECKS, index=False)
    drops.to_csv(DROPS, index=False)
    META.write_text(json.dumps(meta) + "\n")
    log_run("s10_checks", [CHECKS.name, DROPS.name, META.name], {"franchise_check": cfg()["franchise_check"],
                                                                 "robustness": cfg()["robustness"], "instrument": inst})
    return load()


def load() -> dict:
    return {"checks": pd.read_csv(CHECKS), "drops": pd.read_csv(DROPS), "meta": json.loads(META.read_text())}


def _summary(res: dict) -> pd.DataFrame:
    ch, dr = res["checks"], res["drops"]
    rows = []
    for k in ("D1", "D2"):
        a = ch[(ch.descriptor == k) & ch["sample"].str.startswith("(a)")].iloc[0]
        b = ch[(ch.descriptor == k) & ch["sample"].str.startswith("(b)")].iloc[0]
        d = dr[dr.descriptor == k]
        rows.append({"descriptor": k, "ρ (a) all scripts": a.rho, "ρ (b) franchise excluded": b.rho,
                     "ρ (d) random drops, median": d.rho.median(),
                     "(d) 5th–95th percentile": f"[{d.rho.quantile(.05):+.3f}, {d.rho.quantile(.95):+.3f}]",
                     "share of drops at least as far from (a) as (b) is":
                         float((d.rho <= b.rho).mean() if b.rho <= a.rho else (d.rho >= b.rho).mean()),
                     "drops with p < 0.05": f"{int((d.p < 0.05).sum())} of {len(d)}"})
    return pd.DataFrame(rows)


def tables(res: dict) -> dict:
    ch = res["checks"].copy()
    ch["95% CI"] = [f"[{a:+.3f}, {b:+.3f}]" for a, b in zip(ch.ci_low, ch.ci_high)]
    return {"checks": ch[["sample", "descriptor", "N", "franchise films", "rho", "95% CI", "z", "p"]].round(3),
            "drops": _summary(res).round(3)}


def figures(res: dict) -> dict:
    ch, dr = res["checks"], res["drops"]
    out = {}
    for k in ("D1", "D2"):
        d = dr[dr.descriptor == k]
        n, e = np.histogram(d.rho, bins=20)
        f = go.Figure(go.Bar(x=(e[:-1] + e[1:]) / 2, y=n, width=np.diff(e), marker_color=NEUTRAL,
                             name=f"{len(d)} random drops"))
        for prefix, colour, txt in (("(a)", LOW, "(a) all scripts"), ("(b)", HIGH, "(b) franchise excluded")):
            v = ch[(ch.descriptor == k) & ch["sample"].str.startswith(prefix)].rho.iloc[0]
            f.add_vline(x=v, line_color=colour, line_width=3, annotation_text=txt)
        f.update_layout(xaxis_title=f"ρ for {k}, all scripts minus {res['meta']['franchise_in_all']} random "
                        f"non-franchise films", yaxis_title="drops", showlegend=False)
        out[f"drops_{k}"] = style(f, f"{k}: franchise films or just fewer films?", height=340)
    return out


def reading(res: dict) -> str:
    s = _summary(res).set_index("descriptor")
    ch = res["checks"]
    parts = []
    for k, label in (("D1", "volatility (D1)"), ("D2", "ending minus start (D2)")):
        r = s.loc[k]
        a, b, dmed = r["ρ (a) all scripts"], r["ρ (b) franchise excluded"], r["ρ (d) random drops, median"]
        c = ch[(ch.descriptor == k) & ch["sample"].str.startswith("(c)")].rho.iloc[0]
        lo, hi = [float(x) for x in r["(d) 5th–95th percentile"].strip("[]").split(",")]
        inside = lo <= b <= hi
        parts.append(f"For {label}: ρ = {a:+.3f} with all scripts, {b:+.3f} without likely franchise films and "
                     f"{c:+.3f} also without possible transcripts. Dropping the same number of random non-franchise "
                     f"films instead gives a median ρ of {dmed:+.3f} (5th–95th percentile {lo:+.3f} to {hi:+.3f}). "
                     + (f"(a) and (b) barely differ (Δρ = {b - a:+.3f}), so there is little for the franchise "
                        f"films to explain. " if abs(b - a) < 0.02 else "")
                     + ("The franchise-free value lies inside that range, so the difference between (a) and (b) "
                        "is what losing that many films does anyway, not something franchise films carry."
                        if inside else
                        "The franchise-free value lies outside that range, so franchise films themselves carry part "
                        "of the difference between (a) and (b)."))
    return " ".join(parts) + (" Leakage: the scripts were written before release, so a film's success cannot have "
                              "shaped its script. That is an advantage over reviews and ratings. That each MovieSum "
                              "script is the final shooting draft is an unverified assumption (Step 4's script "
                              "sheet checks a sample).")


SECTION = [
    Md("""## Step 10 · Franchise, transcript and leakage checks

**Why.** The earlier word-level branch found a weak signal that vanished once franchise films were removed.
That is the reason the primary sample here excludes likely franchise films. Removing them changes two things
at once, though: *which* films are in the sample and *how many*. This step separates the two:
- **(a)** all arc-ready scripts, franchise films included;
- **(b)** the primary sample, likely franchise films excluded;
- **(c)** the primary sample also without the possible transcripts, which have little description;
- **(d)** all scripts minus the same number of **randomly chosen non-franchise** films, 50 times.

If (b) differs from (a) only because it has fewer films, the random drops in (d) should look like (b). If
the franchise films themselves matter, (d) should look like (a).

**Leakage.** Screenplays are written before release, so a film's success cannot have shaped its script. That
is an advantage over reviews and ratings, which come after. We assume each MovieSum script is the final
shooting draft. We can't verify that from the data; the script sheet of Step 4 checks a sample by hand.

These rows use the same statistic as Step 7. Only (b) repeats the primary sample; the rest is exploratory."""),
    Run("s10_checks"),
    Table("s10_checks", "checks", "The Step 7 statistic on samples (a), (b) and (c)."),
    Table("s10_checks", "drops", "Random drops (d) against (a) and (b)."),
    Fig("s10_checks", "drops_D1", "D1: ρ over 50 random drops (grey), with (a) all scripts and (b) franchise "
        "excluded marked."),
    Fig("s10_checks", "drops_D2", "D2: the same."),
    Reading("s10_checks"),
]
