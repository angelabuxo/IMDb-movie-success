"""EXPLORATORY (added 14 Nov 2026, after the primary analysis): replicate Del Vecchio et al. (2021) on our scripts.

Del Vecchio, Kharlamov, Parry & Pogrebna (2021), J. Operational Research Society 72(5), report that the
"Man in a hole" arc earns more. Their pipeline, as described in the arXiv version (1807.02221):
lexicon sentiment per sentence, DCT low-pass smoothing to 100 points, k-means into six clusters, then one OLS
per cluster of gross revenue (in dollars) on that cluster's dummy alone. No budget, genre or franchise
control in the same model; a budget subsample is reported separately.

We rebuild that on our 100-point arcs and then tighten it one step at a time:
  S1  their test        gross ($M) ~ dummy                                  all 1,087 scripts
  S2  + budget          gross ($M) ~ dummy + budget ($M)                    all 1,087
  S3  our outcome       ln(gross/budget) ~ dummy + ln budget                 all 1,087
  S4  our controls      ln(gross/budget) ~ dummy + ln budget + genre x decade cells + mean sentences per scene
                                                                             834 primary films (no likely franchise)
Each spec is run for every one of the six clusters (six tests, so a naive p < 0.05 somewhere is expected
about a quarter of the time by chance). Robust (HC3) standard errors. Lexicon arm: VADER (syuzhet is not
installed; VADER is the closest lexicon instrument we have). Model arm: SST-2, our primary instrument.

    python -m analysis.arcs.x01_del_vecchio
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.fft import dct, idct
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from .config import cfg
from .paths import OUTPUTS

LOW_PASS = 5            # syuzhet's get_dct_transform default keeps the first 5 DCT components
K = 6
SHAPES = {               # the six Reagan/Del Vecchio shapes as templates on t in [0, 1]
    "Rags to riches": lambda t: t,
    "Riches to rags": lambda t: -t,
    "Man in a hole": lambda t: (2 * t - 1) ** 2,
    "Icarus": lambda t: -(2 * t - 1) ** 2,
    "Cinderella": lambda t: -np.cos(3 * np.pi * t),
    "Oedipus": lambda t: np.cos(3 * np.pi * t),
}


def dct_smooth(x: np.ndarray) -> np.ndarray:
    c = dct(x, norm="ortho")
    c[LOW_PASS:] = 0
    return idct(c, norm="ortho")


def clusters(inst: str, seed: int) -> tuple[pd.Series, dict]:
    a = pd.read_csv(OUTPUTS / f"s05_arcs_{inst}.csv.gz").sort_values(["movie_name", "t"])
    M = a.pivot(index="movie_name", columns="t", values="raw")
    n_all = len(M)
    M = M.dropna()                                   # arcs with a missing point are left out (counted below)
    S = np.apply_along_axis(dct_smooth, 1, M.to_numpy())
    sd = S.std(1, keepdims=True)
    keep = sd[:, 0] > 0                              # a flat arc has no shape to cluster
    M, S, sd = M[keep], S[keep], sd[keep]
    Z = (S - S.mean(1, keepdims=True)) / sd
    km = KMeans(K, n_init=50, random_state=seed).fit(Z)
    t = np.linspace(0, 1, Z.shape[1])
    names = list(SHAPES)
    T = np.array([SHAPES[n](t) for n in names])
    r = np.array([[stats.pearsonr(c, s)[0] for s in T] for c in km.cluster_centers_])
    rows, cols = linear_sum_assignment(-r)
    label = {int(i): names[j] for i, j in zip(rows, cols)}
    match = {names[j]: round(float(r[i, j]), 3) for i, j in zip(rows, cols)}
    match["films_dropped_missing_or_flat"] = n_all - len(M)
    return pd.Series([label[c] for c in km.labels_], index=M.index, name="shape"), match


def run() -> dict:
    seed = cfg()["seed"]
    t = pd.read_csv(OUTPUTS / "analysis_table.csv")
    t = t[t.arc_ready & t.budget_usd.gt(0) & t.gross_usd.gt(0)].copy()
    t["gross_m"], t["budget_m"] = t.gross_usd / 1e6, t.budget_usd / 1e6
    out = {"note": "exploratory, not pre-registered; replication of Del Vecchio et al. (2021)", "instruments": {}}
    rows = []
    for inst in ("vader", "sst2"):
        shape, match = clusters(inst, seed)
        d = t.join(shape, on="movie_name").dropna(subset=["shape"])
        specs = {
            "S1 their test: gross ~ dummy": (d, "gross_m ~ D"),
            "S2 + budget": (d, "gross_m ~ D + budget_m"),
            "S3 our outcome: ln(gross/budget) ~ dummy + ln budget": (d, "log_mult ~ D + log_budget"),
            "S4 + genre x decade, scene length, no franchise": (
                d[d.in_primary], "log_mult ~ D + log_budget + mean_sentences_per_scene + C(cell)"),
        }
        info = {"template_match_r": match, "cluster_sizes": d["shape"].value_counts().to_dict(),
                "kruskal_wallis_log_mult_p": float(stats.kruskal(*[g.log_mult for _, g in d.groupby("shape")]).pvalue)}
        for spec, (df, f) in specs.items():
            for s in SHAPES:
                df = df.assign(D=(df["shape"] == s).astype(int))
                m = smf.ols(f, df).fit(cov_type="HC3")
                rows.append({"instrument": inst, "spec": spec, "shape": s, "n": int(m.nobs), "n_in_shape": int(df.D.sum()),
                             "coef": float(m.params["D"]), "se": float(m.bse["D"]), "p": float(m.pvalues["D"])})
        out["instruments"][inst] = info
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS / "x01_del_vecchio.csv", index=False)
    mih = res[res["shape"] == "Man in a hole"].set_index(["instrument", "spec"])[["n", "n_in_shape", "coef", "p"]]
    out["man_in_a_hole"] = {f"{i} | {s}": {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
                            for (i, s), r in mih.to_dict("index").items()}
    out["shapes_p_below_05"] = {f"{i} | {s}": int((g.p < 0.05).sum())
                                for (i, s), g in res.groupby(["instrument", "spec"], sort=False)}
    (OUTPUTS / "x01_del_vecchio.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
