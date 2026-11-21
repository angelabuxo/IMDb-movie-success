"""Two follow-ups Angi asked for on 2026-11-21, with the primary pipeline of run_tests.py unchanged.

1. Label shuffle inside the 100k+ tier only (Comedy and Action x dialogue and description). The baseline loses
   its tier column there (it is constant), so this asks whether the vector still beats a shuffle once the vote
   tier can't carry the result. Word2Vec (window 5, seeds 1-5), TF-IDF and the genre/era baseline, 200 label
   permutations each, the same permutation seeds and CV settings as primary.csv. The folds are cut from the
   smaller set, so they are not the same films' folds as in primary.csv.
2. A 95 % paired film bootstrap interval for AUC(vector + baseline) and for AUC(vector + baseline) - AUC(baseline)
   on the full labelled set, using the out-of-fold scores (no refit inside the bootstrap, so the interval ignores
   training variance and is too narrow if anything).

Outputs (numbers only) in results/:
  subset_100k.csv                 one row per genre x layer x representation, with shuffle p
  null_100k_<genre>_<layer>.csv   the permutation AUCs
  combo_interval.csv              intervals for vector + baseline

Run:  IMDB_DATA_DIR=... python -I scripts-analysis/word2vec/subset_tests.py [--perms 200] [--jobs 9] [--skip-subset]
"""
import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tests import (  # noqa: E402
    LAYERS, N_BOOT, OUT, SEEDS, CV_SEED, Data, auc, baseline_matrix, evaluate, labelled,
                       mean_auc, oof, perm_one)

GENRES = ("Comedy", "Action")
TIER = "100k+"


def setup(genre, tier=None):
    rows = labelled(genre, tier=tier)
    ids = [fl["id"] for fl, _ in rows]
    y = np.array([lb == "hit" for _, lb in rows], int)
    return ids, y, baseline_matrix(rows, genre)


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def subset(D, perms, jobs):
    out = []
    for genre in GENRES:
        ids, y, Xb = setup(genre, TIER)
        t = time.time()
        base_auc = mean_auc(oof(Xb, y, "dense"), y)
        base_null = None
        for layer in LAYERS:
            obs = evaluate(D, ids, y, layer, Xb)
            null = np.array(Parallel(n_jobs=jobs)(delayed(perm_one)(D, ids, y, layer, Xb, p) for p in range(perms)))
            with open(OUT / f"null_100k_{genre}_{layer}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["perm", "w2v", "tfidf", "baseline"])
                w.writerows([[p, *map(lambda v: round(v, 5), r)] for p, r in enumerate(null)])
            if layer == LAYERS[0]:
                base_null = null[:, 2]
            for j, rep in enumerate(("w2v", "tfidf")):
                m, sd, _ = obs[rep]
                nl = null[:, j]
                out.append({"genre": genre, "tier": TIER, "layer": layer, "representation": rep,
                            "n_hit": int(y.sum()), "n_flop": int(len(y) - y.sum()), "auc": round(m, 4),
                            "auc_sd_over_seeds": "" if np.isnan(sd) else round(sd, 4),
                            "auc_minus_baseline": round(m - base_auc, 4),
                            "shuffle_mean": round(nl.mean(), 4), "shuffle_q95": round(np.quantile(nl, .95), 4),
                            "shuffle_p": round((1 + (nl >= m).sum()) / (1 + len(nl)), 4), "perms": len(nl)})
            print(f"100k+ {genre} {layer}: w2v {obs['w2v'][0]:.3f}, tfidf {obs['tfidf'][0]:.3f}, "
                  f"base {base_auc:.3f} ({time.time() - t:.0f} s)", flush=True)
        out.append({"genre": genre, "tier": TIER, "layer": "-", "representation": "baseline (genre + era)",
                    "n_hit": int(y.sum()), "n_flop": int(len(y) - y.sum()), "auc": round(base_auc, 4),
                    "auc_sd_over_seeds": "", "auc_minus_baseline": 0.0,
                    "shuffle_mean": round(base_null.mean(), 4), "shuffle_q95": round(np.quantile(base_null, .95), 4),
                    "shuffle_p": round((1 + (base_null >= base_auc).sum()) / (1 + len(base_null)), 4),
                    "perms": len(base_null)})
        write(OUT / "subset_100k.csv", out)


def combo_interval(D):
    out = []
    rng = np.random.default_rng(CV_SEED)
    for genre in GENRES:
        ids, y, Xb = setup(genre)
        sb = oof(Xb, y, "dense")                                   # (REPEATS, n)
        for layer in LAYERS:
            sc = np.concatenate([oof(np.hstack([D.rows(ids, D.vec(5, s, layer)), Xb]), y, "dense")
                                 for s in SEEDS])                  # (SEEDS * REPEATS, n)
            obs_c, obs_b = mean_auc(sc, y), mean_auc(sb, y)
            bc, bd = [], []
            while len(bc) < N_BOOT:
                idx = rng.integers(0, len(y), len(y))
                yb = y[idx]
                if yb.min() == yb.max():
                    continue
                c = np.mean([auc(yb, s[idx]) for s in sc])
                b = np.mean([auc(yb, s[idx]) for s in sb])
                bc.append(c)
                bd.append(c - b)
            (clo, chi), (dlo, dhi) = np.percentile(bc, [2.5, 97.5]), np.percentile(bd, [2.5, 97.5])
            out.append({"genre": genre, "layer": layer, "n": len(y), "auc_w2v_plus_baseline": round(obs_c, 4),
                        "ci95_lo": round(clo, 4), "ci95_hi": round(chi, 4), "auc_baseline": round(obs_b, 4),
                        "difference": round(obs_c - obs_b, 4), "diff_ci95_lo": round(dlo, 4),
                        "diff_ci95_hi": round(dhi, 4), "boot": N_BOOT})
            print(f"combo {genre} {layer}: {obs_c:.3f} [{clo:.3f}, {chi:.3f}], minus baseline "
                  f"{obs_c - obs_b:+.3f} [{dlo:+.3f}, {dhi:+.3f}]", flush=True)
    write(OUT / "combo_interval.csv", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--skip-subset", action="store_true")
    a = ap.parse_args()
    warnings.simplefilter("ignore", ConvergenceWarning)
    D = Data()
    combo_interval(D)
    if not a.skip_subset:
        subset(D, a.perms, a.jobs)


if __name__ == "__main__":
    main()
