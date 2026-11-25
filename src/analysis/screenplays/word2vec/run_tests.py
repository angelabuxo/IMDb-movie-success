"""The four tests (Comedy and Action x dialogue and description) plus Drama (exploratory).

Settings: DECISIONS.md, "Test settings fixed before any AUC". Per genre and layer, hit vs flop is predicted by
  w2v       the TF-IDF-weighted mean Word2Vec vector (window 5, seeds 1-5, AUC averaged over seeds)
  tfidf     the TF-IDF representation over the same vocabulary
  baseline  tier + era one-hot + the film's other genres
  w2v+base  the two together (observed only)
Each AUC is compared with a 200-permutation label shuffle of the identical pipeline. The hypothesis contrast
AUC(dialogue) - AUC(description) gets a paired film bootstrap interval.

Outputs (numbers only) in results/:
  primary.csv     one row per genre x layer x representation
  contrast.csv    the dialogue - description contrast per genre and representation
  robustness.csv  windows 2 and 10, plain mean, 100k+ only (observed AUCs, no shuffle)
  null_<genre>_<layer>.csv  the permutation AUCs

Run:  IMDB_DATA_DIR=... python -I analysis/screenplays/word2vec/run_tests.py [--perms 200] [--jobs 8]
"""
import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "genre_counts"))
from final_set import final_films  # noqa: E402
from genre_cells import ERAS, label, pairs_for  # noqa: E402

PRIVATE = HERE.parent / "private"
OUT = HERE / "results"
LAYERS = ("dialogue", "description")
GENRES = (("Comedy", "primary"), ("Action", "primary"), ("Drama", "exploratory"))
PREDICTED = {"Comedy": "dialogue > description", "Action": "description > dialogue", "Drama": "none (exploratory)"}
SEEDS = (1, 2, 3, 4, 5)
CS = np.logspace(-4, 2, 7)
FOLDS, REPEATS, CV_SEED = 5, 5, 2805
N_BOOT = 2000


def auc(y, s):
    """Mann-Whitney AUC with tied scores counted as half."""
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def model(kind):
    if kind == "tfidf":
        return LogisticRegressionCV(Cs=CS, cv=5, scoring="roc_auc", solver="liblinear", max_iter=5000,
                                    l1_ratios=(0,), use_legacy_attributes=False)
    return make_pipeline(StandardScaler(),
                         LogisticRegressionCV(Cs=CS, cv=5, scoring="roc_auc", max_iter=5000,
                                              l1_ratios=(0,), use_legacy_attributes=False))


def oof(X, y, kind):
    """Out-of-fold scores, shape (REPEATS, n)."""
    out = np.zeros((REPEATS, len(y)))
    cv = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=CV_SEED)
    for i, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        m = model(kind).fit(X[tr], y[tr])
        out[i // FOLDS, te] = m.decision_function(X[te])
    return out


def mean_auc(scores, y):
    return float(np.mean([auc(y, s) for s in scores]))


def labelled(genre, tier=None):
    films = final_films()
    lab, _ = label(pairs_for(films, "a"))
    rows = sorted([(fl, lb) for fl, g, _, lb in lab if g == genre and lb in ("hit", "flop")],
                  key=lambda x: x[0]["id"])
    if tier:
        rows = [r for r in rows if r[0]["tier"] == tier]
    return rows


def baseline_matrix(rows, genre):
    from genre_cells import era
    others = sorted({g for fl, _ in rows for g in fl["genres"] if g != genre})
    X = []
    for fl, _ in rows:
        X.append([fl["tier"] == "100k+"] + [era(fl["decade"]) == e for e in ERAS] + [g in fl["genres"] for g in others])
    return np.array(X, float)


class Data:
    def __init__(self):
        self.ids = list(np.load(PRIVATE / "vector_ids.npy"))
        self.pos = {f: i for i, f in enumerate(self.ids)}
        self.tfidf = {k: sparse.load_npz(PRIVATE / f"tfidf_{k}.npz") for k in LAYERS}
        self.cache = {}

    def vec(self, window, seed, layer, agg="tfidf_mean"):
        key = (window, seed)
        if key not in self.cache:
            self.cache[key] = dict(np.load(PRIVATE / f"vectors_w{window}_s{seed}.npz"))
        return self.cache[key][f"{layer}_{agg}"]

    def rows(self, ids, M):
        idx = [self.pos[f] for f in ids]
        return M[idx]


def evaluate(D, ids, y, layer, Xb, window=5, agg="tfidf_mean", with_tfidf=True, with_combo=False):
    """Mean AUCs and OOF scores for one genre x layer."""
    res = {}
    w2v_scores, seed_aucs = [], []
    for s in SEEDS:
        X = D.rows(ids, D.vec(window, s, layer, agg))
        if np.isnan(X).any():
            raise SystemExit(f"empty vector for a labelled film in {layer}")
        sc = oof(X, y, "dense")
        w2v_scores.append(sc)
        seed_aucs.append(mean_auc(sc, y))
    res["w2v"] = (float(np.mean(seed_aucs)), float(np.std(seed_aucs, ddof=1)), np.array(w2v_scores))
    if with_tfidf:
        sc = oof(D.rows(ids, D.tfidf[layer]), y, "tfidf")
        res["tfidf"] = (mean_auc(sc, y), float("nan"), sc[None])
    if with_combo:
        aucs = [mean_auc(oof(np.hstack([D.rows(ids, D.vec(window, s, layer, agg)), Xb]), y, "dense"), y)
                for s in SEEDS]
        res["w2v+base"] = (float(np.mean(aucs)), float(np.std(aucs, ddof=1)), None)
    return res


def perm_one(D, ids, y, layer, Xb, p):
    rng = np.random.default_rng(10_000 + p)
    yp = rng.permutation(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        r = evaluate(D, ids, yp, layer, Xb)
        b = mean_auc(oof(Xb, yp, "dense"), yp) if layer == LAYERS[0] else float("nan")
    return r["w2v"][0], r["tfidf"][0], b


def boot_contrast(sc_dia, sc_des, y, rng):
    """sc_*: (k, REPEATS, n) OOF scores; returns observed and bootstrap differences, averaged over k and repeats."""
    flat_dia = sc_dia.reshape(-1, len(y))
    flat_des = sc_des.reshape(-1, len(y))
    obs = np.mean([auc(y, a) - auc(y, b) for a, b in zip(flat_dia, flat_des)])
    n, boots = len(y), []
    while len(boots) < N_BOOT:
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        boots.append(np.mean([auc(yb, a[idx]) - auc(yb, b[idx]) for a, b in zip(flat_dia, flat_des)]))
    return obs, np.percentile(boots, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--genres", default="Comedy,Action,Drama")
    a = ap.parse_args()
    warnings.simplefilter("ignore", ConvergenceWarning)
    OUT.mkdir(exist_ok=True)
    D = Data()
    primary, contrast, robust = [], [], []
    for genre, status in GENRES:
        if genre not in a.genres.split(","):
            continue
        rows = labelled(genre)
        ids = [fl["id"] for fl, _ in rows]
        y = np.array([lb == "hit" for _, lb in rows], int)
        Xb = baseline_matrix(rows, genre)
        t = time.time()
        sb = oof(Xb, y, "dense")
        base_auc = mean_auc(sb, y)
        scores = {}
        for layer in LAYERS:
            obs = evaluate(D, ids, y, layer, Xb, with_combo=True)
            scores[layer] = obs
            null = Parallel(n_jobs=a.jobs)(delayed(perm_one)(D, ids, y, layer, Xb, p) for p in range(a.perms))
            null = np.array(null)
            with open(OUT / f"null_{genre}_{layer}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["perm", "w2v", "tfidf", "baseline"])
                w.writerows([[p, *map(lambda v: round(v, 5), r)] for p, r in enumerate(null)])
            if layer == LAYERS[0]:
                base_null = null[:, 2]
            for j, rep in enumerate(("w2v", "tfidf")):
                m, sd, _ = obs[rep]
                nl = null[:, j]
                primary.append({"genre": genre, "status": status, "layer": layer, "representation": rep,
                                "n_hit": int(y.sum()), "n_flop": int(len(y) - y.sum()),
                                "auc": round(m, 4), "auc_sd_over_seeds": "" if np.isnan(sd) else round(sd, 4),
                                "auc_minus_baseline": round(m - base_auc, 4),
                                "shuffle_mean": round(nl.mean(), 4), "shuffle_q95": round(np.quantile(nl, .95), 4),
                                "shuffle_p": round((1 + (nl >= m).sum()) / (1 + len(nl)), 4), "perms": len(nl)})
            m, sd, _ = obs["w2v+base"]
            primary.append({"genre": genre, "status": status, "layer": layer, "representation": "w2v+baseline",
                            "n_hit": int(y.sum()), "n_flop": int(len(y) - y.sum()), "auc": round(m, 4),
                            "auc_sd_over_seeds": round(sd, 4), "auc_minus_baseline": round(m - base_auc, 4),
                            "shuffle_mean": "", "shuffle_q95": "", "shuffle_p": "", "perms": 0})
            print(f"{genre} {layer}: w2v {obs['w2v'][0]:.3f}, tfidf {obs['tfidf'][0]:.3f}, "
                  f"base {base_auc:.3f} ({time.time() - t:.0f} s)", flush=True)
        primary.append({"genre": genre, "status": status, "layer": "-", "representation": "baseline",
                        "n_hit": int(y.sum()), "n_flop": int(len(y) - y.sum()), "auc": round(base_auc, 4),
                        "auc_sd_over_seeds": "", "auc_minus_baseline": 0.0,
                        "shuffle_mean": round(base_null.mean(), 4), "shuffle_q95": round(np.quantile(base_null, .95), 4),
                        "shuffle_p": round((1 + (base_null >= base_auc).sum()) / (1 + len(base_null)), 4),
                        "perms": len(base_null)})
        rng = np.random.default_rng(CV_SEED)
        for rep in ("w2v", "tfidf"):
            d, (lo, hi) = boot_contrast(scores["dialogue"][rep][2], scores["description"][rep][2], y, rng)
            contrast.append({"genre": genre, "status": status, "representation": rep, "predicted": PREDICTED[genre],
                             "auc_dialogue": round(scores["dialogue"][rep][0], 4),
                             "auc_description": round(scores["description"][rep][0], 4),
                             "dialogue_minus_description": round(d, 4), "ci95_lo": round(lo, 4),
                             "ci95_hi": round(hi, 4), "boot": N_BOOT})
        # robustness: observed AUCs only
        for name, kw in (("window 2", {"window": 2}), ("window 10", {"window": 10}),
                         ("plain mean", {"agg": "plain_mean"})):
            for layer in LAYERS:
                r = evaluate(D, ids, y, layer, Xb, with_tfidf=False, **kw)
                robust.append({"genre": genre, "check": name, "layer": layer, "representation": "w2v",
                               "n": len(y), "auc": round(r["w2v"][0], 4), "auc_sd_over_seeds": round(r["w2v"][1], 4),
                               "baseline_auc": round(base_auc, 4)})
        rows_t = labelled(genre, tier="100k+")
        ids_t = [fl["id"] for fl, _ in rows_t]
        y_t = np.array([lb == "hit" for _, lb in rows_t], int)
        Xb_t = baseline_matrix(rows_t, genre)
        bt = mean_auc(oof(Xb_t, y_t, "dense"), y_t)
        for layer in LAYERS:
            r = evaluate(D, ids_t, y_t, layer, Xb_t)
            for rep in ("w2v", "tfidf"):
                robust.append({"genre": genre, "check": f"100k+ only ({y_t.sum()} hits, {len(y_t) - y_t.sum()} flops)",
                               "layer": layer, "representation": rep, "n": len(y_t), "auc": round(r[rep][0], 4),
                               "auc_sd_over_seeds": "" if np.isnan(r[rep][1]) else round(r[rep][1], 4),
                               "baseline_auc": round(bt, 4)})
        print(f"{genre}: done ({time.time() - t:.0f} s)", flush=True)
        for fname, data in (("primary.csv", primary), ("contrast.csv", contrast), ("robustness.csv", robust)):
            with open(OUT / fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)


if __name__ == "__main__":
    main()
