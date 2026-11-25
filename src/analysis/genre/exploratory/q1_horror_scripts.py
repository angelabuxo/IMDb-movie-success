"""Exploratory: the Horror word test of q1_words.py on the screenplays instead of the Wikipedia plots. It did not
replicate (no word survives the family-wise shuffle; supernatural words don't separate the films), which is one reason
the plot-word result is exploratory. Moved here from the first q1_scripts_network.py.

Runs on the usable Horror films with a MovieSum script, description and dialogue layers, with the counts of Q3's
preprocessing (features/script_w2v/script_counts.pkl). Outputs in analysis/genre/exploratory/outputs/horror_scripts/.
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.exploratory.q1_horror_scripts [--perms 1000]
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from analysis.genre.exploratory import EXP_OUT
from analysis.genre.exploratory.q1_words import SUPERNATURAL, log_odds_z
from analysis.genre.q1_genre import DATA, SEED, load, prepare
from analysis.genre.q1_tmdb import net_return

OUT = EXP_OUT / "horror_scripts"


def horror_scripts(f, keep, M, perms):
    counts = pickle.loads((DATA / "features" / "script_w2v" / "script_counts.pkl").read_bytes())
    f = f.assign(net=net_return(f))
    h = f[(M[:, keep.index("Horror")] == 1) & f["imdb_id"].isin(counts)].reset_index(drop=True)
    q1, q3 = h["net"].quantile([.25, .75])
    lab = h[(h["net"] <= q1) | (h["net"] >= q3)].reset_index(drop=True)
    y = (lab["net"] >= q3).to_numpy().astype(int)
    rng = np.random.default_rng(SEED + 1900)
    words_out, lists, summ = [], [], {"films_with_script": len(h), "hits": int(y.sum()), "flops": int(len(y) - y.sum()), "perms": perms}
    for layer in ("description", "dialogue"):
        prior = Counter()
        for t in f["imdb_id"]:
            if t in counts:
                prior.update(counts[t][layer])
        df_ = Counter(w for t in lab["imdb_id"] for w in counts[t][layer])
        vocab = sorted(w for w, d in df_.items() if d >= 10)
        vi = {w: j for j, w in enumerate(vocab)}
        C = np.zeros((len(lab), len(vocab)))
        for i, t in enumerate(lab["imdb_id"]):
            for w, c in counts[t][layer].items():
                if w in vi:
                    C[i, vi[w]] = c
        alpha = np.array([prior[w] for w in vocab], float)
        zf = lambda yy: log_odds_z(C[yy == 1].sum(0), C[yy == 0].sum(0), alpha)
        obs = zf(y)
        grp = lab["budget_bin"].to_numpy()
        idx = [np.flatnonzero(grp == g) for g in np.unique(grp)]
        nl = np.empty((perms, len(vocab)))
        for p in range(perms):
            yp = y.copy()
            for ix in idx:
                yp[ix] = y[rng.permutation(ix)]
            nl[p] = zf(yp)
        zc = (obs - nl.mean(0)) / nl.std(0, ddof=1)
        zcn = (nl - nl.mean(0)) / nl.std(0, ddof=1)
        w = pd.DataFrame({"layer": layer, "word": vocab, "calibrated_z": zc.round(2),
                          "hit_films": (C[y == 1] > 0).sum(0), "flop_films": (C[y == 0] > 0).sum(0)})
        words_out.append(pd.concat([w.nlargest(15, "calibrated_z").assign(side="hit"), w.nsmallest(15, "calibrated_z").assign(side="flop")]))
        summ[f"{layer}_vocab"] = len(vocab)
        summ[f"{layer}_familywise_p"] = round(float((1 + (np.abs(zcn).max(1) >= np.abs(zc).max()).sum()) / (1 + perms)), 4)
        for side, sg in (("hit", 1), ("flop", -1)):
            o, n_ = np.sort(sg * zc)[::-1], np.sort(sg * zcn, axis=1)[:, ::-1]
            for k in (5, 10, 20):
                lists.append({"layer": layer, "side": side, "k": k, "p": round(float((1 + (n_[:, k - 1] >= o[k - 1]).sum()) / (1 + perms)), 4)})
        sup = set(SUPERNATURAL)
        summ[f"{layer}_supernatural_words_in_vocab"] = sorted(sup & set(vocab))
    # supernatural set counted in the scripts (description + dialogue), on all Horror films with a script
    sup_film = np.array([any(counts[t][l].get(w, 0) > 0 for l in ("description", "dialogue") for w in SUPERNATURAL) for t in h["imdb_id"]])
    net = h["net"].to_numpy()
    d = net[sup_film].mean() - net[~sup_film].mean()
    nl = np.array([(lambda m: net[m].mean() - net[~m].mean())(rng.permutation(sup_film)) for _ in range(5000)])
    summ["supernatural_in_script"] = {"films": int(sup_film.sum()), "share": round(float(sup_film.mean()), 3),
                                      "net_supernatural": round(float(net[sup_film].mean()), 3), "net_other": round(float(net[~sup_film].mean()), 3),
                                      "difference": round(float(d), 3), "p": round(float((1 + (np.abs(nl) >= abs(d)).sum()) / 5001), 4)}
    return pd.concat(words_out), pd.DataFrame(lists), summ



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    OUT.mkdir(parents=True, exist_ok=True)
    w, l, s = horror_scripts(f, keep, M, a.perms)
    w.to_csv(OUT / "horror_script_words.csv", index=False)
    l.to_csv(OUT / "horror_script_list_null.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
