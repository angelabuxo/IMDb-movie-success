"""Does popularity inside the 100k+ tier explain the vector's AUC there?

The tier is a binary split at 100,000 IMDb votes, and votes still vary more than tenfold inside 100k+. Votes are a
consequence of popularity, which goes with success, so log(votes) is the obvious alternative explanation for
"the signal survives the tier control". For Comedy and Action, all labelled films and 100k+ only:
  log_votes_raw        AUC of log(votes) itself (no model)
  baseline+votes       genre/era baseline (with tier on the full set) plus log(votes), same CV as run_tests
  w2v+baseline+votes   the window-5 vector (seeds 1-5) added to that
Observed AUCs only, no shuffle.

Run:  IMDB_DATA_DIR=... python -I analysis/screenplays/word2vec/votes_check.py
"""
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tests import (LAYERS, OUT, SEEDS, Data, auc, baseline_matrix, labelled, mean_auc,  # noqa: E402
                       oof)
from genre_counts import DATA  # noqa: E402  (run_tests puts genre_counts on the path)


def votes():
    with open(DATA / "features" / "movies_features_10k.csv", newline="") as f:
        return {r["tconst"]: float(r["votes"]) for r in csv.DictReader(f)}


def main():
    warnings.simplefilter("ignore", ConvergenceWarning)
    V, D, out = votes(), Data(), []
    for genre in ("Comedy", "Action"):
        for tier in (None, "100k+"):
            rows = labelled(genre, tier=tier)
            ids = [fl["id"] for fl, _ in rows]
            y = np.array([lb == "hit" for _, lb in rows], int)
            lv = np.log([V[i] for i in ids])[:, None]
            Xb = np.hstack([baseline_matrix(rows, genre), lv])
            base = {"genre": genre, "set": tier or "all labelled", "n_hit": int(y.sum()), "n_flop": int(len(y) - y.sum()),
                    "median_votes_hit": int(np.median(np.exp(lv[y == 1]))),
                    "median_votes_flop": int(np.median(np.exp(lv[y == 0])))}
            out.append({**base, "model": "log_votes_raw", "layer": "-", "auc": round(auc(y, lv.ravel()), 4)})
            out.append({**base, "model": "baseline+votes", "layer": "-", "auc": round(mean_auc(oof(Xb, y, "dense"), y), 4)})
            for layer in LAYERS:
                a = [mean_auc(oof(np.hstack([D.rows(ids, D.vec(5, s, layer)), Xb]), y, "dense"), y) for s in SEEDS]
                out.append({**base, "model": "w2v+baseline+votes", "layer": layer, "auc": round(float(np.mean(a)), 4)})
            print(*[f"{r['model']} {r['layer']} {r['auc']}" for r in out[-4:]], sep=" | ", flush=True)
    with open(OUT / "votes_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
