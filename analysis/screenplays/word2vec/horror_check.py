"""What the Horror hit / flop separation of all_genres.py is made of (post hoc, exploratory).

all_genres.py finds that Horror hit scripts resemble each other, and flop scripts each other, more than across the
two groups. Reading the titles suggested the hits are mostly slasher and haunted-house films and the flops creature
and sci-fi horror, and that cheap films are the hits. This script measures both instead of reading titles:

  composition  share of hits and of flops in each TMDB keyword subgenre (the keyword sets of
               analysis/genre/q1_tmdb.py, fixed there before looking), and their median budgets
  stratified   the separation statistic of all_genres.py with the label shuffle done inside era x subgenre flag
               or era x budget half. If the separation is only the subgenre or the budget level, a shuffle that keeps
               that mix in every stratum reaches the observed value and the p goes up.

Cosine averaged over every window-5 seed found, as in all_genres.py. Writes aggregate numbers only (no per-film
TMDB data) to results/horror_check.csv and results/horror_composition.csv.
Run:  IMDB_DATA_DIR=... python -I analysis/screenplays/word2vec/horror_check.py [--perms 2000]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "genre_counts"))
from all_genres import DATA, LAYERS, OUT, SEED, VECTORS, load_vectors, test  # noqa: E402
from final_set import final_films, read_table  # noqa: E402
from genre_cells import label, pairs_for  # noqa: E402

# The keyword sets of analysis/genre/q1_tmdb.py (SUBGENRES), copied so this script runs without pandas.
SUBGENRES = {
    "supernatural": {"supernatural", "demon", "demonic possession", "possession", "ghost", "haunted house", "haunting",
                     "exorcism", "paranormal", "poltergeist", "witch", "curse", "occult", "evil spirit"},
    "slasher": {"slasher", "serial killer", "masked killer", "stalker"},
    "found footage": {"found footage"},
    "creature": {"monster", "creature", "creature feature", "alien", "giant monster", "zombie", "vampire", "werewolf"},
}


def tmdb_keywords(ids):
    csv.field_size_limit(sys.maxsize)
    out = {}
    with open(DATA / "tmdb" / "tmdb_project_movies.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["imdb_id"] in ids and r["imdb_id"] not in out:
                out[r["imdb_id"]] = {k.strip().lower() for k in (r["keywords"] or "").split("|") if k.strip()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED + 666)

    index, vecs = load_vectors()
    films = [fl for fl in final_films() if fl["id"] in index]
    labelled, _ = label(pairs_for(films, "a"))
    items = [(fl["id"], e, 1 if lab == "hit" else 0) for fl, g, e, lab in labelled if g == "Horror" and lab != "middle"]
    ids = [t[0] for t in items]
    y = np.array([t[2] for t in items])
    print(f"Horror: {y.sum()} hits, {len(y) - y.sum()} flops; seeds: {', '.join(p.stem for p in VECTORS)}")

    budget = {r["imdb_id"]: float(r["budget"]) for r in read_table() if r["imdb_id"] in set(ids)}
    kw = tmdb_keywords(set(ids))
    flags = {name: np.array([bool(kw.get(i, set()) & words) for i in ids]) for name, words in SUBGENRES.items()}
    b = np.array([budget[i] for i in ids])
    flags["budget above Horror median"] = b > np.median(b)

    comp = [{"group": grp, "films": int(m.sum()), "with_tmdb_keywords": int(sum(ids[k] in kw and bool(kw[ids[k]])
                                                                               for k in np.where(m)[0])),
             "median_budget_musd": float(np.median(b[m]) / 1e6),
             **{f"share_{n}": float(flags[n][m].mean()) for n in SUBGENRES}}
            for grp, m in (("hit", y == 1), ("flop", y == 0))]
    for c in comp:
        print(c)

    rows = []
    for name, flag in [("era only", None)] + list(flags.items()):
        cells = defaultdict(list)
        for k, t in enumerate(items):
            cells[(t[1], None if flag is None else bool(flag[k]))].append(k)
        strata = [np.array(v) for v in cells.values()]
        for layer in LAYERS:
            obs, _, null, p, z = test(vecs[layer][:, [index[i] for i in ids]], y, strata, args.perms, rng)
            rows.append({"shuffle_within": "era" if flag is None else f"era x {name}", "layer": layer,
                         "separation": obs, "null_mean": null.mean(), "null_sd": null.std(), "z": z, "p": p})
            print(f"{rows[-1]['shuffle_within']:<40}{layer:<12} sep {obs:+.4f}  null {null.mean():+.4f}"
                  f" ± {null.std():.4f}  z {z:+.2f}  p {p:.4f}")

    for name, data in (("horror_composition.csv", comp), ("horror_check.csv", rows)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            for r in data:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
