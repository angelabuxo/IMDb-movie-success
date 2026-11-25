"""Do hit scripts resemble each other, and flop scripts each other, among films of the same genre, era and budget?

DECISIONS.md D29 (post hoc, criteria written before the five-seed run). Statistic, vectors and labels are those of
all_genres.py. Only the shuffle changes: labels move only between films of the same genre, era and budget group.
Budget groups are cut on all films of the genre x era cell (terciles; halves and quarters as a check).

Captures: whether two scripts of the same outcome are closer than two of different outcomes, beyond genre, era and
budget group. Misses: plot and order (mean vectors), films without a script, films under 10,000 votes, and budget
differences inside a group. Strata with only hits or only flops cannot move (share_movable says how many can).

Run:  IMDB_DATA_DIR=... W2V_VECTOR_DIR=... python -I analysis/screenplays/word2vec/budget_matched.py [--perms 2000]
Writes results/budget_matched.csv (main, halves, quarters, era only), budget_matched_summary.csv,
budget_matched_checks.csv, and
results/budget_matched_words.csv if a counts file is found (numbers and single words only).
"""
import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "words"))
import all_genres as ag  # noqa: E402
from final_set import final_films, read_table  # noqa: E402
from genre_cells import MIN_GROUP, label, pairs_for  # noqa: E402

GROUPINGS = {"era only": 1, "era x budget halves": 2, "era x budget terciles": 3, "era x budget quarters": 4}
MAIN = "era x budget terciles"
SMALL_CHECK = 20


def write(name, rows):
    with open(ag.OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()} for r in rows)


def load():
    index, vecs = ag.load_vectors()
    budget = {r["imdb_id"]: float(r["budget"]) for r in read_table() if r["budget"]}
    lab, _ = label(pairs_for([f for f in final_films() if f["id"] in index], "a"))
    cells, by = defaultdict(list), defaultdict(list)
    for f, g, e, _ in lab:
        cells[g, e].append(budget[f["id"]])
    for f, g, e, lb in lab:
        if lb != "middle":
            grp = {k: int(np.searchsorted(np.quantile(cells[g, e], np.arange(1, k) / k), budget[f["id"]]))
                   for k in set(GROUPINGS.values())}
            by[g].append({"id": f["id"], "era": e, "tier": f["tier"], "bud": grp, "hit": int(lb == "hit")})
    keep = {g: v for g, v in by.items() if min(sum(t["hit"] for t in v), sum(1 - t["hit"] for t in v)) >= MIN_GROUP}
    return index, vecs, keep


def strata(items, key):
    st = defaultdict(list)
    for k, t in enumerate(items):
        st[key(t)].append(k)
    return [np.array(v) for v in st.values()]


def run(items, key, layer, index, vecs, perms, seed):
    y = np.array([t["hit"] for t in items])
    st = strata(items, key)
    obs, _, null, p, z = ag.test(vecs[layer][:, [index[t["id"]] for t in items]], y, st, perms,
                                 np.random.default_rng(seed))
    return {"layer": layer, "hits": int(y.sum()), "flops": int(len(y) - y.sum()),
            "share_movable": float(sum(len(s) for s in st if 0 < y[s].sum() < len(s)) / len(y)),
            "separation": float(obs), "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, .95)),
            "z": float(z), "p": float(p)}


def words(genre, items, perms):
    import words as W
    path = W.PRIVATE / "counts.pkl"
    if not path.exists():
        path = ag.DATA / "features" / "script_w2v" / "script_counts.pkl"
    with open(path, "rb") as f:
        counts = pickle.load(f)
    finals = [f["id"] for f in final_films()]
    ids, y = [t["id"] for t in items], np.array([t["hit"] for t in items], float)
    cells = [(t["era"], t["bud"][3]) for t in items]
    out = []
    for layer in ag.LAYERS:
        vocab = sorted({w for i in finals for w in counts[i][layer]})
        ix = {w: k for k, w in enumerate(vocab)}
        prior = np.asarray(W.matrix(counts, finals, ix, layer).sum(axis=0)).ravel()
        C = W.matrix(counts, ids, ix, layer)
        keep = np.where(np.asarray((C > 0).sum(axis=0)).ravel() >= W.MIN_DF)[0]
        C, a = C[:, keep], prior[keep]
        z = W.z_for(C, y, a)
        null = np.array([W.z_for(C, yp, a) for yp in W.cell_permutations(cells, y, perms, np.random.default_rng(ag.SEED))])
        mu, sd = null.mean(0), null.std(0, ddof=1)
        zc, nc = (z - mu) / sd, np.abs((null - mu) / sd).max(1)
        for j in np.argsort(-np.abs(zc))[:20]:
            out.append({"genre": genre, "layer": layer, "word": vocab[keep[j]], "direction": "hit" if zc[j] > 0 else "flop",
                        "calibrated_z": float(zc[j]), "family_p": float((1 + (nc >= abs(zc[j])).sum()) / (1 + perms)),
                        "hit_films": int((C[y == 1, j] > 0).sum()), "flop_films": int((C[y == 0, j] > 0).sum()),
                        "family_threshold": float(np.quantile(nc, .95)), "words_tested": len(keep)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=2000)
    a = ap.parse_args()
    index, vecs, by = load()
    print("seeds:", [p.stem for p in ag.VECTORS], "genres:", len(by))
    write("budget_matched_summary.csv", [{"films": len({t["id"] for v in by.values() for t in v}), "genres": len(by),
                                          "perms": a.perms, "seeds": len(ag.VECTORS)}])
    rows = []
    for name, k in GROUPINGS.items():
        fam = [{"shuffle_within": name, "genre": g, **run(by[g], lambda t: (t["era"], t["bud"][k]), layer, index, vecs,
                                                          a.perms, ag.SEED + k)}
               for g in sorted(by) for layer in ag.LAYERS]
        for r, h in zip(fam, ag.holm(np.array([r["p"] for r in fam]))):
            r["p_holm"] = float(h)
        rows += fam
        for r in fam:
            print(name, r["genre"], r["layer"], f"sep {r['separation']:.4f} null95 {r['null_q95']:.4f} z {r['z']:.2f} "
                  f"p {r['p']:.4f} holm {r['p_holm']:.4f} movable {r['share_movable']:.2f}", flush=True)
    write("budget_matched.csv", rows)
    passed = sorted({r["genre"] for r in rows if r["shuffle_within"] == MAIN and r["p_holm"] < .05})
    print("passing the main test:", passed)
    fr, checks = ag.likely_franchise(), []
    bt = lambda t: (t["era"], t["bud"][3])  # noqa: E731
    for g in passed:
        for name, keep, key in (("no franchise", lambda t: t["id"] not in fr, bt),
                                ("100k+ only", lambda t: t["tier"] == "100k+", bt),
                                ("terciles x vote tier", lambda t: True, lambda t: (*bt(t), t["tier"]))):
            items = [t for t in by[g] if keep(t)]
            for layer in ag.LAYERS:
                r = {"genre": g, "check": name, **run(items, key, layer, index, vecs, a.perms, ag.SEED + 99)}
                r["too_small"] = min(r["hits"], r["flops"]) < SMALL_CHECK
                checks.append(r)
        for s in range(len(ag.VECTORS)):
            for layer in ag.LAYERS:
                checks.append({"genre": g, "check": f"main, {ag.VECTORS[s].stem} alone",
                               **run(by[g], bt, layer, index, {k: v[s:s + 1] for k, v in vecs.items()}, a.perms,
                                     ag.SEED + 3), "too_small": False})
    if checks:
        write("budget_matched_checks.csv", checks)
        for r in checks:
            print(*r.values())
        write("budget_matched_words.csv", sum((words(g, by[g], 1000) for g in passed), []))


if __name__ == "__main__":
    main()
