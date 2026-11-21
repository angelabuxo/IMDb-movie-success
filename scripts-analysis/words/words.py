"""Which words separate hits from flops (follow-up requested on 2026-11-21). The Scattertext statistic, hand-rolled.

Scattertext 0.2.2 pins scipy 1.13, which doesn't build on Python 3.13, so this computes the statistic Scattertext
offers as LogOddsRatioInformativeDirichletPrior: the log-odds ratio with an informative Dirichlet prior (Monroe,
Colaresi and Quinn 2008, "Fightin' Words", eq. 22), z = delta / sqrt(var).
  y_w^h, y_w^f  count of word w in all hit films / all flop films of the genre (that layer)
  a_w           count of w in the layer over the 1598 films of the final set (the prior), a0 = sum of a_w
  delta_w = log((y^h + a)/(n^h + a0 - y^h - a)) - log((y^f + a)/(n^f + a0 - y^f - a)),  var = 1/(y^h+a) + 1/(y^f+a)
Positive z = more typical of hits.

Same inputs as the Word2Vec tests: the counts in private/counts.pkl (corpus.py, T = 20 names filter), the hit/flop
labels of run_tests.labelled (1598-film set, genre x era quartiles), per layer.

The z treats tokens as independent, and they aren't (a word repeated in one film counts many times), so the test
is a film-level label shuffle: labels are permuted within genre x era cells (hits = flops in each cell by design)
and z is recomputed. A word used heavily by a few films gets a wide shuffle distribution, so the lists are ranked
by the calibrated z, (z - shuffle mean) / shuffle sd of that word; the Monroe-z ranking is written too.
  * per word: shuffle p = (1 + #perms with z >= z_obs) / (1 + perms) in the word's own direction;
  * per list: the k-th largest calibrated z under the shuffle against the observed k-th largest, and the
    largest |calibrated z| over all words (a family-wise check).
label_token_in_films counts the films whose <character> labels contain the word. A word with 20 or more is kept by
the T = 20 names filter, so a high count marks a likely name (nick, steve) that the filter lets through.
Words in fewer than MIN_DF of the genre's labelled films are left out, fixed before looking at any list.
For each listed word, the number of hit and flop films that contain it is reported, as is the same z in the
100k+ tier only (most hits are 100k+, so a word could track the tier).

Outputs (numbers and single words only, no script text):
  results/top_<genre>_<layer>.csv    top TOP words per direction
  results/list_null.csv              observed vs shuffled k-th largest z
With --kwic, also writes private/kwic/<genre>_<layer>.txt (gitignored): for each listed word, one line of at most
15 words from each of up to KWIC_FILMS different films, for reading the words in context. Never commit these.

Run:  IMDB_DATA_DIR=... python -I scripts-analysis/words/words.py [--perms 1000] [--kwic]
"""
import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for d in ("word2vec", "genre_counts", "preprocess", "data_check"):
    sys.path.insert(0, str(ROOT / d))
from final_set import final_films  # noqa: E402
from genre_cells import era  # noqa: E402
from run_tests import labelled  # noqa: E402

PRIVATE = ROOT / "private"
OUT = HERE / "results"
LAYERS = ("dialogue", "description")
GENRES = ("Action", "Comedy")
MIN_DF = 10          # word must appear in at least this many of the genre's labelled films
TOP = 20
KS = (1, 5, 10, 20)
KWIC_FILMS = 8
KWIC_WORDS = 15
PERM_SEED = 2805


def matrix(counts, ids, vocab_index, layer):
    rows, cols, vals = [], [], []
    for r, fid in enumerate(ids):
        for w, c in counts[fid][layer].items():
            j = vocab_index.get(w)
            if j is not None:
                rows.append(r)
                cols.append(j)
                vals.append(c)
    return sparse.csr_matrix((np.array(vals, float), (rows, cols)), shape=(len(ids), len(vocab_index)))


def log_odds_z(yh, yf, a):
    a0 = a.sum()
    nh, nf = yh.sum(axis=-1, keepdims=True), yf.sum(axis=-1, keepdims=True)
    d = np.log((yh + a) / (nh + a0 - yh - a)) - np.log((yf + a) / (nf + a0 - yf - a))
    return d / np.sqrt(1 / (yh + a) + 1 / (yf + a))


def z_for(C, y, a):
    yh = np.asarray(C.T @ y).ravel()
    yf = np.asarray(C.T @ (1 - y)).ravel()
    return log_odds_z(yh, yf, a)


def cell_permutations(cells, y, perms, rng):
    """Permute labels within each genre x era cell."""
    groups = defaultdict(list)
    for i, c in enumerate(cells):
        groups[c].append(i)
    groups = [np.array(g) for g in groups.values()]
    for _ in range(perms):
        yp = y.copy()
        for g in groups:
            yp[g] = y[rng.permutation(g)]
        yield yp


def setup(genre, tier=None):
    rows = labelled(genre, tier=tier)
    ids = [fl["id"] for fl, _ in rows]
    y = np.array([lb == "hit" for _, lb in rows], float)
    cells = [era(fl["decade"]) for fl, _ in rows]
    return ids, y, cells


_PARSED = {}


def parsed_films():
    """{imdb_id: parsed elements} for every script, plus the common-label set (as in corpus.py). Cached."""
    if _PARSED:
        return _PARSED["films"], _PARSED["common"], _PARSED["label_df"]
    from genre_counts import DATA
    from collections import Counter
    from parse_layers import one_per_id, parse_script, read_records
    from preprocess import NAME_MAX_FILMS, label_tokens
    label_df, parsed = Counter(), {}
    for r in one_per_id(read_records(DATA)):
        els = parse_script(r["script"])
        label_df.update(label_tokens(els))
        parsed[r["imdb_id"]] = els
    _PARSED["films"] = parsed
    _PARSED["common"] = {t for t, c in label_df.items() if c >= NAME_MAX_FILMS}
    _PARSED["label_df"] = label_df
    return parsed, _PARSED["common"], label_df


def kwic(layer, words, ids, y, path):
    """One line of at most KWIC_WORDS words per film, from up to KWIC_FILMS films per word, half hits, half flops."""
    from preprocess import film_layers
    parsed, common, _ = parsed_films()
    lab = dict(zip(ids, y))
    rng = np.random.default_rng(PERM_SEED)
    order = list(ids)
    rng.shuffle(order)
    layers = {fid: film_layers(parsed[fid], common)[0][layer] for fid in order}
    half = KWIC_WORDS // 2
    with open(path, "w") as f:
        for w in words:
            f.write(f"== {w}\n")
            shown = {1.0: 0, 0.0: 0}
            rng.shuffle(order)                  # a fresh film order per word, so the same films don't recur
            for fid in order:
                if shown[lab[fid]] >= KWIC_FILMS // 2:
                    continue
                for el in layers[fid]:
                    if w in el:
                        i = el.index(w)
                        start = max(0, i - half)
                        line = " ".join(el[start:start + KWIC_WORDS])
                        f.write(f"{'hit ' if lab[fid] else 'flop'} {fid}  {line}\n")
                        shown[lab[fid]] += 1
                        break
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--kwic", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    with open(PRIVATE / "counts.pkl", "rb") as f:
        counts = pickle.load(f)
    final_ids = [fl["id"] for fl in final_films()]
    _, _, label_df = parsed_films()
    list_rows = []
    for layer in LAYERS:
        vocab = sorted({w for fid in final_ids for w in counts[fid][layer]})
        index = {w: i for i, w in enumerate(vocab)}
        prior = np.asarray(matrix(counts, final_ids, index, layer).sum(axis=0)).ravel()
        for genre in GENRES:
            ids, y, cells = setup(genre)
            C = matrix(counts, ids, index, layer)
            dfh = np.asarray((C[y == 1] > 0).sum(axis=0)).ravel()
            dff = np.asarray((C[y == 0] > 0).sum(axis=0)).ravel()
            keep = np.where(dfh + dff >= MIN_DF)[0]
            C, a_w = C[:, keep], prior[keep]
            z = z_for(C, y, a_w)
            rng = np.random.default_rng(PERM_SEED)
            null = np.array([z_for(C, yp, a_w) for yp in cell_permutations(cells, y, a.perms, rng)])
            mu, sd = null.mean(axis=0), null.std(axis=0, ddof=1)
            zc = (z - mu) / sd                      # z against this word's own shuffle distribution
            null_c = (null - mu) / sd
            p_up = (1 + (null >= z).sum(axis=0)) / (1 + a.perms)
            p_dn = (1 + (null <= z).sum(axis=0)) / (1 + a.perms)
            ids_t, y_t, _ = setup(genre, tier="100k+")
            C_t = matrix(counts, ids_t, index, layer)[:, keep]
            z_t = z_for(C_t, y_t, a_w)
            out = []
            for direction, sign, p in (("hit", 1, p_up), ("flop", -1, p_dn)):
                srt = np.sort(sign * null_c, axis=1)[:, ::-1]
                obs_sorted = np.sort(sign * zc)[::-1]
                for k in KS:
                    nk = srt[:, k - 1]
                    list_rows.append({"genre": genre, "layer": layer, "direction": direction, "k": k,
                                      "observed_kth_calibrated_z": round(obs_sorted[k - 1], 3),
                                      "shuffle_mean_kth": round(nk.mean(), 3),
                                      "shuffle_q95_kth": round(np.quantile(nk, .95), 3),
                                      "shuffle_p": round((1 + (nk >= obs_sorted[k - 1]).sum()) / (1 + a.perms), 4),
                                      "perms": a.perms, "words_tested": len(keep)})
                for rank_by, order in (("calibrated", np.argsort(-sign * zc)), ("monroe_z", np.argsort(-sign * z))):
                    for rank, j in enumerate(order[:TOP], 1):
                        w = vocab[keep[j]]
                        out.append({"ranked_by": rank_by, "direction": direction, "rank": rank, "word": w,
                                    "calibrated_z": round(zc[j], 3), "monroe_z": round(z[j], 3),
                                    "shuffle_p": round(p[j], 4),
                                    "hit_films_with_word": int(dfh[keep[j]]),
                                    "flop_films_with_word": int(dff[keep[j]]),
                                    "count_hits": int(C[y == 1, j].sum()), "count_flops": int(C[y == 0, j].sum()),
                                    "label_token_in_films": label_df.get(w, 0),
                                    "monroe_z_100k_only": round(z_t[j], 3)})
            fw = np.quantile(np.abs(null_c).max(axis=1), .95)
            list_rows.append({"genre": genre, "layer": layer, "direction": "either", "k": "max |calibrated z|",
                              "observed_kth_calibrated_z": round(np.abs(zc).max(), 3), "shuffle_mean_kth": "",
                              "shuffle_q95_kth": round(fw, 3),
                              "shuffle_p": round((1 + (np.abs(null_c).max(axis=1) >= np.abs(zc).max()).sum())
                                                 / (1 + a.perms), 4),
                              "perms": a.perms, "words_tested": len(keep)})
            with open(OUT / f"top_{genre}_{layer}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(out[0]))
                w.writeheader()
                w.writerows(out)
            cal = [r for r in out if r["ranked_by"] == "calibrated"]
            print(f"{genre} {layer}: {len(keep)} words, n {int(y.sum())}/{int(len(y) - y.sum())}, FWER |zc| {fw:.2f}; "
                  f"hit {[r['word'] for r in cal[:10]]}; flop {[r['word'] for r in cal[TOP:TOP + 10]]}", flush=True)
            if a.kwic:
                (PRIVATE / "kwic").mkdir(exist_ok=True)
                kwic(layer, sorted({r["word"] for r in out}), ids, y, PRIVATE / "kwic" / f"{genre}_{layer}.txt")
    with open(OUT / "list_null.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(list_rows[0]))
        w.writeheader()
        w.writerows(list_rows)


if __name__ == "__main__":
    main()
