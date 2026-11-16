"""Tags from what happens in the film, not from its audience.

Run from the data folder (data/raw) after plot_embeddings.py, tag_shares.py
and build_graph.py:

    cd data/raw && ../../.venv/bin/python ../../scripts/reviews/plot_tags.py

tag_profit.py found that the Tag Genome scores carry the size of a film's
audience. Here the same tags are predicted from the film's Wikipedia plot
summary alone: a ridge regression from the plot's embedding (384 numbers) to
the 1,084 Tag Genome scores, trained on the films that have both. The plot
says what happens in the film. It doesn't say how many people saw it.

A. How well a plot recovers each tag. Five-fold: every scored film's tags are
   predicted by a model that never saw that film. Pearson r between predicted
   and Tag Genome score per tag, by tag class, and the AUC of the predicted
   genre relevance against IMDb's genre labels, both for scored films and for
   the films after 2014 that the Tag Genome doesn't cover.
B. Files for every film with a plot: features/plot_tag_relevance.npy (out of
   fold for scored films, the full model for the rest, clipped to 0..1), its
   ids and tags, and features/plot_genre_shares.csv.
C. The leak check: how well the plot embedding predicts ln(IMDb votes), net
   of budget and decade, next to the Tag Genome's content tags.
D. Return. Out-of-sample R2 of the adjusted return (ln(gross / budget) net of
   ln(budget) and decade) from the plot embedding, the plot-predicted content
   tags and IMDb's genre labels: random folds, trained before SPLIT_YEAR and
   tested from it, and trained to 2014 and tested on 2015 onwards. On the
   scored films, the Tag Genome's content tags on the same folds. Per tag, rho
   of the plot-predicted relevance with the adjusted return, against the Tag
   Genome's rho for the same tag.
E. Screenplays as a second pre-release source, on the films that have one:
   the same recovery and return tests from the screenplay embedding.

What it misses: a plot summary is written after release and a famous film
gets a longer one, so the word count is checked too. A mean embedding keeps
topic and setting, not execution. A predicted tag is only as good as its r in
part A, and a reception tag predicted from a plot is a guess at reception.

Writes analysis/tags/plot_tags.json and plot_tags_tags.csv: numbers only.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from tag_profit import HIGH, SPLIT_YEAR, adjust, cv_r2, rho_with_null, split_fit, split_r2, tag_class, zrank
from tag_shares import GENRE_TAG, GENRES, effective_number, shares

NODES = "features/graph/nodes.csv"
OUT = Path(__file__).resolve().parents[2] / "analysis" / "tags"
FIRST_YEAR = 1975
LAST_SCORED_YEAR = 2014
ALPHAS = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
FOLDS = 5
SEED = 0


def standardise(x, ref):
    mu, sd = ref.mean(axis=0), ref.std(axis=0)
    return (x - mu) / np.where(sd == 0, 1, sd)


def pick_alpha(x, y, rng):
    """Ridge penalty with the best mean R2 on a held-out fifth."""
    idx = rng.permutation(len(x))
    te, tr = idx[:len(x) // 5], idx[len(x) // 5:]
    best, score = ALPHAS[0], -np.inf
    for a in ALPHAS:
        p = Ridge(alpha=a).fit(standardise(x[tr], x[tr]), y[tr]).predict(standardise(x[te], x[tr]))
        s = 1 - ((y[te] - p) ** 2).sum(axis=0).sum() / ((y[te] - y[tr].mean(axis=0)) ** 2).sum(axis=0).sum()
        if s > score:
            best, score = a, s
    return best, float(score)


def out_of_fold(x, y, alpha, rng):
    pred = np.empty_like(y, dtype=float)
    for tr, te in KFold(FOLDS, shuffle=True, random_state=int(rng.integers(1 << 31))).split(x):
        pred[te] = Ridge(alpha=alpha).fit(standardise(x[tr], x[tr]), y[tr]).predict(standardise(x[te], x[tr]))
    return pred


def col_r(a, b):
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    return (a * b).sum(axis=0) / np.where(den == 0, 1, den)


def recovery(pred, truth, cls):
    r = col_r(pred, truth)
    return r, {c: {"tags": int((cls == c).sum()), "median_r": float(np.median(r[cls == c])),
                   "p10_r": float(np.percentile(r[cls == c], 10)), "tags_r_over_0.5": int((r[cls == c] > 0.5).sum())}
               for c in ("content", "reception", "commercial")}


def main():
    rng = np.random.default_rng(SEED)
    plot = np.load("features/plot_vectors.npy").astype(float)
    pid = pd.read_csv("features/plot_vector_ids.csv")
    genome = np.load("features/tag_relevance.npy").astype(float)
    gid = pd.read_csv("features/tag_relevance_ids.csv")["tconst"].tolist()
    tags = pd.read_csv("features/tag_relevance_tags.csv", keep_default_na=False)["tag"].tolist()
    cls = np.array([tag_class(t) for t in tags])
    g_cols = [tags.index(GENRE_TAG[g]) for g in GENRES]

    movies = {}
    with open(NODES, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["type"] == "movie" and r["year"]:
                movies[r["id"]] = r
    keep = pid["tconst"].isin(movies).to_numpy()
    plot, pid = plot[keep], pid[keep].reset_index(drop=True)
    P = pid["tconst"].tolist()
    prow = {t: i for i, t in enumerate(P)}
    grow = {t: i for i, t in enumerate(gid)}
    year_all = np.array([int(float(movies[t]["year"])) for t in P])
    labels = np.array([[g in movies[t]["genres"].split(",") for g in GENRES] for t in P])
    both = np.array([t in grow for t in P])
    xb, yb = plot[both], genome[[grow[t] for t in np.array(P)[both]]]

    res = {"films_with_plot": len(P), "films_with_plot_and_tag_scores": int(both.sum()), "films_in_graph": len(movies),
           "plot_words": {"median": float(pid["words"].median()), "p10": float(pid["words"].quantile(0.1)), "p90": float(pid["words"].quantile(0.9))},
           "headings": pid["heading"].value_counts().to_dict()}

    # ---- A. how well a plot recovers each tag
    alpha, val = pick_alpha(xb, yb, rng)
    oof = out_of_fold(xb, yb, alpha, rng)
    r_tag, by_class = recovery(oof, yb, cls)
    full = Ridge(alpha=alpha).fit(standardise(xb, xb), yb)
    pred = full.predict(standardise(plot, xb))
    pred[both] = oof
    pred = pred.clip(0, 1)
    auc = {}
    late = ~both & (year_all > LAST_SCORED_YEAR)
    for j, g in enumerate(GENRES):
        yb_l, late_l = labels[both, j], labels[late, j]
        auc[g] = {"plot_scored_films": float(roc_auc_score(yb_l, pred[both, g_cols[j]])) if yb_l.sum() >= 20 else None,
                  "tag_genome_scored_films": float(roc_auc_score(yb_l, yb[:, g_cols[j]])) if yb_l.sum() >= 20 else None,
                  "plot_films_after_%d" % LAST_SCORED_YEAR: float(roc_auc_score(late_l, pred[late, g_cols[j]])) if late_l.sum() >= 20 else None,
                  "imdb_films_after": int(late_l.sum())}
    mean = lambda k: float(np.mean([v[k] for v in auc.values() if v[k] is not None]))
    sh = shares(pred[:, g_cols])
    top_hit = lambda m: float(labels[m][np.arange(m.sum()), sh[m].argmax(axis=1)].mean())
    res["recovery"] = {
        "alpha": alpha, "held_out_r2_all_tags": val, "by_class": by_class,
        "best_tags": [[tags[j], float(r_tag[j])] for j in np.argsort(-r_tag)[:15]],
        "worst_tags": [[tags[j], float(r_tag[j])] for j in np.argsort(r_tag)[:15]],
        "genre_tags_r": {g: float(r_tag[c]) for g, c in zip(GENRES, g_cols)},
        "genre_auc": auc, "genre_auc_mean": {k: mean(k) for k in next(iter(auc.values())) if k != "imdb_films_after"},
        "films_after_%d" % LAST_SCORED_YEAR: int(late.sum()),
        "top_share_genre_in_imdb": {"scored_films": top_hit(both), "films_after": top_hit(late)},
        "top_genre_agrees_with_tag_genome": float((sh[both].argmax(axis=1) == shares(yb[:, g_cols]).argmax(axis=1)).mean()),
    }
    print(json.dumps({k: v for k, v in res["recovery"].items() if k != "genre_auc"}, indent=1))

    # ---- B. files
    np.save("features/plot_tag_relevance.npy", pred.astype(np.float32))
    pd.DataFrame({"tconst": P, "out_of_fold": both.astype(int)}).to_csv("features/plot_tag_relevance_ids.csv", index=False)
    out = pd.DataFrame({"tconst": P, "primaryTitle": [movies[t]["label"] for t in P], "y": year_all,
                        "genres": [movies[t]["genres"] for t in P], "has_tag_genome": both.astype(int)})
    for j, g in enumerate(GENRES):
        out["rel_" + g] = pred[:, g_cols[j]].round(4)
    for j, g in enumerate(GENRES):
        out["share_" + g] = sh[:, j].round(4)
    out["top_genre"] = [GENRES[j] for j in sh.argmax(axis=1)]
    out["effective_genres"] = effective_number(sh).round(3)
    out.to_csv("features/plot_genre_shares.csv", index=False)

    # ---- the return sample
    clean = np.array([movies[t]["clean"] == "1" and bool(movies[t]["multiple"]) and year_all[i] >= FIRST_YEAR for i, t in enumerate(P)])
    S = np.flatnonzero(clean)
    year = year_all[S]
    budget = np.array([float(movies[P[i]]["budget_usd"]) for i in S])
    gross = np.array([float(movies[P[i]]["gross_usd"]) for i in S])
    ln_votes = np.log(np.array([float(movies[P[i]]["votes"]) for i in S]))
    ln_mult, ln_budget = np.log(gross / budget), np.log(budget)
    adj = adjust(ln_mult, ln_budget, year // 10)
    adj_v = adjust(ln_mult, ln_budget, year // 10, ln_votes)
    xs, lab_s, pred_s = plot[S], labels[S].astype(float), pred[S]
    scored_s = both[S]
    words = np.log(pid["words"].to_numpy(float)[S])

    # ---- C. the leak check
    votes_net = adjust(ln_votes, ln_budget, year // 10)
    gen_s = genome[[grow[P[i]] for i in S[scored_s]]]
    res["leak"] = {
        "films": int(len(S)), "films_scored": int(scored_s.sum()),
        "r2_votes_from_plot_embedding": cv_r2(xs, votes_net, rng, repeats=1),
        "r2_votes_from_plot_content_tags": cv_r2(pred_s[:, cls == "content"], votes_net, rng, repeats=1),
        "r2_votes_from_plot_embedding_scored_films": cv_r2(xs[scored_s], votes_net[scored_s], rng, repeats=1),
        "r2_votes_from_tag_genome_content_tags": cv_r2(gen_s[:, cls == "content"], votes_net[scored_s], rng, repeats=1),
        "r2_votes_from_imdb_genre_labels": cv_r2(lab_s, votes_net, rng, repeats=1),
        "rho_plot_words_with_votes": float(zrank(words[:, None])[:, 0] @ zrank(ln_votes[:, None])[:, 0] / len(S)),
        "rho_plot_words_with_adjusted": float(zrank(words[:, None])[:, 0] @ zrank(adj[:, None])[:, 0] / len(S)),
    }
    print(json.dumps(res["leak"], indent=1))

    # ---- D. return
    # In the splits in time the outcome stays the adjusted return fitted on all films (see tag_profit.py).
    def time_y(mask_train, v=False):
        return adj_v if v else adj

    early, pre15 = year < SPLIT_YEAR, year <= LAST_SCORED_YEAR
    sets = {"plot embedding": xs, "plot-predicted content tags": pred_s[:, cls == "content"], "IMDb genre labels": lab_s,
            "plot embedding + IMDb genre labels": np.column_stack([xs, lab_s])}
    ret = {"films": int(len(S)), "films_from_%d" % SPLIT_YEAR: int((~early).sum()), "films_after_%d" % LAST_SCORED_YEAR: int((~pre15).sum()), "sets": {}}
    for name, x in sets.items():
        null = [cv_r2(x, rng.permutation(adj), rng, repeats=1) for _ in range(5)]
        ret["sets"][name] = {
            "features": int(x.shape[1]), "random_folds": cv_r2(x, adj, rng, repeats=2), "null_max": float(np.max(null)),
            "trained_before_%d" % SPLIT_YEAR: split_r2(x, time_y(early), early),
            "r_trained_before_%d" % SPLIT_YEAR: split_fit(x, adj, early)[1],
            "trained_to_%d_tested_after" % LAST_SCORED_YEAR: split_r2(x, time_y(pre15), pre15),
            "r_trained_to_%d_tested_after" % LAST_SCORED_YEAR: split_fit(x, adj, pre15)[1],
            "votes_fixed_random_folds": cv_r2(x, adj_v, rng, repeats=1),
            "votes_fixed_trained_before_%d" % SPLIT_YEAR: split_r2(x, time_y(early, True), early)}
        print(name, ret["sets"][name])
    # same films, same test: the Tag Genome's content tags against the plot
    ys, es = adj[scored_s], early[scored_s]
    ty, tyv = time_y(early)[scored_s], time_y(early, True)[scored_s]
    ret["scored_films"] = {"films": int(scored_s.sum())}
    for name, x in (("Tag Genome content tags", gen_s[:, cls == "content"]), ("plot embedding", xs[scored_s]),
                    ("plot-predicted content tags", pred_s[scored_s][:, cls == "content"]), ("IMDb genre labels", lab_s[scored_s])):
        ret["scored_films"][name] = {"random_folds": cv_r2(x, ys, rng, repeats=1), "trained_before_%d" % SPLIT_YEAR: split_r2(x, ty, es),
                                     "r_trained_before_%d" % SPLIT_YEAR: split_fit(x, ty, es)[1],
                                     "votes_fixed_trained_before_%d" % SPLIT_YEAR: split_r2(x, tyv, es)}
        print("scored films:", name, ret["scored_films"][name])
    # per tag
    rho_plot, p_plot, fw = rho_with_null(pred_s, adj, rng, 500)
    rho_gen, _, _ = rho_with_null(gen_s, ys, rng, 1)
    rho_plot_sc, _, _ = rho_with_null(pred_s[scored_s], ys, rng, 1)
    cont = cls == "content"
    usable = cont & (r_tag > 0.5)
    ret["per_tag"] = {
        "familywise_95": fw, "content_tags_beyond_familywise": int((np.abs(rho_plot[cont]) > fw).sum()),
        "median_abs_rho_content": float(np.median(np.abs(rho_plot[cont]))), "largest_abs_rho_content": float(np.abs(rho_plot[cont]).max()),
        "content_tags_recovered_r_over_0.5": int(usable.sum()),
        "agreement_with_tag_genome": {"r_between_rhos_all_content": float(np.corrcoef(rho_plot_sc[cont], rho_gen[cont])[0, 1]),
                                      "r_between_rhos_recovered_content": float(np.corrcoef(rho_plot_sc[usable], rho_gen[usable])[0, 1]),
                                      "same_sign_recovered_content": float((np.sign(rho_plot_sc[usable]) == np.sign(rho_gen[usable])).mean()),
                                      "median_abs_rho_tag_genome": float(np.median(np.abs(rho_gen[usable]))),
                                      "median_abs_rho_plot": float(np.median(np.abs(rho_plot_sc[usable])))},
        "top_positive": [[tags[j], float(rho_plot[j]), float(r_tag[j])] for j in np.argsort(-np.where(usable, rho_plot, -9))[:12]],
        "top_negative": [[tags[j], float(rho_plot[j]), float(r_tag[j])] for j in np.argsort(np.where(usable, rho_plot, 9))[:12]],
    }
    sh_s = sh[S]
    rho_sh, _, fw_sh = rho_with_null(sh_s, adj, rng, 500)
    rho_lab, _, _ = rho_with_null(lab_s, adj, rng, 1)
    ret["genre_shares"] = {"familywise_95": fw_sh, "per_genre": {g: {"rho_plot_share": float(a), "rho_imdb_label": float(b), "films_high": int((pred_s[:, c] >= HIGH).sum())}
                                                                  for g, a, b, c in zip(GENRES, rho_sh, rho_lab, g_cols)}}
    res["return"] = ret
    print(json.dumps(ret["per_tag"], indent=1))
    pd.DataFrame({"tag": tags, "class": cls, "recovery_r": r_tag.round(4), "rho_adjusted_plot": rho_plot.round(4),
                  "rho_adjusted_plot_scored_films": rho_plot_sc.round(4), "rho_adjusted_tag_genome": rho_gen.round(4)}
                 ).to_csv(OUT / "plot_tags_tags.csv", index=False)

    # ---- E. screenplays
    sv = np.load("features/embeddings_hpc/script_vectors.npy").astype(float)
    sid = pd.read_csv("features/embeddings_hpc/ids.csv")["tconst"].tolist()
    srow = {}
    for i, t in enumerate(sid):
        srow.setdefault(t, i)
    have = [t for t in sid if t in grow and t in srow]
    have = list(dict.fromkeys(have))
    xsr, ysr = sv[[srow[t] for t in have]], genome[[grow[t] for t in have]]
    a_s, _ = pick_alpha(xsr, ysr, rng)
    r_s, by_class_s = recovery(out_of_fold(xsr, ysr, a_s, rng), ysr, cls)
    in_plot = [t for t in have if t in prow]
    xp2 = plot[[prow[t] for t in in_plot]]
    yp2 = genome[[grow[t] for t in in_plot]]
    a_p, _ = pick_alpha(xp2, yp2, rng)
    r_p2, by_class_p2 = recovery(out_of_fold(xp2, yp2, a_p, rng), yp2, cls)
    scr = {"films_with_script_and_tag_scores": len(have), "recovery_by_class": by_class_s,
           "plot_on_the_same_films": {"films": len(in_plot), "by_class": by_class_p2},
           "r_between_per_tag_recovery_of_script_and_plot": float(np.corrcoef(r_s, r_p2)[0, 1])}
    pos = {P[i]: k for k, i in enumerate(S)}
    cs = [t for t in dict.fromkeys(sid) if t in pos]
    k = np.array([pos[t] for t in cs])
    xsc, xpl = sv[[srow[t] for t in cs]], xs[k]
    e2 = early[k]
    ty2 = time_y(early)[k]
    scr["return"] = {"films": len(cs), "films_from_%d" % SPLIT_YEAR: int((~e2).sum())}
    for name, x in (("screenplay embedding", xsc), ("plot embedding", xpl), ("IMDb genre labels", lab_s[k])):
        null = [cv_r2(x, rng.permutation(adj[k]), rng, repeats=1) for _ in range(5)]
        scr["return"][name] = {"random_folds": cv_r2(x, adj[k], rng, repeats=2), "null_max": float(np.max(null)),
                               "trained_before_%d" % SPLIT_YEAR: split_r2(x, ty2, e2),
                               "r_trained_before_%d" % SPLIT_YEAR: split_fit(x, ty2, e2)[1]}
    res["screenplays"] = scr
    print(json.dumps(scr, indent=1))

    (OUT / "plot_tags.json").write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", OUT / "plot_tags.json")


if __name__ == "__main__":
    main()
