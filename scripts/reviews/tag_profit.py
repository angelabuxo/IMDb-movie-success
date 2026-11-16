"""Do a film's tags and genre mix go with its return or its revenue?

Run from the data folder (data/raw) after tag_shares.py and build_graph.py:

    cd data/raw && ../../.venv/bin/python ../../scripts/reviews/tag_profit.py

Sample: films flagged clean in features/graph/nodes.csv, released 1975 or
later, that have Tag Genome scores (features/tag_relevance.npy). The scores
stop in 2014, so this is a study of 1975 to 2014.

Outcomes, per film:
- gross: ln(worldwide gross), the revenue;
- multiple: ln(gross / budget). Not profit: budgets leave out marketing and
  cinemas keep part of the gross;
- budget: ln(budget), to show which tags simply go with expensive films;
- adjusted: ln(gross / budget) minus its OLS fit on ln(budget) and release
  decade. Because ln(gross) = ln(gross / budget) + ln(budget), adjusting
  ln(gross) for the same two things gives the same number, so once budget is
  held fixed there is one outcome, not two.

Leakage. Every tag score was predicted from what viewers tagged, rated and
wrote after release. Tags that state an opinion, an award or a list
("masterpiece", "boring", "oscar (best picture)", "imdb top 250") record the
reception, and "franchise", "sequel" and "big budget" record commercial facts.
RECEPTION and COMMERCIAL below class those by pattern; everything else is
"content". Even a content tag was measured after release, and films more
people saw have more tags, ratings and reviews behind their scores. So a tag
score can carry the size of the audience, which is the outcome. Part E tests
that: it adds a fifth outcome, adjusted_votes, the return net of ln(budget),
decade and ln(IMDb votes). Votes are partly a result of success, so holding
them fixed removes real effects along with the leak. The truth lies between
adjusted (leak left in) and adjusted_votes (too strict).

A. Every tag against every outcome: Spearman rho over films, with a
   Benjamini-Hochberg q over the 1,084 tags and a family-wise threshold, the
   95th percentile over shuffles of the largest |rho| any tag reaches when the
   outcome is shuffled across films.
B. Genre shares and the spread of a film's mix against the outcomes, the same way.
C. Combinations. For each pair of genres, films are split by whether each
   genre's relevance is at least HIGH. With enough films in all four cells,
   the interaction is the mean adjusted return of films high on both, minus
   what the two genres add separately. Null: the adjusted return shuffled
   across films, BH q over the pairs tested.
D. Prediction. Cross-validated R2 of a ridge regression of the adjusted return
   on: IMDb's genre labels, the 22 genre relevances, the content tags, and all
   tags. Random folds, and a split in time (train before SPLIT_YEAR, test from
   it on). In the split the outcome is the same adjusted return, whose budget
   slope and decade means are fitted on all films: both are known before
   release, and fitting them on the early films only would leave the later
   decades with a shifted mean that no content feature could predict. Null:
   the same with the outcome shuffled.

E. The audience-size check: how well the content tags predict ln(votes), how
   ln(votes) goes with the return, and parts A and D again on adjusted_votes.

Writes analysis/tags/tag_profit.json, tag_profit_tags.csv and
tag_profit_pairs.csv in the repo: numbers only.
"""

import csv
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

NODES = "features/graph/nodes.csv"
MATRIX = "features/tag_relevance.npy"
IDS = "features/tag_relevance_ids.csv"
TAGS = "features/tag_relevance_tags.csv"
GENRES_CSV = "features/genre_shares.csv"
OUT = Path(__file__).resolve().parents[2] / "analysis" / "tags"
FIRST_YEAR = 1975
SPLIT_YEAR = 2005
HIGH = 0.5
MIN_CELL = 30
N_NULL = 1000
N_NULL_PRED = 10
ALPHAS = np.logspace(0, 4, 9)
SEED = 0

# Tags that state how the film was received: opinions, awards, lists.
RECEPTION = re.compile(
    r"oscar|afi 100|imdb top 250|movielens top pick|golden palm|saturn award|criterion|"
    r"\bgood\b|\bgreat\b|\bbad\b|\bbest\b|better|boring|excellent|masterpiece|overrated|underrated|awesome|"
    r"brilliant|perfect|\blame\b|horrible|stupid|\bdumb\b|idiotic|pointless|waste of time|predictable|"
    r"disappointing|funn|hilarious|hillarious|classic|\bcult\b|amazing|stunning|beautiful|breathtaking|"
    r"interesting|entertaining|exciting|original|clever|intelligent|genius|unique|special|\bfun\b|"
    r"pretentious|ridiculous|shallow|cheesy|corny|\bcool\b|crappy|so bad|no plot|plot holes|too long|"
    r"unrealistic|touching|moving|powerful|heartwarming|heartbreaking|inspiring|inspirational|"
    r"thought-provoking|book was better|not as good|guilty pleasure|feel good|feel-good|quotable|"
    r"must see|\bvery\b|exceptional|notable|suprisingly|bullshit|sappy")
NOT_RECEPTION = {"classic car", "classical music", "good versus evil", "special effects"}
# Tags that state a commercial fact about the film.
COMMERCIAL = re.compile(r"franchise|sequel|prequel|trilogy|\bseries\b|big budget|low budget|remake|remade|"
                        r"made for tv|independent film|indie|blockbuster")


def tag_class(tag):
    return "reception" if RECEPTION.search(tag) and tag not in NOT_RECEPTION else "commercial" if COMMERCIAL.search(tag) else "content"


def zrank(a):
    """Columns ranked and standardised, so a product of two of them averages to Spearman's rho."""
    r = rankdata(a, axis=0)
    r = r - r.mean(axis=0)
    sd = r.std(axis=0)
    return r / np.where(sd == 0, 1, sd)


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def rho_with_null(x, y, rng, n_null=N_NULL):
    """Spearman rho of every column of x with y, a two-sided permutation p per column, and the
    95th percentile of the largest |rho| over columns under shuffles of y."""
    xr, yr = zrank(x), zrank(y[:, None])[:, 0]
    n = len(y)
    rho = xr.T @ yr / n
    exceed = np.zeros(x.shape[1])
    top = np.empty(n_null)
    for k in range(n_null):
        r = np.abs(xr.T @ rng.permutation(yr) / n)
        exceed += r >= np.abs(rho)
        top[k] = r.max()
    return rho, (exceed + 1) / (n_null + 1), float(np.percentile(top, 95))


def adjust(y, ln_budget, decade, *more):
    """y minus its OLS fit on ln(budget), decade dummies and any further columns."""
    decs = sorted(set(decade))[1:]
    x = np.column_stack([np.ones(len(y)), ln_budget] + [(decade == d).astype(float) for d in decs] + list(more))
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def cv_r2(x, y, rng, folds=5, repeats=3):
    """Mean out-of-fold R2 of a ridge regression, penalty chosen inside each training fold."""
    out = []
    for _ in range(repeats):
        pred = np.empty(len(y))
        for tr, te in KFold(folds, shuffle=True, random_state=int(rng.integers(1 << 31))).split(x):
            mu, sd = x[tr].mean(axis=0), x[tr].std(axis=0)
            sd = np.where(sd == 0, 1, sd)
            m = RidgeCV(alphas=ALPHAS).fit((x[tr] - mu) / sd, y[tr])
            pred[te] = m.predict((x[te] - mu) / sd)
        out.append(r2_score(y, pred))
    return float(np.mean(out))


def split_fit(x, y, train):
    """(R2, Pearson r) on the films outside `train` of a ridge regression fitted on the films in it."""
    mu, sd = x[train].mean(axis=0), x[train].std(axis=0)
    sd = np.where(sd == 0, 1, sd)
    pred = RidgeCV(alphas=ALPHAS).fit((x[train] - mu) / sd, y[train]).predict((x[~train] - mu) / sd)
    return float(r2_score(y[~train], pred)), float(np.corrcoef(y[~train], pred)[0, 1])


def split_r2(x, y, train):
    return split_fit(x, y, train)[0]


def main():
    rng = np.random.default_rng(SEED)
    rel_all = np.load(MATRIX)
    ids = pd.read_csv(IDS)["tconst"].tolist()
    tags = pd.read_csv(TAGS, keep_default_na=False)["tag"].tolist()
    gs = pd.read_csv(GENRES_CSV).set_index("tconst").reindex(ids)
    genres = [c[4:] for c in gs.columns if c.startswith("rel_")]

    film = {}
    with open(NODES, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["type"] == "movie" and r["clean"] == "1" and r["multiple"] and int(r["year"]) >= FIRST_YEAR:
                film[r["id"]] = (float(r["budget_usd"]), float(r["gross_usd"]), int(r["year"]), r["genres"], r["label"],
                                 float(r["votes"]))
    rows = [i for i, t in enumerate(ids) if t in film]
    keep = [ids[i] for i in rows]
    rel = rel_all[rows].astype(float)
    budget = np.array([film[t][0] for t in keep])
    gross = np.array([film[t][1] for t in keep])
    year = np.array([film[t][2] for t in keep])
    y = {"gross": np.log(gross), "multiple": np.log(gross / budget), "budget": np.log(budget)}
    y["adjusted"] = adjust(y["multiple"], y["budget"], year // 10)
    assert np.allclose(y["adjusted"], adjust(y["gross"], y["budget"], year // 10))
    ln_votes = np.log(np.array([film[t][5] for t in keep]))
    y["adjusted_votes"] = adjust(y["multiple"], y["budget"], year // 10, ln_votes)
    n = len(keep)
    cls = np.array([tag_class(t) for t in tags])

    res = {"sample": {"films": n, "clean_films_in_graph_from_%d" % FIRST_YEAR: len(film), "scored_films": len(ids),
                      "years": [int(year.min()), int(year.max())],
                      "grossed_under_budget": float((gross < budget).mean()),
                      "median_multiple": float(np.median(gross / budget)),
                      "note": "clean flag as stored in features/graph/nodes.csv"},
           "settings": {"high": HIGH, "min_cell": MIN_CELL, "shuffles": N_NULL, "split_year": SPLIT_YEAR},
           "tag_classes": {c: int((cls == c).sum()) for c in ("content", "reception", "commercial")},
           "reception_tags": sorted(t for t, c in zip(tags, cls) if c == "reception"),
           "commercial_tags": sorted(t for t, c in zip(tags, cls) if c == "commercial")}

    # ---- A. every tag against every outcome
    table = pd.DataFrame({"tag": tags, "class": cls, "median_relevance": np.median(rel, axis=0).round(4),
                          "films_high": (rel >= HIGH).sum(axis=0)})
    res["tags"] = {}
    for name, v in y.items():
        rho, p, fw = rho_with_null(rel, v, rng)
        q = bh(p)
        table["rho_" + name], table["q_" + name] = rho.round(4), q.round(4)
        res["tags"][name] = {
            "familywise_95": fw, "tags_beyond_familywise": int((np.abs(rho) > fw).sum()),
            "tags_q_below_0.05": int((q < 0.05).sum()),
            "beyond_familywise_by_class": {c: int(((np.abs(rho) > fw) & (cls == c)).sum()) for c in ("content", "reception", "commercial")},
            "largest_abs_rho_by_class": {c: float(np.abs(rho[cls == c]).max()) for c in ("content", "reception", "commercial")},
            "median_abs_rho": float(np.median(np.abs(rho))),
        }
        for c in ("content", "reception", "commercial"):
            sub = table[table["class"] == c].sort_values("rho_" + name)
            res["tags"][name]["top_" + c] = {
                "positive": [[t, float(r)] for t, r in zip(sub["tag"][::-1][:12], sub["rho_" + name][::-1][:12])],
                "negative": [[t, float(r)] for t, r in zip(sub["tag"][:12], sub["rho_" + name][:12])]}
    # tags that go with the return but not with the budget, and the reverse
    hi = table[(table["rho_adjusted"].abs() > res["tags"]["adjusted"]["familywise_95"]) & (table["class"] == "content")]
    res["tags"]["content_adjusted_dose"] = {}
    for t in list(hi.sort_values("rho_adjusted")["tag"][:4]) + list(hi.sort_values("rho_adjusted")["tag"][-4:]):
        j = tags.index(t)
        bins = [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]
        res["tags"]["content_adjusted_dose"][t] = [
            {"relevance": "%.1f-%.1f" % (a, min(b, 1)), "films": int(((rel[:, j] >= a) & (rel[:, j] < b)).sum()),
             "median_multiple": float(np.median((gross / budget)[(rel[:, j] >= a) & (rel[:, j] < b)])) if ((rel[:, j] >= a) & (rel[:, j] < b)).any() else None,
             "mean_adjusted": float(y["adjusted"][(rel[:, j] >= a) & (rel[:, j] < b)].mean()) if ((rel[:, j] >= a) & (rel[:, j] < b)).any() else None}
            for a, b in bins]
    table.to_csv(OUT / "tag_profit_tags.csv", index=False)

    # ---- B. genre shares and the spread of the mix
    g_rel = gs[["rel_" + g for g in genres]].to_numpy(dtype=float)[rows]
    share = gs[["share_" + g for g in genres]].to_numpy(dtype=float)[rows]
    eff = gs["effective_genres"].to_numpy(dtype=float)[rows]
    imdb = np.array([[g in film[t][3].split(",") for g in genres] for t in keep], dtype=float)
    res["genres"] = {}
    for name, v in y.items():
        rho, p, fw = rho_with_null(np.column_stack([share, eff]), v, rng)
        q = bh(p)
        rho_imdb, _, _ = rho_with_null(imdb, v, rng, 1)
        res["genres"][name] = {"familywise_95": fw, "effective_genres": {"rho": float(rho[-1]), "q": float(q[-1])},
                               "per_genre": {g: {"rho_share": float(rho[j]), "q": float(q[j]), "rho_imdb_label": float(rho_imdb[j]),
                                                 "imdb_films": int(imdb[:, j].sum()), "films_high": int((g_rel[:, j] >= HIGH).sum())}
                                             for j, g in enumerate(genres)}}
    res["genres"]["by_top_genre"] = {
        g: {"films": int(m.sum()), "median_multiple": float(np.median((gross / budget)[m])),
            "grossed_under_budget": float((gross < budget)[m].mean()), "median_gross_musd": float(np.median(gross[m]) / 1e6),
            "median_budget_musd": float(np.median(budget[m]) / 1e6), "mean_adjusted": float(y["adjusted"][m].mean())}
        for g in genres for m in [share.argmax(axis=1) == genres.index(g)] if m.sum() >= MIN_CELL}

    # ---- C. combinations of two genres
    high = g_rel >= HIGH
    adj = y["adjusted"]
    pairs, stats = [], []
    for a, b in itertools.combinations(range(len(genres)), 2):
        cells = [(high[:, a] == i) & (high[:, b] == j) for i in (0, 1) for j in (0, 1)]   # 00, 01, 10, 11
        if min(c.sum() for c in cells) < MIN_CELL:
            continue
        pairs.append((a, b))
        stats.append(np.column_stack(cells).astype(float))
    def interactions(v):
        return np.array([(c[:, 3] @ v / c[:, 3].sum()) - (c[:, 2] @ v / c[:, 2].sum())
                         - (c[:, 1] @ v / c[:, 1].sum()) + (c[:, 0] @ v / c[:, 0].sum()) for c in stats])
    obs = interactions(adj)
    exceed, top = np.zeros(len(pairs)), np.empty(N_NULL)
    for k in range(N_NULL):
        r = np.abs(interactions(rng.permutation(adj)))
        exceed += r >= np.abs(obs)
        top[k] = r.max()
    p_pair = (exceed + 1) / (N_NULL + 1)
    q_pair = bh(p_pair)
    mult = gross / budget
    pt = pd.DataFrame([{
        "genre_a": genres[a], "genre_b": genres[b], "films_both": int(c[:, 3].sum()),
        "films_only_a": int(c[:, 2].sum()), "films_only_b": int(c[:, 1].sum()),
        "median_multiple_both": float(np.median(mult[c[:, 3] > 0])), "median_multiple_only_a": float(np.median(mult[c[:, 2] > 0])),
        "median_multiple_only_b": float(np.median(mult[c[:, 1] > 0])), "median_multiple_neither": float(np.median(mult[c[:, 0] > 0])),
        "under_budget_both": float((mult < 1)[c[:, 3] > 0].mean()), "mean_adjusted_both": float(adj[c[:, 3] > 0].mean()),
        "interaction": float(o), "p": float(p), "q": float(q)}
        for (a, b), c, o, p, q in zip(pairs, stats, obs, p_pair, q_pair)])
    pt.round(4).sort_values("interaction").to_csv(OUT / "tag_profit_pairs.csv", index=False)
    res["pairs"] = {"pairs_possible": len(genres) * (len(genres) - 1) // 2, "pairs_tested": len(pairs),
                    "familywise_95": float(np.percentile(top, 95)), "largest_abs_interaction": float(np.abs(obs).max()),
                    "pairs_q_below_0.05": int((q_pair < 0.05).sum()), "pairs_beyond_familywise": int((np.abs(obs) > np.percentile(top, 95)).sum()),
                    "smallest_q": float(q_pair.min()),
                    "strongest": pt.reindex(pt["interaction"].abs().sort_values(ascending=False).index).head(8).round(4).to_dict("records")}

    # ---- D. prediction of the adjusted return
    sets = {"IMDb genre labels": imdb, "22 genre relevances": g_rel, "content tags": rel[:, cls == "content"], "all tags": rel}
    train = year < SPLIT_YEAR
    res["prediction"] = {"train_films": int(train.sum()), "test_films": int((~train).sum()),
                         "variance_explained_by_budget_and_decade": float(1 - adj.var() / y["multiple"].var()),
                         "variance_explained_adding_votes": float(1 - y["adjusted_votes"].var() / y["multiple"].var()),
                         "sets": {}}
    for name, x in sets.items():
        out = {"features": int(x.shape[1])}
        for o in ("adjusted", "adjusted_votes"):
            null = [cv_r2(x, rng.permutation(y[o]), rng, repeats=1) for _ in range(N_NULL_PRED)]
            r2, r = split_fit(x, y[o], train)
            out[o] = {"cv_r2": cv_r2(x, y[o], rng), "cv_r2_null_max": float(np.max(null)), "time_split_r2": r2, "time_split_r": r}
        res["prediction"]["sets"][name] = out
        print(name, out)

    # ---- E. the audience-size check
    votes_net = adjust(ln_votes, y["budget"], year // 10)
    noisy = [t for t in tags if (rel_all[:, tags.index(t)] >= HIGH).mean() > 0.5]
    res["audience"] = {
        "rho_votes_with_adjusted": float(zrank(ln_votes[:, None])[:, 0] @ zrank(adj[:, None])[:, 0] / n),
        "rho_votes_with_gross": float(zrank(ln_votes[:, None])[:, 0] @ zrank(y["gross"][:, None])[:, 0] / n),
        "cv_r2_content_tags_predicting_votes": cv_r2(rel[:, cls == "content"], votes_net, rng, repeats=1),
        "cv_r2_genre_labels_predicting_votes": cv_r2(imdb, votes_net, rng, repeats=1),
        "tags_strong_on_over_half_of_films": {
            t: {"rho_adjusted": float(table.loc[tags.index(t), "rho_adjusted"]),
                "rho_adjusted_votes": float(table.loc[tags.index(t), "rho_adjusted_votes"]),
                "rho_with_votes": float(zrank(rel[:, [tags.index(t)]])[:, 0] @ zrank(ln_votes[:, None])[:, 0] / n)}
            for t in noisy},
        "median_abs_rho_content": {o: float(table.loc[table["class"] == "content", "rho_" + o].abs().median())
                                   for o in ("adjusted", "adjusted_votes")},
    }
    print(json.dumps(res["audience"], indent=1))

    (OUT / "tag_profit.json").write_text(json.dumps(res, indent=1) + "\n")
    show = {k: v for k, v in res.items() if k not in ("reception_tags", "commercial_tags", "genres", "tags")}
    print(json.dumps({k: v for k, v in show.items() if k != "pairs"}, indent=1))
    for name in ("adjusted_votes",):
        t = res["tags"][name]
        print(name, {k: t[k] for k in ("familywise_95", "tags_beyond_familywise", "tags_q_below_0.05", "beyond_familywise_by_class",
                                       "largest_abs_rho_by_class", "median_abs_rho")})
        for c in ("content", "reception", "commercial"):
            print("  ", c, "+", t["top_" + c]["positive"][:8]); print("  ", c, "-", t["top_" + c]["negative"][:8])
    print("reception:", res["reception_tags"]); print("commercial:", res["commercial_tags"])


if __name__ == "__main__":
    main()
