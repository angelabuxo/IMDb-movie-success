"""How much each movie is each tag, from MovieLens Tag Genome 2021 relevance scores.

Run from the data folder (data/raw):

    cd data/raw && ../../.venv/bin/python ../../archive/scripts/reviews/tag_shares.py

IMDb gives a movie up to three genres with no weights. The Tag Genome scores
1,084 tags per movie from 0 (doesn't apply) to 1 (applies strongly), predicted
from MovieLens user tags, ratings and IMDb review text and trained on survey
answers. It ships two sets of scores, TagDL (a neural network) and glmer (a
regression). We clip both to 0..1 and use their mean: it agrees with IMDb's
genres a little better than either, and TagDL alone scores 21 tags at 0.5 or
more on over half of all films ("original" and "catastrophe" on every film).
This script joins the scores to our movies by IMDb ID and writes:

- features/tag_relevance.npy: movies x tags, the mean score, with the row
  order in features/tag_relevance_ids.csv and the column order in
  features/tag_relevance_tags.csv. Each tag is scored on its own, so a row
  doesn't sum to anything.
- features/tag_stats.csv: per tag, its median relevance over films and the
  share of films it applies to strongly (0.5 or more).
- features/genre_shares.csv: per movie, rel_<genre> (the relevance of the tag
  that names each IMDb genre) and share_<genre> (relevance ** POWER divided by
  the movie's total over the genres, so shares sum to 1), plus the top genre,
  the effective number of genres (exp of the shares' entropy) and the top
  genre under each scoring method on its own.
- features/tag_top.csv: each movie's ten most distinctive tags, ranked by
  lift (relevance minus the tag's median over films), so a tag that scores
  high on every film doesn't head every list.
- analysis/tags/tag_shares.json: the checks, numbers only: coverage by decade,
  agreement with IMDb's genre labels against a shuffled null, agreement
  between the two scoring methods, and the scores against the survey's human
  answers. notebooks/tag_shares.ipynb and the explorer's Tags page read it.

Why POWER = 2: a genre that doesn't apply still scores a little (the median
"action" score of films IMDb doesn't call Action is about 0.15), and over 22 genres
those leftovers add up. Squaring shrinks them without changing the order.

The scores exist only for the 9,734 movies of the original Tag Genome set, so
nothing after 2014 is covered. They come from what viewers wrote and rated
after release, so tags such as "oscar (best picture)" or "boring" hold the
film's reception. Don't use those as predictors of profit.
"""

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ZIP = "reviews/genome/genome_2021.zip"
FEATURES = "features/movies_features_10k.csv"
OUT_MATRIX = "features/tag_relevance.npy"
OUT_IDS = "features/tag_relevance_ids.csv"
OUT_TAGS = "features/tag_relevance_tags.csv"
OUT_GENRES = "features/genre_shares.csv"
OUT_TOP = "features/tag_top.csv"
OUT_STATS = "features/tag_stats.csv"
STRONG = 0.5
OUT_CHECKS = Path(__file__).resolve().parents[3] / "archive" / "analysis" / "tags" / "tag_shares.json"
N_TOP = 10
N_NULL = 200
POWER = 2
SEED = 0

# IMDb genre -> the genome tag that names it. Where the genome has several spellings we take the one
# that scores lowest on films IMDb doesn't give the genre ("family" also means family relationships,
# "science fiction" scores 0.21 on films that aren't).
GENRE_TAG = {
    "Action": "action", "Adventure": "adventure", "Animation": "animation", "Biography": "biographical",
    "Comedy": "comedy", "Crime": "crime", "Documentary": "documentary", "Drama": "drama",
    "Family": "kids and family", "Fantasy": "fantasy", "Film-Noir": "noir", "History": "history",
    "Horror": "horror", "Music": "music", "Musical": "musical", "Mystery": "mystery", "Romance": "romance",
    "Sci-Fi": "sci-fi", "Sport": "sports", "Thriller": "thriller", "War": "war", "Western": "western",
}
GENRES = list(GENRE_TAG)


def shares(rel, power=POWER):
    """Rows of relevance -> rows that sum to 1."""
    p = np.asarray(rel, dtype=float) ** power
    return p / p.sum(axis=1, keepdims=True)


def effective_number(sh):
    """exp(entropy): 1 when one genre holds everything, k when k genres split evenly."""
    return np.exp(-(sh * np.log(np.clip(sh, 1e-12, None))).sum(axis=1))


def read_scores(z, name, item_to_tconst, tags, tconsts):
    """Movies x tags matrix of one score file, for our movies only."""
    s = pd.read_csv(z.open(name), dtype={"tag": "category", "item_id": "int32", "score": "float32"})
    s["tconst"] = s["item_id"].map(item_to_tconst)
    s = s[s["tconst"].isin(tconsts)]
    wide = s.pivot_table(index="tconst", columns="tag", values="score", aggfunc="mean", observed=True)
    m = wide.reindex(index=tconsts, columns=tags).to_numpy(dtype=np.float32).clip(0, 1)
    for i, j in np.argwhere(np.isnan(m)):  # both files lack one pair (Toy Story, "airplane")
        m[i, j] = np.nanmedian(m[:, j])
        print("no score for", tconsts[i], repr(tags[j]), "in", name.rsplit("/", 1)[-1],
              "- filled with the tag's median %.4f" % m[i, j])
    return m


def label_fit(sh, labels):
    """How a share matrix lines up with IMDb's genre lists."""
    n = labels.sum(axis=1)
    top = sh.argmax(axis=1)
    rank = (-sh).argsort(axis=1).argsort(axis=1)
    return {
        "top_genre_in_imdb": float(labels[np.arange(len(top)), top].mean()),
        "imdb_genres_in_top_k": float(np.mean([labels[i, rank[i] < n[i]].sum() / n[i] for i in range(len(n)) if n[i]])),
        "share_on_imdb_genres": float((sh * labels).sum(axis=1).mean()),
        "effective_genres_median": float(np.median(effective_number(sh))),
    }


def main():
    rng = np.random.default_rng(SEED)
    z = zipfile.ZipFile(ZIP)
    pick = lambda k: next(n for n in z.namelist() if n.endswith(k) and "__MACOSX" not in n)

    item_to_tconst = {}
    with z.open(pick("raw/metadata.json")) as f:
        for line in io.TextIOWrapper(f, encoding="utf8"):
            r = json.loads(line)
            if r.get("imdbId") not in (None, ""):
                item_to_tconst[r["item_id"]] = "tt%07d" % int(r["imdbId"])

    movies = pd.read_csv(FEATURES)
    head = pd.read_csv(z.open(pick("scores/tagdl.csv")), usecols=["tag", "item_id"])
    tags = sorted(head["tag"].unique())
    scored = {item_to_tconst[i] for i in head["item_id"].unique() if i in item_to_tconst}
    del head
    movies["scored"] = movies["tconst"].isin(scored)
    kept = movies[movies["scored"]].sort_values("tconst").reset_index(drop=True)
    tconsts = kept["tconst"].tolist()

    tagdl = read_scores(z, pick("scores/tagdl.csv"), item_to_tconst, tags, tconsts)
    glmer = read_scores(z, pick("scores/glmer.csv"), item_to_tconst, tags, tconsts)

    both = (tagdl + glmer) / 2
    np.save(OUT_MATRIX, both)
    pd.DataFrame({"tconst": tconsts}).to_csv(OUT_IDS, index=False)
    pd.DataFrame({"tag": tags}).to_csv(OUT_TAGS, index=False)

    col = {t: i for i, t in enumerate(tags)}
    g_cols = [col[GENRE_TAG[g]] for g in GENRES]
    rel, rel1, rel2 = both[:, g_cols], tagdl[:, g_cols], glmer[:, g_cols]
    sh, sh1, sh2 = shares(rel), shares(rel1), shares(rel2)
    out = kept[["tconst", "primaryTitle", "y", "genres", "tier"]].copy()
    for j, g in enumerate(GENRES):
        out["rel_" + g] = rel[:, j].round(4)
    for j, g in enumerate(GENRES):
        out["share_" + g] = sh[:, j].round(4)
    out["top_genre"] = [GENRES[j] for j in sh.argmax(axis=1)]
    out["top_share"] = sh.max(axis=1).round(4)
    out["effective_genres"] = effective_number(sh).round(3)
    out["top_genre_tagdl"] = [GENRES[j] for j in sh1.argmax(axis=1)]
    out["top_genre_glmer"] = [GENRES[j] for j in sh2.argmax(axis=1)]
    out.to_csv(OUT_GENRES, index=False)

    tag_median = np.median(both, axis=0)
    order = np.argsort(-(both - tag_median), axis=1)[:, :N_TOP]
    with open(OUT_TOP, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tconst", "rank", "tag", "relevance", "lift"])
        for i, t in enumerate(tconsts):
            for k, j in enumerate(order[i], 1):
                w.writerow([t, k, tags[j], round(float(both[i, j]), 4), round(float(both[i, j] - tag_median[j]), 4)])
    strong = {name: (m >= STRONG).mean(axis=0) for name, m in (("mean", both), ("tagdl", tagdl), ("glmer", glmer))}
    pd.DataFrame({"tag": tags, "median_relevance": tag_median.round(4), "share_of_films_strong": strong["mean"].round(4)}
                 ).to_csv(OUT_STATS, index=False)

    res = {"source": "MovieLens Tag Genome 2021: mean of scores/tagdl.csv and scores/glmer.csv, each clipped to 0..1",
           "power": POWER, "strong": STRONG, "null_shuffles": N_NULL, "genre_tag": GENRE_TAG}
    res["tag_baselines"] = {
        name: {"tags_strong_on_over_half_of_films": sorted(tags[j] for j in np.flatnonzero(s > 0.5)),
               "median_strong_tags_per_film": float(np.median((m >= STRONG).sum(axis=1))),
               "median_tag_share_of_films_strong": float(np.median(s))}
        for (name, s), m in zip(strong.items(), (both, tagdl, glmer))}

    # ---- coverage
    dec = movies.assign(decade=movies["y"] // 10 * 10).groupby("decade")["scored"].agg(["sum", "count"])
    by_year = movies[movies["scored"]].groupby("y").size()
    res["coverage"] = {
        "movies": len(movies), "scored": len(kept), "tags": len(tags),
        "by_tier": {t: {"scored": int(r["sum"]), "movies": int(r["count"])}
                    for t, r in movies.groupby("tier")["scored"].agg(["sum", "count"]).iterrows()},
        "by_decade": {str(int(d)): {"scored": int(r["sum"]), "movies": int(r["count"])} for d, r in dec.iterrows()},
        "scored_by_year_from_2010": {str(int(y)): int(n) for y, n in by_year.items() if y >= 2010},
    }

    # ---- check 1: IMDb genre labels, against shuffled labels
    labels = np.array([[g in str(x).split(",") for g in GENRES] for x in kept["genres"]])
    per_genre = {}
    for j, g in enumerate(GENRES):
        y = labels[:, j]
        per_genre[g] = {
            "tag": GENRE_TAG[g], "imdb_films": int(y.sum()),
            "auc": float(roc_auc_score(y, rel[:, j])),
            "auc_tagdl": float(roc_auc_score(y, rel1[:, j])), "auc_glmer": float(roc_auc_score(y, rel2[:, j])),
            "auc_null_max": float(max(roc_auc_score(rng.permutation(y), rel[:, j]) for _ in range(N_NULL))),
            "median_relevance_labelled": float(np.median(rel[y, j])),
            "median_relevance_unlabelled": float(np.median(rel[~y, j])),
            "share_p10_p50_p90_labelled": [float(v) for v in np.percentile(sh[y, j], [10, 50, 90])],
            "median_share_unlabelled": float(np.median(sh[~y, j])),
        }
    top = sh.argmax(axis=1)
    null_hit = [labels[rng.permutation(len(top)), top].mean() for _ in range(N_NULL)]
    aucs = [v["auc"] for v in per_genre.values()]
    res["imdb_labels"] = {
        "per_genre": per_genre, "auc_mean": float(np.mean(aucs)), "auc_min": float(np.min(aucs)),
        "auc_mean_tagdl": float(np.mean([v["auc_tagdl"] for v in per_genre.values()])),
        "auc_mean_glmer": float(np.mean([v["auc_glmer"] for v in per_genre.values()])),
        "fit": label_fit(sh, labels), "fit_power_1": label_fit(shares(rel, 1), labels),
        "fit_tagdl": label_fit(sh1, labels), "fit_glmer": label_fit(sh2, labels),
        "top_genre_in_imdb_null_mean": float(np.mean(null_hit)), "top_genre_in_imdb_null_max": float(np.max(null_hit)),
        "imdb_genre_count_median": float(np.median(labels.sum(axis=1))),
        "top_share_median": float(np.median(sh.max(axis=1))),
        "top_share_over_half": float((sh.max(axis=1) > 0.5).mean()),
    }

    # ---- check 2: the two scoring methods
    per_tag = np.array([np.corrcoef(tagdl[:, j], glmer[:, j])[0, 1] for j in range(len(tags))])
    per_g = np.array([np.corrcoef(rel1[:, j], rel2[:, j])[0, 1] for j in range(len(GENRES))])
    res["two_methods"] = {
        "r_per_tag_median": float(np.median(per_tag)), "r_per_tag_p10": float(np.percentile(per_tag, 10)),
        "r_per_tag_min": float(per_tag.min()), "r_per_tag_min_tag": tags[int(per_tag.argmin())],
        "tags_with_r_below_0.5": sorted(tags[j] for j in np.flatnonzero(per_tag < 0.5)),
        "r_per_genre": {g: float(v) for g, v in zip(GENRES, per_g)},
        "same_top_genre": float((sh1.argmax(axis=1) == sh2.argmax(axis=1)).mean()),
        "mean_abs_share_difference": float(np.abs(sh1 - sh2).mean()),
    }

    # ---- check 3: survey answers (1 = not at all, 5 = very much)
    tag_name = {}
    with z.open(pick("raw/tags.json")) as f:
        for line in io.TextIOWrapper(f, encoding="utf8"):
            r = json.loads(line)
            tag_name[r["id"]] = r["tag"]
    row = {t: i for i, t in enumerate(tconsts)}
    pairs = {}
    with z.open(pick("raw/survey_answers.json")) as f:
        for line in io.TextIOWrapper(f, encoding="utf8"):
            r = json.loads(line)
            t, tag = item_to_tconst.get(r["item_id"]), tag_name.get(r["tag_id"])
            if r["score"] > 0 and t in row and tag in col:
                pairs.setdefault((row[t], col[tag]), []).append(r["score"])
    idx = np.array(list(pairs))
    human = np.array([np.mean(v) for v in pairs.values()])
    is_genre = np.isin(idx[:, 1], g_cols)
    survey = {"pairs": len(human), "films": len(set(idx[:, 0])), "genre_pairs": int(is_genre.sum())}
    for name, m in (("mean", both), ("tagdl", tagdl), ("glmer", glmer)):
        pred = m[idx[:, 0], idx[:, 1]]
        survey[name] = {
            "spearman": float(spearmanr(human, pred)[0]),
            "spearman_genre_tags": float(spearmanr(human[is_genre], pred[is_genre])[0]),
            "mean_score_by_answer": {str(k): float(pred[np.round(human) == k].mean()) for k in range(1, 6)},
        }
    mae = [ln.split(";") for ln in z.read(pick("predictions/performance_results_tenfolds.txt")).decode().splitlines() if ln.strip()]
    survey["held_out_mae_5_point_scale"] = {
        "tagdl": float(np.mean([float(p[1].split(":")[1]) for p in mae])),
        "glmer": float(np.mean([float(p[2].split(":")[1]) for p in mae])),
        "note": "ten-fold means from the dataset's predictions/performance_results_tenfolds.txt",
    }
    res["survey"] = survey

    OUT_CHECKS.parent.mkdir(parents=True, exist_ok=True)
    OUT_CHECKS.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "genre_tag"}, indent=1))

    # ---- a sample to read by hand
    print("\nsample to read")
    for i in rng.choice(len(kept), 20, replace=False):
        g3 = ", ".join("%s %d%%" % (GENRES[j], round(100 * sh[i, j])) for j in (-sh[i]).argsort()[:3])
        t5 = ", ".join(tags[j] for j in order[i][:5])
        print("%s (%d) [IMDb: %s] -> %s | %s" % (kept["primaryTitle"][i], kept["y"][i], kept["genres"][i], g3, t5))


if __name__ == "__main__":
    main()
