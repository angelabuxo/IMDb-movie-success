"""Four follow-ups to tag_profit.py that don't lean on tags as predictors.

Run from the data folder (data/raw) after tag_shares.py, build_graph.py,
film_features.py and franchise_flag.py:

    cd data/raw && ../../.venv/bin/python ../../archive/scripts/reviews/tag_followups.py

Sample: films flagged clean in features/graph/nodes.csv, released 1975 or
later. The adjusted return is ln(gross / budget) minus its OLS fit on
ln(budget) and release decade, as in tag_profit.py. Gross over budget isn't
profit.

A. Usual mix. Does a director or studio return more when a film stays close
   to the genre mix of their earlier films? A film's mix is its genre-share
   vector (Tag Genome, films to 2014) or, as a second method that covers every
   year, its IMDb genre labels as a 0/1 vector. The usual mix of a director
   or studio at a film is the mean vector of their strictly earlier films
   (at least MIN_PRIOR of them), and the departure is 1 minus the cosine
   between the film's vector and that mean. Only earlier films are used, so
   the film's own outcome can't leak in. We report rho(departure, adjusted
   return), the same with the film's own genres also held fixed (so leaving
   for a weak genre isn't counted as the cost of leaving), and whether the
   track record predicts the return better for films that stay.

B. Seen against earning. ln(IMDb votes) and the IMDb rating against the
   return, and which genres are seen or liked more than their return says.
   Attention is ln(votes) minus its fit on ln(gross) and release year. Films
   are also split by whether they are rated above their decade's median and
   whether they returned more than budget and decade predict; a genre's lift
   in a quadrant is its share there over its share of all films.

C. Critics against audience. The gap is the film's Metacritic score minus its
   IMDb rating, each standardised within release decade. By genre, against
   the return, and again with Rotten Tomatoes as the second method.

D. Controls. The franchise flag (Wikidata) checked against the Tag Genome's
   sequel tags and against numbered titles; the release month parsed from
   Wikipedia, by month; and the main results of tag_profit.py again with
   "follows another film" and the month held fixed. The full flag also marks
   films that later got a sequel, which happens because a film earned, so
   holding it fixed would remove part of the outcome. It is reported as the
   strict variant only.

Nulls shuffle the outcome across films (N_NULL times). Writes
analysis/tags/tag_followups.json: numbers only.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.metrics import roc_auc_score

from tag_profit import SPLIT_YEAR, adjust, bh, cv_r2, rho_with_null, split_r2, tag_class, zrank

NODES = "features/graph/nodes.csv"
EDGES = "features/graph/edges.csv"
TRACK = "features/film_network_features.csv"
FRANCHISE = "features/franchise.csv"
INFOBOX = ["features/wiki_infobox.csv", "features/wiki_infobox_10k.csv"]
OUT = Path(__file__).resolve().parents[3] / "archive" / "analysis" / "tags" / "tag_followups.json"
FIRST_YEAR = 1975
MIN_PRIOR = 2
MIN_GROUP = 30
N_NULL = 1000
SEED = 0
NUMBERED = re.compile(r"\b(2|3|4|5|6|7|8|II|III|IV|V|VI)\b|\bPart (\d|One|Two|Three|I+)\b|Chapter (\d|Two|Three)|Returns\b|Reloaded|Strikes Back", re.I)


def rho(a, b):
    return float(zrank(np.asarray(a, float)[:, None])[:, 0] @ zrank(np.asarray(b, float)[:, None])[:, 0] / len(a))


def rho_p(a, b, rng, n_null=N_NULL):
    """Spearman rho with a two-sided permutation p."""
    ar, br = zrank(np.asarray(a, float)[:, None])[:, 0], zrank(np.asarray(b, float)[:, None])[:, 0]
    obs = ar @ br / len(ar)
    null = np.array([ar @ rng.permutation(br) / len(ar) for _ in range(n_null)])
    return {"rho": float(obs), "p": float(((np.abs(null) >= abs(obs)).sum() + 1) / (n_null + 1)), "films": int(len(ar)),
            "null_95": float(np.percentile(np.abs(null), 95))}


def net_of(y, *cols):
    x = np.column_stack([np.ones(len(y))] + list(cols))
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def zscore_within(v, groups):
    out = np.full(len(v), np.nan)
    for g in set(groups):
        m = (groups == g) & ~np.isnan(v)
        if m.sum() >= 20:
            out[m] = (v[m] - v[m].mean()) / v[m].std()
    return out


def main():
    rng = np.random.default_rng(SEED)
    num = lambda s: float(s) if s not in ("", None) else np.nan
    movies = {}
    with open(NODES, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["type"] == "movie" and r["year"]:
                movies[r["id"]] = {"title": r["label"], "year": int(float(r["year"])), "genres": [g for g in r["genres"].split(",") if g],
                                   "clean": r["clean"] == "1" and bool(r["multiple"]), "budget": num(r["budget_usd"]),
                                   "gross": num(r["gross_usd"]), "votes": num(r["votes"]), "rating": num(r["rating"]),
                                   "rt": num(r["rt"]), "mc": num(r["mc"])}
    gs = pd.read_csv("features/genre_shares.csv").set_index("tconst")
    genres = [c[6:] for c in gs.columns if c.startswith("share_")]
    share = {t: v for t, v in zip(gs.index, gs[["share_" + g for g in genres]].to_numpy(float))}
    label_vec = lambda t: np.array([float(g in movies[t]["genres"]) for g in genres])

    S = [t for t, m in movies.items() if m["clean"] and m["year"] >= FIRST_YEAR]          # the sample
    ix = {t: i for i, t in enumerate(S)}
    year = np.array([movies[t]["year"] for t in S])
    budget, gross = np.array([movies[t]["budget"] for t in S]), np.array([movies[t]["gross"] for t in S])
    ln_mult, ln_budget, decade = np.log(gross / budget), np.log(budget), year // 10
    adj = adjust(ln_mult, ln_budget, decade)
    imdb = np.array([label_vec(t) for t in S])
    adj_genre = net_of(adj, imdb[:, imdb.sum(axis=0) > 0][:, 1:])
    votes, rating = np.array([movies[t]["votes"] for t in S]), np.array([movies[t]["rating"] for t in S])
    scored = np.array([t in share for t in S])
    res = {"sample": {"films": len(S), "with_tag_scores": int(scored.sum()), "years": [int(year.min()), int(year.max())],
                      "first_year": FIRST_YEAR, "shuffles": N_NULL}}

    # ---- A. usual mix
    credits = defaultdict(list)                       # (role, entity) -> [(year, film)]
    with open(EDGES, newline="") as f:
        for r in csv.DictReader(f):
            if r["role"] in ("director", "studio") and r["target"] in movies:
                credits[(r["role"], r["source"])].append((movies[r["target"]]["year"], r["target"]))
    by_film = defaultdict(lambda: defaultdict(list))  # film -> role -> [entity]
    for (role, e), films in credits.items():
        for _, t in films:
            by_film[t][role].append(e)
    track = pd.read_csv(TRACK).set_index("tconst")
    usual = {}
    for vec_name, vec in (("genre shares (Tag Genome, to 2014)", lambda t: share.get(t)), ("IMDb genre labels (all years)", label_vec)):
        usual[vec_name] = {}
        for role in ("director", "studio"):
            dep, n_prior, rows = [], [], []
            for t in S:
                v = vec(t)
                if v is None or not np.any(v):
                    continue
                ds, ns = [], []
                for e in by_film[t][role]:
                    prior = [p for p in (vec(u) for y, u in credits[(role, e)] if y < movies[t]["year"]) if p is not None and np.any(p)]
                    if len(prior) >= MIN_PRIOR:
                        mean = np.mean(prior, axis=0)
                        ds.append(1 - float(v @ mean / (np.linalg.norm(v) * np.linalg.norm(mean))))
                        ns.append(len(prior))
                if ds:
                    dep.append(np.mean(ds)); n_prior.append(np.mean(ns)); rows.append(ix[t])
            dep, rows = np.array(dep), np.array(rows)
            out = {"films": len(rows), "median_departure": float(np.median(dep)),
                   "median_prior_films": float(np.median(n_prior)),
                   "against_adjusted": rho_p(dep, adj[rows], rng),
                   "against_adjusted_genre_fixed": rho_p(dep, adj_genre[rows], rng),
                   "against_budget": rho_p(dep, ln_budget[rows], rng, 200)}
            q = np.quantile(dep, [0.25, 0.5, 0.75])
            grp = np.digitize(dep, q)
            out["by_quartile"] = [{"quartile": k + 1, "films": int((grp == k).sum()), "mean_departure": float(dep[grp == k].mean()),
                                   "median_multiple": float(np.exp(np.median(ln_mult[rows][grp == k]))),
                                   "grossed_under_budget": float((ln_mult[rows][grp == k] < 0).mean()),
                                   "mean_adjusted": float(adj[rows][grp == k].mean()),
                                   "mean_adjusted_genre_fixed": float(adj_genre[rows][grp == k].mean())} for k in range(4)]
            tr = track.reindex([S[i] for i in rows])[role + "_track"].to_numpy(float)
            has = ~np.isnan(tr)
            if has.sum() >= 200:
                stay = dep[has] <= np.median(dep[has])
                a, y_ = tr[has], adj[rows][has]
                obs = rho(a[stay], y_[stay]) - rho(a[~stay], y_[~stay])
                null = []
                for _ in range(N_NULL):
                    s = rng.permutation(stay)
                    null.append(rho(a[s], y_[s]) - rho(a[~s], y_[~s]))
                out["track_record"] = {"films": int(has.sum()), "rho_track_when_staying": rho(a[stay], y_[stay]),
                                       "rho_track_when_leaving": rho(a[~stay], y_[~stay]), "difference": float(obs),
                                       "p": float(((np.abs(null) >= abs(obs)).sum() + 1) / (N_NULL + 1)),
                                       "rho_departure_net_of_track": rho_p(dep[has], net_of(y_, a), rng)["rho"]}
            usual[vec_name][role] = out
            print("usual mix", vec_name, role, {k: out[k] for k in ("films", "against_adjusted", "against_adjusted_genre_fixed")},
                  out.get("track_record"))
    res["usual_mix"] = usual

    # ---- B. seen against earning
    ln_votes = np.log(votes)
    attention = net_of(ln_votes, np.log(gross), year, year ** 2)
    rating_dec = zscore_within(rating, decade)
    aud = {"votes_vs_adjusted": rho_p(ln_votes, adj, rng), "votes_vs_gross": rho_p(ln_votes, np.log(gross), rng, 200),
           "rating_vs_adjusted": rho_p(rating, adj, rng), "rating_vs_gross": rho_p(rating, np.log(gross), rng),
           "rating_vs_budget": rho_p(rating, ln_budget, rng, 200), "rating_vs_votes": rho_p(rating, ln_votes, rng, 200),
           "rating_vs_adjusted_by_decade": {str(int(d) * 10): rho(rating[decade == d], adj[decade == d]) for d in sorted(set(decade))
                                            if (decade == d).sum() >= 100}}
    liked, earned = rating_dec > 0, adj > 0
    quads = {"liked and earned": liked & earned, "liked, didn't earn": liked & ~earned,
             "not liked, earned": ~liked & earned, "not liked, didn't earn": ~liked & ~earned}
    aud["quadrants"] = {k: float(v.mean()) for k, v in quads.items()}
    per, ps = {}, []
    for j, g in enumerate(genres):
        m = imdb[:, j] > 0
        if m.sum() < MIN_GROUP:
            continue
        per[g] = {"films": int(m.sum()), "mean_attention": float(attention[m].mean()), "mean_rating_z": float(rating_dec[m].mean()),
                  "mean_adjusted": float(adj[m].mean()), "median_multiple": float(np.exp(np.median(ln_mult[m]))),
                  "lift": {k: float(m[v].mean() / m.mean()) for k, v in quads.items()}}
        for k, v in quads.items():
            hits = int((m & v).sum())
            p = min(1.0, 2 * min(hypergeom.cdf(hits, len(S), int(m.sum()), int(v.sum())),
                                 hypergeom.sf(hits - 1, len(S), int(m.sum()), int(v.sum()))))
            ps.append((g, k, p))
    for (g, k, _), q in zip(ps, bh([p for *_, p in ps])):
        per[g].setdefault("lift_q", {})[k] = float(q)
    aud["per_genre"] = per
    r_att, p_att, fw = rho_with_null(imdb, attention, rng)
    aud["attention_by_label"] = {"familywise_95": fw, "rho": {g: float(r) for g, r in zip(genres, r_att)}}
    sh = np.array([share[t] for t in np.array(S)[scored]])
    r_sh, _, fw_sh = rho_with_null(sh, attention[scored], rng)
    aud["attention_by_share"] = {"films": int(scored.sum()), "familywise_95": fw_sh, "rho": {g: float(r) for g, r in zip(genres, r_sh)}}
    res["audience"] = aud
    print("audience", {k: aud[k] for k in ("votes_vs_adjusted", "rating_vs_adjusted", "rating_vs_gross", "quadrants")})

    # ---- C. critics against audience
    mc, rt = np.array([movies[t]["mc"] for t in S]), np.array([movies[t]["rt"] for t in S])
    crit = {}
    for name, score in (("metacritic", mc), ("rotten_tomatoes", rt)):
        ok = ~np.isnan(score)
        z = np.full(len(S), np.nan)
        z[ok] = zscore_within(score, decade)[ok] - zscore_within(np.where(ok, rating, np.nan), decade)[ok]
        ok = ~np.isnan(z)
        c = {"films": int(ok.sum()), "rho_score_vs_rating": rho(score[ok], rating[ok]),
             "gap_vs_adjusted": rho_p(z[ok], adj[ok], rng), "score_vs_adjusted": rho_p(score[ok], adj[ok], rng),
             "rating_vs_adjusted_same_films": rho_p(rating[ok], adj[ok], rng, 200),
             "gap_vs_budget": rho_p(z[ok], ln_budget[ok], rng, 200), "gap_vs_votes": rho_p(z[ok], ln_votes[ok], rng, 200)}
        r_g, p_g, fw = rho_with_null(imdb[ok], z[ok], rng)
        q_g = bh(p_g)
        c["familywise_95"] = fw
        c["per_genre"] = {g: {"films": int(imdb[ok][:, j].sum()), "mean_gap": float(z[ok][imdb[ok][:, j] > 0].mean()),
                              "rho": float(r_g[j]), "q": float(q_g[j])} for j, g in enumerate(genres) if imdb[ok][:, j].sum() >= MIN_GROUP}
        both = ok & scored
        shb = np.array([share[t] for t in np.array(S)[both]])
        r_s, _, fw_s = rho_with_null(shb, z[both], rng)
        c["by_share"] = {"films": int(both.sum()), "familywise_95": fw_s, "rho": {g: float(r) for g, r in zip(genres, r_s)}}
        crit[name] = c
        crit[name + "_gap"] = z
    okb = ~np.isnan(crit["metacritic_gap"]) & ~np.isnan(crit["rotten_tomatoes_gap"])
    crit["two_methods"] = {"films": int(okb.sum()), "rho_between_gaps": rho(crit["metacritic_gap"][okb], crit["rotten_tomatoes_gap"][okb]),
                           "rho_per_genre_mean_gaps": rho([crit["metacritic"]["per_genre"][g]["mean_gap"] for g in crit["metacritic"]["per_genre"] if g in crit["rotten_tomatoes"]["per_genre"]],
                                                          [crit["rotten_tomatoes"]["per_genre"][g]["mean_gap"] for g in crit["metacritic"]["per_genre"] if g in crit["rotten_tomatoes"]["per_genre"]])}
    del crit["metacritic_gap"], crit["rotten_tomatoes_gap"]
    res["critics"] = crit
    print("critics", {k: {a: crit[k][a] for a in ("films", "gap_vs_adjusted", "score_vs_adjusted")} for k in ("metacritic", "rotten_tomatoes")}, crit["two_methods"])

    # ---- D. controls: franchise and release month
    fr = pd.read_csv(FRANCHISE).drop_duplicates("tconst").set_index("tconst")
    flag = np.array([float(fr["franchise"].get(t, np.nan)) for t in S])
    follows = np.array([isinstance(fr["follows"].get(t, np.nan), str) for t in S])
    numbered = np.array([bool(NUMBERED.search(movies[t]["title"])) for t in S])
    rel = np.load("features/tag_relevance.npy")
    rel_ids = pd.read_csv("features/tag_relevance_ids.csv")["tconst"].tolist()
    tags = pd.read_csv("features/tag_relevance_tags.csv", keep_default_na=False)["tag"].tolist()
    row = {t: i for i, t in enumerate(rel_ids)}
    sc_rows = np.array([row[t] for t in np.array(S)[scored]])
    known = ~np.isnan(flag)
    ctl = {"franchise": {"films_with_flag": int(known.sum()), "flagged": float(np.nanmean(flag)), "follows_another_film": float(follows[known].mean()),
                         "numbered_title": float(numbered.mean()),
                         "numbered_titles_flagged": float(flag[known & numbered].mean()),
                         "flagged_with_numbered_title": float(numbered[known & (flag == 1)].mean()),
                         "follows_with_numbered_title": float(numbered[follows].mean()),
                         "tag_check": {}}}
    ks = known & scored
    for t in ("sequel", "sequels", "franchise", "series"):
        v = rel[sc_rows, tags.index(t)][known[scored]]
        ctl["franchise"]["tag_check"][t] = {
            "auc_for_flag": float(roc_auc_score(flag[ks], v)), "auc_for_follows": float(roc_auc_score(follows[ks], v)),
            "flagged_among_tag_at_0.5": float(flag[ks][v >= 0.5].mean()) if (v >= 0.5).any() else None,
            "films_tag_at_0.5": int((v >= 0.5).sum()),
            "tag_at_0.5_among_flagged": float((v[flag[ks] == 1] >= 0.5).mean()), "tag_at_0.5_among_follows": float((v[follows[ks]] >= 0.5).mean())}
    ctl["franchise"]["films_scored_with_flag"] = int(ks.sum())
    f0 = np.where(known, flag, 0.0)
    ctl["franchise"]["mean_adjusted"] = {"flagged": float(adj[f0 == 1].mean()), "not_flagged": float(adj[f0 == 0].mean()),
                                         "follows": float(adj[follows].mean()), "numbered_title": float(adj[numbered].mean()),
                                         "flag_vs_adjusted": rho_p(f0, adj, rng), "follows_vs_adjusted": rho_p(follows, adj, rng)}
    info = pd.concat([pd.read_csv(p, usecols=["tconst", "release_month"]) for p in INFOBOX]).drop_duplicates("tconst").set_index("tconst")
    month = np.array([float(info["release_month"].get(t, np.nan)) for t in S])
    hasm = ~np.isnan(month)
    dummies = np.column_stack([(month == k).astype(float) for k in range(2, 13)])
    eta = lambda v: 1 - net_of(v[hasm], dummies[hasm]).var() / v[hasm].var()
    obs = eta(adj)
    null = [1 - net_of(rng.permutation(adj[hasm]), dummies[hasm]).var() / adj[hasm].var() for _ in range(N_NULL)]
    ctl["release_month"] = {"films_with_month": int(hasm.sum()), "share_of_variance_in_adjusted": float(obs),
                            "share_null_95": float(np.percentile(null, 95)), "share_of_variance_in_budget": float(eta(ln_budget)),
                            "by_month": {str(k): {"films": int((month == k).sum()), "mean_adjusted": float(adj[month == k].mean()),
                                                  "median_budget_musd": float(np.median(budget[month == k]) / 1e6),
                                                  "median_multiple": float(np.exp(np.median(ln_mult[month == k])))} for k in range(1, 13)},
                            "sample_to_check": [{"title": movies[S[i]]["title"], "year": int(year[i]), "month": int(month[i])}
                                                for i in rng.choice(np.flatnonzero(hasm & (votes > 300000)), 25, replace=False)]}
    # the tag results again with franchise and month held fixed
    full = hasm
    extra = np.column_stack([follows.astype(float), dummies])
    adj2 = np.full(len(S), np.nan)
    adj2[full] = net_of(adj[full], extra[full])
    share_of = lambda *cols: float(1 - net_of(adj[full], *cols).var() / adj[full].var())
    ctl["with_controls"] = {"films": int(full.sum()), "controls": "follows another film, release month",
                            "variance_of_adjusted_explained_by_follows_and_month": share_of(extra[full]),
                            "by_follows_alone": share_of(follows[full].astype(float)),
                            "by_full_flag_alone": share_of(f0[full]),
                            "by_full_flag_follows_and_month": share_of(f0[full], extra[full])}
    adj_strict = np.full(len(S), np.nan)
    adj_strict[full] = net_of(adj[full], f0[full], extra[full])
    m2 = full & scored
    relS = rel[np.array([row[t] for t in np.array(S)[m2]])].astype(float)
    cls = np.array([tag_class(t) for t in tags])
    r1, _, _ = rho_with_null(relS, adj[m2], rng, 1)
    r2, _, fw2 = rho_with_null(relS, adj2[m2], rng, 200)
    top = np.argsort(-np.abs(np.where(cls == "content", r1, 0)))[:40]
    ctl["with_controls"]["tags"] = {
        "films": int(m2.sum()), "rho_between_before_and_after": float(np.corrcoef(r1, r2)[0, 1]),
        "median_abs_rho_content": {"before": float(np.median(np.abs(r1[cls == "content"]))), "after": float(np.median(np.abs(r2[cls == "content"])))},
        "top_40_content_same_sign": float((np.sign(r1[top]) == np.sign(r2[top])).mean()),
        "top_40_content_mean_shrink": float(np.mean(np.abs(r2[top]) / np.abs(r1[top]))),
        "commercial": {t: {"before": float(r1[tags.index(t)]), "after": float(r2[tags.index(t)])} for t in ("sequel", "franchise", "sequels", "series", "trilogy")},
        "largest_changes": [{"tag": tags[j], "before": float(r1[j]), "after": float(r2[j])} for j in np.argsort(-np.abs(r1 - r2))[:10]]}
    sh2 = np.array([share[t] for t in np.array(S)[m2]])
    g1, _, _ = rho_with_null(sh2, adj[m2], rng, 1)
    g2, _, _ = rho_with_null(sh2, adj2[m2], rng, 1)
    g3, _, _ = rho_with_null(sh2, adj_strict[m2], rng, 1)
    r3, _, _ = rho_with_null(relS, adj_strict[m2], rng, 1)
    ctl["with_controls"]["tags"]["strict_full_flag"] = {
        "top_40_content_same_sign": float((np.sign(r1[top]) == np.sign(r3[top])).mean()),
        "top_40_content_mean_shrink": float(np.mean(np.abs(r3[top]) / np.abs(r1[top]))),
        "sequel": float(r3[tags.index("sequel")])}
    ctl["with_controls"]["genre_shares"] = {g: {"before": float(a), "after": float(b), "strict": float(c)} for g, a, b, c in zip(genres, g1, g2, g3)}
    train = (year < SPLIT_YEAR)[m2]
    content = relS[:, cls == "content"]
    ctl["with_controls"]["content_tags_r2"] = {
        "before": {"random_folds": cv_r2(content, adj[m2], rng, repeats=1), "in_time": split_r2(content, adj[m2], train)},
        "after": {"random_folds": cv_r2(content, adj2[m2], rng, repeats=1), "in_time": split_r2(content, adj2[m2], train)}}
    res["controls"] = ctl
    print(json.dumps(ctl["with_controls"], indent=1)[:5000])

    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
