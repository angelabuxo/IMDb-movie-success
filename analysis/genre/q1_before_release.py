"""Q1 tied to the people (Q2) and the screenplays (Q3), using only what exists before release. Replaces the first
version (q1_scripts_network.py, now gone; its numbers are in git history), which had two flaws:

  - a team's earlier films came only from the 12,673-film set, i.e. films that later got 10,000+ IMDb votes, so a
    team with many earlier credits was partly a team with a noticed, successful past (success leaking into the
    predictor). Earlier films now come from IMDb's full title.principals and title.basics: every earlier
    theatrical film (titleType movie) a person is credited on, with IMDb's genres, noticed or not.
  - the shuffle p-values came from a smaller model than the coefficients reported. Every shuffle now refits the
    same model as the reported coefficient.

  A specialists   A film's team: its directors, writers, producers and first three billed actors/actresses, from
                  title.principals. For each, their movies released in an earlier year. Per film:
                    genre_experience  share of the team's earlier credits carrying one of the film's genres
                                      (pooled over people, averaged over the film's genres)
                    prior_credits     log(1 + the team's earlier credits)
                    genre_breadth     entropy of the team's earlier credits over the kept genres
                  Outcome: ln(gross / budget) net of the budget decile x era mean. OLS with genre dummies and a sequel
                  flag (Wikidata 'follows'), HC3; also without sequels. Shuffle: genre_experience permuted within the
                  film's first genre x budget decile, the full model refitted each time. The old 12,673-film history
                  is kept as a comparison row ("history: noticed films only").
  B typicality    Cosine of a script's embedding (all-MiniLM-L6-v2, character labels removed) to the leave-one-out mean
                  of the other scripts in each of its genres, averaged, minus its cosine to the mean of all scripts.
                  Same outcome and controls, plus script-quality controls from Q3's data check
                  (scripts-analysis/data_check): log tokens, dialogue share, unparsed share, scene headings inside
                  dialogue, upper-case and punctuation shares of the description, a tiny-layer flag, and on the final
                  1,598 films the symbol-noise share. If typicality only measured clean script files, the quality
                  controls would take its coefficient away.

Outputs in analysis/genre/outputs/before_release/. Numbers only.
Run (needs data/raw/imdb/title.principals.tsv.gz and title.basics.tsv.gz, from scripts/get_data.py imdb):
  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.q1_before_release [--perms 1000]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from analysis.genre.q1_genre import DATA, OUT as MAIN_OUT, ROOT, SEED, load, prepare

OUT = MAIN_OUT / "before_release"
ROLES = ("director", "writer", "producer")
ACTOR = ("actor", "actress")
TOP_ACTORS = 3
CHECK = ROOT / "scripts-analysis" / "data_check"


def net_return(f):
    cell = f["budget_bin"].astype(str) + "|" + f["era_bin"].astype(str)
    return (f["ln_gross_over_budget"] - f.groupby(cell)["ln_gross_over_budget"].transform("mean")).to_numpy()


def sequel_flag(f):
    """True when Wikidata says the film follows another (features/franchise.csv, 'follows'), the cleaner flag."""
    fr = pd.read_csv(DATA / "features" / "franchise.csv").drop_duplicates("tconst").set_index("tconst")["follows"]
    v = f["imdb_id"].map(fr)
    return v.notna().to_numpy() & v.astype(str).str.len().gt(3).to_numpy()


def design(X, f, M, keep, extra=None):
    """Standardised X, then genre dummies, the sequel flag and any extra controls (standardised)."""
    ctrl = pd.get_dummies(pd.DataFrame(M, columns=keep).astype(int).astype(str), drop_first=True, dtype=float).to_numpy()
    cols = [(X - X.mean(0)) / X.std(0), ctrl, sequel_flag(f).astype(float)[:, None]]
    if extra is not None and extra.shape[1]:
        sd = extra.std(0)
        cols.append((extra - extra.mean(0)) / np.where(sd > 0, sd, 1))
    Z = np.column_stack(cols)
    keepcol = Z.std(0) > 0
    keepcol[:X.shape[1]] = True
    return np.column_stack([np.ones(len(Z)), Z[:, keepcol]])


def ols(y, X, names, f, M, keep, extra=None, label=""):
    import statsmodels.api as sm
    fit = sm.OLS(y, design(X, f, M, keep, extra)).fit(cov_type="HC3")
    ci = fit.conf_int()
    return [{"films": label, "term": n, "coef_per_sd": round(fit.params[1 + i], 4), "lo": round(ci[1 + i, 0], 4),
             "hi": round(ci[1 + i, 1], 4), "p": float(f"{fit.pvalues[1 + i]:.3g}"), "n": int(fit.nobs)} for i, n in enumerate(names)]


def shuffle_p(y, X, j, f, M, keep, extra, groups, perms, rng):
    """Permute column j of X within groups, refit the same OLS each time; two-sided p for its coefficient."""
    def coef(Xv):
        D = design(Xv, f, M, keep, extra)
        return np.linalg.lstsq(D, y, rcond=None)[0][1 + j]
    obs = coef(X)
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    nl = np.empty(perms)
    Xv = X.copy()
    for p in range(perms):
        v = X[:, j].copy()
        for ix in idx:
            v[ix] = X[ix, j][rng.permutation(len(ix))]
        Xv[:, j] = v
        nl[p] = coef(Xv)
    return float(obs), float((1 + (np.abs(nl) >= abs(obs)).sum()) / (1 + perms))


def imdb_tables(f, keep):
    """Teams of the usable films and every earlier movie of their members, from IMDb's own files."""
    base = DATA / "imdb"
    tb = pd.read_csv(base / "title.basics.tsv.gz", sep="\t", usecols=["tconst", "titleType", "startYear", "genres"],
                     na_values="\\N", dtype=str, quoting=3)
    movies = tb[tb["titleType"] == "movie"].dropna(subset=["startYear"])
    movies = movies.assign(year=pd.to_numeric(movies["startYear"], errors="coerce")).dropna(subset=["year"])
    gi = {g: j for j, g in enumerate(keep)}
    mg = {t: [gi[g] for g in str(gs).split(",") if g in gi] for t, gs in zip(movies["tconst"], movies["genres"])}
    myear = dict(zip(movies["tconst"], movies["year"]))
    usable = set(f["imdb_id"])
    team_rows, chunks = [], []
    for ch in pd.read_csv(base / "title.principals.tsv.gz", sep="\t", usecols=["tconst", "ordering", "nconst", "category"],
                          dtype={"tconst": str, "nconst": str, "category": str}, chunksize=5_000_000, quoting=3):
        ch = ch[ch["category"].isin(ROLES + ACTOR) & ch["tconst"].isin(myear.keys())]
        chunks.append(ch[["tconst", "nconst", "category", "ordering"]])
    pr = pd.concat(chunks, ignore_index=True)
    team = pr[pr["tconst"].isin(usable)].copy()
    team["ordering"] = pd.to_numeric(team["ordering"], errors="coerce")
    act = team[team["category"].isin(ACTOR)].sort_values(["tconst", "ordering"]).groupby("tconst").head(TOP_ACTORS)
    team = pd.concat([team[team["category"].isin(ROLES)], act])
    people = set(team["nconst"])
    hist = pr[pr["nconst"].isin(people)].drop_duplicates(["nconst", "tconst"])
    hist = hist.assign(year=hist["tconst"].map(myear))
    return team.groupby("tconst")["nconst"].apply(lambda s: sorted(set(s))).to_dict(), hist, mg


def graph_tables(films, genres, keep):
    """The first version's history: credits and genres of the 12,673-film set only."""
    cr = pd.read_csv(DATA / "features" / "graph" / "credits.csv")
    cr = cr[cr["role"].isin(ROLES) | ((cr["role"] == "actor") & (cr["ordering"] <= TOP_ACTORS))]
    gi = {g: j for j, g in enumerate(keep)}
    mg = defaultdict(list)
    for t, g in zip(genres["imdb_id"], genres["genre"]):
        if g in gi:
            mg[t].append(gi[g])
    hist = cr.assign(year=cr["tconst"].map(films.set_index("imdb_id")["year"])).dropna(subset=["year"])
    return cr.groupby("tconst")["nconst"].apply(lambda s: sorted(set(s))).to_dict(), hist, mg


def team_measures(f, keep, M, team, hist, mg):
    by_person = {p: d.sort_values("year")[["tconst", "year"]].to_numpy() for p, d in hist.groupby("nconst")}
    rows = []
    for t, yr, mrow in zip(f["imdb_id"], f["year"], M):
        mine = np.flatnonzero(mrow)
        cnt, n = np.zeros(len(keep)), 0
        for p in team.get(t, []):
            for t2, y2 in by_person.get(p, ()):
                if y2 >= yr:
                    break
                n += 1
                for j in mg.get(t2, ()):
                    cnt[j] += 1
        if n == 0 or len(mine) == 0:
            rows.append((np.nan, np.nan, np.nan))
            continue
        q = cnt / cnt.sum() if cnt.sum() > 0 else cnt
        rows.append((float(np.mean(cnt[mine] / n)), np.log1p(n), float(-(q[q > 0] * np.log(q[q > 0])).sum())))
    return np.array(rows)


def specialists(films, genres, f, keep, M, perms):
    names = ["genre_experience", "prior_credits", "genre_breadth"]
    y_all = net_return(f)
    res, summ, quint = [], {}, []
    for src, (team, hist, mg) in (("history: all earlier IMDb movies", imdb_tables(f, keep)),
                                   ("history: noticed films only (first version)", graph_tables(films, genres, keep))):
        X = team_measures(f, keep, M, team, hist, mg)
        ok = ~np.isnan(X).any(1)
        fo, Mo, y, Xo = f[ok].reset_index(drop=True), M[ok], y_all[ok], X[ok]
        res += ols(y, Xo, names, fo, Mo, keep, label=f"{src}, sequel flag controlled")
        ns = ~sequel_flag(fo)
        res += ols(y[ns], Xo[ns], names, fo[ns], Mo[ns], keep, label=f"{src}, sequels left out")
        if src.startswith("history: all"):
            grp = pd.factorize(pd.Series(Mo.argmax(1)).astype(str) + "|" + fo["budget_bin"].astype(str))[0]
            obs, p = shuffle_p(y, Xo, 0, fo, Mo, keep, None, grp, perms, np.random.default_rng(SEED + 2100))
            q = pd.qcut(Xo[:, 0], 5, labels=False, duplicates="drop")
            quint = [{"quintile": k + 1, "films": int((q == k).sum()),
                      "genre_experience_range": f"{Xo[q == k, 0].min():.2f}-{Xo[q == k, 0].max():.2f}",
                      "mean_net_return": round(float(y[q == k].mean()), 4)} for k in range(int(q.max()) + 1)]
            summ = {"films": int(ok.sum()), "films_without_earlier_credits": int((~ok).sum()), "sequels": int((~ns).sum()),
                    "median_earlier_credits": float(np.median(np.expm1(Xo[:, 1]))),
                    "shuffle_coef_genre_experience": round(obs, 4), "shuffle_p_genre_experience": round(p, 4),
                    "shuffle_model": "same as the 'sequel flag controlled' row", "perms": perms}
        else:
            summ["first_version_median_earlier_credits"] = float(np.median(np.expm1(Xo[:, 1])))
    return pd.DataFrame(res), pd.DataFrame(quint), summ


def quality(ids):
    """Script-quality measures from scripts-analysis/data_check, by IMDb id."""
    ls = pd.read_csv(CHECK / "layer_stats.csv").drop_duplicates("imdb_id").set_index("imdb_id")
    elements = ls[[c for c in ls.columns if c.endswith("_elements")]].sum(axis=1)
    q = pd.DataFrame({
        "log_tokens": np.log1p(ls["dialogue_tokens"] + ls["description_tokens"]),
        "dialogue_share": ls["dialogue_share_of_dd"],
        "unparsed_share": ls["unparsed"] / elements.where(elements > 0),
        "heading_in_dialogue_share": ls["heading_in_dialogue"] / ls["dialogue_elements"].where(ls["dialogue_elements"] > 0),
        "description_upper_share": ls["description_upper_share"],
        "description_punct_share": ls["description_punct_share"],
        "tiny_layer": ls["tiny_layer"].astype(float),
    })
    odd = pd.read_csv(CHECK / "prevalence_by_film.csv").drop_duplicates("imdb_id").set_index("imdb_id")["odd_share"]
    return q.reindex(ids), odd.reindex(ids)


def typicality_scores(f, M):
    V = np.load(DATA / "features" / "embeddings_hpc" / "script_vectors.npy")
    ids = pd.read_csv(DATA / "features" / "embeddings_hpc" / "ids.csv")["tconst"]
    pos = {t: i for i, t in enumerate(ids)}
    r = np.array([pos.get(t, -1) for t in f["imdb_id"]])
    ok = r >= 0
    X = V[r[ok]].astype(float)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Mo, tot, n = M[ok], X.sum(0), M[ok].sum(0)
    sums = Mo.T @ X
    typ = np.full(len(f), np.nan)
    vals = []
    for i in range(len(X)):
        sims = [X[i] @ ((sums[j] - X[i]) / (n[j] - 1)) / np.linalg.norm((sums[j] - X[i]) / (n[j] - 1))
                for j in np.flatnonzero(Mo[i]) if n[j] >= 2]
        allc = (tot - X[i]) / (len(X) - 1)
        vals.append((np.mean(sims) if sims else np.nan) - X[i] @ allc / np.linalg.norm(allc))
    typ[ok] = vals
    return typ


def typicality(f, keep, M, perms):
    typ = typicality_scores(f, M)
    q, odd = quality(f["imdb_id"])
    good = ~np.isnan(typ) & q.notna().all(axis=1).to_numpy()
    fo, Mo, t = f[good].reset_index(drop=True), M[good], typ[good][:, None]
    y = net_return(fo)
    Q = q[good].to_numpy()
    res = ols(y, t, ["script_typicality"], fo, Mo, keep, label="sequel flag controlled")
    res += ols(y, t, ["script_typicality"], fo, Mo, keep, extra=Q, label="+ script-quality controls")
    ns = ~sequel_flag(fo)
    res += ols(y[ns], t[ns], ["script_typicality"], fo[ns], Mo[ns], keep, extra=Q[ns], label="+ script-quality controls, sequels left out")
    od = odd[good].notna().to_numpy()
    Q2 = np.column_stack([Q[od], odd[good].to_numpy()[od]])
    res += ols(y[od], t[od], ["script_typicality"], fo[od], Mo[od], keep, extra=Q2, label="+ quality and symbol-noise controls (final 1,598-film set)")
    grp = pd.factorize(pd.Series(Mo.argmax(1)).astype(str) + "|" + fo["budget_bin"].astype(str))[0]
    obs, p = shuffle_p(y, t, 0, fo, Mo, keep, Q, grp, perms, np.random.default_rng(SEED + 2200))
    from scipy.stats import spearmanr
    corr = [{"quality_measure": c, "spearman_with_typicality": round(float(spearmanr(t[:, 0], Q[:, k])[0]), 3),
             "spearman_with_net_return": round(float(spearmanr(y, Q[:, k])[0]), 3)} for k, c in enumerate(q.columns)]
    qq = pd.qcut(t[:, 0], 5, labels=False)
    quint = [{"quintile": k + 1, "films": int((qq == k).sum()), "mean_net_return": round(float(y[qq == k].mean()), 4)} for k in range(5)]
    per_genre = [{"genre": g, "films": int((Mo[:, j] == 1).sum()),
                  "corr_typicality_net_return": round(float(np.corrcoef(t[Mo[:, j] == 1, 0], y[Mo[:, j] == 1])[0, 1]), 3)}
                 for j, g in enumerate(keep) if (Mo[:, j] == 1).sum() >= 100]
    summ = {"films": len(fo), "shuffle_coef": round(obs, 4), "shuffle_p": round(p, 4),
            "shuffle_model": "same as the '+ script-quality controls' row", "perms": perms}
    return pd.DataFrame(res), pd.DataFrame(quint), pd.DataFrame(per_genre), pd.DataFrame(corr), summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {}
    r, q, s = specialists(films, genres, f, keep, M, a.perms)
    r.to_csv(OUT / "specialists_ols.csv", index=False)
    q.to_csv(OUT / "specialists_quintiles.csv", index=False)
    summ["specialists"] = s
    print(r.to_string(index=False), q.to_string(index=False), s, sep="\n", flush=True)
    r, q, pg, qc, s = typicality(f, keep, M, a.perms)
    r.to_csv(OUT / "typicality_ols.csv", index=False)
    q.to_csv(OUT / "typicality_quintiles.csv", index=False)
    pg.to_csv(OUT / "typicality_per_genre.csv", index=False)
    qc.to_csv(OUT / "typicality_quality.csv", index=False)
    summ["typicality"] = s
    print(r.to_string(index=False), q.to_string(index=False), qc.to_string(index=False), s, sep="\n", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
