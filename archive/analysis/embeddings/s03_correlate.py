"""Step 3: do the vectors go with profit and revenue?

    python -m analysis.embeddings.s03_correlate

For each corpus (reviews, screenplays, summaries), each text variant (raw, clean) and each representation
(the sentence-embedding model, and TF-IDF as a second method) it measures:

- alone: how well the vectors by themselves predict ln(gross/budget), ln(gross) and ln(budget) on films
  the model didn't see (out-of-fold R2 and correlation), against a plain shuffle;
- net: what the vectors add once budget, year and genre are known, against a shuffle inside genre x decade
  cells. With budget controlled, profit and revenue are the same question: ln(gross) = ln(multiple) + ln(budget);
- time: the same, fitted on the earliest 70% of films and scored on the later ones;
- network: Newman's assortativity of the outcome on the nearest-neighbour network of the vectors.

Writes outputs/s03_results.csv (one row per measurement), s03_nulls.json (the shuffle distributions behind
the figures), s03_words.csv (words that separate the two ends of each direction) and s03_summary.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

from . import stats as S
from .paths import OUTPUTS, PRIVATE, VECTORS, cfg, film_table

CORPORA = ("reviews", "screenplays", "summaries")
TARGETS = {"log_mult": "profitability: ln(gross / budget)", "log_gross": "revenue: ln(gross)",
           "log_budget": "budget: ln(budget)"}


def controls(d: pd.DataFrame, with_rating: bool = False, franchise: bool = False, year_sq: bool = True,
             extra: str | None = None) -> np.ndarray:
    k = cfg()["controls"]
    g = d.genres.fillna("").str.get_dummies(sep=",")
    g = g[[c for c in g if k["min_films_per_genre"] <= g[c].sum() <= len(d) - k["min_films_per_genre"]]]
    yr = (d.year - d.year.mean()) / 10
    lb = d.log_budget - d.log_budget.mean()
    cols = [np.ones(len(d)), lb, lb ** 2, yr] + ([yr ** 2] if year_sq else []) + [g.to_numpy(float)]
    if franchise:
        cols.append(d.franchise.to_numpy(float))
    if extra:
        cols.append(d[extra].to_numpy(float))
    if with_rating:
        cols.append(d.rating.fillna(d.rating.mean()).to_numpy(float))
    return np.column_stack(cols)


def cells(d: pd.DataFrame) -> np.ndarray:
    """Genre x decade cells for the stratified shuffle; small cells pooled."""
    m = cfg()["controls"]["min_films_per_cell"]
    c = d.primary_genre + " | " + d.decade_bin
    c = c.where(c.map(c.value_counts()) >= m, d.primary_genre + " | pooled")
    c = c.where(c.map(c.value_counts()) >= m, "pooled")
    return pd.factorize(c)[0]


def film_texts(corpus: str, variant: str) -> pd.Series:
    df = pd.read_parquet(PRIVATE / f"{corpus}.parquet", columns=["tconst", variant])
    return df.groupby("tconst")[variant].agg(" ".join)


def tfidf_vectors(texts: pd.Series) -> np.ndarray:
    t, c = cfg()["tfidf"], cfg()["clean"]
    stop = list(ENGLISH_STOP_WORDS | {c["name_placeholder"].lower(), "film", "movie", "films", "movies"})
    X = TfidfVectorizer(lowercase=True, token_pattern=c["token_regex"], stop_words=stop, min_df=t["min_df"],
                        max_df=t["max_df"], max_features=t["max_features"], sublinear_tf=t["sublinear_tf"]).fit_transform(texts)
    return TruncatedSVD(t["svd_components"], random_state=cfg()["seed"]).fit_transform(X)


def ocr_damage() -> pd.Series:
    """Share of OCR-damaged words per screenplay (clean.junk_share), cached in private/."""
    p = PRIVATE / "screenplay_ocr.parquet"
    if not p.exists():
        from .clean import junk_share
        s = pd.read_parquet(PRIVATE / "screenplays.parquet", columns=["tconst", "clean"])
        s.assign(ocr=s.clean.map(junk_share))[["tconst", "ocr"]].to_parquet(p, index=False)
    return pd.read_parquet(p).set_index("tconst").ocr


def load(corpus: str, variant: str, rep: str, films: pd.DataFrame,
         drop_ocr: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Films and vectors of one corpus. Screenplays with OCR damage above the cutoff are left out unless
    drop_ocr is False (then the damage share is a column, for the check that controls for it)."""
    ids = pd.read_csv(VECTORS / f"{corpus}_ids.csv", dtype=str).tconst
    if rep == "embedding":
        V = np.load(VECTORS / f"{corpus}_{variant}.npy").astype(float)
    else:
        V = tfidf_vectors(film_texts(corpus, variant).reindex(ids))
    d = films.set_index("tconst").loc[ids].reset_index()
    if corpus == "screenplays":
        d["ocr"] = d.tconst.map(ocr_damage()).to_numpy()
        if drop_ocr:
            keep = (d.ocr <= cfg()["post_hoc"]["max_ocr_damaged_word_share"]).to_numpy()
            d, V = d[keep].reset_index(drop=True), V[keep]
    return d, V


def measure(d, V, tag: dict, full: bool, nulls: dict, keep_null: str | None = None) -> tuple[list[dict], np.ndarray]:
    """Every measurement for one set of films and vectors. Returns result rows and the out-of-fold
    'what the vectors add' score per film (used for the word lists)."""
    a, seed = cfg()["analysis"], cfg()["seed"]
    alphas = np.logspace(*a["ridge_alphas"])
    k = min(a["pca_components"], V.shape[1], len(d) // 10)
    Z = PCA(k, random_state=seed).fit_transform(V)      # unsupervised, so it never sees the outcomes
    ones, C, strata = np.ones((len(d), 1)), controls(d), cells(d)
    rows = []

    def add(analysis, target, **kw):
        rows.append({**tag, "n": len(d), "analysis": analysis, "target": target, **kw})

    def repeated(y, ctrl):
        runs = [S.oof(Z, y, ctrl, alphas, a["folds"], seed + r) for r in range(a["repeats"])]
        sc = pd.DataFrame([S.summarise(y, b, e) for b, e in runs]).mean().to_dict()
        return sc, runs[0], np.mean([e for _, e in runs], axis=0)

    # alone
    for target in (TARGETS if full else ["log_mult", "log_gross"]):
        y = d[target].to_numpy()
        sc, _, _ = repeated(y, ones)
        null = S.shuffle_gains(Z, y, ones, np.zeros(len(d), int), alphas, a["folds"], a["permutations"] if full else 200, seed)
        add("alone", target, r2=sc["r2_full"], pearson=sc["partial_pearson"], spearman=sc["partial_spearman"],
            null_mean=float(np.mean(null)), null_p95=float(np.percentile(null, 95)), p=S.p_upper(sc["r2_gain"], null))

    # net of budget, year and genre
    y = d.log_mult.to_numpy()
    sc, (b0, e0), score = repeated(y, C)
    null = S.shuffle_gains(Z, y, C, strata, alphas, a["folds"], a["permutations"] if full else 200, seed)
    lo, hi = S.bootstrap_gain(y, b0, e0, a["bootstrap"], seed)
    add("net", "log_mult", r2_base=sc["r2_base"], r2=sc["r2_full"], r2_gain=sc["r2_gain"], gain_lo=lo, gain_hi=hi,
        pearson=sc["partial_pearson"], spearman=sc["partial_spearman"], null_mean=float(np.mean(null)),
        null_p95=float(np.percentile(null, 95)), p=S.p_upper(sc["r2_gain"], null))
    if keep_null:
        nulls[keep_null] = {"gain": sc["r2_gain"], "null": [round(x, 5) for x in null]}

    # ... and of the franchise flag: reviews of sequels say "sequel", so without it the text can just
    # detect franchise films, which the budget doesn't fully capture.
    CF = controls(d, franchise=True)
    sc, (b0, e0), score = repeated(y, CF)
    null = S.shuffle_gains(Z, y, CF, strata, alphas, a["folds"], a["permutations"] if full else 200, seed)
    lo, hi = S.bootstrap_gain(y, b0, e0, a["bootstrap"], seed)
    add("net_of_franchise_too", "log_mult", r2_base=sc["r2_base"], r2=sc["r2_full"], r2_gain=sc["r2_gain"],
        gain_lo=lo, gain_hi=hi, pearson=sc["partial_pearson"], spearman=sc["partial_spearman"],
        null_mean=float(np.mean(null)), null_p95=float(np.percentile(null, 95)), p=S.p_upper(sc["r2_gain"], null))
    if keep_null:
        nulls[keep_null + "_franchise"] = {"gain": sc["r2_gain"], "null": [round(x, 5) for x in null]}
    if not full:
        return rows, score

    # net of the above and of the IMDb rating: is it more than "people liked it"?
    CR = controls(d, with_rating=True, franchise=True)
    sc, _, _ = repeated(y, CR)
    null = S.shuffle_gains(Z, y, CR, strata, alphas, a["folds"], 200, seed)
    add("net_of_rating_too", "log_mult", r2_base=sc["r2_base"], r2=sc["r2_full"], r2_gain=sc["r2_gain"],
        spearman=sc["partial_spearman"], null_mean=float(np.mean(null)), null_p95=float(np.percentile(null, 95)),
        p=S.p_upper(sc["r2_gain"], null))

    # fitted on early films, scored on later ones
    # year squared is left out: fitted on early films it extrapolates wildly past the training years
    ts = S.time_split(Z, y, controls(d, franchise=True, year_sq=False), d.year.to_numpy(),
                      a["time_split_train_share"], alphas)
    add("time_split", "log_mult", r2_base=ts["r2_base"], r2=ts["r2_full"], r2_gain=ts["r2_gain"],
        spearman=ts["partial_spearman"], n_test=ts["n_test"], cutoff_year=ts["cutoff_year"])

    # number of components
    for kk in a["pca_grid"]:
        if kk == k or kk > min(V.shape[1], len(d) // 10):
            continue
        Zk = PCA(kk, random_state=seed).fit_transform(V)
        g = np.mean([S.cv_scores(Zk, y, CF, alphas, a["folds"], seed + r)["r2_gain"] for r in range(3)])
        add(f"net_k{kk}", "log_mult", r2_gain=float(g))

    # the similarity network
    edges = S.knn_edges(V, a["knn"])
    beta, *_ = np.linalg.lstsq(CF, y, rcond=None)
    for name, attr, st in (("log_mult", y, np.zeros(len(d), int)), ("log_gross", d.log_gross.to_numpy(), np.zeros(len(d), int)),
                           ("log_mult_net_of_controls", y - CF @ beta, strata)):
        obs = S.assortativity(edges, attr)
        null = S.assortativity_null(edges, attr, st, a["permutations"], seed)
        add("network_assortativity", name, assortativity=obs, null_mean=float(np.mean(null)),
            null_p95=float(np.percentile(null, 95)), z=float((obs - np.mean(null)) / np.std(null)),
            p=S.p_upper(obs, null), edges=len(edges))
    return rows, score


def ocr_check(films: pd.DataFrame, rep: str, tag: dict) -> list[dict]:
    """Does OCR damage itself go with profit, and does the screenplay result hold on every script once the
    damage share is a control?"""
    a, seed = cfg()["analysis"], cfg()["seed"]
    d, V = load("screenplays", "clean", rep, films, drop_ocr=False)
    y, CF = d.log_mult.to_numpy(), controls(d, franchise=True)
    beta, *_ = np.linalg.lstsq(CF, y, rcond=None)
    rho, p = sps.spearmanr(d.ocr, y - CF @ beta)
    alphas = np.logspace(*a["ridge_alphas"])
    Z = PCA(min(a["pca_components"], V.shape[1]), random_state=seed).fit_transform(V)
    CO = controls(d, franchise=True, extra="ocr")
    g = np.mean([S.cv_scores(Z, y, CO, alphas, a["folds"], seed + r)["r2_gain"] for r in range(a["repeats"])])
    null = S.shuffle_gains(Z, y, CO, cells(d), alphas, a["folds"], 200, seed)
    base = {**tag, "sample": "every script, OCR damage as a control", "n": len(d), "target": "log_mult"}
    return [{**base, "analysis": "ocr_vs_profit", "spearman": float(rho), "p": float(p),
             "damaged": int((d.ocr > cfg()["post_hoc"]["max_ocr_damaged_word_share"]).sum())},
            {**base, "analysis": "net_of_franchise_too", "r2_gain": float(g), "null_mean": float(np.mean(null)),
             "null_p95": float(np.percentile(null, 95)), "p": S.p_upper(g, null)}]


def words(corpus: str, ids: pd.Series, score: np.ndarray) -> pd.DataFrame:
    """Words found in more films at one end of the 'what the vectors add' score than at the other: the top
    fifth against the bottom fifth, by the share of films that use the word at all."""
    a, c = cfg()["analysis"], cfg()["clean"]
    texts = film_texts(corpus, "clean").reindex(ids)
    lo, hi = np.quantile(score, [0.2, 0.8])
    stop = list(ENGLISH_STOP_WORDS | {c["name_placeholder"].lower()})
    cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b[A-Za-z][A-Za-z]{2,}\b", stop_words=stop, binary=True,
                         min_df=a["word_min_doc_freq"])
    X = cv.fit_transform(texts)
    top, bot = np.asarray(X[score >= hi].mean(axis=0)).ravel(), np.asarray(X[score <= lo].mean(axis=0)).ravel()
    t = pd.DataFrame({"corpus": corpus, "word": cv.get_feature_names_out(), "share_top_fifth": top,
                      "share_bottom_fifth": bot, "log10_ratio": np.log10((top + 0.005) / (bot + 0.005))})
    t = t[(t.share_top_fifth + t.share_bottom_fifth) >= 0.05].sort_values("log10_ratio")
    out = pd.concat([t.tail(a["top_words"]).assign(end="more profit than expected"),
                     t.head(a["top_words"]).assign(end="less profit than expected")])
    return out.round(4)


def run() -> pd.DataFrame:
    films = pd.read_parquet(film_table())
    rows, nulls, wordlists, scores = [], {}, [], {}
    have = {c: set(pd.read_csv(VECTORS / f"{c}_ids.csv", dtype=str).tconst) for c in CORPORA}
    common = set.intersection(*have.values())
    for corpus in CORPORA:
        for rep, variant in (("embedding", "clean"), ("embedding", "raw"), ("tfidf", "clean"), ("tfidf", "raw")):
            d, V = load(corpus, variant, rep, films)
            tag = {"corpus": corpus, "variant": variant, "representation": rep}
            primary = rep == "embedding"
            r, score = measure(d, V, {**tag, "sample": "all clean films"}, primary, nulls,
                               f"{corpus}_{variant}" if primary else None)
            rows += r
            print(corpus, variant, rep, len(d), flush=True)
            if primary and variant == "clean":
                wordlists.append(words(corpus, d.tconst, score))
                scores[corpus] = pd.Series(score, index=d.tconst)
            if variant != "clean":
                continue
            for name, mask in (("100k+ votes only", d.tier == "100k+"), ("no franchise films", d.franchise == 0),
                               ("films in all three corpora", d.tconst.isin(common))):
                m = mask.to_numpy()
                rows += measure(d[m].reset_index(drop=True), V[m], {**tag, "sample": name}, False, nulls)[0]
            if corpus == "screenplays":
                rows += ocr_check(films, rep, tag)
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS / "s03_results.csv", index=False, float_format="%.5f")
    (OUTPUTS / "s03_nulls.json").write_text(json.dumps(nulls) + "\n")
    pd.concat(wordlists).to_csv(OUTPUTS / "s03_words.csv", index=False)
    # Do the three corpora point the same way? Correlation of their "adds" scores on shared films.
    sc = pd.DataFrame(scores).dropna()
    agree = {f"{a} ~ {b}": float(sps.spearmanr(sc[a], sc[b])[0]) for i, a in enumerate(CORPORA) for b in CORPORA[i + 1:]}
    (OUTPUTS / "s03_summary.json").write_text(json.dumps(
        {"films_in_all_three_corpora": len(common), "films_per_corpus": {c: len(v) for c, v in have.items()},
         "score_agreement_spearman_on_shared_films": agree, "shared_films_for_agreement": len(sc)}, indent=1) + "\n")
    pd.DataFrame(scores).to_parquet(PRIVATE / "scores.parquet")
    return res


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 400)
    print(r[r["sample"] == "all clean films"].drop(columns=["sample"]).to_string())
