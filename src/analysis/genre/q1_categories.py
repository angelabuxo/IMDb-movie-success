"""Finer film categories two independent ways, and which of them beat their budgets. Added after the main results.

  A Wikidata genres   Wikidata's genre property (P136, features/wd_genre.tsv, from scripts/get_data.py wikidata), curated
                      by Wikidata editors, with one level of parent genres (P279, features/wd_genre_parents.tsv) so that
                      "supernatural horror film" also counts as "horror film".
                        agreement   per IMDb genre, Cohen's kappa against the matching Wikidata genre (with parents)
                        categories  every Wikidata genre (with parents) held by at least MIN_FILMS usable-money films,
                                    against the budget-decile shuffle of q1_genre.py, Holm over all of them
                        Horror      inside IMDb Horror, each Wikidata subgenre of horror with at least MIN_SUB films
                                    against other Horror (return net of budget and era, shuffled within budget deciles)
  B keyword network   TMDB keywords (data/raw/tmdb, never committed) as a network (course weeks 1, 2, 4). Rules fixed
                      before any community was looked at:
                        keywords    in at least MIN_KW usable-money films; production facts (sequel, prequel, remake,
                                    reboot, credit stingers, woman director, "based on ..."), places written with a comma,
                                    decades, and TMDB's mood words (viewer reactions such as "hilarious" or "tense", listed
                                    in MOODS) left out
                        null        curveball swaps of the film x keyword matrix, which keep every film's number of
                                    keywords and every keyword's number of films (week 2); z per pair, two-sided normal p,
                                    Holm over all pairs; the backbone is the pairs with Holm p < .05 and more films than
                                    the null expects
                        categories  Louvain on the backbone (weights = observed minus expected films) over SEEDS seeds; the
                                    best partition, its agreement with the other seeds (NMI) and its modularity against the
                                    same backbone rewired (double edge swaps, weights shuffled)
                        films       a film belongs to the categories of its kept keywords, weighted by its share of them;
                                    each category's weighted mean return against the budget-decile shuffle, Holm; NMI of
                                    each film's main category with its IMDb main genre and with its main Wikidata genre
What these measure: how films are described by curators (Wikidata) and by TMDB's users, both after release. They say
which kinds of film beat their budgets, not why.
Outputs in analysis/genre/outputs/categories/, numbers and labels only.
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.q1_categories [--null 200] [--perms 10000]
"""
from __future__ import annotations

import argparse
import json
import re

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from analysis.genre.exploratory.q1_network import curveball, labels, louvain
from analysis.genre.q1_genre import DATA, OUT as MAIN_OUT, SEED, holm, load, permuted_means, prepare, strata
from analysis.genre.q1_tmdb import kappa, net_return

OUT = MAIN_OUT / "categories"
MIN_FILMS = 40
MIN_SUB = 20
MIN_KW = 30
SEEDS = 20
WD_MATCH = {"Action": "action film", "Adventure": "adventure film", "Animation": "animated film", "Biography": "biographical film",
            "Comedy": "comedy film", "Crime": "crime film", "Drama": "drama film", "Family": "family film", "Fantasy": "fantasy film",
            "History": "historical film", "Horror": "horror film", "Music": "musical film", "Mystery": "mystery film",
            "Romance": "romance film", "Sci-Fi": "science fiction film", "Sport": "sport film", "Thriller": "thriller film", "War": "war film"}
META = {"sequel", "prequel", "remake", "reboot", "duringcreditsstinger", "aftercreditsstinger", "woman director"}
MOODS = {"absurd", "admiring", "adoring", "affectation", "aggressive", "ambiguous", "ambivalent", "amused", "angry", "antagonistic",
         "anxious", "appreciative", "assertive", "audacious", "awestruck", "baffled", "biting", "blunt", "bold", "campy", "candid",
         "cautionary", "celebratory", "cheerful", "cliché", "comforting", "compassionate", "complex", "complicated", "cruel",
         "defiant", "depressing", "distressing", "disturbed", "dramatic", "dreary", "empathetic", "enchant", "enthusiastic",
         "euphoric", "excited", "exhilarated", "familiar", "frantic", "frightened", "harsh", "hilarious", "hopeful", "horrified",
         "inspirational", "intense", "intimate", "introspective", "joyful", "joyous", "lighthearted", "macabre", "melancholy",
         "melodramatic", "mysterious", "nostalgic", "philosophical", "playful", "powerful", "provocative", "reflective",
         "sentimental", "shocking", "sinister", "somber", "suspenseful", "sympathetic", "tense", "thoughtful", "tragic", "vibrant",
         "whimsical", "witty", "wonder"}


def _clean(s):
    return s.astype(str).str.strip('"').str.replace(r'^"|"@en$|@en$', "", regex=True).str.strip('"')


def wikidata(f):
    w = pd.read_csv(DATA / "features" / "wd_genre.tsv", sep="\t")
    w.columns = ["imdb", "g", "name"]
    w["imdb"], w["name"] = _clean(w["imdb"]), _clean(w["name"])
    w = w[w["imdb"].isin(set(f["imdb_id"])) & ~w["g"].str.contains("genid") & w["name"].ne("nan")]
    p = pd.read_csv(DATA / "features" / "wd_genre_parents.tsv", sep="\t")
    p.columns = ["g", "parent", "pname"]
    p["pname"] = _clean(p["pname"])
    p = p[p["pname"].str.endswith("film") | p["pname"].isin(set(w["name"]))]       # genres, not abstract classes
    up = w.merge(p[["g", "pname"]], on="g")[["imdb", "pname"]].rename(columns={"pname": "name"})
    return pd.concat([w[["imdb", "name"]], up]).drop_duplicates()


def membership(f, pairs, names):
    idx = {t: i for i, t in enumerate(f["imdb_id"])}
    col = {n: j for j, n in enumerate(names)}
    W = np.zeros((len(f), len(names)))
    for t, n in zip(pairs["imdb"], pairs["name"]):
        if n in col and t in idx:
            W[idx[t], col[n]] = 1
    return W


def shuffle_test(f, y, W, names, perms, seed, label):
    obs = y @ W / W.sum(0)
    nl = permuted_means(y, W, strata(f, "budget"), perms, np.random.default_rng(seed))
    mu, sd = nl.mean(0), nl.std(0, ddof=1)
    p = (1 + (np.abs(nl - mu) >= np.abs(obs - mu) - 1e-12).sum(0)) / (1 + perms)
    return [{"run": label, "category": n, "films": round(float(W[:, j].sum()), 1), "mean": round(obs[j], 4),
             "excess": round(obs[j] - mu[j], 4), "z": round((obs[j] - mu[j]) / sd[j], 2), "p_holm": round(h, 5)}
            for j, (n, h) in enumerate(zip(names, holm(p)))]


def part_wikidata(f, keep, M, perms):
    wd = wikidata(f)
    counts = wd["name"].value_counts()
    agree = []
    for g, name in WD_MATCH.items():
        if g in keep and name in counts:
            Wg = membership(f, wd, [name])[:, 0]
            i_ = M[:, keep.index(g)]
            agree.append({"imdb_genre": g, "wikidata_genre": name, "imdb_films": int(i_.sum()), "wikidata_films": int(Wg.sum()),
                          "kappa": round(float(kappa(i_, Wg)), 3), "p_wikidata_given_imdb": round(float((i_ * Wg).sum() / i_.sum()), 3)})
    names = sorted(counts[counts >= MIN_FILMS].index)
    W = membership(f, wd, names)
    rows = shuffle_test(f, f["ln_gross_over_budget"].to_numpy(), W, names, perms, SEED + 4000, "Wikidata genre")
    # Horror subgenres: Wikidata genres whose parent is horror film, inside IMDb Horror
    p = pd.read_csv(DATA / "features" / "wd_genre_parents.tsv", sep="\t")
    p.columns = ["g", "parent", "pname"]
    w = pd.read_csv(DATA / "features" / "wd_genre.tsv", sep="\t")
    w.columns = ["imdb", "g", "name"]
    sub_g = set(p.loc[_clean(p["pname"]) == "horror film", "g"])
    subs = sorted(set(_clean(w.loc[w["g"].isin(sub_g), "name"])))
    h = M[:, keep.index("Horror")] == 1
    fh = f[h].reset_index(drop=True)
    Wh = membership(fh, wd, subs)
    ok = Wh.sum(0) >= MIN_SUB
    sub_rows = []
    if ok.any():
        yh = net_return(f)[h]
        sub_rows = shuffle_test(fh, yh, Wh[:, ok], list(np.array(subs)[ok]), perms, SEED + 4010, "Wikidata horror subgenre, inside IMDb Horror")
        for r in sub_rows:
            c = Wh[:, subs.index(r["category"])]
            r["median"], r["median_other_horror"] = round(float(np.median(yh[c == 1])), 4), round(float(np.median(yh[c == 0])), 4)
    main_wd = pd.Series(np.array(names)[W.argmax(1)], index=f["imdb_id"]).where(W.sum(1) > 0)
    return pd.DataFrame(agree), pd.DataFrame(rows), pd.DataFrame(sub_rows), main_wd, {"films_with_wikidata_genre": int(f["imdb_id"].isin(set(wd["imdb"])).sum()),
                                                                                         "distinct_genres": int(counts.size), "genres_tested": len(names)}


def keywords(f):
    t = pd.read_csv(DATA / "tmdb" / "tmdb_project_movies.csv", usecols=["imdb_id", "keywords"]).drop_duplicates("imdb_id").set_index("imdb_id")
    kw = t["keywords"].reindex(f["imdb_id"]).fillna("").map(lambda s: {x.strip().lower() for x in s.split("|") if x.strip()})
    from collections import Counter
    c = Counter(k for s in kw for k in s)
    def drop(k):
        return (k in META or k in MOODS or k.startswith("based on ") or ", " in k or re.fullmatch(r"\d{4}s?", k) is not None)
    vocab = sorted(k for k, v in c.items() if v >= MIN_KW and not drop(k))
    vi = {k: j for j, k in enumerate(vocab)}
    # sorted, so each film's keyword ids go in in the same order in every process (string hashing is randomised per
    # process, and the curveball null depends on set order): without it the backbone changed from run to run
    rows = [set(vi[k] for k in sorted(s) if k in vi) for s in kw]
    return vocab, rows, {"keywords_in_min_films": sum(v >= MIN_KW for v in c.values()), "kept": len(vocab)}


def to_sparse(rows, k):
    r = [i for i, s in enumerate(rows) for _ in s]
    c = [j for s in rows for j in s]
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows), k))


def part_keywords(f, keep, M, main_wd, n_null, perms):
    from scipy.stats import norm
    from sklearn.metrics import normalized_mutual_info_score as nmi
    vocab, rows, info = keywords(f)
    k = len(vocab)
    X = to_sparse(rows, k)
    C = (X.T @ X).toarray()
    rng = np.random.default_rng(SEED + 4100)
    work = [set(s) for s in rows]
    work = curveball(work, 10 * len(work), rng)
    s1, s2 = np.zeros((k, k)), np.zeros((k, k))
    for _ in range(n_null):
        work = curveball(work, len(work), rng)
        Cn = (to_sparse(work, k).T @ to_sparse(work, k)).toarray()
        s1 += Cn
        s2 += Cn ** 2
    mu = s1 / n_null
    sd = np.sqrt(np.maximum(s2 / n_null - mu ** 2, 1e-12) * n_null / (n_null - 1))
    z = (C - mu) / sd
    iu = np.triu_indices(k, 1)
    ph = np.ones_like(z)
    ph[iu] = holm(2 * norm.sf(np.abs(z[iu])))
    back = (ph < .05) & (C > mu)
    np.fill_diagonal(back, False)
    G = nx.Graph()
    G.add_nodes_from(vocab)
    for i, j in zip(*iu):
        if back[i, j]:
            G.add_edge(vocab[i], vocab[j], weight=float(C[i, j] - mu[i, j]))
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    parts, mods = louvain(G, SEEDS)
    best = int(np.argmax(mods))
    part = sorted([c for c in parts[best] if len(c) >= 3], key=len, reverse=True)
    nodes = list(G)
    labs = [labels(pp, nodes) for pp in parts]
    stab = float(np.mean([nmi(labs[best], l) for i, l in enumerate(labs) if i != best]))
    w = np.array([d["weight"] for *_, d in G.edges(data=True)])
    null_mod = []
    for s in range(50):
        Gn = nx.Graph(G)
        nx.double_edge_swap(Gn, nswap=5 * Gn.number_of_edges(), max_tries=10 ** 6, seed=SEED + s)
        for (u, v), wt in zip(Gn.edges(), rng.permutation(w)):
            Gn[u][v]["weight"] = float(wt)
        null_mod.append(max(louvain(Gn, 2)[1]))
    strength = dict(G.degree(weight="weight"))
    cats = []
    for c_i, c in enumerate(part):
        top = sorted(c, key=lambda n: -strength[n])[:8]
        cats.append({"category": c_i + 1, "keywords": len(c), "top_keywords": ", ".join(top)})
    # films: share of their kept keywords in each category
    kix = {n: i for i, n in enumerate(vocab)}
    F = np.zeros((len(f), len(part)))
    for c_i, c in enumerate(part):
        cols = [kix[n] for n in c]
        F[:, c_i] = np.asarray(X[:, cols].sum(1)).ravel()
    tot = F.sum(1, keepdims=True)
    has = tot[:, 0] >= 2
    Fw = np.where(tot > 0, F / np.maximum(tot, 1), 0)
    fs, ys, Ws = f[has].reset_index(drop=True), f["ln_gross_over_budget"].to_numpy()[has], Fw[has]
    names = [f"{c['category']}: {c['top_keywords'].split(', ')[0]}" for c in cats]
    rows_t = shuffle_test(fs, ys, Ws, names, perms, SEED + 4200, "keyword category (weighted by share of a film's keywords)")
    main_c = Ws.argmax(1)
    imdb_main = M[has].argmax(1)       # alphabetical first IMDb genre is arbitrary, so use the rarest genre a film has instead
    rar = M.sum(0)
    imdb_rare = np.array([np.argmin(np.where(m > 0, rar, np.inf)) for m in M[has]])
    wdm = main_wd.reindex(fs["imdb_id"]).to_numpy()
    okw = pd.notna(wdm)
    summ = {**info, "null_samples": n_null, "backbone_links": int(back[iu].sum()), "nodes_in_backbone": G.number_of_nodes(),
            "louvain_seeds": SEEDS, "modularity": round(mods[best], 4), "mean_nmi_best_vs_other_seeds": round(stab, 3),
            "rewired_modularity_mean": round(float(np.mean(null_mod)), 4), "rewired_modularity_max": round(float(np.max(null_mod)), 4),
            "modularity_p": round(float((1 + sum(m >= mods[best] for m in null_mod)) / (1 + len(null_mod))), 4),
            "categories": len(part), "films_with_two_kept_keywords_in_categories": int(has.sum()),
            "nmi_main_category_vs_rarest_imdb_genre": round(float(nmi(main_c, imdb_rare)), 3),
            "nmi_main_category_vs_main_wikidata_genre": round(float(nmi(main_c[okw], pd.factorize(wdm[okw])[0])), 3)}
    del imdb_main
    return pd.DataFrame(cats), pd.DataFrame(rows_t), summ, G, part


def figure(G, part, tests, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ex = {int(r["category"].split(":")[0]): r["excess"] for _, r in tests.iterrows()}
    keepn = [n for c in part for n in c]
    H = G.subgraph(keepn)
    pos = nx.spring_layout(H, weight="weight", seed=SEED, k=.35, iterations=80)
    cmap = matplotlib.colormaps["RdBu"]
    lim = max(abs(v) for v in ex.values())
    fig, ax = plt.subplots(figsize=(9, 8))
    nx.draw_networkx_edges(H, pos, ax=ax, width=.3, edge_color="#bbb", alpha=.5)
    strength = dict(H.degree(weight="weight"))
    for c_i, c in enumerate(part):
        col = cmap(.5 + ex[c_i + 1] / (2 * lim))
        nl = list(c)
        nx.draw_networkx_nodes(H, pos, nodelist=nl, node_size=[8 + strength[n] / 40 for n in nl], node_color=[col], ax=ax,
                               edgecolors="#333", linewidths=.3)
        for n in sorted(nl, key=lambda n: -strength[n])[:3]:
            ax.text(*pos[n], n, fontsize=6, ha="center", va="bottom")
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(-lim, lim))
    fig.colorbar(sm, ax=ax, shrink=.5, label="category's excess ln(gross / budget), budget-matched")
    ax.set_title("TMDB keywords linked when films share them more than a curveball null expects; colour = Louvain category's return", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--perms", type=int, default=10000)
    a = ap.parse_args()
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    OUT.mkdir(parents=True, exist_ok=True)
    agree, wd_tests, wd_horror, main_wd, s_wd = part_wikidata(f, keep, M, a.perms)
    agree.to_csv(OUT / "wikidata_agreement.csv", index=False)
    wd_tests.to_csv(OUT / "wikidata_genre_tests.csv", index=False)
    wd_horror.to_csv(OUT / "wikidata_horror_subgenres.csv", index=False)
    print(json.dumps(s_wd), agree.to_string(index=False), flush=True)
    cats, kw_tests, s_kw, G, part = part_keywords(f, keep, M, main_wd, a.null, a.perms)
    cats.to_csv(OUT / "keyword_categories.csv", index=False)
    kw_tests.to_csv(OUT / "keyword_category_tests.csv", index=False)
    figure(G, part, kw_tests, OUT / "figures" / "q1_keyword_network")
    (OUT / "summary.json").write_text(json.dumps({"wikidata": s_wd, "keyword_network": s_kw, "perms": a.perms}, indent=1))
    print(json.dumps(s_kw, indent=1))
    print(cats.to_string(index=False))
    print(kw_tests.sort_values("excess").to_string(index=False))
    sig = wd_tests[wd_tests["p_holm"] < .05].sort_values("excess")
    print(sig.to_string(index=False))
    print(wd_horror.to_string(index=False))


if __name__ == "__main__":
    main()
