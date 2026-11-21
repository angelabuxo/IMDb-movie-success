"""Write and execute notebooks/q1_genre.ipynb, which walks through the saved outputs of analysis/genre/q1_genre.py
and analysis/genre/q1_checks.py. It reads only analysis/genre/outputs/, so it needs no raw data. Run from the
repo root:

    python -m analysis.genre.build_notebook            # write and execute
    python -m analysis.genre.build_notebook --no-run   # write only

No number is typed into the notebook's text: every number in it is printed by a cell from the saved outputs.
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "q1_genre.ipynb"

CELLS = [
    ("md", """# Q1. Do some genres earn more than others at the same budget?

This notebook asks whether films of some genres return more on their budget than films of the same budget in
other genres. The return is ln(gross / budget): 0 means the gross equalled the budget, 0.69 means it doubled it.
Gross over budget isn't profit, because marketing isn't in the budget and cinemas keep part of the gross.

It reads the saved outputs of `python -m analysis.genre.q1_genre` (the main test) and
`python -m analysis.genre.q1_checks` (five follow-up checks) and needs no raw data. The method and its limits are
in [`analysis/genre/README.md`](../analysis/genre/README.md) and [`CHECKS.md`](../analysis/genre/CHECKS.md).

Sections:
1. The films and the settings
2. The main test: genres against budget-matched shuffles
3. Does the null matter? Four versions of the shuffle
4. A second method: regression
5. On the common set (the films Q2 and Q3 also use)
6. Check 1: a smooth budget curve instead of bins
7. Check 2: genre x budget tier
8. Check 3: without each genre's biggest hits
9. Check 4: matched pairs
10. Check 5: hit share and the share below budget
11. The genre network: co-occurrence, null, backbone and families (weeks 1, 2, 4)
12. Do films made by the same people share genre? Assortativity (week 3)
13. Other ways to say what genre a film is: Tag Genome shares, plot shares, plot clusters (weeks 6, 7)
14. Inside Horror: which plot words go with beating the budget (week 6)
15. The genre distribution (Tag Genome shares) through the same analyses
16. Fixing the genre distribution's weaknesses
17. Genre with the people and the screenplays: specialists, script typicality, Horror scripts
18. Summary"""),
    ("code", """import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path.cwd().resolve().parent   # notebooks -> repo root
SRC = REPO / "analysis" / "genre" / "outputs"
CHK = SRC / "checks"
S = json.loads((SRC / "summary.json").read_text())
sh = pd.read_csv(SRC / "genre_shuffle.csv")
ols = pd.read_csv(SRC / "genre_ols.csv")
overlap = pd.read_csv(SRC / "overlap.csv")
smooth = pd.read_csv(CHK / "smooth.csv")
tiers = pd.read_csv(CHK / "tiers.csv")
trim = pd.read_csv(CHK / "trim.csv")
tops = pd.read_csv(CHK / "top_films.csv")
matched = pd.read_csv(CHK / "matched.csv")
shares = pd.read_csv(CHK / "shares.csv")
EXP = REPO / "analysis" / "genre" / "exploratory" / "outputs"   # weak evidence, not used for claims
BR = SRC / "before_release"
NET, SHR, WRD = EXP / "network", EXP / "shares", EXP / "words"
SN = json.loads((NET / "summary.json").read_text())
pairs = pd.read_csv(NET / "genre_pairs.csv")
families = pd.read_csv(NET / "genre_families.csv")
assort = pd.read_csv(NET / "film_assortativity.csv")
SS = json.loads((SHR / "summary.json").read_text())
share_tests = pd.read_csv(SHR / "share_tests.csv")
agreement = pd.read_csv(SHR / "agreement.csv")
cv = pd.read_csv(SHR / "cv_r2.csv")
clusters = pd.read_csv(SHR / "plot_clusters.csv")
SW = json.loads((WRD / "horror_summary.json").read_text())
wtop = pd.read_csv(WRD / "horror_top.csv")
wlists = pd.read_csv(WRD / "horror_list_null.csv")
from IPython.display import Image, display
SF = EXP / "share_followups"
SFS = json.loads((SF / "summary.json").read_text())
FX = EXP / "share_fixes"
FXS = json.loads((FX / "summary.json").read_text())
BRS = json.loads((BR / "summary.json").read_text())
HSC = EXP / "horror_scripts"
HSS = json.loads((HSC / "summary.json").read_text())

BLUE, RED, GREY, INK = "#2f6db5", "#c0504d", "#b8b8b8", "#222"
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
pd.options.display.float_format = "{:,.3f}".format

def view(set_="usable_money", outcome="ln_ratio", null="budget"):
    return sh[(sh["set"] == set_) & (sh["outcome"] == outcome) & (sh["null"] == null)].set_index("genre")

main = view()
SEVEN = main.index[main["p_holm"] < .05].intersection(
    view(null="budget_era").index[view(null="budget_era")["p_holm"] < .05]).intersection(
    view(null="none").index[view(null="none")["p_holm"] < .05]).intersection(
    view(outcome="rank").index[view(outcome="rank")["p_holm"] < .05]).tolist()
def colour(r):
    return GREY if r["p_holm"] >= .05 else (BLUE if r["excess"] > 0 else RED)"""),
    ("md", """## 1. The films and the settings

The films are every film with `usable_money` in the shared analysis set (budget and gross both in US dollars,
the shared clean rule from `analysis/clean_rule.py`, no flagged figures). A film counts in every IMDb genre it
has, and genres with fewer than the minimum number of films are left out. The settings were fixed in the script
before any genre result was looked at."""),
    ("code", """for s, v in S["sets"].items():
    print(f"{s}: {v['films']:,} films, {len(v['genres_kept'])} genres with at least {S['min_films']} films, "
          f"{v['mean_genres_per_film']} genres per film, mean ln(gross/budget) {v['all_film_mean_ln_ratio']:.3f}")
print(f"shuffles: {S['perms']:,}, budget deciles: {S['budget_bins']}, seed {S['seed']}")"""),
    ("md", """Genres overlap a lot, so the tests below aren't independent. The largest overlaps, as the share of the first
genre's films that also carry the second:"""),
    ("code", """overlap.sort_values("share_of_a", ascending=False).head(8).reset_index(drop=True)"""),
    ("md", """## 2. The main test

For each genre, the mean return of its films is compared with shuffles in which returns are swapped only
between films in the same decile of ln(budget). The genre tags stay where they are. So the band shows what the
genre's films would return on average if genre didn't matter, given their budgets. A bar outside the band (Holm
p < .05 over the genres) means the genre returns more, or less, than its budgets predict."""),
    ("code", """d = main.sort_values("mean")
fig, ax = plt.subplots(figsize=(7.2, 5.6))
yy = np.arange(len(d))
ax.barh(yy, d["mean"], color=[colour(r) for _, r in d.iterrows()], height=.55, zorder=2)
for k, (_, r) in enumerate(d.iterrows()):
    ax.add_patch(plt.Rectangle((r["null_lo95"], k - .38), r["null_hi95"] - r["null_lo95"], .76,
                               facecolor="#000", alpha=.12, edgecolor=INK, lw=.7, zorder=3))
overall = S["sets"]["usable_money"]["all_film_mean_ln_ratio"]
ax.axvline(overall, color="#555", ls=":", lw=.8)
ax.set_yticks(yy, [f"{g} ({n})" for g, n in zip(d.index, d["n_films"])], fontsize=8)
ax.set_xlabel("mean ln(gross / budget)")
ax.legend(handles=[Patch(color=BLUE, label="above the band"), Patch(color=RED, label="below the band"),
                   Patch(color=GREY, label="inside"), Patch(facecolor="#000", alpha=.12, edgecolor=INK, label="95% of budget-matched shuffles"),
                   Line2D([], [], color="#555", ls=":", label="all-film average")], fontsize=7, loc="lower right", frameon=False)
ax.set_title("Genre returns against returns shuffled within budget deciles", fontsize=10)
plt.show()"""),
    ("code", """t = main[["n_films", "mean", "null_mean", "excess", "z", "p_holm"]].copy()
t["times"] = np.exp(t["excess"])
t.rename(columns={"null_mean": "expected", "times": "x gross/budget"}).sort_values("excess", ascending=False)"""),
    ("md", """`excess` is the genre's mean minus what its budgets predict, and `x gross/budget` turns it into a factor: a
factor above 1 means the genre's films return that much more gross per dollar of budget than films of the same
budgets (1.2 would be 20 % more)."""),
    ("md", """## 3. Does the null matter?

The same test with three other nulls: no budget matching at all, matching on budget decile x era (returns fall a
lot by decade), and the budget-matched shuffle on the percentile rank of the return, where each film counts once
however extreme it is. The genres that pass all four are the result we report."""),
    ("code", """rows = {}
for null, outcome, name in [("none", "ln_ratio", "no matching"), ("budget", "ln_ratio", "budget"),
                            ("budget_era", "ln_ratio", "budget x era"), ("budget_fine", "ln_ratio", "50 budget bins"),
                            ("budget", "rank", "ranks")]:
    v = view(outcome=outcome, null=null)
    rows[(name, "excess")] = v["excess"]
    rows[(name, "p_holm")] = v["p_holm"]
pd.DataFrame(rows).sort_values(("budget", "excess"), ascending=False)"""),
    ("code", """print("Pass all four (no matching, budget, budget x era, ranks):", ", ".join(SEVEN))
raw, matched_b = view(null="none")["excess"], main["excess"]
same = (np.sign(raw) == np.sign(matched_b.reindex(raw.index))).sum()
print(f"Same sign with and without budget matching: {same} of {len(raw)} genres")
print(f"Horror: {raw['Horror']:+.3f} without matching, {matched_b['Horror']:+.3f} with budget deciles, "
      f"{view(null='budget_fine').loc['Horror', 'excess']:+.3f} with 50 budget bins")"""),
    ("md", """Budget matching changes little. The genres that win aren't winning just because they're cheap. (The 50-bin
shuffle was added after the main results, when the lowest budget decile turned out to be very wide.)"""),
    ("md", """## 4. A second method: regression

An OLS of ln(gross / budget) on all genres at once (multi-hot), budget decile and era, with HC3 errors. Its genre
coefficients hold the film's other genres fixed, so they answer a slightly different question. Almost every film
has at least one of the kept genres, so the raw coefficients sit against a near-empty reference; the centred
coefficient (minus the mean over genres) is the one to compare."""),
    ("code", """o = ols[ols["set"] == "usable_money"].set_index("genre").sort_values("coef_centred")
fig, ax = plt.subplots(figsize=(6.4, 5))
yy = np.arange(len(o))
ax.errorbar(o["coef_centred"], yy, xerr=[o["coef_centred"] - o["ci95_lo_centred"], o["ci95_hi_centred"] - o["coef_centred"]],
            fmt="o", color=INK, ms=4, lw=1, capsize=2)
ax.scatter(main["excess"].reindex(o.index), yy, marker="|", s=120, color=BLUE, zorder=3, label="shuffle excess (budget deciles)")
ax.axvline(0, color="#999", lw=.6)
ax.set_yticks(yy, o.index, fontsize=8)
ax.set_xlabel("centred OLS coefficient with 95% interval")
ax.legend(fontsize=7, frameon=False, loc="lower right")
plt.show()
for s, v in S["sets"].items():
    print(f"{s}: R2 budget + era {v['r2_budget_era']:.3f} -> with genre {v['r2_budget_era_genre']:.3f} "
          f"(adds {v['r2_added_by_genre']:.3f}, adjusted {v['adj_r2_added_by_genre']:.3f}); "
          f"Spearman of OLS vs shuffle excess {v['spearman_ols_vs_shuffle_excess']:.2f}")"""),
    ("md", """The regression ranks the genres almost as the shuffle does. Genre adds only a small part of the variance on
top of budget and era: it's a real but small effect."""),
    ("md", """## 5. On the common set

The common set is the films that have usable money and a MovieSum screenplay, the set on which Q1, Q2 and Q3 are
compared head to head. It has fewer genres with enough films (no Animation, no Family)."""),
    ("code", """rows = {}
for null, name in [("none", "no matching"), ("budget", "budget"), ("budget_era", "budget x era"), ("budget_fine", "50 budget bins")]:
    v = view("in_common_set", null=null)
    rows[(name, "excess")] = v["excess"]
    rows[(name, "p_holm")] = v["p_holm"]
c = pd.DataFrame(rows)
c.insert(0, ("", "n_films"), view("in_common_set")["n_films"])
c.sort_values(("budget", "excess"), ascending=False)"""),
    ("md", """## 6. Check 1: a smooth budget curve instead of bins

ln(gross / budget) minus a natural cubic spline in ln(budget) plus era; the genre means of that leftover against
a plain shuffle of it. No bin edges to choose, so this is the cleanest answer to "are the deciles too coarse?"."""),
    ("code", """sm = smooth.set_index("genre")
pd.DataFrame({"main excess": main["excess"], "main p_holm": main["p_holm"],
              "smooth excess": sm["excess"], "smooth p_holm": sm["p_holm"]}).sort_values("main excess", ascending=False)"""),
    ("md", """## 7. Check 2: genre x budget tier

Four fixed budget tiers. Inside each, every genre against a shuffle within budget deciles among that tier's
films. Cells with too few films are grey; * marks Holm p < .05 over all filled cells. This shows *where* on the
budget scale genre matters."""),
    ("code", """names = ["under $5M", "$5-30M", "$30-100M", "$100M+"]
order = main.sort_values("excess", ascending=False).index
E = tiers.pivot(index="genre", columns="tier", values="excess").reindex(index=order, columns=names)
P = tiers.pivot(index="genre", columns="tier", values="p_holm").reindex(index=order, columns=names)
N = tiers.pivot(index="genre", columns="tier", values="n_films").reindex(index=order, columns=names)
lim = np.nanmax(np.abs(E.to_numpy()))
cmap = plt.get_cmap("RdBu").copy()
cmap.set_bad("#eeeeee")
fig, ax = plt.subplots(figsize=(6.2, 6.4))
im = ax.imshow(np.ma.masked_invalid(E.to_numpy()), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
for i in range(len(order)):
    for k in range(len(names)):
        e, p, n = E.iat[i, k], P.iat[i, k], N.iat[i, k]
        txt = f"n={int(n)}" if np.isnan(e) else f"{e:+.2f}{' *' if p < .05 else ''}\\nn={int(n)}"
        ax.text(k, i, txt, ha="center", va="center", fontsize=6,
                color=INK if np.isnan(e) or abs(e) < .6 * lim else "white")
ax.set_xticks(range(len(names)), names, fontsize=8)
ax.set_yticks(range(len(order)), order, fontsize=8)
ax.spines[:].set_visible(True)
fig.colorbar(im, ax=ax, shrink=.6, label="excess ln(gross / budget) over the shuffle")
plt.show()"""),
    ("code", """f = tiers[tiers["excess"].notna()]
print(f"{len(f)} filled cells, {(f['p_holm'] < .05).sum()} pass:")
for _, r in f[f["p_holm"] < .05].sort_values(["tier", "excess"]).iterrows():
    print(f"  {r['tier']:>9}  {r['genre']:<10} {r['excess']:+.2f}  ({int(r['n_films'])} films, Holm p {r['p_holm']:.3f})")"""),
    ("md", """Genre matters most for mid-budget films. At $100M and up only Drama differs, and in the cheapest tier returns
spread so widely that even large excesses don't pass. Horror's advantage comes from its $5-30M films, not from
the micro-budget hits."""),
    ("md", """## 8. Check 3: without each genre's biggest hits

Each genre's 5 and 10 highest ratios are dropped, and the shuffle drops the top films the same way. If a few
extreme films (Blair Witch, Paranormal Activity, a possible parser error) carried a genre, its excess would
shrink here."""),
    ("code", """tr = trim.pivot(index="genre", columns="check", values="excess")
pd.DataFrame({"main": main["excess"], "without top 5": tr["trim_top5"], "without top 10": tr["trim_top10"],
              "p_holm top 10": trim[trim["check"] == "trim_top10"].set_index("genre")["p_holm"]}).sort_values("main", ascending=False)"""),
    ("code", """print("The films dropped from Horror:")
tops[(tops["genre"] == "Horror") & (tops["rank"] <= 10)][["title", "year", "budget", "gross", "ln_ratio"]].reset_index(drop=True)"""),
    ("md", """**Outliers at both ends, across all films.** Instead of each genre's top films, remove the most extreme films
overall, the biggest hits and the worst flops together, recut the budget deciles and rerun the main test. This asks
whether a few films drive the result, not whether their figures are right (Blair Witch is extreme and real)."""),
    ("code", """ol = pd.read_csv(CHK / "outliers.csv")
tab = pd.DataFrame({("main", "excess"): main["excess"], ("main", "Holm p"): main["p_holm"]})
for c in ol["check"].unique():
    x = ol[ol["check"] == c].set_index("genre")
    lab = f"{c} ({x['films_removed'].iloc[0]} films)"
    tab[(lab, "excess")], tab[(lab, "Holm p")] = x["excess"], x["p_holm"]
tab.columns = pd.MultiIndex.from_tuples(tab.columns)
tab.sort_values(("main", "excess"), ascending=False)"""),
    ("code", """print("The 20 films removed in the first cut:")
pd.read_csv(CHK / "outliers_removed.csv")[["side", "title", "year", "genres", "ln_gross_over_budget"]]"""),
    ("md", """## 9. Check 4: matched pairs

Each film against the 5 films outside its genre with the closest budget in the same era; the mean difference
with a 95 % bootstrap interval over the genre's films. The most literal reading of "compared with films of
similar budgets". The intervals aren't corrected for the number of genres."""),
    ("code", """m = matched.set_index("genre").reindex(order[::-1])
fig, ax = plt.subplots(figsize=(6.4, 5))
yy = np.arange(len(m))
ax.errorbar(m["excess"], yy, xerr=[m["excess"] - m["lo95"], m["hi95"] - m["excess"]], fmt="o", ms=4, lw=1, capsize=2,
            color=INK)
ax.scatter(main["excess"].reindex(m.index), yy, marker="|", s=120, color=BLUE, zorder=3, label="main test excess")
ax.axvline(0, color="#999", lw=.6)
ax.set_yticks(yy, m.index, fontsize=8)
ax.set_xlabel("mean difference from nearest-budget films outside the genre (95% bootstrap interval)")
ax.legend(fontsize=7, frameon=False, loc="lower right")
plt.show()"""),
    ("md", """## 10. Check 5: hit share and the share below budget

A hit is a film in the top quarter of returns within its era. The share below budget is the share of films that
grossed less than their budget. Both against the budget-matched shuffle, in percentage points. This says *how*
a genre gets its result: by producing more hits, or by flopping less."""),
    ("code", """h = shares[shares["check"] == "hit_share"].set_index("genre").reindex(order)
b = shares[shares["check"] == "below_budget_share"].set_index("genre").reindex(order)
fig, axes = plt.subplots(1, 2, figsize=(8.4, 5), sharey=True)
for ax, s, title in [(axes[0], h, "hit share (top quarter within era)"), (axes[1], b, "share that grossed below budget")]:
    yy = np.arange(len(s))[::-1]
    ax.barh(yy, s["excess_pp"], color=[colour({"p_holm": p, "excess": e}) for p, e in zip(s["p_holm"], s["excess_pp"])], height=.55)
    for y, lo, hi in zip(yy, s["lo95_pp"], s["hi95_pp"]):
        ax.add_patch(plt.Rectangle((lo, y - .38), hi - lo, .76, facecolor="#000", alpha=.1, edgecolor=INK, lw=.5))
    ax.axvline(0, color="#999", lw=.6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("percentage points above the shuffle")
axes[0].set_yticks(np.arange(len(h))[::-1], h.index, fontsize=8)
plt.tight_layout()
plt.show()"""),
    ("code", """pd.DataFrame({"hit share": h["share"], "hit expected": h["expected"], "hit p_holm": h["p_holm"],
              "below budget share": b["share"], "below expected": b["expected"], "below p_holm": b["p_holm"]})"""),
    ("md", """## 11. The genre network

Genres are linked by the number of films that carry both (a projection of the film-genre network, week 1). Big
genres share many films just by being big, so each pair is compared with a curveball null that swaps genres
between films while keeping every film's number of genres and every genre's number of films (week 2). The links
that beat the null (Holm over all pairs) form the backbone; the disparity filter (week 4) is the second method.
Louvain (week 4) splits the backbone into genre families, and its modularity is compared with the same backbone
rewired."""),
    ("code", """print(f"{SN['genre_network']['pairs']} genre pairs, {SN['genre_network']['null_samples']} null samples")
g = SN["genre_network"]
print(f"backbone (null model): {g['backbone_links']} links; disparity filter: {g['disparity_links']}; both: {g['both']}, "
      f"null only: {g['backbone_only']}, disparity only: {g['disparity_only']}")
print(f"Louvain: {g['families']} families, modularity {g['modularity']:.3f} (seeds {g['modularity_min_max_over_seeds']}), "
      f"agreement across seeds NMI {g['mean_nmi_best_vs_other_seeds']:.2f}")
print(f"rewired backbone: modularity {g['null_modularity_mean']:.3f} on average, max {g['null_modularity_max']:.3f}; p = {g['modularity_p']}")
display(Image(filename=str(NET / "figures" / "q1_genre_network.png"), width=650))"""),
    ("code", """print("Strongest links over the null:")
pairs[pairs["backbone"]].sort_values("z", ascending=False).head(12).reset_index(drop=True)"""),
    ("code", """print("Weakest (pairs that co-occur less than chance):")
pairs.sort_values("z").head(6).reset_index(drop=True)"""),
    ("md", """Each family's return, with every film split across families by its share of genres, against the
budget-decile shuffle:"""),
    ("code", """families"""),
    ("md", """The families are what genres *travel with*. If returns belonged to families, the families would differ here;
the main test's winners and losers sit in the same families instead (Animation next to Action, Horror next to
Crime), so the return effect belongs to single genres."""),
    ("md", """## 12. Do films made by the same people share genre?

Films are linked when they share a director, writer, producer or one of the first three billed actors. For each
genre, the assortativity coefficient of its 0/1 indicator on that network (week 3, mixing patterns), against the
genre sets shuffled between films of the same era (each film keeps its links). This is the link between Q1 and
Q2: if people stick to genres, part of what Q2's network measures is genre."""),
    ("code", """fn = SN["film_network"]
print(f"{fn['films']:,} films, {fn['people']:,} people, {fn['links']:,} links, mean degree {fn['mean_degree']}, "
      f"{fn['films_with_no_link']} films without a link")
print(f"links whose two films share a genre: {fn['share_links_sharing_a_genre']:.1%} "
      f"(shuffle {fn['share_null_mean']:.1%}, 97.5th pct {fn['share_null_hi95']:.1%}, p {fn['share_p']})")
a_ = assort.sort_values("assortativity")
fig, ax = plt.subplots(figsize=(6.2, 4.8))
yy = np.arange(len(a_))
for k, (_, r) in enumerate(a_.iterrows()):
    ax.add_patch(plt.Rectangle((r["null_lo95"], k - .35), r["null_hi95"] - r["null_lo95"], .7, color="#000", alpha=.12))
ax.scatter(a_["assortativity"], yy, color=[BLUE if p < .05 else GREY for p in a_["p_holm"]], zorder=3)
ax.set_yticks(yy, a_["genre"], fontsize=8)
ax.axvline(0, color="#999", lw=.6)
ax.set_xlabel("genre assortativity on the shared-people film network (band: era-matched shuffle)")
plt.show()
a_.sort_values("assortativity", ascending=False).reset_index(drop=True)"""),
    ("md", """## 13. Other ways to say what genre a film is

IMDb's genres are up to three unweighted tags. The Tag Genome gives each film a share per genre from what users
tagged; the plot shares predict the same from the Wikipedia plot's embedding; and k-means on the plot embeddings
makes genres from text alone (weeks 6 and 7). How far do they agree, and which predicts returns best?"""),
    ("code", """print(f"films: {SS['films']:,}; with Tag Genome {SS['with_tag_genome']:,}; with a plot vector {SS['with_plot_vector']:,}; "
      f"with all {SS['films_with_all']:,}. k-means k = {SS['kmeans_k']}, stability across seeds NMI {SS['kmeans_mean_nmi_best_vs_other_seeds']}")
agreement"""),
    ("code", """c2 = cv.copy()
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.barh(c2["model"][::-1], c2["added_over_budget_era"][::-1], xerr=c2["sd_over_repeats"][::-1], color=BLUE, height=.55)
ax.axvline(0, color="#999", lw=.6)
ax.set_xlabel("cross-validated R2 added over budget decile + era")
plt.show()
cv"""),
    ("md", """The plot embedding adds the most, but a Wikipedia plot is written after release and the embedding has hundreds
of numbers, so part of it can be how famous the film is rather than what it is about. The plot clusters, which
know nothing about returns, add almost nothing. The weighted Tag Genome shares do better than IMDb's 0/1 tags."""),
    ("code", """tg = share_tests[share_tests["films"] == "tag_genome"].pivot(index="genre", columns="representation", values=["excess", "p_holm"])
tg.sort_values(("excess", "imdb_tags"), ascending=False)"""),
    ("md", """Same films, three ways to count genre: IMDb's tags, each film split by its Tag Genome shares, and each film
given its single dominant Tag Genome genre (no overlap, so the tests are separate). The plot clusters, described
by their most common IMDb genres:"""),
    ("code", """clusters.sort_values("excess", ascending=False).reset_index(drop=True)"""),
    ("md", """## 14. Inside Horror: which plot words go with beating the budget

Horror films are split by their return net of budget decile and era: top quarter against bottom quarter. Their
Wikipedia plots are compared word by word with the log-odds ratio and an informative Dirichlet prior (week 6;
the statistic Q3 uses on scripts), and the labels are shuffled within budget deciles. Character names are dropped
(words mostly capitalised mid-sentence). Only words and counts are shown; no plot text."""),
    ("code", """print(f"{SW['films_with_plot']} Horror films with a plot; {SW['hits']} above, {SW['flops']} below; vocabulary {SW['vocab']:,} words")
print(f"largest |calibrated z| {SW['max_abs_calibrated_z']}, family-wise shuffle p {SW['familywise_p_max_abs_calibrated_z']}")
print(f"median plot length: {SW['median_plot_words_hit']} words (above), {SW['median_plot_words_flop']} (below)")
wlists"""),
    ("code", """t = wtop.assign(rank=wtop.groupby("side").cumcount() + 1).pivot(index="rank", columns="side", values="word")
t["hit z"] = wtop[wtop.side == "hit"]["calibrated_z"].to_numpy()
t["flop z"] = wtop[wtop.side == "flop"]["calibrated_z"].to_numpy()
t[["hit", "hit z", "flop", "flop z"]]"""),
    ("md", """The words describe the plots as Wikipedia editors remember them, so they can't separate what made a film work
from how its story is told afterwards. They do say what kind of Horror beats its budget."""),
    ("md", """## 15. The genre distribution through the same analyses

`notebooks/tag_shares.ipynb` turned each film's Tag Genome genre scores into shares that add up to 1. Here every
follow-up above is rerun with those shares as weights, next to the 0/1 tags on the same films (the films with
Tag Genome scores, which stop in 2014). `notebooks/tag_profit.ipynb` found that tag scores carry audience size, so
the first question is whether the shares do too."""),
    ("code", """lk = pd.read_csv(SF / "leak.csv")
print(f"{SFS['films']:,} films with Tag Genome shares")
lk.pivot(index="model", columns="target", values="added").loc[["+ IMDb tags", "+ Tag Genome shares"]]"""),
    ("md", """`ln_votes`: how much each adds to predicting ln(IMDb votes) beyond budget and era; if the shares carried
audience size they would add more here than the tags. `ln_ratio_votes_in_base`: what each adds to the return when
ln(votes) is already in the model (votes are partly the outcome, so this is a sensitivity check)."""),
    ("code", """t_imdb = pd.read_csv(SF / "tiers_imdb_tags.csv")
t_sh = pd.read_csv(SF / "tiers_tag_genome_shares.csv")
for name, t in (("0/1 tags", t_imdb), ("shares", t_sh)):
    ok = t[t["excess"].notna()]
    print(f"tiers with {name}: {(ok['p_holm'] < .05).sum()} of {len(ok)} cells pass: "
          + ", ".join(f"{r['genre']} {r['tier']} {r['excess']:+.2f}" for _, r in ok[ok['p_holm'] < .05].iterrows()))
display(Image(filename=str(SF / "figures" / "q1_tiers_tag_genome_shares.png"), width=520))"""),
    ("code", """sm_i = pd.read_csv(SF / "smooth_imdb_tags.csv").set_index("genre")
sm_s = pd.read_csv(SF / "smooth_tag_genome_shares.csv").set_index("genre")
hs_i = pd.read_csv(SF / "hitshare_imdb_tags.csv")
hs_s = pd.read_csv(SF / "hitshare_tag_genome_shares.csv")
hit = lambda d: d[d["check"] == "hit_share"].set_index("genre")
pd.DataFrame({"smooth, tags": sm_i["excess"], "p": sm_i["p_holm"], "smooth, shares": sm_s["excess"], "p ": sm_s["p_holm"],
              "hit pp, tags": hit(hs_i)["excess_pp"], "hit pp, shares": hit(hs_s)["excess_pp"]}).sort_values("smooth, shares", ascending=False)"""),
    ("md", """**Genre mix.** A film's effective number of genres (1 / sum of squared shares) is how spread its mix is. Mean
return per quintile of it, and per IMDb number of genres, against the budget-decile shuffle:"""),
    ("code", """sp = pd.read_csv(SF / "spread.csv")
print({k: v for k, v in SFS["spread"].items()})
sp"""),
    ("md", """The two measures point opposite ways: films with one IMDb genre beat their budgets, while films the Tag Genome
sees as most mixed do too. The spread of the Tag Genome mix goes with IMDb votes (printed above), so a broad mix
partly means a widely seen film, as `tag_profit.ipynb` found; IMDb's count of genres does not."""),
    ("code", """sn = SFS["share_network"]
print(f"share network: {sn['backbone_links']} backbone links ({sn['shared_links']} also in the IMDb-tag backbone of "
      f"{sn['imdb_backbone_links']}); Louvain {sn['families']} families, modularity {sn['modularity']}; "
      f"NMI with the IMDb-tag families {sn['nmi_with_imdb_families']}")
pd.read_csv(SF / "share_network_families.csv")"""),
    ("code", """asr = pd.read_csv(SF / "assortativity.csv").pivot(index="genre", columns="representation", values="assortativity")
asr.sort_values("tag_genome_shares", ascending=False)"""),
    ("md", """**Horror words, three ways.** The same word test as the Horror-plot section on (a) every IMDb-tagged Horror film,
(b) IMDb-tagged Horror films with Tag Genome scores, (c) films whose largest Tag Genome share is Horror. If (b) and
(c) agree with each other but not with (a), the difference comes from which films are in (the Tag Genome stops in
2014), not from how Horror is defined."""),
    ("code", """runs = {"(a) IMDb Horror": "horror", "(b) IMDb Horror, Tag Genome films": "horror_imdb_on_tag_genome",
        "(c) Tag Genome Horror": "horror_tag_genome"}
cols = {}
for name, tag in runs.items():
    s_ = json.loads((WRD / f"{tag}_summary.json").read_text())
    t_ = pd.read_csv(WRD / f"{tag}_top.csv")
    print(f"{name}: {s_['films_with_plot']} films, {s_['hits']} vs {s_['flops']}, family-wise p {s_['familywise_p_max_abs_calibrated_z']}")
    cols[(name, "above")] = t_[t_.side == "hit"]["word"].head(10).to_numpy()
    cols[(name, "below")] = t_[t_.side == "flop"]["word"].head(10).to_numpy()
pd.DataFrame(cols)"""),
    ("md", """## 16. Fixing the genre distribution's weaknesses

The section before used the Tag Genome shares as they were built for `tag_shares.ipynb`. A critical look found six
problems: the shares carry reception (ratings, critics), the power of 2 in their construction was arbitrary and moved
the result, the films covered are older and much-tagged, everything in them is measured after release, each genre
rested on one tag, and the "effective number of genres" partly measured fame. `q1_share_fixes.py` fixes each one,
choosing every setting from agreement with IMDb's labels and never from returns."""),
    ("md", """**Which tag scores each genre.** Each genre's candidate tags (and their mean) against IMDb's label, by AUC; the
best is kept. The old mapping is marked `current`."""),
    ("code", """mp = pd.read_csv(FX / "mapping.csv")
print("genres whose tag changed:", ", ".join(FXS["mapping_changed"]))
mp[mp["chosen"] | mp["current"]].reset_index(drop=True)"""),
    ("md", """**Which power.** Shares = relevance to the power p, divided by the film's total. p is chosen by how well each
share reproduces IMDb's label (cross-validated log loss, lower is better):"""),
    ("code", """pw = pd.read_csv(FX / "power.csv")
print(f"chosen power: {FXS['power_chosen']}")
pw"""),
    ("md", """**Shares, total tagging and reception.** Raw relevance mixes how much a film is tagged at all (which goes with
fame) with what kind of film it is. The log of the total enters as its own variable; rating and ln(votes) go into
the base model so that the shares are judged on what they add beyond reception. CV R2 added over each base:"""),
    ("code", """cr = pd.read_csv(FX / "cv_r2.csv")
cr[cr["films"].str.startswith("tag_genome") | cr["films"].str.startswith("films with")].reset_index(drop=True)"""),
    ("code", """print("What each one predicts besides returns (CV R2 added over budget + era):")
cr[cr["films"].str.startswith("target")].reset_index(drop=True)"""),
    ("md", """The log total predicts votes far better than it predicts anything about genre, so it is mostly fame; the
shares on their own predict IMDb rating better than the tags do, so they carry some reception. With rating and
votes held fixed, the shares still add more than the tags."""),
    ("code", """ge = pd.read_csv(FX / "genre_effects.csv").pivot(index="genre", columns="representation", values="coef_centred")
print("rank agreement of genre effects (Spearman):", FXS["spearman_effects"])
ge.sort_values("IMDb tags", ascending=False)"""),
    ("md", """Share coefficients are per whole unit of share (0 to 1) and shares are small, so read their order, not their size.
Family and Animation shares overlap strongly (Animation films score high on the family tags), which is why the
Animation share turns negative next to the Family share."""),
    ("md", """**Coverage.** Budget-netted genre means with 0/1 tags on the full set, on the Tag Genome subset, and on the subset
reweighted to look like the full set (by decade x vote tier x budget decile):"""),
    ("code", """cv_ = pd.read_csv(FX / "coverage.csv")
print(FXS["coverage"])
fig, ax = plt.subplots(figsize=(5, 4.4))
ax.scatter(cv_["full_set"], cv_["subset"], color=GREY, label="subset")
ax.scatter(cv_["full_set"], cv_["subset_reweighted"], color=BLUE, label="subset, reweighted")
for _, r in cv_.iterrows():
    ax.annotate(r["genre"], (r["full_set"], r["subset_reweighted"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
lo, hi = cv_[["full_set", "subset", "subset_reweighted"]].min().min(), cv_[["full_set", "subset", "subset_reweighted"]].max().max()
ax.plot([lo, hi], [lo, hi], color="#999", lw=.6)
ax.set_xlabel("full set: genre mean of budget-netted return")
ax.set_ylabel("Tag Genome subset")
ax.legend(fontsize=7, frameon=False)
plt.show()"""),
    ("md", """**Genre mix, without the fame in it.** The effective number of genres against the budget-netted return, per
standard deviation, alone and with the log total, rating and votes held fixed; IMDb's count of genres for
comparison:"""),
    ("code", """pd.read_csv(FX / "spread.csv")"""),
    ("md", """**Genre shares from the screenplay, known before release.** A ridge regression from each script's embedding to
its Tag Genome shares (out-of-fold for the films used in training). A script holds no ratings and no audience, so
these shares can't carry reception. First, are they a valid measure of genre?"""),
    ("code", """pd.read_csv(FX / "script_shares_validation.csv")"""),
    ("code", """print(f"{FXS['common_set_films']:,} films in the common set")
pd.read_csv(FX / "script_shares_cv_r2.csv")"""),
    ("code", """ss_ = pd.read_csv(FX / "script_shares_shuffle.csv").pivot(index="genre", columns="representation", values=["excess", "p_holm"])
ss_.sort_values(("excess", "IMDb tags"), ascending=False)"""),
    ("md", """**Hand check (to be filled in by the group).** Films whose dominant share is none of their IMDb genres are where
the two sources disagree most. A random sample of them is in `analysis/genre/outputs/share_fixes/hand_check_sheet.csv`:
for each, read the film's synopsis and write in `which_is_right` whether IMDb, the Tag Genome, both or neither
names its genre better. The count below updates when the sheet is filled."""),
    ("code", """hc = pd.read_csv(FX / "hand_check_sheet.csv").fillna("")
print(f"{FXS['films_dominant_not_in_imdb']} films disagree; sample of {len(hc)}; labelled so far: {(hc['which_is_right'] != '').sum()}")
if (hc["which_is_right"] != "").any():
    print(hc.loc[hc["which_is_right"] != "", "which_is_right"].str.lower().value_counts())
hc[["title", "year", "imdb_genres", "tag_genome_top3"]].head(10)"""),
    ("md", """## 17. Before release: genre specialists and script typicality

Two measures that tie genre to Q2's people and Q3's screenplays and exist before a film comes out.

**Genre specialists.** A film's team is its directors, writers, producers and first three billed actors (IMDb's
`title.principals`). Their earlier films are every earlier theatrical film they're credited on in IMDb's full
`title.basics`, noticed or not. (The first version counted only films that later got 10,000+ IMDb votes, which let
a successful past leak into "experience"; its numbers are kept below as a comparison.) Per film: the share of the
team's earlier credits in the film's genres (`genre_experience`), how many earlier credits (`prior_credits`), and how
spread over genres they are (`genre_breadth`). Outcome: the return net of budget decile and era; genre dummies and a
sequel flag held fixed. The shuffle refits exactly the reported model."""),
    ("code", """print(BRS["specialists"])
pd.read_csv(BR / "specialists_ols.csv")"""),
    ("md", """**Script typicality.** How close a screenplay's embedding is to the usual screenplay of its genres. A garbled or
oddly formatted script file would also sit far from its genre, so Q3's script-quality measures (length, dialogue
share, unparsed elements, scene headings inside dialogue, upper-case and punctuation noise, tiny layers, and on the
final set the symbol-noise share) are held fixed too. If typicality only measured clean files, they would take its
coefficient away."""),
    ("code", """print(BRS["typicality"])
display(pd.read_csv(BR / "typicality_ols.csv"))
print("How each quality measure goes with typicality and with the return:")
pd.read_csv(BR / "typicality_quality.csv")"""),
    ("code", """sq = pd.read_csv(BR / "specialists_quintiles.csv")
tq = pd.read_csv(BR / "typicality_quintiles.csv")
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
axes[0].bar(sq["quintile"], sq["mean_net_return"], color=BLUE)
axes[0].set_xticks(sq["quintile"], sq["genre_experience_range"], fontsize=7)
axes[0].set_xlabel("team's earlier experience in the film's genres (share of credits)")
axes[0].set_ylabel("return net of budget and era")
axes[1].bar(tq["quintile"], tq["mean_net_return"], color=BLUE)
axes[1].set_xlabel("script typicality for its genres (quintile)")
for ax in axes:
    ax.axhline(0, color="#999", lw=.6)
plt.tight_layout()
plt.show()
pd.read_csv(BR / "typicality_per_genre.csv")"""),
    ("md", """## 19. Inside Horror, in the screenplays

The word test of the Horror-plot section on the screenplays instead of the Wikipedia plots (Q3's preprocessing,
description and dialogue layers), and the same supernatural word set counted in the scripts. It does not replicate."""),
    ("code", """print({k: v for k, v in HSS.items() if k != "supernatural_in_script"})
print("supernatural words in the script:", HSS["supernatural_in_script"])
hw = pd.read_csv(HSC / "horror_script_words.csv")
pd.DataFrame({(l, sd): hw[(hw.layer == l) & (hw.side == sd)]["word"].head(10).to_numpy()
              for l in ("description", "dialogue") for sd in ("hit", "flop")})"""),
    ("code", """pd.read_csv(HSC / "horror_script_list_null.csv").pivot(index=["layer", "side"], columns="k", values="p")"""),
    ("md", """## 18. Summary

Printed from the cells above, so it changes with the data."""),
    ("code", """up = [g for g in SEVEN if main.loc[g, "excess"] > 0]
down = [g for g in SEVEN if main.loc[g, "excess"] < 0]
v = S["sets"]["usable_money"]
print("Core:")
print(f"1. Main test ({v['films']:,} films): above their budgets: " + ", ".join(f"{g} x{np.exp(main.loc[g, 'excess']):.2f}" for g in up)
      + "; below: " + ", ".join(f"{g} x{np.exp(main.loc[g, 'excess']):.2f}" for g in down) + ".")
print(f"2. These pass with no matching, budget deciles, budget x era and ranks. Genre adds {v['r2_added_by_genre']:.3f} to R2 "
      f"on top of budget and era ({v['r2_budget_era']:.3f}).")
keep_ = all(np.sign(sm.loc[g, 'excess']) == np.sign(main.loc[g, 'excess']) and
            np.sign(tr.loc[g, 'trim_top10']) == np.sign(main.loc[g, 'excess']) for g in SEVEN)
print(f"3. Same sign with the smooth budget curve and without each genre's top 10 films, for all of them: {keep_}.")
ft = tiers[tiers["excess"].notna()]
mid = ft[ft["p_holm"] < .05]
print(f"4. Budget tiers: {len(mid)} of {len(ft)} cells pass, " + ", ".join(f"{t} {(mid['tier'] == t).sum()}" for t in names) + ".")
cs = view("in_common_set")
print("5. Common set: passes the budget shuffle: " + ", ".join(f"{g} {cs.loc[g, 'excess']:+.3f}" for g in cs.index[cs["p_holm"] < .05]) + ".")
o_ = pd.read_csv(BR / "specialists_ols.csv").set_index(["films", "term"])
t_ = pd.read_csv(BR / "typicality_ols.csv").set_index(["films", "term"])
print(f"6. Before release: a team's earlier experience in the film's genres "
      f"{o_.loc[('history: all earlier IMDb movies, sequels left out', 'genre_experience'), 'coef_per_sd']:+.3f} per SD (sequels left out; "
      f"shuffle p {BRS['specialists']['shuffle_p_genre_experience']}); script typicality "
      f"{t_.loc[('+ script-quality controls, sequels left out', 'script_typicality'), 'coef_per_sd']:+.3f} per SD with script quality held fixed "
      f"(shuffle p {BRS['typicality']['shuffle_p']}).")
print()
print("Exploratory (weak evidence, not used for claims):")
g = SN["genre_network"]
print(f"   Genre network: {g['backbone_links']} of {g['pairs']} pairs beat the null; {g['families']} Louvain families, "
      f"modularity no stronger than a rewired network (p {g['modularity_p']}).")
print(f"   Film network: {fn['share_links_sharing_a_genre']:.0%} of shared-people links join films of a common genre "
      f"against {fn['share_null_mean']:.0%} after shuffling (a description, not a test of anything surprising).")
r = cv.set_index("model")["added_over_budget_era"]
print("   CV R2 added over budget + era: " + ", ".join(f"{k.lstrip('+ ')} {v:+.3f}" for k, v in r.items() if k != "budget + era") + ".")
fc_ = cr.set_index(["films", "model"])["added"]
print(f"   Fixed Tag Genome shares, rating and votes held fixed: {fc_[('tag_genome_films, rating and votes in base', 'shares (chosen)')]:+.3f} "
      f"against {fc_[('tag_genome_films, rating and votes in base', 'IMDb tags')]:+.3f} for the tags.")
print(f"   Horror plots: above-budget words " + ", ".join(wtop[wtop.side == 'hit']['word'].head(6))
      + "; in the scripts this does not replicate (supernatural words "
      f"{HSS['supernatural_in_script']['difference']:+.3f}, p {HSS['supernatural_in_script']['p']}).")"""),
]


CORE = ["The films and the settings", "The main test", "Does the null matter?", "A second method: regression",
        "On the common set", "Check 1: a smooth budget curve", "Check 2: genre x budget tier",
        "Check 3: without each genre's biggest hits", "Check 5: hit share", "Before release:"]
EXPLORATORY = ["Check 4: matched pairs", "The genre network", "Do films made by the same people share genre?",
               "Other ways to say what genre a film is", "Inside Horror: which plot words",
               "Inside Horror, in the screenplays", "The genre distribution through the same analyses",
               "Fixing the genre distribution's weaknesses"]
WHY = {
    "Check 4: matched pairs": "a different estimand from the main test, not corrected for testing 18 genres, and it repeats what the shuffle and the regression already show",
    "The genre network": "the backbone is a fine description, but the Louvain families are no stronger than a rewired network, so testing their returns tests groups that may not exist",
    "Do films made by the same people share genre?": "the null shuffles away all structure, so a huge z was guaranteed; people repeating genres is expected and largely sequels and franchises",
    "Other ways to say what genre a film is": "Tag Genome and plots are written after release and carry reception; plot clusters are a null result",
    "Inside Horror: which plot words": "post-release text, post-hoc word set, and it does not replicate in the screenplays",
    "Inside Horror, in the screenplays": "the replication attempt of the plot words; it fails",
    "The genre distribution through the same analyses": "post-release measure that carries reception, with many variants tried",
    "Fixing the genre distribution's weaknesses": "careful, but it repairs a measure the analysis doesn't rely on; the script-predicted shares learn from a target that carries reception",
}


def ordered(cells):
    """Intro and setup, the core sections, one exploratory part (each section with a line on why it's there), summary.
    Sections are renumbered; nothing is dropped."""
    blocks, cur, head = {}, None, []
    for kind, src in cells:
        if kind == "md" and src.startswith("## "):
            cur = src.split("\n", 1)[0][3:].split(". ", 1)[1]
            blocks[cur] = []
        (blocks[cur] if cur else head).append([kind, src])
    find = lambda key: next(k for k in blocks if k.startswith(key))
    out, n, toc = list(head), 0, []
    for key in CORE:
        n += 1
        k = find(key)
        toc.append(f"{n}. {k}")
        b = blocks.pop(k)
        rest = b[0][1].split("\n", 1)[1] if "\n" in b[0][1] else ""
        b[0][1] = f"## {n}. {k}\n" + rest
        out += b
    out.append(["md", "# Exploratory: weak evidence, not used for claims\n\nThese analyses were run and are kept for the record, "
                      "with their outputs in `analysis/genre/exploratory/`. Each says in one line why it is here."])
    toc.append("Exploratory part:")
    for key in EXPLORATORY:
        n += 1
        k = find(key)
        toc.append(f"{n}. {k}")
        b = blocks.pop(k)
        rest = b[0][1].split("\n", 1)[1] if "\n" in b[0][1] else ""
        b[0][1] = f"## {n}. {k}\n\n*Exploratory: {WHY[key]}.*\n" + rest
        out += b
    k = find("Summary")
    n += 1
    b = blocks.pop(k)
    b[0][1] = f"## {n}. Summary" + ("\n" + b[0][1].split("\n", 1)[1] if "\n" in b[0][1] else "")
    toc.append(f"{n}. Summary")
    out += b
    assert not blocks, f"sections not placed: {list(blocks)}"
    intro = out[0][1]
    out[0][1] = intro.split("Sections:")[0] + "Sections:\n" + "\n".join(toc)
    return [tuple(c) for c in out]


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in ordered(CELLS)]
    if "--no-run" not in sys.argv:
        from nbconvert.preprocessors import ExecutePreprocessor
        ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
    nbf.write(nb, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
