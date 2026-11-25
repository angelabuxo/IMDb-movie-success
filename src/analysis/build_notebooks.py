"""Write and execute the five notebooks in notebooks/, one per part of the paper. They read only committed outputs,
so they need no raw data, and no result is typed into their text.

Run from src/:  python -m analysis.build_notebooks
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks"

SETUP = """import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "analysis").exists())
A = ROOT / "src" / "analysis"
pd.set_option("display.max_colwidth", 90)"""

NB = {
"1_data": [
("md", """# 1. Data: which films we keep, and why

**Question.** Which films can we measure a return for?

**Measure.** Return is ln(gross / budget). 0 means the gross equalled the budget.

**Method.** Start from the IMDb films with 10,000+ votes, keep those with comparable money, then those that also have a usable screenplay. The money rule is in one module, `src/analysis/clean_rule.py`, and the columns are described in `data/ANALYSIS_SET.md`.

**What it misses.** Films under 10,000 votes, films without both figures, and everything a budget leaves out (marketing) or a gross leaves out (home video, streaming). Gross over budget isn't profit."""),
("code", SETUP),
("md", "## The film sets"),
("code", """links = pd.read_csv(ROOT / "data" / "movie_links.csv")
cov = pd.read_csv(A / "screenplays/word2vec/results/prerelease_coverage.csv")
h2h = json.loads((A / "compare/outputs/head_to_head/summary.json").read_text())
pd.Series({"IMDb films with 10,000+ votes": links.imdb_id.nunique(),
           "with comparable budget and gross": int(cov[cov.group == "all films"].films.sum()),
           "also with a usable screenplay (common set)": h2h["films"]}).to_frame("films")"""),
("md", "## Who has a screenplay?\n\nScripts exist mainly for popular films. The table compares films with money that do and don't have one (any MovieSum script, before the token rule)."),
("code", """cov[cov.group == "all films"][["label", "films", "median_votes", "median_budget_musd", "mean_ln_return"]]"""),
("md", "The share of hits and flops that have a script, with a film counted once per genre. The gap is compared with 2,000 shuffles of the labels within genre and era."),
("code", """d = cov[cov.group == "all film-genre pairs"].pivot(index="tier", columns="label", values="coverage")[["hit", "middle", "flop"]]
gap = pd.read_csv(A / "screenplays/word2vec/results/prerelease_coverage_gap.csv").query("group == 'all film-genre pairs'").set_index("tier")
d.join(gap[["gap", "null_lo", "null_hi", "p_two_sided"]]).round(3)"""),
("md", "## A second source for the money\n\nThe Wikipedia figures compared with TMDB's, from `src/scripts/imdb/tmdb_compare.py`."),
("code", """t = json.loads((A / "checks/tmdb_compare.json").read_text())
pd.Series({k: v for k, v in t.items() if not isinstance(v, (dict, list))}).to_frame("value")"""),
("md", "## Were dialogue and description split correctly?\n\nOne of us read a sample of script elements by hand and checked the layer each was put in."),
("code", """pd.read_csv(A / "screenplays/data_check/manual_check_summary.csv")"""),
],
"2_genre": [
("md", """# 2. Genre: do some genres return more than films of the same budget?

**Method.** A genre's statistic is the mean return of its films (a film counts in every genre it has).

**Compared to what?** 10,000 shuffles that swap returns only between films in the same budget decile, so a cheap genre can't win by being cheap. Holm correction over the 18 genres.

**Checks.** Other shuffles, ranks, TMDB as a second source, and Horror subgenres from two independent labels.

**What it misses.** It can't separate a genre from the companies that specialise in it. It isn't causal.

The long version, with every check, is `archive/notebooks/q1_genre.ipynb`."""),
("code", SETUP),
("md", "## Main test"),
("code", """g = pd.read_csv(A / "genre/outputs/genre_shuffle.csv")
m = g.query("set == 'usable_money' and outcome == 'ln_ratio' and null == 'budget'").sort_values("excess", ascending=False)
m[["genre", "n_films", "mean", "null_lo95", "null_hi95", "excess", "p_holm"]].round(3)"""),
("code", """o = m.sort_values("mean").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6, 5))
col = ["tab:blue" if (p < .05 and e > 0) else "tab:red" if p < .05 else "silver" for p, e in zip(o.p_holm, o.excess)]
ax.barh(o.genre, o["mean"], color=col)
ax.hlines(range(len(o)), o.null_lo95, o.null_hi95, color="k", lw=2)
ax.set_xlabel("mean ln(gross / budget); black: middle 95% of shuffles within budget decile"); plt.show()"""),
("md", "## Does the result depend on the null?\n\nWhich shuffles each genre passes after Holm correction, on the 6,346 films. `+` is above, `-` is below, empty is inside."),
("code", """x = g[(g.set == "usable_money")].assign(sign=lambda d: (d.excess > 0).map({True: "+", False: "-"}).where(d.p_holm < .05, ""))
x.assign(run=x.outcome + " / " + x.null).pivot(index="genre", columns="run", values="sign")"""),
("md", "## How much does genre explain?"),
("code", """s = json.loads((A / "genre/outputs/summary.json").read_text())["sets"]
{k: {a: b for a, b in v.items() if not isinstance(b, (dict, list))} for k, v in s.items()}"""),
("md", "## Inside Horror: two labels made in different ways\n\nTMDB keywords (list fixed before looking) and Wikidata's genres, against other Horror films of the same budget and era."),
("code", """pd.read_csv(A / "genre/outputs/tmdb/horror_subgenres.csv").round(3)"""),
("code", """pd.read_csv(A / "genre/outputs/categories/wikidata_horror_subgenres.csv").round(3)"""),
("md", "## A third view: categories from a keyword network\n\nKeywords linked when they share more films than a curveball null expects, grouped with Louvain."),
("code", """pd.read_csv(A / "genre/outputs/categories/keyword_category_tests.csv").sort_values("excess", ascending=False).round(3)"""),
],
"3_people": [
("md", """# 3. People: do films made by the same people earn alike?

**Method.** Link two films that share a director, a top-three billed actor, a producer or a studio. Assortativity is the correlation of the two films' returns, after removing budget, era and genre. Then split the network into communities with Louvain.

**Compared to what?** 1,000 shuffles of returns within era bins that keep the network. For communities, the network rewired with every film's degree kept. For their returns, shuffles within budget decile.

**What it misses.** A shared person also means a shared era, budget tier and company, so this doesn't show the person causes it. Only films with money data are tested, all with 10,000+ votes.

The runs declared before running are in `src/analysis/network/Q2_NETWORK.md`."""),
("code", SETUP),
("md", "## Assortativity by role, common set"),
("code", """q = json.loads((A / "network/q2_network.json").read_text())["samples"]
def tab(k):
    return pd.DataFrame({r: {c: v[c] for c in ("pairs", "assortativity", "null_mean", "null_95", "p")} for r, v in q[k].items() if isinstance(v, dict) and "null_95" in v}).T
tab("in_common_set_adjusted").round(4)"""),
("md", "The same on every film with usable money."),
("code", """tab("usable_money_adjusted").round(4)"""),
("code", """t = tab("in_common_set_adjusted")
ax = t.assortativity.plot.bar(figsize=(5, 3), color="tab:blue")
ax.hlines(t.null_95, [i - .3 for i in range(len(t))], [i + .3 for i in range(len(t))], color="red"); ax.set_ylabel("assortativity (red: 95% of shuffles)"); plt.show()"""),
("md", "## Does it need a shared genre?"),
("code", """b = pd.read_csv(A / "compare/outputs/network_by_genre/assortativity_by_genre.csv")
b[b.split.isin(["all", "same genre", "different genre"])].round(4)"""),
("md", "## Communities\n\nModularity against the rewired network, and how well 10 Louvain seeds agree (NMI). Modularity says how grouped the network is, not whether the groups are right."),
("code", """c = json.loads((A / "compare/outputs/communities_usable_money/summary.json").read_text())
pd.Series({k: v for k, v in c.items() if not isinstance(v, (dict, list))}).to_frame("value")"""),
("code", """p = pd.read_csv(A / "compare/outputs/communities_usable_money/community_profiles.csv")
p[["films", "median_budget_musd", "median_multiple", "excess", "p_holm", "top_studios"]].sort_values("excess", ascending=False).round(3)"""),
("md", "Without studio links, the same test."),
("code", """p = pd.read_csv(A / "compare/outputs/communities_usable_money_no_studio/community_profiles.csv")
p[p.p_holm < .05][["films", "median_budget_musd", "excess", "p_holm", "top_studios"]].round(3)"""),
("md", "## Do people do better in a genre they know?\n\nCoefficients per standard deviation, with experience counted over every earlier IMDb film."),
("code", """e = pd.read_csv(A / "genre/outputs/before_release/specialists_ols.csv")
e[e.films.str.contains("all earlier IMDb")].round(4)"""),
],
"4_screenplays": [
("md", """# 4. Screenplays: do hit and flop scripts differ beyond genre, era and budget?

**Question.** Do hit scripts resemble each other, and flop scripts each other, more than a hit resembles a flop?

**Method.** A hit is in the top quarter of return within genre and era, a flop in the bottom quarter. Dialogue and scene description each become one vector (the TF-IDF-weighted mean of Word2Vec vectors, 5 seeds). Resemblance is cosine similarity. The gap is the mean similarity within hits and within flops minus that between them.

**Compared to what?** 2,000 shuffles of the labels, only between films of the same genre, era and budget third. Holm correction over 18 tests.

**What it misses.** Plot and order, budget differences inside a group, films without a script, films under 10,000 votes. Groups are small, so a test that doesn't pass isn't evidence of no difference.

**How we chose it.** This test is exploratory. A planned test came first (the last section here), and its baseline used votes, which come after release. The criteria are D29 in `src/analysis/screenplays/DECISIONS.md`, written after a one-seed run and before the five-seed run."""),
("code", SETUP + "\nW = A / 'screenplays/word2vec/results'"),
("md", "## The 18 tests"),
("code", """b = pd.read_csv(W / "budget_matched.csv")
b[b.shuffle_within == "era x budget terciles"].drop(columns="shuffle_within").sort_values("p").round(4)"""),
("code", """from IPython.display import Image
Image(str(A / "screenplays/figures/fig_script_similarity.png"), width=520)"""),
("md", "Dark blue passes with budget thirds, halves and quarters. Light blue passes with thirds and halves only. Red: 95% of shuffles lie below.\n\n## Does the choice of budget groups matter?\n\nHolm p for each grouping. `era only` doesn't hold budget fixed."),
("code", """b.assign(test=b.genre + " " + b.layer).pivot(index="test", columns="shuffle_within", values="p_holm").sort_values("era x budget terciles").round(3)"""),
("md", "## Checks on Horror, the only genre that passes\n\n`share_movable` is the share of films in groups where the shuffle can change anything."),
("code", """pd.read_csv(W / "budget_matched_checks.csv").round(4)"""),
("md", "## Which words differ?\n\nLog-odds words with labels shuffled within era and budget third. `family_p` corrects for testing every word."),
("code", """w = pd.read_csv(W / "budget_matched_words.csv")
w.groupby("layer").head(8)[["layer", "word", "direction", "calibrated_z", "family_p", "hit_films", "flop_films"]].round(3)"""),
("md", "## The planned test, for comparison\n\nA cross-validated classifier (AUC; 0.5 is guessing). `tier+era+genres` contains whether a film has 100,000+ votes. The last model adds the vector to facts known before release. We don't interpret the Action gain, because the similarity test doesn't support it."),
("code", """pd.read_csv(W / "prerelease_auc.csv").round(3)"""),
],
"5_head_to_head": [
("md", """# 5. Head to head: genre, people and screenplays on the same films

**Question.** Which of the three goes most with return, on the same films and the same measure?

**Method.** Three networks on the common set: films sharing a genre, sharing people, or with similar screenplays (each linked to its 10 nearest by cosine similarity of a sentence embedding, all-MiniLM-L6-v2, which is not the Word2Vec of notebook 4). Assortativity of the return beyond films of the same budget decile and era.

**Compared to what?** The same 1,000 shuffles within era bins for all three.

**What it misses.** The networks differ in size: shared genre links almost every film to every other, so this measures genre badly, and notebook 2 measures it better. It isn't causal."""),
("code", SETUP),
("code", """a = pd.read_csv(A / "compare/outputs/head_to_head/assortativity.csv")
a.round(4)"""),
("code", """o = a.iloc[::-1]
fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(o.network + ": " + o.links, o.assortativity, color="tab:blue")
ax.vlines(o.null_95, [i - .4 for i in range(len(o))], [i + .4 for i in range(len(o))], color="red")
ax.set_xlabel("return assortativity (red: 95% of shuffles)"); plt.show()"""),
],
}


def main():
    OUT.mkdir(exist_ok=True)
    for name, cells in NB.items():
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        for k, (kind, text) in enumerate(cells):
            c = nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
            c["id"] = f"{name}-{k}".replace("_", "-")
            nb.cells.append(c)
        NotebookClient(nb, timeout=300, resources={"metadata": {"path": str(OUT)}}).execute()
        for c in nb.cells:
            c.metadata.pop("execution", None)
        nbf.write(nb, OUT / f"{name}.ipynb")
        print("written", name)


if __name__ == "__main__":
    main()
