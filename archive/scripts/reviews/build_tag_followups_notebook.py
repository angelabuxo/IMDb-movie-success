"""Write and execute notebooks/tag_followups.ipynb: five follow-ups to the tag and
profit analysis. It reads only the saved outputs of scripts/reviews/plot_tags.py
and tag_followups.py in analysis/tags/, so it needs no raw data. Run from the
repo root:

    python scripts/reviews/build_tag_followups_notebook.py            # write and execute
    python scripts/reviews/build_tag_followups_notebook.py --no-run   # write only
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "archive" / "notebooks" / "tag_followups.ipynb"

CELLS = [
    ("md", """# Five follow-ups to the tag and profit analysis

`notebooks/tag_profit.ipynb` found that the Tag Genome's tags go with a film's return, and that most of it is the
audience leaking in: the scores are built from what viewers tagged and wrote, and they predict a film's vote count.
This notebook does the five things that finding pointed to.

1. Tags measured from the plot instead of the audience.
2. Whether directors and studios return more when they stay in their usual genre mix.
3. Which films are seen and liked but don't earn, and the reverse.
4. Where critics and viewers disagree, and whether the gap goes with the return.
5. Franchise and release month as controls.

It reads the saved outputs of `scripts/reviews/plot_tags.py` and `scripts/reviews/tag_followups.py`. The return
throughout is ln(gross ÷ budget) with ln(budget) and release decade held fixed. Gross over budget isn't profit:
budgets leave out marketing and cinemas keep part of the gross."""),
    ("code", """import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

SRC = Path.cwd().resolve().parents[1] / "archive" / "analysis" / "tags"
plot = json.loads((SRC / "plot_tags.json").read_text())
fol = json.loads((SRC / "tag_followups.json").read_text())
plot_tags = pd.read_csv(SRC / "plot_tags_tags.csv", keep_default_na=False)
INK, ACCENT, GREY, LIGHT, RED = "#1F2933", "#1F6F8B", "#8A969C", "#D5E3E8", "#B5482F"
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

rec, leak, ret, scr = plot["recovery"], plot["leak"], plot["return"], plot["screenplays"]
um, au, cr, ct = fol["usual_mix"], fol["audience"], fol["critics"], fol["controls"]
lab = "IMDb genre labels (all years)"
pe, sf = ret["sets"]["plot embedding"], ret["scored_films"]
display(Markdown(f\"\"\"**Short answers.**

1. A plot summary recovers a film's genres well (AUC {rec['genre_auc_mean']['plot_films_after_2014']:.2f} against IMDb's
   labels on films after 2014, which the Tag Genome doesn't cover) and carries little of the audience: it predicts
   votes with R² {leak['r2_votes_from_plot_embedding']:.2f}, against {leak['r2_votes_from_tag_genome_content_tags']:.2f}
   for the Tag Genome's tags. Measured this way, content accounts for {pe['random_folds']:.0%} of the return on
   random folds and {max(pe['trained_before_2005'], 0):.0%} of the return of later films, against
   {sf['Tag Genome content tags']['trained_before_2005']:.0%} for the Tag Genome's tags on the films it covers.
2. Films far from their director's or studio's usual genre mix return less: rho
   {um[lab]['director']['against_adjusted_genre_fixed']['rho']:+.2f} for directors and
   {um[lab]['studio']['against_adjusted_genre_fixed']['rho']:+.2f} for studios, with the film's own genres held fixed.
3. Liking and earning agree only loosely: the IMDb rating goes with the return at rho
   {au['rating_vs_adjusted']['rho']:+.2f}. {au['quadrants']["liked, didn't earn"]:.0%} of films are liked and
   didn't earn, and {au['quadrants']['not liked, earned']:.0%} are the reverse.
4. Critics and viewers disagree by genre in a stable way, and the size of the gap doesn't go with the return
   (rho {cr['metacritic']['gap_vs_adjusted']['rho']:+.2f} with Metacritic, {cr['rotten_tomatoes']['gap_vs_adjusted']['rho']:+.2f}
   with Rotten Tomatoes).
5. Following another film accounts for {ct['with_controls']['by_follows_alone']:.1%} of the variance in the return and
   release month for {ct['release_month']['share_of_variance_in_adjusted']:.1%}. Holding both fixed leaves the tag results
   in place.\"\"\"))"""),
    ("md", """## 1. Tags from plots

**What we did.** Each film's Wikipedia plot section is embedded with the same sentence model as the screenplays
(`scripts/reviews/plot_embeddings.py`). A ridge regression then predicts all 1,084 Tag Genome scores from that
embedding, trained on the films that have both. Every scored film's tags come from a model that never saw that
film. A plot says what happens in the film. It doesn't say how many people watched it, which is the leak."""),
    ("code", """print(f"{plot['films_with_plot']:,} of {plot['films_in_graph']:,} films have a plot section; {plot['films_with_plot_and_tag_scores']:,} also have Tag Genome scores")
print(f"plot length: median {plot['plot_words']['median']:.0f} words, 10th to 90th percentile {plot['plot_words']['p10']:.0f} to {plot['plot_words']['p90']:.0f}")
bc = pd.DataFrame(rec["by_class"]).T
bc.columns = ["tags", "median r", "10th percentile r", "tags with r over 0.5"]
print("\\nhow well a plot recovers a tag (Pearson r with the Tag Genome score, out of fold):")
display(bc.style.format({"tags": "{:.0f}", "median r": "{:.2f}", "10th percentile r": "{:.2f}", "tags with r over 0.5": "{:.0f}"}))
print("best recovered:", ", ".join(f"{t} {r:.2f}" for t, r in rec["best_tags"][:10]))
print("worst recovered:", ", ".join(f"{t} {r:.2f}" for t, r in rec["worst_tags"][:10]))"""),
    ("md", """A plot recovers what a film is about (sports, war, space, murder) and can't recover who made it
("tarantino", "aardman") or a single odd subject. Reception tags are recovered about as well as content tags,
which says that a good part of "funny" or "boring" follows from the kind of film it is.

The check against IMDb's genre labels is the important one, because it can be run on the films after 2014 that the
Tag Genome never scored."""),
    ("code", """auc = pd.DataFrame(rec["genre_auc"]).T
auc.columns = ["plot, scored films", "Tag Genome, scored films", "plot, films after 2014", "IMDb-labelled films after 2014"]
display(auc.sort_values("plot, films after 2014").style.format({c: "{:.3f}" for c in auc.columns[:3]} | {auc.columns[3]: "{:.0f}"}, na_rep="too few"))
m = rec["genre_auc_mean"]
print(f"mean AUC: plot on scored films {m['plot_scored_films']:.3f}, Tag Genome on the same films {m['tag_genome_scored_films']:.3f}, "
      f"plot on the {rec['films_after_2014']:,} films after 2014 {m['plot_films_after_2014']:.3f}")
t = rec["top_share_genre_in_imdb"]
print(f"largest plot-based genre share is one of the film's IMDb genres: {t['scored_films']:.1%} of scored films, {t['films_after']:.1%} of films after 2014")
print(f"same top genre as the Tag Genome: {rec['top_genre_agrees_with_tag_genome']:.1%}")"""),
    ("md", """The plot-based genres are a little weaker than the Tag Genome's and hold up almost unchanged on films from
after 2014. Their top genre matches the Tag Genome's less than half the time, mostly because a predicted score is
smoother, so the largest share often goes to a broad genre such as Drama or Action. Use them for "which genres",
not for "exactly how much".

**The leak check.** How well each set of features predicts ln(IMDb votes), with budget and decade held fixed."""),
    ("code", """rows = {"Tag Genome content tags (scored films)": leak["r2_votes_from_tag_genome_content_tags"],
        "plot embedding (scored films)": leak["r2_votes_from_plot_embedding_scored_films"],
        "plot embedding (all clean films)": leak["r2_votes_from_plot_embedding"],
        "plot-predicted content tags": leak["r2_votes_from_plot_content_tags"],
        "IMDb genre labels": leak["r2_votes_from_imdb_genre_labels"]}
display(pd.Series(rows, name="R² for votes").to_frame().style.format("{:.3f}"))
print(f"plot length against votes: rho {leak['rho_plot_words_with_votes']:+.3f}; against the return: rho {leak['rho_plot_words_with_adjusted']:+.3f}")"""),
    ("md", """The plot carries a small part of the audience signal the Tag Genome carries. It isn't zero: better-known
films get longer plot summaries, and some kinds of film do draw bigger audiences.

**The return.** Out-of-sample R² of the adjusted return. "r" is the correlation between predicted and actual
returns on the later films, which shows the direction even when R² is near zero."""),
    ("code", """rows = {}
for name, v in ret["sets"].items():
    rows[name] = {"features": v["features"], "random folds": v["random_folds"], "highest with outcome shuffled": v["null_max"],
                  "trained before 2005: R²": v["trained_before_2005"], "r": v["r_trained_before_2005"],
                  "trained to 2014, tested after: R²": v["trained_to_2014_tested_after"], "r ": v["r_trained_to_2014_tested_after"],
                  "votes fixed, random folds": v["votes_fixed_random_folds"]}
p = pd.DataFrame(rows).T
display(p.style.format({c: "{:.3f}" for c in p.columns[1:]} | {"features": "{:.0f}"}))
print(f"{ret['films']:,} clean films with a plot; {ret['films_from_2005']:,} from 2005 on; {ret['films_after_2014']:,} after 2014")
same = pd.DataFrame({k: v for k, v in sf.items() if isinstance(v, dict)}).T
same.columns = ["random folds", "trained before 2005: R²", "r", "votes fixed, trained before 2005: R²"]
print(f"\\nthe same test on the {sf['films']:,} films that also have Tag Genome scores:")
display(same.style.format("{:.3f}"))"""),
    ("md", """Read the second table first. On the same films and the same split, the Tag Genome's tags account for several
times more of the return than the plot does, and holding votes fixed closes little of the gap. Two things could
make up the difference. The Tag Genome may describe content more finely than a 600-word summary can. Or its scores,
built from reviews and ratings, hold more of the reception than the vote count captures. These data can't separate
the two.

What can be said: content measured from the plot accounts for a few percent of the return, a little more than
IMDb's genre labels, and it carries forward to later films, including the films after 2014. So the range for "what
a film is about" runs from a few percent (plots) to about a quarter (Tag Genome), and only the low end is free of
the audience."""),
    ("code", """pt = ret["per_tag"]
ag = pt["agreement_with_tag_genome"]
print(f"{pt['content_tags_recovered_r_over_0.5']} content tags are recovered with r over 0.5; family-wise threshold {pt['familywise_95']:.3f}; "
      f"largest |rho| of a content tag {pt['largest_abs_rho_content']:.3f}")
print(f"plot-based rho against Tag Genome rho for the same tag, same films: r {ag['r_between_rhos_recovered_content']:.2f}, "
      f"same sign for {ag['same_sign_recovered_content']:.0%}; median |rho| {ag['median_abs_rho_plot']:.3f} from plots, {ag['median_abs_rho_tag_genome']:.3f} from the Tag Genome")
d = plot_tags[(plot_tags["class"] == "content") & (plot_tags["recovery_r"] > 0.5)].sort_values("rho_adjusted_plot")
d = pd.concat([d.head(12), d.tail(12)])
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
a.barh(d["tag"], d["rho_adjusted_plot"], color=[RED if v < 0 else ACCENT for v in d["rho_adjusted_plot"]])
for v in (pt["familywise_95"], -pt["familywise_95"]):
    a.axvline(v, color=INK, ls="--", lw=0.8)
a.set_xlabel("rho with the adjusted return"); a.set_title("Plot-predicted tags", loc="left", fontsize=9)
u = plot_tags[(plot_tags["class"] == "content") & (plot_tags["recovery_r"] > 0.5)]
b.scatter(u["rho_adjusted_tag_genome"], u["rho_adjusted_plot_scored_films"], s=8, color=ACCENT, alpha=0.6)
lim = [-0.4, 0.4]; b.plot(lim, lim, color=GREY, lw=0.8); b.axhline(0, color=GREY, lw=0.5); b.axvline(0, color=GREY, lw=0.5)
b.set_xlabel("rho from the Tag Genome"); b.set_ylabel("rho from the plot"); b.set_title("Same tag, same films", loc="left", fontsize=9)
plt.tight_layout(); plt.show()"""),
    ("md", """The direction survives: family, fairy-tale and children's films above their budget, gangster, corruption
and character-study films below. The sizes are about half of what the Tag Genome showed, and two thirds of the
tags keep their sign. So the earlier picture had roughly the right shape and was too large.

**Screenplays** are the second pre-release source, on the films that have one."""),
    ("code", """s = scr
print(f"{s['films_with_script_and_tag_scores']:,} films with a screenplay and Tag Genome scores")
print(f"content tags, median r: screenplay {s['recovery_by_class']['content']['median_r']:.2f}, plot on the same films {s['plot_on_the_same_films']['by_class']['content']['median_r']:.2f}; "
      f"the two recover the same tags well and badly (r {s['r_between_per_tag_recovery_of_script_and_plot']:.2f} across tags)")
r = pd.DataFrame({k: v for k, v in s["return"].items() if isinstance(v, dict)}).T
r.columns = ["random folds", "highest with outcome shuffled", "trained before 2005: R²", "r"]
print(f"\\nthe return, {s['return']['films']:,} clean films with a screenplay ({s['return']['films_from_2005']:,} from 2005):")
display(r.style.format("{:.3f}"))"""),
    ("md", """A screenplay recovers tags slightly better than a plot summary, and the two agree closely on which tags
are recoverable, so the plot results aren't an oddity of Wikipedia's summaries. On the return a screenplay does no
better than a plot. The screenplay sample is small and tilted to well-known films, so its numbers are rough."""),
    ("md", """### Any tag, for every year

The table behind all of this, `features/plot_tag_relevance.npy`, has all 1,084 tags for every film with a plot,
including the films after 2014 that the Tag Genome never scored. It comes in the `plot-tags-2026-11-16` bundle, so
this part reads `data/raw` and needs `./setup.sh --data` first.

Only tags a plot recovers well are worth reading this way. The cell below keeps tags with r of at least 0.5 against
the Tag Genome and, for a few of them, lists the films the plot scores highest."""),
    ("code", """FEAT = Path.cwd().resolve().parents[1] / "data" / "raw" / "features"
if not (FEAT / "plot_tag_relevance.npy").exists():
    print("data/raw/features/plot_tag_relevance.npy is missing: run ./setup.sh --data from the repo root")
else:
    prel = np.load(FEAT / "plot_tag_relevance.npy")
    pids = pd.read_csv(FEAT / "plot_tag_relevance_ids.csv")
    tag_list = pd.read_csv(FEAT / "tag_relevance_tags.csv", keep_default_na=False)["tag"].tolist()
    pfilms = pd.read_csv(FEAT / "plot_genre_shares.csv").set_index("tconst").reindex(pids["tconst"]).reset_index()
    grel = np.load(FEAT / "tag_relevance.npy")
    gids = {t: i for i, t in enumerate(pd.read_csv(FEAT / "tag_relevance_ids.csv")["tconst"])}
    rec = dict(zip(plot_tags["tag"], plot_tags["recovery_r"]))
    good = [t for t in tag_list if rec.get(t, 0) >= 0.5]
    late = (pfilms["y"] > 2014).to_numpy()
    print(f"{prel.shape[0]:,} films x {prel.shape[1]:,} tags; {late.sum():,} films after 2014; {len(good)} tags recovered at r 0.5 or more")

    def top_films(tag, n=10):
        j = tag_list.index(tag)
        order = np.argsort(-prel[:, j])[:n]
        out = pfilms.loc[order, ["primaryTitle", "y", "genres"]].assign(plot_score=prel[order, j].round(3))
        out["Tag Genome score"] = [round(float(grel[gids[t], j]), 3) if t in gids else None for t in pfilms["tconst"][order]]
        return out.reset_index(drop=True)

    for tag in [t for t in ("space", "heist", "vampires", "martial arts", "high school", "zombies") if t in good][:4]:
        d = top_films(tag)
        print(f"\\n{tag} (recovered at r {rec[tag]:.2f}); {int((d['y'] > 2014).sum())} of the top 10 are after 2014")
        display(d)"""),
    ("code", """if (FEAT / "plot_tag_relevance.npy").exists():
    # does the plot rank films like the Tag Genome does, tag by tag, on the films that have both?
    both = np.array([t in gids for t in pfilms["tconst"]])
    pr, gr = prel[both], grel[[gids[t] for t in pfilms["tconst"][both]]]
    k = 20
    overlap = []
    for t in good:
        j = tag_list.index(t)
        a, b = set(np.argsort(-pr[:, j])[:k]), set(np.argsort(-gr[:, j])[:k])
        overlap.append(len(a & b) / k)
    print(f"on the {both.sum():,} films with both scores, the plot's top {k} films for a tag share a median of "
          f"{np.median(overlap):.0%} with the Tag Genome's top {k} (10th to 90th percentile {np.percentile(overlap, 10):.0%} to "
          f"{np.percentile(overlap, 90):.0%}), over the {len(good)} well-recovered tags")
    # how common each tag is among later films, against earlier ones. Predicted scores are pulled towards the middle,
    # so a film counts as strong on a tag when it is in that tag's top 5% over all films, not above a fixed 0.5
    cut = np.percentile(prel, 95, axis=0)
    strong = prel >= cut
    early = ~late & (pfilms["y"] >= 1990).to_numpy()
    ch = pd.DataFrame({"tag": tag_list, "1990-2014": strong[early].mean(axis=0), "after 2014": strong[late].mean(axis=0)})
    ch = ch[ch["tag"].isin(good)].assign(change=lambda d: d["after 2014"] - d["1990-2014"]).sort_values("change")
    print(f"highest plot score of any film, median over tags: {np.median(prel.max(axis=0)):.2f}")
    print("\\nshare of films in a tag's top 5%, 1990 to 2014 against after 2014 (5% would mean no change; well-recovered tags):")
    display(pd.concat([ch.head(8), ch.tail(8)]).style.format({"1990-2014": "{:.1%}", "after 2014": "{:.1%}", "change": "{:+.1%}"}).hide(axis="index"))"""),
    ("md", """The top lists hold films that fit the tag, from before and after 2014 alike. But they are not the Tag
Genome's lists: on films that have both, the two top-20 lists share only a few films. A plot score is a regression
prediction, pulled towards the middle (a plot score rarely passes 0.7, where the Tag Genome gives 0.99), and with a
per-tag r of 0.5 to 0.75 the ordering at the very top is loose. So use plot scores to compare groups of films or to
find candidates for a tag, and the Tag Genome where it exists for one film's exact profile.

The last table compares what plots were about before and after 2014, using each tag's top 5% so the pulled-in scale
doesn't matter. It describes the films in our sample, which over-represents well-known films, and Wikipedia's way
of summarising plots may have changed over time as well as the films."""),
    ("md", """## 2. The usual genre mix

A film's mix is its genre vector. A director's or studio's usual mix at a film is the mean vector of their
strictly earlier films (at least two), so the film's own outcome can't leak in. The departure is 1 minus the
cosine between the two: 0 is "the same mix as always", 1 is "nothing in common".

The mix is measured two ways. IMDb's genre labels cover every year and were not built from audience data, so they
are the main version. The Tag Genome's genre shares are finer and stop in 2014."""),
    ("code", """rows = []
for vec, by_role in um.items():
    for role, o in by_role.items():
        tr = o["track_record"]
        rows.append({"mix measured by": vec, "whose": role, "films": o["films"], "median departure": o["median_departure"],
                     "rho with return": o["against_adjusted"]["rho"], "rho, own genres fixed": o["against_adjusted_genre_fixed"]["rho"],
                     "p": o["against_adjusted_genre_fixed"]["p"], "null 95%": o["against_adjusted_genre_fixed"]["null_95"],
                     "rho, track record also fixed": tr["rho_departure_net_of_track"]})
display(pd.DataFrame(rows).style.format({"median departure": "{:.2f}", "rho with return": "{:+.3f}", "rho, own genres fixed": "{:+.3f}",
                                         "p": "{:.3f}", "null 95%": "{:.3f}", "rho, track record also fixed": "{:+.3f}"}).hide(axis="index"))
fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
for ax, role in zip(axes, ("director", "studio")):
    for vec, col in zip(um, (GREY, ACCENT)):
        q = pd.DataFrame(um[vec][role]["by_quartile"])
        ax.plot(q["quartile"], q["mean_adjusted_genre_fixed"], marker="o", color=col, label=vec.split(" (")[0])
    ax.axhline(0, color=INK, lw=0.6); ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("quartile of departure (1 = closest to the usual mix)")
    ax.set_title(role, loc="left", fontsize=9)
axes[0].set_ylabel("adjusted return, own genres fixed"); axes[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()"""),
    ("md", """Films close to the usual mix return more, for directors and for studios, under both measures. The effect
is small with IMDb's labels and larger with the Tag Genome's shares, which carry the audience leak, so the label
rows are the ones to quote. For directors measured by labels the line isn't a steady slope: the quarter of films
closest to the usual mix stands out and the other three are level. It holds with the film's own genres fixed, so it isn't just "they left for a weaker
genre", and most of it remains with the track record held fixed.

Does a track record count for more when the film stays in the usual mix?"""),
    ("code", """rows = []
for vec, by_role in um.items():
    for role, o in by_role.items():
        tr = o["track_record"]
        rows.append({"mix measured by": vec, "whose": role, "films": tr["films"], "track record when staying": tr["rho_track_when_staying"],
                     "when leaving": tr["rho_track_when_leaving"], "difference": tr["difference"], "p": tr["p"]})
display(pd.DataFrame(rows).style.format({"track record when staying": "{:+.3f}", "when leaving": "{:+.3f}", "difference": "{:+.3f}", "p": "{:.3f}"}).hide(axis="index"))"""),
    ("md", """The track record predicts the return better for films that stay in the usual mix in all four rows, and the
difference clears chance only for directors measured by IMDb's labels. So there is a lead, mostly for directors: a
director's past returns say more about a film of the kind they usually make.

What this doesn't show: who leaves their usual mix isn't random. A studio short of money, or a director hired for
someone else's project, may both leave the usual mix and return less for other reasons."""),
    ("md", """## 3. Seen and liked against earned

The project's second question is whether money agrees with whether people liked the film. Here every clean film
from 1975 is used, not only those with tag scores."""),
    ("code", """for k, name in (("votes_vs_adjusted", "IMDb votes against the return"), ("votes_vs_gross", "IMDb votes against revenue"),
                ("rating_vs_adjusted", "IMDb rating against the return"), ("rating_vs_gross", "IMDb rating against revenue"),
                ("rating_vs_budget", "IMDb rating against budget"), ("rating_vs_votes", "IMDb rating against votes")):
    print(f"{name}: rho {au[k]['rho']:+.3f} over {au[k]['films']:,} films (95% of shuffles under {au[k]['null_95']:.3f})")
print("rating against the return by decade:", {d: round(v, 2) for d, v in au["rating_vs_adjusted_by_decade"].items()})
q = pd.Series(au["quadrants"], name="share of films")
display(q.to_frame().style.format("{:.1%}"))"""),
    ("md", """How many people saw a film goes with its return far more than how much they liked it. The rating goes with
the return weakly and with revenue more weakly still, because expensive films are rated slightly lower.

"Liked" below means rated above the median of the film's decade, and "earned" means it returned more than its
budget and decade predict. A genre's lift in a group is its share of that group over its share of all films."""),
    ("code", """g = pd.DataFrame({k: {"films": v["films"], **v["lift"]} for k, v in au["per_genre"].items()}).T
qv = pd.DataFrame({k: v["lift_q"] for k, v in au["per_genre"].items()}).T
order = g.sort_values("liked, didn't earn").index
display(g.loc[order].style.format({"films": "{:.0f}"} | {c: "{:.2f}" for c in g.columns[1:]})
        .apply(lambda col: ["font-weight: bold" if qv.loc[i, col.name] < 0.05 else "" for i in col.index] if col.name in qv else [""] * len(col), axis=0))
print("bold: q below 0.05 (hypergeometric test, Benjamini-Hochberg over all genre and group pairs)")
att = pd.DataFrame({k: {"rating": v["mean_rating_z"], "return": v["mean_adjusted"], "attention": v["mean_attention"], "films": v["films"]}
                    for k, v in au["per_genre"].items()}).T
fig, ax = plt.subplots(figsize=(6.5, 4.4))
sc = ax.scatter(att["return"], att["rating"], s=att["films"] / 12, c=att["attention"], cmap="RdBu_r", vmin=-0.5, vmax=0.5, edgecolor=GREY)
for name, r in att.iterrows():
    ax.annotate(name, (r["return"], r["rating"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
ax.set_xlabel("mean adjusted return"); ax.set_ylabel("mean rating (z within decade)")
plt.colorbar(sc, label="votes for its gross"); plt.show()"""),
    ("md", """Money and liking part ways by genre. History, Biography and War films are about twice as common among
films that were liked and didn't earn. Horror is the reverse, nearly twice as common among films that earned
without being liked, with Family behind it. Documentary and Animation are over-represented where both are true.

The colour is attention: votes relative to gross. Sci-Fi gets by far the most votes for the money it took, and
Family the fewest. IMDb's voters are not the cinema audience, which is worth remembering whenever votes or ratings
stand in for "people"."""),
    ("md", """## 4. Critics against audience

The gap is the film's critic score minus its IMDb rating, each standardised within release decade. A positive gap
means critics were kinder than viewers. Metacritic is the main score and Rotten Tomatoes the second method."""),
    ("code", """rows = {}
for k, name in (("metacritic", "Metacritic"), ("rotten_tomatoes", "Rotten Tomatoes")):
    c = cr[k]
    rows[name] = {"films": c["films"], "critics vs IMDb rating": c["rho_score_vs_rating"], "critics vs return": c["score_vs_adjusted"]["rho"],
                  "IMDb rating vs return, same films": c["rating_vs_adjusted_same_films"]["rho"], "gap vs return": c["gap_vs_adjusted"]["rho"],
                  "p of the gap": c["gap_vs_adjusted"]["p"], "gap vs budget": c["gap_vs_budget"]["rho"], "gap vs votes": c["gap_vs_votes"]["rho"]}
t = pd.DataFrame(rows).T
display(t.style.format({"films": "{:.0f}", "p of the gap": "{:.3f}"} | {c: "{:+.3f}" for c in t.columns if c not in ("films", "p of the gap")}))
tm = cr["two_methods"]
print(f"the two gaps correlate at {tm['rho_between_gaps']:.2f} over {tm['films']:,} films; the genre means rank alike (rho {tm['rho_per_genre_mean_gaps']:.2f})")
mc, rt = pd.DataFrame(cr["metacritic"]["per_genre"]).T, pd.DataFrame(cr["rotten_tomatoes"]["per_genre"]).T
g = pd.DataFrame({"Metacritic": mc["mean_gap"], "Rotten Tomatoes": rt["mean_gap"]}).dropna().sort_values("Metacritic")
fig, ax = plt.subplots(figsize=(6.5, 4.6))
y = np.arange(len(g))
ax.barh(y - 0.2, g["Metacritic"], 0.4, color=ACCENT, label="Metacritic"); ax.barh(y + 0.2, g["Rotten Tomatoes"], 0.4, color=LIGHT, label="Rotten Tomatoes")
ax.set_yticks(y, g.index); ax.axvline(0, color=INK, lw=0.6); ax.legend(frameon=False)
ax.set_xlabel("critics minus viewers (standard deviations within decade)"); plt.show()
sig = mc[mc["q"] < 0.05].index.tolist()
print("genres whose gap differs from the rest at q below 0.05 (Metacritic):", ", ".join(sig))"""),
    ("md", """Critics rate Horror, Family, Documentary and Comedy above where viewers put them, and Sport, Biography,
History, War, Action and Crime below. Both critic scores give the same order, so it isn't a quirk of one site.

Critics and viewers each go with the return to about the same degree, and the gap between them doesn't. A film
that critics liked more than viewers did, or less, earns no differently for its budget. The gap is larger for cheap
and little-seen films, which is where the two groups are watching different films."""),
    ("md", """## 5. Franchise and release month

**The franchise flag** comes from Wikidata. It is known to be loose, so here it is checked against two other
signals: the Tag Genome's own sequel tags, and titles that carry a number or "Part"."""),
    ("code", """fr = ct["franchise"]
print(f"{fr['flagged']:.1%} of {fr['films_with_flag']:,} clean films are flagged; {fr['follows_another_film']:.1%} follow another film; "
      f"{fr['numbered_title']:.1%} have a numbered title")
print(f"numbered titles that are flagged: {fr['numbered_titles_flagged']:.1%}; flagged films with a numbered title: {fr['flagged_with_numbered_title']:.1%}; "
      f"films that follow another with a numbered title: {fr['follows_with_numbered_title']:.1%}")
tc = pd.DataFrame(fr["tag_check"]).T
tc.columns = ["AUC for the flag", "AUC for 'follows'", "flagged, among films with the tag at 0.5+", "films with the tag at 0.5+",
              "tag at 0.5+, among flagged", "tag at 0.5+, among films that follow"]
display(tc.style.format({c: "{:.3f}" for c in tc.columns if c != "films with the tag at 0.5+"} | {"films with the tag at 0.5+": "{:.0f}"}))
ma = fr["mean_adjusted"]
print(f"mean adjusted return: flagged {ma['flagged']:+.2f}, not flagged {ma['not_flagged']:+.2f}, follows another film {ma['follows']:+.2f}")
print(f"flag against the return: rho {ma['flag_vs_adjusted']['rho']:+.3f}; 'follows another film' against the return: rho {ma['follows_vs_adjusted']['rho']:+.3f}")"""),
    ("md", """The flag is precise and broad. Nearly every film the Tag Genome calls a sequel is flagged, and nearly every
numbered title. But most flagged films are neither: the flag also marks first films that later got a sequel, and
films that belong to a loose series.

That matters for using it as a control. A film gets a sequel because it earned, so the flag partly records the
outcome, and its correlation with the return is twice that of "follows another film". The fact known before
release is that a film follows another one, so that is the control used below. The full flag is the strict variant.

**Release month** is parsed from the Wikipedia infobox."""),
    ("code", """rm = ct["release_month"]
mo = pd.DataFrame(rm["by_month"]).T
mo.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 2.6))
a.bar(mo.index, mo["films"], color=LIGHT); a.set_ylabel("films")
b.bar(mo.index, mo["mean_adjusted"], color=[RED if v < 0 else ACCENT for v in mo["mean_adjusted"]]); b.axhline(0, color=INK, lw=0.6)
b.set_ylabel("mean adjusted return"); plt.tight_layout(); plt.show()
print(f"{rm['films_with_month']:,} films with a month. Month accounts for {rm['share_of_variance_in_adjusted']:.1%} of the variance in the adjusted return "
      f"(95% of shuffles under {rm['share_null_95']:.1%}) and {rm['share_of_variance_in_budget']:.1%} of the variance in ln(budget)")
display(pd.DataFrame(rm["sample_to_check"]).T.style.hide(axis="columns"))"""),
    ("md", """September has almost twice the films of a typical month and the lowest return. That is the autumn
festivals: the infobox lists the first showing, and the parser takes the first date.

The table above is a sample of 25 well-known films. Read from memory of when each film opened in cinemas, and not
checked against a source, 15 of the 25 months are the general release and the other 10 are an earlier premiere or
festival (The Social Network, Batman Begins, Fight Club, About Time, The Last Samurai, Hugo, Logan, Ant-Man and the
Wasp, Captain America: Civil War, Poor Things). So the column is "month of first showing", wrong for the cinema
release in roughly four films in ten, and the September dip is mostly festival films, which are smaller and
artier. A summer and a holiday effect are visible, and small.

**With the controls.** The tag results from `tag_profit.ipynb` again, with "follows another film" and month held
fixed."""),
    ("code", """wc = ct["with_controls"]
print(f"share of the variance in the adjusted return: follows another film {wc['by_follows_alone']:.1%}, with month {wc['variance_of_adjusted_explained_by_follows_and_month']:.1%}; "
      f"the full flag alone {wc['by_full_flag_alone']:.1%}")
tg = wc["tags"]
print(f"per-tag rho before and after: r {tg['rho_between_before_and_after']:.2f} across all tags; the 40 strongest content tags keep their sign in "
      f"{tg['top_40_content_same_sign']:.0%} of cases and {tg['top_40_content_mean_shrink']:.0%} of their size")
print(f"under the strict variant: {tg['strict_full_flag']['top_40_content_same_sign']:.0%} keep their sign and {tg['strict_full_flag']['top_40_content_mean_shrink']:.0%} of their size")
display(pd.DataFrame(tg["commercial"]).T.style.format("{:+.3f}"))
r2 = pd.DataFrame(wc["content_tags_r2"]).T
r2.columns = ["random folds", "trained before 2005"]
print("Tag Genome content tags, R² for the return:"); display(r2.style.format("{:.3f}"))
gsh = pd.DataFrame(wc["genre_shares"]).T.sort_values("after")
fig, ax = plt.subplots(figsize=(6.5, 4.8))
y = np.arange(len(gsh))
ax.barh(y - 0.27, gsh["before"], 0.27, color=LIGHT, label="budget and decade fixed")
ax.barh(y, gsh["after"], 0.27, color=ACCENT, label="plus follows-a-film and month")
ax.barh(y + 0.27, gsh["strict"], 0.27, color=INK, label="plus the full franchise flag")
ax.set_yticks(y, gsh.index); ax.axvline(0, color=GREY, lw=0.6); ax.set_xlabel("rho of the genre's share with the return"); ax.legend(frameon=False, fontsize=8)
plt.show()"""),
    ("md", """The controls do what they should: the sequel and franchise tags lose a large part of their correlation with
the return, and nearly all of it under the strict variant. The content tags keep their signs and most of their
size, so the earlier findings weren't sequels in disguise.

Genre shares move in a consistent way. Action, Adventure, Animation, Fantasy and Sci-Fi go down, because much of
their return comes from sequels, and Drama, Biography and History go up.

## What to take from this

- **For the paper's first question**, what a film is about, measured without its audience, accounts for a few
  percent of its return, and that small part does hold on later films. The Tag Genome's much larger figure is an
  upper bound. The plot-based tags are the version to use if tags go into a model, and they exist for every year.
- **The usual-mix result is new and small.** It is leak-free by construction and survives its controls. It is an
  association, and worth a paragraph next to the network results.
- **For the second question**, liking and earning agree weakly, and the disagreement has a clear shape by genre.
  This part needs no tag scores and uses the whole clean sample.
- **Critics against viewers** is a stable description of genres and tells us nothing about money.
- **Use "follows another film" as the franchise control**, and treat release month as the month of first showing
  until the parser picks the general release.
- **Limits that apply throughout**: gross over budget isn't profit, the clean flags are the ones stored in the
  graph bundle, a plot summary is written after release, and every correlation here is over films that reported
  both a budget and a gross."""),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    if "--no-run" not in sys.argv:
        from nbconvert.preprocessors import ExecutePreprocessor
        ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
    for cell in nb.cells:
        cell.pop("id", None)
    nb.nbformat_minor = 4
    nbf.write(nb, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
