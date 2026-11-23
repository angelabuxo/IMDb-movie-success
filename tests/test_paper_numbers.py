"""Every number in paper/main.tex, recomputed from the committed outputs.

Each claim is a snippet of the paper with `@` where a number goes, and a function that reads the same numbers from
an output file. The test finds the snippet, rounds each source value to as many decimals as the paper prints, and
fails if they differ. A rerun that moves a result therefore fails here until the paper is updated.

`test_every_number_is_accounted_for` fails on any number in the paper that no claim covers and that isn't a
design constant (vote thresholds, CI level, ...), a year, or a small count. A number typed in by hand has to be
added to CLAIMS with its source, or it fails.

The reception numbers come from analysis/reviews/outputs on the branch reviews-exploration, which isn't merged.
They are read with `git show` when a ref to that branch exists and skipped, with that reason, when it doesn't.

Reads only committed files (outputs, notebooks, data/movie_links.csv): nothing from data/raw or private/.
"""

import json
import re
import subprocess
from functools import cached_property
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "main.tex"
A = ROOT / "analysis"
REVIEW_REFS = ("public/reviews-exploration", "origin/reviews-exploration", "reviews-exploration")

NUM = r"(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"


class SourceMissing(Exception):
    pass


class Sources:
    """Lazily loaded outputs. Every value a claim reads comes from one of these."""

    def _json(self, rel):
        return json.loads((A / rel).read_text())

    @cached_property
    def roi(self):
        return self._json("network/roi_similarity.json")

    def sim(self, variant):
        roles = self.roi["results"][variant]["roles"]
        return [r["similarity"] for r in roles.values()]

    def null95(self, variant):
        return [r["similarity_null_95"] for r in self.roi["results"][variant]["roles"].values()]

    def within_studio_kept(self):
        """Share of each people role's similarity above the within-studio null mean."""
        adj, ws = self.roi["results"]["adjusted"]["roles"], self.roi["results"]["adjusted_within_studio"]["roles"]
        people = [k for k in ws if k not in ("studio", "distributor")]
        return [(ws[k]["similarity"] - ws[k]["similarity_null_mean"]) / adj[k]["similarity"] for k in people]

    @cached_property
    def s01(self):
        return self._json("scripts_text/outputs/s01_summary.json")

    @cached_property
    def t06(self):
        return self._json("scripts_text/outputs/s06_summary.json")

    @cached_property
    def t09(self):
        return self._json("scripts_text/outputs/s09_summary.json")

    @cached_property
    def t10(self):
        return self._json("scripts_text/outputs/s10_summary.json")

    @cached_property
    def t10_runs(self):
        return pd.read_csv(A / "scripts_text/outputs/s10_results.csv").set_index("run")

    @cached_property
    def links(self):
        return pd.read_csv(ROOT / "data/movie_links.csv", usecols=["imdb_id"])

    @cached_property
    def nb(self):
        """All printed output of notebooks/what_predicts_profit.ipynb, as one string."""
        cells = json.loads((ROOT / "notebooks/what_predicts_profit.ipynb").read_text())["cells"]
        out = []
        for c in cells:
            for o in c.get("outputs", []):
                out.append("".join(o.get("text", [])) or "".join(o.get("data", {}).get("text/plain", [])))
        return "\n".join(out)

    def nb_find(self, pattern):
        m = re.search(pattern, self.nb)
        if not m:
            raise AssertionError(f"notebook output no longer matches {pattern!r}")
        return [float(g.rstrip(".")) for g in m.groups()]  # a value can end a sentence

    def _branch(self, rel):
        for ref in REVIEW_REFS:
            r = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                return r.stdout
        raise SourceMissing(f"{rel} is only on the unmerged branch reviews-exploration; no ref {REVIEW_REFS} here")

    @cached_property
    def rev_rho(self):
        from io import StringIO
        d = pd.read_csv(StringIO(self._branch("analysis/reviews/outputs/spearman_correlations.csv")))
        return {(r.a, r.b): r for r in d.itertuples()}

    @cached_property
    def rev_cov(self):
        from io import StringIO
        d = pd.read_csv(StringIO(self._branch("analysis/reviews/outputs/movie_table_coverage.csv")))
        return dict(zip(d.iloc[:, 0], d.value))

    @cached_property
    def rev_vader(self):
        return json.loads(self._branch("analysis/reviews/outputs/sentiment_validation.json"))["stanford_vader"]

    @cached_property
    def rev_readme(self):
        return self._branch("analysis/reviews/README.md")

    # ---- Q1 genre (analysis/genre/outputs) ----
    @cached_property
    def genre(self):
        return self._json("genre/outputs/summary.json")

    @cached_property
    def gsh(self):
        d = pd.read_csv(A / "genre/outputs/genre_shuffle.csv")
        return d[(d.set == "usable_money") & (d.outcome == "ln_ratio") & (d.null == "budget")].set_index("genre")

    @cached_property
    def tiers(self):
        d = pd.read_csv(A / "genre/outputs/checks/tiers.csv")
        return d.set_index(["tier", "genre"])

    @cached_property
    def tmdb(self):
        return self._json("genre/outputs/tmdb/summary.json")

    @cached_property
    def hsub(self):
        return pd.read_csv(A / "genre/outputs/tmdb/horror_subgenres.csv").set_index("subgenre")

    @cached_property
    def wdh(self):
        return pd.read_csv(A / "genre/outputs/categories/wikidata_horror_subgenres.csv").set_index("category")

    @cached_property
    def kwn(self):
        return self._json("genre/outputs/categories/summary.json")["keyword_network"]

    def kw_excess(self, first_keyword):
        d = pd.read_csv(A / "genre/outputs/categories/keyword_category_tests.csv")
        row = d[d.category.str.split(": ").str[1] == first_keyword].iloc[0]
        assert row.p_holm < .05, f"keyword category {first_keyword} no longer passes"
        return row.excess

    @cached_property
    def stab(self):
        return self._json("genre/outputs/category_stability/summary.json")

    def stab_always(self, label):
        d = pd.read_csv(A / "genre/outputs/category_stability/per_category.csv")
        d = d[(d.category.str.split(": ").str[1] == label) & d.share_same_sign_holm_05.notna()]
        assert len(d) and (d.share_same_sign_holm_05 == 1).all(), f"{label} does not pass in every stability run"
        return True

    @cached_property
    def br(self):
        return self._json("genre/outputs/before_release/summary.json")

    def br_row(self, file, films, term):
        d = pd.read_csv(A / "genre/outputs/before_release" / file)
        return d[(d.films == films) & (d.term == term)].iloc[0]

    # ---- Q2 people network (analysis/network/q2_network.json, merged from Q2-Niklas in PR 40) ----
    @cached_property
    def q2(self):
        if (A / "network/q2_network.json").exists():
            return self._json("network/q2_network.json")
        for ref in ("public/Q2-Niklas", "origin/Q2-Niklas", "Q2-Niklas"):
            r = subprocess.run(["git", "show", f"{ref}:analysis/network/q2_network.json"], cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                return json.loads(r.stdout)
        raise SourceMissing("analysis/network/q2_network.json is only on the unmerged branch Q2-Niklas; no ref to it here")

    def q2r(self, sample, role, key="assortativity"):
        return self.q2["samples"][sample][role][key]

    def q2_null95_max(self, sample):
        d = self.q2["samples"][sample]
        return max(v["null_95"] for v in d.values() if isinstance(v, dict) and "null_95" in v)

    # ---- Q3 screenplays (scripts-analysis) ----
    @cached_property
    def w2v(self):
        return pd.read_csv(ROOT / "scripts-analysis/word2vec/results/primary.csv")

    def auc(self, genre, layer, rep):
        d = self.w2v
        return d[(d.genre == genre) & (d.layer == layer) & (d.representation == rep)].iloc[0]

    @cached_property
    def combo(self):
        return pd.read_csv(ROOT / "scripts-analysis/word2vec/results/combo_interval.csv").set_index(["genre", "layer"])

    def votes(self, genre, sset, model, layer):
        d = pd.read_csv(ROOT / "scripts-analysis/word2vec/results/votes_check.csv")
        return d[(d.genre == genre) & (d.set == sset) & (d.model == model) & (d.layer == layer)].iloc[0].auc

    @cached_property
    def manual(self):
        d = pd.read_csv(ROOT / "scripts-analysis/data_check/manual_check_summary.csv")
        return d[d.sheet == "main"]

    @cached_property
    def manual_films(self):
        return len(pd.read_csv(ROOT / "scripts-analysis/data_check/sample_films.csv"))

    # ---- head to head and network by genre (analysis/compare/outputs) ----
    @cached_property
    def h2h(self):
        return pd.read_csv(A / "compare/outputs/head_to_head/assortativity.csv").set_index(["network", "links"])

    @cached_property
    def h2h_sum(self):
        return self._json("compare/outputs/head_to_head/summary.json")

    @cached_property
    def nbg(self):
        return pd.read_csv(A / "compare/outputs/network_by_genre/assortativity_by_genre.csv").set_index(["sample", "role", "split"])

    def nbx(self, sample, role, split, col="assortativity"):
        return self.nbg.loc[(sample, role, split), col]

    # ---- communities of the people network (analysis/compare/network_communities.py) ----
    @cached_property
    def com(self):
        return self._json("compare/outputs/communities/summary.json")

    @cached_property
    def com_methods(self):
        return pd.read_csv(A / "compare/outputs/communities/methods.csv").set_index("method")

    @cached_property
    def com_prof(self):
        return pd.read_csv(A / "compare/outputs/communities/community_profiles.csv")

    @cached_property
    def com_ret(self):
        return pd.read_csv(A / "compare/outputs/communities/community_returns.csv")

    def money(self, score):
        return self.rev_rho[("log_gross_over_budget", score)]


S = Sources()
G = "log_gross_over_budget"

# (id, snippet of main.tex with @ for each number, function returning the source values in the same order)
CLAIMS = [
    # Abstract
    ("abs graph", "into a network of @ films", lambda s: [s.links.imdb_id.nunique()]),
    ("abs scripts", "add @ screenplays, and measure", lambda s: [s.genre["sets"]["in_common_set"]["films"]]),
    ("abs genre R2", "genre adds only @ to the explained variance", lambda s: [s.genre["sets"]["usable_money"]["r2_added_by_genre"]]),
    ("abs assort", "shuffles allow (@ against at most @)",
     lambda s: [s.q2r("in_common_set_adjusted", "director"), s.q2_null95_max("in_common_set_adjusted")]),
    # Results: genre
    ("genre films", "Among the @ films with clean money figures, we compared each IMDb genre's mean return with @ shuffles",
     lambda s: [s.genre["sets"]["usable_money"]["films"], s.genre["perms"]]),
    ("genre above", "Animation (+@), Family (+@) and Horror (+@) return more",
     lambda s: [s.gsh.loc[g, "excess"] for g in ("Animation", "Family", "Horror")]),
    ("genre below", r"Drama ($-@$), Action ($-@$), Crime ($-@$), History ($-@$) and War ($-@$) return less",
     lambda s: [-s.gsh.loc[g, "excess"] for g in ("Drama", "Action", "Crime", "History", "War")]),
    ("genre tiers", r"between \$5M and \$30M, Animation is +@, Family +@ and Horror +@",
     lambda s: [s.tiers.loc[("$5-30M", g), "excess"] for g in ("Animation", "Family", "Horror")]),
    ("genre R2", r"Budget decile and era give $R^2=@$, and all @ genres add @. On the @ films with a screenplay, genre adds @.",
     lambda s: [s.genre["sets"]["usable_money"]["r2_budget_era"], len(s.genre["sets"]["usable_money"]["genres_kept"]),
                s.genre["sets"]["usable_money"]["r2_added_by_genre"], s.genre["sets"]["in_common_set"]["films"],
                s.genre["sets"]["in_common_set"]["r2_added_by_genre"]]),
    ("supernatural tmdb", r"beat other horror films of the same budget and era by +@ (@ films; median +@ against +@, and @\% beat the median of other horror)",
     lambda s: [s.hsub.loc["supernatural", "excess"], s.hsub.loc["supernatural", "n_films"], s.hsub.loc["supernatural", "median"],
                s.hsub.loc["supernatural", "median_other_horror"], 100 * s.hsub.loc["supernatural", "share_beating_other_horror_median"]]),
    ("supernatural wikidata", "its supernatural horror films beat other Horror by +@ (@ films)",
     lambda s: [s.wdh.loc["supernatural horror film", "excess"], s.wdh.loc["supernatural horror film", "films"]]),
    ("keyword network", "(@ links among @ keywords) into @ categories with modularity @, against at most @ for the same backbone rewired",
     lambda s: [s.kwn["backbone_links"], s.kwn["nodes_in_backbone"], s.kwn["categories"], s.kwn["modularity"],
                s.stab["rewired_modularity_best_of_20_max"]]),
    ("keyword categories", r"supernatural horror (+@) and superheroes with 3D animation (+@) above, dystopia ($-@$) and action with gangsters ($-@$) below",
     lambda s: [s.kw_excess("supernatural horror"), s.kw_excess("villain"), -s.kw_excess("dystopia"), -s.kw_excess("action hero")]),
    ("keyword stability", r"Rebuilt over @ null draws, @ subsamples of @\% of the films and five changed settings, the supernatural horror category passes in every run",
     lambda s: [s.stab["null_draws"], s.stab["boot"], 100 * s.stab["frac"]] if s.stab_always("supernatural horror") else []),
    ("fig genre films", r"a film counts in every genre it has), @ films. Grey boxes: the middle 95\% of @ shuffles",
     lambda s: [s.genre["sets"]["usable_money"]["films"], s.genre["perms"]]),
    ("fig genre holm", "Holm correction over the @ genres", lambda s: [len(s.genre["sets"]["usable_money"]["genres_kept"])]),
    # Results: people
    ("q2 films", "On the @ films of the common set, we linked", lambda s: [s.q2["samples"]["in_common_set"]["films"]]),
    ("q2 roles", "Every role beats @ shuffles of returns within era bins: @ for directors and @ for producers, @ for actors and @ for studios, against null 95th percentiles of at most @",
     lambda s: [s.q2["shuffles"], *[s.q2r("in_common_set_adjusted", r) for r in ("director", "producer", "actor", "studio")],
                s.q2_null95_max("in_common_set_adjusted")]),
    ("q2 wide", "On all @ films the director and producer values rise to @",
     lambda s: [s.q2["samples"]["usable_money"]["films"], s.q2r("usable_money_adjusted", "director")]
     if round(s.q2r("usable_money_adjusted", "director"), 3) == round(s.q2r("usable_money_adjusted", "producer"), 3) else [None]),
    ("R2", r"explain none of the variance in return ($R^2=@$), and adding each role's track record, experience and centrality, computed only from earlier films, raises $R^2$ to @",
     lambda s: s.nb_find(r"ridge R2: base (-?[\d.]+); base \+ network ([\d.]+)")),
    ("specialists", r"it beats its budget by +@ per standard deviation (95\% interval +@ to +@; @ films, sequels controlled), and by +@ with sequels left out. Shuffling that experience within genre $\times$ budget decile gives $p=@$. Experience in general does not help ($-@$ per standard deviation)",
     lambda s: [*(lambda r: [r.coef_per_sd, r.lo, r.hi, r.n])(s.br_row("specialists_ols.csv", "history: all earlier IMDb movies, sequel flag controlled", "genre_experience")),
                s.br_row("specialists_ols.csv", "history: all earlier IMDb movies, sequels left out", "genre_experience").coef_per_sd,
                s.br["specialists"]["shuffle_p_genre_experience"],
                -s.br_row("specialists_ols.csv", "history: all earlier IMDb movies, sequel flag controlled", "prior_credits").coef_per_sd]),
    ("fig people", r"on the @ films of the common set. Red marks: 95th percentile of @ shuffles of returns within era bins. Every role lies above it ($p=@$, the smallest possible)",
     lambda s: [s.q2["samples"]["in_common_set"]["films"], s.q2["shuffles"],
                max(s.q2r("in_common_set_adjusted", r, "p") for r in ("director", "actor", "producer", "studio"))]),
    # Results: screenplays
    ("q3 groups", "(Comedy @ against @ films, Action @ against @)",
     lambda s: [s.auc("Comedy", "dialogue", "w2v").n_hit, s.auc("Comedy", "dialogue", "w2v").n_flop,
                s.auc("Action", "dialogue", "w2v").n_hit, s.auc("Action", "dialogue", "w2v").n_flop]),
    ("q3 auc", r"The vectors beat @ label shuffles in all four tests (Fig.~\ref{fig:script}), with AUC from @ to @",
     lambda s: [s.auc("Comedy", "dialogue", "w2v").perms,
                *(lambda v: [min(v), max(v)])([s.auc(g, l, "w2v").auc for g in ("Comedy", "Action") for l in ("dialogue", "description")])]
     if all(s.auc(g, l, "w2v").shuffle_p < .05 for g in ("Comedy", "Action") for l in ("dialogue", "description")) else [None]),
    ("q3 baseline", "baseline of vote tier, era and the film's other genres (@ in Comedy, @ in Action)",
     lambda s: [s.auc("Comedy", "-", "baseline").auc, s.auc("Action", "-", "baseline").auc]),
    ("q3 action gain", r"they help in Action (+@ for dialogue, 95\% interval +@ to +@; +@ for description, +@ to +@) but not in Comedy",
     lambda s: [v for l in ("dialogue", "description") for v in s.combo.loc[("Action", l), ["difference", "diff_ci95_lo", "diff_ci95_hi"]]]
     if all(s.combo.loc[("Comedy", l), "diff_ci95_lo"] < 0 < s.combo.loc[("Comedy", l), "diff_ci95_hi"] for l in ("dialogue", "description")) else [None]),
    ("q3 votes", "a baseline with log votes reaches @ and adding either vector gives @ and @",
     lambda s: [s.votes("Action", "100k+", "baseline+votes", "-"), s.votes("Action", "100k+", "w2v+baseline+votes", "dialogue"),
                s.votes("Action", "100k+", "w2v+baseline+votes", "description")]),
    ("franchise", "drops from $z=@$ to $z=@$ once likely franchise films are removed",
     lambda s: [s.t10_runs.loc["primary", "z"], s.t10_runs.loc["without likely franchise films", "z"]]),
    ("typicality", r"Typical scripts beat their budgets by +@ per standard deviation (95\% interval +@ to +@; @ films), with length, dialogue share and parsing noise held fixed, and shuffling typicality within genre $\times$ budget decile gives $p=@$",
     lambda s: [*(lambda r: [r.coef_per_sd, r.lo, r.hi, r.n])(s.br_row("typicality_ols.csv", "+ script-quality controls", "script_typicality")),
                s.br["typicality"]["shuffle_p"]]),
    ("fig script", r"Grey band: 95\% of @ label shuffles", lambda s: [s.auc("Comedy", "dialogue", "w2v").perms]),
    # Results: people by genre (analysis/compare/network_by_genre.py)
    ("nbg across", "between films that share no genre (directors @, producers @, actors @, studios @), and pairs within a genre never differ",
     lambda s: [s.nbx("in_common_set", r, "different genre") for r in ("director", "producer", "actor", "studio")]
     if all(s.nbx("in_common_set", r, "different genre", "p") < .05 and s.nbx("in_common_set", r, "same minus different", "p") > .05
            for r in ("director", "producer", "actor", "studio")) else [None]),
    ("nbg horror", "films sharing a producer correlate at @ and films sharing a studio at @",
     lambda s: [s.nbx("in_common_set", "producer", "genre: Horror"), s.nbx("in_common_set", "studio", "genre: Horror")]),
    ("nbg wide", "On all @ films, producers (@ within a genre against @ across) and studios (@ against @) do earn alike more within a genre",
     lambda s: [6346, s.nbx("usable_money", "producer", "same genre"), s.nbx("usable_money", "producer", "different genre"),
                s.nbx("usable_money", "studio", "same genre"), s.nbx("usable_money", "studio", "different genre")]
     if all(s.nbx("usable_money", r, "same minus different", "p") < .05 for r in ("producer", "studio")) else [None]),
    ("communities", "finds @ groups among the @ linked films, with modularity @ against at most @ for the network rewired with every film's degree kept, and @ seeds agree with NMI @",
     lambda s: [s.com["communities"], s.com["films_in_network"], s.com["modularity"], s.com["rewired_modularity_max"],
                s.com["seeds"], s.com["mean_nmi_best_vs_other_seeds"]] if s.com["modularity_p"] < .05 else [None]),
    ("community methods", "Leiden (NMI @ with Louvain) and Infomap (@) find much the same split",
     lambda s: [s.com_methods.loc["Leiden", "nmi_with_louvain"], s.com_methods.loc["Infomap", "nmi_with_louvain"]]),
    ("community genre", "their NMI with each film's genre is @, above shuffled genres (@) but small",
     lambda s: [s.com["nmi_with_rarest_genre"], s.com["nmi_genre_null_95"]] if s.com["nmi_genre_p"] < .05 else [None]),
    ("community returns", r"only one of the @ communities with at least @ films differs from films of the same budget, a group with twice the usual share of biographies ($-@$)",
     lambda s: [s.com["communities_min_size"], s.com["min_size"],
                -s.com_ret[s.com_ret.p_holm < .05].iloc[0].excess] if (s.com_ret.p_holm < .05).sum() == 1
                and "Biography" in s.com_ret[s.com_ret.p_holm < .05].iloc[0].category else [None]),
    ("community horror", "The three groups that lean towards horror sit above their budgets (+@ to +@) but do not pass",
     lambda s: (lambda h: [h.excess.min(), h.excess.max()] if len(h) == 3 and (h.p_holm >= .05).all() else [None])(
                s.com_ret[s.com_ret.category.str.contains("Horror")])),
    ("community art house", r"with Film4 and BBC Film as its most frequent studios (@ films, median budget \$@M)",
     lambda s: (lambda r: [r.films, r.median_budget_musd] if r.p_holm < .05 and r.top_studios.startswith("Film4")
                and "BBC Film" in r.top_studios and (s.com_prof.p_holm < .05).sum() == 1 else [None])(
                s.com_prof[s.com_prof.p_holm < .05].iloc[0])),
    ("community horror houses", "form around Blumhouse (+@) and around New Line with Wes Craven (+@)",
     lambda s: (lambda t: [t.iloc[0].excess, t.iloc[1].excess] if t.iloc[0].top_studios.startswith("Blumhouse")
                and t.iloc[1].top_studios.startswith("New Line") and "Wes Craven" in t.iloc[1].top_directors
                and (t.iloc[:2].p_holm >= .05).all() else [None])(s.com_prof.sort_values("excess", ascending=False))),
    ("methods communities", "against @ rewirings that keep every film's degree", lambda s: [s.com["rewires"]]),
    # Results: head to head (analysis/compare/head_to_head.py)
    ("h2h setup", "On the same @ films we built three networks", lambda s: [s.h2h_sum["films"]]),
    ("h2h k", "each script linked to its @ nearest scripts", lambda s: [s.h2h_sum["k_script"]]),
    ("h2h shuffles", "against the same @ shuffles within era bins. Shared people", lambda s: [s.h2h_sum["shuffles"]]),
    ("h2h people", "Shared people give @ (null 95th percentile @), and @ between films of different genres",
     lambda s: [s.h2h.loc[("people", "all links"), "assortativity"], s.h2h.loc[("people", "all links"), "null_95"],
                s.h2h.loc[("people", "links with no shared genre"), "assortativity"]]),
    ("h2h script", r"Similar screenplays give @ (null @), and all of it comes from films of the same genre: @ for those links and $-@$ across genres ($p=@$)",
     lambda s: [s.h2h.loc[("script", "all links"), "assortativity"], s.h2h.loc[("script", "all links"), "null_95"],
                s.h2h.loc[("script", "links within a shared genre"), "assortativity"],
                -s.h2h.loc[("script", "links with no shared genre"), "assortativity"], s.h2h.loc[("script", "links with no shared genre"), "p"]]),
    ("h2h genre", "Shared genre gives @, which passes its null (@) but links almost every film to every other (mean degree @)",
     lambda s: [s.h2h.loc[("genre", "all links"), "assortativity"], s.h2h.loc[("genre", "all links"), "null_95"],
                s.h2h.loc[("genre", "all links"), "mean_degree"]] if s.h2h.loc[("genre", "all links"), "p"] < .05 else [None]),
    # Discussion
    ("missing flops", r"TMDB has figures for @ films Wikipedia lacks, with a mean return of $-@$ against @, and @\% of them grossed less than their budget against @\%",
     lambda s: [s.tmdb["money"]["films_wikipedia_lacks_with_clean_tmdb_money"], -s.tmdb["money"]["mean_ln_ratio_wikipedia_lacks"],
                s.tmdb["money"]["mean_ln_ratio_usable"], 100 * s.tmdb["money"]["share_below_budget_wikipedia_lacks"],
                100 * s.tmdb["money"]["share_below_budget_usable"]]),
    ("reception", r"at only $\rho\approx@$",
     lambda s: [{round(s.money(k).rho, 1) for k in ("imdb_rating", "metacritic", "rotten_tomatoes")}.pop()]),
    # Methods
    ("methods films", "came out between 1975 and 2025 (@ films)", lambda s: [s.genre["sets"]["usable_money"]["films"]]),
    ("methods common", "in both the dialogue and the description layer (@ films)", lambda s: [s.genre["sets"]["in_common_set"]["films"]]),
    ("methods genres", "IMDb genres with at least @ films (@ genres)",
     lambda s: [s.genre["min_films"], len(s.genre["sets"]["usable_money"]["genres_kept"])]),
    ("methods shuffles", "the mean return of its films, against @ shuffles within budget deciles", lambda s: [s.genre["perms"]]),
    ("methods tmdb", "(the two returns correlate at @)", lambda s: [s.tmdb["money"]["corr_ln_ratio_wikipedia_tmdb"]]),
    ("methods keywords", "uses the @ TMDB keywords in at least @ films, after removing production facts, places, decades and mood words, with @ curveball samples and Louvain over @ seeds",
     lambda s: [s.kwn["kept"], 30, s.kwn["null_samples"], s.kwn["louvain_seeds"]]),
    ("methods q2", "against @ shuffles within era bins that keep the graph", lambda s: [s.q2["shuffles"]]),
    ("manual check", "hand-checked @ elements from @ screenplays drawn by era and found @ placed in the wrong layer",
     lambda s: [s.manual.n.sum(), s.manual_films, s.manual.k.sum()]),
]

# Numbers that are design choices, not results. Each is allowed anywhere in the text.
CONSTANTS = {
    "02805": "course code",
    "10,000": "vote threshold of the 10k tier",
    "100,000": "vote threshold of the 100k tier",
    "1,000": "minimum budget and gross ($1,000) and shuffle counts",
    "95": "confidence level and null percentile",
    "100": "the 100k-vote tier",
    "500": "minimum tokens per screenplay layer",
    "20": "character-name threshold (films) in the Word2Vec preprocessing",
}


def normalise(tex):
    s = re.sub(r"(?<!\\)%.*", "", tex)  # comments
    s = re.sub(r"\\(cite|label|ref|includegraphics|doi|dates|graphicspath|documentclass|bibliography|articletype|"
               r"templatetype|thispagestyle)(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\(author|affil|bibsplit)\[[^\]]*\]", " ", s)
    return re.sub(r"\s+", " ", s)


@pytest.fixture(scope="module")
def body():
    return normalise(TEX.read_text())


def pattern(snippet):
    parts = [re.escape(p) for p in normalise(snippet).split("@")]
    return re.compile(NUM.join(parts))


def decimals(lit):
    return len(lit.split(".")[1]) if "." in lit else 0


@pytest.mark.parametrize("cid,snippet,fn", CLAIMS, ids=[c[0] for c in CLAIMS])
def test_claim_matches_output(body, cid, snippet, fn):
    m = pattern(snippet).search(body)
    assert m, f"[{cid}] snippet not found in main.tex; update CLAIMS if the sentence was reworded:\n  {snippet}"
    try:
        values = fn(S)
    except SourceMissing as e:
        pytest.skip(str(e))
    lits = m.groups()
    assert len(lits) == len(values), f"[{cid}] {len(lits)} numbers in the text, {len(values)} from the source"
    for lit, v in zip(lits, values):
        d = decimals(lit)
        got = round(float(v), d)
        want = float(lit.replace(",", ""))
        assert abs(got - want) < 1e-9, f"[{cid}] paper says {lit}, the output gives {float(v):.6g} (→ {got:.{d}f})"


def _tokens(s):
    """Numbers in the text, skipping digits that belong to a word (D1, SST-2, R^2 exponents are small anyway)."""
    s = s.replace("--", "  ")
    for m in re.finditer(NUM, s):
        i = m.start()
        if i and (s[i - 1].isdigit() or s[i - 1] == "."):
            continue
        j = i
        while j and s[j - 1].isalpha():
            j -= 1
        if j < i and (j == 0 or s[j - 1] != "\\"):
            continue  # part of a word such as D1
        yield m


def test_every_number_is_accounted_for(body):
    covered = set()
    for cid, snippet, _ in CLAIMS:
        m = pattern(snippet).search(body)
        if m:
            covered.update(range(m.start(), m.end()))
    loose = []
    for m in _tokens(body):
        lit = m.group(0).lstrip("-")
        if m.start() in covered or lit in CONSTANTS:
            continue
        if "," not in lit and "." not in lit and (int(lit) < 20 or 1900 <= int(lit) <= 2030):
            continue
        loose.append(f"{m.group(0)!r} in ...{body[max(0, m.start() - 50):m.end() + 30]}...")
    assert not loose, "numbers in main.tex with no source in CLAIMS:\n  " + "\n  ".join(loose)


def test_figure_inputs_match_outputs():
    """paper/make_figures.py hard-codes the R2 ladder (Fig. 1A) and the reception rows (Fig. 3)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_figures", ROOT / "paper" / "make_figures.py")
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    ladder = [v for _, v in mf.LADDER]
    want = S.nb_find(r"ridge R2: base (-?[\d.]+); base \+ network ([\d.]+); all pre-release ([\d.]+); "
                     r"everything incl\. reviews ([\d.]+)")
    assert ladder == want, f"Fig. 1A ladder {ladder} != notebook {want}"
    try:
        rows = {k: (rho, m, sd) for k, rho, m, sd in mf.RECEPTION}
        for label, key in (("IMDb rating", "imdb_rating"), ("Metacritic", "metacritic"), ("Rotten Tomatoes", "rotten_tomatoes")):
            r = S.money(key)
            assert rows[label] == (round(r.rho, 3), round(r.null_mean_strat, 3), round(r.null_std_strat, 3)), label
    except SourceMissing as e:
        pytest.skip(str(e))
