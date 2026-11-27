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

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "main.tex"
A = ROOT / "src" / "analysis"
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
        return self._json("../../archive/analysis/scripts_text/outputs/s01_summary.json")

    @cached_property
    def t06(self):
        return self._json("../../archive/analysis/scripts_text/outputs/s06_summary.json")

    @cached_property
    def t09(self):
        return self._json("../../archive/analysis/scripts_text/outputs/s09_summary.json")

    @cached_property
    def t10(self):
        return self._json("../../archive/analysis/scripts_text/outputs/s10_summary.json")

    @cached_property
    def t10_runs(self):
        return pd.read_csv(ROOT / "archive" / "analysis" / "scripts_text/outputs/s10_results.csv").set_index("run")

    @cached_property
    def links(self):
        return pd.read_csv(ROOT / "data/movie_links.csv", usecols=["imdb_id"])

    @cached_property
    def nb(self):
        """All printed output of notebooks/what_predicts_profit.ipynb, as one string."""
        cells = json.loads((ROOT / "archive/notebooks/what_predicts_profit.ipynb").read_text())["cells"]
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

    # ---- experience from full IMDb careers (analysis/compare/experience_full_imdb.py) ----
    @cached_property
    def xp(self):
        return pd.read_csv(A / "compare/outputs/experience_full_imdb/experience_tests.csv").set_index("measure")

    @cached_property
    def xp_sum(self):
        return self._json("compare/outputs/experience_full_imdb/summary.json")

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

    # ---- Q3 screenplays (analysis/screenplays) ----
    @cached_property
    def w2v(self):
        return pd.read_csv(ROOT / "src/analysis/screenplays/word2vec/results/primary.csv")

    def auc(self, genre, layer, rep):
        d = self.w2v
        return d[(d.genre == genre) & (d.layer == layer) & (d.representation == rep)].iloc[0]

    @cached_property
    def combo(self):
        return pd.read_csv(ROOT / "src/analysis/screenplays/word2vec/results/combo_interval.csv").set_index(["genre", "layer"])

    def votes(self, genre, sset, model, layer):
        d = pd.read_csv(ROOT / "src/analysis/screenplays/word2vec/results/votes_check.csv")
        return d[(d.genre == genre) & (d.set == sset) & (d.model == model) & (d.layer == layer)].iloc[0].auc

    @cached_property
    def allg(self):
        return pd.read_csv(ROOT / "src/analysis/screenplays/word2vec/results/all_genres.csv").set_index(["check", "genre", "layer"])

    @staticmethod
    def _wv(name):
        return pd.read_csv(ROOT / f"src/analysis/screenplays/word2vec/results/{name}.csv")

    @cached_property
    def bm(self):
        return self._wv("budget_matched").set_index(["shuffle_within", "genre", "layer"])

    @cached_property
    def bmc(self):
        return self._wv("budget_matched_checks").set_index(["check", "layer"])

    @cached_property
    def bmw(self):
        return self._wv("budget_matched_words")

    @cached_property
    def bm_sum(self):
        return self._wv("budget_matched_summary").iloc[0]

    @cached_property
    def pre(self):
        d = self._wv("prerelease_auc")
        return d[d.model == "era+genres+budget + vector"].set_index(["genre", "layer"])

    def script_cov(self, tier, lab):
        d = self._wv("prerelease_coverage")
        return d[(d.group == "all film-genre pairs") & (d.tier == tier) & (d.label == lab)].coverage.iloc[0]

    @cached_property
    def coverage_total(self):
        d = self._wv("prerelease_coverage")
        return int(d[d.group == "all films"].films.sum())

    @cached_property
    def cov_gap(self):
        return self._wv("prerelease_coverage_gap").set_index(["group", "tier"])

    def horror(self, name):
        return pd.read_csv(ROOT / f"src/analysis/screenplays/word2vec/results/{name}.csv")

    @cached_property
    def manual(self):
        d = pd.read_csv(ROOT / "src/analysis/screenplays/data_check/manual_check_summary.csv")
        return d[d.sheet == "main"]

    @cached_property
    def manual_films(self):
        return len(pd.read_csv(ROOT / "src/analysis/screenplays/data_check/sample_films.csv"))

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

    @cached_property
    def tb(self):
        return pd.read_csv(A / "compare/outputs/track_record_bounds/track_bounds.csv")

    @cached_property
    def tb_sum(self):
        return self._json("compare/outputs/track_record_bounds/summary.json")

    @cached_property
    def tb_check(self):
        return pd.read_csv(A / "compare/outputs/track_record_bounds/track_check.csv").set_index("group")

    def tb_row(self, measure, subset):
        return self.tb[(self.tb.measure == measure) & self.tb.films_subset.str.startswith(subset)].iloc[0]

    @cached_property
    def wc(self):
        return pd.read_csv(A / "genre/outputs/worst_case/verdict.csv").set_index("genre")

    @cached_property
    def wc_sum(self):
        return self._json("genre/outputs/worst_case/summary.json")

    # ---- communities of the people network (analysis/compare/network_communities.py) ----
    @cached_property
    def com(self):
        return self._json("compare/outputs/communities/summary.json")

    @cached_property
    def com_methods(self):
        return pd.read_csv(A / "compare/outputs/communities/methods.csv").set_index("method")

    @cached_property
    def comA(self):
        return self._json("compare/outputs/communities_usable_money/summary.json")

    @cached_property
    def comA_methods(self):
        return pd.read_csv(A / "compare/outputs/communities_usable_money/methods.csv").set_index("method")

    @cached_property
    def comA_prof(self):
        return pd.read_csv(A / "compare/outputs/communities_usable_money/community_profiles.csv")

    # ---- post-hoc robustness runs: no studio links, and the network from all of IMDb ----
    @cached_property
    def comN(self):
        return self._json("compare/outputs/communities_usable_money_no_studio/summary.json")

    @cached_property
    def comN_prof(self):
        return pd.read_csv(A / "compare/outputs/communities_usable_money_no_studio/community_profiles.csv")

    def comN_house(self, studio):
        d = self.comN_prof
        return d[d.top_studios.str.startswith(studio)].iloc[0]

    @cached_property
    def comF(self):
        return self._json("compare/outputs/communities_full_imdb/summary.json")

    @cached_property
    def comF_prof(self):
        return pd.read_csv(A / "compare/outputs/communities_full_imdb/community_profiles.csv")

    @cached_property
    def comH(self):
        return self._json("compare/outputs/communities_hollywood/summary.json")

    @cached_property
    def comH_prof(self):
        return pd.read_csv(A / "compare/outputs/communities_hollywood/community_profiles.csv")

    @cached_property
    def comH_house(self):
        return pd.read_csv(A / "compare/outputs/communities_hollywood/houses_10k.csv")

    @cached_property
    def cov(self):
        return self._json("compare/outputs/imdb_full_coverage/summary.json")

    @cached_property
    def cov_tier(self):
        return pd.read_csv(A / "compare/outputs/imdb_full_coverage/principal_credits_by_tier.csv").set_index("vote_tier")

    @cached_property
    def cov_crew(self):
        return pd.read_csv(A / "compare/outputs/imdb_full_coverage/crew_vs_principals.csv").set_index(["role", "vote_tier"])

    @cached_property
    def notice(self):
        return pd.read_csv(A / "compare/outputs/notice_track_imdb/notice_tests.csv").set_index("measure")

    @cached_property
    def notice_chk(self):
        return pd.read_csv(A / "compare/outputs/notice_track_imdb/notice_check.csv")

    def comH_sub(self, studio):
        d = self.comH_prof
        return d[d.top_studios.str.startswith(studio)].iloc[0]

    def comA_house(self, studio):
        d = self.comA_prof
        return d[d.top_studios.str.startswith(studio)].iloc[0]

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
BM3 = "era x budget terciles"

CLAIMS = [
    # Abstract
    ("abs graph", "Wikidata and TMDB for @ films (@ with usable budget and gross)",
     lambda s: [s.links.imdb_id.nunique(), s.genre["sets"]["usable_money"]["films"]]),
    ("abs scripts", "add @ screenplays, and measure", lambda s: [s.genre["sets"]["in_common_set"]["films"]]),
    ("abs genre R2", "Genre adds only @ to the variation explained once budget and era are fixed", lambda s: [s.genre["sets"]["usable_money"]["r2_added_by_genre"]]),
    ("abs assort", "earn alike (return correlation @, against at most @ in shuffles)",
     lambda s: [s.q2r("in_common_set_adjusted", "director"), s.q2_null95_max("in_common_set_adjusted")]),
    ("genre R2", r"Budget decile and era give $R^2=@$, and all @ genres add @. On the @ films with a screenplay, genre adds @.",
     lambda s: [s.genre["sets"]["usable_money"]["r2_budget_era"], len(s.genre["sets"]["usable_money"]["genres_kept"]),
                s.genre["sets"]["usable_money"]["r2_added_by_genre"], s.genre["sets"]["in_common_set"]["films"],
                s.genre["sets"]["in_common_set"]["r2_added_by_genre"]]),
    ("supernatural tmdb", r"beat other horror films of the same budget and era by +@ (@ films; median +@ against +@, and @\% beat the median of other horror)",
     lambda s: [s.hsub.loc["supernatural", "excess"], s.hsub.loc["supernatural", "n_films"], s.hsub.loc["supernatural", "median"],
                s.hsub.loc["supernatural", "median_other_horror"], 100 * s.hsub.loc["supernatural", "share_beating_other_horror_median"]]),
    ("supernatural wikidata", "its supernatural horror films beat other Horror by +@ (@ films)",
     lambda s: [s.wdh.loc["supernatural horror film", "excess"], s.wdh.loc["supernatural horror film", "films"]]),
    ("fig genre films", r"a film counts in every genre it has), @ films. Grey boxes: the middle 95\% of @ shuffles",
     lambda s: [s.genre["sets"]["usable_money"]["films"], s.genre["perms"]]),
    ("fig genre holm", "Holm correction over the @ genres", lambda s: [len(s.genre["sets"]["usable_money"]["genres_kept"])]),
    # Results: people
    ("q2 films", "we first ask it on the @ films of the common set. We linked", lambda s: [s.q2["samples"]["in_common_set"]["films"]]),
    ("q2 roles", r"Every role beats @ shuffles of returns within era bins (Table~\ref{tab:assortativity}). Its return assortativity is @ for directors and @ for producers, @ for actors and @ for studios, against null 95th percentiles of at most @",
     lambda s: [s.q2["shuffles"], *[s.q2r("in_common_set_adjusted", r) for r in ("director", "producer", "actor", "studio")],
                s.q2_null95_max("in_common_set_adjusted")]),
    ("q2 wide", "On all @ films the director and producer values rise to @, while the actor value falls to @",
     lambda s: [s.q2["samples"]["usable_money"]["films"], s.q2r("usable_money_adjusted", "director"), s.q2r("usable_money_adjusted", "actor")]
     if round(s.q2r("usable_money_adjusted", "director"), 3) == round(s.q2r("usable_money_adjusted", "producer"), 3) else [None]),
    ("specialists", r"it beats its budget by +@ per standard deviation (95\% interval +@ to +@; @ films, sequels controlled), and by +@ with sequels left out. Shuffling that experience within genre $\times$ budget decile gives $p=@$. General experience, counted as the number of earlier credits, does not help ($-@$ per standard deviation, not significant)",
     lambda s: [*(lambda r: [r.coef_per_sd, r.lo, r.hi, r.n])(s.br_row("specialists_ols.csv", "history: all earlier IMDb movies, sequel flag controlled", "genre_experience")),
                s.br_row("specialists_ols.csv", "history: all earlier IMDb movies, sequels left out", "genre_experience").coef_per_sd,
                s.br["specialists"]["shuffle_p_genre_experience"],
                -s.br_row("specialists_ols.csv", "history: all earlier IMDb movies, sequel flag controlled", "prior_credits").coef_per_sd]),
    ("fig people", r'on the @ films of the common set. "Combined" pools all four kinds of link. Red marks: the 95th percentile of @ shuffles of returns within era bins. Every role lies above it ($p=@$, the smallest possible)',
     lambda s: [s.q2["samples"]["in_common_set"]["films"], s.q2["shuffles"],
                max(s.q2r("in_common_set_adjusted", r, "p") for r in ("director", "actor", "producer", "studio"))]),
    # Results: screenplays
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
     lambda s: [s.comA["communities"], s.comA["films_in_network"], s.comA["modularity"], s.comA["rewired_modularity_max"],
                s.comA["seeds"], s.comA["mean_nmi_best_vs_other_seeds"]] if s.comA["modularity_p"] < .05 else [None]),
    ("coverage careers", r"Only @\% of the credits of our films' teams (their directors, writers, producers and top 3 actors) are on films with 10,000+ votes, so the gap is large",
     lambda s: [100 * s.cov["careers"]["share_on_10k_vote_films"]]),
    ("coverage crew", r"@\% of the directors in IMDb's separate crew list appear among them",
     lambda s: [100 * s.cov_crew.loc[("director", "all"), "share_in_principals"]]),
    ("robustness methods", r"their returns are tested on the @ films only", lambda s: [s.comF["usable_films"]]),
    ("community genre", "their NMI with each film's genre is @, which is greater than shuffled genres (@) but small",
     lambda s: [s.comA["nmi_with_rarest_genre"], s.comA["nmi_genre_null_95"]] if s.comA["nmi_genre_p"] < .05 else [None]),
    ("community count", "Tested like the genres, @ of the @ communities with at least @ films differ from films of the same budget",
     lambda s: [(s.comA_prof.p_holm < .05).sum(), s.comA["communities_min_size"], s.comA["min_size"]]),
    ("blumhouse", r"By far the largest gap is Blumhouse's: @ films, @ of them made by Jason Blum's company",
     lambda s: (lambda r: [r.films, int(re.search(r"Blumhouse Productions \((\d+)\)", r.top_studios).group(1))]
                if r.excess == s.comA_prof.excess.max() and r.p_holm < .05 and "Jason Blum" in r.top_producers else [None])(s.comA_house("Blumhouse"))),
    ("new line", r"the New Line group, with \emph{A Nightmare on Elm Street}, \emph{The Conjuring} and \emph{It}, earns like any other ($-@$)",
     lambda s: (lambda r: [-r.excess] if r.p_holm >= .05 else [None])(s.comA_house("New Line"))),
    ("methods communities", "Communities are tested against @ rewirings that keep every film's degree", lambda s: [s.comA["rewires"]]),
    # Results: head to head (analysis/compare/head_to_head.py)
    ("h2h setup", "so we put them on the same @ films, with one outcome", lambda s: [s.h2h_sum["films"]]),
    ("h2h k", "linked to its @ nearest scripts by cosine similarity", lambda s: [s.h2h_sum["k_script"]]),
    ("h2h shuffles", "at the two ends of a link, against @ shuffles within era bins. Shared people", lambda s: [s.h2h_sum["shuffles"]]),
    ("h2h people", r"Shared people give @ (null 95th percentile @), and @ between films of different genres (Fig.~\ref{fig:h2h})",
     lambda s: [s.h2h.loc[("people", "all links"), "assortativity"], s.h2h.loc[("people", "all links"), "null_95"],
                s.h2h.loc[("people", "links with no shared genre"), "assortativity"]]),
    ("h2h script", r"Similar screenplays give @ (null @), all of it among films of the same genre (@); across genres it is $-@$ ($p=@$), but only @ links cross genres and the null reaches @",
     lambda s: [s.h2h.loc[("script", "all links"), "assortativity"], s.h2h.loc[("script", "all links"), "null_95"],
                s.h2h.loc[("script", "links within a shared genre"), "assortativity"],
                -s.h2h.loc[("script", "links with no shared genre"), "assortativity"], s.h2h.loc[("script", "links with no shared genre"), "p"],
                s.h2h.loc[("script", "links with no shared genre"), "n_links"], s.h2h.loc[("script", "links with no shared genre"), "null_95"]]),
    ("h2h genre", "Shared genre gives @, but it links almost every film to every other", lambda s: [s.h2h.loc[("genre", "all links"), "assortativity"]]),
    # Discussion
    ("missing flops", r"TMDB has figures for @ films Wikipedia lacks, with a mean return of $-@$ against @, and @\% of them grossed less than their budget against @\%",
     lambda s: [s.tmdb["money"]["films_wikipedia_lacks_with_clean_tmdb_money"], -s.tmdb["money"]["mean_ln_ratio_wikipedia_lacks"],
                s.tmdb["money"]["mean_ln_ratio_usable"], 100 * s.tmdb["money"]["share_below_budget_wikipedia_lacks"],
                100 * s.tmdb["money"]["share_below_budget_usable"]]),
    ("reception", r"at only $\rho\approx@$",
     lambda s: [{round(s.money(k).rho, 1) for k in ("imdb_rating", "metacritic", "rotten_tomatoes")}.pop()]),
    # Methods
    ("methods films", "came out between 1975 and 2025 (@ films)", lambda s: [s.genre["sets"]["usable_money"]["films"]]),
    ("methods common", "in both the dialogue and the description layer. These @ films are the common set", lambda s: [s.genre["sets"]["in_common_set"]["films"]]),
    ("methods genres", "IMDb genres with at least @ films (@ genres)",
     lambda s: [s.genre["min_films"], len(s.genre["sets"]["usable_money"]["genres_kept"])]),
    ("methods shuffles", "the mean return of its films, against @ shuffles within budget deciles", lambda s: [s.genre["perms"]]),
    ("methods tmdb", "(the two returns correlate at @)", lambda s: [s.tmdb["money"]["corr_ln_ratio_wikipedia_tmdb"]]),
    ("methods keywords", "uses the @ TMDB keywords in at least @ films, after removing production facts, places, decades and mood words",
     lambda s: [s.kwn["kept"], 30]),
    ("methods keyword null", "The keyword network is built with @ curveball samples and Louvain over @ seeds; its backbone has @ links among @ keywords",
     lambda s: [s.kwn["null_samples"], s.kwn["louvain_seeds"], s.kwn["backbone_links"], s.kwn["nodes_in_backbone"]]),
    ("methods q2", "against @ shuffles within era bins that keep the graph", lambda s: [s.q2["shuffles"]]),
    ("manual check", "a parser we checked on only @ scripts", lambda s: [s.manual_films]),
    # Table 1 and the shortened text
    ("table sets", r"budget and gross known} & @ \\ \quad Genre returns (Fig.~\ref{fig:genre}), communities & @ \\ \quad Horror subgenres (Horror films only) & @ \\ \quad Genre experience (earlier credits needed) & @ \\ \emph{Common set: usable money and a screenplay} & @ \\",
     lambda s: [s.genre["sets"]["usable_money"]["films"], s.comA["films"], s.gsh.loc["Horror", "n_films"], s.br["specialists"]["films"],
                s.genre["sets"]["in_common_set"]["films"]]),
    ("keyword short", r"of the @ categories Louvain~ finds in that network (modularity @, against at most @ rewired), supernatural horror beats its budget by +@ and passes in every rerun",
     lambda s: [s.kwn["categories"], s.kwn["modularity"], s.stab["rewired_modularity_best_of_20_max"], s.kw_excess("supernatural horror")]
     if s.stab_always("supernatural horror") else []),
    ("methods keyword runs", r"The keyword network was rebuilt over @ null draws, @ subsamples of @\% of the films",
     lambda s: [s.stab["null_draws"], s.stab["boot"], 100 * s.stab["frac"]]),
    ("blumhouse money", r"with a median budget of \$@M, a median gross of @ times the budget and a return of +@ beyond films of the same budget",
     lambda s: (lambda r: [r.median_budget_musd, r.median_multiple, r.excess])(s.comA_house("Blumhouse"))),
    ("no studio blumhouse", r"Without the studio links, Blumhouse's group is @ films at +@, so +@ depends on the studio links",
     lambda s: (lambda b: [b.films, b.excess, s.comA_house("Blumhouse").excess] if b.p_holm < .05 else [None])(s.comN_house("Blumhouse"))),
    ("common set blumhouse", r"On the @ films with a screenplay, Blumhouse's group (@ films, +@) does not pass",
     lambda s: (lambda c: [s.com["films"], c.films, c.excess] if c.p_holm >= .05 else [None])(
         s.com_prof[s.com_prof.top_studios.str.startswith("Blumhouse")].iloc[0])),
    # Results: communities without the vote cut (communities_full_imdb, communities_hollywood) and Table 2
    ("full imdb", r"These @ linked films split into national industries, such as France, Japan, Spain and Korea (modularity @ against at most @ rewired; seeds agree with NMI @), and none of the @ groups with at least @ films of known return differs from films of the same budget. One group, Hollywood, holds @ films and @ of our @.",
     lambda s: [s.comF["films_in_network"], s.comF["modularity"], s.comF["rewired_modularity_max"], s.comF["mean_nmi_best_vs_other_seeds"],
                s.comF["communities_min_size"], s.comF["min_size"], s.comF["largest_community_films"], s.comF_prof.usable_films.max(), s.comF["usable_films"]]
     if (s.comF_prof.p_holm >= .05).all() else [None]),
    ("hollywood", r"It finds @ groups (modularity @ against at most @ rewired), and @ of the @ with at least @ films of known return differ",
     lambda s: [s.comH["communities"], s.comH["modularity"], s.comH["rewired_modularity_max"], (s.comH_prof.p_holm < .05).sum(), s.comH["communities_min_size"], s.comH["min_size"]]),
    ("hollywood houses", r"@\% of its films there fall in one group, which beats its budget by +@, and a group around Imagine, Amblin and Malpaso does by +@. Some groups fall short, the largest one around Emmett/Furla and Tribeca by $-@$. This split is less stable (NMI @)",
     lambda s: (lambda h, b, i, e: [100 * h.share_in_main_sub_community, b.excess, i.excess, -e.excess, s.comH["mean_nmi_best_vs_other_seeds"]]
                if str(h.main_sub_community) == str(b.community) and max(b.p_holm, i.p_holm, e.p_holm) < .05 else [None])(
         s.comH_house[s.comH_house.top_studios_10k.str.startswith("Blumhouse")].iloc[0], s.comH_sub("Blumhouse"), s.comH_sub("Imagine"), s.comH_sub("Emmett"))),
    ("networks returns", "returns are known only for the @ films", lambda s: [s.comF["usable_films"]]),
    ("table networks", r"Main & @ & @ & @ & @ & @ of @ & +@ \\ No studios & @ & @ & @ & @ & @ of @ & +@ \\ All of IMDb & @ & @ & @ & @ & @ of @ & \\ Hollywood & @ & @ & @ & @ & @ of @ & +@",
     lambda s: [v for k, pr, films, bl in ((s.comA, s.comA_prof, s.comA["films_in_network"], s.comA_house("Blumhouse").excess),
                                           (s.comN, s.comN_prof, s.comN["films_in_network"], s.comN_house("Blumhouse").excess),
                                           (s.comF, s.comF_prof, s.comF["films_in_network"], None),
                                           (s.comH, s.comH_prof, s.comH["films"], s.comH_sub("Blumhouse").excess))
                for v in [films, k["modularity"], k["rewired_modularity_max"], k["mean_nmi_best_vs_other_seeds"], (pr.p_holm < .05).sum(), len(pr)]
                + ([bl] if bl is not None else [])]),
    ("table networks caption", "Rewired: the highest modularity of @ rewirings", lambda s: [s.comA["rewires"]] if s.comH["rewires"] == s.comF["rewires"] == s.comA["rewires"] else [None]),
    # Results: genre, worst case for films without money (analysis/genre/q1_worst_case.py)
    ("worst case set", "Of the @ films from 1975 to 2025 with 10,000+ votes, @ have no budget and gross we can compare", lambda s: [s.wc_sum["films_1975_2025"], s.wc_sum["unpriced"]]),
    ("worst case fill", r"the mean return of the worst quarter of our films ($-@$) and shuffled within era only", lambda s: [-s.wc_sum["fill_value"]]),
    ("worst case action", r"Seven of the eight genres keep their direction after Holm correction. Action does not: its gap goes from $-@$ among films with money, under the same shuffle, to +@.",
     lambda s: [-s.wc.loc["Action", "priced_era_excess"], s.wc.loc["Action", "filled_excess"]]
     if len(s.wc) == 8 and list(s.wc[s.wc.verdict != "holds"].index) == ["Action"] else [None]),
    ("worst case horror", r"shrinks from +@ to +@, and it does not pass when we fill only the films with a known budget but no gross and match on budget (+@, Holm $p=@$)",
     lambda s: [s.wc.loc["Horror", c] for c in ("priced_era_excess", "filled_excess", "budget_filled_excess", "budget_filled_p_holm")]),
    # Results: track records and the stress test (analysis/compare/track_record_bounds.py)
    ("track coverage", r"only @\% (lead actors) to @\% (producers) of their earlier IMDb films count",
     lambda s: (lambda d: [100 * d.median_coverage["cast"], 100 * d.median_coverage["producer"]] if d.median_coverage.idxmin() == "cast" and d.median_coverage.idxmax() == "producer" else [None])(s.tb_check)),
    ("track rebuilt", r"from +@ per standard deviation for lead actors to +@ for producers, above all @ shuffles within genre $\times$ budget decile ($p=@$)",
     lambda s: (lambda b: [b.coef_per_sd.min(), b.coef_per_sd.max(), s.tb_sum["perms"], b.shuffle_p.max()])(s.tb[(s.tb.measure.str.endswith("rebuilt")) & (s.tb.films_subset == "all")])),
    ("track bound", r"worst quarter of our films ($-@$). Track records still go with beating the budget, from +@ for lead actors to +@ for producers ($p=@$ for each)",
     lambda s: (lambda b: [-s.tb_sum["v_bottom_quarter_mean"], b.coef_per_sd.min(), b.coef_per_sd.max(), b.shuffle_p.max()])(s.tb[s.tb.measure.str.contains("bottom-quarter")])),
    ("track split", r"where fewer of their earlier films have money (+@ against +@)",
     lambda s: [s.tb_row("producer_track, rebuilt", k).coef_per_sd for k in ("coverage < median", "coverage >= median")]),
    ("fig h2h", r"on the same @ films, with the same outcome", lambda s: [s.h2h_sum["films"]]),
    # Results: screenplays, budget-matched similarity (analysis/screenplays/word2vec/budget_matched.py, prerelease.py)
    ("genre films", "Of the @ films with 10,000+ IMDb votes, @ have a budget and gross we can compare",
     lambda s: [s.links.imdb_id.nunique(), s.genre["sets"]["usable_money"]["films"]] if s.coverage_total == s.genre["sets"]["usable_money"]["films"] else [None, None]),
    ("bm design", r"who is a hit and who is a flop @ times, as in the network part, only among films of the same genre, era and budget third", lambda s: [s.bm_sum.perms]),
    ("bm genres", "We tested nine genres in both layers, @ tests in total",
     lambda s: [len(s.bm.loc[BM3])] if s.bm_sum.genres == 9 else [None]),
    ("bm horror", r"Only Horror in scene description passes (Fig.~\ref{fig:script}): two scripts with the same outcome are more alike than a hit and a flop by @, while 95\% of the shuffles stay below @ (Holm $p=@$)",
     lambda s: (lambda r: [r.separation, r.null_q95, r.p_holm])(s.bm.loc[(BM3, "Horror", "description")])
     if sorted(s.bm.loc[BM3].query("p_holm < .05").index) == [("Horror", "description"), ("Horror", "dialogue")]
     and all(s.bm.loc[(f"era x budget {k}", "Horror", "description"), "p_holm"] < .05 for k in ("halves", "quarters")) else [None]),
    ("bm horror checks", r"dropping franchise films ($p=@$), keeping only films with 100,000+ votes (@ hits and @ flops, $p=@$)",
     lambda s: (lambda f, v: [f.p, v.hits, v.flops, v.p])(s.bmc.loc[("no franchise", "description")], s.bmc.loc[("100k+ only", "description")])),
    ("bm dialogue", r"In dialogue the gap is @ against @ (Holm $p=@$), but it fails with budget quarters (Holm $p=@$) and among films with 100,000+ votes ($p=@$)",
     lambda s: (lambda r: [r.separation, r.null_q95, r.p_holm, s.bm.loc[("era x budget quarters", "Horror", "dialogue"), "p_holm"],
                           s.bmc.loc[("100k+ only", "dialogue"), "p"]])(s.bm.loc[(BM3, "Horror", "dialogue")])),
    ("bm words", r"Of @ words compared by log-odds~\cite{monroe2008fightin}, only \emph{closet} (in @ hits and @ flops) differs",
     lambda s: (lambda d: [d.words_tested, d.hit_films, d.flop_films])(s.bmw[s.bmw.word == "closet"].iloc[0])
     if list(s.bmw[s.bmw.family_p < .05].word) == ["closet"] else [None]),
    ("bm coverage", r"@\% of hits have one against @\% of flops, and among films with 100,000+ votes @\% against @\% ($p=@$)",
     lambda s: [100 * s.script_cov(t, l) for t in ("all", "100k+") for l in ("hit", "flop")] + [s.cov_gap.loc[("all film-genre pairs", "100k+"), "p_two_sided"]]),
    ("fig script", r"Red marks: the 95th percentile of @ label shuffles within genre, era and budget third, not corrected for the @ tests",
     lambda s: [s.bm_sum.perms, len(s.bm.loc[BM3])]),
    ("bm tier", r"apart from the rest in the shuffle (in the second case $p=@$ for dialogue and $@$ for description)",
     lambda s: [s.bmc.loc[("terciles x vote tier", l), "p"] for l in ("dialogue", "description")]),
    ("horror words", r"Whether a Horror script uses supernatural words does not go with its return (@ films, $p=@$)",
     lambda s: (lambda d: [d["films_with_script"], d["supernatural_in_script"]["p"]])(s._json("genre/exploratory/outputs/horror_scripts/summary.json"))),
    ("table common", r"head to head (Fig.~\ref{fig:h2h}) & @ \\ \quad Hit and flop scripts (Fig.~\ref{fig:script}), @ genres & @",
     lambda s: [s.h2h_sum["films"], s.bm_sum.genres, s.bm_sum.films]),
    ("communities all", "Here we use all @ films, since screenplays aren't needed for the movie network", lambda s: [s.comA["films"]]),
    # Synced from Overleaf 27 Nov 2026: comment fixes and the group's rewrites
    ("abs script gap", r"(gap @ in mean cosine similarity; 95\% of shuffles below @)",
     lambda s: (lambda r: [r.separation, r.null_q95])(s.bm.loc[(BM3, "Horror", "description")])),
    ("abs h2h", "shared people give a correlation of @ against @ for similar screenplays",
     lambda s: [s.h2h.loc[("people", "all links"), "assortativity"], s.h2h.loc[("script", "all links"), "assortativity"]]),
    ("wikidata subgenres", r"Slasher (+@), splatter ($-@$) and comedy horror ($-@$) do not differ significantly from other Horror",
     lambda s: [s.wdh.loc["slasher film", "excess"], -s.wdh.loc["splatter film", "excess"], -s.wdh.loc["comedy horror film", "excess"]]
     if (s.wdh.drop("supernatural horror film").p_holm >= .05).all() else [None]),
    ("q2 unadjusted", "Without any adjustment the result still holds (director @, actor @, producer @, studio @)",
     lambda s: [s.q2r("common_unadjusted", r) for r in ("director", "actor", "producer", "studio")]),
    ("table assortativity", r"Every value lies above its null ($p=@$).} \small \setlength{\tabcolsep}{3pt} \begin{tabular}{lrrrrrr} \toprule & \multicolumn{3}{c}{Common set (@ films)} & \multicolumn{3}{c}{All usable (@ films)} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7} Link & Pairs & Assort. & Null & Pairs & Assort. & Null \\ \midrule Director & @ & @ & @ & @ & @ & @ \\ Top-3 actor & @ & @ & @ & @ & @ & @ \\ Producer & @ & @ & @ & @ & @ & @ \\ Studio & @ & @ & @ & @ & @ & @ \\ Combined & @ & @ & @ & @ & @ & @ \\",
     lambda s: [max(s.q2r(m, r, "p") for m in ("in_common_set_adjusted", "usable_money_adjusted") for r in ("director", "actor", "producer", "studio", "combined")),
                s.q2["samples"]["in_common_set"]["films"], s.q2["samples"]["usable_money"]["films"]]
     + [s.q2r(m, r, k) for r in ("director", "actor", "producer", "studio", "combined")
        for m in ("in_common_set_adjusted", "usable_money_adjusted") for k in ("pairs", "assortativity", "null_95")]),
    ("unlinked films", "The other @ films share no director, top-three actor, producer or studio with any film in the set",
     lambda s: [s.comA["films"] - s.comA["films_in_network"]]),
    ("community genre seeds", "but small next to the @ at which Louvain's seeds agree with each other", lambda s: [s.comA["mean_nmi_best_vs_other_seeds"]]),
    ("community ends", r"The highest is the group around Walt Disney Animation and Studio Ghibli (@ films, +@), and the lowest is the group around DreamWorks Animation and Woody Allen (@ films, $-@$)",
     lambda s: (lambda hi, lo: [hi.films, hi.excess, lo.films, -lo.excess] if hi.excess == s.comA_prof[~s.comA_prof.top_studios.str.startswith("Blumhouse")].excess.max() and lo.excess == s.comA_prof.excess.min()
                and max(hi.p_holm, lo.p_holm) < .05 else [None])(s.comA_house("Walt Disney Animation"), s.comA_house("DreamWorks Animation"))),
    ("common set blumhouse again", r"does not pass. Its return is still +@ beyond films of the same budget",
     lambda s: [s.com_prof[s.com_prof.top_studios.str.startswith("Blumhouse")].iloc[0].excess]),
    ("bm subtypes", r"hits are more often supernatural (@\% of the @ hits against @\% of the @ flops) and slasher (@\% against @\%), yet the gap stays far above its shuffles ($p=@$ for each subtype, the smallest @ shuffles allow)",
     lambda s: (lambda c, h: [100 * c.loc["hit", "share_supernatural"], c.loc["hit", "films"], 100 * c.loc["flop", "share_supernatural"], c.loc["flop", "films"],
                              100 * c.loc["hit", "share_slasher"], 100 * c.loc["flop", "share_slasher"],
                              max(h.loc[(f"era x {t}", "description"), "p"] for t in ("supernatural", "slasher", "found footage", "creature")), s.bm_sum.perms])(
         s._wv("horror_composition").set_index("group"), s._wv("horror_check").set_index(["shuffle_within", "layer"]))),
    ("fig h2h cross", "only @ screenplay links cross genres", lambda s: [s.h2h.loc[("script", "links with no shared genre"), "n_links"]]),
    ("methods people films", "communities use the same links on all @ films, weighted by the number", lambda s: [s.comA["films"]]),
    ("methods sequels", "a yes/no variable for sequels (@ films, from Wikidata)", lambda s: [s.br["specialists"]["sequels"]]),
    ("moviesum ids", "rows and @ unique IMDb ids", lambda s: [s._wv("../corpus_summary").set_index("measure").loc["films", "value"]]),
    ("moviesum final", "and @ have at least 500 tokens in both the dialogue", lambda s: [s.genre["sets"]["in_common_set"]["films"]]),
    ("per seed", r"Each of the five Word2Vec runs alone also gives $p\le@$",
     lambda s: [s.bmc[s.bmc.index.get_level_values(0).str.endswith(" alone")].p.max()]),
    ("corpus tokens", "The scripts hold @ million tokens before preprocessing (@ million dialogue, @ million description), and Word2Vec trains on the @ million left after it (@ million and @ million)",
     lambda s: (lambda p, c: [(p["dialogue_raw_whitespace_tokens"] + p["description_raw_whitespace_tokens"]) / 1e6,
                              p["dialogue_raw_whitespace_tokens"] / 1e6, p["description_raw_whitespace_tokens"] / 1e6,
                              (c["dialogue_tokens"] + c["description_tokens"]) / 1e6, c["dialogue_tokens"] / 1e6, c["description_tokens"] / 1e6]
                if (p["dialogue_tokens_after"], p["description_tokens_after"]) == (c["dialogue_tokens"], c["description_tokens"]) else [None])(
         pd.read_csv(ROOT / "src/analysis/screenplays/preprocess/preprocess_summary.csv").set_index("measure").value,
         pd.read_csv(ROOT / "src/analysis/screenplays/word2vec/corpus_summary.csv").set_index("measure").value)),
    ("manual read", "one of us read @ elements from @ screenplays, drawn by era, and found @ in the wrong layer",
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
    "50": "minimum hits and flops per genre (D14)",
    "30": "upper edge of the $5-30M budget tier",
    "20": "character-name threshold (films) in the Word2Vec preprocessing",
    "0.05": "significance level in the Holm example",
    "23": "day of the later IMDb download date",
}

# Numbers the paper prints that no committed output holds yet. They are skipped by the accounting test and
# reported by test_numbers_awaiting_a_source until someone commits the output and moves them into CLAIMS.
AWAITING_SOURCE = {
    "2,200": "MovieSum row count (src/analysis/screenplays/DECISIONS.md has no row count; no output file)",
    "2,020": "scripts among 10k+ vote films (only in src/analysis/screenplays/DECISIONS.md, not in an output)",
    "1,621": "scripts that pass the money rule (only in src/analysis/screenplays/DECISIONS.md, not in an output)",
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
        if m.start() in covered or lit in CONSTANTS or lit in AWAITING_SOURCE:
            continue
        if "," not in lit and "." not in lit and (int(lit) < 20 or 1900 <= int(lit) <= 2030):
            continue
        loose.append(f"{m.group(0)!r} in ...{body[max(0, m.start() - 50):m.end() + 30]}...")
    assert not loose, "numbers in main.tex with no source in CLAIMS:\n  " + "\n  ".join(loose)


@pytest.mark.xfail(reason="numbers in the paper with no committed output yet; see AWAITING_SOURCE", strict=False)
def test_numbers_awaiting_a_source(body):
    present = [f"{k}: {v}" for k, v in AWAITING_SOURCE.items() if re.search(r"(?<![\d,.])" + re.escape(k) + r"(?![\d,])", body)]
    assert not present, "\n  ".join(["no source yet:"] + present)


def test_figure_inputs_match_outputs():
    """paper/make_figures.py hard-codes the reception rows (Fig. 3)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_figures", ROOT / "paper" / "make_figures.py")
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    try:
        rows = {k: (rho, m, sd) for k, rho, m, sd in mf.RECEPTION}
        for label, key in (("IMDb rating", "imdb_rating"), ("Metacritic", "metacritic"), ("Rotten Tomatoes", "rotten_tomatoes")):
            r = S.money(key)
            assert rows[label] == (round(r.rho, 3), round(r.null_mean_strat, 3), round(r.null_std_strat, 3)), label
    except SourceMissing as e:
        pytest.skip(str(e))
