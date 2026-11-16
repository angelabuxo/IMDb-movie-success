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
        return self._json("arcs/outputs/s01_summary.json")

    @cached_property
    def s05(self):
        return self._json("arcs/outputs/s05_summary_sst2.json")

    @cached_property
    def s07(self):
        return self._json("arcs/outputs/s07_primary.json")

    @cached_property
    def s08(self):
        return self._json("arcs/outputs/s08_prediction.json")

    @cached_property
    def x01(self):
        return pd.read_csv(A / "arcs/outputs/x01_del_vecchio.csv")

    def dv(self, instrument, shape="Man in a hole", spec="S1 their test: gross ~ dummy"):
        x = self.x01
        return x[(x.instrument == instrument) & (x.spec == spec) & (x["shape"] == shape)].iloc[0]

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

    def money(self, score):
        return self.rev_rho[("log_gross_over_budget", score)]


S = Sources()
G = "log_gross_over_budget"

# (id, snippet of main.tex with @ for each number, function returning the source values in the same order)
CLAIMS = [
    # Abstract
    ("abs graph", "one network of @ films", lambda s: [s.links.imdb_id.nunique()]),
    ("abs scripts", "add @ screenplays", lambda s: [s.s01["checks"]["clean_with_usable_script"]["got"]]),
    ("abs R2", r"explain nothing ($R^2=@$), and adding leakage-safe network track records raises $R^2$ to @",
     lambda s: s.nb_find(r"ridge R2: base (-?[\d.]+); base \+ network ([\d.]+)")),
    ("abs similarity", "shuffles allow (@--@ against at most @)",
     lambda s: [min(s.sim("adjusted")), max(s.sim("adjusted")), max(s.null95("adjusted"))]),
    ("abs arcs", r"null ($\rho=@$, $p=@$) or a near miss ($\rho=@$, $p=@$, threshold @)",
     lambda s: [s.s07["results"]["D1"]["rho"], s.s07["results"]["D1"]["p"], s.s07["results"]["D2"]["rho"],
                s.s07["results"]["D2"]["p"], s.s07["alpha"]]),
    ("abs reception", r"only moderately ($\rho\approx@$)",
     lambda s: [{round(s.money(k).rho, 1) for k in ("imdb_rating", "metacritic", "rotten_tomatoes")}.pop()]),
    # Results: network
    ("R2 base", r"out of sample ($R^2=@$;", lambda s: s.nb_find(r"ridge R2: base (-?[\d.]+);")),
    ("R2 ladder", r"raises $R^2$ to @. Release month and a franchise flag add little more (@). Critic and audience scores raise it to @",
     lambda s: s.nb_find(r"base \+ network ([\d.]+); all pre-release ([\d.]+); everything incl\. reviews ([\d.]+)")),
    ("studio track", r"studio's track record goes with the multiple at $\rho=+@$, net of budget and year (@ films), against $\rho=+@$ for the distributor's",
     lambda s: [*s.nb_find(r"studio track, net of budget and year\s+\d+\s+([\d.]+)"),
                *s.nb_find(r"studio track, net of budget and year\s+(\d+)"),
                *s.nb_find(r"distributor track, net of budget and year\s+\d+\s+([\d.]+)")]),
    ("director streak", r"a hit (top quarter) @\% of the time after a hit and @\% after a non-hit",
     lambda s: s.nb_find(r"hit after a hit (\d+)%, after a non-hit (\d+)%")),
    ("director shuffle", "about half that gap (@ of @)",
     lambda s: s.nb_find(r"gap \+([\d.]+) real vs \+([\d.]+) shuffled")[::-1]),
    ("pagerank", r"plain experience counts ($\rho=+@$)",
     lambda s: s.nb_find(r"Key people's PageRank net of experience: rho \+?(-?[\d.]+)")),
    ("sim shuffles", "residual returns than @ shuffles allow", lambda s: [s.roi["shuffles"]]),
    ("sim range", "from @ for the top-three cast to @ for producers, against a 95th null percentile of at most @",
     lambda s: [s.roi["results"]["adjusted"]["roles"]["cast"]["similarity"],
                s.roi["results"]["adjusted"]["roles"]["producer"]["similarity"], max(s.null95("adjusted"))]),
    ("sim robust", "without franchise films (@--@), with ranked returns (@--@) and from 1980 on (@--@)",
     lambda s: [f(s.sim(v)) for v in ("adjusted_no_franchise", "adjusted_rank", "adjusted_1980_on") for f in (min, max)]),
    ("sim within studio", "between @\\% and @\\% of each people role's similarity survives",
     lambda s: [100 * min(s.within_studio_kept()), 100 * max(s.within_studio_kept())]),
    ("fig1 caption", "1,000 shuffles that keep every group's size", lambda s: []),
    ("fig1 films", "among films sharing each role, @ films", lambda s: [s.roi["films"]]),
    # Results: screenplay text
    ("text scripts", "least profitable quarter of @ scripts", lambda s: [s.s01["checks"]["clean_with_usable_script"]["got"]]),
    ("text null", r"stratified shuffle null ($z=@$, $p=@$;", lambda s: [s.t06["count_test"]["z"], s.t06["count_test"]["p"]]),
    ("franchise", r"which uses @ shuffles per run, the same comparison gives $z=@$ with all films and $z=@$ without the @ likely franchise films",
     lambda s: [s.t10["null_shuffles"], s.t10_runs.loc["primary", "z"], s.t10_runs.loc["without likely franchise films", "z"],
                s.t10["likely_franchise"]]),
    ("random drop", r"never falls that low (@ of @ draws; median $z=@$)",
     lambda s: [round(s.t10["random_drop"]["share_z_at_or_below_franchise_drop"] * s.t10["random_drop"]["replicates"]),
                s.t10["random_drop"]["replicates"], s.t10["random_drop"]["z_median"]]),
    ("auc", "(AUC @ against @)", lambda s: [s.t09["auc_text_mean"], s.t09["auc_baseline_mean"]]),
    # Results: arcs
    ("arc sample", "every sentence of @ non-franchise scripts", lambda s: [s.s07["n"]]),
    ("arc points", "built a @-point arc per film", lambda s: [s.s05["params"]["time_points"]]),
    ("D1", r"($\rho=@$, 95\% CI $[@, @]$, $p=@$)",
     lambda s: [s.s07["results"]["D1"][k] for k in ("rho", "ci_low", "ci_high", "p")]),
    ("D2", r"near miss ($\rho=@$, $[@, @]$, $p=@$ against a Bonferroni threshold of @;",
     lambda s: [*(s.s07["results"]["D2"][k] for k in ("rho", "ci_low", "ci_high", "p")), s.s07["alpha"]]),
    ("arc prediction", r"($\Delta R^2=@$)", lambda s: [s.s08["gain"]]),
    ("reliability", "descriptors is @ for swing and @ for ending lift",
     lambda s: [r["Spearman-Brown"] for r in s.s05["reliability"] if r["descriptor"] in ("D1", "D2")]),
    ("del vecchio", r"$+\$@$M, $p=@$; SST-2: $+\$@$M, $p=@$",
     lambda s: [s.dv("vader").coef, s.dv("vader").p, s.dv("sst2").coef, s.dv("sst2").p]),
    ("del vecchio other", r"lower revenue for one shape ($p=@$)",
     lambda s: [s.x01[(s.x01.instrument == "vader") & s.x01.spec.str.startswith("S1")].p.min()]),
    ("fig2 caption", "all @ labelled films, @ drops of @ random films, and the drop of the @ likely franchise films",
     lambda s: [s.t10["labelled_films"], s.t10["random_drop"]["replicates"], s.t10["likely_franchise"], s.t10["likely_franchise"]]),
    ("fig2 shuffles", "(@ shuffles per run)", lambda s: [s.t10["null_shuffles"]]),
    ("fig2 arcs", r"with $\ln(\text{gross}/\text{budget})$, @ films", lambda s: [s.s07["n"]]),
    # Results: reception
    ("reception", r"the multiple correlates with the IMDb rating at $\rho=@$ (@ films), Metacritic at @ (@) and Rotten Tomatoes at @ (@)",
     lambda s: [s.money("imdb_rating").rho, s.money("imdb_rating").n, s.money("metacritic").rho, s.money("metacritic").n,
                s.money("rotten_tomatoes").rho, s.money("rotten_tomatoes").n]),
    ("reception null", "which itself sits near @", lambda s: [{round(s.money(k).null_mean_strat, 1) for k in
                                                               ("imdb_rating", "metacritic", "rotten_tomatoes")}.pop()]),
    ("critics agree", "(Metacritic with Rotten Tomatoes, @)", lambda s: [s.rev_rho[("metacritic", "rotten_tomatoes")].rho]),
    ("review tone", "tracks the multiple weakly (@)",
     lambda s: [float(re.search(r"VADER ↔ log\(g/b\) \| [^|]+ \| ([\d.]+)", s.rev_readme).group(1))]),
    ("loved flops", "there are @ critically loved flops and @ panned hits",
     lambda s: s.nb_find(r"(\d+) loved flops, (\d+) panned hits")),
    ("fig3 caption", "on films up to 2017 (@ to @ per measure;",
     lambda s: [f(s.money(k).n for k in ("imdb_rating", "metacritic", "rotten_tomatoes")) for f in (min, max)]),
    # Methods
    ("clean sample", "release before 2026 (@ films; @ with 100,000+ votes)",
     lambda s: [s.roi["films"], s.s01["checks"]["clean_movies"]["got"]]),
    ("methods scripts", "(@ clean films with 100,000+ votes)", lambda s: [s.s01["checks"]["clean_with_usable_script"]["got"]]),
    ("methods reviews", "(@ films, none after 2017)", lambda s: [s.rev_cov["n_100k_with_genome"]]),
    ("louvain", "(1980--2017), ten seeds", lambda s: []),
    ("arc smoothing", "resampled to @ points and smoothed over @",
     lambda s: [s.s05["params"]["time_points"], s.s05["params"]["smoothing_window"]]),
    ("arc tests", r"@ within-cell permutations, @ bootstrap resamples, Bonferroni $\alpha=@$",
     lambda s: [s.s07["permutations"], s.s07["bootstrap"], s.s07["alpha"]]),
    ("vader check", r"on @\% of @ non-neutral reviews",
     lambda s: [100 * s.rev_vader["accuracy_vs_stanford"], s.rev_vader["n_non_neutral"]]),
    # Discussion
    ("power", "powered for correlations of about @", lambda s: [s.s07["r_min"]["alpha_0.025"]]),
]

# Numbers that are design choices, not results. Each is allowed anywhere in the text.
CONSTANTS = {
    "02805": "course code",
    "10,000": "vote threshold of the 10k tier",
    "100,000": "vote threshold of the 100k tier",
    "1,000": "minimum budget and gross ($1,000) and shuffle counts",
    "95": "confidence level and null percentile",
    "1.96": "z threshold drawn in Fig. 2A",
    "100": "the 100k-vote tier",
    "50": "first 50 reviews per film; 50 hand-labelled reviews",
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
