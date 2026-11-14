"""Step 13: interpretation and limitations. Nothing is computed here: every number is read from the saved
outputs of Steps 1-12, so the notebook and the page say the same thing. Results that do not exist yet are
shown as PENDING, never filled in."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.scripts_text.report import Md, Reading, Table

from .config import cfg
from .paths import OUTPUTS

PENDING = "PENDING: not run yet"


def _j(name: str):
    p = OUTPUTS / name
    return json.loads(p.read_text()) if p.exists() else None


def load() -> dict:
    d = {k: _j(f) for k, f in {"s01": "s01_summary.json", "s03": "s03_summary.json", "decision": "s03_instrument_decision.json",
                                "s04": "s04_validation_results.json", "s07": "s07_primary.json", "s08": "s08_prediction.json",
                                "s10": "s10_meta.json", "s12": "s12_meta.json", "s02": "s02_summary.json"}.items()}
    inst = d["decision"]["primary_instrument"] if d["decision"] else "vader"
    d["instrument"] = inst
    d["s05"] = _j(f"s05_summary_{inst}.json")
    d["s06"] = _j(f"s06_summary_{inst}.json")
    d["s09"] = pd.read_csv(OUTPUTS / "s09_robustness.csv") if (OUTPUTS / "s09_robustness.csv").exists() else None
    d["s10_checks"] = pd.read_csv(OUTPUTS / "s10_checks.csv") if (OUTPUTS / "s10_checks.csv").exists() else None
    d["s10_drops"] = pd.read_csv(OUTPUTS / "s10_random_drops.csv") if (OUTPUTS / "s10_random_drops.csv").exists() else None
    s12 = OUTPUTS / "s12_back_to_text.csv"
    d["s12_filled"] = int((pd.read_csv(s12, dtype=str, keep_default_na=False)[["score_matches_text", "driver"]] != "")
                          .sum().sum()) if s12.exists() else 0
    return d


def _validated(d: dict) -> bool:
    return bool(d["s04"] and d["s04"]["status"] == "validated")


def _rel(d: dict, k: str) -> str:
    if not d["s05"]:
        return "?"
    return f"{next(r['Spearman-Brown'] for r in d['s05']['reliability'] if r['descriptor'] == k):.2f}"


def _shuffle(d: dict, k: str) -> str:
    if not d["s06"]:
        return PENDING
    r = next(a for a in d["s06"]["aggregate"] if a["sample"] == "primary 834" and a["descriptor"] == k)
    return f"{100 * r['share']:.0f}%"


def _primary(d: dict, k: str) -> str:
    if not d["s07"]:
        return PENDING
    r = d["s07"]["results"][k]
    return f"ρ = {r['rho']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}], p = {r['p']:.3f}"


def analyses(d: dict) -> pd.DataFrame:
    s08 = d["s08"]
    rows = [
        ("1 Sample and outcome", "Films with 100k+ IMDb votes, a public screenplay, and log(worldwide gross / budget).",
         "Films without a public script or with fewer votes; gross/budget is not profit."),
        ("2 Text units", "Scenes in order of appearance, split into dialogue and description sentences.",
         "Headings, speaker cues and parentheticals; whether the file's scene order is the story order."),
        ("3 Instruments", f"Sentence tone by VADER (a lexicon){' and contextual models' if d['decision'] else ''}.",
         "What a viewer feels; sarcasm, negated events (\"I didn't kill him.\"), and events told in neutral words."),
        ("4 Validation sheets", "A blind human check of 50 scenes and 25 scripts, with pre-registered scoring.",
         "Everything until the sheets are filled: " + ("filled." if _validated(d) else "NOT YET VALIDATED.")),
        ("5 Arcs", f"Tone over 100 points of story time; D1 swing and D2 ending minus start (reliability "
                   f"{_rel(d, 'D1')} and {_rel(d, 'D2')}).",
         "Within-scene order, music, image and performance; a share of about 1 minus the reliability of the between-film "
         "variation in D1 and D2 is scene-level noise."),
        ("6 Scene-order null", f"Whether the order of scenes carries tone structure (D1 beyond null: {_shuffle(d, 'D1')} "
                               f"of scripts; D2: {_shuffle(d, 'D2')}).",
         "Differences between films; it compares each script only with itself."),
        ("7 Primary analysis", f"Association with profit after genre × decade, budget and scene length. D1: {_primary(d, 'D1')}; "
                               f"D2: {_primary(d, 'D2')}.",
         "Causation, unmeasured confounders (marketing, stars, release date), and associations below r_min."),
        ("8 Prediction", PENDING if not s08 else f"Out-of-sample gain in R² from the arc: {s08['gain']:+.4f} "
                                                 f"[{s08['gain_range'][0]:+.4f}, {s08['gain_range'][1]:+.4f}].",
         "Non-linear relations a linear model cannot use."),
        ("9 Robustness", PENDING if d["s09"] is None else f"How the Step 7 statistic moves across {d['s09'].variant.nunique() - 1} "
                                                         "other analysis choices.",
         "Choices not varied: other lexicons, other outcomes, other time units."),
        ("10 Franchise checks", PENDING if d["s10_checks"] is None else "Whether dropping franchise films differs from "
                                                                         "dropping the same number of random films.",
         "Franchises missed by the noisy Wikidata flag."),
        ("11 Reception", "Not run: whether the review scores use the same instrument is still an open question for the group.",
         "Audience tone entirely, until it runs."),
        ("12 Back to the text", "The scenes and sentences behind the most extreme arcs, for a human to read.",
         "Everything until a person has filled the sheet" + (": filled." if d["s12_filled"] else ": NOT YET VALIDATED.")),
    ]
    return pd.DataFrame(rows, columns=["analysis", "what it captures", "what it misses"])


def limitations(d: dict) -> pd.DataFrame:
    s1 = d["s01"]
    eb, ep = s1["earlier_groups_after_exclusion"], s1["earlier_groups_primary"]
    rows = [
        ("Hit-skewed sample", f"Scripted films lean towards hits (earlier quartile groups B {eb['B']['high']} top vs "
                              f"{eb['B']['low']} bottom; C {eb['C']['high']} vs {eb['C']['low']}). In the primary "
                              f"sample this mostly disappears (B {ep['B']['high']} vs {ep['B']['low']}, C {ep['C']['high']} "
                              f"vs {ep['C']['low']}), but the 100k-vote tier and public scripts are still not a random draw."),
        ("Outcome", "gross / budget is not profit: no marketing costs, exhibitors keep part of the gross, no inflation "
                    "adjustment. It favours micro-budget horror."),
        ("Franchise flag", f"Noisy (Wikidata 'follows' {s1['franchise']['follows']}, named franchise "
                           f"{s1['franchise']['named']}). Excluding franchise films from the primary sample was "
                           "informed by the earlier branch's result."),
        ("Timeline", f"The scene order in the file is taken as the story order. Only {d['s02']['numbered_scenes']} "
                     "scripts number their scenes, so this cannot be checked."),
        ("Text quality", "Pre-tokenised scripts (joined before scoring) and possible transcripts (flagged, dropped in Step 10)."),
        ("Script tone is not viewer emotion", "A score of a written line, not of what an audience feels watching it."),
        ("Instruments", "VADER and the contextual models disagree on constructed sentences (Step 3)."
                        + ("" if d["decision"] else " The contextual run has not returned, so only VADER is in use.")),
        ("Validation", "Filled." if _validated(d) else "NOT YET VALIDATED: the human sheets are empty."),
        ("Reviews", f"Tag Genome stops at {s1['latest_year_with_reviews']} and has no ratings or dates (Step 11)."),
        ("Multiple testing", "The primary family (D1, D2, Bonferroni α = 0.025) was fixed in the config before any "
                             "association. Everything else is exploratory, with forking-paths risk."),
        ("Data not used", "The 10k-vote tier: its money fields are not validated."),
        ("Reliability", f"Split-half reliability of D1 {_rel(d, 'D1')} and D2 {_rel(d, 'D2')} caps any correlation "
                        "with profit."),
    ]
    return pd.DataFrame(rows, columns=["limitation", "what it means here"])


def figures_for_paper(d: dict) -> pd.DataFrame:
    return pd.DataFrame([
        ("1 Real vs shuffled arcs (Step 6, z histogram)", "Shows the arc exists as an ordered object before asking "
         "whether it relates to profit. Without it, a null in Step 7 could just mean the descriptors are noise."),
        ("2 Primary association against its permutation null, with r_min (Step 7)", "The one pre-registered test, "
         "shown against what chance gives under the same cell structure, with the smallest detectable effect marked."),
        ("3 Robustness forest plot (Step 9)", "Shows the answer does not hinge on one instrument, layer, window, "
         "sample or seed. That is the main defence against forking paths."),
    ], columns=["figure", "why"])


def finding(d: dict) -> str:
    if not d["s07"]:
        return f"**Finding: {PENDING}.** The primary analysis (Step 7) waits for the instrument decision."
    r = d["s07"]["results"]
    rm = d["s07"]["r_min"]["alpha_0.025"]
    sig = [k for k in ("D1", "D2") if r[k]["significant"]]
    order = _shuffle(d, "D1")
    pred = d["s08"]
    tail = (" Adding the arc to a baseline model " + ("does" if pred and pred["adds_predictive_value"] else "does not")
            + " improve out-of-sample prediction." if pred else "")
    if not sig:
        return (f"**Finding (negative):** screenplays have an emotional order ({order} of scripts are more volatile "
                f"than shuffled versions of themselves), but once genre, decade, budget and scene length are held fixed, "
                f"neither their volatility ({_primary(d, 'D1')}) nor their ending relative to their start "
                f"({_primary(d, 'D2')}) is associated with log(gross/budget) in {d['s07']['n']} films; associations "
                f"stronger than |ρ| ≈ {rm:.2f} are unlikely.{tail} The sentiment scores are "
                + ("validated." if _validated(d) else "NOT YET VALIDATED."))
    k = sig[0]
    name = {"D1": "volatility", "D2": "ending relative to the start"}[k]
    return (f"**Finding:** once genre, decade, budget and scene length are held fixed, a screenplay's {name} is "
            f"associated with log(gross/budget) ({_primary(d, k)}, N = {d['s07']['n']}). This is an association, "
            f"not an effect of the script.{tail} The sentiment scores are "
            + ("validated." if _validated(d) else "NOT YET VALIDATED."))


def tables(res: dict) -> dict:
    return {"analyses": analyses(res), "limitations": limitations(res), "paper_figures": figures_for_paper(res)}


def figures(res: dict) -> dict:
    return {}


def reading(res: dict) -> str:
    return finding(res)


SECTION = [
    Md("""## Step 13 · Interpretation and limitations

For every analysis: one line on what it captures and one on what it misses. Then our own limitations, the
finding in one sentence, and the three figures we would put in a six-page paper. Every number here is read
from the saved outputs of the earlier steps; anything not yet run says PENDING."""),
    Table("s13_interpretation", "analyses", "Each analysis: what it captures and what it misses."),
    Table("s13_interpretation", "limitations", "Limitations of this study."),
    Reading("s13_interpretation"),
    Table("s13_interpretation", "paper_figures", "The three figures for a six-page paper, and why each."),
]
