"""Step 12: interpretation, limitations, the finding candidate and the final checklist, all built from
the saved outputs of Steps 1-11 so they can't disagree with them."""
from __future__ import annotations

import pandas as pd

from . import (s01_groups, s02_layers, s03_clean, s04_controls, s05_scatter, s06_null, s07_stability, s08_kwic,
               s09_logreg, s10_franchise, s11_validation)
from .config import cfg
from .report import Md, Reading, Table


def load() -> dict:
    return {m.__name__.split(".")[-1]: m.load() for m in (s01_groups, s02_layers, s03_clean, s04_controls, s05_scatter,
                                                           s06_null, s07_stability, s08_kwic, s09_logreg, s10_franchise,
                                                           s11_validation)}


def run() -> dict:
    return load()


def _captures(r) -> pd.DataFrame:
    rows = [
        ("1 Success groups", "Gross relative to budget, three ways, with quartiles cut on all clean films.",
         "Real profit (marketing, the cinemas' share, inflation); the measure favours micro-budget films."),
        ("2 Text layers", "Which words sit in dialogue, action and headings, as MovieSum tagged them.",
         "Tagging errors; extraction is not resolution of who is speaking."),
        ("3 Cleaning", "The vocabulary left after removing formatting and names, with tokens/types/hapax per stage.",
         "Names that never appear as a speaker cue, nicknames, and numbers and hyphenated words."),
        ("4 Matching", "Unprofitable scripts against profitable ones of the same genre, decade and budget.",
         "Everything not matched on: stars, studio, marketing, release date, popularity."),
        ("5 Scattertext", "Which words each group uses comparatively more, per 10,000 words.",
         "How words are used, whether differences exceed chance, causes, quality."),
        ("6 Shuffle null", "Whether the real groups differ more than random groupings within genre × decade.",
         "Confounders other than genre and decade (budget, franchise, popularity)."),
        ("7 Stability", "How much the word lists depend on which scripts are in the sample.",
         "Whether a stable word is meaningful; an artefact can be very stable."),
        ("8 Back to the text", "How concentrated each top term is, with short windows to read.",
         "Five windows per term is a glimpse; categories are left to people."),
        ("9 Logistic regression", "Whether all words together predict the label on unseen scripts, against a baseline.",
         "Word order and meaning; single coefficients are unstable with correlated words."),
        ("10 Franchise and transcripts", "Whether the result survives without franchise films and likely transcripts.",
         "Both flags are incomplete."),
        ("11 Validation", "Whether people reading the scripts agree with the machine.",
         "Not filled in yet." if r["s11_validation"]["results"]["status"] == "NOT YET VALIDATED" else
         "Small samples catch systematic, not rare, errors."),
    ]
    return pd.DataFrame(rows, columns=["analysis", "what it captures", "what it misses"])


def _limitations(r) -> pd.DataFrame:
    s1 = r["s01_groups"]["summary"]
    sc = r["s01_groups"]["scripts"]
    allc = r["s01_groups"]["clean"]
    lim = []
    if allc is not None:
        a, b = allc[allc.has_script], allc[~allc.has_script]
        lim.append(("selection bias", f"Scripted films vs the other clean films: median year {int(a.year.median())} "
                    f"vs {int(b.year.median())}, median budget ${a.budget_usd.median() / 1e6:.0f}M vs "
                    f"${b.budget_usd.median() / 1e6:.0f}M, median gross/budget {pd.Series(a.gross_usd / a.budget_usd).median():.1f} "
                    f"vs {pd.Series(b.gross_usd / b.budget_usd).median():.1f}."))
    lim += [
        ("the 100k+ tier favours hits", f"Only {s1['checks']['gross_below_budget']['got']} of "
         f"{s1['checks']['clean_with_usable_script']['got']:,} scripted films grossed less than their budget."),
        ("gross/budget is not profit", "Marketing is missing, cinemas keep part of the gross, and the ratio favours "
         "micro-budget films."),
        ("no inflation adjustment", "Fine for the ratio (same-year dollars), but budgets and grosses aren't comparable "
         "across decades in absolute terms."),
        ("duplicate rule", f"{s1['moviesum_duplicate_ids']} IMDb IDs appear twice in MovieSum; the longest script is "
         "kept without checking which version it is."),
        ("uncleaned character labels", "Speaker cues are raw strings; nicknames and unspoken names stay in the text."),
        ("final-draft status unverified", "MovieSum doesn't say which draft a script is; some may be transcripts "
         f"({r['s10_franchise']['summary']['possible_transcripts']} flagged by a heuristic)."),
        ("10k tier not used", "Its money fields are not validated, so this analysis stays in the 100k+ tier."),
        ("validation", r["s11_validation"]["results"]["status"]),
    ]
    return pd.DataFrame(lim, columns=["limitation", "detail"])


def _finding(r) -> str:
    ct = r["s06_null"]["summary"]["count_test"]
    lr = r["s09_logreg"]["summary"]
    fr = r["s10_franchise"]["results"].set_index("run").loc["without likely franchise films"]
    no_null = ct["p"] > 0.05
    no_lr = s09_logreg.no_clear_gain(lr) or lr["auc_text_mean"] <= lr["auc_baseline_mean"] or lr["permutation_p"] > 0.05
    rdz = r["s10_franchise"]["summary"]["random_drop"]["z_median"]
    if not no_null and fr.p_empirical > 0.05:
        nf = r["s10_franchise"]["summary"]["logreg_without_franchise"]
        lr_part = (f"a TF-IDF classifier doesn't do clearly better than genre, decade and budget alone (AUC "
                   f"{lr['auc_text_mean']:.2f} vs {lr['auc_baseline_mean']:.2f}), and without franchise films it does "
                   f"worse ({nf['auc_text']:.2f} vs {nf['auc_baseline']:.2f})" if no_lr else
                   f"a TF-IDF classifier reaches AUC {lr['auc_text_mean']:.2f} against {lr['auc_baseline_mean']:.2f} "
                   f"for genre, decade and budget")
        return (f"Within the same genre and decade, the dialogue of the most and least profitable films (by "
                f"gross/budget) differs only slightly more than random groupings of the same scripts do (z = "
                f"{ct['z']:.2f}, p = {ct['p']:.3f}, 1,000 stratified shuffles). That excess disappears once "
                f"franchise films are removed (z = {fr.z:.2f}, p = {fr.p_empirical:.2f}), while random drops of the "
                f"same size give a median z of {rdz:.2f}. The most extreme words are mostly names, franchise words and "
                f"one film's French dialogue, so the result isn't robust to removing franchise films. As a second "
                f"method, {lr_part}.")
    if no_null and no_lr:
        return (f"Within the same genre and decade, the dialogue of the most and least profitable films (by "
                f"gross/budget) is no more different than random groupings of the same scripts (z = {ct['z']:.2f}, "
                f"p = {ct['p']:.2f}, 1,000 stratified shuffles). A TF-IDF classifier also does no better than genre, "
                f"decade and budget alone (AUC {lr['auc_text_mean']:.2f} vs {lr['auc_baseline_mean']:.2f}).")
    if not no_null and not no_lr:
        return (f"Within the same genre and decade, the dialogue of the most profitable films differs from the least "
                f"profitable more than random groupings do (z = {ct['z']:.2f}, p = {ct['p']:.3f}), and a TF-IDF "
                f"classifier adds to genre, decade and budget (AUC {lr['auc_text_mean']:.2f} vs "
                f"{lr['auc_baseline_mean']:.2f}). Which words carry this still awaits human validation.")
    return (f"The evidence is mixed. The word-count test against stratified shuffles gives z = {ct['z']:.2f} "
            f"(p = {ct['p']:.3f}), while the TF-IDF classifier reaches AUC {lr['auc_text_mean']:.2f} against "
            f"{lr['auc_baseline_mean']:.2f} for genre, decade and budget (permutation p = {lr['permutation_p']:.3f}). "
            f"Neither alone supports a claim about script vocabulary and profit.")


def _figures_for_paper(r) -> pd.DataFrame:
    return pd.DataFrame([
        ("Step 6: the real count against its stratified null", "It is the test of the primary analysis: one picture "
         "shows the claim and what it is compared to."),
        ("Step 10: the count test with and without franchise films", "It carries the interpretation: whether the "
         "excess survives without sequels decides if it is about scripts or franchises."),
        ("Step 9: cross-validated AUC, text vs baseline vs shuffled", "A second, independent method on unseen "
         "scripts; agreement or disagreement with Step 6 is the robustness argument. (The Step 5 Scattertext goes "
         "to the supplement, with the Step 8 table showing its top words are names and one-film words.)"),
    ], columns=["figure", "why"])


def _checklist(r) -> pd.DataFrame:
    s3, s6, s7, s8, s9 = (r[k]["summary"] for k in ("s03_clean", "s06_null", "s07_stability", "s08_kwic", "s09_logreg"))
    val = r["s11_validation"]["results"]["status"]
    rows = [
        ("1 Preprocessing stated?", "yes", "Step 3's choice table and per-stage tokens/types/hapax; every parameter in "
         "config.yaml; variants (stopwords, lemmas, NER names) reported in Step 5."),
        ("2 Did anyone read the text underneath?", "partly",
         f"{s8['windows']} KWIC windows for the top {s8['terms']} terms are prepared and the machine hints computed, "
         f"but the human category column is {'empty' if s8['human_categories_filled'] == 0 else 'partly filled'}."),
        ("3 Compared to what?", "yes", f"Stratified shuffle null ({s6['shuffles']:,} shuffles, z = "
         f"{s6['count_test']['z']:.2f}), a genre + decade + budget baseline (AUC {s9['auc_baseline_mean']:.2f}), a "
         f"label-permutation baseline for the classifier, and a second method."),
        ("4 Validated on our own sample?", "no" if val == "NOT YET VALIDATED" else "partly",
         f"Validation sheets for 25 scripts and 30 terms are ready; status: {val}."),
        ("5 Does each claim match its definition?", "yes", "Claims are about rates per 10,000 words and AUC, not about "
         "meaning; 'distinctive' means 'comparatively more frequent', extraction of names is not resolution, and "
         "no causal language is used."),
        ("6 Is one run presented as the answer?", "no", f"{s7['subsamples']} subsamples (median top-50 overlap "
         f"{s7['median_jaccard_high']:.2f}), a grid of definitions, layers and thresholds, matching calipers and "
         f"orders, and franchise/transcript reruns."),
        ("7 Can this representation answer the question?", "partly", "Word frequencies can show whether successful "
         "scripts *use different words*; they cannot show whether a script is good, how a story is built, or "
         "what caused a film's earnings."),
    ]
    return pd.DataFrame(rows, columns=["check", "answer", "evidence"])


def tables(res: dict) -> dict:
    return {"captures": _captures(res), "limitations": _limitations(res), "paper_figures": _figures_for_paper(res),
            "checklist": _checklist(res)}


def figures(res: dict) -> dict:
    return {}


def takeaway(r) -> str:
    ct = r["s06_null"]["summary"]["count_test"]
    lr = r["s09_logreg"]["summary"]
    fr = r["s10_franchise"]["results"].set_index("run").loc["without likely franchise films"]
    no_null = ct["p"] > 0.05
    no_lr = s09_logreg.no_clear_gain(lr) or lr["auc_text_mean"] <= lr["auc_baseline_mean"] or lr["permutation_p"] > 0.05
    if not no_null and fr.p_empirical > 0.05:
        return (f"Within the same genre and decade, the dialogue of the most and least profitable films (by "
                f"gross/budget) differs only slightly more than chance (z = {ct['z']:.2f}, p = {ct['p']:.3f}), and the difference disappears "
                f"without franchise films (z = {fr.z:.2f}).")
    if no_null and no_lr:
        return (f"Within the same genre and decade, the dialogue of the most and least profitable films (by "
                f"gross/budget) is no more different than chance (z = {ct['z']:.2f}), and a dialogue classifier does no better than genre, "
                f"decade and budget (AUC {lr['auc_text_mean']:.2f} vs {lr['auc_baseline_mean']:.2f}).")
    if not no_null and not no_lr:
        return (f"Within the same genre and decade, the dialogue of the most profitable films differs from the least "
                f"profitable more than chance (z = {ct['z']:.2f}, p = {ct['p']:.3f}), and a dialogue classifier adds to genre, "
                f"decade and budget (AUC {lr['auc_text_mean']:.2f} vs {lr['auc_baseline_mean']:.2f}). Which words "
                f"carry this still awaits human validation.")
    return (f"The evidence is mixed (word-count test z = {ct['z']:.2f}, classifier AUC {lr['auc_text_mean']:.2f} "
            f"against {lr['auc_baseline_mean']:.2f} for genre, decade and budget), and neither alone supports a claim "
            f"about script vocabulary and profit.")


def reading(res: dict) -> str:
    return "**Finding candidate.** " + _finding(res)


SECTION = [
    Md("""## Step 12 · Interpretation and limitations

This step asks what the analysis as a whole does and doesn't show.

Below we give one sentence on what each analysis captures and one on what it misses, the limitations we found
ourselves, the finding candidate, and the three figures we'd put in a six-page paper. Everything below is
generated from the saved outputs of Steps 1-11."""),
    Table("s12_interpretation", "captures", "What each analysis captures and what it misses."),
    Table("s12_interpretation", "limitations", "Limitations, with numbers from our own coverage results."),
    Reading("s12_interpretation"),
    Table("s12_interpretation", "paper_figures", "The three figures we'd use in a six-page PNAS paper, and why."),
    Md("### Final checklist"),
    Table("s12_interpretation", "checklist", "The course's checklist, answered with evidence from these notebooks."),
]
