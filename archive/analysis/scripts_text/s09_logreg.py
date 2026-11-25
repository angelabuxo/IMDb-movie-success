"""Step 9: a second method. Logistic regression on TF-IDF against a genre + decade + budget baseline."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import s01_groups, s05_scatter, s06_null
from . import textstats as ts
from .config import cfg, log_run
from .paths import OUTPUTS
from .report import ACCENT, HIGH, LOW, NEUTRAL, Fig, Md, Reading, Run, Table, style

CV = OUTPUTS / "s09_cv.csv"
PERM = OUTPUTS / "s09_permutations.csv"
COEF = OUTPUTS / "s09_coefficients.csv"
SUMMARY = OUTPUTS / "s09_summary.json"
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def tfidf(X_train, len_train, X_test, len_test, min_df):
    """Course definitions: tf = count / |d|, idf = ln(N / df), with N and df from the training documents
    only. Terms below min_df in the training fold are dropped."""
    df = np.asarray((X_train > 0).sum(axis=0)).ravel()
    keep = np.where(df >= min_df)[0]
    idf = np.log(X_train.shape[0] / df[keep])
    def f(X, lengths):
        tf = sp.diags(1.0 / np.maximum(lengths, 1)) @ X[:, keep].astype(float)
        return (tf @ sp.diags(idf)).tocsr()
    return f(X_train, len_train), f(X_test, len_test), keep


def _fit_text(Xtr, ytr, C):
    return LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(Xtr, ytr)


def _choose_C(Xtr, ytr, seed):
    """Inner 3-fold CV over the C grid, on the training fold only."""
    best, best_auc = None, -1
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    for C in cfg()["logreg"]["C_grid"]:
        aucs = []
        for a, b in inner.split(Xtr, ytr):
            m = _fit_text(Xtr[a], ytr[a], C)
            aucs.append(roc_auc_score(ytr[b], m.decision_function(Xtr[b])))
        if np.mean(aucs) > best_auc:
            best, best_auc = C, np.mean(aucs)
    return best


def baseline_matrix(meta: pd.DataFrame):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat = enc.fit_transform(meta[["primary_genre", "decade_bin"]])
    return np.column_stack([cat, meta.log_budget.to_numpy()])


def cv_once(X, lengths, y, B, strat, seed, C=None):
    """One repeat of stratified k-fold: AUC of the text model, the baseline and both together."""
    c = cfg()
    out = []
    folds = StratifiedKFold(c["logreg"]["folds"], shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(folds.split(X, strat)):
        Xtr, Xte, _ = tfidf(X[tr], lengths[tr], X[te], lengths[te], c["primary"]["min_doc_freq"])
        Ck = C if C is not None else _choose_C(Xtr, y[tr], seed)
        text = _fit_text(Xtr, y[tr], Ck)
        sc = StandardScaler().fit(B[tr])
        base = LogisticRegression(max_iter=3000).fit(sc.transform(B[tr]), y[tr])
        both = LogisticRegression(C=Ck, max_iter=3000, solver="liblinear").fit(
            sp.hstack([Xtr, sp.csr_matrix(sc.transform(B[tr]))]).tocsr(), y[tr])
        out.append({"seed": seed, "fold": k, "C": Ck,
                    "auc_text": roc_auc_score(y[te], text.decision_function(Xte)),
                    "auc_baseline": roc_auc_score(y[te], base.decision_function(sc.transform(B[te]))),
                    "auc_text_plus_baseline": roc_auc_score(y[te], both.decision_function(
                        sp.hstack([Xte, sp.csr_matrix(sc.transform(B[te]))]).tocsr()))})
    return out


def run() -> dict:
    c = cfg()
    X, vocab, ids, lengths, y, strata = s06_null.primary_inputs()
    meta = s01_groups.load()["scripts"].set_index("tconst").loc[ids]
    B = baseline_matrix(meta)
    strat = np.array([f"{g}|{l}" for g, l in zip(meta.primary_genre, y)])  # label x genre in every fold
    rows = []
    for r in range(c["logreg"]["repeats"]):
        rows += cv_once(X, lengths, y, B, strat, c["seed"] + r)
    cv = pd.DataFrame(rows)
    cv.to_csv(CV, index=False)
    C_fixed = float(cv.C.median())

    rng = np.random.default_rng(c["seed"] + 7)
    perm = []
    for i in range(c["logreg"]["permutations"]):
        yp = ts.stratified_permutation(y, strata, rng)
        res = cv_once(X, lengths, yp, B, np.array([f"{g}|{l}" for g, l in zip(meta.primary_genre, yp)]),
                      c["seed"], C=C_fixed)
        perm.append({"permutation": i + 1, "auc_text": float(np.mean([r_["auc_text"] for r_ in res]))})
        if (i + 1) % 20 == 0:
            print(f"label permutations: {i + 1}/{c['logreg']['permutations']}", flush=True)
    perm = pd.DataFrame(perm)
    perm.to_csv(PERM, index=False)

    Xall, _, keep = tfidf(X, lengths, X, lengths, c["primary"]["min_doc_freq"])
    model = _fit_text(Xall, y, C_fixed)
    coef = pd.DataFrame({"term": vocab[keep], "coefficient": model.coef_[0]}).sort_values("coefficient", ascending=False)
    coef.to_csv(COEF, index=False)

    n = c["scattertext"]["top_n"]
    st_hi, st_lo = ts.top_terms(s05_scatter.load()["primary"], n)
    co_hi, co_lo = set(coef.head(n).term), set(coef.tail(n).term)
    per_repeat = cv.groupby("seed")[["auc_text", "auc_baseline", "auc_text_plus_baseline"]].mean()
    real_text = float(cv.auc_text.mean())
    summary = {
        "documents": int(len(y)), "successful": int(y.sum()), "repeats": c["logreg"]["repeats"],
        "folds": c["logreg"]["folds"], "C_chosen_median": C_fixed,
        **{f"{k}_mean": float(cv[k].mean()) for k in ("auc_text", "auc_baseline", "auc_text_plus_baseline")},
        **{f"{k}_interval": [float(np.percentile(cv[k], 2.5)), float(np.percentile(cv[k], 97.5))]
           for k in ("auc_text", "auc_baseline", "auc_text_plus_baseline")},
        **{f"{k}_repeat_range": [float(per_repeat[k].min()), float(per_repeat[k].max())]
           for k in ("auc_text", "auc_baseline", "auc_text_plus_baseline")},
        "text_beats_baseline_in_folds": float((cv.auc_text > cv.auc_baseline).mean()),
        "permutation_mean": float(perm.auc_text.mean()), "permutation_std": float(perm.auc_text.std(ddof=1)),
        "permutation_p": float(ts.empirical_p((perm.auc_text >= real_text).sum(), len(perm))),
        "permutations": int(len(perm)),
        "jaccard_coef_vs_scattertext_high": ts.jaccard(co_hi, st_hi),
        "jaccard_coef_vs_scattertext_low": ts.jaccard(co_lo, st_lo),
        "shared_high": sorted(co_hi & st_hi), "shared_low": sorted(co_lo & st_lo),
    }
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    log_run("s09_logreg", [CV.name, PERM.name, COEF.name, SUMMARY.name],
            {"logreg": c["logreg"], "min_doc_freq": c["primary"]["min_doc_freq"], "C_fixed_for_permutations": C_fixed})
    return load()


def load() -> dict:
    return {"cv": pd.read_csv(CV), "perm": pd.read_csv(PERM), "coef": pd.read_csv(COEF),
            "summary": json.loads(SUMMARY.read_text())}


def tables(res: dict) -> dict:
    s = res["summary"]
    rows = [{"model": label, "mean AUC": round(s[f"{k}_mean"], 3),
             "95% of folds within": f"{s[f'{k}_interval'][0]:.2f}–{s[f'{k}_interval'][1]:.2f}",
             "range of the 10 repeat means": f"{s[f'{k}_repeat_range'][0]:.3f}–{s[f'{k}_repeat_range'][1]:.3f}"}
            for k, label in (("auc_text", "TF-IDF text only"), ("auc_baseline", "baseline: genre + decade + log budget"),
                             ("auc_text_plus_baseline", "text + baseline"))]
    rows.append({"model": "text only, labels shuffled (null)", "mean AUC": round(s["permutation_mean"], 3),
                 "95% of folds within": f"sd {s['permutation_std']:.3f} over {s['permutations']} shuffles",
                 "range of the 10 repeat means": "–"})
    co = res["coef"]
    n = 20
    coef = pd.DataFrame({"positive (successful)": co.head(n).term.values,
                         "coefficient": co.head(n).coefficient.round(2).values,
                         "negative (unsuccessful)": co.tail(n).term.values[::-1],
                         "coefficient ": co.tail(n).coefficient.round(2).values[::-1]})
    return {"auc": pd.DataFrame(rows), "coefficients": coef}


def figures(res: dict) -> dict:
    cv, perm, s = res["cv"], res["perm"], res["summary"]
    fig = go.Figure()
    for k, label, colour in (("auc_baseline", "baseline", NEUTRAL), ("auc_text", "text only", HIGH),
                             ("auc_text_plus_baseline", "text + baseline", ACCENT)):
        fig.add_trace(go.Box(y=cv[k], name=label, marker_color=colour, boxpoints="all", jitter=0.3, pointpos=0))
    fig.add_trace(go.Box(y=perm.auc_text, name="text, shuffled labels", marker_color=LOW, boxpoints=False))
    fig.add_hline(line_width=1.5, y=0.5, line_dash="dash", line_color="#444", annotation_text="chance", layer="above")
    fig.update_layout(yaxis_title="cross-validated AUC (each dot: one test fold)", showlegend=False)
    return {"auc": style(fig, "Can the dialogue predict the success label?", height=440)}


def no_clear_gain(s: dict) -> bool:
    """True when the text model's repeat-mean AUCs overlap the baseline's: no clear gain over genre, decade
    and budget."""
    t, b = s["auc_text_repeat_range"], s["auc_baseline_repeat_range"]
    return t[0] <= b[1] and b[0] <= t[1]


def compare_to_baseline(s: dict) -> str:
    if no_clear_gain(s):
        return ("Text and baseline are **not clearly different**, because the means of the 10 repeats overlap "
                f"({s['auc_text_repeat_range'][0]:.3f} to {s['auc_text_repeat_range'][1]:.3f} for text, "
                f"{s['auc_baseline_repeat_range'][0]:.3f} to {s['auc_baseline_repeat_range'][1]:.3f} for the baseline)")
    if s["auc_text_mean"] > s["auc_baseline_mean"]:
        return "The text model beats the baseline"
    return "The text model **doesn't beat** the baseline"


def takeaway(res: dict) -> str:
    s = res["summary"]
    if no_clear_gain(s):
        verdict = "Text and baseline aren't clearly different."
    elif s["auc_text_mean"] > s["auc_baseline_mean"]:
        verdict = "The text model beats the baseline."
    else:
        verdict = "The text model doesn't beat the baseline."
    return (f"A classifier that reads the dialogue reaches a mean AUC of {s['auc_text_mean']:.3f}, against "
            f"{s['auc_baseline_mean']:.3f} for genre, decade and budget alone (0.5 = chance). {verdict}")


def reading(res: dict) -> str:
    s = res["summary"]
    verdict = compare_to_baseline(s)
    return (f"On {s['documents']} scripts ({s['successful']} successful), {s['repeats']} repeats of {s['folds']}-fold "
            f"cross-validation give a mean AUC of {s['auc_text_mean']:.3f} for the TF-IDF text model, "
            f"{s['auc_baseline_mean']:.3f} for the genre + decade + budget baseline and "
            f"{s['auc_text_plus_baseline_mean']:.3f} for both together (0.5 = chance). With labels shuffled within "
            f"genre × decade cells, the text model averages {s['permutation_mean']:.3f} (sd "
            f"{s['permutation_std']:.3f}, {s['permutations']} shuffles), so the real text AUC has an empirical "
            f"p = {s['permutation_p']:.3f}. "
            + verdict + f". Fold by fold, the text model is ahead in {s['text_beats_baseline_in_folds']:.0%} of test "
            f"folds. Its "
            f"top-50 positive coefficients overlap the Scattertext top 50 with a Jaccard of "
            f"{s['jaccard_coef_vs_scattertext_high']:.2f} ({s['jaccard_coef_vs_scattertext_low']:.2f} on the "
            f"negative side).")


SECTION = [
    Md("""## Step 9 · A second method: logistic regression on TF-IDF

This step asks whether a model that uses all the words at once can tell successful from unsuccessful scripts
better than genre, decade and budget alone.

### What we do

We turn each script's dialogue into a TF-IDF vector, a set of word weights that are high for words the script
uses often and few other scripts use. We follow the course definitions: tf = count / |d| (the word's count
divided by the script's length) and idf = ln(N / df) (N scripts, df the number of scripts using the word),
computed on the training scripts only so the test scripts stay unseen. A logistic regression then predicts the
success label of definition (C) from that vector. We use cross-validation, which fits the model on part of the
scripts and tests it on the rest, rotating the held-out part. It's 5-fold, repeated 10 times, with folds
balanced on label × primary genre. We compare the model with a **baseline** that only knows genre, decade and
log budget, and with the text model trained on shuffled labels.

### Why

Scattertext looks at one word at a time. A classifier uses all the words together and is scored on scripts it
hasn't seen, so it answers a different question: is there *any* reliable signal in the dialogue? Agreement
between the two methods is evidence, and disagreement is a warning.

### What it measures

AUC, the chance that a randomly chosen successful script gets a higher score than a randomly chosen
unsuccessful one. 0.5 is a coin flip and 1 is perfect.

### How to read a coefficient

A positive coefficient means that, holding the other words fixed, more of this word goes with a higher
predicted chance of being in the successful group, in this corpus. It's an association, not a cause, and with
thousands of correlated words a single coefficient is unstable.

### What it misses

A linear model on word frequencies ignores order, meaning and context. We don't use scikit-learn's
`TfidfVectorizer`, because its idf adds 1 even without smoothing. We compute the course formula directly
instead."""),
    Run("s09_logreg"),
    Fig("s09_logreg", "auc", "Cross-validated AUC for each test fold: the baseline (genre, decade, budget), text "
        "only, both together, and the text model trained on shuffled labels. The dashed line is chance."),
    Table("s09_logreg", "auc", "Mean AUC, with the spread over folds and over the 10 repeats."),
    Table("s09_logreg", "coefficients", "The 20 most positive and most negative coefficients of the text model "
          "fitted on all scripts. They're associations in this corpus only."),
    Reading("s09_logreg"),
]
