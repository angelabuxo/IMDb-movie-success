"""Step 3: sentiment instruments. VADER on every sentence; contraction effect; distributions per layer;
stress tests; agreement with the contextual models once the HPC job has returned (ingest_contextual).
No outcome is used anywhere in this step."""
from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from analysis.scripts_text.report import HIGH, LOW, NEUTRAL, Fig, Md, Private, Reading, Run, Table, style

from . import s02_units
from .config import cfg, log_run
from .paths import OUTPUTS, PRIVATE

SUMMARY = OUTPUTS / "s03_summary.json"
DECISION = OUTPUTS / "s03_instrument_decision.json"   # written only once the fallback rule can be applied
STRESS_CTX = OUTPUTS / "s03_stress_contextual.csv"    # constructed sentences scored on HPC (no script text)
LOCAL_RUN = PRIVATE / "contextual_local"              # run of record: the same job on this Mac (MPS), LOG 2026-11-14
HPC_RUN = PRIVATE / "contextual"                      # HPC job outputs as fetched by the watcher: reproducibility check only
JOB = "local MPS run (HPC jobs 30053002 / 30053443 kept as a check)"


def _complete(d) -> bool:
    """A run counts only once every model has finished both passes."""
    m = d / "meta.json"
    if not m.exists():
        return False
    models = json.loads(m.read_text())["models"]
    return all(k in models and "scene_seconds" in models[k] for k in ("sst2", "roberta3")) or \
        any(not v.get("within_limit", True) for v in models.values())


CONTEXTUAL = LOCAL_RUN
# Amendment (d), robustness only. Two runs race (LOG 2026-11-14): the local MPS run and an HPC gpua10 copy. The run
# that FINISHED FIRST is used; the other is only a reproducibility check.
MODERNBERT_RUNS = {"local MPS": PRIVATE / "contextual_modernbert", "HPC gpua10": PRIVATE / "contextual_modernbert_hpc"}
THREE_CLASS = ("roberta3", "modernbert3")


def _finished_at(d) -> float | None:
    """Epoch seconds at which a ModernBERT run finished both passes, or None if it has not."""
    m = d / "meta.json"
    if not m.exists() or "scene_seconds" not in json.loads(m.read_text())["models"].get("modernbert3", {}):
        return None
    f = d / "finished_at.txt"                       # HPC: LSF finish time, written by the watcher
    return float(f.read_text()) if f.exists() else (d / "scene_modernbert3.csv.gz").stat().st_mtime


def modernbert_run() -> tuple[str | None, object]:
    done = {k: _finished_at(d) for k, d in MODERNBERT_RUNS.items()}
    done = {k: v for k, v in done.items() if v is not None}
    if not done:
        return None, None
    k = min(done, key=done.get)
    return k, MODERNBERT_RUNS[k]


def _source(key: str):
    return modernbert_run()[1] if key == "modernbert3" else CONTEXTUAL
KEYS = ["movie_name", "scene", "pos"]
LAYERS = ("both", "dialogue", "action")


def _sentence_path(inst: str):
    return PRIVATE / f"scores_{inst}.parquet"         # sentence-level scores; private (large, keyed to the text table)


def _scene_path(inst: str):
    return OUTPUTS / f"s03_scene_{inst}.csv.gz"       # scene means per layer; numbers only


# Constructed stress sentences (author-written, not from any script). "intent" is the author's construction
# goal, NOT a human validation label.
STRESS = [
    ("negation", "I love you.", "+"), ("negation", "I don't love you.", "-"),
    ("negation", "This is not bad.", "+ (mild)"), ("negation", "I'm not unhappy.", "+ (mild)"),
    ("negated violence", "I killed him.", "-"), ("negated violence", "I didn't kill him.", "0 / relief"),
    ("contrast and order", "The hero died, but saved everyone.", "mixed, ends +"),
    ("contrast and order", "The hero saved everyone, but died.", "mixed, ends -"),
    ("word order", "The dog bit the man.", "-"), ("word order", "The man bit the dog.", "- (same words)"),
    ("sarcasm", "Oh great, another flat tire.", "-"), ("sarcasm", "Yeah, I just love waiting in line all day.", "-"),
    ("neutral action line", "He opens the door and walks in.", "0"), ("neutral action line", "She looks at the clock.", "0"),
    ("violence without sentiment words", "Blood pools under the body.", "-"),
    ("violence without sentiment words", "He pulls the trigger.", "-"),
    ("shouting", "NO! NO! NO!", "-"),
]


def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def _label(v: np.ndarray, band: float) -> np.ndarray:
    return np.where(np.abs(v) < band, 0, np.sign(v)).astype(int)


def _split_inverse():
    """Exact inverse of the frozen join regex: "don't" -> "do n't" (the pre-tokenised form)."""
    pat = re.sub(r"\\s\+", "", cfg()["text"]["split_contraction_regex"])   # (\w)(n't|'s|...)\b
    return re.compile(pat, re.I)


# ---------------------------------------------------------------- run

def run() -> dict:
    c = cfg()
    band = c["instruments"]["lexicon"]["neutral_band"]
    s = s02_units.load_sentences()
    tab = s02_units.load()["table"]
    prim = set(tab.movie_name[tab.in_primary])
    pretok = set(s02_units.load()["report"].query("pretokenised").movie_name)
    an = _vader()

    t0 = time.perf_counter()
    s["vader"] = np.fromiter((an.polarity_scores(x)["compound"] for x in s.text), np.float32, len(s))
    secs = time.perf_counter() - t0
    s[["movie_name", "scene", "layer", "pos", "vader"]].to_parquet(_sentence_path("vader"), index=False)
    _scene_means(s, "vader").to_csv(_scene_path("vader"), index=False, compression="gzip")

    # Contraction effect: rescore every sentence holding a contraction in its split (pre-tokenised) form.
    inv, join = _split_inverse(), re.compile(c["text"]["split_contraction_regex"], re.I)
    roundtrip = {w: join.sub(r"\1\2", inv.sub(r"\1 \2", w)) == w for w in ("can't", "won't", "I'm sure she's done.")}
    assert all(roundtrip.values()), roundtrip
    has = s.text.str.contains(r"\w(?:n't|'s|'re|'ve|'ll|'d|'m)\b", case=False, regex=True)
    sub = s.loc[has, ["movie_name", "text", "vader"]].copy()
    sub["split"] = np.fromiter((an.polarity_scores(inv.sub(r"\1 \2", x))["compound"] for x in sub.text), np.float32, len(sub))

    def effect(d: pd.DataFrame, what: str) -> dict:
        diff = (d.split - d.vader).abs()
        return {"set": what, "sentences": int(len(d)), "score_changed": int((diff > 1e-6).sum()),
                "largest_change": float(diff.max()) if len(d) else 0.0,
                "crossed_neutral_band": int((_label(d.split.to_numpy(), band) != _label(d.vader.to_numpy(), band)).sum())}
    contraction = [effect(sub, "every sentence with a contraction"),
                   effect(sub[sub.movie_name.isin(pretok)], f"… in the {len(pretok)} pre-tokenised scripts")]
    # Which contraction forms sit in the sentences whose score moves (single tokens, no script text).
    form = re.compile(r"\b\w+(?:n't|'s|'re|'ve|'ll|'d|'m)\b", re.I)
    moved = sub[(sub.split - sub.vader).abs() > 1e-6]
    cnt_all = pd.Series([w.lower() for x in sub.text for w in form.findall(x)]).value_counts()
    cnt_mv = pd.Series([w.lower() for x in moved.text for w in form.findall(x)]).value_counts()
    forms = [{"form": w, "in moved sentences": int(n), "in all contraction sentences": int(cnt_all.get(w, 0))}
             for w, n in cnt_mv.head(12).items()]

    dist = []
    for sample, keep in (("primary 834", s.movie_name.isin(prim)), ("all 1,087", np.ones(len(s), bool))):
        for layer in LAYERS:
            v = s.vader[keep & (s.layer.eq(layer) if layer != "both" else True)].to_numpy()
            lab = _label(v, band)
            dist.append({"sample": sample, "layer": layer, "sentences": int(len(v)),
                         "negative": float((lab < 0).mean()), "neutral": float((lab == 0).mean()),
                         "positive": float((lab > 0).mean()), "exactly zero": float((v == 0).mean()),
                         "mean": float(v.mean()), "median |score| if non-neutral": float(np.median(np.abs(v[lab != 0])))})

    stress = [{"group": g, "sentence": x, "constructed intent (author)": i,
               "vader": an.polarity_scores(x)["compound"]} for g, x, i in STRESS]

    summary = {"sentences": int(len(s)), "sentences_primary": int(s.movie_name.isin(prim).sum()),
               "vader_seconds": round(secs, 1), "vader_version": "3.3.2", "neutral_band": band,
               "contraction_roundtrip": roundtrip, "contraction": contraction, "contraction_forms": forms, "distribution": dist, "stress": stress}
    summary["contextual"] = ingest_contextual(s, prim)
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    log_run("s03_instruments", [SUMMARY.name, _scene_path("vader").name],
            {"instruments": c["instruments"], "seed": c["seed"]})
    return load()


def _scene_means(s: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per scene: mean sentence score over both layers (all sentences, not the mean of layer means), and per layer."""
    both = s.groupby(["movie_name", "scene"], sort=False)[col].agg(["mean", "size"])
    both.columns = [f"{col}_both", "n_both"]
    lay = s.groupby(["movie_name", "scene", "layer"], sort=False)[col].agg(["mean", "size"]).unstack("layer")
    lay.columns = [f"{col}_{l}" if a == "mean" else f"n_{l}" for a, l in lay.columns]
    out = both.join(lay).reset_index()
    for l in ("dialogue", "action"):
        out[f"n_{l}"] = out[f"n_{l}"].fillna(0).astype(int)
    return out[["movie_name", "scene", f"{col}_both", "n_both", f"{col}_dialogue", "n_dialogue", f"{col}_action", "n_action"]]


# ---------------------------------------------------------------- contextual (HPC job)

def ingest_contextual(s: pd.DataFrame, prim: set) -> dict:
    """Bring in the HPC job's scores if they are present; apply the fallback rule; compute agreement.
    Returns {"status": "pending"} until the job's files are in private/contextual/."""
    meta_p = CONTEXTUAL / "meta.json"
    if not _complete(CONTEXTUAL):
        return {"status": "pending", "job": JOB}
    c = cfg()
    meta = json.loads(meta_p.read_text())
    limit = c["instruments"]["max_runtime_hours"]
    m = meta["models"].get("sst2", {})
    ok = bool(m.get("within_limit")) and (CONTEXTUAL / "sentence_sst2.csv.gz").exists()
    decision = {"date": time.strftime("%Y-%m-%d"), "job": JOB, "device": meta.get("device"),
                "hardware": "Apple M5 laptop GPU (MPS, fp16)" if meta.get("device") == "mps" else meta.get("gpu"),
                "contextual_model": c["instruments"]["contextual"]["name"],
                "available_after_user_permission": True,
                "projected_hours_sentence_all": m.get("projected_hours_sentence_all"), "limit_hours": limit,
                "within_limit": bool(m.get("within_limit")), "full_run_present": ok,
                "primary_instrument": "sst2" if ok else "vader",
                "reading": "availability read as met: the user allowed the download and the run completed on the "
                           "hardware named here (LOG.md, 2026-11-14 entries on amendment (b) and the local run)"}
    DECISION.write_text(json.dumps(decision, indent=1) + "\n")
    out = {"status": "done", "meta": meta, "decision": decision}
    sc = s[KEYS + ["layer", "words", "text", "vader"]]
    band = c["instruments"]["lexicon"]["neutral_band"]
    for key in ("sst2", "roberta3", "modernbert3"):
        p = _source(key) / f"sentence_{key}.csv.gz"
        if key == "modernbert3":
            which, _ = modernbert_run()
            if which is None:
                out[key] = "not finished yet"
                continue
            out["modernbert3_run"] = which
        if not p.exists():
            out[key] = "not run (over the runtime limit or missing)"
            continue
        d = pd.read_csv(p, dtype={"movie_name": str})
        assert len(d) == len(s) == 3_149_729, (len(d), len(s))
        sc = sc.merge(d, on=KEYS, how="left", validate="one_to_one", suffixes=("", f"_{key}"))
        assert (sc.layer == sc[f"layer_{key}"]).all(), f"layer mismatch for {key}"
        sc = sc.drop(columns=[f"layer_{key}"])
        sc[KEYS + ["layer", key]].to_parquet(_sentence_path(key), index=False)
        _scene_means(sc, key).to_csv(_scene_path(key), index=False, compression="gzip")
        ctx = _source(key) / f"scene_{key}.csv.gz"
        if ctx.exists():
            pd.read_csv(ctx, dtype={"movie_name": str}).to_csv(OUTPUTS / f"s03_scene_context_{key}.csv.gz",
                                                               index=False, compression="gzip")
    out["agreement"] = _agreement(sc, prim, band)
    out["matrix"] = _matrix(sc, prim)
    out["disagreements_available"] = "sst2" in sc
    if "sst2" in sc:   # private: kept on disk for the notebook's private cell only
        d = sc[(_label(sc.vader.to_numpy(), band) * np.sign(sc.sst2.to_numpy()) < 0) & sc.words.between(3, 15)]
        n = c["instruments"]["agreement"]["disagreements"]["n"]
        d.sample(min(n, len(d)), random_state=c["seed"])[["movie_name", "layer", "text", "vader", "sst2"]].to_parquet(
            PRIVATE / "s03_disagreements.parquet", index=False)
    return out


def _agreement(sc: pd.DataFrame, prim: set, band: float) -> list[dict]:
    rows = []
    vl = _label(sc.vader.to_numpy(), band)
    for k3 in THREE_CLASS:
        if k3 in sc:
            p_neu = sc[f"{k3}_neutral"].to_numpy()
            sp = sc[k3].to_numpy()
            probs = np.stack([(1 - p_neu - sp) / 2, p_neu, (1 - p_neu + sp) / 2], 1)   # neg, neu, pos
            sc = sc.assign(**{f"{k3}_label": probs.argmax(1) - 1})
    for sample, keep in (("primary 834", sc.movie_name.isin(prim).to_numpy()), ("all 1,087", np.ones(len(sc), bool))):
        nn = keep & (vl != 0)
        for key in ("sst2",) + THREE_CLASS:
            if key not in sc:
                continue
            x = sc[keep]
            scene = x.groupby(["movie_name", "scene"])[["vader", key]].mean()
            film = scene.groupby("movie_name").mean()
            r = {"sample": sample, "instrument": key,
                 "sign agreement on VADER-non-neutral sentences": float(
                     (np.sign(sc[key].to_numpy()[nn]) == vl[nn]).mean()) if key == "sst2" else float(
                     (sc[f"{key}_label"].to_numpy()[nn] == vl[nn]).mean()),
                 "Spearman, sentences": float(spearmanr(x.vader, x[key]).statistic),
                 "Spearman, scenes": float(spearmanr(scene.vader, scene[key]).statistic),
                 "Spearman, films": float(spearmanr(film.vader, film[key]).statistic)}
            if key in THREE_CLASS:
                r["Cohen's kappa, 3 classes"] = float(cohen_kappa_score(vl[keep], sc[f"{key}_label"].to_numpy()[keep]))
                r["neutral share"] = float((sc[f"{key}_label"].to_numpy()[keep] == 0).mean())
            rows.append(r)
    return rows


def _matrix(sc: pd.DataFrame, prim: set) -> dict:
    """Pairwise Spearman between every instrument that ran, primary 834: sentences, scenes, films."""
    inst = [k for k in ("vader", "sst2", "roberta3", "modernbert3") if k in sc]
    x = sc[sc.movie_name.isin(prim)]
    scene = x.groupby(["movie_name", "scene"])[inst].mean()
    film = scene.groupby("movie_name").mean()
    return {lvl: d.rank().corr().round(4).to_dict() for lvl, d in (("sentences", x[inst]), ("scenes", scene), ("films", film))}


def load() -> dict:
    return {"summary": json.loads(SUMMARY.read_text()), "scenes": load_scores("vader", "scene"),
            "table": s02_units.load()["table"]}


def load_scores(instrument: str = "vader", level: str = "scene") -> pd.DataFrame:
    """The one accessor later steps use. level = 'sentence' (private) or 'scene' (committed)."""
    if level == "sentence":
        return pd.read_parquet(_sentence_path(instrument))
    return pd.read_csv(_scene_path(instrument), dtype={"movie_name": str})


# ---------------------------------------------------------------- tables

PENDING = "NOT YET AVAILABLE: the contextual run has not finished"


def tables(res: dict) -> dict:
    s = res["summary"]
    ctx = s["contextual"]
    scoring = pd.DataFrame([
        ("instrument", f"VADER {s['vader_version']} compound score, one per sentence"),
        ("sentences scored (all 1,087 scripts)", f"{s['sentences']:,}"),
        ("… of which in the primary 834", f"{s['sentences_primary']:,}"),
        ("time on one laptop core", f"{s['vader_seconds']:.0f} s"),
        ("neutral band", f"|score| < {s['neutral_band']}"),
        ("contextual instruments", PENDING if ctx["status"] == "pending" else
         f"primary instrument by the fallback rule: {ctx['decision']['primary_instrument']}")], columns=["item", "value"])
    dist = pd.DataFrame(s["distribution"])
    for k in ("negative", "neutral", "positive", "exactly zero"):
        dist[k] = (100 * dist[k]).round(1).astype(str) + "%"
    dist[["mean", "median |score| if non-neutral"]] = dist[["mean", "median |score| if non-neutral"]].round(3)
    con = pd.DataFrame(s["contraction"]).round(3)
    forms = pd.DataFrame(s["contraction_forms"])
    st = pd.DataFrame(s["stress"])
    if STRESS_CTX.exists():
        st = st.merge(pd.read_csv(STRESS_CTX), on="sentence", how="left")
    else:
        st["sst2"] = st["roberta3"] = PENDING
    if ctx["status"] == "pending":
        agree = pd.DataFrame([{"comparison": "VADER vs sst2, VADER vs roberta3", "result": PENDING}])
        matrix = pd.DataFrame([{"result": PENDING}])
    else:
        agree = pd.DataFrame(ctx["agreement"]).round(3)
        matrix = pd.concat({lvl: pd.DataFrame(m) for lvl, m in ctx["matrix"].items()}, names=["level", "instrument"]
                           ).round(3).reset_index()
    sc = res["scenes"]
    prim = set(res["table"].movie_name[res["table"].in_primary])
    sc = sc[sc.movie_name.isin(prim)]
    rows = []
    for l in LAYERS:
        v = sc[f"vader_{l}"].dropna()
        rows.append({"layer": l, "scenes with a score": len(v), "exactly zero": f"{100 * (v == 0).mean():.1f}%",
                     **{f"q{int(q * 100)}": round(float(v.quantile(q)), 3) for q in (.05, .25, .5, .75, .95)},
                     "mean": round(float(v.mean()), 3)})
    film = sc.groupby("movie_name").vader_both.agg(["mean", "std"])
    films = film.describe(percentiles=[.1, .5, .9]).T.round(3)
    films.index = ["mean scene score per film", "SD of scene scores per film"]
    return {"scoring": scoring, "distribution": dist, "contraction": con, "contraction_forms": forms, "matrix": matrix, "stress": st.round(3), "agreement": agree,
            "scenes": pd.DataFrame(rows), "films": films.reset_index(names="per film (primary 834)")}


def private(res: dict) -> dict:
    """Script sentences, each cut to 15 words. Notebook only; outputs cleared before commit."""
    c = cfg()
    s = s02_units.load_sentences()
    v = load_scores("vader", "sentence")
    s = s.assign(vader=v.vader.to_numpy())
    prim = set(res["table"].movie_name[res["table"].in_primary])
    s = s[s.movie_name.isin(prim) & s.words.between(3, 15)]
    ext = pd.concat([s[s.vader >= 0.9].sample(5, random_state=c["seed"]), s[s.vader <= -0.9].sample(5, random_state=c["seed"])])
    out = {"extremes": ext[["movie_name", "layer", "text", "vader"]]}
    p = PRIVATE / "s03_disagreements.parquet"
    out["disagreements"] = pd.read_parquet(p) if p.exists() else pd.DataFrame([{"status": PENDING}])
    return out


# ---------------------------------------------------------------- figures

def _pending_fig(title: str) -> go.Figure:
    f = go.Figure()
    f.add_annotation(text=PENDING, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False)
    return style(f, title, height=240)


def figures(res: dict) -> dict:
    s = res["summary"]
    d = pd.DataFrame(s["distribution"]).query("sample == 'primary 834'")
    bars = go.Figure()
    for k, colour in (("negative", LOW), ("neutral", NEUTRAL), ("positive", HIGH)):
        bars.add_trace(go.Bar(x=d.layer, y=100 * d[k], name=k, marker_color=colour))
    bars.update_layout(barmode="stack", yaxis_title="% of sentences", xaxis_title="layer")
    style(bars, "Most sentences are neutral to VADER, action lines even more so", height=360)

    v = load_scores("vader", "sentence")
    tab = res["table"]
    v = v[v.movie_name.isin(set(tab.movie_name[tab.in_primary])) & (v.vader != 0)]
    hist = go.Figure()
    for layer, colour in (("dialogue", HIGH), ("action", LOW)):
        n, e = np.histogram(v.vader[v.layer == layer], bins=np.linspace(-1, 1, 81))
        hist.add_trace(go.Bar(x=(e[:-1] + e[1:]) / 2, y=100 * n / n.sum(), width=0.025, name=layer, opacity=0.6,
                              marker_color=colour))
    hist.update_layout(barmode="overlay", xaxis_title="VADER compound (sentences with a non-zero score)",
                       yaxis_title="% of those sentences")
    style(hist, "Non-zero sentence scores by layer", height=360)

    sc = res["scenes"]
    sc = sc[sc.movie_name.isin(set(tab.movie_name[tab.in_primary]))]
    n, e = np.histogram(sc.vader_both.dropna(), bins=np.linspace(-1, 1, 101))
    sh = go.Figure(go.Bar(x=(e[:-1] + e[1:]) / 2, y=n, width=0.02, marker_color=NEUTRAL))
    sh.update_layout(xaxis_title="scene score: mean VADER over the scene's sentences", yaxis_title="scenes",
                     showlegend=False)
    style(sh, "Scene scores pile up near zero", height=340)

    ctx = s["contextual"]
    if ctx["status"] == "pending" or not (OUTPUTS / "s03_scene_sst2.csv.gz").exists():
        ag = _pending_fig("Scene scores: VADER against the contextual model")
    else:
        m = sc.merge(load_scores("sst2", "scene"), on=["movie_name", "scene"])
        h, ex, ey = np.histogram2d(m.vader_both, m.sst2_both, bins=60, range=[[-1, 1], [-1, 1]])
        ag = go.Figure(go.Heatmap(x=(ex[:-1] + ex[1:]) / 2, y=(ey[:-1] + ey[1:]) / 2, z=np.log1p(h.T), colorscale="Greys",
                                  colorbar_title="log(1 + scenes)"))
        ag.update_layout(xaxis_title="scene score, VADER", yaxis_title="scene score, DistilBERT SST-2")
        style(ag, "Scene scores: VADER against the contextual model", height=420)
    return {"layers": bars, "nonzero": hist, "scenes": sh, "agreement": ag}


# ---------------------------------------------------------------- reading

def reading(res: dict) -> str:
    s = res["summary"]
    d = {(r["sample"], r["layer"]): r for r in s["distribution"]}
    p = lambda l, k: f"{100 * d[('primary 834', l)][k]:.0f}%"
    con = s["contraction"]
    st = {r["sentence"]: r["vader"] for r in s["stress"]}
    kill = st["I didn't kill him."]
    died, saved = st["The hero died, but saved everyone."], st["The hero saved everyone, but died."]
    top = ", ".join(f"\"{f['form']}\"" for f in s["contraction_forms"][:4])
    txt = (
        f"VADER scored all {s['sentences']:,} sentences in {s['vader_seconds']:.0f} seconds. In the primary sample "
        f"{p('both', 'neutral')} of sentences fall inside the neutral band: {p('dialogue', 'neutral')} of dialogue "
        f"and {p('action', 'neutral')} of action lines. Of the rest, dialogue leans positive "
        f"({p('dialogue', 'positive')} positive vs {p('dialogue', 'negative')} negative) and action lines lean "
        f"negative ({p('action', 'negative')} vs {p('action', 'positive')}). "
        f"Re-scoring every sentence that holds a contraction in its split, pre-tokenised form changed "
        f"{con[0]['score_changed']:,} of {con[0]['sentences']:,} scores "
        f"({100 * con[0]['score_changed'] / con[0]['sentences']:.1f}%; largest change {con[0]['largest_change']:.2f}; "
        f"{con[0]['crossed_neutral_band']:,} crossed the neutral band). The forms most often involved are {top}. "
        f"An extra token shifts VADER's three-word negation and booster windows, and a split possessive exposes a "
        f"lexicon word (\"God 's\"). So the contraction fix is not neutral for VADER after all. What it guarantees "
        f"is that pre-tokenised and ordinary scripts are scored the same way. "
        f"The stress tests show the blind spots of a word lexicon: \"I didn't kill him.\" scores {kill:+.2f}; "
        f"sarcasm reads as positive; violent action lines with no sentiment words score 0. VADER's \"but\" rule "
        f"does follow order (\"died, but saved everyone\" {died:+.2f}, \"saved everyone, but died\" {saved:+.2f}). "
    )
    if STRESS_CTX.exists():
        k = pd.read_csv(STRESS_CTX).set_index("sentence")
        neutral = ["He opens the door and walks in.", "She looks at the clock.", "He pulls the trigger."]
        kk = "I didn't kill him."
        forced = int((k.loc[neutral, "sst2"].abs() > 0.5).sum())
        txt += (f"On the same sentences the contextual models differ in telling ways. SST-2, which has no neutral class, "
                f"gives {forced} of the {len(neutral)} plain action lines a score beyond ±0.5 "
                f"(\"He opens the door and walks in.\" {k.loc[neutral[0], 'sst2']:+.2f}); the three-class RoBERTa calls "
                f"them mostly neutral (P(neutral) {k.loc[neutral, 'roberta3_neutral'].min():.2f} to "
                f"{k.loc[neutral, 'roberta3_neutral'].max():.2f}). \"I didn't kill him.\" is "
                f"{k.loc[kk, 'sst2']:+.2f} for SST-2 and "
                f"{k.loc[kk, 'roberta3']:+.2f} for RoBERTa. Both models read "
                f"\"Oh great, another flat tire.\" as negative, and both miss the sarcasm of loving to wait in line. "
                f"No instrument is right on every case; this is why Step 4 validates against human labels. ")
    ctx = s["contextual"]
    if ctx["status"] == "pending":
        txt += (f"Contextual scores for the scripts: NOT YET AVAILABLE (the contextual run has not finished). Until they are, "
                f"the fallback rule cannot be applied "
                f"and Step 7 does not run.")
    else:
        ag = {(a["sample"], a["instrument"]): a for a in ctx["agreement"]}
        s2, r3 = ag[("primary 834", "sst2")], ag[("primary 834", "roberta3")]
        txt += (f"On the scripts themselves the instruments agree only moderately. Against VADER, SST-2 has the same "
                f"sign on {100 * s2['sign agreement on VADER-non-neutral sentences']:.0f}% of the sentences VADER calls "
                f"non-neutral, and the Spearman correlation is {s2['Spearman, sentences']:.2f} for sentences, "
                f"{s2['Spearman, scenes']:.2f} for scenes and {s2['Spearman, films']:.2f} for films. The three-class "
                f"RoBERTa calls {100 * r3['neutral share']:.0f}% of sentences neutral (κ = "
                f"{r3["Cohen's kappa, 3 classes"]:.2f} with VADER's three labels), and correlates "
                f"{r3['Spearman, scenes']:.2f} with VADER for scenes and {r3['Spearman, films']:.2f} for films. "
                f"By the pre-registered fallback rule (projected {ctx['decision']['projected_hours_sentence_all']:.2f} h "
                f"≤ {ctx['decision']['limit_hours']:.0f} h) the primary instrument is DistilBERT SST-2 "
                f"({ctx['decision']['primary_instrument']}); VADER and RoBERTa are robustness instruments (Step 9).")
        if ("primary 834", "modernbert3") in ag:
            mb, mx = ag[("primary 834", "modernbert3")], ctx["matrix"]["films"]
            txt += (f" The ModernBERT-large model added later (amendment (d), robustness only) calls "
                    f"{100 * mb['neutral share']:.0f}% of sentences neutral (κ = {mb["Cohen's kappa, 3 classes"]:.2f} "
                    f"with VADER). At film level it correlates {mx['modernbert3']['sst2']:.2f} with SST-2, "
                    f"{mx['modernbert3']['roberta3']:.2f} with RoBERTa and {mx['modernbert3']['vader']:.2f} with VADER.")
    return txt


SECTION = [
    Md("""## Step 3 · Sentiment instruments

**What we do.** We score every sentence with VADER, a word lexicon with rules for negation, intensifiers,
capitals and "but". Each sentence gets a compound score in [-1, 1], and |score| < 0.05 counts as neutral. Two
contextual transformer models score the same sentences on DTU HPC:
- DistilBERT fine-tuned on SST-2, which is pre-registered;
- a three-class RoBERTa, which is robustness only.

A scene's score is the mean of its sentence scores (neutral sentences included). That is what the arcs in
Step 5 are built from.

**Why two kinds of instrument.**
- A lexicon is fast and transparent: every score can be traced to words. It does not understand order,
  sarcasm, or what an event means.
- A contextual model reads the whole sentence. It was trained on review text, not screenplays, though.

Where the two disagree, the arc depends on the instrument. Step 9 reports the analysis with each.

**What this step checks.**
1. How scores are distributed in dialogue and in action lines.
2. Whether the contraction fix from Step 2 changes any VADER score.
3. A small set of constructed stress sentences. The "intent" column is the author's construction goal, not a
   human validation; human validation is Step 4.
4. Agreement between the instruments, by rules fixed before the contextual scores were read.

No outcome is used in this step."""),
    Run("s03_instruments"),
    Table("s03_instruments", "scoring", "What was scored, how fast, and the state of the contextual run."),
    Table("s03_instruments", "distribution", "Share of negative, neutral and positive sentences per layer, primary "
          "834 first and all 1,087 scripts second."),
    Fig("s03_instruments", "layers", "Stacked shares per layer (primary 834). Action lines are mostly neutral to a "
        "lexicon: they describe events in plain words."),
    Fig("s03_instruments", "nonzero", "Distribution of non-zero sentence scores, dialogue against action, each as a "
        "percentage of its own layer."),
    Table("s03_instruments", "contraction", "Every sentence that contains a contraction, re-scored in its split "
          "('do n't') form: how many scores move. The second row is limited to the pre-tokenised scripts, where "
          "the split form actually occurred."),
    Table("s03_instruments", "stress", "Constructed stress sentences (not from any script). The contextual columns "
          "were scored on the HPC login node's CPU in fp32; the full job runs in fp16 on a GPU, so the last "
          "decimal can differ."),
    Table("s03_instruments", "contraction_forms", "The contraction forms that most often appear in sentences whose "
          "score moved, against how often they appear in all contraction sentences (single tokens only)."),
    Private("s03_instruments", "extremes", "ten real sentences at the extremes of VADER (≥ 0.9 or ≤ −0.9), 3 to 15 "
            "words (notebook only)."),
    Table("s03_instruments", "scenes", "Scene scores in the primary sample, per layer: quantiles and the share of "
          "scenes that score exactly zero."),
    Fig("s03_instruments", "scenes", "Scene scores (both layers) in the primary sample."),
    Table("s03_instruments", "films", "Per film: the mean scene score and the SD of scene scores. That SD is a raw "
          "version of D1 before smoothing; no outcome is shown."),
    Table("s03_instruments", "agreement", "Agreement between VADER and the contextual models, using the rules "
          "pre-registered in `instruments.agreement`."),
    Table("s03_instruments", "matrix", "Spearman correlation between every pair of instruments that ran, primary 834: "
          "per sentence, per scene (mean of its sentences) and per film (mean of its scenes)."),
    Fig("s03_instruments", "agreement", "Scene scores from VADER (x) against DistilBERT SST-2 (y), primary sample."),
    Private("s03_instruments", "disagreements", "20 sentences where VADER and SST-2 have opposite signs, drawn by the "
            "pre-registered rule (notebook only)."),
    Reading("s03_instruments"),
]
