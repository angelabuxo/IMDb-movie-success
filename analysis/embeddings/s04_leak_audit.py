"""Step 4: do names or titles slip through the cleaning into the words that drive the results?

    python -m analysis.embeddings.s04_leak_audit

For each corpus it takes the top TOP words at both ends of the out-of-fold "more profit than the controls
predict" score (the same comparison as s03's word lists, but longer) and flags each word that could be
an identity leak rather than content:

- proper: written with a capital letter in at least PROPER_SHARE of its uses in the cleaned text
  (a name or a title that survived; sentence-initial capitals keep ordinary words well below this);
- genome_name: a word of a person's name or of a film title in the Tag Genome's 84,661 films;
- few_films: used by fewer than FEW_FILMS films, so it can only come from one film or one franchise.

A word flagged "proper" is read by hand below; the rest are context. Writes outputs/s04_leaks.csv (single
words and counts only) and outputs/s04_summary.json.
"""
from __future__ import annotations

import io
import json
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from . import clean as C
from .paths import GENOME, OUTPUTS, PRIVATE, cfg

TOP = 50
PROPER_SHARE = 0.8
FEW_FILMS = 30
CORPORA = ("reviews", "screenplays", "summaries")


def genome_words() -> set[str]:
    from .s01_corpora import genome_meta
    out = set()
    for m in genome_meta():
        for field in ("title", "directedBy", "starring"):
            out |= {w.lower() for w in C.WORD.findall(m.get(field) or "") if len(w) >= 3}
    return out - set(ENGLISH_STOP_WORDS)


def audit(corpus: str, score: pd.Series, names: set[str]) -> pd.DataFrame:
    a, c = cfg()["analysis"], cfg()["clean"]
    df = pd.read_parquet(PRIVATE / f"{corpus}.parquet", columns=["tconst", "clean"])
    texts = df.groupby("tconst").clean.agg(" ".join).reindex(score.index)
    lo, hi = np.quantile(score, [0.2, 0.8])
    stop = list(ENGLISH_STOP_WORDS | {c["name_placeholder"].lower()})
    cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b[A-Za-z][A-Za-z]{2,}\b", stop_words=stop, binary=True,
                         min_df=a["word_min_doc_freq"])
    X = cv.fit_transform(texts)
    vocab = cv.get_feature_names_out()
    top, bot = np.asarray(X[score.to_numpy() >= hi].mean(axis=0)).ravel(), np.asarray(X[score.to_numpy() <= lo].mean(axis=0)).ravel()
    films = np.asarray(X.sum(axis=0)).ravel()
    t = pd.DataFrame({"word": vocab, "films": films, "share_top_fifth": top, "share_bottom_fifth": bot,
                      "log10_ratio": np.log10((top + 0.005) / (bot + 0.005))})
    t = t[(t.share_top_fifth + t.share_bottom_fifth) >= 0.05].sort_values("log10_ratio")
    t = pd.concat([t.tail(TOP).assign(end="more profit than expected"), t.head(TOP).assign(end="less profit than expected")])
    want = set(t.word)
    caps, total = Counter(), Counter()
    for text in texts:
        for w in C.WORD.findall(text):
            lw = w.lower()
            if lw in want:
                total[lw] += 1
                caps[lw] += w[:1].isupper()
    t["capitalised_share"] = t.word.map(lambda w: caps[w] / total[w] if total[w] else 0.0)
    t["proper"] = t.capitalised_share >= PROPER_SHARE
    t["genome_name"] = t.word.isin(names)
    t["few_films"] = t.films < FEW_FILMS
    return t.assign(corpus=corpus).round(4)


def run() -> pd.DataFrame:
    scores = pd.read_parquet(PRIVATE / "scores.parquet")
    names = genome_words()
    out = pd.concat([audit(c, scores[c].dropna(), names) for c in CORPORA])
    out.to_csv(OUTPUTS / "s04_leaks.csv", index=False)
    summary = {c: {"words_checked": int((out.corpus == c).sum()),
                   "proper": sorted(out.word[(out.corpus == c) & out.proper]),
                   "few_films": sorted(out.word[(out.corpus == c) & out.few_films]),
                   "genome_name_and_proper": sorted(out.word[(out.corpus == c) & out.proper & out.genome_name])}
               for c in CORPORA}
    (OUTPUTS / "s04_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return out


if __name__ == "__main__":
    r = run()
    for c in CORPORA:
        d = r[r.corpus == c]
        print(f"\n== {c}")
        for end in ("more profit than expected", "less profit than expected"):
            e = d[d.end == end].sort_values("log10_ratio", ascending=end.startswith("less"))
            print(end + ":", ", ".join(w + ("*" if p else "") + ("†" if f else "")
                                       for w, p, f in zip(e.word, e.proper, e.few_films)))
    print("\n* capitalised in 80%+ of uses   † used by fewer than 30 films")
