"""Step 1: the film sample and the three cleaned corpora.

    python -m analysis.embeddings.s01_corpora

Writes cleaned and raw text to private/ (gitignored) and counts of what each cleaning rule did to
outputs/s01_cleaning.json. No text goes to outputs/.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import re
import zipfile

import numpy as np
import pandas as pd

from . import clean as C
from .paths import FEATURES, GENOME, MOVIESUM, OUTPUTS, PRIVATE, cfg

FILMS = PRIVATE / "films.parquet"
REPORT = OUTPUTS / "s01_cleaning.json"


def films() -> pd.DataFrame:
    """The clean theatrical sample with outcomes and controls."""
    s, k = cfg()["sample"], cfg()["controls"]
    m = pd.read_csv(FEATURES / s["table"], dtype={"tconst": str})
    ok = ((m.budget_cur == "USD") & (m.gross_cur == "USD") & (m.b >= s["min_money_usd"]) & (m.g >= s["min_money_usd"])
          & ~m.gross_flag.fillna("").str.contains(s["drop_gross_flags"])
          & ~m.distributors.fillna("").str.contains(s["streaming_distributors"], case=False)
          & (m.y >= s["min_year_inclusive"]) & (m.y < s["max_year_exclusive"]))
    d = m[ok].copy()
    d["year"] = d.y.astype(int)
    d["log_mult"], d["log_gross"], d["log_budget"] = np.log(d.g / d.b), np.log(d.g), np.log(d.b)

    def genre(g):
        have = set((g or "").split(","))
        return next((x for x in k["primary_genre_priority"] if x in have), "Other")

    def decade(y):
        return next(label for lo, hi, label in k["decade_bins"] if lo <= y <= hi)

    d["primary_genre"] = d.genres.fillna("").map(genre)
    d["decade_bin"] = d.year.map(decade)
    fr = pd.read_csv(FEATURES / "franchise.csv", dtype={"tconst": str}).set_index("tconst").franchise
    d["franchise"] = d.tconst.map(fr).fillna(0).astype(int)
    keep = ["tconst", "primaryTitle", "year", "genres", "votes", "rating", "tier", "log_mult", "log_gross",
            "log_budget", "primary_genre", "decade_bin", "franchise"]
    return d[keep].reset_index(drop=True)


def moviesum() -> dict[str, dict]:
    """One MovieSum record per IMDb ID: the longest script, as everywhere else in the repo."""
    best = {}
    for path in MOVIESUM:
        with open(path, encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                if r["imdb_id"] not in best or len(r["script"] or "") > len(best[r["imdb_id"]]["script"] or ""):
                    best[r["imdb_id"]] = r
    return best


def build_screenplays(d: pd.DataFrame, ms: dict) -> dict:
    p = cfg()["clean"]
    s = p["screenplays"]
    rows, rep = [], {"clean_films_with_a_moviesum_script": 0, "dropped_too_short": 0}
    for t in d.tconst:
        if t not in ms:
            continue
        rep["clean_films_with_a_moviesum_script"] += 1
        raw = C.raw_script(ms[t]["script"])
        text, st = C.clean_script(ms[t]["script"], s["name_max_lowercase_share"], s["name_min_count"],
                                  s["name_min_length"], p["name_placeholder"])
        if C.n_words(text) < s["min_words"]:
            rep["dropped_too_short"] += 1
            continue
        rows.append({"tconst": t, "raw": raw, "clean": text, "unmasked": st["unmasked"], "names": st["names"],
                     "words_raw": C.n_words(raw), "words_clean": C.n_words(text), "names_masked": st["names_masked"]})
    df = pd.DataFrame(rows)
    df.to_parquet(PRIVATE / "screenplays.parquet", index=False)
    rep.update(films=len(df), median_words_raw=float(df.words_raw.median()),
               median_words_clean=float(df.words_clean.median()),
               share_of_raw_words_removed=float(1 - df.words_clean.sum() / df.words_raw.sum()),
               median_distinct_names_masked=float(df.names.map(len).median()),
               share_of_clean_words_that_are_name_placeholders=float(df.names_masked.sum() / df.words_clean.sum()),
               films_with_no_name_masked=int((df.names_masked == 0).sum()))
    rep["name_rule_vs_spacy"] = ner_check(df)
    return rep


def ner_check(df: pd.DataFrame) -> dict:
    """Second method for the name rule: on a sample of scripts, compare the words the label rule masks with
    the words spaCy tags PERSON. Neither is ground truth; this is agreement, word by word."""
    import spacy
    p = cfg()["clean"]
    s = p["screenplays"]
    nlp = spacy.load(p["spacy_model"], disable=["lemmatizer"])
    sample = df.sample(min(s["ner_check_scripts"], len(df)), random_state=cfg()["seed"])
    both = rule_only = ner_only = 0
    for r in sample.itertuples():
        words = r.unmasked.split()
        mid = max(0, len(words) // 2 - s["ner_check_words"] // 2)
        doc = nlp(" ".join(words[mid:mid + s["ner_check_words"]]))
        names = set(r.names)
        person = {tok.i for ent in doc.ents if ent.label_ == "PERSON" for tok in ent}
        for tok in doc:
            if not tok.is_alpha:
                continue
            a, b = tok.text.upper() in names, tok.i in person
            both += a and b
            rule_only += a and not b
            ner_only += b and not a
    return {"scripts": len(sample), "words_per_script": s["ner_check_words"], "masked_by_both": both,
            "masked_by_label_rule_only": rule_only, "tagged_person_by_spacy_only": ner_only,
            "share_of_rule_words_spacy_also_tags": both / max(1, both + rule_only),
            "share_of_spacy_person_words_the_rule_masks": both / max(1, both + ner_only)}


def build_summaries(d: pd.DataFrame, ms: dict) -> dict:
    import spacy
    p = cfg()["clean"]
    nlp = spacy.load(p["spacy_model"], disable=["lemmatizer"])
    title = d.set_index("tconst").primaryTitle
    rows, rep = [], {"clean_films_with_a_moviesum_summary": 0, "dropped_too_short": 0}
    tot = {"brackets": 0, "actor_parens": 0, "titles": 0, "people": 0}
    for t in d.tconst:
        if t not in ms or not (ms[t].get("summary") or "").strip():
            continue
        rep["clean_films_with_a_moviesum_summary"] += 1
        raw = C.basic(ms[t]["summary"])
        text, st = C.clean_summary(ms[t]["summary"], nlp, C.title_pattern([title[t]]), p["name_placeholder"],
                                   p["title_placeholder"])
        if C.n_words(text) < p["summaries"]["min_words"]:
            rep["dropped_too_short"] += 1
            continue
        for k in tot:
            tot[k] += st[k]
        rows.append({"tconst": t, "raw": raw, "clean": text, "words_raw": C.n_words(raw), "words_clean": C.n_words(text),
                     "people": st["people"]})
    df = pd.DataFrame(rows)
    df.to_parquet(PRIVATE / "summaries.parquet", index=False)
    rep.update(films=len(df), median_words_raw=float(df.words_raw.median()),
               median_words_clean=float(df.words_clean.median()), removed=tot,
               share_of_clean_words_that_are_name_placeholders=float(df.people.sum() / df.words_clean.sum()),
               films_with_no_name_masked=int((df.people == 0).sum()))
    return rep


def build_reviews(d: pd.DataFrame) -> dict:
    p = cfg()["clean"]
    rv = p["reviews"]
    keep = set(d.tconst)
    imdb_title = d.set_index("tconst").primaryTitle
    z = zipfile.ZipFile(GENOME)
    pick = lambda k: next(n for n in z.namelist() if n.endswith(k) and "__MACOSX" not in n)
    item, meta = {}, {}
    with z.open(pick("raw/metadata.json")) as f:
        for line in io.TextIOWrapper(f, encoding="utf8"):
            r = json.loads(line)
            if r.get("imdbId") in (None, ""):
                continue
            t = "tt%07d" % int(r["imdbId"])
            if t in keep:
                item[r["item_id"]] = t
                meta[t] = r
    rng = random.Random(cfg()["seed"])
    seen_hash, valid, pool = {}, {}, {}
    rep = {"reviews_read_for_clean_films": 0, "dropped_empty_or_short": 0, "dropped_not_english": 0,
           "dropped_duplicate": 0}
    with z.open(pick("raw/reviews.json")) as f:
        for i, line in enumerate(io.TextIOWrapper(f, encoding="utf8"), 1):
            if i % 1_000_000 == 0:
                print(f"read {i:,} reviews", flush=True)
            r = json.loads(line)
            t = item.get(r["item_id"])
            if t is None:
                continue
            rep["reviews_read_for_clean_films"] += 1
            txt = C.basic(r.get("txt") or "")
            if C.n_words(txt) < rv["min_words"]:
                rep["dropped_empty_or_short"] += 1
                continue
            if not C.looks_english(txt, rv["min_ascii_letter_share"], rv["min_english_function_words"]):
                rep["dropped_not_english"] += 1
                continue
            h = hashlib.md5(re.sub(r"\W+", "", txt.lower())[:400].encode()).digest()[:8]
            if h in seen_hash.setdefault(t, set()):
                rep["dropped_duplicate"] += 1
                continue
            seen_hash[t].add(h)
            n = valid[t] = valid.get(t, 0) + 1
            box = pool.setdefault(t, [])
            if len(box) < rv["per_film"]:          # reservoir sampling: every valid review equally likely
                box.append(txt)
            else:
                j = rng.randrange(n)
                if j < rv["per_film"]:
                    box[j] = txt
    rows = []
    tot = {"sentences": 0, "money_sentences": 0, "titles": 0, "people": 0}
    with_money = short_after = too_few = 0
    for t in sorted(pool):
        if len(pool[t]) < rv["min_per_film"]:
            too_few += 1
            continue
        m = meta[t]
        tp = C.title_pattern([m.get("title") or "", imdb_title[t]])
        pp = C.people_pattern([x.strip() for x in ((m.get("directedBy") or "") + "," + (m.get("starring") or "")).split(",")])
        for k, raw in enumerate(pool[t]):
            text, st = C.clean_review(raw, tp, pp, p["name_placeholder"], p["title_placeholder"])
            for key in tot:
                tot[key] += st[key]
            with_money += st["money_sentences"] > 0
            if C.n_words(text) < rv["min_words"]:
                short_after += 1
                continue
            rows.append({"tconst": t, "k": k, "raw": raw, "clean": text, "valid_reviews": valid[t]})
    df = pd.DataFrame(rows)
    n = df.groupby("tconst").k.transform("size")
    too_few += int(df.tconst[n < rv["min_per_film"]].nunique())
    df = df[n >= rv["min_per_film"]]
    df.to_parquet(PRIVATE / "reviews.parquet", index=False)
    per = df.groupby("tconst").size()
    rep.update(clean_films_with_any_valid_review=len(pool), films_dropped_fewer_than_min_reviews=too_few,
               films=int(per.size), reviews_sampled=int(len(df)), median_reviews_per_film=float(per.median()),
               sampled_reviews_with_a_money_sentence=int(with_money),
               share_of_sentences_dropped_for_money=tot["money_sentences"] / tot["sentences"],
               title_mentions_masked=tot["titles"], cast_and_director_mentions_masked=tot["people"],
               sampled_reviews_too_short_after_cleaning=int(short_after),
               latest_film_year=int(d.set_index("tconst").year[per.index].max()))
    return rep


def run() -> dict:
    d = films()
    d.to_parquet(FILMS, index=False)
    ms = moviesum()
    report = {"sample": {"clean_films": len(d), "in_100k_tier": int((d.tier == "100k+").sum()),
                         "gross_below_budget": int((d.log_mult < 0).sum()), "franchise": int(d.franchise.sum())}}
    print(report["sample"], flush=True)
    report["screenplays"] = build_screenplays(d, ms)
    print("screenplays", report["screenplays"]["films"], flush=True)
    report["summaries"] = build_summaries(d, ms)
    print("summaries", report["summaries"]["films"], flush=True)
    report["reviews"] = build_reviews(d)
    print("reviews", report["reviews"]["films"], flush=True)
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
