"""Step 1: the film sample and the three cleaned corpora.

    python -m analysis.embeddings.s01_corpora            # everything
    python -m analysis.embeddings.s01_corpora reviews    # only the reviews

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
from collections import Counter

import numpy as np
import pandas as pd

from ..clean_rule import clean_mask
from . import clean as C
from .paths import FEATURES, FILMS_SHARED, GENOME, MOVIESUM, OUTPUTS, PRIVATE, cfg

FILMS = PRIVATE / "films.parquet"
REPORT = OUTPUTS / "s01_cleaning.json"


def films() -> pd.DataFrame:
    """The clean theatrical sample with outcomes and controls."""
    s, k = cfg()["sample"], cfg()["controls"]
    m = pd.read_csv(FEATURES / s["table"], dtype={"tconst": str})
    d = m[clean_mask(m)].copy()          # the shared clean theatrical rule (analysis/clean_rule.py)
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


def build_screenplays(d: pd.DataFrame, ms: dict, people: "C.OtherNames | None" = None) -> dict:
    """people: full names of Tag Genome people, masked in title case (a real person named in a description,
    "a young Brando type"); extra: the names added by hand after the leak audit."""
    p = cfg()["clean"]
    s = p["screenplays"]
    extra = C.extra_names_pattern(p["extra_names"])
    rows, rep = [], {"clean_films_with_a_moviesum_script": 0, "dropped_too_short": 0}
    for t in d.tconst:
        if t not in ms:
            continue
        rep["clean_films_with_a_moviesum_script"] += 1
        raw = C.raw_script(ms[t]["script"])
        text, st = C.clean_script(ms[t]["script"], s["name_max_lowercase_share"], s["name_min_count"],
                                  s["name_min_length"], p["name_placeholder"])
        text, k1 = people.mask(text) if people else (text, {"other_people": 0})
        text, k2 = C.mask(text, extra, p["name_placeholder"])
        st["names_masked"] += k1["other_people"] + k2
        rep["genome_people_masked"] = rep.get("genome_people_masked", 0) + k1["other_people"]
        rep["hand_added_names_masked"] = rep.get("hand_added_names_masked", 0) + k2
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


def build_summaries(d: pd.DataFrame, ms: dict, people: "C.OtherNames | None" = None) -> dict:
    import spacy
    p = cfg()["clean"]
    nlp = spacy.load(p["spacy_model"], disable=["lemmatizer"])
    title = d.set_index("tconst").primaryTitle
    rows, rep = [], {"clean_films_with_a_moviesum_summary": 0, "dropped_too_short": 0}
    tot = {"brackets": 0, "actor_parens": 0, "titles": 0, "people": 0}
    extra = C.extra_names_pattern(p["extra_names"])
    for t in d.tconst:
        if t not in ms or not (ms[t].get("summary") or "").strip():
            continue
        rep["clean_films_with_a_moviesum_summary"] += 1
        raw = C.basic(ms[t]["summary"])
        text, st = C.clean_summary(ms[t]["summary"], nlp, C.title_pattern([title[t]]), p["name_placeholder"],
                                   p["title_placeholder"])
        text, k1 = people.mask(text) if people else (text, {"other_people": 0})
        text, k2 = C.mask(text, extra, p["name_placeholder"])
        st["people"] += k1["other_people"] + k2
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


POOL = PRIVATE / "reviews_pool.parquet"
SAMPLING = PRIVATE / "reviews_sampling.json"


def film_meta(d: pd.DataFrame, meta_all: list[dict]) -> tuple[dict, dict]:
    """Genome item id -> IMDb id, and IMDb id -> metadata, for the clean films."""
    keep, item, meta = set(d.tconst), {}, {}
    for r in meta_all:
        if r.get("imdbId") in (None, ""):
            continue
        t = "tt%07d" % int(r["imdbId"])
        if t in keep:
            item[r["item_id"]] = t
            meta[t] = r
    return item, meta


def build_reviews(d: pd.DataFrame) -> dict:
    sample_reviews(d)
    return mask_reviews(d)


def sample_reviews(d: pd.DataFrame) -> dict:
    """One pass over the 3.9 GB review file: drop empty, short, non-English and duplicate reviews, then draw
    per_film reviews per film by seeded reservoir sampling. Writes the raw sample to private/ (no cleaning)."""
    p = cfg()["clean"]
    rv = p["reviews"]
    z = zipfile.ZipFile(GENOME)
    pick = lambda k: next(n for n in z.namelist() if n.endswith(k) and "__MACOSX" not in n)
    item, _ = film_meta(d, genome_meta())
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
    rows = [{"tconst": t, "k": k, "raw": x, "valid_reviews": valid[t]} for t in sorted(pool) for k, x in enumerate(pool[t])]
    pd.DataFrame(rows).to_parquet(POOL, index=False)
    rep["clean_films_with_any_valid_review"] = len(pool)
    SAMPLING.write_text(json.dumps(rep, indent=1) + "\n")
    return rep


def mask_reviews(d: pd.DataFrame) -> dict:
    """Clean and mask the sampled reviews (private/reviews_pool.parquet), write private/reviews.parquet."""
    p = cfg()["clean"]
    rv = p["reviews"]
    imdb_title = d.set_index("tconst").primaryTitle
    meta_all = genome_meta()
    _, meta = film_meta(d, meta_all)
    rep = json.loads(SAMPLING.read_text())
    sample = pd.read_parquet(POOL)
    pool = {t: g.sort_values("k").raw.tolist() for t, g in sample.groupby("tconst")}
    valid = sample.groupby("tconst").valid_reviews.first().to_dict()
    others, singles_table = other_names(meta_all, pool)
    extra = C.extra_names_pattern(p["extra_names"])
    singles_table.to_csv(PRIVATE / "review_single_names.csv", index=False)   # for checking which words were masked
    rows = []
    tot = {"sentences": 0, "money_sentences": 0, "titles": 0, "people": 0, "other_titles": 0, "other_people": 0}
    with_money = short_after = too_few = 0
    for t in sorted(pool):
        if len(pool[t]) < rv["min_per_film"]:
            too_few += 1
            continue
        m = meta[t]
        tp = C.title_pattern([m.get("title") or "", imdb_title[t]])
        pp = C.people_pattern([x.strip() for x in ((m.get("directedBy") or "") + "," + (m.get("starring") or "")).split(",")])
        for k, raw in enumerate(pool[t]):
            text, st = C.clean_review(raw, tp, pp, p["name_placeholder"], p["title_placeholder"], others)
            text, k2 = C.mask(text, extra, p["name_placeholder"])
            st["other_people"] += k2
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
    rep.update(films_dropped_fewer_than_min_reviews=too_few,
               films=int(per.size), reviews_sampled=int(len(df)), median_reviews_per_film=float(per.median()),
               sampled_reviews_with_a_money_sentence=int(with_money),
               share_of_sentences_dropped_for_money=tot["money_sentences"] / tot["sentences"],
               title_mentions_masked=tot["titles"], cast_and_director_mentions_masked=tot["people"],
               other_film_title_mentions_masked=tot["other_titles"],
               other_people_mentions_masked=tot["other_people"],
               reference_titles_and_names=len(others.phrases),
               single_words_masked_on_their_own=len(others.singles),
               sampled_reviews_too_short_after_cleaning=int(short_after),
               latest_film_year=int(d.set_index("tconst").year[per.index].max()))
    return rep


def director_lift(meta_all: list[dict], pool: dict[str, list[str]], surnames: set[str], min_lift: float) -> set[str]:
    """Keep the surnames that the reviews of the director's own films use far more than other reviews do.
    "Kubrick" is everywhere in reviews of Kubrick films; "Paris" or "English", surnames of prolific directors
    too, are not, because in reviews they mostly mean the city or the language."""
    own = {}
    for m in meta_all:
        if m.get("imdbId") in (None, ""):
            continue
        t = "tt%07d" % int(m["imdbId"])
        if t in pool:
            for x in (m.get("directedBy") or "").split(","):
                w = x.strip().split()[-1].lower() if x.strip() else ""
                if w in surnames:
                    own.setdefault(w, set()).add(t)
    hits_own, words_own, hits_all, words_all = Counter(), Counter(), Counter(), 0
    for t, box in pool.items():
        for text in box:
            ws = [v for w in C.WORD.findall(text) if w[:1].isupper() for v in C._variants(w)[:2] if v in surnames]
            n = C.n_words(text)
            words_all += n
            for w in ws:
                hits_all[w] += 1
            for w in surnames:
                if t in own.get(w, ()):
                    words_own[w] += n
                    hits_own[w] += ws.count(w)
    keep = set()
    for w in own:
        if words_own[w] and hits_all[w]:
            lift = (hits_own[w] / words_own[w]) / (hits_all[w] / words_all)
            if lift >= min_lift:
                keep.add(w)
    return keep


def other_names(meta_all: list[dict], pool: dict[str, list[str]]) -> tuple[C.OtherNames, pd.DataFrame]:
    """Titles and people of every film in the Tag Genome (84,661 films), to mask when a review of one film
    mentions another. Single words are chosen from the sampled reviews themselves (see clean.OtherNames)."""
    o = cfg()["clean"]["other_names"]
    titles = [m.get("title") or "" for m in meta_all]
    people = [x.strip() for m in meta_all
              for x in ((m.get("directedBy") or "") + "," + (m.get("starring") or "")).split(",") if x.strip()]
    phrases = C.name_phrases(titles, people, o["min_title_chars"])
    texts = [x for box in pool.values() for x in box]
    import spacy
    nlp = spacy.load(cfg()["clean"]["spacy_model"], disable=["lemmatizer", "parser"])
    phrases, _, dropped = C.drop_places_and_dates(phrases, {}, nlp)       # before counting, so "New York" stays
    # Directors with many films: their surnames are used alone ("the Coens", "Kubrick's"), so for them being
    # capitalised is enough; the place check in context still applies (it keeps "Paris" and "English").
    directed = Counter(x.strip() for m in meta_all for x in (m.get("directedBy") or "").split(",") if x.strip())
    famous = {n.split()[-1].lower() for n, k in directed.items()
              if k >= o["famous_director_min_films"] and len(n.split()) >= 2 and len(n.split()[-1]) >= o["single_min_length"]}
    famous = director_lift(meta_all, pool, famous, o["famous_director_min_lift"])
    singles, tab = C.distinctive_singles(texts, phrases, o["single_min_count"], o["single_max_lowercase_share"],
                                         o["single_min_share_in_full_name"], o["single_min_length"], famous)
    # What each single word is where it is used: spaCy's transformer NER on context_sentences mentions per
    # word (entities.py). It agreed better with the hand labels of the Tag Genome tags than the small model
    # (outputs/tag_entity_typing.json). A word used mostly as a place, a group or a time stays.
    from . import entities as E
    typer = spacy.load(o["entity_model"])
    votes = E.type_votes(E.contexts(sorted(singles), texts, o["context_sentences"], cfg()["seed"]), typer)
    labels = dict(zip(votes["item"], votes["type"]))
    votes.to_csv(PRIVATE / "review_single_types.csv", index=False)
    dropped_single = sorted(w for w, lab in labels.items() if lab in ("place", "group", "time"))
    singles = {w: k for w, k in singles.items() if w not in dropped_single}
    if not tab.empty:
        tab["context_type"] = tab.word.map(labels).fillna("")
        tab["kept_as_place_or_date"] = tab.word.isin(dropped_single)
        tab["masked"] &= ~tab.kept_as_place_or_date
    (PRIVATE / "review_places_and_dates_not_masked.txt").write_text("\n".join(dropped + dropped_single) + "\n")
    return C.OtherNames(phrases, singles), tab


GENOME_META = PRIVATE / "genome_meta.jsonl"


def genome_meta() -> list[dict]:
    """The Tag Genome's film metadata (titles, directors, stars). Read from the zip once and kept in private/,
    so the steps after sampling (and an HPC job) don't need the 1.9 GB zip."""
    if not GENOME_META.exists():
        z = zipfile.ZipFile(GENOME)
        name = next(n for n in z.namelist() if n.endswith("raw/metadata.json") and "__MACOSX" not in n)
        with z.open(name) as f, open(GENOME_META, "w", encoding="utf8") as out:
            for line in io.TextIOWrapper(f, encoding="utf8"):
                if line.strip():
                    out.write(line if line.endswith("\n") else line + "\n")
    with open(GENOME_META, encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def genome_people() -> C.OtherNames:
    """Full names of everyone credited as director or star in the Tag Genome, as an OtherNames masker."""
    people = [x.strip() for m in genome_meta()
              for x in ((m.get("directedBy") or "") + "," + (m.get("starring") or "")).split(",") if x.strip()]
    return C.OtherNames({k: v for k, v in C.name_phrases([], people).items() if v == "person"})


def run() -> dict:
    d = films()
    d.to_parquet(FILMS, index=False)
    FILMS_SHARED.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(FILMS_SHARED, index=False)    # no text, so it can go in the bundle with the vectors
    ms = moviesum()
    report = {"sample": {"clean_films": len(d), "in_100k_tier": int((d.tier == "100k+").sum()),
                         "gross_below_budget": int((d.log_mult < 0).sum()), "franchise": int(d.franchise.sum())}}
    print(report["sample"], flush=True)
    people = genome_people()
    report["screenplays"] = build_screenplays(d, ms, people)
    print("screenplays", report["screenplays"]["films"], flush=True)
    report["summaries"] = build_summaries(d, ms, people)
    print("summaries", report["summaries"]["films"], flush=True)
    report["reviews"] = build_reviews(d)
    print("reviews", report["reviews"]["films"], flush=True)
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    return report


def run_step(step: str) -> dict:
    """sample-reviews (needs the zip), mask-reviews (needs only private/), reviews (both), or texts
    (screenplays and summaries only)."""
    d = pd.read_parquet(FILMS)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    if step == "sample-reviews":
        return sample_reviews(d)
    if step == "texts":
        ms, people = moviesum(), genome_people()
        report["screenplays"] = build_screenplays(d, ms, people)
        report["summaries"] = build_summaries(d, ms, people)
    else:
        report["reviews"] = mask_reviews(d) if step == "mask-reviews" else build_reviews(d)
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    print(json.dumps(run_step(args[0]) if args else run(), indent=1))
