"""Sentiment and embeddings of audience reviews, per clean movie, from MovieLens
Tag Genome 2021 (IMDb reviews with IMDb IDs, nothing after 2017).

Run from the data folder (data/raw):

    cd data/raw && ../../.venv/bin/python ../../scripts/reviews/review_features.py

For each movie in the clean sample (the rule in scripts/imdb/genome_stats.py,
over features/movies_features_10k.csv) it takes the first N_SENT reviews in
the file's order, so reruns pick the same ones, and writes to
features/review_features.csv:
- reviews: how many the genome has for the movie,
- vader_mean: mean VADER compound score (-1 to 1) of the first N_SENT,
- vader_pos_share: share of those scoring above 0.05 (VADER's own cut-off).

It embeds the first N_EMB reviews with all-MiniLM-L6-v2, the model used for
the screenplay vectors, and writes each movie's mean vector, normalised, to
features/review_vectors.npy, with the row order in features/review_vector_ids.csv.

Review text stays in the zip and is never written out.
"""

import csv
import io
import json
import re
import zipfile

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ZIP = "reviews/genome/genome_2021.zip"
FEATURES = "features/movies_features_10k.csv"
OUT = "features/review_features.csv"
VECTORS = "features/review_vectors.npy"
VECTOR_IDS = "features/review_vector_ids.csv"
N_SENT = 50
N_EMB = 20
MODEL = "all-MiniLM-L6-v2"


def clean_ids():
    """IMDb IDs of the clean theatrical sample, as genome_stats.py defines it."""
    def ok(r):
        return (r["budget_cur"] == "USD" and r["gross_cur"] == "USD" and float(r["b"] or 0) >= 1000
                and float(r["g"] or 0) >= 1000 and "rentals" not in r["gross_flag"]
                and "regional" not in r["gross_flag"]
                and not re.search("netflix|amazon|apple|hulu", r["distributors"] or "", re.I)
                and int(r["y"] or 0) < 2026)
    with open(FEATURES, newline="") as f:
        return {r["tconst"] for r in csv.DictReader(f) if ok(r)}


def main():
    keep = clean_ids()
    z = zipfile.ZipFile(ZIP)
    pick = lambda k: next(n for n in z.namelist() if n.endswith(k) and "__MACOSX" not in n)
    item = {}
    with z.open(pick("metadata.json")) as f:
        for line in io.TextIOWrapper(f, encoding="utf8"):
            r = json.loads(line)
            if r.get("imdbId") not in (None, ""):
                t = "tt%07d" % int(r["imdbId"])
                if t in keep:
                    item[r["item_id"]] = t
    counts, texts = {}, {}
    with z.open(pick("reviews.json")) as f:
        for i, line in enumerate(io.TextIOWrapper(f, encoding="utf8"), 1):
            r = json.loads(line)
            t = item.get(r["item_id"])
            if t is None:
                continue
            counts[t] = counts.get(t, 0) + 1
            txt = (r.get("txt") or "").strip()
            if txt and len(texts.setdefault(t, [])) < N_SENT:
                texts[t].append(txt)
            if i % 500_000 == 0:
                print(f"read {i:,} reviews", flush=True)
    ids = sorted(texts)
    vader = SentimentIntensityAnalyzer()
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tconst", "reviews", "scored", "vader_mean", "vader_pos_share"])
        for n, t in enumerate(ids, 1):
            s = [vader.polarity_scores(x)["compound"] for x in texts[t]]
            w.writerow([t, counts[t], len(s), round(float(np.mean(s)), 6),
                        round(float(np.mean([x > 0.05 for x in s])), 6)])
            if n % 500 == 0:
                print(f"sentiment {n}/{len(ids)} movies", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device="cpu")
    vecs = []
    for n, t in enumerate(ids, 1):
        e = model.encode(texts[t][:N_EMB], batch_size=64, normalize_embeddings=True)
        m = e.mean(axis=0)
        vecs.append(m / np.linalg.norm(m))
        if n % 500 == 0:
            print(f"embedded {n}/{len(ids)} movies", flush=True)
    np.save(VECTORS, np.array(vecs, dtype=np.float32))
    with open(VECTOR_IDS, "w", newline="") as f:
        csv.writer(f).writerows([["tconst"]] + [[t] for t in ids])
    print(f"wrote {OUT} and {VECTORS}: {len(ids)} movies")


if __name__ == "__main__":
    main()
