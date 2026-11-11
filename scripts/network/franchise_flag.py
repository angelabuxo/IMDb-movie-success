"""Which movies belong to a series or franchise, from each film's Wikidata
entry. Run from the data folder:

    cd data/raw && ../../.venv/bin/python ../../scripts/network/franchise_flag.py

A movie is flagged when its Wikidata item has any of:
- follows (P155) or followed by (P156) pointing to another film: sequels, and
  first films of a series, which "follows" alone misses;
- part of the series (P179) pointing to an item that is an instance of a film
  series or a media franchise (or a subclass of either one step down). This
  leaves out lists such as the BBC's "100 greatest films", which P179 also
  holds.

The Wikidata item comes from data/movie_links.csv. Writes
features/franchise.csv: tconst, qid, follows, followed_by, series (the
series item IDs that passed), franchise (1 or 0). API answers are cached in
features/franchise_cache/.
"""

import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

LINKS = "../../data/movie_links.csv"
OUT = "features/franchise.csv"
CACHE = Path("features/franchise_cache")
API = "https://www.wikidata.org/w/api.php"
AGENT = "IMDb-movie-success course project (DTU 02805; https://github.com/angelabuxo/IMDb-movie-success)"
SERIES_CLASSES = {"Q24856", "Q196600"}  # film series, media franchise


def get(params):
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=60) as r:
                data = json.loads(r.read())
            break
        except Exception:
            time.sleep(2 ** attempt)
    else:
        raise SystemExit(f"Wikidata API kept failing for {url[:120]}")
    CACHE.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".part").write_text(json.dumps(data))
    path.with_suffix(".part").rename(path)
    time.sleep(0.2)
    return data


def claims(qids, props):
    out = {}
    qids = sorted(qids)
    for i in range(0, len(qids), 50):
        ents = get({"action": "wbgetentities", "ids": "|".join(qids[i:i + 50]), "props": "claims",
                    "format": "json"}).get("entities", {})
        for q, e in ents.items():
            c = e.get("claims", {})
            out[q] = {p: sorted({s["mainsnak"]["datavalue"]["value"]["id"] for s in c.get(p, [])
                                 if s.get("rank") != "deprecated" and "datavalue" in s["mainsnak"]})
                      for p in props}
        if (i // 50) % 50 == 0:
            print(f"read {min(i + 50, len(qids))}/{len(qids)} items", flush=True)
    return out


def main():
    film = {}
    with open(Path(__file__).resolve().parents[2] / "data" / "movie_links.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["wikidata_url"]:
                film.setdefault(r["imdb_id"], r["wikidata_url"].rsplit("/", 1)[-1])
    fc = claims(set(film.values()), ["P155", "P156", "P179"])
    series = {s for c in fc.values() for s in c["P179"]}
    sc = claims(series, ["P31", "P279"])
    sub = {c for v in sc.values() for c in v["P31"] if c not in SERIES_CLASSES}
    subc = claims(sub, ["P279"])  # classes one step below film series or media franchise
    ok_classes = SERIES_CLASSES | {c for c, v in subc.items() if set(v["P279"]) & SERIES_CLASSES}
    good_series = {s for s, v in sc.items() if set(v["P31"]) & ok_classes}

    rows = []
    for t in sorted(film):
        c = fc.get(film[t], {"P155": [], "P156": [], "P179": []})
        ser = [s for s in c["P179"] if s in good_series]
        flag = int(bool(c["P155"] or c["P156"] or ser))
        rows.append({"tconst": t, "qid": film[t], "follows": "|".join(c["P155"]), "followed_by": "|".join(c["P156"]),
                     "series": "|".join(ser), "franchise": flag})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    print(f"wrote {OUT}: {sum(r['franchise'] for r in rows):,} of {n:,} films flagged "
          f"({sum(bool(r['follows']) for r in rows):,} follow a film, {sum(bool(r['followed_by']) for r in rows):,} "
          f"are followed by one, {sum(bool(r['series']) for r in rows):,} in a film series or franchise); "
          f"{len(series) - len(good_series):,} of {len(series):,} P179 targets refused as not a film series")


if __name__ == "__main__":
    main()
