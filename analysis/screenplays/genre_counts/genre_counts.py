"""Task 2b: how many films per genre fall in hit / middle / flop?

Uses ONLY genre, budget, gross and release year from movies_features.csv, plus
the list of imdb ids that have a MovieSum script. No script text is read.

A film with several genres is counted once in EACH of its genres.

Run:  python3 analysis/screenplays/genre_counts/genre_counts.py
Set IMDB_DATA_DIR if the data is not in data/raw.
"""
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("IMDB_DATA_DIR", ROOT / "data" / "raw"))
OUT = Path(__file__).resolve().parent
MIN_GROUP = 50  # decided by the group (D14): groups below this are exploratory or out


FLAGGED = []


def script_ids():
    """imdb ids with a MovieSum script (a set, so duplicated ids count once)."""
    ids = set()
    for split in ["train", "val", "test"]:
        with open(DATA / "scripts" / "moviesum" / f"{split}.jsonl") as f:
            for line in f:
                ids.add(json.loads(line)["imdb_id"])  # only the id is kept
    return ids


def tiny_ids():
    """imdb ids whose dialogue or description layer is empty or tiny.

    Rule (data_check/layer_stats.py): fewer than 500 whitespace tokens in the
    dialogue layer or in the description layer. It uses only tags and token
    counts, never hit/flop, budget or gross.
    """
    with open(ROOT / "analysis" / "screenplays" / "data_check" / "layer_stats.csv") as f:
        return {r["imdb_id"] for r in csv.DictReader(f) if r["tiny_layer"] == "1"}


def load_films(exclude_tiny=True):
    ids = script_ids()
    tiny = tiny_ids() if exclude_tiny else set()
    films = []
    with open(DATA / "features" / "movies_features.csv") as f:
        for r in csv.DictReader(f):
            if r["tconst"] not in ids or not r["b"] or not r["g"]:
                continue
            if r["tconst"] in tiny:  # decided by the group (D9): excluded from the whole analysis
                continue
            b, g = float(r["b"]), float(r["g"])
            if b <= 0 or g <= 0:
                continue
            # first money rule, superseded by D6: drop films whose budget or gross carries an infobox flag
            # (regional, rentals, net_gross, direct_to_video): not comparable
            if r["budget_flag"] or r["gross_flag"]:
                FLAGGED.append(r["tconst"])
                continue
            # a ratio only makes sense if budget and gross share a currency
            if r["budget_cur"] != r["gross_cur"]:
                continue
            films.append({
                "id": r["tconst"],
                "genres": r["genres"].split(","),
                "decade": int(r["y"]) // 10 * 10,
                "ratio": g / b,
                "ln": math.log(g / b),
            })
    return films


def absolute(group):
    """(i) hit: ratio >= 2, flop: ratio < 1, middle: in between."""
    c = {"hit": 0, "middle": 0, "flop": 0}
    for fl in group:
        c["hit" if fl["ratio"] >= 2 else "flop" if fl["ratio"] < 1 else "middle"] += 1
    return c


def relative(group):
    """(ii)/(iii) hit: ln ratio >= Q3, flop: <= Q1, middle: strictly between."""
    c = {"hit": 0, "middle": 0, "flop": 0}
    if len(group) < 4:
        return None
    q1, _, q3 = statistics.quantiles([fl["ln"] for fl in group], n=4, method="inclusive")
    for fl in group:
        c["hit" if fl["ln"] >= q3 else "flop" if fl["ln"] <= q1 else "middle"] += 1
    return c


def main(films):
    multi = sum(len(fl["genres"]) > 1 for fl in films)
    print(f"films with script + budget + gross (same currency, no infobox flag): {len(films)}"
          f"  (excluded for a flag: {len(FLAGGED)})")
    print(f"films with more than one genre: {multi}")

    by_genre = defaultdict(list)
    by_gd = defaultdict(list)
    for fl in films:
        for g in fl["genres"]:
            by_genre[g].append(fl)
            by_gd[(g, fl["decade"])].append(fl)

    rows = []
    for g in sorted(by_genre, key=lambda g: -len(by_genre[g])):
        grp = by_genre[g]
        a = absolute(grp)
        r = relative(grp) or {"hit": 0, "middle": 0, "flop": 0}  # <4 films: no quartiles
        # (iii): quartiles cut within genre x decade; sum the counts over decades
        d = {"hit": 0, "middle": 0, "flop": 0}
        small_decades = 0
        for (g2, dec), sub in by_gd.items():
            if g2 != g:
                continue
            rr = relative(sub)
            if rr is None:
                small_decades += len(sub)  # too few films to cut quartiles
                continue
            for k in d:
                d[k] += rr[k]
        row = {
            "genre": g, "n_films": len(grp),
            "abs_hit": a["hit"], "abs_middle": a["middle"], "abs_flop": a["flop"],
            "rel_hit": r["hit"], "rel_middle": r["middle"], "rel_flop": r["flop"],
            "reldec_hit": d["hit"], "reldec_middle": d["middle"], "reldec_flop": d["flop"],
            "reldec_films_in_decades_under_4": small_decades,
        }
        row["flag_abs"] = min(a["hit"], a["flop"]) < MIN_GROUP
        row["flag_rel"] = min(r["hit"], r["flop"]) < MIN_GROUP
        row["flag_reldec"] = min(d["hit"], d["flop"]) < MIN_GROUP
        rows.append(row)

    with open(OUT / "genre_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nflag = hit or flop group has fewer than {MIN_GROUP} films (* = flagged)")
    print(f"{'genre':<10}{'n':>5} | {'(i) hit/mid/flop':>17} | {'(ii) hit/mid/flop':>18} | {'(iii) hit/mid/flop':>19}")
    for r in rows:
        def cell(p, flag):
            s = f"{r[p+'_hit']}/{r[p+'_middle']}/{r[p+'_flop']}"
            return s + (" *" if r[flag] else "")
        print(f"{r['genre']:<10}{r['n_films']:>5} | {cell('abs','flag_abs'):>17} | "
              f"{cell('rel','flag_rel'):>18} | {cell('reldec','flag_reldec'):>19}")


def decade_table(films):
    """Share of hits per decade under definition (ii), quartiles cut within each genre only.

    Uses genre, money and year only. Pooled over film-genre pairs, and for each genre.
    """
    by_genre = defaultdict(list)
    for fl in films:
        for g in fl["genres"]:
            by_genre[g].append(fl)
    rows = []
    for g, grp in by_genre.items():
        if len(grp) < 4:
            continue
        q1, _, q3 = statistics.quantiles([fl["ln"] for fl in grp], n=4, method="inclusive")
        for fl in grp:
            rows.append((g, fl["decade"], fl["ln"] >= q3, fl["ln"] <= q1))
    big = [g for g, grp in by_genre.items() if len(grp) >= 200]
    decades = sorted({r[1] for r in rows})
    out = []
    for scope in ["ALL genres pooled"] + sorted(big):
        for dec in decades:
            sel = [r for r in rows if r[1] == dec and (scope.startswith("ALL") or r[0] == scope)]
            n = len(sel)
            out.append({"scope": scope, "decade": dec, "n": n,
                        "hit_pct": round(100 * sum(r[2] for r in sel) / n, 1) if n else "",
                        "flop_pct": round(100 * sum(r[3] for r in sel) / n, 1) if n else ""})
    with open(OUT / "decade_hits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "decade", "n", "hit_pct", "flop_pct"])
        w.writeheader()
        w.writerows(out)
    print("\nhit % per decade under (ii) (about 25 % everywhere if the decade did not matter; n in brackets)")
    print(f"{'':<20}" + "".join(f"{d:>13}" for d in decades))
    for scope in ["ALL genres pooled"] + sorted(big):
        cells = {o["decade"]: o for o in out if o["scope"] == scope}
        print(f"{scope:<20}" + "".join(
            f"{(str(cells[d]['hit_pct']) + ' (' + str(cells[d]['n']) + ')'):>13}" for d in decades))
    return out


if __name__ == "__main__":
    films = load_films()
    main(films)
    decade_table(films)
