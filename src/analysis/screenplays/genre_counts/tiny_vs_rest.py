"""Are the films excluded for an empty/tiny layer a random subset? (no outcomes used)

Compares era and genre of the excluded films with the rest. Uses only year,
genre and the tiny flag. Output: tiny_vs_rest.csv. Run with python3 -I.
With 21 films the comparison has little power: it can show a big skew, not rule out a small one.
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from genre_cells import ERAS, era  # noqa: E402
from genre_counts import OUT, ROOT, load_films, tiny_ids  # noqa: E402

tiny = tiny_ids()
stats = list(csv.DictReader(open(ROOT / "src" / "analysis" / "screenplays" / "data_check" / "layer_stats.csv")))

# 1. era over all 2188 unique ids (year read from the movie_name suffix, e.g. "_1999")
def year(name):
    m = re.search(r"_(\d{4})$", name)
    return int(m.group(1)) if m else None
c_t, c_r = Counter(), Counter()
for r in stats:
    y = year(r["movie_name"])
    (c_t if r["imdb_id"] in tiny else c_r)[era(y // 10 * 10)] += 1
nt, nr = sum(c_t.values()), sum(c_r.values())
rows = []
print(f"== era, all unique ids: {nt} excluded vs {nr} rest")
print(f"{'era':<10}{'excl n':>8}{'excl %':>8}{'rest n':>8}{'rest %':>8}")
for e in ERAS:
    print(f"{e:<10}{c_t[e]:>8}{100*c_t[e]/nt:>8.1f}{c_r[e]:>8}{100*c_r[e]/nr:>8.1f}")
    rows.append({"table": "era_all_ids", "key": e, "excluded": c_t[e], "rest": c_r[e]})

# 2. era and genre in the analysis set BEFORE the exclusion (1125 films)
films = load_films(exclude_tiny=False)
t_f = [f for f in films if f["id"] in tiny]
r_f = [f for f in films if f["id"] not in tiny]
print(f"\n== analysis set before exclusion: {len(t_f)} excluded vs {len(r_f)} rest")
ge_t, ge_r = Counter(), Counter()
for f in films:
    for g in f["genres"]:
        (ge_t if f["id"] in tiny else ge_r)[g] += 1
print(f"{'genre':<10}{'excl n':>8}{'excl %':>8}{'rest n':>8}{'rest %':>8}   (% of films of that group tagged with the genre)")
for g in ["Drama", "Comedy", "Action", "Adventure", "Crime", "Thriller", "Horror", "Mystery", "Sci-Fi", "Romance"]:
    print(f"{g:<10}{ge_t[g]:>8}{100*ge_t[g]/len(t_f):>8.1f}{ge_r[g]:>8}{100*ge_r[g]/len(r_f):>8.1f}")
    rows.append({"table": "genre_analysis_set", "key": g, "excluded": ge_t[g], "rest": ge_r[g]})
ea_t, ea_r = Counter(era(f["decade"]) for f in t_f), Counter(era(f["decade"]) for f in r_f)
print(f"\n{'era':<10}{'excl n':>8}{'excl %':>8}{'rest n':>8}{'rest %':>8}")
for e in ERAS:
    print(f"{e:<10}{ea_t[e]:>8}{100*ea_t[e]/len(t_f):>8.1f}{ea_r[e]:>8}{100*ea_r[e]/len(r_f):>8.1f}")
    rows.append({"table": "era_analysis_set", "key": e, "excluded": ea_t[e], "rest": ea_r[e]})

# 3. films removed per genre x era bin (main genres)
rem = defaultdict(int)
for f in t_f:
    for g in f["genres"]:
        rem[(g, era(f["decade"]))] += 1
print("\n== films removed per genre x era (main genres)")
mains = ["Drama", "Comedy", "Action", "Adventure", "Crime", "Thriller"]
print(f"{'':<10}" + "".join(f"{e:>10}" for e in ERAS) + f"{'total':>8}")
for g in mains:
    print(f"{g:<10}" + "".join(f"{rem[(g, e)]:>10}" for e in ERAS) + f"{sum(rem[(g, e)] for e in ERAS):>8}")
    for e in ERAS:
        rows.append({"table": "removed_genre_era", "key": f"{g}|{e}", "excluded": rem[(g, e)], "rest": ""})
with open(OUT / "tiny_vs_rest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["table", "key", "excluded", "rest"])
    w.writeheader()
    w.writerows(rows)

# 4. is the skew towards older films larger than chance? (Fisher exact test, before 1990 vs 1990 and later)
from scipy.stats import fisher_exact  # noqa: E402
old = ["pre-1980", "1980s"]
a, b = sum(c_t[e] for e in old), sum(c_r[e] for e in old)
odds, p = fisher_exact([[a, nt - a], [b, nr - b]])
print(f"\nbefore 1990, all ids: {a}/{nt} excluded vs {b}/{nr} rest, odds ratio {odds:.2f}, Fisher p = {p:.4f}")
a2, b2 = sum(ea_t[e] for e in old), sum(ea_r[e] for e in old)
odds2, p2 = fisher_exact([[a2, len(t_f) - a2], [b2, len(r_f) - b2]])
print(f"before 1990, analysis set: {a2}/{len(t_f)} excluded vs {b2}/{len(r_f)} rest, odds ratio {odds2:.2f}, Fisher p = {p2:.4f}")
