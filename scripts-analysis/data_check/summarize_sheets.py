"""Summarise the filled manual-check sheets into numbers (no script text).

Reads private/sheet_main.csv, sheet_redflags.csv and sheet_excluded.csv (filled by hand; Excel may have saved
them with ';' as separator, both are read). Writes data_check/manual_check_summary.csv: one row per count.

Error rates are reported per era group and per layer and are NOT pooled (DECISIONS.md). The 95 % interval is a
Wilson interval that treats the points as independent. They are not: 10 points per film, 25 films. So the
interval is optimistic and should be read as a lower bound on the real uncertainty.

Run:  python3 -I scripts-analysis/data_check/summarize_sheets.py
"""
import csv
import io
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIVATE = HERE.parent / "private"


def load(name):
    text = (PRIVATE / name).read_text(encoding="utf-8-sig")
    first = text.split("\n", 1)[0]
    delim = ";" if first.count(";") > first.count(",") else ","
    return list(csv.DictReader(io.StringIO(text), delimiter=delim))


def wilson(k, n, z=1.96):
    if n == 0:
        return ("", "")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, centre - half), 3), round(min(1, centre + half), 3))


def yn(x):
    return x.strip().lower()


def main():
    rows = []

    def add(sheet, group, layer, k, n, what):
        lo, hi = wilson(k, n)
        rows.append({"sheet": sheet, "group": group, "layer": layer, "k": k, "n": n,
                     "share": round(k / n, 3) if n else "", "ci_low": lo, "ci_high": hi, "what": what})

    main_rows = load("sheet_main.csv")
    cell = defaultdict(lambda: [0, 0])
    for x in main_rows:
        key = (x["era_group"], x["layer_parser"])
        cell[key][1] += 1
        cell[key][0] += yn(x["correct_yes_no"]) == "no"
    for (era, layer), (k, n) in sorted(cell.items()):
        add("main", era, layer, k, n, "points judged wrong by the parser's layer")
    print("main sheet error types:", Counter(x["error_type"].strip() for x in main_rows if yn(x["correct_yes_no"]) == "no"))

    flags = load("sheet_redflags.csv")
    by_rule = defaultdict(Counter)
    for x in flags:
        by_rule[x["rule"]][yn(x["is_real_problem_yes_no"]) or "blank"] += 1
    for rule, c in sorted(by_rule.items()):
        n = sum(c.values())
        add("redflags", "all", rule, c["yes"], n,
            "real problems among suspicious elements (hunting list, NOT an error rate); other answers: " +
            ", ".join(f"{k}={v}" for k, v in sorted(c.items()) if k != "yes"))
    films_with_real = {x["imdb_id"] for x in flags if yn(x["is_real_problem_yes_no"]) == "yes"}
    add("redflags", "all", "films_with_a_real_problem", len(films_with_real), len({x["imdb_id"] for x in flags}),
        "films (of those in the sheet) with at least one real problem")

    excl = load("sheet_excluded.csv")
    col = {"description": "dialogue_inside_description_yes_no_unclear",
           "dialogue": "description_inside_dialogue_yes_no_unclear"}
    per_film = defaultdict(Counter)
    for x in excl:
        ans = yn(x[col[x["layer_parser"]]])
        per_film[(x["film_type"], x["layer_parser"], x["movie_name"])][ans or "blank"] += 1
    for (ftype, layer, name), c in sorted(per_film.items()):
        n = sum(c.values())
        add("excluded_by_film", ftype, layer, c["yes"], n,
            f"{name}: other layer's content found in this layer; no={c['no']}, unclear={c['unclear']}")
    by_type = defaultdict(Counter)
    for (ftype, layer, _), c in per_film.items():
        by_type[(ftype, layer)].update(c)
    for (ftype, layer), c in sorted(by_type.items()):
        n = sum(c.values())
        add("excluded", ftype, layer, c["yes"], n,
            f"points; no={c['no']}, unclear={c['unclear']}; films={sum(1 for k in per_film if k[:2] == (ftype, layer))}")

    out = HERE / "manual_check_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["sheet"], r["group"], r["layer"], f"{r['k']}/{r['n']}", r["ci_low"], r["ci_high"])
    print("wrote", out)


if __name__ == "__main__":
    main()
