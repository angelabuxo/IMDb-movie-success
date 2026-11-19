"""Prevalence of two failure modes found in the manual check, over the final set (1598 films).

  1. Scene headings (INT./EXT.) inside dialogue elements: share of dialogue elements and of films.
  2. Corrupted text (OCR-like): per film, the share of whitespace tokens of the dialogue and description layers
     that contain a symbol outside letters (any language), digits and ordinary punctuation (e.g. ~ \\ ^ | = * # @ _ { } [ ] < >).

Uses only structure (tags, characters), never hit/flop, budget or gross. Writes numbers only:
    prevalence_by_film.csv   one row per film of the final set
    prevalence_summary.csv   distribution and the films far from the rest

The two films the manual check called "text seems encrypted" (tt0097368, tt0430912) are shown as a check of the
measure: if they do not stand out, the measure does not capture what we saw.

Run:  python3 -I scripts-analysis/data_check/prevalence.py
"""
import csv
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "genre_counts"))
from genre_counts import DATA  # noqa: E402
from final_set import final_films  # noqa: E402
from parse_layers import one_per_id, parse_script, read_records  # noqa: E402

HEADING = re.compile(r"\b(INT|EXT)\.")
ODD = re.compile(r"[^\w\s.,;:!?'\"“”‘’()\-–—/&%$…]|_")  # a symbol that is not ordinary prose (accented letters are fine)
KNOWN_GARBLED = ["tt0097368", "tt0430912"]


def main():
    final = {f["id"]: f for f in final_films()}
    records = {r["imdb_id"]: r for r in one_per_id(read_records(DATA)) if r["imdb_id"] in final}
    rows = []
    for fid, rec in records.items():
        els = parse_script(rec["script"])
        dlg = [e["text"] for e in els if e["layer"] == "dialogue" and e["text"]]
        desc = [e["text"] for e in els if e["layer"] == "description" and e["text"]]
        head = sum(1 for t in dlg if HEADING.search(t))
        toks = [w for t in dlg + desc for w in t.split()]
        odd = sum(1 for w in toks if ODD.search(w))
        rows.append({"imdb_id": fid, "movie_name": rec["movie_name"], "year": final[fid]["year"],
                     "dialogue_elements": len(dlg), "dialogue_with_heading": head,
                     "tokens": len(toks), "odd_tokens": odd,
                     "odd_share": round(odd / len(toks), 5) if toks else 0})
    with open(HERE / "prevalence_by_film.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["odd_share"]))

    n = len(rows)
    tot_d = sum(r["dialogue_elements"] for r in rows)
    tot_h = sum(r["dialogue_with_heading"] for r in rows)
    films_h = sum(1 for r in rows if r["dialogue_with_heading"] > 0)
    share_h = sorted(r["dialogue_with_heading"] / r["dialogue_elements"] for r in rows if r["dialogue_elements"])
    odd = sorted(r["odd_share"] for r in rows)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    med, p95, p99 = q(odd, 0.5), q(odd, 0.95), q(odd, 0.99)
    q1, q3 = q(odd, 0.25), q(odd, 0.75)
    fence = q3 + 3 * (q3 - q1)
    far = [r for r in rows if r["odd_share"] > fence]
    ranks = {r["imdb_id"]: i + 1 for i, r in enumerate(sorted(rows, key=lambda r: -r["odd_share"]))}

    out = [
        ("films_in_final_set", n),
        ("dialogue_elements_total", tot_d),
        ("dialogue_elements_with_INT_EXT", tot_h),
        ("share_of_dialogue_elements_with_heading", round(tot_h / tot_d, 5)),
        ("films_with_at_least_one_heading_in_dialogue", films_h),
        ("share_of_films_with_a_heading_in_dialogue", round(films_h / n, 3)),
        ("films_where_headings_exceed_1pct_of_dialogue", sum(1 for s in share_h if s > 0.01)),
        ("odd_share_median", med), ("odd_share_p95", p95), ("odd_share_p99", p99),
        ("odd_share_max", odd[-1]), ("tukey_far_fence_q3+3iqr", round(fence, 5)),
        ("films_above_far_fence", len(far)),
    ]
    for fid in KNOWN_GARBLED:
        r = next((x for x in rows if x["imdb_id"] == fid), None)
        out.append((f"check_{fid}_odd_share_and_rank", f"{r['odd_share']} rank {ranks[fid]} of {n}" if r else "not in final set"))
    with open(HERE / "prevalence_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measure", "value"])
        w.writerows(out)
        for r in sorted(far, key=lambda r: -r["odd_share"]):
            w.writerow([f"far_{r['imdb_id']}_{r['movie_name']}", r["odd_share"]])
    for k, v in out:
        print(k, v)
    print("films above far fence:", [(r["imdb_id"], r["movie_name"], r["odd_share"]) for r in sorted(far, key=lambda r: -r["odd_share"])][:25])


if __name__ == "__main__":
    main()
