"""Task 2 step 3: per-film sanity statistics of the layer split.

Output (numbers only, no script text):
    layer_stats.csv  one row per record (one per imdb id: 2188; duplicated ids keep the longest script)
    outliers.csv     records that are extreme on one statistic (Tukey far fences)

Run:  python3 -I analysis/screenplays/data_check/layer_stats.py
Set IMDB_DATA_DIR if the data is not in data/raw.

"Token" here = a whitespace-separated word. It is only a counting unit for
these statistics, NOT the tokenisation used for Word2Vec (still an open decision).
"""
import csv
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_layers import one_per_id, parse_script, read_records  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("IMDB_DATA_DIR", ROOT / "data" / "raw"))
OUT = Path(__file__).resolve().parent

LAYERS = ["dialogue", "description", "parenthetical", "char", "stage"]  # other split in two
TINY = 500  # a text layer with fewer tokens than this counts as "tiny"

HEADING = re.compile(r"^(\d+[A-Z]?\s+)?(INT|EXT|I/E)\b")
DETACHED = {"n't", "'s", "'ll", "'re", "'ve", "'d", "'m"}
ATTACHED = re.compile(r"\w('s|'ll|'re|'ve|'d|'m|n't)$")


def has_alpha(tok):
    return any(c.isalpha() for c in tok)


def is_upper_word(tok):
    """Fully uppercase word of at least 2 letters (so 'I' and 'A' do not count)."""
    letters = [c for c in tok if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def clean_word(tok):
    return "".join(c for c in tok if c.isalpha())


def film_stats(rec):
    els = parse_script(rec["script"])
    layer = lambda e: "char" if e["tag"] == "character" else "stage" if e["tag"] == "stage_direction" else e["layer"]

    # names of this film = cues that are a single alphabetic word (e.g. WELLES), not roles with "THE"
    names = set()
    for e in els:
        if e["tag"] == "character":
            words = re.sub(r"\(.*?\)", "", e["text"]).split()
            if len(words) == 1 and clean_word(words[0]) == words[0] and len(words[0]) >= 2:
                names.add(words[0].upper())

    s = {"imdb_id": rec["imdb_id"], "movie_name": rec["movie_name"], "split": rec["split"]}
    for L in LAYERS:
        for k in ["elements", "empty", "tokens", "punct_tokens", "upper", "alpha",
                  "detached", "attached", "upper_names", "ci_names"]:
            s[f"{L}_{k}"] = 0
    s["unparsed"] = 0
    s["empty_scenes"] = 0
    s["heading_in_dialogue"] = s["heading_in_description"] = 0
    s["nameonly_in_dialogue"] = s["nameonly_in_description"] = 0
    s["char_no_letters"] = s["char_has_lowercase"] = s["stage_not_heading"] = 0

    for e in els:
        if e["layer"] == "structure":
            s["empty_scenes"] += 1
            continue
        if e["layer"] == "unparsed":
            s["unparsed"] += 1
            continue
        L = layer(e)
        s[f"{L}_elements"] += 1
        text = e["text"]
        if not text:
            s[f"{L}_empty"] += 1
            continue
        toks = text.split()
        s[f"{L}_tokens"] += len(toks)
        for t in toks:
            if not any(c.isalnum() for c in t):
                s[f"{L}_punct_tokens"] += 1
            if has_alpha(t):
                s[f"{L}_alpha"] += 1
                if is_upper_word(t):
                    s[f"{L}_upper"] += 1
                    if clean_word(t) in names:
                        s[f"{L}_upper_names"] += 1
                if clean_word(t).upper() in names:
                    s[f"{L}_ci_names"] += 1
            if t.lower() in DETACHED:
                s[f"{L}_detached"] += 1
            elif ATTACHED.search(t):
                s[f"{L}_attached"] += 1
        # leakage proxies
        if L in ("dialogue", "description"):
            if HEADING.match(text):
                s[f"heading_in_{L}"] += 1
            if clean_word(text).upper() in names and len(toks) <= 2:
                s[f"nameonly_in_{L}"] += 1
        if L == "char":
            if not has_alpha(text):
                s["char_no_letters"] += 1
            elif any(c.islower() for c in text):
                s["char_has_lowercase"] += 1
        if L == "stage" and not HEADING.match(text):
            s["stage_not_heading"] += 1

    total = sum(s[f"{L}_tokens"] for L in LAYERS)
    for L in LAYERS:
        s[f"{L}_share_total"] = round(s[f"{L}_tokens"] / total, 4) if total else 0
        s[f"{L}_upper_share"] = round(s[f"{L}_upper"] / s[f"{L}_alpha"], 4) if s[f"{L}_alpha"] else 0
        s[f"{L}_punct_share"] = round(s[f"{L}_punct_tokens"] / s[f"{L}_tokens"], 4) if s[f"{L}_tokens"] else 0
    dd = s["dialogue_tokens"] + s["description_tokens"]
    s["dialogue_share_of_dd"] = round(s["dialogue_tokens"] / dd, 4) if dd else 0
    s["dialogue_turns"] = s["dialogue_elements"] - s["dialogue_empty"]
    s["tiny_layer"] = int(s["dialogue_tokens"] < TINY or s["description_tokens"] < TINY)
    return s


def quartiles(xs):
    q1, q2, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return q1, q2, q3


def main():
    rows = [film_stats(r) for r in one_per_id(read_records(DATA))]
    with open(OUT / "layer_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    print(f"records: {n}  unique ids: {len({r['imdb_id'] for r in rows})}")

    def summ(name, key):
        xs = [r[key] for r in rows]
        q1, q2, q3 = quartiles(xs)
        print(f"{name:<44} min {min(xs):>9.4g}  Q1 {q1:>9.4g}  median {q2:>9.4g}  Q3 {q3:>9.4g}  max {max(xs):>9.4g}")

    print("\n== size of each layer (tokens per film)")
    for L in LAYERS:
        summ(f"{L} tokens", f"{L}_tokens")
    print("\n== share of all tokens (each layer)")
    for L in LAYERS:
        summ(f"{L} share of total", f"{L}_share_total")
    summ("dialogue share of (dialogue+description)", "dialogue_share_of_dd")
    summ("dialogue turns (non-empty dialogue elements)", "dialogue_turns")
    print("\n== share of fully uppercase words (>=2 letters) among alphabetic tokens")
    for L in LAYERS:
        summ(f"{L} uppercase share", f"{L}_upper_share")

    print("\n== counts over all films")
    tot = lambda k: sum(r[k] for r in rows)
    print("unparsed lines:", tot("unparsed"), " empty <scene /> markers:", tot("empty_scenes"),
          " films with some:", sum(r["empty_scenes"] > 0 for r in rows))
    for L in LAYERS:
        print(f"{L:<14} elements {tot(L+'_elements'):>8}  empty {tot(L+'_empty'):>6}")
    print("films with a tiny layer (dialogue or description <", TINY, "tokens):", tot("tiny_layer"))
    print("films with zero dialogue tokens:", sum(r["dialogue_tokens"] == 0 for r in rows),
          " zero description tokens:", sum(r["description_tokens"] == 0 for r in rows))

    print("\n== possible leakage (proxies, see README for what they miss)")
    for k in ["heading_in_dialogue", "heading_in_description", "nameonly_in_dialogue",
              "nameonly_in_description", "char_no_letters", "char_has_lowercase", "stage_not_heading"]:
        print(f"{k:<28} total {tot(k):>7}  films affected {sum(r[k] > 0 for r in rows):>5}")
    for L in ["dialogue", "description"]:
        print(f"{L}: uppercase film-name tokens (CAPS name from own cues) {tot(L+'_upper_names')}"
              f" of {tot(L+'_alpha')} alphabetic tokens;"
              f" any-case name tokens {tot(L+'_ci_names')}")

    print("\n== tokenisation per layer (all films pooled)")
    for L in ["dialogue", "description", "parenthetical"]:
        t = tot(f"{L}_tokens")
        d, a = tot(f"{L}_detached"), tot(f"{L}_attached")
        print(f"{L:<14} tokens {t:>9}  punctuation-only {tot(L+'_punct_tokens')/t:6.2%}"
              f"  contractions detached {d:>7}  attached {a:>7}  share detached {d/(d+a) if d+a else 0:6.2%}")
    for L in ["dialogue", "description"]:
        xs = [r[f"{L}_punct_share"] for r in rows if r[f"{L}_tokens"]]
        q1, q2, q3 = quartiles(xs)
        print(f"{L} punctuation-only share per film: Q1 {q1:.3f} median {q2:.3f} Q3 {q3:.3f}")

    # outliers: Tukey far fences (3 x IQR) on a few statistics
    metrics = ["dialogue_tokens", "description_tokens", "dialogue_share_of_dd",
               "dialogue_upper_share", "description_upper_share", "dialogue_punct_share",
               "description_punct_share", "char_share_total", "stage_share_total"]
    out = []
    for m in metrics:
        q1, _, q3 = quartiles([r[m] for r in rows])
        lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
        for r in rows:
            if r[m] < lo or r[m] > hi:
                out.append({"metric": m, "imdb_id": r["imdb_id"], "movie_name": r["movie_name"],
                            "value": r[m], "fence_low": round(lo, 4), "fence_high": round(hi, 4)})
    for r in rows:
        if r["tiny_layer"]:
            out.append({"metric": "tiny_layer", "imdb_id": r["imdb_id"], "movie_name": r["movie_name"],
                        "value": min(r["dialogue_tokens"], r["description_tokens"]),
                        "fence_low": TINY, "fence_high": ""})
    with open(OUT / "outliers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "imdb_id", "movie_name", "value", "fence_low", "fence_high"])
        w.writeheader()
        w.writerows(out)
    print(f"\n== outliers (3 x IQR fences): {len(out)} rows in outliers.csv")
    from collections import Counter
    by = Counter(o["metric"] for o in out)
    for m, c in by.most_common():
        worst = sorted((o for o in out if o["metric"] == m), key=lambda o: -abs(o["value"] - (o["fence_low"] if o["value"] < o["fence_low"] else o["fence_high"] or o["fence_low"])))[:3]
        print(f"  {m:<26}{c:>4}   e.g. " + "; ".join(f"{o['movie_name']} ({o['value']})" for o in worst))


if __name__ == "__main__":
    main()
