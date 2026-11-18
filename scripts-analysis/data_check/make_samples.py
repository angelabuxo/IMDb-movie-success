"""Task 2 step 4: build the manual-check sheets.

Draws (fixed seed, stored in sample_films.csv):
  A. main sample: 25 films from the 1104 final films, 10 before 1990 and 15 from 1990 on.
     Per film 10 points: 4 dialogue, 4 description, 2 "other" elements
     (parenthetical, stage_direction or character, drawn uniformly from those three together).
  B. red-flag sheet (exploratory, no error rate): up to 5 automatically suspicious
     elements per film for the same 25 films.
  C. small separate sample of the 30 excluded films: 5 dialogue-poor and 3 description-poor
     films, 10 random elements each from the other layer. Never mixed into A.

Files written
  committed (ids, tags and EMPTY columns, no script text):
     sample_films.csv, sheet_main_blank.csv, sheet_redflags_blank.csv, sheet_excluded_blank.csv
  private/ (gitignored, with script text, at most 15 words per snippet):
     sheet_main.csv, sheet_redflags.csv, sheet_excluded.csv

Run:  python3 -I scripts-analysis/data_check/make_samples.py
The text is RAW (straight from the tags, no preprocessing).
"""
import csv
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "genre_counts"))
from genre_counts import DATA, load_films  # noqa: E402
from parse_layers import one_per_id, parse_script, read_records  # noqa: E402

SEED = 2805
N_OLD, N_NEW = 10, 15          # films before 1990 / from 1990
PER_LAYER = {"dialogue": 4, "description": 4, "other": 2}
REDFLAGS_PER_FILM = 5
EXCL_DIALOGUE_POOR, EXCL_DESCRIPTION_POOR, EXCL_POINTS = 5, 3, 10
MAX_WORDS = 15
PRIVATE = HERE.parent / "private"
PRIVATE.mkdir(exist_ok=True)

OTHER_TAGS = {"parenthetical", "stage_direction", "character"}
HEADING_ANYWHERE = re.compile(r"\b(INT|EXT)\.")
NAME_COLON = re.compile(r"^[A-Z][A-Z .'-]{1,30}:")
CAPS_RUN = re.compile(r"(?:\b[A-Z][A-Z'.,-]+\b\s+){4,}[A-Z][A-Z'.,-]+\b")  # 5+ uppercase words in a row
QUOTE = re.compile(r"[\"“”]")


def snippet(text, start_char=0):
    """At most 15 words, from 5 words before start_char (so a match is visible)."""
    words = text.split()
    idx = len(text[:start_char].split())
    first = max(0, idx - 5) if start_char else 0
    s = " ".join(words[first:first + MAX_WORDS])
    return ("... " if first else "") + s + (" ..." if first + MAX_WORDS < len(words) else "")


def neighbours(els, i):
    prev = els[i - 1]["tag"] if i > 0 else "(start)"
    nxt = els[i + 1]["tag"] if i + 1 < len(els) else "(end)"
    return prev, nxt


def points(els, rng, wanted):
    """wanted: {group: n}. Returns list of (group, element index)."""
    pools = {"dialogue": [], "description": [], "other": []}
    for i, e in enumerate(els):
        if not e["text"]:
            continue
        if e["tag"] == "dialogue":
            pools["dialogue"].append(i)
        elif e["tag"] == "scene_description":
            pools["description"].append(i)
        elif e["tag"] in OTHER_TAGS:
            pools["other"].append(i)
    out = []
    for g, n in wanted.items():
        out += [(g, i) for i in sorted(rng.sample(pools[g], n))]
    return out


def write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    rng = random.Random(SEED)
    records = {r["imdb_id"]: r for r in one_per_id(read_records(DATA))}
    films = sorted(load_films(), key=lambda f: f["id"])   # the final 1104, in a fixed order
    old = [f for f in films if f["decade"] < 1990]
    new = [f for f in films if f["decade"] >= 1990]
    chosen = [(f, "pre-1990") for f in rng.sample(old, N_OLD)] + [(f, "1990+") for f in rng.sample(new, N_NEW)]
    print(f"final films: {len(films)} ({len(old)} before 1990, {len(new)} from 1990)")

    write(HERE / "sample_films.csv",
          [{"imdb_id": f["id"], "movie_name": records[f["id"]]["movie_name"], "era_group": g, "seed": SEED}
           for f, g in chosen], ["imdb_id", "movie_name", "era_group", "seed"])

    main_rows, flag_rows = [], []
    flag_counts = {"quote_in_description": 0, "name_colon_in_description": 0,
                   "heading_in_dialogue": 0, "caps_run_in_dialogue": 0}
    for f, group in chosen:
        rec = records[f["id"]]
        els = parse_script(rec["script"])
        base = {"imdb_id": f["id"], "movie_name": rec["movie_name"], "era_group": group}
        for n, (layer, i) in enumerate(points(els, rng, PER_LAYER), 1):
            e = els[i]
            prev, nxt = neighbours(els, i)
            main_rows.append({**base, "point_id": f"{f['id']}_{n:02d}", "layer_parser": e["layer"],
                              "tag": e["tag"], "element_index": i, "prev_tag": prev, "next_tag": nxt,
                              "snippet": snippet(e["text"]), "correct_yes_no": "", "error_type": "", "comment": ""})
        # red flags: every suspicious element of the film, then up to 5 at random
        found = []
        for i, e in enumerate(els):
            t = e["text"]
            if not t:
                continue
            hits = []
            if e["tag"] == "scene_description":
                m = QUOTE.search(t)
                if m:
                    hits.append(("quote_in_description", m.start()))
                m = NAME_COLON.search(t)
                if m:
                    hits.append(("name_colon_in_description", 0))
            if e["tag"] == "dialogue":
                m = HEADING_ANYWHERE.search(t)
                if m:
                    hits.append(("heading_in_dialogue", m.start()))
                m = CAPS_RUN.search(t)
                if m:
                    hits.append(("caps_run_in_dialogue", m.start()))
            found += [(rule, i, pos) for rule, pos in hits]
        for rule, *_ in found:
            flag_counts[rule] += 1
        for n, (rule, i, pos) in enumerate(sorted(rng.sample(found, min(REDFLAGS_PER_FILM, len(found))), key=lambda x: x[1]), 1):
            e = els[i]
            prev, nxt = neighbours(els, i)
            flag_rows.append({**base, "flag_id": f"{f['id']}_f{n:02d}", "rule": rule, "layer_parser": e["layer"],
                              "tag": e["tag"], "element_index": i, "prev_tag": prev, "next_tag": nxt,
                              "snippet": snippet(e["text"], pos), "is_real_problem_yes_no": "", "comment": ""})

    # C. excluded films
    tiny = list(csv.DictReader(open(HERE / "tiny_layers.csv")))
    d_poor = sorted(r["imdb_id"] for r in tiny if r["tiny_layer"] == "dialogue")
    s_poor = sorted(r["imdb_id"] for r in tiny if r["tiny_layer"] == "description")
    picks = [(i, "dialogue-poor", "dialogue") for i in rng.sample(d_poor, EXCL_DIALOGUE_POOR)] + \
            [(i, "description-poor", "description") for i in rng.sample(s_poor, EXCL_DESCRIPTION_POOR)]
    excl_rows = []
    for imd, kind, poor in picks:
        rec = records[imd]
        els = parse_script(rec["script"])
        # sample the layer that is NOT the poor one: description for dialogue-poor films, dialogue for description-poor
        tag = "scene_description" if poor == "dialogue" else "dialogue"
        pool = [i for i, e in enumerate(els) if e["tag"] == tag and e["text"]]
        for n, i in enumerate(sorted(rng.sample(pool, EXCL_POINTS)), 1):
            prev, nxt = neighbours(els, i)
            excl_rows.append({"imdb_id": imd, "movie_name": rec["movie_name"], "film_type": kind,
                              "point_id": f"{imd}_x{n:02d}", "layer_parser": els[i]["layer"], "tag": tag,
                              "element_index": i, "prev_tag": prev, "next_tag": nxt, "snippet": snippet(els[i]["text"]),
                              "dialogue_inside_description_yes_no_unclear": "",
                              "description_inside_dialogue_yes_no_unclear": "", "comment": ""})

    f_main = ["point_id", "imdb_id", "movie_name", "era_group", "layer_parser", "tag", "element_index", "prev_tag",
              "next_tag", "snippet", "correct_yes_no", "error_type", "comment"]
    f_flag = ["flag_id", "imdb_id", "movie_name", "era_group", "rule", "layer_parser", "tag", "element_index",
              "prev_tag", "next_tag", "snippet", "is_real_problem_yes_no", "comment"]
    f_excl = ["point_id", "imdb_id", "movie_name", "film_type", "layer_parser", "tag", "element_index", "prev_tag",
              "next_tag", "snippet", "dialogue_inside_description_yes_no_unclear",
              "description_inside_dialogue_yes_no_unclear", "comment"]
    for name, rows, fields in [("main", main_rows, f_main), ("redflags", flag_rows, f_flag), ("excluded", excl_rows, f_excl)]:
        write(PRIVATE / f"sheet_{name}.csv", rows, fields)                      # with text, private
        write(HERE / f"sheet_{name}_blank.csv", rows, [c for c in fields if c != "snippet"])  # no text
    print(f"main sheet: {len(main_rows)} points; red flags: {len(flag_rows)} rows; excluded sample: {len(excl_rows)} points")
    print("suspicious elements found in the 25 films (before choosing 5 per film):", flag_counts)
    print("excluded films sampled:", [(r["movie_name"], r["film_type"]) for r in {x['imdb_id']: x for x in excl_rows}.values()])


if __name__ == "__main__":
    main()
