#!/usr/bin/env python3
"""Build the shared analysis-set tables: one row per film, one row per film-genre pair, and a data dictionary.

Run from the data folder (data/raw), like the other scripts in scripts/imdb:

    cd data/raw && ../../.venv/bin/python ../../scripts/imdb/build_analysis_set.py [--bundle]

Reads   features/movies_features_10k.csv       (all 12,673 films with 10,000+ IMDb votes)
        scripts/moviesum/{train,val,test}.jsonl (only to count tokens per layer; no script text is kept)
Writes  features/film_analysis_set.csv         one row per film, every film kept, with flags and exclusion reasons
        features/film_genres.csv               one row per film-genre pair (a film appears once per genre)
        features/film_analysis_set.md          the data dictionary, next to the tables
        data/ANALYSIS_SET.md                   the same dictionary, readable on GitHub
        data/bundles/analysis-set-<date>.tar.zst   with --bundle: the three files above, as the repo's other bundles

Rows are never deleted: a film that is excluded says why in a column.

The shared cleaning rule is analysis/clean_rule.py, CALLED here and not written out again (the repo's tests forbid a
second copy). `clean_team_rule` is its result. `usable_money` is that rule and no net_gross flag and no flag on the
budget. `usable_ratio_same_currency` is `usable_money` with the US-dollar condition replaced by "budget and gross are in
the same currency" (the rule's other conditions are taken from its own steps, not rewritten). The money figures are the Wikipedia infobox figures from parse_infobox.py; nothing from TMDB or Wikidata is used.

What it captures: which films have budget and gross figures that are comparable theatrical amounts, and which films
have usable screenplay layers.
What it misses: it trusts the parser's flags, so a film the parser misread can be wrongly kept or dropped, and gross
over budget still isn't profit.

The output is deterministic: same inputs, same bytes (rows keep the input order, the bundle has fixed metadata).
"""
import csv
import io
import json
import math
import re
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))   # repo root, for the shared clean rule
import pandas as pd  # noqa: E402
from analysis.clean_rule import is_clean, steps, why_not_clean  # noqa: E402

BUNDLE_DATE = "2026-11-19"
MOVIES = "features/movies_features_10k.csv"
MOVIESUM = [f"scripts/moviesum/{s}.jsonl" for s in ("train", "val", "test")]
TINY_TOKENS = 500   # a film with fewer tokens than this in the dialogue or the description layer is excluded
LAYER_LINE = re.compile(r"<(dialogue|scene_description)>(.*)</\1>")

COLUMNS = ["imdb_id", "title", "year", "decade", "era_bin", "tier", "genres", "n_genres",
           "budget", "budget_cur", "gross", "gross_cur", "budget_flag", "gross_flag", "ln_gross_over_budget",
           "net_gross_flag", "clean_team_rule", "usable_money", "money_exclude_reason", "usable_ratio_same_currency",
           "in_moviesum", "in_common_set", "common_set_exclude_reason"]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def era_bin(year):
    if year is None:
        return ""
    return "before 1980" if year < 1980 else f"{year // 10 * 10}s"


def moviesum_layers():
    """{imdb_id: (dialogue tokens, description tokens)} for the longest script of each id.

    Tokens are whitespace-separated words of the non-empty <dialogue> and <scene_description> elements.
    A duplicated id keeps the record with the most characters (the first one on a tie).
    """
    best = {}
    for path in MOVIESUM:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                tok = {"dialogue": 0, "scene_description": 0}
                for l in r["script"].split("\n"):
                    m = LAYER_LINE.fullmatch(l.strip())
                    if m:
                        tok[m.group(1)] += len(m.group(2).split())
                old = best.get(r["imdb_id"])
                if old is None or len(r["script"]) > old[0]:
                    best[r["imdb_id"]] = (len(r["script"]), tok["dialogue"], tok["scene_description"])
    return {k: (v[1], v[2]) for k, v in best.items()}


def build():
    with open(MOVIES, newline="") as f:
        movies = list(csv.DictReader(f))
    layers = moviesum_layers()

    # the shared rule's result and first reason, straight from analysis/clean_rule.py
    df = pd.DataFrame(movies)
    first = why_not_clean(df).tolist()
    # every condition of the rule except the US-dollar one, taken from the rule's own steps
    kept = [mask for reason, mask in steps(df) if "USD" not in reason]
    assert len(kept) == len(steps(df)) - 1, "expected exactly one US-dollar step in analysis/clean_rule.py"
    others_pass = pd.concat(kept, axis=1).all(axis=1).tolist()
    rows, genre_rows = [], []
    for r, why, others in zip(movies, first, others_pass):
        team = is_clean(r)
        assert team == (why == "clean"), r["tconst"]      # the two entry points of the rule agree
        year = int(num(r["y"])) if num(r["y"]) is not None else None
        genres = [g for g in (r["genres"] or "").split(",") if g and g != "\\N"]
        b, g = num(r["b"]), num(r["g"])
        net_gross = "net_gross" in (r["budget_flag"] or "") or "net_gross" in (r["gross_flag"] or "")
        budget_flag = bool(r["budget_flag"])
        usable = team and not net_gross and not budget_flag
        if why != "clean":
            money_why = why
        elif net_gross:
            money_why = "net_gross flag"
        elif budget_flag:
            money_why = "budget carries an infobox flag"
        else:
            money_why = ""
        same_currency = bool(r["budget_cur"]) and r["budget_cur"] == r["gross_cur"]
        ratio_ok = others and same_currency and not net_gross and not budget_flag
        assert not usable or ratio_ok, r["tconst"]   # usable_money implies usable_ratio_same_currency
        in_ms = r["tconst"] in layers
        tiny = in_ms and min(layers[r["tconst"]]) < TINY_TOKENS
        if not in_ms:
            common_why = "no MovieSum script"
        elif not usable:
            common_why = "money: " + money_why
        elif tiny:
            common_why = f"dialogue or description layer under {TINY_TOKENS} tokens"
        else:
            common_why = ""
        ratio = ""
        if b and g and b > 0 and g > 0 and r["budget_cur"] == r["gross_cur"]:
            ratio = f"{math.log(g / b):.6f}"
        rows.append({
            "imdb_id": r["tconst"], "title": r["primaryTitle"], "year": year if year is not None else "",
            "decade": year // 10 * 10 if year is not None else "", "era_bin": era_bin(year), "tier": r["tier"],
            "genres": r["genres"] if genres else "", "n_genres": len(genres),
            "budget": r["b"], "budget_cur": r["budget_cur"], "gross": r["g"], "gross_cur": r["gross_cur"],
            "budget_flag": r["budget_flag"], "gross_flag": r["gross_flag"], "ln_gross_over_budget": ratio,
            "net_gross_flag": net_gross, "clean_team_rule": team, "usable_money": usable,
            "money_exclude_reason": money_why, "usable_ratio_same_currency": ratio_ok, "in_moviesum": in_ms, "in_common_set": common_why == "",
            "common_set_exclude_reason": common_why})
        genre_rows += [{"imdb_id": r["tconst"], "genre": x} for x in genres]
    return rows, genre_rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def table(counter, order=None):
    keys = order or [k for k, _ in counter.most_common()]
    return "\n".join(f"| {k or '(none: the film passes)'} | {counter[k]:,} |" for k in keys)


def dictionary(rows):
    n = len(rows)
    tier = Counter(r["tier"] for r in rows)
    money = Counter(r["money_exclude_reason"] for r in rows)
    common = Counter(r["common_set_exclude_reason"] for r in rows)
    team = [r for r in rows if r["clean_team_rule"]]
    team_not_usable = [r for r in team if not r["usable_money"]]
    final = [r for r in rows if r["in_common_set"]]
    ft = Counter(r["tier"] for r in final)
    ng = sum(1 for r in team if r["net_gross_flag"])
    bf = sum(1 for r in team if r["budget_flag"] and not r["net_gross_flag"])
    by_tier = {t: sum(1 for r in team if r["tier"] == t) for t in tier}
    nd = sum(1 for r in team_not_usable if r["tier"] == "100k+")
    ratio = [r for r in rows if r["usable_ratio_same_currency"]]
    extra = [r for r in ratio if not r["usable_money"]]
    extra_cur = Counter(r["budget_cur"] for r in extra)
    extra_tier = Counter(r["tier"] for r in extra)
    ratio_script = sum(1 for r in ratio if r["in_moviesum"])
    nonusd = [r for r in rows if r["money_exclude_reason"] == "budget or gross not in USD"]
    nonusd_same = sum(1 for r in nonusd if r["usable_ratio_same_currency"])
    sc = sum(1 for r in rows if r["in_moviesum"])
    return f"""# Analysis set: one table of films with the cleaning flags

Built by `scripts/imdb/build_analysis_set.py` from `features/movies_features_10k.csv` (all {n:,} films with 10,000+ IMDb votes)
and the MovieSum screenplays (only to count tokens per layer; no script text is in these files).
Three files: `film_analysis_set.csv` (one row per film), `film_genres.csv` (one row per film-genre pair) and this dictionary.
**No row is deleted.** A film that is excluded says why in a column, so every question can filter its own way.

## Which flag to use

- `usable_money` is the **primary cleaning rule** for money. It is the repo's shared clean rule
  (`analysis/clean_rule.py`, called and not rewritten) **and** no `net_gross` flag **and** no flag on the budget.
- `in_common_set` is the set on which all three questions are compared head to head. Each question can use the wider
  set for its own analysis (for example every film with `usable_money`, with or without a script).
- `usable_ratio_same_currency` is wider than `usable_money`: it also keeps the films whose budget and gross are in the same non-dollar
  currency (for example euros or rupees). Use it for the **ratio** gross/budget (`ln_gross_over_budget`), which has no unit when both
  figures are in the same currency. Every film with `usable_money` also has `usable_ratio_same_currency`.
- `clean_team_rule` is the result of `analysis/clean_rule.py` alone, kept so the two can be compared.

| Column | Keeps | Does not keep |
|---|---|---|
| `clean_team_rule` | films passing `analysis/clean_rule.py`. The rule does not look at `net_gross` or at flags on the budget, so those films pass | non-dollar films, missing figures, amounts under $1,000, grosses flagged rentals-only, regional or direct-to-video, streaming distributors, films outside 1975 to 2025 |
| `usable_money` | `clean_team_rule`, no `net_gross` flag, no flag on the budget; budget and gross in **US dollars** | non-dollar films, and everything `clean_team_rule` drops |
| `usable_ratio_same_currency` | the same conditions as `usable_money`, but budget and gross in the **same currency** (dollars or not) | films with budget and gross in different currencies, and every other exclusion of `usable_money` |
| `in_common_set` | `usable_money` + a MovieSum script + both layers with at least {TINY_TOKENS} tokens | everything else. Unchanged by `usable_ratio_same_currency` |

## Files

| File | One row per | Rows |
|---|---|---|
| `film_analysis_set.csv` | film | {n:,} |
| `film_genres.csv` | film-genre pair (a film appears once per genre it has) | {sum(r['n_genres'] for r in rows):,} |

## Columns of `film_analysis_set.csv`

| Column | Meaning | Source |
|---|---|---|
| `imdb_id` | IMDb title ID, the key for joining everything | IMDb non-commercial datasets |
| `title` | IMDb primary title | IMDb |
| `year` | release year (IMDb start year) | IMDb |
| `decade` | year rounded down to the decade | derived |
| `era_bin` | before 1980, 1980s, 1990s, 2000s, 2010s, 2020s | derived from `year` |
| `tier` | `100k+` for 100,000+ IMDb votes, `10k-100k` for 10,000 to 99,999 | IMDb votes |
| `genres` | the original IMDb genre string. The genres are in **alphabetical order, not by importance** | IMDb |
| `n_genres` | number of genres | derived |
| `budget`, `gross` | the figures parsed from the Wikipedia infobox, in the currency of `budget_cur` and `gross_cur` (US dollars for every film with `usable_money`) | English Wikipedia infobox (`scripts/imdb/parse_infobox.py`). Not TMDB, not Wikidata |
| `budget_cur`, `gross_cur` | currency code of each figure | the parser |
| `budget_flag`, `gross_flag` | the parser's flags on the figure: `regional`, `rentals`, `net_gross`, `direct_to_video` (blank when none) | the parser |
| `ln_gross_over_budget` | ln(gross / budget). Filled only when both figures are positive and in the same currency. Gross over budget is **not profit**. Use it with `usable_money` | derived |
| `net_gross_flag` | True when the budget or the gross carries a `net_gross` flag | the parser's flags |
| `clean_team_rule` | True when the film passes `analysis/clean_rule.py` | the shared clean rule |
| `usable_money` | `clean_team_rule` and no `net_gross` flag and no flag on the budget. **The primary rule** | see Rules |
| `usable_ratio_same_currency` | `usable_money` with the US-dollar condition replaced by "budget and gross are in the same currency". Use it for the ratio gross/budget only (see Rules) | see Rules |
| `money_exclude_reason` | why `usable_money` is False: the first reason the shared rule reports, then `net_gross flag`, then `budget carries an infobox flag`. Blank when True | derived |
| `in_moviesum` | True when MovieSum has a screenplay for this ID | MovieSum |
| `in_common_set` | True when the film has a script, `usable_money`, and both layers have at least {TINY_TOKENS} tokens | see Rules |
| `common_set_exclude_reason` | why `in_common_set` is False (`no MovieSum script`, `money: <reason>`, `dialogue or description layer under {TINY_TOKENS} tokens`). Blank when True | derived |

`film_genres.csv` has two columns, `imdb_id` and `genre`. A film appears once for each genre it has, so "genre X" means
"film tagged X". Results for different genres are not independent: films are in several genres at once.

Hit and flop labels are **not** in these files: they depend on genre x era cells of the set a question uses. Token counts per layer
and script text are not in them either.

## Rules, with the reasons

1. **The money rule (`usable_money`).** `analysis/clean_rule.py` requires budget and gross in US dollars and at least $1,000,
   drops grosses flagged rentals-only, regional or direct-to-video, drops films from a streaming distributor, and keeps only
   films released 1975 to 2025. On top of that, a film with a `net_gross` flag, or with any flag on the budget, is excluded.
   *Reason:* a gross that is not comparable with a worldwide theatrical gross makes a film look like a flop through measurement,
   not performance. That applies to flagged infobox figures, to films before 1975 (the repo's README says their grosses are mostly
   distributor rentals; we have not checked that claim), and to streaming-distributor films. The rule was chosen on this principle and not by looking at hits or flops.
   **Non-dollar films.** `usable_money` needs US dollars, so it drops every film whose figures are in another currency
   ({sum(1 for r in rows if r['money_exclude_reason'] == 'budget or gross not in USD'):,} films across the whole table, mixed or not). `usable_ratio_same_currency` keeps those whose budget and gross are in the
   *same* currency, because a ratio of two amounts in one currency has no unit. **Budgets in different currencies are not comparable in absolute
   amounts**: an analysis that compares films of similar budget, or uses the budget as a variable, needs US dollars (`usable_money`). The minimum
   amount of 1,000 in the rule is a plain number, so for a non-dollar film it is 1,000 units of its own currency.
2. **The layer rule.** A film is excluded when its dialogue layer or its scene-description layer has fewer than {TINY_TOKENS} tokens
   (whitespace-separated words of the `<dialogue>` and `<scene_description>` elements of the screenplay). It applies to the whole
   analysis, not layer by layer. *Reasons:* a script vector is the mean of the word vectors of a layer, so with no tokens there is no vector;
   the comparison between layers only makes sense on the same films; in the films with almost no tagged dialogue the description layer is about twice as long as
   normal, which suggests the dialogue sits untagged inside it (a hypothesis from counts, not verified by reading); and the rule uses only the structure of the scripts, never the outcome.
   The threshold {TINY_TOKENS} is a first pick, not a tuned value.
3. **Duplicated screenplay IDs.** Twelve IDs appear twice in MovieSum with different versions; the record with the most characters is used.
4. **Universe.** All films of the 10k+ table, not only the 100k+ tier. *Reasons:* group sizes are comfortable;
   selection on IMDb votes is partly selection on success (votes follow popularity), and the 10k tier reduces this bias without removing it;
   and it is the universe the network and text questions already use. The 100k+ films alone are a robustness check (`tier`).
5. **Era bins** are before 1980, 1980s, 1990s, 2000s, 2010s, 2020s. The era effect is the main confounder and hit rates differ strongly between the 1970s and the 1980s.

## Counts (exact, from the build)

- Films: {n:,} (`100k+`: {tier['100k+']:,}, `10k-100k`: {tier['10k-100k']:,}). With a MovieSum script: {sc:,}.
- Pass `clean_team_rule`: {len(team):,} ({by_tier['100k+']:,} at 100k+). Pass `usable_money`: {sum(1 for r in rows if r['usable_money']):,}.
- Pass `usable_ratio_same_currency`: {len(ratio):,} ({sum(1 for r in ratio if r['tier'] == '100k+'):,} at 100k+; {ratio_script:,} with a MovieSum script), which is `usable_money` plus
  {len(extra):,} films in another currency ({", ".join(f"{k} {v}" for k, v in extra_cur.most_common())}; {extra_tier['100k+']} at 100k+, {extra_tier['10k-100k']} at 10k-100k).
  Of the {len(nonusd):,} films that `usable_money` drops for "not in USD", {nonusd_same:,} have budget and gross in the same currency and pass this column; the rest have two different currencies or fail another condition.
- **`in_common_set`: {len(final):,} films** ({ft['100k+']:,} at 100k+, {ft['10k-100k']:,} at 10k-100k).

`money_exclude_reason`, all {n:,} films:

| Reason | Films |
|---|---|
{table(money)}

`common_set_exclude_reason`, all {n:,} films:

| Reason | Films |
|---|---|
{table(common)}

### How the two money rules differ

`usable_money` is the shared clean rule with two extra exclusions, so it never keeps a film that `clean_team_rule` drops.
Among the {len(team):,} films that pass `clean_team_rule`, {len(team_not_usable):,} do not pass `usable_money`
({nd} of them at 100k+): {ng} because of a `net_gross` flag and {bf} because the budget carries another flag.
The repo's old clean rule accepts those films; the new rule does not, for the reason given in Rule 1.
**The paper needs a single rule for all three questions.** `usable_money` is the proposal; the group confirms it.

## Warnings

- **Gross is not profit.** Marketing is not in the budget and cinemas keep part of the gross.
- **Flops are under-represented.** A film needs a budget and a gross in its Wikipedia infobox, which favours films with reported figures.
- **The films removed by the layer rule are concentrated before 1990.** The results apply less to older films.
- **Genre means "film tagged X".** Most films have several genres, so results for different genres are not independent.
- **Selection on votes is partly selection on success.** Films with 10,000+ votes are the films that got an audience.

## Licences and credits

Derived from IMDb's non-commercial datasets (personal and non-commercial use under IMDb's terms), English Wikipedia
(CC BY-SA 4.0; the infobox figures), and MovieSum (CC BY-NC 4.0; Saxena and Keller, Findings of ACL 2024) for `in_moviesum`
and the layer rule. Non-commercial use only. The screenplays are not in these files and remain under their rights holders' copyright.
"""


def check_dictionary(text):
    """Every column and every rule must be described in the dictionary."""
    columns_part = text.split("## Columns of")[1].split("## Rules")[0]
    missing = [c for c in COLUMNS if f"`{c}`" not in columns_part]
    assert not missing, f"columns missing from the dictionary: {missing}"
    for word in ("Rules, with the reasons", "layer rule", "Duplicated screenplay IDs", "Universe", "Era bins", "Non-dollar films"):
        assert word in text, f"rule missing from the dictionary: {word}"


def main():
    bundle = "--bundle" in sys.argv
    rows, genre_rows = build()
    Path("features").mkdir(exist_ok=True)
    write_csv("features/film_analysis_set.csv", rows, COLUMNS)
    write_csv("features/film_genres.csv", genre_rows, ["imdb_id", "genre"])
    text = dictionary(rows)
    check_dictionary(text)
    Path("features/film_analysis_set.md").write_text(text)
    (REPO / "data").mkdir(exist_ok=True)
    (REPO / "data" / "ANALYSIS_SET.md").write_text(text)

    print(f"films {len(rows)}, film-genre pairs {len(genre_rows)}")
    print("money_exclude_reason:", dict(Counter(r["money_exclude_reason"] for r in rows)))
    print("common_set_exclude_reason:", dict(Counter(r["common_set_exclude_reason"] for r in rows)))
    print("in_common_set:", sum(r["in_common_set"] for r in rows))
    print("usable_money:", sum(r["usable_money"] for r in rows), " usable_ratio_same_currency:", sum(r["usable_ratio_same_currency"] for r in rows))

    if bundle:   # fixed metadata, so the same inputs give the same bytes
        stamp = int(datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp())
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for name in ("film_analysis_set.csv", "film_analysis_set.md", "film_genres.csv"):
                data = Path("features", name).read_bytes()
                info = tarfile.TarInfo(f"features/{name}")
                info.size, info.mtime, info.mode = len(data), stamp, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(data))
        packed = subprocess.run(["zstd", "-19", "-q", "-c"], input=buf.getvalue(), check=True, capture_output=True).stdout
        dest = REPO / "data" / "bundles" / f"analysis-set-{BUNDLE_DATE}.tar.zst"
        dest.write_bytes(packed)
        print(f"wrote {dest.relative_to(REPO)} ({len(packed):,} bytes)")


if __name__ == "__main__":
    main()
