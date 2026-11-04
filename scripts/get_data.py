#!/usr/bin/env python3
"""Download the project's datasets into a git-ignored folder. Standard library only.

  python3 scripts/get_data.py --list            show what can be fetched
  python3 scripts/get_data.py --status          show what's already here
  python3 scripts/get_data.py imdb moviesum     fetch those datasets
  python3 scripts/get_data.py --all             fetch all files and Wikidata tables
  python3 scripts/get_data.py wikidata          fetch only the Wikidata tables
  python3 scripts/get_data.py lists             build the movie lists from IMDb
  python3 scripts/get_data.py wikipedia         pull articles (about an hour, resumes)

Files already present are skipped. Partial downloads never use the final name.
"""
import argparse, csv, gzip, json, os, shutil, subprocess, sys, time
import urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = "DTU-02805-course-project (https://github.com/angelabuxo/IMDb-movie-success)"
MANIFEST = json.loads((ROOT / "data" / "datasets.json").read_text())


def fetch(url, dest):
    if dest.exists() and not dest.with_name(dest.name + ".aria2").exists():
        print(f"  have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get  {dest.name}", flush=True)
    if shutil.which("aria2c"):
        subprocess.run(["aria2c", "-x", "8", "-s", "8", "-k", "1M", "-c", "--auto-file-renaming=false",
                        "--console-log-level=warn", "--summary-interval=0", f"--user-agent={UA}",
                        "-d", str(dest.parent), "-o", dest.name, url], check=True)
        return
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
        expected = r.headers.get("Content-Length")
        encoded = r.headers.get("Content-Encoding")
    if expected and not encoded and part.stat().st_size != int(expected):
        raise SystemExit(f"{dest.name}: got {part.stat().st_size} bytes, expected {expected}. Run again.")
    part.rename(dest)


def get_dataset(name, data):
    d = MANIFEST["datasets"][name]
    print(f"{name}: {d['about']} ({d['size']})\n  terms: {d['license']}")
    for f in d["files"]:
        fetch(f["url"], data / f["path"])


def get_wikidata(data):
    w = MANIFEST["wikidata"]
    print(f"wikidata: {w['about']}\n  terms: {w['license']}")
    for path, query in w["queries"].items():
        dest = data / path
        if dest.exists():
            print(f"  have {dest.name}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  query {dest.name}", flush=True)
        body = urllib.parse.urlencode({"query": f"{w['prefixes']} {query}"}).encode()
        req = urllib.request.Request(w["endpoint"], data=body, headers={
            "User-Agent": UA, "Accept": "text/tab-separated-values"})
        part = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(req, timeout=300) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        part.rename(dest)
        time.sleep(2)


def build_lists(data):
    """Write the non-adult movie lists the Wikipedia pull needs: 100k+ and 10k+ votes."""
    ratings = {}
    with gzip.open(data / "raw/title.ratings.tsv.gz", "rt", encoding="utf8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if int(r["numVotes"]) >= 10000:
                ratings[r["tconst"]] = r
    (data / "wiki").mkdir(parents=True, exist_ok=True)
    head = ["tconst", "primaryTitle", "startYear", "genres", "votes", "averageRating"]
    with open(data / "wiki/movies_10k.csv", "w", newline="") as a, open(data / "wiki/movies_100k.csv", "w", newline="") as b:
        wa, wb = csv.writer(a), csv.writer(b)
        wa.writerow(head), wb.writerow(head)
        n10 = n100 = 0
        with gzip.open(data / "raw/title.basics.tsv.gz", "rt", encoding="utf8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                t = ratings.get(r["tconst"])
                if not t or r["titleType"] != "movie" or r["isAdult"] != "0":
                    continue
                clean = lambda v: "" if v == "\\N" else v
                row = [r["tconst"], r["primaryTitle"], clean(r["startYear"]), clean(r["genres"]), t["numVotes"], t["averageRating"]]
                wa.writerow(row)
                n10 += 1
                if int(t["numVotes"]) >= 100000:
                    wb.writerow(row)
                    n100 += 1
    print(f"lists: {n10} movies with 10k+ votes, {n100} with 100k+")


def pull_wikipedia(data):
    if not (data / "wiki/wd_tt_enwiki.tsv").exists():
        get_wikidata(data)
    if not (data / "wiki/movies_10k.csv").exists():
        build_lists(data)
    print("wikipedia: text is CC BY-SA. This takes about an hour and can resume.", flush=True)
    for script in ("pull_wiki.py", "pull_wiki_10k.py"):
        subprocess.run([sys.executable, "-I", str(ROOT / "scripts/imdb" / script)], cwd=data, check=True)


def status(data):
    print(f"data folder: {data}")
    for name, d in MANIFEST["datasets"].items():
        have = sum((data / f["path"]).exists() and not (data / (f["path"] + ".aria2")).exists() for f in d["files"])
        print(f"  {name}: {have}/{len(d['files'])} files")
    q = MANIFEST["wikidata"]["queries"]
    print(f"  wikidata: {sum((data / p).exists() for p in q)}/{len(q)} tables")
    print(f"  lists: {'yes' if (data / 'wiki/movies_10k.csv').exists() else 'no'}")
    for p in ("wiki/articles.jsonl", "wiki/articles_10k.jsonl"):
        f = data / p
        print(f"  {p}: {sum(1 for _ in open(f, 'rb')) if f.exists() else 0} pages")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="dataset names, or wikidata, lists, wikipedia")
    ap.add_argument("--all", action="store_true", help="all files and Wikidata tables, then the movie lists")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    a = ap.parse_args()
    data = Path(a.data_dir).resolve()
    names = list(MANIFEST["datasets"])
    if a.list:
        for n in names:
            d = MANIFEST["datasets"][n]
            print(f"{n} ({d['size']}): {d['about']}\n  terms: {d['license']}")
        print("wikidata: " + MANIFEST["wikidata"]["about"])
        print("lists: movie lists built from the IMDb files")
        print("wikipedia: articles for the movie lists, about an hour")
        for n, why in MANIFEST["not_downloadable"].items():
            print(f"not here, {n}: {why}")
        return
    if a.status:
        return status(data)
    targets = (names + ["wikidata", "lists"]) if a.all else a.targets
    if not targets:
        ap.print_help()
        return
    bad = [t for t in targets if t not in names + ["wikidata", "lists", "wikipedia"]]
    if bad:
        raise SystemExit(f"unknown: {bad}. See --list.")
    for t in targets:
        if t in names:
            get_dataset(t, data)
        elif t == "wikidata":
            get_wikidata(data)
        elif t == "lists":
            build_lists(data)
        else:
            pull_wikipedia(data)
    status(data)


if __name__ == "__main__":
    main()
