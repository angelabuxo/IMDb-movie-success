"""Give each studio and distributor name a Wikidata ID, so the graph merges a
company's name variants only when they are the same entity.

Run from the data folder before build_graph.py:

    cd data/raw && ../../.venv/bin/python ../../scripts/network/resolve_companies.py

Names come from the Wikipedia infobox ("studio" and "distributor" fields, as
parsed into features/movies_features_10k.csv). parse_infobox.py keeps a link's
target, so most names are English Wikipedia article titles. Each name is
resolved in this order, and only by identifier, never by similar spelling:

1. wikipedia: the name is itself an English Wikipedia article, and that
   article's Wikidata item is the ID. Disambiguation pages are refused.
2. redirect: the name redirects to an article. A redirect can mean a rename
   ("20th Century Fox" to "20th Century Studios") but also a label, subsidiary,
   parent or owner ("Magnet Releasing" to Magnolia Pictures, "Endgame
   Entertainment" to a person), and Wikidata's film entries repeat the same
   conflation, so a film can't confirm it. A redirect is accepted only when
   (a) the target's own Wikidata item lists the name as its label or an alias,
   or (b) the two names are the same once case, spacing, punctuation and a
   legal suffix ("Inc.", "Corp.", "Company", "Ltd.", "LLC") are dropped, as
   with "Lions Gate Films" and "Lionsgate Films". Wikidata aliases are noisy
   too ("Endgame Entertainment" is an alias of a person), so a redirect
   accepted by alias is still refused when a Wikidata search finds another
   item with that name as its label or alias: the name then belongs to a
   separate entity, or is ambiguous. It is also refused, everywhere, if any
   film credits both the name and the company it redirects to.
3. film: otherwise, the film's own Wikidata entry (features/wd_studio.tsv and
   wd_distributor.tsv) lists exactly one company whose label equals the name
   after case and punctuation are normalised.
4. Otherwise the name stays unresolved and becomes its own node, as do
   redirects that fail rule 2.

So labels and subsidiaries are never folded into a parent, at the cost of
leaving some renames unmerged when Wikidata lacks the old name as an alias.

Writes features/graph/companies.csv (one row per film, role and name with the
ID it got and how) and prints every ID that absorbed more than one name, for
checking by hand. API answers are cached in features/graph/company_cache/.
"""

import csv
import hashlib
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

MOVIES = "features/movies_features_10k.csv"
WD = {"studio": "features/wd_studio.tsv", "distributor": "features/wd_distributor.tsv"}
OUT = "features/graph/companies.csv"
CACHE = Path("features/graph/company_cache")
API = "https://en.wikipedia.org/w/api.php"
WIKIDATA = "https://www.wikidata.org/w/api.php"
AGENT = "IMDb-movie-success course project (DTU 02805; https://github.com/angelabuxo/IMDb-movie-success)"
CHUNK = 50


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def get(params, api=API):
    key = hashlib.sha1(json.dumps([api, params], sort_keys=True).encode()).hexdigest()[:20]
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    url = api + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            break
        except Exception:
            time.sleep(2 ** attempt)
    else:
        raise SystemExit(f"Wikipedia API kept failing for {url[:120]}")
    CACHE.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".part").write_text(json.dumps(data))
    path.with_suffix(".part").rename(path)
    time.sleep(0.3)
    return data


def resolve_titles(names):
    """{name: (qid, article title, redirected)} for names that lead to a Wikipedia article."""
    out = {}
    names = sorted(names)
    for i in range(0, len(names), CHUNK):
        part = names[i:i + CHUNK]
        q = get({"action": "query", "titles": "|".join(part), "redirects": 1, "prop": "pageprops",
                 "ppprop": "wikibase_item|disambiguation", "format": "json", "formatversion": 2})["query"]
        step, redirect = {}, set()
        for kind in ("normalized", "redirects"):
            for m in q.get(kind, []):
                step[m["from"]] = m["to"]
                if kind == "redirects":
                    redirect.add(m["from"])
        pages = {p["title"]: p for p in q.get("pages", [])}
        for n in part:
            t, seen, redirected = n, set(), False
            while t in step and t not in seen:
                seen.add(t)
                redirected |= t in redirect
                t = step[t]
            p = pages.get(t)
            if p and not p.get("missing") and "pageprops" in p and "disambiguation" not in p["pageprops"] \
                    and p["pageprops"].get("wikibase_item"):
                out[n] = (p["pageprops"]["wikibase_item"], p["title"], redirected)
        if (i // CHUNK) % 20 == 0:
            print(f"looked up {min(i + CHUNK, len(names))}/{len(names)} names", flush=True)
    return out


def names_of(qids):
    """{qid: [label and aliases in English or 'mul']} from Wikidata."""
    out = {}
    qids = sorted(qids)
    for i in range(0, len(qids), CHUNK):
        ents = get({"action": "wbgetentities", "ids": "|".join(qids[i:i + CHUNK]), "props": "labels|aliases",
                    "languages": "en|mul", "format": "json"}, WIKIDATA).get("entities", {})
        for q, e in ents.items():
            vals = [v["value"] for v in e.get("labels", {}).values()]
            vals += [a["value"] for lang in e.get("aliases", {}).values() for a in lang]
            out[q] = vals
    return out


def other_items(name, target):
    """Wikidata items other than target whose label or alias is the name."""
    hits = get({"action": "wbsearchentities", "search": name, "language": "en", "type": "item", "limit": 20,
                "format": "json"}, WIKIDATA).get("search", [])
    return sorted({h["id"] for h in hits if h["id"] != target
                   and same_name(name, h.get("match", {}).get("text", h.get("label", "")))})


LEGAL = r"\b(inc|incorporated|corp|corporation|company|co|ltd|limited|llc|plc|the)\b"


def same_name(a, b):
    """True when two names differ only in case, spacing, punctuation and legal suffixes."""
    squash = lambda s: re.sub(r"[^a-z0-9]", "", re.sub(LEGAL, " ", norm(s)))
    return squash(a) == squash(b)


def film_companies():
    """{(tconst, role): {normalised label: {qid, ...}}} from the films' Wikidata entries."""
    out = defaultdict(lambda: defaultdict(set))
    labels = {}
    for role, path in WD.items():
        with open(path, encoding="utf8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                t = parts[0].strip('"')
                qid = parts[1].strip("<>").rsplit("/", 1)[-1]
                m = re.match(r'"(.*)"@\w+', parts[2])
                if not m:
                    continue
                out[(t, role)][norm(m.group(1))].add(qid)
                labels[qid] = m.group(1)
    return out, labels


def main():
    uses = []
    with open(MOVIES, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            for role, field in (("studio", "studios"), ("distributor", "distributors")):
                for n in sorted({s.strip() for s in (r[field] or "").split("|") if s.strip()}):
                    uses.append((r["tconst"], role, n))
    names = {n for _, _, n in uses}
    wiki = resolve_titles(names)
    anchors, wd_labels = film_companies()

    rows, how = [], defaultdict(int)
    # IDs each film's Wikidata entry lists for each role, for confirming redirects
    film_ids = defaultdict(set)
    for (t, role), by_label in anchors.items():
        for qs in by_label.values():
            film_ids[(t, role)] |= qs
    # Which item each film's other names point to directly. If a film credits both a redirecting
    # name and the company it redirects to ("Anton Capital Entertainment" and "StudioCanal"), the
    # two are different companies, whatever the redirect says: that veto holds everywhere.
    direct = defaultdict(set)
    for t, role, n in uses:
        if n in wiki and not wiki[n][2]:
            direct[(t, role)].add(wiki[n][0])
    vetoed = {n for t, role, n in uses if n in wiki and wiki[n][2] and wiki[n][0] in direct[(t, role)]}
    # Otherwise a redirect is accepted when the names match apart from legal suffixes, or the
    # target's Wikidata item lists the name as its label or an alias.
    redirected = {n for n in wiki if wiki[n][2]} - vetoed
    known = names_of({wiki[n][0] for n in redirected})
    confirmed, reason, rejected = set(), {}, {}
    for n in sorted(redirected):
        if same_name(n, wiki[n][1]):
            confirmed.add(n); reason[n] = "same name"
        elif any(same_name(n, a) for a in known.get(wiki[n][0], [])):
            others = other_items(n, wiki[n][0])
            if others:
                rejected[n] = others
            else:
                confirmed.add(n); reason[n] = "Wikidata alias of target, no other item with the name"
    for t, role, n in uses:
        hits = anchors.get((t, role), {}).get(norm(n), set())
        if n in wiki and not wiki[n][2]:
            qid, label = wiki[n][:2]
            method = "wikipedia"
        elif n in wiki and n in confirmed:
            qid, label = wiki[n][:2]
            method = "redirect"
        elif len(hits) == 1:
            qid = next(iter(hits))
            method, label = "film", wd_labels.get(qid, n)
        else:
            qid, label = "", n
            method = "redirect_unconfirmed" if n in wiki else "unresolved"
        how[method] += 1
        rows.append({"tconst": t, "role": role, "name": n, "qid": qid, "label": label, "method": method})
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_qid = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["qid"]:
            by_qid[r["qid"]][r["name"]] += 1
    merged = {q: v for q, v in by_qid.items() if len(v) > 1}
    films_of = defaultdict(int)
    for t, role, n in uses:
        films_of[n] += 1
    print(f"redirects: {len(confirmed)} accepted, {len(vetoed)} vetoed (named on a film beside their target), "
          f"{len({n for n in wiki if wiki[n][2]} - confirmed - vetoed)} unconfirmed")
    print(f"{len(rejected)} alias matches refused because another Wikidata item has the name, e.g.:")
    for n in sorted(rejected, key=lambda x: -films_of[x])[:15]:
        print(f"  {n} -> {wiki[n][1]} refused ({films_of[n]} films; also {', '.join(rejected[n][:3])})")
    print("accepted redirects with 10+ films (name -> article, reason):")
    for n in sorted(confirmed, key=lambda x: -films_of[x]):
        if films_of[n] >= 10:
            print(f"  {n} -> {wiki[n][1]} ({films_of[n]} films; {reason[n]})")
    print(f"\n{len(rows):,} film-company links, {len(names):,} distinct names -> "
          f"{len(by_qid):,} Wikidata IDs + {len({r['name'] for r in rows if not r['qid']}):,} unresolved names")
    print("links by method:", dict(how))
    print(f"{len(merged)} IDs absorbed more than one name. Largest by films:")
    for q, v in sorted(merged.items(), key=lambda kv: -sum(kv[1].values()))[:40]:
        print(f"  {q:11} " + " | ".join(f"{n} ({c})" for n, c in sorted(v.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
