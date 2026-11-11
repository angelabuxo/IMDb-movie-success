"""Character networks, text measures and sentiment arcs from MovieSum screenplays.

Run from the data folder (data/raw), like the scripts in scripts/imdb/:

    cd data/raw && ../../.venv/bin/python ../../scripts/screenplays/screenplay_features.py

MovieSum marks up each screenplay as <scene> blocks holding <stage_direction>
(the heading), <scene_description>, <character>, <parenthetical> and
<dialogue>. A character's line is the <dialogue> right after its <character>.

Per film it writes one row to features/screenplay_features.csv:
- network: two characters are linked for every scene they both speak in. A
  character needs MIN_LINES lines to be a node. Measures: characters, density,
  the lead's share of lines, degree centralisation, weighted clustering, Louvain
  modularity and community count (the partition found most often over
  LOUVAIN_SEEDS seeds), and the Gini of lines across characters.
- text: scenes, words, speeches per scene, words per speech, dialogue share
  (spoken words over all words).
- sentiment: each speech's VADER compound score (-1 to 1); the speeches, in
  order, are cut into BINS equal parts and each part's mean is one point of
  the arc. Kept: mean, swing (standard deviation of the smoothed arc), where
  the low point falls (0 start, 1 end) and the ending (last tenth minus mean).

It also writes features/screenplay_arcs.npy (one row of BINS points for each
film with has_arc = 1, in the CSV's order) and features/screenplay_words.json (dialogue word counts
per film, for the distinctive-words comparison in the notebook).

A screenplay with fewer than MIN_SCENES scenes is skipped: its scene markup is
missing, so the whole film reads as one scene. Character labels in MovieSum
aren't cleaned: "JULES (V.O.)" and "JULES" are
one person here, after extensions in brackets and "CONT'D" are stripped. 12
IMDb IDs appear twice; the longest script is kept, as in the embeddings.
"""

import csv
import html
import json
import random
import re
from collections import Counter
from itertools import combinations

import igraph as ig
import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter1d
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SPLITS = ["scripts/moviesum/train.jsonl", "scripts/moviesum/val.jsonl", "scripts/moviesum/test.jsonl"]
OUT = "features/screenplay_features.csv"
ARCS = "features/screenplay_arcs.npy"
WORDS = "features/screenplay_words.json"
SEED = 1
MIN_LINES = 3
LOUVAIN_SEEDS = 10
BINS = 40
SMOOTH = 2.0
MIN_SPEECHES = 200     # fewer than this makes a 40-point arc too noisy: no arc
MIN_SCENES = 20        # fewer means the scene markup is missing

SCENE = re.compile(r"<scene>(.*?)</scene>", re.S)
TAG = re.compile(r"<(character|dialogue|parenthetical|scene_description|stage_direction)>(.*?)</\1>", re.S)
WORD = re.compile(r"[A-Za-z']+")


# Headers MovieSum sometimes tags as a speaker: "THE END" in 307 scripts, "CUT TO:", "SHOOTING SCRIPT".
NOT_A_NAME = re.compile(r"\b(THE END|CUT TO|FADE|DISSOLVE|SCRIPT|DRAFT|REVISION|CONTINUED|SMASH CUT)\b|^(INT|EXT)\b")


def clean_name(raw):
    raw = html.unescape(raw)
    if NOT_A_NAME.search(raw.upper()):
        return ""
    name = re.sub(r"\([^)]*\)", " ", raw).upper()
    name = re.sub(r"\bCONT'?D\b|[^A-Z0-9 .'-]", " ", name)
    return re.sub(r"\s+", " ", name).strip(" .-'")


def parse(script):
    """[(scene, [(name, dialogue), ...]), ...] and the script's total word count."""
    scenes, total = [], 0
    for block in SCENE.findall(script):
        lines, speaker = [], None
        for tag, body in TAG.findall(block):
            body = html.unescape(body)
            total += len(WORD.findall(body)) if tag != "character" else 0
            if tag == "character":
                speaker = clean_name(body) or None
            elif tag == "dialogue" and speaker:
                lines.append((speaker, body))
            elif tag != "parenthetical":
                speaker = None
        scenes.append(lines)
    return scenes, total


def gini(xs):
    xs = np.sort(np.asarray(xs, dtype=float))
    n = len(xs)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(xs) / (n * xs.sum())) if n and xs.sum() else 0.0


def louvain(g, seed):
    nodes = sorted(g)
    idx = {n: i for i, n in enumerate(nodes)}
    h = ig.Graph(n=len(nodes), edges=[(idx[a], idx[b]) for a, b in g.edges()])
    w = [g[a][b]["weight"] for a, b in g.edges()]
    ig.set_random_number_generator(random.Random(seed))
    part = h.community_multilevel(weights=w)
    return [tuple(sorted(nodes[i] for i in c)) for c in part], h.modularity(part, weights=w)


def most_frequent_partition(g):
    if not g.number_of_edges():
        return 1, 0.0
    seen = {}
    for i in range(LOUVAIN_SEEDS):
        parts, mod = louvain(g, SEED + i)
        key = tuple(sorted(parts))
        seen[key] = (seen.get(key, (0, mod))[0] + 1, mod)
    key = max(seen, key=lambda k: (seen[k][0], round(seen[k][1], 12), k))
    return len(key), seen[key][1]


def network(scenes):
    lines = Counter(n for s in scenes for n, _ in s)
    keep = {n for n, c in lines.items() if c >= MIN_LINES}
    g = nx.Graph()
    g.add_nodes_from(sorted(keep))
    for s in scenes:
        for a, b in combinations(sorted({n for n, _ in s} & keep), 2):
            w = g[a][b]["weight"] + 1 if g.has_edge(a, b) else 1
            g.add_edge(a, b, weight=w)
    return g, {n: lines[n] for n in keep}


def features(scenes, total):
    g, lines = network(scenes)
    n = g.number_of_nodes()
    if n < 2:
        return None
    deg = [d for _, d in g.degree()]
    communities, mod = most_frequent_partition(g)
    said = [t for s in scenes for _, t in s]
    spoken = sum(len(WORD.findall(t)) for t in said)
    return {
        "characters": n, "density": nx.density(g), "lead_share": max(lines.values()) / sum(lines.values()),
        "centralisation": sum(max(deg) - d for d in deg) / ((n - 1) * (n - 2)) if n > 2 else 0.0,
        "clustering": nx.average_clustering(g, weight="weight"), "modularity": mod,
        "communities": communities, "line_gini": gini(list(lines.values())),
        "scenes": len(scenes), "words": total, "speeches": len(said),
        "speeches_per_scene": len(said) / len(scenes) if scenes else 0.0,
        "words_per_speech": spoken / len(said) if said else 0.0,
        "dialogue_share": spoken / total if total else 0.0,
    }


def arc(scores):
    a = np.array([p.mean() for p in np.array_split(np.asarray(scores), BINS)])
    sm = gaussian_filter1d(a, SMOOTH, mode="nearest")
    return a, {"sent_mean": float(a.mean()), "sent_swing": float(sm.std()),
               "sent_low_point": float(np.argmin(sm) / (BINS - 1)),
               "sent_ending": float(a[-BINS // 10:].mean() - a.mean())}


def main():
    films = {}
    for path in SPLITS:
        with open(path, encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                if r["imdb_id"] not in films or len(r["script"]) > len(films[r["imdb_id"]]["script"]):
                    films[r["imdb_id"]] = r
    vader = SentimentIntensityAnalyzer()
    rows, arcs, words = [], [], {}
    ids = sorted(films)
    for i, tconst in enumerate(ids, 1):
        scenes, total = parse(films[tconst]["script"])
        if len(scenes) < MIN_SCENES:
            continue
        f = features(scenes, total)
        if f is None:
            continue
        said = [t for s in scenes for _, t in s]
        names = {w.lower() for s in scenes for n, _ in s for w in n.split()}
        words[tconst] = {"counts": dict(Counter(w for t in said for w in re.findall(r"[a-z']+", t.lower()))),
                         "names": sorted(names)}
        row = {"tconst": tconst, "movie_name": films[tconst]["movie_name"], **f}
        if len(said) >= MIN_SPEECHES:
            a, s = arc([vader.polarity_scores(t)["compound"] for t in said])
            row.update(s, has_arc=1)
            arcs.append(a)
        else:
            row.update(sent_mean="", sent_swing="", sent_low_point="", sent_ending="", has_arc=0)
        rows.append(row)
        if i % 200 == 0 or i == len(ids):
            print(f"{i}/{len(ids)} screenplays", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    np.save(ARCS, np.array(arcs))
    with open(WORDS, "w") as f:
        json.dump(words, f)
    print(f"wrote {OUT}: {len(rows)} films, {len(arcs)} with arcs")


if __name__ == "__main__":
    main()
