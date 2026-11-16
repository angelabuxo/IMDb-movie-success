"""Which Tag Genome tags name an entity (a person, a studio, a franchise) rather than describe content?

    python -m analysis.embeddings.tag_entity_check

scripts/reviews/tag_profit.py sorts the 1,084 tags into reception, commercial and content by pattern. A tag
such as "pixar", "kubrick" or "james bond" lands in content, but it identifies who made the film or which
series it belongs to, which is the network half of the project (and, for franchises, a commercial fact). This
check flags every tag three ways, the same flags as the word audit (s04):

- capitalised: in the sampled reviews, the share of the tag's mentions written with a capital letter
  (mentions at the start of a sentence are left out, since those are capitalised anyway);
- genome_name: the tag is, or contains, the full name of a person credited in the Tag Genome, the surname of
  a director with 5+ films there, or a Tag Genome film title of two words or more;
- few_films: fewer than 30 films have the tag at relevance 0.5 or more (films_high in tag_profit_tags.csv).

hand_label is a separate, hand-made reading of all 922 content tags (by Claude, 2026-11-16, to be checked by
the group): person, studio, franchise, or empty. The flags are measurements; the hand label is a judgement.
Writes outputs/tag_entity_check.csv (tag names and numbers only).
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter

import pandas as pd

from . import clean as C
from .paths import GENOME, OUTPUTS, PRIVATE, REPO

TAGS = REPO / "analysis" / "tags" / "tag_profit_tags.csv"
FEW_FILMS = 30
CAPITALISED = 0.8

HAND = {
    "person": ["almodovar", "amy smart", "arnold", "author:alan moore", "author:neil gaiman", "beatles", "c.s. lewis",
               "carrie-anne moss", "chris tucker", "coen bros", "coen brothers", "depp & burton", "dr. seuss",
               "emma watson", "ennio morricone", "francis ford copolla", "george orwell", "gilliam", "hitchcock",
               "jane austen", "kubrick", "kurosawa", "liv tyler", "lynch", "marx brothers", "miyazaki", "monty python",
               "neil gaiman", "philip k. dick", "roald dahl", "robert downey jr", "robert ludlum", "shakespeare",
               "sophia coppola", "spielberg", "tarantino", "tolkien", "truman capote"],
    "studio": ["aardman", "dc comics", "disney", "disney animated feature", "dreamworks", "marvel", "mtv", "muppets",
               "pixar", "pixar animation", "studio ghibli", "view askew"],
    "franchise": ["007", "batman", "bond", "firefly", "hannibal lecter", "harry potter", "indiana jones", "james bond",
                  "jay and silent bob", "spock", "star trek", "star wars"],
}


def genome_names() -> tuple[set[str], set[str], set[str]]:
    from .s01_corpora import genome_meta
    people, titles, directed = set(), set(), Counter()
    for m in genome_meta():
        if True:
            for x in ((m.get("directedBy") or "") + "," + (m.get("starring") or "")).split(","):
                if len(x.split()) >= 2:
                    people.add(x.strip().lower())
            for x in (m.get("directedBy") or "").split(","):
                if len(x.split()) >= 2:
                    directed[x.strip()] += 1
            t = re.sub(r"\s*\(\d{4}\)\s*$", "", m.get("title") or "").strip().lower()
            if len(t.split()) >= 2:
                titles.add(t)
    surnames = {n.split()[-1].lower() for n, k in directed.items() if k >= 5}
    return people, titles, surnames


def capitalised_share(tags: list[str], texts) -> dict[str, tuple[int, float]]:
    """(mentions not at a sentence start, share capitalised) per tag, over the review texts."""
    pats = {t: re.compile(r"(?<![A-Za-z])" + r"\s+".join(map(re.escape, t.split())) + r"(?![A-Za-z])", re.I)
            for t in tags if re.search("[a-z]", t)}
    n, cap = Counter(), Counter()
    for text in texts:
        low = text.lower()
        for t, p in pats.items():
            if t.split()[0] not in low:
                continue
            for m in p.finditer(text):
                before = text[:m.start()].rstrip()
                if not before or before[-1] in ".!?\"":
                    continue
                n[t] += 1
                cap[t] += m.group()[:1].isupper()
    return {t: (n[t], cap[t] / n[t] if n[t] else float("nan")) for t in pats}


def run() -> pd.DataFrame:
    t = pd.read_csv(TAGS)
    people, titles, surnames = genome_names()
    texts = pd.read_parquet(PRIVATE / "reviews.parquet", columns=["raw"]).raw
    cs = capitalised_share(t.tag.tolist(), texts)
    t["mentions"] = t.tag.map(lambda x: cs.get(x, (0, float("nan")))[0])
    t["capitalised_share"] = t.tag.map(lambda x: cs.get(x, (0, float("nan")))[1])
    t["capitalised"] = t.capitalised_share >= CAPITALISED

    def name_hit(tag):
        x = tag.lower().replace("author:", "")
        return (x in people or x in titles or (x in surnames and " " not in x)
                or any(p in x for p in people if " " in p and len(p) > 6 and p in x))

    t["genome_name"] = t.tag.map(name_hit)
    t["few_films"] = t.films_high < FEW_FILMS
    hand = {tag: kind for kind, tags in HAND.items() for tag in tags}
    missing = set(hand) - set(t.tag)
    assert not missing, f"hand-labelled tags not in the tag list: {missing}"
    t["hand_label"] = t.tag.map(hand).fillna("")
    content = t["class"] == "content"
    for col in ("rho_adjusted", "rho_adjusted_votes"):
        r = t[content][col].rank(ascending=False)
        t[f"rank_{col}_of_content"] = r.reindex(t.index)
    keep = ["tag", "class", "hand_label", "capitalised", "capitalised_share", "mentions", "genome_name", "few_films",
            "films_high", "rho_adjusted", "q_adjusted", "rho_adjusted_votes", "rank_rho_adjusted_of_content",
            "rank_rho_adjusted_votes_of_content"]
    out = t[keep].round(4)
    out.to_csv(OUTPUTS / "tag_entity_check.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_rows", 200)
    r = run()
    c = r[r["class"] == "content"]
    n = len(c)
    print(f"{n} content tags; hand-labelled entities: {(c.hand_label != '').sum()} "
          f"({c.hand_label.value_counts().to_dict()})")
    for flag in ("capitalised", "genome_name", "few_films"):
        f = c[flag]
        lab = c.hand_label != ""
        print(f"{flag}: {int(f.sum())} tags; of the hand-labelled entities it flags {int((f & lab).sum())}/{int(lab.sum())}, "
              f"and {int((f & ~lab).sum())} of its flags are not entities")
    any_flag = c.capitalised | c.genome_name
    lab = c.hand_label != ""
    print(f"capitalised or genome_name: catches {int((any_flag & lab).sum())}/{int(lab.sum())} entities, "
          f"{int((any_flag & ~lab).sum())} false flags: {sorted(c.tag[any_flag & ~lab])}")
    print("entities neither flag catches:", sorted(c.tag[lab & ~any_flag]))
    for col in ("rho_adjusted", "rho_adjusted_votes"):
        rk = f"rank_{col}_of_content"
        e = c[lab]
        print(f"\n{col}: entities in the top 25 of {n}: {sorted(e.tag[e[rk] <= 25])}; "
              f"in the bottom 25: {sorted(e.tag[e[rk] > n - 25])}")


def typed(model: str, per_item: int = 20) -> pd.DataFrame:
    """Entity type of every tag in review context (entities.py), and its agreement with the hand labels."""
    import spacy

    from . import entities as E
    t = pd.read_csv(OUTPUTS / "tag_entity_check.csv", keep_default_na=False, na_values={"capitalised_share": [""]})
    c = t[t["class"] == "content"].reset_index(drop=True)
    texts = pd.read_parquet(PRIVATE / "reviews.parquet", columns=["raw"]).raw.tolist()
    ctx = E.contexts(c.tag.str.replace("author:", "", regex=False).tolist(), texts, per_item, seed=20261116)
    nlp = spacy.load(model)
    v = E.type_votes(ctx, nlp).rename(columns={"item": "tag_text"})
    c["tag_text"] = c.tag.str.replace("author:", "", regex=False)
    c = c.merge(v, on="tag_text", how="left")
    c["model"] = model
    return c
