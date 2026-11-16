"""Entity typing in context: what kind of thing is a word or phrase where people actually use it?

The same string means different things: "Paris" is a city and a Hilton, "Christmas" a date and part of a
title, "Washington" a city and Denzel, "Bond" a word and a franchise. A word list can't tell them apart, and
neither can a named-entity model reading the bare word ("Tarantino" alone is a place to spaCy's small model).
So each candidate is read where it occurs: up to `per_item` sentences that use it, through spaCy's NER, and
the label of the entity that covers the match is a vote. The votes give a distribution over coarse types and
a share for the winner, so a split ("Paris": place 0.85, person 0.10) shows up instead of being hidden.

Coarse types: person, org (studios, companies, agencies), work (titles, franchises), place, group
(nationalities, religions), time (dates, holidays, events), none (no entity covers the match).
"""
from __future__ import annotations

import random
import re
from collections import Counter

import pandas as pd

from . import clean as C

COARSE = {"PERSON": "person", "ORG": "org", "WORK_OF_ART": "work", "PRODUCT": "work", "LAW": "work",
          "GPE": "place", "LOC": "place", "FAC": "place", "NORP": "group", "LANGUAGE": "group",
          "DATE": "time", "TIME": "time", "EVENT": "time"}
ENTITY_TYPES = {"person", "org", "work"}   # the types that identify who made a film or which film it is


def item_pattern(item: str) -> re.Pattern:
    words = [re.escape(w) for w in item.split()]
    return re.compile(r"(?<![A-Za-z])" + r"\s+".join(words) + r"(?:'s|s'|es|s)?(?![A-Za-z])", re.I)


def contexts(items: list[str], texts, per_item: int, seed: int, window: int = 300,
             prefer_capitalised: bool = False) -> dict[str, list[tuple[str, int, int]]]:
    """Up to per_item (snippet, start, end) per item, from texts in a seeded random order, one mention per
    text. Mentions are sampled as they come, capitalised or not: preferring capitalised ones over-samples the
    item inside other names ("Magic Mike", "Reservoir Dogs") and makes common words look like titles."""
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    pats = {it: item_pattern(it) for it in items}
    first = {it: it.split()[0].lower() for it in items}
    found = {it: [] for it in items}
    lower = {it: [] for it in items}
    for i in order:
        t = texts[i]
        tl = t.lower()
        for it, p in pats.items():
            if len(found[it]) >= per_item or first[it] not in tl:
                continue
            for m in p.finditer(t):
                a, b = max(0, m.start() - window // 2), min(len(t), m.end() + window // 2)
                snip = (t[a:b], m.start() - a, m.end() - a)
                (found if (m.group()[:1].isupper() or not prefer_capitalised) else lower)[it].append(snip)
                break
        if all(len(v) >= per_item for v in found.values()):
            break
    return {it: (found[it] + lower[it])[:per_item] for it in items}


def same_span(ent, a: int, b: int, slack: int = 4) -> bool:
    """The entity is the item itself: it ends where the match ends (give or take "'s"), and starts at the
    match (or at a "the " before it). A person may start earlier, since a surname ends a full name ("Steven
    Spielberg"); any other entity that only contains the item ("Magic Mike", "Reservoir Dogs") is another name."""
    if not b - 3 <= ent.end_char <= b + 2:
        return False
    return ent.start_char <= a and (ent.label_ == "PERSON" or a - slack <= ent.start_char)


def type_votes(ctx: dict[str, list[tuple[str, int, int]]], nlp, batch_size: int = 64) -> pd.DataFrame:
    """One row per item: votes per coarse type, the winner and its share."""
    jobs = [(it, s, a, b) for it, xs in ctx.items() for s, a, b in xs]
    votes = {it: Counter() for it in ctx}
    for (it, _, a, b), doc in zip(jobs, nlp.pipe([s for _, s, _, _ in jobs], batch_size=batch_size)):
        lab = next((COARSE.get(e.label_, "none") for e in doc.ents if same_span(e, a, b)), "none")
        votes[it][lab] += 1
    rows = []
    for it, v in votes.items():
        n = sum(v.values())
        top, k = v.most_common(1)[0] if n else ("none", 0)
        rows.append({"item": it, "contexts": n, "type": top, "share": k / n if n else float("nan"),
                     **{f"n_{t}": v.get(t, 0) for t in ("person", "org", "work", "place", "group", "time", "none")}})
    return pd.DataFrame(rows)


def agreement(pred: pd.Series, truth: pd.Series) -> dict:
    """Precision, recall and Cohen's kappa of a yes/no prediction against hand labels."""
    p, t = pred.astype(bool), truth.astype(bool)
    tp, fp, fn, tn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"), "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan")}
