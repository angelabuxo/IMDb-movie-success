"""One Wikipedia article per film.

Films are linked to articles through Wikidata's IMDb ID property, and two Wikidata items can carry the same IMDb
ID: a film and its franchise (Horizon: An American Saga / ... – Chapter 1), or a film and a recut (Apocalypse
Now / Apocalypse Now Redux). Every table built from the articles then has two rows for one film, and whichever
reader keeps "the first row" gets whichever article happened to come first.

parse_infobox.py calls best_article() so its tables hold one row per IMDb ID: the article whose title is closest
to the film's IMDb title, ties going to the first. Readers never have to choose.

What it captures: the usual case, where the film's own article is titled after the film.
What it misses: two articles with equally close titles (then the first is kept, as before), and a film whose
own article title differs more from IMDb's than a related article's does.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_PAREN = re.compile(r"\s*\([^)]*\)\s*$")   # "(film)", "(2019 film)", "(American film)"


def norm(title: str) -> str:
    """Lowercase, accents and punctuation removed, a trailing parenthetical dropped: 'Horizon: An American Saga –
    Chapter 1' and 'Horizon: An American Saga - Chapter 1' both become 'horizonanamericansagachapter1'."""
    t = _PAREN.sub("", title or "")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", t.lower())


def title_score(article_title: str, imdb_title: str) -> float:
    a, b = norm(article_title), norm(imdb_title)
    if not a or not b:
        return 0.0
    return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()


def best_article(article_titles: list[str], imdb_title: str | None) -> int:
    """Index of the article to keep for one film. Without an IMDb title, or on a tie, the first one."""
    if not imdb_title or len(article_titles) < 2:
        return 0
    scores = [title_score(t, imdb_title) for t in article_titles]
    return scores.index(max(scores))
