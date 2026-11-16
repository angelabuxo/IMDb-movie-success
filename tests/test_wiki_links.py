"""One Wikipedia article per film (analysis/wiki_links.py), including the two real cases where two Wikidata
items share an IMDb ID."""

import pytest

from analysis.wiki_links import best_article, norm, title_score


@pytest.mark.parametrize("articles,imdb,keep", [
    (["Horizon: An American Saga", "Horizon: An American Saga – Chapter 1"], "Horizon: An American Saga - Chapter 1", 1),
    (["Apocalypse Now", "Apocalypse Now Redux"], "Apocalypse Now", 0),
    (["Apocalypse Now Redux", "Apocalypse Now"], "Apocalypse Now", 1),     # order no longer decides
    (["Heat (1995 film)", "Heat (soundtrack)"], "Heat", 0),                # tie on the title: first kept
    (["A", "B"], None, 0),                                                 # no IMDb title: first kept
    (["Only one"], "Something else", 0),
])
def test_best_article(articles, imdb, keep):
    assert best_article(articles, imdb) == keep


def test_norm_ignores_dashes_accents_punctuation_and_a_film_suffix():
    assert norm("Horizon: An American Saga – Chapter 1") == norm("Horizon: An American Saga - Chapter 1")
    assert norm("Amélie (film)") == norm("Amelie")
    assert title_score("Se7en", "Se7en") == 1.0
    assert title_score("", "Anything") == 0.0
