"""The preprocessing of analysis/screenplays/preprocess/preprocess.py on toy strings (no data, no script text)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "screenplays" / "preprocess"))
import preprocess as pp  # noqa: E402


def el(tag, layer, text):
    return {"tag": tag, "layer": layer, "text": text, "scene": 0}


def test_detached_contractions_are_joined():
    assert pp.tokenize("I do n't know , she 's here") == ["i", "don't", "know", "she's", "here"]
    assert pp.tokenize("We ca n't and wo n't") == ["we", "can't", "and", "won't"]


def test_attached_contractions_stay_one_token_and_curly_apostrophes_are_straightened():
    assert pp.tokenize("you’re here, don't go") == ["you're", "here", "don't", "go"]


def test_both_layers_give_the_same_tokens_for_the_same_words():
    assert pp.tokenize("Do n't go , please .") == pp.tokenize("Don't go, please.")


def test_symbols_and_digits_are_removed_and_letters_of_any_language_kept():
    assert pp.tokenize("• ~ \\* [ ] 1984 café à la") == ["café", "à", "la"]
    assert pp.tokenize("walk-in 'quoted' word_") == ["walk", "in", "quoted", "word"]


def test_stopwords_are_kept():
    assert pp.tokenize("The man and the dog") == ["the", "man", "and", "the", "dog"]


def test_name_tokens_drop_parenthesised_extensions():
    assert pp.name_tokens(["JACK (V.O.)", "MARY ANN", "(CONT'D)"]) == {"jack", "mary", "ann"}


def test_film_layers_remove_names_keep_layers_apart_and_ignore_other_layers():
    els = [el("character", "other", "JACK"),
           el("dialogue", "dialogue", "Jack , come here ."),
           el("scene_description", "description", "Jack walks to the door ."),
           el("parenthetical", "parenthetical", "(quietly)"),
           el("stage_direction", "other", "CUT TO:"),
           el("dialogue", "dialogue", "")]
    layers, names, stats = pp.film_layers(els)
    assert names == {"jack"}
    assert layers["dialogue"] == [["come", "here"]]
    assert layers["description"] == [["walks", "to", "the", "door"]]
    assert stats["dialogue"] == {"before": 3, "names_removed": 1}


def test_each_element_is_its_own_context_unit():
    els = [el("dialogue", "dialogue", "one two"), el("dialogue", "dialogue", "three four")]
    layers, _, _ = pp.film_layers(els)
    assert layers["dialogue"] == [["one", "two"], ["three", "four"]]


def test_an_element_that_is_only_names_or_symbols_is_dropped():
    els = [el("character", "other", "JACK"), el("dialogue", "dialogue", "Jack ~ *")]
    layers, _, _ = pp.film_layers(els)
    assert layers["dialogue"] == []


def test_common_label_tokens_are_not_removed_as_names():
    els = [el("character", "other", "THE MAN"), el("character", "other", "KOWALSKI"),
           el("dialogue", "dialogue", "The man says Kowalski left .")]
    layers, names, _ = pp.film_layers(els, common_labels={"the", "man"})
    assert names == {"kowalski"}
    assert layers["dialogue"] == [["the", "man", "says", "left"]]


def test_name_max_films_is_fixed_at_twenty():
    assert pp.NAME_MAX_FILMS == 20
