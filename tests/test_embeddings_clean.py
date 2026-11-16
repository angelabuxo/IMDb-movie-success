"""The cleaning rules of analysis/embeddings, on made-up text."""
import re

import numpy as np

from analysis.embeddings import clean as C


def test_basic_strips_markup_entities_and_urls():
    assert C.basic("Tom &amp;amp; Jerry <br/> see http://x.org/a  now") == "Tom & Jerry see now"


def test_detokenise_restores_written_english():
    assert C.detokenise("He did n't go . It 's ( really ) over , ok ?") == "He didn't go. It's (really) over, ok?"


def test_money_sentences_are_dropped_and_others_kept():
    text = ("The acting is superb. It flopped at the box office. It cost $40 million to make. "
            "A gross and silly comedy. I laughed a lot.")
    kept, dropped, seen = C.drop_money_sentences(text)
    assert (dropped, seen) == (2, 5)
    assert kept == "The acting is superb. A gross and silly comedy. I laughed a lot."


def test_money_pattern_examples():
    for s in ("a low-budget gem", "the highest-grossing film", "it made a lot of money", "a huge hit",
              "bombed in cinemas", "a commercial failure", "earned 300 million"):
        assert C.MONEY.search(s), s
    for s in ("the bomb squad arrives", "gross-out humour", "a hit man", "he hit her", "rich characters"):
        assert not C.MONEY.search(s), s


def test_title_mask_multiword_any_case_single_word_needs_capital():
    p = C.title_pattern(["Toy Story (1995)", "Toy Story"])
    assert C.mask("I love TOY STORY and toy  story 2", p, "the film")[0] == "I love the film and the film 2"
    p = C.title_pattern(["Godfather, The (1972)"])
    assert C.mask("The Godfather is long", p, "the film")[0] == "the film is long"
    p = C.title_pattern(["Alien (1979)"])
    assert C.mask("Alien has an alien in it", p, "the film")[0] == "the film has an alien in it"
    assert C.title_pattern(["Up (2009)"]) is None        # too short to mask safely


def test_people_mask_full_names_and_capitalised_surnames():
    p = C.people_pattern(["Tom Hanks", "Jack Black", ""])
    out, n = C.mask("tom hanks is great. Hanks's best. Black shines in a black comedy.", p, "Person")
    assert out == "Person is great. Person best. Person shines in a black comedy." and n == 3


def test_looks_english():
    assert C.looks_english("This is the best film of the year and it is a joy to watch.", 0.9, 3)
    assert not C.looks_english("Это лучший фильм года, и его приятно смотреть снова.", 0.9, 3)
    assert not C.looks_english("Una película maravillosa que emociona desde el principio.", 0.9, 3)


SCRIPT = """<script><scene><stage_direction>INT. KITCHEN - DAY</stage_direction>
<scene_description>MARLA enters . A man follows her . CUT TO:</scene_description>
<character>MARLA (CONT'D)</character><parenthetical>(angry)</parenthetical>
<dialogue>I did n't ask you , Doctor .</dialogue>
<character>DOCTOR</character><dialogue>Marla , the doctor is in . Marla ?</dialogue>
<character>MAN</character><dialogue>The man said nothing to Marla .</dialogue>
<character>THE END</character></scene></script>"""


def test_script_blocks_keep_description_and_dialogue_only():
    blocks, labels = C.script_blocks(SCRIPT)
    assert blocks[0] == "MARLA enters. A man follows her"
    assert blocks[1] == "I didn't ask you, Doctor"
    assert not any("KITCHEN" in b or "angry" in b or "CUT TO" in b for b in blocks)
    assert labels[0] == "MARLA (CONT'D)"


def test_names_are_masked_and_common_nouns_kept():
    text, st = C.clean_script(SCRIPT, max_lower=0.1, min_count=3, min_len=3, name_ph="Person")
    assert st["names"] == ["MARLA"]                       # "doctor" and "man" also appear in lower case
    assert "Marla" not in text and "MARLA" not in text
    assert st["names_masked"] == 4 and "doctor is in" in text and "man said" in text


def test_summary_cleaning_masks_people_brackets_and_actor_names():
    class Ent:
        label_ = "PERSON"

        def __init__(self, m):
            self.start_char, self.end_char = m.span()

    class Doc:
        def __init__(self, t):
            self.ents = [Ent(m) for m in re.finditer(r"Nanisca|Nawi", t)]

    out, st = C.clean_summary("General Nanisca ( Viola Davis ) trains Nawi [ 2 ] in Dahomey .", Doc, None, "Person",
                              "the film")
    assert out == "General Person trains Person in Dahomey."
    assert st["people"] == 2 and st["actor_parens"] == 1 and st["brackets"] == 1


def test_passages_drop_only_a_short_tail():
    words = " ".join(str(i) for i in range(400))
    assert [len(p.split()) for p in C.passages(words, 180, 30)] == [180, 180, 40]
    assert [len(p.split()) for p in C.passages(" ".join(["w"] * 370), 180, 30)] == [180, 180]
    assert C.passages("only ten words here", 180, 30) == ["only ten words here"]


def test_stats_helpers_on_planted_signal():
    from analysis.embeddings import stats as S
    rng = np.random.default_rng(0)
    n = 600
    Z = rng.normal(size=(n, 10))
    ctrl = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = 0.8 * ctrl[:, 1] + 0.6 * Z[:, 0] + rng.normal(size=n)
    alphas = np.logspace(-2, 4, 25)
    got = S.cv_scores(Z, y, ctrl, alphas, folds=5, seed=1)
    assert got["r2_gain"] > 0.1 and got["partial_spearman"] > 0.3
    strata = np.zeros(n, dtype=int)
    null = S.shuffle_gains(Z, y, ctrl, strata, alphas, folds=5, perms=50, seed=1)
    assert got["r2_gain"] > max(null) and abs(np.mean(null)) < 0.02
    # and none when the vectors are noise
    noise = S.cv_scores(rng.normal(size=(n, 10)), y, ctrl, alphas, folds=5, seed=1)
    assert noise["r2_gain"] < 0.02
    # a kNN graph over Z puts films with similar Z[:, 0] together, so y is assortative on it
    edges = S.knn_edges(Z, 10)
    assert S.assortativity(edges, y) > 0.05
    assert abs(S.assortativity(edges, rng.permutation(y))) < 0.05


def test_junk_share_counts_ocr_damage_only():
    assert C.junk_share("He said, \"Don't go.\" T.S. left at 10:00 (quietly) - well-known.") == 0
    assert C.junk_share("gest:ure and dart:s away M~ralda ~ot::s we~") == 5 / 7
    assert C.junk_share("\\* Person(35) walks in \\* M-16 lounge/club") == 0
    assert C.junk_share("") == 0.0
