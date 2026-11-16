"""Write and execute notebooks/embeddings.ipynb from the saved outputs. Run from the repo root:

    python -m analysis.embeddings.build_notebook            # write and execute
    python -m analysis.embeddings.build_notebook --no-run   # write only

The notebook reads analysis/embeddings/outputs/ through report.py, so its numbers are the explorer page's.
Only the film map (Section 6) needs the local vectors; without them it says so and moves on.
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "embeddings.ipynb"

CELLS = [
    ("md", """# Reviews, screenplays and summaries in an embedding space: do they line up with profit?

Each film's text is turned into one vector, so that films whose text says similar things sit close together. Then we
ask whether films that sit close together also have similar money outcomes. Three texts, each with a different
relationship to the outcome:

| Text | Source | Written | What it could know about money |
|---|---|---|---|
| Audience reviews | MovieLens Tag Genome 2021 (IMDb reviews, nothing after 2017) | after release | the outcome itself ("it flopped"), hype, sequel status |
| Screenplays | MovieSum | before release | only what is on the page |
| Plot summaries | MovieSum (from Wikipedia) | after release, about the story | little, once names are removed |

Profitability is ln(worldwide gross / budget), revenue is ln(gross). Neither is real profit: marketing isn't in the
budget and cinemas keep part of the gross. The sample is the clean theatrical rule used everywhere in the repo
(USD budget and gross, 1975 to 2025, no rentals, regional or direct-to-video grosses, no streaming distributors).

Run order, from the repo root: `python -m analysis.embeddings.s01_corpora` (clean), `s02_embed` (vectors),
`s03_correlate` (tests), then this notebook. Every parameter is in `analysis/embeddings/config.yaml`; the two
added after the first results are under `post_hoc` and are labelled wherever they're used."""),
    ("code", """import sys
from pathlib import Path

REPO = Path.cwd().resolve().parent
sys.path.insert(0, str(REPO))
import pandas as pd
from IPython.display import Markdown, display

from analysis.embeddings import report as R

pd.set_option("display.max_columns", 30, "display.width", 200)
res = R.load()
r = res["results"]
say = R.reading(res)
print(f"{len(r):,} measurements")"""),
    ("md", """## 1. Cleaning

A language model reads everything in the text, including things that aren't the content. What each rule removes
and why:

- **Markup, HTML entities, URLs, tokeniser spacing** (MovieSum stores "did n't"): how the text was stored.
- **Review sentences about money** ("flopped at the box office", "$200 million"): a review is written after
  release, so this is the outcome leaking into the predictor. We drop the sentence, not the review.
- **The film's title, its cast and director (reviews), character names (screenplays) and every person's name
  (summaries, with spaCy NER)**: the model has read about famous films and stars, so with names left in, a vector
  can encode *which* film or *who* made it rather than what the text says. Who made it is the network half of the
  project; here we want the words.
- **Other films and other people named in a review** ("better than The Shawshank Redemption", "the Coens' best"):
  the titles and directing and starring credits of all 84,661 Tag Genome films. Full titles and names are masked
  in title case. A single word ("Shawshank", "Kubrick") is masked on its own only when the reviews show it is
  distinctive: almost always capitalised, and either used inside the full title or name a fifth of the time or more,
  or the surname of a prolific director whose own films' reviews use it at least five times as often. Words that
  spaCy, reading them in 20 review sentences each, mostly tags as a place, nationality, date or event ("Paris",
  "Christmas", "English") stay, as do title phrases that are places ("New York"). Plural and possessive forms
  ("Coens", "Kubrick's") count as the name.
- **Screenplay scene headings, speaker labels, parentheticals and transitions**: formatting.
- **Duplicate, very short and non-English reviews.** Each film gets 30 reviews sampled at random (seeded), so long
  runs of one reviewer or the file's order don't decide which ones we read.

Character names in screenplays are found without a model: a word from a speaker label counts as a name if the
script almost never writes it in lower case ("MARLA" is a name, "DOCTOR" isn't, because "the doctor" appears in
the descriptions). spaCy's PERSON tags are the second method; their agreement is below."""),
    ("code", """display(R.cleaning_table(res["cleaning"]))
chk = res["cleaning"]["screenplays"]["name_rule_vs_spacy"]
print(f"Name rule vs spaCy PERSON on {chk['scripts']} scripts x {chk['words_per_script']:,} words: "
      f"spaCy also tags {chk['share_of_rule_words_spacy_also_tags']:.0%} of the words the rule masks, and the rule "
      f"masks {chk['share_of_spacy_person_words_the_rule_masks']:.0%} of the words spaCy tags.")"""),
    ("md", """Reading the disagreements by hand: most words only the rule masks are character names spaCy misses (invented
names such as Linguini or Balian), with a few false ones (SERGEANT, Jerusalem). Most words only spaCy tags are
ordinary names that never speak (Jack, Steve) and some mistakes (Outside, Hesitates). Neither is ground truth;
both leave a small residue of names.

**Two problems found after the first results** (fixed, and marked post hoc in the config):
- OCR damage. Some MovieSum scripts were scanned and read badly ("gest:ure", "t~e"). The first word list for the
  "more profitable than expected" screenplays was entirely such fragments, so scripts with more than 0.5%
  damaged words are left out (`clean.junk_share`), and a check below puts every script back with the damage share
  as a control.
- Sequels. The first review word list was "installment", "prequel", "predecessor". Reviews say when a film is a
  sequel, and sequels return more than their budget predicts. The franchise flag (Wikidata) is now a control in
  the primary specification; the version without it is reported next to it."""),
    ("md", """## 2. The vectors alone know budget, not profit

Every text is cut into 180-word passages (the model reads at most 256 word pieces), each passage is embedded with
all-MiniLM-L6-v2, and a film's vector is the normalised mean. The vectors are reduced to 50 principal components
(unsupervised, so this step never sees the outcome) and a ridge regression predicts the outcome. Every R² is
**out-of-fold**: each film is predicted by a model fitted on the other four fifths, ten times over."""),
    ("code", """R.fig_alone(r).show()
display(Markdown(say["alone"]))"""),
    ("md", """## 3. What the text adds beyond budget, year, genre and franchise

The question that matters for profit: once we know the budget, the year, the genres and whether the film is a
sequel, does the text tell us anything more? The controls are fitted by least squares on the training folds; the
ridge regression on the text predicts what they leave. **The null** gives each film the vector of another film
from its own genre × decade cell, 500 times, which breaks the link between text and money and keeps everything else.
With budget controlled, profitability and revenue are the same question: ln(gross) = ln(gross/budget) + ln(budget)."""),
    ("code", """R.fig_net(res).show()
display(R.main_table(r))
display(Markdown(say["net"]))"""),
    ("md", """**What this measures and what it misses.** The R² gain measures how much a linear map from the text's 50
main directions improves the prediction of profit on unseen films. It captures what a general-purpose sentence model
encodes (topic, tone, genre markers, register); it misses anything the model doesn't encode and any non-linear
pattern. For reviews it is not a forecast: reviews are written after release, and even with money sentences removed,
"everyone saw this" or "nobody talks about it" survives. It says what audiences write about films that did better
than their budget predicted. Screenplays and summaries are the before-release question."""),
    ("md", """## 4. Checks: a second method, later films, and the rating

- **TF-IDF** is a different representation of the same cleaned text: word counts weighted by rarity, reduced to
  300 dimensions. If it agrees, the result isn't a quirk of one model.
- **Time split**: fitted on the earliest 70% of films and scored on the rest, the only version that could have been
  used in advance (year squared is dropped here, because fitted on early years it extrapolates wildly).
- **Rating**: is the text more than "people liked it"? The IMDb rating joins the controls.
- **Raw vs clean text**: in the table in Section 3. Cleaning should not *create* signal; if raw text scores higher,
  the extra came from formatting, names or money talk."""),
    ("code", """R.fig_methods(r).show()
display(R.checks_table(r))
display(Markdown(say["checks"]))"""),
    ("code", """display(R.ocr_table(r))
display(Markdown(say.get("ocr", "")))"""),
    ("md", """## 5. Other samples and the number of components

The 100k+ tier has hand-checked money but few flops; dropping franchise films removes the sequel effect entirely
instead of controlling for it; the shared sample puts the three corpora on the same films."""),
    ("code", """R.fig_samples(r).show()
display(R.samples_table(r))
display(Markdown(say["samples"]))
display(R.components_table(r))"""),
    ("md", """## 6. Text as a network

The course's second way to join text and networks: link each film to its 10 nearest films by cosine similarity of
the vectors, and ask whether linked films have similar outcomes (Newman's assortativity, the correlation of the
outcome at the two ends of an edge). The null shuffles the outcome across films, inside genre × decade cells for the
residual. This is a check on the regression from a different angle: it uses all 384 dimensions and no fitting."""),
    ("code", """R.fig_network(r).show()
display(Markdown(say["network"]))
display(Markdown(say["agreement"]))"""),
    ("code", """# Local only: needs data/raw and the private film table. Hover a point for the film.
for corpus in R.CORPORA:
    d = R.film_map(corpus)
    if d is None:
        print("No local vectors; run s01 and s02 to see the map.")
        break
    R.fig_map(d, corpus).show()"""),
    ("md", """## 7. Which words go with the direction

For each corpus, films are ranked by their out-of-fold "more profit than the controls predict" score from the text.
The words below are used by a larger share of films in the top fifth than in the bottom fifth (and the reverse). They
describe the direction the regression found; they are not causes."""),
    ("code", """for c in R.CORPORA:
    print(R.LABEL[c])
    display(R.words_table(res["words"], c))"""),
    ("md", """## 8. Do names slip through? Entities used in different ways

The same string can be a name or not: "Paris" is a city and a Hilton, "Christmas" a date and part of a title,
"Bond" a word and a franchise. Each word in the lists above is flagged three ways (`s04_leak_audit.py`):
capitalised in 80% or more of its uses (*), part of a Tag Genome person name or film title, and used by fewer than
30 films (†). A flag is a prompt to read the word, not a verdict: places, agencies and nationalities are
capitalised too and are story content.

Which reading a word gets is decided in context: spaCy's transformer NER reads each candidate in 20 review
sentences and votes on its type (person, organisation, work, place, group, time). It was checked against hand
labels of the 922 content tags of the Tag Genome (person, studio, franchise or none; one person's labels, to be
checked by a second), for spaCy's small and transformer models; the transformer is the one used."""),
    ("code", """import json
typing = json.loads((R.OUTPUTS / "tag_entity_typing.json").read_text())
display(pd.DataFrame({m: {k: v[k] for k in ("precision", "recall", "kappa", "tp", "fp", "fn")} for m, v in typing.items()}).T)
for m, v in typing.items():
    print(m, "misses:", ", ".join(v["missed"]))"""),
    ("code", """for c in R.CORPORA:
    print(R.LABEL[c], "(* capitalised 80%+ of uses, † fewer than 30 films)")
    display(R.leak_table(c))"""),
    ("md", """The same three flags on the Tag Genome tags show why a flag alone is not enough, and that entity tags sit
at the extremes of the tag-profit result in `notebooks/tag_profit.ipynb`: studios and directors there measure who
made the film, which is the network half of the project, not content."""),
    ("code", """flags, extremes = R.tag_entities()
display(flags)
display(extremes)"""),
    ("md", """## 9. What to take from this

The numbers are in the readings above, generated from the saved outputs.

- **Reviews** carry a real signal about profit beyond budget, year, genre and franchise. It passes the shuffle null,
  a second method (TF-IDF finds more, not less), a control for the IMDb rating, and the no-franchise sample. It is
  not a forecast: the reviews are written after release, and fitted on films up to 2009 the gain on later films is
  much smaller, so part of what the model learns is period-specific. The words point at audience-pleasing drama
  ("tearjerker", "heartwarming") doing better than its budget predicts and at genre fare (westerns, apocalyptic
  and futuristic films, "artsy") doing worse.
- **Plot summaries** add a small amount that survives without franchise films.
- **Screenplays** add almost nothing: the small gain on all films vanishes once franchise films are left out, so
  what the scripts carried was mostly whether the film is a sequel. Cleaning matters here: before the OCR filter,
  damaged scans made the screenplay word list.
- **Budget, not profit**: every text predicts budget and revenue far better than profitability. A correlation of
  text with revenue mostly says big films read like big films.

Limits: MovieSum covers mostly well-known films, which skews towards hits, and its scripts mix drafts; name masking
is a set of rules, not a model (common first names are masked as people even when they are characters, and a few
names spaCy reads as places stay); the franchise flag is noisy; the OCR cutoff, the franchise control and the
masking of other films' names were added after the first results; gross over budget isn't profit."""),
]


def main(run: bool = True):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    OUT.write_text(nbf.writes(nb))
    if run:
        from nbconvert.preprocessors import ExecutePreprocessor
        ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
        for c in nb.cells:   # the film maps hold every title and weigh over a megabyte: run them locally
            if c.cell_type == "code" and c.source.startswith("# Local only"):
                c.outputs, c.execution_count = [], None
        OUT.write_text(nbf.writes(nb))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main("--no-run" not in sys.argv)
