# Profit attribution

This is an optional analysis. It asks how a film's return splits up between its budget, genre and franchise, the people who made it, the companies behind it, and what's left over. Nothing else in the repo reads its outputs, so the group can use it in the paper or leave it out.

The decisions and assumptions behind it are also logged in the group's Google Doc ("Social Graphs Final Project", section "Decisions and assumptions"), which is the one place we record them. Every number below comes from [`analysis/attribution/outputs/summary.json`](../analysis/attribution/outputs/summary.json), written by the run of 16 November 2026 on the 6,413 clean films (1975 to 2025).

## Run it

From the repo root, with the movie graph in `data/raw` (about 17 minutes):

```bash
python -m analysis.attribution.run
```

```bash
python -m analysis.attribution.build_notebook
```

The first writes four files to `analysis/attribution/outputs/`. The second rebuilds [`notebooks/profit_attribution.ipynb`](../notebooks/profit_attribution.ipynb) from them. The explorer's **Profit attribution** page reads the same files. The notebook and the page need no raw data.

| File | What's in it |
|---|---|
| `summary.json` | Sample, penalties, R², each block's share, the nulls, reliability, the profit measures |
| `entities.csv` | 14,104 effects: one row per person or company and role, with a bootstrap interval and dollars |
| `films.csv` | Every film's return split into blocks, in logs and in dollars, fitted two ways |
| `film_entities.csv.gz` | The same split down to each person and company on each film |

## Profit measures

We have no real profit for any film. Between the budget and the worldwide gross, two things are missing: the cinemas' cut of the gross, and marketing, which isn't in the budget. That leaves three measures, none of them right.

- Gross ÷ budget. It ignores inflation and film size, and it favours cheap films.
- Gross − budget in dollars. It favours big films and counts all of the gross as the studio's.
- Share × gross − budget, where the share stands for what's left after the cinemas' cut. The share isn't published per film, so it's an assumption. The run saves results at 40%, 50% and 60%, and the explorer page has a slider.

At a 50% share, 51.3% of films clear break-even (28.0% gross less than their budget outright), and the top tenth of films earns 69.2% of all gains. The measures agree on the order of most films (Spearman 0.861 between the multiple and estimated dollars) and disagree on the winners: only 36.8% of the top tenth by multiple is also in the top tenth by dollars. The top tenth by multiple has a median budget of $6.0M, and the top tenth by dollars $71.5M.

The model uses ln(gross ÷ budget). The dollar figures are splits of gross, which you scale by the share you believe.

## What published papers do

We checked how papers handle the same gap between gross and profit. "Read" means we opened the paper or filing. "Second-hand" means another paper's description of it.

| Question | What they do | Source |
|---|---|---|
| Outcome | Most model revenue, usually log box office with log budget as a predictor. Those that model profit use ROI = (gross − budget) ÷ budget | Lash & Zhao 2016 and Lee & Lim 2018 (read); Walls 2005 for log revenue with year dummies (read) |
| Cinemas' cut | Profit = 0.5 × box office − budget, on domestic box office | De Vany & Walls 1999, second-hand from Jansen 2002, footnote 3 |
| Cinemas' cut, measured | Cinemark paid 53.9% of its admissions revenue in film rentals and advertising in 2013 (53.5% in 2012): 55.8% in the US and 48.9% abroad | [Cinemark's 10-K for 2013](https://www.sec.gov/Archives/edgar/data/0001385280/000119312514077445/d641540d10k.htm) (read) |
| Marketing | None of the profit papers we opened includes it. We found no free per-film dataset of marketing cost or net profit | Lash & Zhao 2016, Lee & Lim 2018 (read) |
| Inflation | Year dummies, or nothing | Walls 2005; Lash & Zhao 2016 and Lee & Lim 2018 use nominal figures (read) |
| Other income | Named and left out | Lash & Zhao 2016 (read) |
| Break-even | A rule that a film grossing twice its budget worldwide was "very likely" profitable held for 72% to 75% of films with known accounts | [Stephen Follows' analysis](https://stephenfollows.com/hollywood-movies-make-a-profit/) (read; not peer reviewed) |

So our choices are the usual ones. ln(gross ÷ budget) is the log of the ratio Ravid (1999, second-hand) and the ROI papers use, and the 50% default is De Vany and Walls' rule. Cinemark's filings put the measured US figure a little higher, at about 55%, and that line also holds the cinema's own advertising, so it slightly overstates the rental rate. A fixed share only shifts ln(gross ÷ budget) by a constant, so it moves the break-even line and the dollar figures and leaves every effect and ranking in the model as it is.

What would make the share real per film is a split of each film's gross into US, international and China, with a rate for each. We don't have that split: the infobox gives one worldwide figure.

### What we couldn't make real

- **Marketing.** No free per-film data exists. We found industry-wide averages up to 2007 only in secondary sources and couldn't confirm them from a primary one, so they aren't used.
- **A share per film.** See above: it needs the gross split by territory.
- **A share per year.** Cinemark's and AMC's annual filings would give one. We've only read the 2013 filing so far.
- **Papers we couldn't open.** De Vany & Walls (2004), Eliashberg, Elberse & Leenders (2006) and Litman (1983), so nothing here cites them for how profit is measured.

## The model

A film's ln(gross ÷ budget) is written as a sum:

> intercept + budget + year + genre + franchise + one effect per person and company + what's left

The people are the director, the three top-billed actors, writers, producers, composer, cinematographer and editor. The companies are the studios and distributors. Anyone with at least two films in the sample gets an effect for that role. All the effects are fitted at once by ridge regression, so an actor's effect is what remains of their films' returns after the director, the studio, the budget and everyone else have taken their part. Sports analysts know this model as adjusted plus-minus.

Each role has its own penalty, picked by cross-validation: 10 for distributors, 20 for directors and studios, 30 for cast and writers, 50 for producers, 100 for composers, cinematographers and editors. A heavier penalty means the data can tell individuals in that role apart less well.

Because the outcome is a log, effects multiply. An effect of +0.10 means the film grosses about 1.105 times what it would with an average person in that seat.

### Two versions of every split

- **Out of fold** (`ln_*` and `usd_*` in `films.csv`). The effects used for a film are fitted without it and without the other films of its franchise, on films from every year. It answers "how do this director's other films do?".
- **Earlier years only** (`fwd_*`). The effects are fitted on films released in earlier years. It answers "what could have been known before release?", and it's the one to use for prediction. It starts in 1990.

### Dollars

In logs the parts add. In dollars they multiply, so a director's dollar value depends on which other factors are counted first. We use the Shapley value, which averages over every order. For a product it has an exact formula (`shapley_product` in `model.py`), and a film's parts add up to its gross minus a baseline to within a dollar. The baseline is the film's budget times the average multiple.

## What it finds

Cross-validated R² rises from 0.115 with budget, year and genre to 0.299 with franchise, people and companies. 71.0% of the variance in returns stays unexplained.

The gain splits as follows (Shapley shares of cross-validated R², which sum to the gain of 0.181):

| Block | Share | Added alone | Lost when dropped | Split-half r |
|---|---:|---:|---:|---:|
| Franchise | 0.068 | 0.071 | 0.066 | |
| Distributor | 0.054 | 0.071 | 0.042 | 0.62 |
| Studio | 0.027 | 0.051 | 0.012 | 0.40 |
| Producer | 0.012 | 0.029 | 0.001 | 0.27 |
| Cast | 0.008 | 0.010 | 0.008 | 0.16 |
| Writer | 0.004 | 0.009 | 0.001 | 0.18 |
| Director | 0.004 | 0.009 | 0.001 | 0.21 |
| Composer | 0.002 | 0.004 | 0.001 | 0.11 |
| Editor | 0.001 | 0.004 | 0.000 | 0.04 |
| Cinematographer | 0.001 | 0.003 | 0.000 | 0.11 |

Companies and the franchise flag carry most of what can be attributed. Directors, cast and writers each add under one point of R². From earlier years only, R² is 0.219 for films from 1990 on, against 0.059 for budget, year and genre.

## Checks

- **Shuffle null.** Returns shuffled among films of the same release year, 200 times. The model's gain over budget, year and genre is +0.181 on the real returns and at most -0.026 on shuffled ones. Each block added alone beats all 200 of its shuffles.
- **Split halves.** Effects fitted on two random halves of the films, 50 times, and correlated for people or companies with at least three films in each half (the last column above). Distributor effects repeat (0.62). Studio effects partly repeat (0.40). Individual people's effects mostly don't: 0.21 for directors, 0.16 for cast.
- **Bootstrap.** Films resampled by franchise group, 200 times. A 90% interval excludes zero for 13% of director effects, 11% of cast effects and 17% of distributor effects, where chance alone gives about 10%. Ridge intervals aren't exactly calibrated, so read this as a rough guide.
- **A second method.** The shrunk mean of each entity's residuals, ignoring collaborators. Its ranking agrees with the joint model's at Spearman 0.89 to 0.92 in every role, and its effects are more spread out (directors: sd 0.112 against 0.075), which is the part collaborators account for.
- **Unit tests.** `tests/test_attribution.py` plants known effects in made-up films and checks they're recovered, that a film's own return never reaches its own split, that the earlier-years version ignores later films, and that the dollar formula matches brute force.

## What it measures and what it misses

It measures who goes with high returns once everyone else on the film is accounted for. It doesn't measure why.

- **People aren't assigned to films at random.** A star gets better offers, and a large distributor picks films it expects to sell. All of that lands in their effect.
- **The distributor effect is mostly release size.** That's our reading, not something the data shows: the lowest distributors (Vertical Entertainment, Gravitas Ventures, Magnolia Pictures) are ones we know for limited and video-on-demand releases, and the highest are the major studios' distribution arms. The model doesn't see screen counts, so it can't separate the two.
- **Individual names are noisy.** The block of directors helps predict a little, but a ranking of directors would mostly rank noise (split-half r 0.21). The same goes for cast and writers. We'd keep leaderboards of people out of the paper, or show them with their intervals.
- **Collaborators who always work together can't be told apart.** The model gives them equal effects.
- **People with one film in the sample have no effect.** Their part sits in the unexplained term.
- **Inflation is handled with US prices only.** The saved dollars are nominal. The notebook and the explorer page convert them to 2025 dollars with the US consumer price index of the release year ([`data/cpi_us_annual.csv`](../../data/cpi_us_annual.csv), BLS CPI-U from FRED; `python -m analysis.attribution.cpi` refreshes it). A 1975 dollar counts as 5.98 dollars of 2025. Budgets are spent before release and a worldwide gross isn't all earned in the US, so it's an approximation. It changes rankings of films (16 of the top 25 by 50% of gross minus budget are the same in nominal and real dollars) and barely changes the blocks' shares of dollars. 2025 is the mean of eleven months, because no October figure was published.
- **Gross isn't profit.** The share slider stands in for the cinemas' cut, and marketing, home video and television are missing whatever the share.
- **Credits are partial.** IMDb's principals file lists about ten people a title.
