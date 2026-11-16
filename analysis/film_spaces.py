"""The clean film sample and every space a film can be placed in, for the embedding-map notebooks.

    from analysis.film_spaces import sample, load_spaces
    d = sample()                 # one row per clean film, indexed by tconst
    S = load_spaces(d)           # {name: (tconst array, matrix)}, rows in the clean sample only

Each space comes from a file another script writes to data/raw/features/:

- Screenplays, MovieSum summaries, Reviews: text_vectors/*_clean.npy from analysis/embeddings/s02_embed.py
  (all-MiniLM-L6-v2, 180-word passages, averaged and normalised).
- Wikipedia plots: plot_vectors.npy from scripts/reviews/plot_embeddings.py.
- Tag relevance, Genre shares: tag_relevance.npy and genre_shares.csv from scripts/reviews/tag_shares.py.
- Tags without reception: tag relevance minus the RECEPTION tags below.
- Cast and crew: graph/credits.csv (IMDb principals: actors, director, writers, producers, editor, composer,
  cinematographer), see cast_crew().
- Sentiment arcs: screenplay_arcs.npy from scripts/screenplays/screenplay_features.py (40 points of mean VADER
  score along the script).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse

from analysis.clean_rule import clean_mask

REPO = Path(__file__).resolve().parents[1]
FEATURES = REPO / "data" / "raw" / "features"
SEED = 1
PRIORITY = ["Horror", "Sci-Fi", "Thriller", "Adventure", "Action", "Comedy", "Drama"]

# Tags that hold the film's reception rather than its content, picked by hand from the tag names before any result
# was computed for them: awards, rankings and lists; cult and classic status; judgements of quality. Every tag that
# starts with "oscar (" is dropped too.
RECEPTION = {
    "afi 100", "afi 100 (movie quotes)", "best war films", "criterion", "funniest movies", "golden palm",
    "good romantic comedies", "imdb top 250", "movielens top pick", "oscar", "oscar winner",
    "saturn award (best special effects)",
    "classic", "cult", "cult classic", "cult film", "scifi cult", "masterpiece", "guilty pleasure",
    "highly quotable", "quotable",
    "amazing cinematography", "amazing photography", "awesome", "awesome soundtrack", "bad", "bad acting", "bad cgi",
    "bad ending", "bad plot", "bad science", "bad script", "bad sequel", "beautifully filmed", "better than expected",
    "better than the american version", "book was better", "boring", "boring!", "breathtaking", "brilliant",
    "crappy sequel", "disappointing", "dumb", "dumb but funny", "excellent", "excellent script", "exceptional acting",
    "good", "good acting", "good action", "good dialogue", "good music", "good sequel", "good soundtrack",
    "good story", "great", "great acting", "great cinematography", "great dialogue", "great ending", "great movie",
    "great music", "great soundtrack", "horrible", "idiotic", "lame", "no plot", "not as good as the first",
    "not funny", "notable soundtrack", "overrated", "perfect", "plot holes", "pointless", "predictable",
    "pretentious", "sad but good", "so bad it's funny", "so bad it's good", "stunning", "stupid", "stupid as hell",
    "stupidity", "suprisingly clever", "too long", "underrated", "unfunny", "unintentionally funny", "very good",
    "very interesting", "visually stunning", "waste of time",
}


def sample(features: Path = FEATURES) -> pd.DataFrame:
    """The clean theatrical sample of analysis/embeddings/config.yaml (2026-11-15), indexed by tconst.

    log_mult_net is ln(multiple) net of budget, year and genre: the residual of an ordinary least-squares fit on
    ln(budget), year, year squared and a dummy for every IMDb genre with 50+ films, over the whole sample.
    """
    m = pd.read_csv(features / "movies_features_10k.csv", dtype={"tconst": str})
    d = m[clean_mask(m)].copy().set_index("tconst")   # the shared clean rule
    d["year"] = d.y.astype(int)
    d["title"] = d.primaryTitle
    d["log_mult"] = np.log(d.g / d.b)
    d["primary_genre"] = d.genres.fillna("").map(lambda g: next((x for x in PRIORITY if x in g.split(",")), "Other"))
    G = d.genres.fillna("").str.get_dummies(sep=",")
    G = G[[c for c in G if G[c].sum() >= 50]]
    yr = (d.year - d.year.mean()) / d.year.std()
    X = np.column_stack([np.ones(len(d)), np.log(d.b), yr, yr ** 2, G.to_numpy()])
    beta, *_ = np.linalg.lstsq(X, d.log_mult.to_numpy(), rcond=None)
    d["log_mult_net"] = d.log_mult - X @ beta
    d.attrs["net_r2"] = float(1 - d.log_mult_net.var() / d.log_mult.var())
    d.attrs["genre_dummies"] = G.shape[1]
    return d


def reception_mask(tags: pd.Series) -> np.ndarray:
    missing = RECEPTION - set(tags)
    assert not missing, missing
    return (tags.isin(RECEPTION) | tags.str.startswith("oscar (")).to_numpy()


def cast_crew(d: pd.DataFrame, features: Path = FEATURES, dims: int = 100, min_films: int = 2):
    """Each film as the people who made it: a film x person matrix over the clean sample, TF-IDF weighted so
    someone on many films counts less, reduced to `dims` dimensions with truncated SVD.

    Only people on min_films+ clean films are kept, since someone on one film makes no two films alike. Films left
    with nobody are dropped. Who made a film is known before release and no outcome enters the vectors, but the SVD
    is fitted on every film at once, so a film's neighbours can be later films.
    """
    c = pd.read_csv(features / "graph" / "credits.csv", dtype={"tconst": str, "nconst": str},
                    usecols=["tconst", "nconst"]).drop_duplicates()
    c = c[c.tconst.isin(d.index)]
    n = c.nconst.value_counts()
    c = c[c.nconst.isin(n[n >= min_films].index)]
    films, fi = np.unique(c.tconst, return_inverse=True)
    people, pi = np.unique(c.nconst, return_inverse=True)
    M = sparse.csr_matrix((np.ones(len(c)), (fi, pi)), shape=(len(films), len(people)))
    W = TfidfTransformer(sublinear_tf=False).fit_transform(M)
    V = TruncatedSVD(dims, random_state=SEED).fit_transform(W)
    V /= np.linalg.norm(V, axis=1, keepdims=True).clip(1e-12)
    return films, V, {"people": len(people), "credits": len(c)}


def load_spaces(d: pd.DataFrame, features: Path = FEATURES) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{name: (tconst array, float matrix)} for every space, keeping only films in d, in a fixed order."""
    tv = features / "text_vectors"
    raw: dict[str, tuple] = {}
    for name, stem in (("Screenplays", "screenplays"), ("MovieSum summaries", "summaries")):
        raw[name] = (pd.read_csv(tv / f"{stem}_ids.csv", dtype={"tconst": str}).tconst, np.load(tv / f"{stem}_clean.npy"))
    raw["Wikipedia plots"] = (pd.read_csv(features / "plot_vector_ids.csv", dtype={"tconst": str}).tconst,
                              np.load(features / "plot_vectors.npy"))
    raw["Reviews"] = (pd.read_csv(tv / "reviews_ids.csv", dtype={"tconst": str}).tconst, np.load(tv / "reviews_clean.npy"))
    tag_ids = pd.read_csv(features / "tag_relevance_ids.csv", dtype={"tconst": str}).tconst
    T = np.load(features / "tag_relevance.npy")
    drop = reception_mask(pd.read_csv(features / "tag_relevance_tags.csv").tag)
    raw["Tag relevance"] = (tag_ids, T)
    raw["Tags without reception"] = (tag_ids, T[:, ~drop])
    g = pd.read_csv(features / "genre_shares.csv", dtype={"tconst": str})
    raw["Genre shares"] = (g.tconst, g[[c for c in g.columns if c.startswith("share_")]].to_numpy())
    films, V, _ = cast_crew(d, features)
    raw["Cast and crew"] = (films, V)
    s = pd.read_csv(features / "screenplay_features.csv", dtype={"tconst": str})
    raw["Sentiment arcs"] = (s[s.has_arc == 1].tconst, np.load(features / "screenplay_arcs.npy"))

    out = {}
    for name, (ids, V) in raw.items():
        ids = pd.Series(ids, dtype=str).reset_index(drop=True)
        assert len(ids) == len(V) and not ids.duplicated().any(), name
        keep = ids.isin(d.index).to_numpy()
        out[name] = (ids[keep].to_numpy(), np.asarray(V, dtype=float)[keep])
    return out
