"""Four 2D maps of one set of film vectors, each scored against the full vectors.

Used by notebooks/embedding_maps.ipynb. Every method gets the same input and the same seed:

- PCA: linear, keeps the directions of largest spread, so large distances survive.
- Isomap: distances along a graph of each film's 15 nearest films.
- t-SNE and UMAP: keep who sits next to whom and give up on large distances, so cluster sizes and the gaps
  between clusters mean little on their maps.

Isomap, t-SNE and UMAP run on the first 50 principal components (or every dimension, if there are fewer).

Each map is scored on three things:

- neighbours kept: the share of each film's k nearest films in the full space that are also among its k
  nearest on the map. Chance is about k / number of films.
- distance rank: the Spearman correlation between pairwise distances in the full space and on the map, over
  n_pairs random films.
- profit smoothness: the Spearman correlation between a film's outcome and the mean outcome of its k nearest
  films. compare() also gives it for the full space, with a shuffle null (outcomes permuted between films,
  neighbours kept), so a map that makes profit look more clustered than the vectors are shows up. Given a second
  outcome y_net (the multiple net of budget, year and genre), it scores that too, next to the raw one.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.neighbors import NearestNeighbors

with warnings.catch_warnings():   # umap pulls in tqdm, which warns about missing ipywidgets
    warnings.simplefilter("ignore")
    import umap

GENRE_COLOURS = {"Horror": "#d62728", "Sci-Fi": "#9467bd", "Thriller": "#8c564b", "Adventure": "#ff7f0e",
                 "Action": "#1f77b4", "Comedy": "#e7ba52", "Drama": "#2ca02c", "Other": "#bbbbbb"}


def knn(X: np.ndarray, k: int = 10) -> np.ndarray:
    return NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]


def project(V: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    Z = PCA(min(50, V.shape[1]), random_state=seed).fit_transform(V)
    return {
        "PCA": PCA(2, random_state=seed).fit_transform(V),
        "Isomap": Isomap(n_neighbors=15, n_components=2).fit_transform(Z),
        "t-SNE": TSNE(2, perplexity=30, init="pca", random_state=seed).fit_transform(Z),
        "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed, n_jobs=1,
                          compatibility_layout=True).fit_transform(Z),
    }


def smoothness(y: np.ndarray, nb: np.ndarray) -> float:
    return float(stats.spearmanr(y, y[nb].mean(1)).statistic)


def compare(V: np.ndarray, y: np.ndarray, genre: np.ndarray, title: str, y_net: np.ndarray | None = None,
            k: int = 10, seed: int = 1, n_pairs: int = 3000,
            perms: int = 200) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Draw the four maps (top row coloured by y, bottom by genre), print the scores, return them and the maps."""
    maps = project(V, seed)
    nv = knn(V, k)
    rng = np.random.default_rng(seed)
    full = smoothness(y, nv)
    null = [smoothness(rng.permutation(y), nv) for _ in range(perms)]
    if y_net is not None:
        full_net = smoothness(y_net, nv)
        null_net = [smoothness(rng.permutation(y_net), nv) for _ in range(perms)]
    pairs = rng.choice(len(V), min(n_pairs, len(V)), replace=False)
    dv = pdist(V[pairs])

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.8))
    rows = []
    for j, (name, Y) in enumerate(maps.items()):
        ny = knn(Y, k)
        kept = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nv, ny)]))
        dist = float(stats.spearmanr(dv, pdist(Y[pairs])).statistic)
        rows.append({"space": title, "films": len(V), "dims": V.shape[1], "method": name, "neighbours kept": kept,
                     "distance rank": dist, "profit smoothness": smoothness(y, ny), "full-space smoothness": full,
                     "null 97.5%": float(np.percentile(null, 97.5)), "chance kept": k / (len(V) - 1)})
        if y_net is not None:
            rows[-1].update({"net smoothness": smoothness(y_net, ny), "full-space net smoothness": full_net,
                             "net null 97.5%": float(np.percentile(null_net, 97.5))})
        lo, hi = np.percentile(Y, [0.5, 99.5], axis=0)   # a few outliers would otherwise squash the cloud
        pad = 0.05 * (hi - lo)
        sc = axes[0, j].scatter(Y[:, 0], Y[:, 1], c=y, cmap="viridis", s=3, vmin=-2, vmax=3, alpha=0.7)
        for g, col in GENRE_COLOURS.items():
            m = genre == g
            axes[1, j].scatter(Y[m, 0], Y[m, 1], color=col, s=3, alpha=0.7, label=f"{g} ({m.sum():,})")
        for ax in axes[:, j]:
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, j].set_title(f"{name}\nneighbours kept {kept:.2f} | distance rank {dist:.2f}", loc="left", fontsize=9)
    axes[0, 0].set_ylabel("coloured by ln(gross / budget)"); axes[1, 0].set_ylabel("coloured by primary genre")
    fig.colorbar(sc, ax=axes[0, :], label="ln(gross / budget)", shrink=0.8, pad=0.01)
    axes[1, 3].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, markerscale=3, fontsize=8)
    fig.suptitle(f"{title}: {len(V):,} films, {V.shape[1]:,} dimensions, four projections "
                 f"(axes cut at the 0.5th and 99.5th percentiles)", x=0.02, ha="left")
    plt.show()

    out = pd.DataFrame(rows)
    print(f"chance level for neighbours kept: {k / (len(V) - 1):.4f}; profit smoothness in the full space: {full:.3f} "
          f"(shuffled outcomes, 95% range {np.percentile(null, 2.5):+.3f} to {np.percentile(null, 97.5):+.3f})")
    cols = ["neighbours kept", "distance rank", "profit smoothness"]
    if y_net is not None:
        print(f"net of budget, year and genre: full space {full_net:.3f} (shuffled, 95% range "
              f"{np.percentile(null_net, 2.5):+.3f} to {np.percentile(null_net, 97.5):+.3f})")
        cols.append("net smoothness")
    print(out.set_index("method")[cols].round(3).to_string())
    return out, maps
