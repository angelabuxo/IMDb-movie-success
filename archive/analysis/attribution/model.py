"""The maths of the profit attribution, with no file reading, so the tests can run it on made-up films.

The model. A film's return y = ln(gross / budget) is written as a sum:

    y = intercept + controls (budget, year, genre, franchise) + one effect per person and company + residual

Every director, top-billed actor, writer, producer, composer, cinematographer, editor, studio and distributor
with at least two films in the sample gets its own effect, and all of them are fitted together by ridge
regression, each role with its own penalty. This is the adjusted plus-minus used for basketball players, or a
crossed random-effects model read as its best linear unbiased predictions: an actor's effect is what is left of
their films' returns after the director, the studio, the budget and everyone else have taken their part, pulled
towards zero when there are few films to go on.

Because y is a log, effects multiply: an effect of +0.10 means gross / budget is exp(0.10) = 1.105 times what it
would be with an average person in that seat. `shapley_product` turns the factors of one film into dollars that
add up exactly to the film's gross minus a baseline.

What this can't do: it describes, it doesn't identify causes. People aren't assigned to films at random, so an
effect also holds whatever comes with that person and isn't in the model (the projects they are offered, release
dates, marketing).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge

ROLES = ["director", "cast", "writer", "producer", "composer", "cinematographer", "editor", "studio", "distributor"]
CONTROLS = ["budget", "year", "genre", "franchise"]
CONTROL_PENALTY = 1e-3


@dataclass
class Design:
    """A films x columns matrix with, for every column, the block it belongs to and what it stands for."""
    X: sparse.csr_matrix
    block: np.ndarray            # block name per column: a control or a role
    key: np.ndarray              # entity id for role columns, a label for control columns
    films: np.ndarray            # film id per row
    extra: dict = field(default_factory=dict)

    def cols(self, block: str) -> np.ndarray:
        return np.flatnonzero(self.block == block)


def build_design(films, credits, min_films: int = 2, roles=ROLES) -> Design:
    """films: DataFrame indexed by film id with log_budget, year, genres (comma separated), franchise (0/1).
    credits: DataFrame with film, entity, role (one row per credit, already cut to the top-billed cast).
    An entity gets a column for a role when it has that role on at least `min_films` films."""
    import pandas as pd

    ids = np.asarray(films.index)
    row = {t: i for i, t in enumerate(ids)}
    n = len(ids)
    mats, block, key = [], [], []

    lb = films.log_budget.to_numpy(float)
    lbc = lb - lb.mean()
    mats.append(sparse.csr_matrix(np.column_stack([lbc, lbc ** 2 - (lbc ** 2).mean()])))
    block += ["budget", "budget"]
    key += ["ln(budget)", "ln(budget) squared"]

    yr = films.year.to_numpy(float)
    decade = (yr // 10 * 10).astype(int)
    D = pd.get_dummies(decade).astype(float)
    mats.append(sparse.csr_matrix(np.column_stack([(yr - yr.mean()) / 10.0, D.to_numpy()])))
    block += ["year"] * (1 + D.shape[1])
    key += ["year (per decade)"] + [f"{d}s" for d in D.columns]

    G = films.genres.fillna("").str.get_dummies(sep=",").astype(float)
    G = G.loc[:, G.sum() >= 5]
    mats.append(sparse.csr_matrix(G.to_numpy()))
    block += ["genre"] * G.shape[1]
    key += list(G.columns)

    mats.append(sparse.csr_matrix(films.franchise.fillna(0).to_numpy(float).reshape(-1, 1)))
    block.append("franchise")
    key.append("franchise")

    c = credits[credits.film.isin(row)].drop_duplicates(["film", "entity", "role"])
    for r in roles:
        cr = c[c.role == r]
        counts = cr.entity.value_counts()
        keep = np.sort(counts.index[counts >= min_films].to_numpy())
        col = {e: j for j, e in enumerate(keep)}
        cr = cr[cr.entity.isin(col)]
        M = sparse.csr_matrix((np.ones(len(cr)), (cr.film.map(row).to_numpy(), cr.entity.map(col).to_numpy())),
                              shape=(n, len(keep)))
        mats.append(M)
        block += [r] * len(keep)
        key += list(keep)
    return Design(sparse.hstack(mats, format="csr"), np.array(block, dtype=object), np.array(key, dtype=object), ids)


def penalties_vector(design: Design, penalties: dict) -> np.ndarray:
    """One ridge penalty per column. Controls get a tiny one, enough to keep collinear dummies solvable."""
    return np.array([penalties.get(b, CONTROL_PENALTY) for b in design.block], dtype=float)


def fit(design: Design, y: np.ndarray, penalties: dict, rows=None, blocks=None) -> tuple[float, np.ndarray]:
    """Ridge with a penalty per column, fitted on `rows` (default all) and the columns in `blocks` (default all).
    Returns the intercept and a coefficient per column of the full design (zero for left-out columns).

    Scaling column j by 1/sqrt(penalty_j) and fitting with penalty 1 is the same problem."""
    lam = penalties_vector(design, penalties)
    use = np.ones(len(lam), bool) if blocks is None else np.isin(design.block, list(blocks))
    use &= np.isfinite(lam)
    beta = np.zeros(len(lam))
    X = design.X if rows is None else design.X[rows]
    yy = y if rows is None else y[rows]
    if not use.any():
        return float(yy.mean()), beta
    scale = 1.0 / np.sqrt(lam[use])
    Xs = X[:, use] @ sparse.diags(scale)
    m = Ridge(alpha=1.0, solver="sparse_cg", fit_intercept=True, tol=1e-6, max_iter=5000).fit(Xs, yy)
    beta[use] = m.coef_ * scale
    return float(m.intercept_), beta


def grouped_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """A fold number per film, with every film of a group (a franchise) in the same fold."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % k))
    return np.array([fold_of[g] for g in groups])


def out_of_fold(design: Design, y: np.ndarray, penalties: dict, folds: np.ndarray, blocks=None):
    """For every film, the intercept and coefficients fitted without its fold. Returns (prediction per film,
    intercept per film, a films x columns sparse matrix of each column's contribution to its film)."""
    n = len(y)
    pred, icpt = np.zeros(n), np.zeros(n)
    parts = []
    for f in np.unique(folds):
        te = np.flatnonzero(folds == f)
        a, beta = fit(design, y, penalties, rows=np.flatnonzero(folds != f), blocks=blocks)
        C = design.X[te] @ sparse.diags(beta)
        pred[te] = a + np.asarray(C.sum(axis=1)).ravel()
        icpt[te] = a
        parts.append((te, C))
    order = np.concatenate([te for te, _ in parts])
    C = sparse.vstack([c for _, c in parts], format="csr")[np.argsort(order)]
    return pred, icpt, C


def forward(design: Design, y: np.ndarray, penalties: dict, year: np.ndarray, first_year: int):
    """The strict version: a film's effects come only from films released in earlier years. Films before
    `first_year` get no prediction (NaN). Returns the same three things as `out_of_fold`."""
    n = len(y)
    pred, icpt = np.full(n, np.nan), np.full(n, np.nan)
    C = sparse.lil_matrix(design.X.shape)
    for yv in sorted(set(year[year >= first_year])):
        te, tr = np.flatnonzero(year == yv), np.flatnonzero(year < yv)
        a, beta = fit(design, y, penalties, rows=tr)
        c = design.X[te] @ sparse.diags(beta)
        C[te] = c
        pred[te] = a + np.asarray(c.sum(axis=1)).ravel()
        icpt[te] = a
    return pred, icpt, C.tocsr()


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ok = np.isfinite(pred)
    return float(1 - ((y[ok] - pred[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())


def cv_r2(design: Design, y: np.ndarray, penalties: dict, folds: np.ndarray, blocks=None) -> float:
    return r2(y, out_of_fold(design, y, penalties, folds, blocks)[0])


def tune(design: Design, y: np.ndarray, folds: np.ndarray, roles, grid, start: float = 30.0, passes: int = 2,
         log=None) -> tuple[dict, list]:
    """One penalty per role by coordinate descent on cross-validated R2. np.inf in the grid switches a role off.
    Returns the penalties and the path of (role, penalty, R2) tried."""
    pen = {r: start for r in roles}
    best = cv_r2(design, y, pen, folds)
    path = []
    for p in range(passes):
        for r in roles:
            for g in [g for g in grid if g != pen[r]]:
                trial = {**pen, r: g}
                s = cv_r2(design, y, trial, folds)
                path.append((r, g, s))
                if s > best + 1e-6:
                    best, pen = s, trial
            if log:
                log(f"pass {p + 1}, {r}: penalty {pen[r]:g}, cross-validated R2 {best:.4f}")
    return pen, path


def block_contributions(design: Design, C: sparse.csr_matrix, blocks) -> dict:
    """Sum the per-column contributions of each film inside every block."""
    return {b: np.asarray(C[:, design.cols(b)].sum(axis=1)).ravel() for b in blocks}


def shapley_blocks(value, players) -> dict:
    """Exact Shapley values of `value(frozenset of players) -> float`, by enumerating every coalition."""
    from itertools import combinations
    from math import factorial

    n = len(players)
    v = {frozenset(s): value(frozenset(s)) for k in range(n + 1) for s in combinations(players, k)}
    out = {}
    for p in players:
        rest = [q for q in players if q != p]
        tot = 0.0
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for s in combinations(rest, k):
                tot += w * (v[frozenset(s) | {p}] - v[frozenset(s)])
        out[p] = tot
    return out, v


_GL_T, _GL_W = np.polynomial.legendre.leggauss(64)
_GL_T, _GL_W = (_GL_T + 1) / 2, _GL_W / 2


def shapley_product(base: float, terms: np.ndarray) -> np.ndarray:
    """Split base * (exp(sum(terms)) - 1) between the terms.

    A film's gross is base * exp(t_1) * ... * exp(t_n). With a product there is no single answer to "how many
    dollars is term i worth": it depends on which other terms are already switched on. The Shapley value
    averages a term's gain over every order in which the terms could be switched on. For a product it has a
    closed form, phi_i = base * (f_i - 1) * integral over t in [0, 1] of prod_{j != i} (1 + t (f_j - 1)), with
    f = exp(term), which is a polynomial and so exact under Gauss-Legendre quadrature (64 nodes cover 127
    terms). The parts sum to base * (prod f - 1)."""
    terms = np.asarray(terms, float)
    if len(terms) > 127:
        raise ValueError("more than 127 terms")
    g = np.expm1(terms)
    one = 1 + np.outer(_GL_T, g)                      # nodes x terms, every entry > 0
    total = np.exp(np.log(one).sum(axis=1))
    return base * g * (_GL_W[:, None] * total[:, None] / one).sum(axis=0)


def split_half(design: Design, y: np.ndarray, penalties: dict, groups: np.ndarray, roles, reps: int, seed: int,
               min_each: int = 3) -> dict:
    """Fit the effects on two random halves of the films and correlate them, for entities with at least
    `min_each` films in each half. Near zero means a role's effects are mostly noise; near one, stable."""
    out = {r: [] for r in roles}
    counts = {r: [] for r in roles}
    for k in range(reps):
        half = grouped_folds(groups, 2, seed + k)
        a_rows, b_rows = np.flatnonzero(half == 0), np.flatnonzero(half == 1)
        _, ba = fit(design, y, penalties, rows=a_rows)
        _, bb = fit(design, y, penalties, rows=b_rows)
        na = np.asarray(design.X[a_rows].sum(axis=0)).ravel()
        nb = np.asarray(design.X[b_rows].sum(axis=0)).ravel()
        for r in roles:
            c = design.cols(r)
            c = c[(na[c] >= min_each) & (nb[c] >= min_each)]
            counts[r].append(len(c))
            if len(c) >= 10 and ba[c].std() > 0 and bb[c].std() > 0:
                out[r].append(float(np.corrcoef(ba[c], bb[c])[0, 1]))
    return {r: {"mean_r": float(np.mean(v)) if v else None, "sd_r": float(np.std(v)) if v else None,
                "entities": int(np.mean(counts[r])) if counts[r] else 0, "splits": len(v)} for r, v in out.items()}


def shuffle_within(y: np.ndarray, strata: np.ndarray, rng) -> np.ndarray:
    """Shuffle y among films of the same stratum (release year): keeps each year's level, breaks who made what."""
    out = y.copy()
    for s in np.unique(strata):
        i = np.flatnonzero(strata == s)
        out[i] = y[rng.permutation(i)]
    return out
