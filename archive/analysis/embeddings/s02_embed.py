"""Step 2: one vector per film for each corpus, from the raw and the cleaned text.

    python -m analysis.embeddings.s02_embed            # all three corpora
    python -m analysis.embeddings.s02_embed reviews    # one

Each text is cut into passages of 180 words (the model reads about 190), every passage is embedded and
L2-normalised, and a film's vector is the normalised mean of its passages. For reviews the mean is taken
within each review first, so a long review counts the same as a short one.

Writes data/raw/features/text_vectors/<corpus>_<variant>.npy with the row order in <corpus>_ids.csv,
and outputs/s02_embedding.json. Vectors only, no text.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import clean as C
from .paths import OUTPUTS, PRIVATE, VECTORS, cfg

CORPORA = ("reviews", "screenplays", "summaries")
VARIANTS = ("raw", "clean")


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def embed(model, texts: list[str], e: dict) -> tuple[np.ndarray, int]:
    """One normalised vector per text: the mean over its passages."""
    owner, chunks = [], []
    for i, t in enumerate(texts):
        ps = C.passages(t, e["passage_words"], e["min_last_passage_words"])
        owner += [i] * len(ps)
        chunks += ps
    order = np.argsort([len(c) for c in chunks])[::-1]          # similar lengths batch together
    vec = np.empty((len(chunks), model.get_sentence_embedding_dimension()), dtype=np.float32)
    vec[order] = model.encode([chunks[i] for i in order], batch_size=e["batch_size"], normalize_embeddings=True,
                              show_progress_bar=False)
    out = np.zeros((len(texts), vec.shape[1]), dtype=np.float64)
    np.add.at(out, np.array(owner), vec)
    return unit(out), len(chunks)


def run(corpora=CORPORA) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer
    e = cfg()["embed"]
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(e["model"], device=device)
    VECTORS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / "s02_embedding.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update({"model": e["model"], "device": device, "passage_words": e["passage_words"],
                   "max_seq_length": model.max_seq_length})
    for corpus in corpora:
        df = pd.read_parquet(PRIVATE / f"{corpus}.parquet", columns=["tconst", "raw", "clean"])
        ids = sorted(df.tconst.unique())
        pos = {t: i for i, t in enumerate(ids)}
        pd.DataFrame({"tconst": ids}).to_csv(VECTORS / f"{corpus}_ids.csv", index=False)
        for variant in VARIANTS:
            v, n = embed(model, df[variant].tolist(), e)
            film = np.zeros((len(ids), v.shape[1]))
            np.add.at(film, df.tconst.map(pos).to_numpy(), v)   # one row per film already, except for reviews
            np.save(VECTORS / f"{corpus}_{variant}.npy", unit(film).astype(np.float32))
            report[f"{corpus}_{variant}"] = {"films": len(ids), "texts": len(df), "passages": n}
            print(corpus, variant, report[f"{corpus}_{variant}"], flush=True)
    path.write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    import sys
    run(tuple(a for a in sys.argv[1:] if a in CORPORA) or CORPORA)
