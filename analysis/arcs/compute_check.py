"""Step 3 compute check (run BEFORE any full scoring run): count sentences, benchmark each available
instrument on a seeded sample of sentences, project runtime and memory, and apply the pre-registered
fallback rule's runtime condition. Writes outputs/s03_compute_check.json. Scores nothing in full.

    python -m analysis.arcs.compute_check
"""
from __future__ import annotations

import json
import os
import platform
import time

import numpy as np

from . import s02_units
from .config import cfg, log_run
from .paths import OUTPUTS

OUT = OUTPUTS / "s03_compute_check.json"


def _model_cached(name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(name, "config.json"), str)
    except Exception:
        return False


def main() -> dict:
    c = cfg()
    ins = c["instruments"]
    sents = s02_units.load_sentences()
    table = s02_units.load()["table"]
    prim = set(table.movie_name[table.in_primary])
    n_all, n_prim = len(sents), int(sents.movie_name.isin(prim).sum())
    rng = np.random.default_rng(c["seed"])
    sample = sents.text.to_numpy()[rng.choice(n_all, ins["compute_check_sentences"], replace=False)].tolist()
    words = sents.words.to_numpy()

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    t0 = time.perf_counter()
    an = SentimentIntensityAnalyzer()
    load_s = time.perf_counter() - t0
    reps = []
    for _ in range(3):
        t0 = time.perf_counter()
        for s in sample:
            an.polarity_scores(s)
        reps.append((time.perf_counter() - t0) / len(sample))
    per = float(np.median(reps))
    cores = os.cpu_count() or 1
    workers = max(1, cores - 2)
    vader = {"available": True, "load_seconds": round(load_s, 2), "seconds_per_sentence": per,
             "sentences_per_second_one_core": round(1 / per),
             "projected_hours_one_core_all": round(n_all * per / 3600, 3),
             "projected_hours_parallel_all": round(n_all * per / 3600 / workers, 3), "parallel_workers": workers}

    name = ins["contextual"]["name"]
    cached = _model_cached(name)
    contextual = {"name": name, "available": cached}
    if not cached:
        contextual["status"] = ("NOT AVAILABLE: not in the local Hugging Face cache. Downloading it needs the user's "
                                "explicit permission; no throughput can be measured, so no runtime is projected.")
    else:   # benchmark only when the model is present
        import torch
        from transformers import pipeline
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        clf = pipeline("sentiment-analysis", model=name, device=dev, truncation=True)
        clf(sample[:8])
        t0 = time.perf_counter()
        clf(sample, batch_size=ins["contextual"]["batch_size"])
        per_c = (time.perf_counter() - t0) / len(sample)
        hours = n_all * per_c / 3600
        contextual |= {"device": dev, "seconds_per_sentence": per_c, "projected_hours_all": round(hours, 2),
                       "within_limit": hours <= ins["max_runtime_hours"]}

    out = {"date": time.strftime("%Y-%m-%d"), "machine": f"{platform.machine()} {platform.platform()}", "cpu_cores": cores,
           "sentences_all_scripts": int(n_all), "sentences_primary": n_prim,
           "sentence_words": {"median": float(np.median(words)), "p99": float(np.percentile(words, 99)),
                              "max": int(words.max()), "over_350_words": int((words > 350).sum())},
           "benchmark_sentences": len(sample), "max_runtime_hours": ins["max_runtime_hours"],
           "memory_note": "sentence table on disk ~0.1 GB; scores are one float per sentence per instrument (~25 MB)",
           "vader": vader, "contextual": contextual,
           "fallback_decision": ("contextual primary" if contextual.get("within_limit") else
                                 "lexicon (VADER) primary unless the contextual model becomes available and fits the limit")}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    log_run("s03_compute_check", [OUT.name], {"instruments": ins})
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
