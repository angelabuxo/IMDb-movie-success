"""Contextual sentiment scores for the arc analysis, run as a GPU batch job on DTU HPC.

    python score_contextual.py IN_SENTENCES.jsonl.gz OUT_DIR --models sst2,roberta3 [--benchmark-only]

Input: the private sentence table from Step 2 (movie_name, scene, layer, pos, text). Script text never
leaves the owner-only HPC folder; the outputs are numbers only.

Two configurations per model (pre-registered in config.yaml):
  sentence  every sentence scored alone (the primary unit; scenes are the mean of their sentences later)
  scene     each scene's 'both' text, in order, cut at sentence boundaries into pieces of at most 510 word
            pieces; each piece scored; the scene score is the mean over pieces weighted by their length.
            This lets the model see context (the week-7 chunking point).
Score = P(positive) - P(negative), in [-1, 1]. The three-class model also gives P(neutral).

The job first times the pre-registered benchmark (200 sentences drawn with the config seed), writes the
projected runtime, and stops before the full run if the projection exceeds --max-hours.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = {"sst2": "distilbert-base-uncased-finetuned-sst-2-english",
          "roberta3": "cardiffnlp/twitter-roberta-base-sentiment-latest"}
MAX_PIECES = 510
SEED = 20261114


class Scorer:
    def __init__(self, name: str, device: str):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        if device == "cuda":
            self.model.half()
        self.device = device
        lab = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.i_pos, self.i_neg, self.i_neu = lab["positive"], lab["negative"], lab.get("neutral")

    @torch.inference_mode()
    def _probs(self, ids: list[list[int]], batch_tokens: int = 64_000) -> np.ndarray:
        """Softmax probabilities for pre-tokenised inputs (each already <= 512 incl. special tokens).
        Inputs are sorted by length and batched by token budget, so short sentences go fast."""
        order = np.argsort([len(x) for x in ids], kind="stable")
        out = np.zeros((len(ids), self.model.config.num_labels), np.float32)
        pad = self.tok.pad_token_id
        i, n = 0, len(order)
        while i < n:   # ascending lengths: the last item of a batch is its longest
            j = i + 1
            while j < n and (j - i + 1) * len(ids[order[j]]) <= batch_tokens and j - i < 4096:
                j += 1
            idx = order[i:j]
            L = len(ids[idx[-1]])
            x = torch.full((len(idx), L), pad, dtype=torch.long)
            m = torch.zeros((len(idx), L), dtype=torch.long)
            for r, k in enumerate(idx):
                x[r, :len(ids[k])] = torch.tensor(ids[k])
                m[r, :len(ids[k])] = 1
            logits = self.model(input_ids=x.to(self.device), attention_mask=m.to(self.device)).logits.float()
            out[idx] = torch.softmax(logits, -1).cpu().numpy()
            i = j
        return out

    def _wrap(self, pieces: list[int]) -> list[int]:
        return self.tok.build_inputs_with_special_tokens(pieces)

    def sentences(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score each sentence; a sentence over 510 word pieces is cut into 510-piece chunks, each scored,
        and averaged (count returned)."""
        enc = self.tok(texts, add_special_tokens=False, truncation=False)["input_ids"]
        ids, owner = [], []
        for n, e in enumerate(enc):
            chunks = [e[k:k + MAX_PIECES] for k in range(0, max(len(e), 1), MAX_PIECES)] or [[]]
            for ch in chunks:
                ids.append(self._wrap(ch))
                owner.append(n)
        p = self._probs(ids)
        owner = np.asarray(owner)
        score = p[:, self.i_pos] - p[:, self.i_neg]
        neu = p[:, self.i_neu] if self.i_neu is not None else np.full(len(p), np.nan, np.float32)
        cnt = np.bincount(owner, minlength=len(enc))
        s = np.bincount(owner, weights=score, minlength=len(enc)) / cnt
        u = np.bincount(owner, weights=neu, minlength=len(enc)) / cnt
        return s.astype(np.float32), u.astype(np.float32), (cnt > 1)

    def scenes(self, groups: list[list[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score each scene with context: its sentences in order, packed greedily into pieces of at most
        510 word pieces (a sentence longer than that is cut). Scene score = length-weighted mean."""
        flat = [s for g in groups for s in g]
        enc = self.tok(flat, add_special_tokens=False, truncation=False)["input_ids"]
        ids, owner, weight = [], [], []
        k = 0
        for n, g in enumerate(groups):
            cur = []
            for _ in g:
                e = enc[k]
                k += 1
                for c in [e[a:a + MAX_PIECES] for a in range(0, max(len(e), 1), MAX_PIECES)]:
                    if len(cur) + len(c) > MAX_PIECES and cur:
                        ids.append(self._wrap(cur)); owner.append(n); weight.append(len(cur))
                        cur = []
                    cur = cur + c
            if cur:
                ids.append(self._wrap(cur)); owner.append(n); weight.append(len(cur))
        p = self._probs(ids)
        owner, weight = np.asarray(owner), np.asarray(weight, float)
        score = p[:, self.i_pos] - p[:, self.i_neg]
        neu = p[:, self.i_neu] if self.i_neu is not None else np.full(len(p), np.nan, np.float32)
        w = np.bincount(owner, weights=weight, minlength=len(groups))
        s = np.bincount(owner, weights=score * weight, minlength=len(groups)) / w
        u = np.bincount(owner, weights=neu * weight, minlength=len(groups)) / w
        return s.astype(np.float32), u.astype(np.float32), np.bincount(owner, minlength=len(groups))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sentences")
    ap.add_argument("out")
    ap.add_argument("--models", default="sst2,roberta3")
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--benchmark-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = pd.read_json(a.sentences, lines=True, compression="gzip", dtype={"movie_name": str, "text": str})
    rng = np.random.default_rng(SEED)
    bench = s.text.to_numpy()[rng.choice(len(s), 200, replace=False)].tolist()
    scenes = s.groupby(["movie_name", "scene"], sort=False).text.apply(list)
    meta = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
            "torch": torch.__version__, "sentences": int(len(s)), "scenes": int(len(scenes)), "models": {}}
    for key in a.models.split(","):
        name = MODELS[key]
        t0 = time.perf_counter()
        sc = Scorer(name, dev)
        load = time.perf_counter() - t0
        sc.sentences(bench[:16])                      # warm-up
        t0 = time.perf_counter()
        sc.sentences(bench)
        per = (time.perf_counter() - t0) / len(bench)
        # Batches of 200 underuse a GPU; also time 20,000 sentences for a realistic projection.
        big = s.text.sample(20_000, random_state=SEED).tolist()
        t0 = time.perf_counter()
        sc.sentences(big)
        per_big = (time.perf_counter() - t0) / len(big)
        proj = len(s) * per_big / 3600
        m = {"name": name, "load_seconds": round(load, 1), "seconds_per_sentence_bench200": per,
             "seconds_per_sentence_20k": per_big, "projected_hours_sentence_all": round(proj, 3),
             "within_limit": proj <= a.max_hours}
        meta["models"][key] = m
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(m), flush=True)
        if a.benchmark_only or not m["within_limit"]:
            continue
        t0 = time.perf_counter()
        score, neu, chunked = sc.sentences(s.text.tolist())
        pd.DataFrame({"movie_name": s.movie_name.to_numpy(), "scene": s.scene.to_numpy(), "layer": s.layer.to_numpy(),
                      "pos": s.pos.to_numpy(), f"{key}": score, f"{key}_neutral": neu, f"{key}_chunked": chunked}).to_csv(
            out / f"sentence_{key}.csv.gz", index=False, compression="gzip")
        m["sentence_seconds"] = round(time.perf_counter() - t0, 1)
        m["sentences_chunked"] = int(chunked.sum())
        t0 = time.perf_counter()
        ss, sn, npieces = sc.scenes(scenes.tolist())
        pd.DataFrame({"movie_name": scenes.index.get_level_values(0), "scene": scenes.index.get_level_values(1),
                      f"{key}_scene": ss, f"{key}_scene_neutral": sn, f"{key}_pieces": npieces}).to_csv(
            out / f"scene_{key}.csv.gz", index=False, compression="gzip")
        m["scene_seconds"] = round(time.perf_counter() - t0, 1)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(m), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
