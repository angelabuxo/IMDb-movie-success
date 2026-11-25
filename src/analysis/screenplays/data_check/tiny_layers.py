"""Why are 31 films' dialogue or description layers empty or tiny?

Looks only at tags and counts (no script text is printed or saved).
Output: tiny_layers.csv. Run: python3 -I analysis/screenplays/data_check/tiny_layers.py
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = [r for r in csv.DictReader(open(HERE / "layer_stats.csv")) if r["tiny_layer"] == "1"]
out = []
for r in rows:
    for layer in ["dialogue", "description"]:
        if int(r[f"{layer}_tokens"]) >= 500:
            continue
        els, empty, toks = int(r[f"{layer}_elements"]), int(r[f"{layer}_empty"]), int(r[f"{layer}_tokens"])
        other = "description" if layer == "dialogue" else "dialogue"
        if els == 0:
            why = "no elements with this tag at all"
        elif els == empty:
            why = "elements exist but are all empty"
        else:
            why = "few elements"
        out.append({"imdb_id": r["imdb_id"], "movie_name": r["movie_name"], "tiny_layer": layer,
                    "elements": els, "empty_elements": empty, "tokens": toks,
                    "other_layer_tokens": r[f"{other}_tokens"], "char_elements": r["char_elements"],
                    "empty_scene_markers": r["empty_scenes"], "reason": why})
with open(HERE / "tiny_layers.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
for o in out:
    print(f"{o['movie_name'][:34]:<35}{o['tiny_layer']:<12}els {o['elements']:>6} empty {o['empty_elements']:>5} tokens {o['tokens']:>5}"
          f"  char els {o['char_elements']:>5}  empty scenes {o['empty_scene_markers']:>3}  {o['reason']}")
