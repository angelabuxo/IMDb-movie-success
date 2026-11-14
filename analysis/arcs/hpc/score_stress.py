"""Score the constructed Step 3 stress sentences (no script text) with both contextual models, on CPU.

    python score_stress.py stress.json stress_contextual.csv

stress.json is the list of sentences in analysis.arcs.s03_instruments.STRESS. Same Scorer and score definition
(P(pos) - P(neg)) as the full job, but fp32 on CPU instead of fp16 on GPU. Copy the CSV back to
analysis/arcs/outputs/s03_stress_contextual.csv.
"""
import json
import sys

import numpy as np
import pandas as pd

from score_contextual import MODELS, Scorer

x = json.load(open(sys.argv[1]))
d = {"sentence": x}
for k, name in MODELS.items():
    s, u, _ = Scorer(name, "cpu").sentences(x)
    d[k] = np.round(s, 4)
    if k == "roberta3":
        d["roberta3_neutral"] = np.round(u, 4)
pd.DataFrame(d).to_csv(sys.argv[2], index=False)
print(pd.DataFrame(d).to_string())
