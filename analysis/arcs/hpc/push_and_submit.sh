#!/bin/bash
# One connection to DTU HPC: upload the job and the private sentence table, download the two models into the
# scratch Hugging Face cache, smoke-test both on 200 sentences on the login node's CPU, and submit the job.
# Run from the repo root:  bash analysis/arcs/hpc/push_and_submit.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
COPYFILE_DISABLE=1 tar czf - --no-xattrs -C analysis/arcs hpc/score_contextual.py hpc/score.lsf private/sentences.jsonl.gz | \
ssh hpc 'bash -lc "
set -e; umask 077
mkdir -p ~/02805-embed/arcs && cd ~/02805-embed/arcs
tar xzf - && mv -f hpc/score_contextual.py hpc/score.lsf . && rmdir hpc
chmod 700 . private; chmod 600 private/*
module load python3/3.12.11 >/dev/null 2>&1
export HF_HOME=/dtu/blackhole/0e/223278/.sg02805/hf
P=~/venvs/sg02805/bin/python
\$P -I - <<PY
import gzip, json, itertools
from transformers import pipeline
rows = [json.loads(l) for l in itertools.islice(gzip.open(\"private/sentences.jsonl.gz\", \"rt\"), 200)]
for name in (\"distilbert-base-uncased-finetuned-sst-2-english\", \"cardiffnlp/twitter-roberta-base-sentiment-latest\"):
    clf = pipeline(\"sentiment-analysis\", model=name, device=\"cpu\", truncation=True)
    out = clf([r[\"text\"] for r in rows], batch_size=32)
    print(\"smoke\", name, len(out), sorted({o[\"label\"] for o in out}))
PY
bsub < score.lsf
bjobs -w | tail -3
"'
