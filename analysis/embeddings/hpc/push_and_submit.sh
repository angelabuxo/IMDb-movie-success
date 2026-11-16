#!/bin/bash
# Upload the embedding analysis and its private inputs to DTU HPC and submit run.lsf. Run from the repo root
# after the local steps (films, texts, sample-reviews):  bash analysis/embeddings/hpc/push_and_submit.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=analysis/embeddings/private
COPYFILE_DISABLE=1 tar czf - --no-xattrs analysis/__init__.py analysis/clean_rule.py analysis/tags/tag_profit_tags.csv \
  analysis/embeddings/*.py analysis/embeddings/config.yaml analysis/embeddings/hpc/run.lsf \
  analysis/embeddings/outputs/s01_cleaning.json analysis/embeddings/outputs/s02_embedding.json \
  $P/films.parquet $P/reviews_pool.parquet $P/reviews_sampling.json $P/genome_meta.jsonl \
  $P/screenplays.parquet $P/summaries.parquet | \
ssh hpc 'bash -lc "
set -e; umask 077
mkdir -p /dtu/blackhole/0e/223278/.sg02805/embeddings && chmod 700 /dtu/blackhole/0e/223278/.sg02805/embeddings && cd /dtu/blackhole/0e/223278/.sg02805/embeddings
rm -rf analysis/embeddings/private/screenplay_ocr.parquet
tar xzf -
chmod -R go-rwx .
# Two copies, as for the arc job: a GPU node and a 16-core CPU node. watch_and_fetch.sh keeps whichever
# starts first and cancels the other; both write to the same folders, so only one may run.
bsub < analysis/embeddings/hpc/run.lsf
sed -e \"s/^#BSUB -q gpuv100/#BSUB -q hpc/\" -e \"/^#BSUB -gpu/d\" -e \"s/^#BSUB -n 8/#BSUB -n 16/\" \
  -e \"s/^#BSUB -J emb02805/#BSUB -J emb02805cpu/\" analysis/embeddings/hpc/run.lsf | bsub
bjobs -w | tail -3
"'
