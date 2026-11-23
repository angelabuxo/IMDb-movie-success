#!/bin/bash
# When the HPC job is done, copy back its outputs (numbers), the private tables it wrote and the vectors.
#   bash archive/analysis/embeddings/hpc/fetch.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
ssh -o BatchMode=yes hpc 'cd /dtu/blackhole/0e/223278/.sg02805/embeddings && tar cf - archive/analysis/embeddings/outputs archive/analysis/embeddings/private \
  --exclude=archive/analysis/embeddings/private/reviews_pool.parquet lsf_*.out lsf_*.err nvidia-smi_*.txt' | tar xf - -C . 
mkdir -p data/raw/features/text_vectors
ssh -o BatchMode=yes hpc 'cd /dtu/blackhole/0e/223278/.sg02805/embeddings/data/features/text_vectors && tar cf - .' | tar xf - -C data/raw/features/text_vectors
mkdir -p archive/analysis/embeddings/private/hpc && mv -f lsf_*.out lsf_*.err nvidia-smi_*.txt archive/analysis/embeddings/private/hpc/ 2>/dev/null || true
echo fetched
