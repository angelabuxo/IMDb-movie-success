#!/bin/bash
# Watch the two copies of the embedding job (GPU and CPU); keep whichever starts first, cancel the other, and
# fetch the results when the kept one finishes. Uses BatchMode SSH; gives up after MAX_FAILS failed polls.
#   bash archive/analysis/embeddings/hpc/watch_and_fetch.sh [poll_seconds]
# Exit 0: fetched. 2: SSH lost. 3: the kept job failed (its log is fetched).
set -uo pipefail
cd "$(dirname "$0")/../../.."
POLL=${1:-120}; MAX_FAILS=15
LOG=archive/analysis/embeddings/private/watch.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
hpc() { ssh -o BatchMode=yes -o ConnectTimeout=20 hpc "bash -lc '$1'" 2>/dev/null; }
fails=0; keep=""
while true; do
  st=$(hpc "bjobs -a -noheader -o \"jobid stat job_name\" -J emb02805 2>/dev/null; bjobs -a -noheader -o \"jobid stat job_name\" -J emb02805cpu 2>/dev/null")
  if [ -z "$st" ]; then
    fails=$((fails + 1)); say "ssh failed or no jobs ($fails/$MAX_FAILS)"
    [ "$fails" -ge "$MAX_FAILS" ] && { say "giving up"; exit 2; }
    sleep "$POLL"; continue
  fi
  fails=0
  gpu=$(awk '$3 == "emb02805" {print $1, $2}' <<< "$st" | tail -1); cpu=$(awk '$3 == "emb02805cpu" {print $1, $2}' <<< "$st" | tail -1)
  if [ -z "$keep" ]; then
    for j in "$gpu" "$cpu"; do
      id=${j% *}; s=${j#* }
      if [ -n "$j" ] && [ "$s" != "PEND" ]; then keep=$id; break; fi
    done
    if [ -n "$keep" ]; then
      for j in "$gpu" "$cpu"; do id=${j% *}; [ -n "$j" ] && [ "$id" != "$keep" ] && hpc "bkill $id" >/dev/null; done
      say "job $keep started first; cancelled the other (gpu: $gpu, cpu: $cpu)"
    else
      say "both pending (gpu: $gpu, cpu: $cpu)"
    fi
  fi
  if [ -n "$keep" ]; then
    sk=$(awk -v j="$keep" '$1 == j {print $2}' <<< "$st")
    say "kept job $keep: $sk"
    case "$sk" in
      DONE) bash archive/analysis/embeddings/hpc/fetch.sh && say "fetched" && exit 0 ;;
      EXIT) ssh -o BatchMode=yes hpc "cd /dtu/blackhole/0e/223278/.sg02805/embeddings && cat lsf_$keep.out lsf_$keep.err" > archive/analysis/embeddings/private/lsf_$keep.failed.txt 2>/dev/null
            say "job $keep FAILED; log in archive/analysis/embeddings/private/lsf_$keep.failed.txt"; exit 3 ;;
    esac
  fi
  sleep "$POLL"
done
