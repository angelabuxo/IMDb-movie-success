#!/bin/bash
# Watch one HPC job; when it ends, copy its numeric outputs (no script text) and its LSF finish time back.
#   bash analysis/arcs/hpc/watch_job.sh JOBID REMOTE_OUT_DIR LOCAL_DEST [poll_seconds]
# Uses the existing SSH control connection only (BatchMode). Exit 0: fetched. 2: SSH lost. 3: job failed.
set -uo pipefail
cd "$(dirname "$0")/../../.."
JOB=$1; OUT=$2; DEST=$3; POLL=${4:-300}; MAX_FAILS=12
LOG=analysis/arcs/private/watch_$JOB.log
mkdir -p "$DEST"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
fails=0
while true; do
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=20 hpc "bash -lc 'bjobs -a -noheader -o \"stat\" $JOB'" 2>/dev/null)
  if [ -z "$st" ]; then
    fails=$((fails + 1)); say "ssh failed ($fails/$MAX_FAILS)"
    [ "$fails" -ge "$MAX_FAILS" ] && { say "giving up: SSH connection lost"; exit 2; }
    sleep "$POLL"; continue
  fi
  fails=0; say "job $JOB: $st"
  case "$st" in
    DONE)
      ssh -o BatchMode=yes hpc "cd ~/02805-embed/arcs/$OUT && tar cf - ." | tar xf - -C "$DEST"
      fin=$(ssh -o BatchMode=yes hpc "bash -lc 'bjobs -a -noheader -o \"finish_time\" $JOB'" 2>/dev/null)
      python3 -c "import sys, time, datetime as d; s=sys.argv[1].strip().rstrip(' L'); y=time.localtime().tm_year; print(d.datetime.strptime(f'{y} {s}', '%Y %b %d %H:%M').timestamp())" "$fin" > "$DEST/finished_at.txt"
      say "fetched; LSF finish time: $fin"
      exit 0 ;;
    EXIT)
      ssh -o BatchMode=yes hpc "cd ~/02805-embed/arcs && cat lsf*_$JOB.out lsf*_$JOB.err" > "$DEST/lsf_$JOB.failed.txt" 2>/dev/null
      say "job $JOB FAILED; log in $DEST"; exit 3 ;;
  esac
  sleep "$POLL"
done
