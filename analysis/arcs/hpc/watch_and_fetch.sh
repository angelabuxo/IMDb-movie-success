#!/bin/bash
# Watch the two copies of the contextual job; keep whichever starts first and cancel the other (rule in
# LOG.md); when the kept job ends, copy its numeric outputs (no script text) to analysis/arcs/private/contextual/.
# Uses the existing SSH control connection only (BatchMode): if it is gone, retry; give up after MAX_FAILS polls.
#   bash analysis/arcs/hpc/watch_and_fetch.sh [poll_seconds]
# Exit 0: results fetched. 2: SSH lost. 3: kept job failed (its LSF log is fetched).
set -uo pipefail
cd "$(dirname "$0")/../../.."
A40=30053002; A10=30053443
POLL=${1:-300}; MAX_FAILS=12
DEST=analysis/arcs/private/contextual
LOG=analysis/arcs/private/watch.log
mkdir -p "$DEST"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
hpc() { ssh -o BatchMode=yes -o ConnectTimeout=20 hpc "bash -lc '$1'" 2>/dev/null; }
stat_of() { awk -v j="$1" '$1 == j {print $2}' <<< "$2"; }

fails=0; keep=""
while true; do
  st=$(hpc "bjobs -a -noheader -o \"jobid stat\" $A40 $A10")
  if [ -z "$st" ]; then
    fails=$((fails + 1)); say "ssh failed ($fails/$MAX_FAILS)"
    [ "$fails" -ge "$MAX_FAILS" ] && { say "giving up: SSH connection lost"; exit 2; }
    sleep "$POLL"; continue
  fi
  fails=0
  s40=$(stat_of $A40 "$st"); s10=$(stat_of $A10 "$st")
  if [ -z "$keep" ]; then
    if [ "$s40" != "PEND" ] && [ -n "$s40" ]; then keep=$A40; other=$A10; out=out
    elif [ "$s10" != "PEND" ] && [ -n "$s10" ]; then keep=$A10; other=$A40; out=out_a10; fi
    if [ -n "$keep" ]; then
      hpc "bkill $other" >/dev/null
      say "job $keep started first ($s40 / $s10); cancelled $other"
    else
      say "both pending (a40=$s40 a10=$s10)"
    fi
  fi
  if [ -n "$keep" ]; then
    sk=$(stat_of $keep "$st")
    say "kept job $keep: $sk"
    case "$sk" in
      DONE)
        ssh -o BatchMode=yes hpc "cd ~/02805-embed/arcs/$out && tar cf - ." 2>/dev/null | tar xf - -C "$DEST"
        ssh -o BatchMode=yes hpc "cd ~/02805-embed/arcs && cat lsf*_$keep.out" 2>/dev/null > "$DEST/lsf_$keep.out"
        say "fetched: $(ls "$DEST" | tr '\n' ' ')"
        echo "$keep" > "$DEST/job_id.txt"
        exit 0 ;;
      EXIT)
        ssh -o BatchMode=yes hpc "cd ~/02805-embed/arcs && cat lsf*_$keep.out lsf*_$keep.err" 2>/dev/null > "$DEST/lsf_$keep.failed.txt"
        say "kept job $keep FAILED; log in $DEST/lsf_$keep.failed.txt"
        exit 3 ;;
    esac
  fi
  sleep "$POLL"
done
