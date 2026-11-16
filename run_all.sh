#!/bin/sh
# One command from a fresh clone to a checked, running explorer. From the repo root:
#
#   ./run_all.sh                 set up, check every page and the quick tests, then start the explorer
#   ./run_all.sh --check         the same checks, without starting the explorer (exit code 1 if anything broke)
#   ./run_all.sh --full          also open every notebook step and view, and run the slow tests
#   ./run_all.sh --no-check      set up and start the explorer straight away
#
# Steps:
#   1. ./setup.sh: make .venv, install requirements.txt, unpack data/bundles into data/raw (skipped when done)
#   2. python -m explorer.check_pages: open every explorer page headless and list crashes, errors and every
#      input file a page couldn't find; report in logs/check_pages.log
#   3. python -m pytest: the test suite on synthetic data (about 35 s here; --full adds the slow tests)
#   4. ./explorer/run.sh: Streamlit on http://localhost:8501, logging to logs/explorer.log and logs/streamlit.log
#
# A failing check doesn't stop the explorer from starting; read the summary above the Streamlit banner.
# Extra arguments after the flags go to Streamlit, e.g. ./run_all.sh --server.port 8502
set -e
cd "$(dirname "$0")"

CHECK=1 LAUNCH=1 FULL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --check) LAUNCH=0 ;;
    --full) FULL=1 ;;
    --no-check) CHECK=0 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) break ;;
  esac
  shift
done

echo "== 1/4 setup"
./setup.sh
mkdir -p logs

STATUS=0
if [ "$CHECK" = 1 ]; then
  echo "== 2/4 explorer pages (report: logs/check_pages.log)"
  if [ "$FULL" = 1 ]; then PAGES_ARGS=""; else PAGES_ARGS="--quick"; fi
  # Streamlit's bare-mode chatter goes to stderr; keep it out of the terminal but in a log
  .venv/bin/python -m explorer.check_pages $PAGES_ARGS 2>>logs/check_pages.stderr.log || STATUS=1

  echo "== 3/4 tests"
  if [ "$FULL" = 1 ]; then
    .venv/bin/python -m pytest -q || STATUS=1
  else
    .venv/bin/python -m pytest -q -m "not slow" || STATUS=1
  fi
  if [ "$STATUS" = 0 ]; then echo "== checks passed"; else echo "== SOME CHECKS FAILED: see above and logs/"; fi
fi

if [ "$LAUNCH" = 0 ]; then
  exit "$STATUS"
fi
echo "== 4/4 explorer on http://localhost:8501 (Ctrl+C to stop)"
exec ./explorer/run.sh "$@"
