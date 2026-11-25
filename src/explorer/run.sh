#!/bin/sh
# Start the local explorer from a fresh clone: ./src/explorer/run.sh  (./run_all.sh also checks every page first)
# Runs ./setup.sh first (makes .venv, installs requirements.txt when it changes, unpacks data/bundles into
# data/raw when the bundles change), then runs streamlit. Extra arguments go to streamlit,
# e.g. ./src/explorer/run.sh --server.port 8502
#
# Logs: the explorer's crashes, errors, warnings and missing input files go to logs/explorer.log
# (explorer/log.py); Streamlit's own terminal output is copied to logs/streamlit.log.
set -e
cd "$(dirname "$0")/../.."

if [ -t 0 ] && [ -t 1 ]; then
  trap 'printf "\nPress Enter to close this terminal..."; IFS= read -r _ || true' 0
fi

./setup.sh
mkdir -p logs
echo "Logging to logs/explorer.log and logs/streamlit.log"

# localhost only: data/raw holds licensed data and copyrighted script text
if [ -x .venv/bin/streamlit ]; then
  STREAMLIT=.venv/bin/streamlit
elif [ -x .venv/Scripts/streamlit.exe ]; then
  STREAMLIT=.venv/Scripts/streamlit.exe
else
  echo "Could not find Streamlit in .venv. Check the setup output above." >&2
  exit 1
fi
"$STREAMLIT" run src/explorer/app.py --server.address localhost "$@" 2>&1 | tee -a logs/streamlit.log
