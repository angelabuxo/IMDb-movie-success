#!/bin/sh
# Start the local explorer from a fresh clone: ./explorer/run.sh
# Runs ./setup.sh first (makes .venv, installs requirements.txt when it changes, unpacks data/bundles into
# data/raw when the bundles change), then runs streamlit. Extra arguments go to streamlit,
# e.g. ./explorer/run.sh --server.port 8502
set -e
cd "$(dirname "$0")/.."

./setup.sh

# localhost only: data/raw holds licensed data and copyrighted script text
exec .venv/bin/streamlit run explorer/app.py --server.address localhost "$@"
