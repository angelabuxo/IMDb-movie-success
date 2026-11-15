#!/bin/sh
# Start the local explorer from a fresh clone: ./explorer/run.sh
# Makes .venv and installs requirements.txt (again only when it changes), unpacks data/bundles into
# data/raw when data/raw has no movie graph yet, then runs streamlit. Extra arguments go to streamlit,
# e.g. ./explorer/run.sh --server.port 8502
set -e
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv"
  python3 -m venv .venv
fi
if [ ! .venv/.requirements-installed -nt requirements.txt ]; then
  echo "Installing requirements (first run takes a few minutes)"
  .venv/bin/pip install -q -r requirements.txt
  touch .venv/.requirements-installed
fi

DATA="${IMDB_DATA_DIR:-data/raw}"
if [ ! -f "$DATA/features/graph/nodes.csv" ]; then
  echo "Unpacking data/bundles into $DATA (oldest first, so newer bundles replace older files)"
  mkdir -p "$DATA"
  for f in $(ls data/bundles/*.tar.zst | sed -E 's/.*-([0-9]{4}-[0-9]{2}-[0-9]{2})\.tar\.zst$/\1 &/' | sort | cut -d" " -f2); do
    echo "  $f"
    tar --zstd -xf "$f" -C "$DATA"
  done
fi

# localhost only: data/raw holds licensed data and copyrighted script text
exec .venv/bin/streamlit run explorer/app.py --server.address localhost "$@"
