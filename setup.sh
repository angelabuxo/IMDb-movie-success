#!/bin/sh
# Set up everything from a clone, in one command, from the repo root:
#
#   ./setup.sh          make .venv, install requirements.txt, unpack data/bundles into data/raw
#   ./setup.sh --data   only unpack the bundles
#
# Safe to run again: packages are reinstalled only when requirements.txt changes, and the bundles are
# unpacked again only when the set of bundles changes (after a pull that adds one). Bundles are unpacked
# oldest first, so a newer bundle replaces the files it updates. Set IMDB_DATA_DIR to unpack somewhere
# other than data/raw. Needs python3 and zstd (brew install zstd on a Mac).
set -e
cd "$(dirname "$0")"

if [ -t 0 ] && [ -t 1 ]; then
  trap 'printf "\nPress Enter to close this terminal..."; IFS= read -r _ || true' 0
fi

if [ "$1" != "--data" ]; then
  if [ ! -x .venv/bin/python ] && [ ! -x .venv/Scripts/python.exe ]; then
    echo "Creating .venv"
    if command -v python3 >/dev/null 2>&1; then
      python3 -m venv .venv
    else
      python -m venv .venv
    fi
  fi
  if [ -x .venv/bin/python ]; then
    VENV_PYTHON=.venv/bin/python
  elif [ -x .venv/Scripts/python.exe ]; then
    VENV_PYTHON=.venv/Scripts/python.exe
  else
    echo "Could not find the Python executable in .venv." >&2
    exit 1
  fi
  if [ ! .venv/.requirements-installed -nt requirements.txt ]; then
    echo "Installing requirements (the first time takes a few minutes)"
    "$VENV_PYTHON" -m pip install -q --upgrade pip setuptools wheel
    "$VENV_PYTHON" -m pip install -q -r requirements.txt
    touch .venv/.requirements-installed
  fi
fi

if [ -x .venv/bin/python ]; then
  VENV_PYTHON=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
  VENV_PYTHON=.venv/Scripts/python.exe
else
  VENV_PYTHON=
fi

if [ -n "$VENV_PYTHON" ] && ! "$VENV_PYTHON" -c "import zstandard" >/dev/null 2>&1; then
  echo "Installing zstd support"
  "$VENV_PYTHON" -m pip install -q zstandard
fi

DATA="${IMDB_DATA_DIR:-data/raw}"
STAMP="$DATA/.bundles-unpacked"
BUNDLES=$(ls data/bundles/*.tar.zst | sed -E 's/.*-([0-9]{4}-[0-9]{2}-[0-9]{2})\.tar\.zst$/\1 &/' | sort | cut -d" " -f2)
WANT=$(for f in $BUNDLES; do echo "$f $(wc -c < "$f" | tr -d ' ')"; done)
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$WANT" ]; then
  echo "Bundles already unpacked in $DATA"
else
  echo "Unpacking data/bundles into $DATA (oldest first, so newer bundles replace older files)"
  mkdir -p "$DATA"
  for f in $BUNDLES; do
    echo "  $f"
    if command -v zstd >/dev/null 2>&1; then
      tar --zstd -xf "$f" -C "$DATA"
    elif [ -n "$VENV_PYTHON" ] && "$VENV_PYTHON" -c "import zstandard" >/dev/null 2>&1; then
      "$VENV_PYTHON" -c '
import sys
import tarfile
import zstandard

source, destination = sys.argv[1:3]
with open(source, "rb") as compressed:
    with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            archive.extractall(destination, filter="data")
' "$f" "$DATA"
    else
      echo "zstd support is missing and could not be installed." >&2
      exit 1
    fi
  done
  echo "$WANT" > "$STAMP"
fi

echo "Done. Start the explorer with ./src/explorer/run.sh, or run the tests with .venv/bin/python -m pytest"
