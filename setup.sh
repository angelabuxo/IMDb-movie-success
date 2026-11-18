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
    case "$(uname -s)" in
      Linux*)
        echo "The existing .venv was created for Windows. Remove it and rerun this script in WSL to create a Linux venv:" >&2
        echo "  rm -rf .venv && ./explorer/run.sh" >&2
        exit 1
        ;;
      *)
        VENV_PYTHON=.venv/Scripts/python.exe
        ;;
    esac
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

if ! command -v zstd >/dev/null 2>&1; then
  echo "zstd is missing, so the bundles can't be unpacked. Install it (brew install zstd, or apt install zstd) and run ./setup.sh again." >&2
  exit 1
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
    tar --zstd -xf "$f" -C "$DATA"
  done
  echo "$WANT" > "$STAMP"
fi

echo "Done. Start the explorer with ./explorer/run.sh, or run the tests with .venv/bin/python -m pytest"
