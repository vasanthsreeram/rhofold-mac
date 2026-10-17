#!/usr/bin/env bash
# Predict an RNA 3D structure with RhoFold+ on this Mac.
#   ./run.sh <input.fasta> <out_dir> [--a3m MSA.a3m] [--device auto|mps|cpu] [--official]
# Single-sequence mode unless --a3m is given (no MSA search, no Amber relaxation).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ $# -lt 2 ]; then
  sed -n '2,4p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi
if [ ! -x "$DIR/.venv/bin/python" ] || [ ! -f "$DIR/RhoFold/pretrained/RhoFold_pretrained.pt" ]; then
  echo "Not set up yet. Run ./setup.sh first." >&2; exit 1
fi
start=$(date +%s)
"$DIR/.venv/bin/python" "$DIR/predict.py" "$@"
echo "Total wall time (Python start-up + model load + prediction): $(( $(date +%s) - start )) s"
