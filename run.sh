#!/usr/bin/env bash
# Predict an RNA 3D structure with RhoFold+ on this Mac (Apple GPU via PyTorch MPS, CPU fallback).
#   ./run.sh                              fold the bundled 52-nt sample  -> out/sample/
#   ./run.sh my_rna.fasta                 fold your sequence             -> out/my_rna/
#   ./run.sh my_rna.fasta results/        choose the output folder
# Options: --a3m MSA.a3m  --device auto|mps|cpu  --official (turn off the speed patch)
# Single-sequence mode unless --a3m is given (no MSA search, no Amber relaxation).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
case "${1:-}" in -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac
if [ ! -x "$DIR/.venv/bin/python" ] || [ ! -f "$DIR/RhoFold/pretrained/RhoFold_pretrained.pt" ]; then
  echo "Not set up yet. Run ./setup.sh first (one time, ~1-5 min)." >&2; exit 1
fi
start=$(date +%s)
"$DIR/.venv/bin/python" "$DIR/predict.py" "$@"
echo "Total wall time (Python start-up + model load + prediction): $(( $(date +%s) - start )) s"
