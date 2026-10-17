#!/usr/bin/env bash
# One-time setup for rhofold-mac. Safe to re-run: every step is skipped if already done.
#   1. Python env in .venv (uv if installed, else python3.10+ venv) with pinned deps
#   2. Upstream RhoFold cloned into ./RhoFold at a fixed commit, Mac patch applied
#   3. Official RhoFold+ weights (~508 MB) downloaded from upstream's Hugging Face repo, sha256-checked
set -euo pipefail
cd "$(dirname "$0")"

RHOFOLD_REPO=https://github.com/ml4bio/RhoFold.git
RHOFOLD_COMMIT=6bdfbda720184409eb682ce08c05d258162ddc48   # upstream main, 2025-05-29
HF_REV=4458c1c5484a3a10a7f3059b9f0e8ca0447b23ac            # huggingface.co/cuhkaih/rhofold
WEIGHTS_URL="https://huggingface.co/cuhkaih/rhofold/resolve/$HF_REV/rhofold_pretrained_params.pt"
WEIGHTS_SHA256=3adb621978dfcd7ea0dc0edeb520d249f423d61530df85ff58a1fad2f33e5608
WEIGHTS=RhoFold/pretrained/RhoFold_pretrained.pt
WITH_MLX=0
[ "${1:-}" = "--with-mlx" ] && WITH_MLX=1

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31mError:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 0. checks -------------------------------------------------------------
[ "$(uname -s)" = Darwin ] || die "this runner is for macOS on Apple Silicon (for Linux/CUDA use upstream RhoFold)."
[ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = 1 ] && die "this terminal runs under Rosetta (x86 emulation), so PyTorch would not see the Apple GPU. Open a native Terminal (Finder > Terminal > Get Info > untick 'Open using Rosetta') and re-run ./setup.sh"
[ "$(uname -m)" = arm64 ] || die "needs an Apple Silicon Mac (M1 or newer). Intel Macs are not supported."
[ "$(sw_vers -productVersion | cut -d. -f1)" -ge 13 ] || echo "Warning: macOS $(sw_vers -productVersion). The Apple GPU (MPS) needs 12.3+; tested on 15.5. Older systems fall back to the CPU." >&2
command -v git >/dev/null || die "git not found. Run: xcode-select --install"
command -v curl >/dev/null || die "curl not found."
free_gb=$(df -g . | awk 'NR==2 {print $4}')
[ "${free_gb:-99}" -ge 3 ] || die "need about 3 GB free disk (env ~1.5 GB + weights 0.5 GB); only ${free_gb} GB free."

# --- 1. Python environment ---------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then
    say "Creating .venv with uv (Python 3.12)"
    uv venv --python 3.12 .venv
  else
    PY=""
    for c in python3.12 python3.13 python3.11 python3.10 python3; do
      if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then PY=$c; break; fi
    done
    [ -n "$PY" ] || die "need Python 3.10+ (macOS's built-in python3 is 3.9). Easiest fix: install uv (curl -LsSf https://astral.sh/uv/install.sh | sh) and re-run ./setup.sh"
    say "Creating .venv with $PY ($($PY --version))"
    "$PY" -m venv .venv
  fi
fi
say "Installing pinned Python packages"
if command -v uv >/dev/null; then
  uv pip install --python .venv/bin/python -q -r requirements.txt
  [ $WITH_MLX = 1 ] && uv pip install --python .venv/bin/python -q -r requirements-mlx.txt
else
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
  [ $WITH_MLX = 1 ] && .venv/bin/python -m pip install -q -r requirements-mlx.txt
fi

# --- 2. Upstream RhoFold at a fixed commit + Mac patch ------------------------
if [ ! -d RhoFold/.git ]; then
  say "Cloning upstream RhoFold @ ${RHOFOLD_COMMIT:0:7}"
  git clone -q "$RHOFOLD_REPO" RhoFold
fi
if [ "$(git -C RhoFold rev-parse HEAD)" != "$RHOFOLD_COMMIT" ]; then
  git -C RhoFold checkout -q "$RHOFOLD_COMMIT" || die "could not check out $RHOFOLD_COMMIT (local edits in RhoFold/? delete the folder and re-run)."
fi
if git -C RhoFold apply --reverse --check ../rhofold-mac-fast.patch 2>/dev/null; then
  say "Mac patch already applied"
else
  say "Applying rhofold-mac-fast.patch"
  git -C RhoFold apply ../rhofold-mac-fast.patch || die "patch did not apply. Delete RhoFold/ and re-run ./setup.sh"
fi

# --- 3. Official weights ------------------------------------------------------
check_sha() { [ "$(shasum -a 256 "$1" | awk '{print $1}')" = "$WEIGHTS_SHA256" ]; }
if [ -f "$WEIGHTS" ] && check_sha "$WEIGHTS"; then
  say "Weights present and verified"
else
  say "Downloading RhoFold+ weights (~508 MB) from huggingface.co/cuhkaih/rhofold"
  mkdir -p RhoFold/pretrained
  curl -fL -# --retry 3 -C - -o "$WEIGHTS.part" "$WEIGHTS_URL" || die "weights download failed (network, proxy, or Hugging Face down).
  Re-run ./setup.sh to resume where it stopped. Or download by hand in a browser:
    $WEIGHTS_URL
  and save it as $(pwd)/$WEIGHTS"
  check_sha "$WEIGHTS.part" || { rm -f "$WEIGHTS.part"; die "the downloaded weights are damaged (SHA-256 mismatch), so they were deleted. Re-run ./setup.sh to download again."; }
  mv "$WEIGHTS.part" "$WEIGHTS"
fi

# --- 4. Self-check -------------------------------------------------------------
./doctor.sh || die "setup finished but the self-check above found a problem."
say "Setup done. Fold the bundled sample with:  ./run.sh"
