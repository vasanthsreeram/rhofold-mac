#!/usr/bin/env bash
# Self-check: is this Mac ready to run RhoFold+? Read-only, takes a few seconds.
# Prints one line per check and what to do if it fails. Exit code 1 if anything is wrong.
cd "$(dirname "$0")"
WEIGHTS=RhoFold/pretrained/RhoFold_pretrained.pt
WEIGHTS_SHA256=3adb621978dfcd7ea0dc0edeb520d249f423d61530df85ff58a1fad2f33e5608
RHOFOLD_COMMIT=6bdfbda720184409eb682ce08c05d258162ddc48
bad=0
if [ -t 1 ]; then G=$'\033[32m' Y=$'\033[33m' R=$'\033[31m' N=$'\033[0m'; else G= Y= R= N=; fi
ok()   { printf '  %sok%s    %s\n' "$G" "$N" "$*"; }
warn() { printf '  %swarn%s  %s\n' "$Y" "$N" "$*"; }
no()   { printf '  %sFAIL%s  %s\n' "$R" "$N" "$*"; bad=1; }

echo "rhofold-mac doctor"
[ "$(uname -s)" = Darwin ] && ok "macOS $(sw_vers -productVersion)" || no "not macOS. This runner is for Apple Silicon Macs; on Linux use upstream RhoFold."
if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = 1 ]; then
  no "this terminal runs under Rosetta (x86 emulation). Open a native Terminal (Get Info > untick 'Open using Rosetta')."
elif [ "$(uname -m)" = arm64 ]; then
  ok "Apple Silicon ($(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB RAM)"
else
  no "Intel Mac: not supported (no Apple GPU for MPS, and current PyTorch has no Intel Mac builds)."
fi
major=$(sw_vers -productVersion | cut -d. -f1); minor=$(sw_vers -productVersion | cut -d. -f2)
{ [ "$major" -gt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -ge 3 ]; }; } || no "macOS 12.3+ needed for the Apple GPU (MPS). Update macOS, or run with --device cpu."

if [ -x .venv/bin/python ]; then
  pv=$(.venv/bin/python -c 'import platform,sys; print(f"{sys.version_info[0]}.{sys.version_info[1]} {platform.machine()}")')
  case "$pv" in
    *x86_64) no "Python in .venv is x86_64 (Rosetta). Delete .venv and re-run ./setup.sh from a native terminal." ;;
    3.1[0-9]*) ok "Python $pv in .venv" ;;
    *) no "Python $pv in .venv is too old (need 3.10+). Delete .venv and re-run ./setup.sh." ;;
  esac
  out=$(.venv/bin/python -c 'import torch; print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())' 2>&1)
  if [ $? -ne 0 ]; then
    no "PyTorch not importable in .venv. Re-run ./setup.sh."
  else
    set -- $out
    [ "$3" = True ] && ok "PyTorch $1, Apple GPU (MPS) available: runs on the GPU" \
                    || warn "PyTorch $1, MPS not available (built: $2): runs on the CPU, slower. See the checks above."
  fi
else
  no "no .venv yet. Run ./setup.sh"
fi

if [ -d RhoFold/.git ]; then
  [ "$(git -C RhoFold rev-parse HEAD)" = "$RHOFOLD_COMMIT" ] && ok "RhoFold code at ${RHOFOLD_COMMIT:0:7}" || no "RhoFold/ is not at ${RHOFOLD_COMMIT:0:7}. Delete RhoFold/ and re-run ./setup.sh"
  git -C RhoFold apply --reverse --check ../rhofold-mac-fast.patch 2>/dev/null && ok "Mac speed patch applied" || no "Mac patch not applied. Re-run ./setup.sh"
else
  no "RhoFold code missing. Run ./setup.sh"
fi
if [ -f "$WEIGHTS" ]; then
  [ "$(shasum -a 256 "$WEIGHTS" | awk '{print $1}')" = "$WEIGHTS_SHA256" ] && ok "weights present, checksum matches" \
    || no "weights file is damaged (checksum mismatch). Delete $WEIGHTS and re-run ./setup.sh"
else
  no "weights missing. Run ./setup.sh (downloads ~508 MB)"
fi
free_gb=$(df -g . | awk 'NR==2 {print $4}')
[ "${free_gb:-0}" -ge 1 ] && ok "${free_gb} GB free disk" || warn "under 1 GB free disk; outputs are small, but setup needs ~3 GB"

if [ $bad = 0 ]; then echo "All good. Try:  ./run.sh"; else echo "Fix the FAIL lines above, then run ./doctor.sh again."; exit 1; fi
