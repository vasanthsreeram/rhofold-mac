#!/bin/bash
# Full benchmark matrix. Each config runs in its own process (clean peak-memory reading).
cd "$(dirname "$0")/.."; mkdir -p benchmarks/results out/bench-pdb
PY=.venv/bin/python; E=RhoFold/example/input; R=benchmarks/results
run() { # name device fast extra-args...
  local name=$1 dev=$2 fast=$3; shift 3
  [ -f $R/$name.json ] && { echo "skip $name"; return; }
  echo "== $name"; RHOFOLD_FAST=$fast $PY benchmarks/bench.py --device $dev --out $R/$name.json "$@" 2>&1 | grep -E '^\{|Error' | tail -2
}
S1="3meiA:$E/3meiA/3meiA 6tb7A:$E/6tb7A/6tb7A 4l81A:$E/4l81A/4l81A 4P8Z:benchmarks/4P8Z 7EZ0:benchmarks/7EZ0"
for sp in $S1; do n=${sp%%:*}; f=${sp#*:}
  run mps_fast_ss_$n mps 1 --fasta $f.fasta --pdb out/bench-pdb/${n}_mps_fast_ss.pdb
done
for sp in $S1; do n=${sp%%:*}; f=${sp#*:}; [ $n = 7EZ0 ] && continue
  run mps_off_ss_$n mps 0 --fasta $f.fasta --pdb out/bench-pdb/${n}_mps_off_ss.pdb
done
run mps_off_ss_7EZ0 mps 0 --fasta benchmarks/7EZ0.fasta --runs 1 --warmup 0   # expected OOM: record it
for n in 3meiA 6tb7A 4l81A; do
  run mps_fast_msa_$n mps 1 --fasta $E/$n/$n.fasta --a3m $E/$n/$n.a3m --pdb out/bench-pdb/${n}_mps_fast_msa.pdb
  run mps_off_msa_$n mps 0 --fasta $E/$n/$n.fasta --a3m $E/$n/$n.a3m
done
run mps_fast_msa128timing_4P8Z mps 1 --fasta benchmarks/4P8Z.fasta --a3m benchmarks/4P8Z_timing128.a3m
for n in 6tb7A 4l81A; do run mps_fastcompile_ss_$n mps 1 --compile --fasta $E/$n/$n.fasta; done
run mps_fastcompile_ss_4P8Z mps 1 --compile --fasta benchmarks/4P8Z.fasta
run mps_fastcompile_msa_4l81A mps 1 --compile --fasta $E/4l81A/4l81A.fasta --a3m $E/4l81A/4l81A.a3m
for sp in $S1; do n=${sp%%:*}; f=${sp#*:}; [ $n = 7EZ0 ] && continue
  run cpu_fast_ss_$n cpu 1 --fasta $f.fasta --pdb out/bench-pdb/${n}_cpu_fast_ss.pdb
done
for n in 3meiA 6tb7A 4l81A; do run cpu_off_ss_$n cpu 0 --fasta $E/$n/$n.fasta --pdb out/bench-pdb/${n}_cpu_off_ss.pdb; done
run cpu_fast_msa_4l81A cpu 1 --fasta $E/4l81A/4l81A.fasta --a3m $E/4l81A/4l81A.a3m
run cpu_fast_ss_7EZ0 cpu 1 --fasta benchmarks/7EZ0.fasta --runs 1 --warmup 0 --pdb out/bench-pdb/7EZ0_cpu_fast_ss.pdb
echo ALLDONE
