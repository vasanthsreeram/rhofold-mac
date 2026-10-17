# rhofold-mac

**RhoFold+ on Apple Silicon via PyTorch MPS; MLX path experimental.**

Predict an RNA's 3D structure from its sequence on an M-series Mac, on the Apple GPU, with the official [RhoFold+](https://github.com/ml4bio/RhoFold) model. One setup command, one run command.

## Quick start

**You need:** a Mac with Apple Silicon (M1 or newer), macOS 12.3+ (tested on 15.5), git (`xcode-select --install`), about 3 GB free disk, and an internet connection for the one-time download. Python is handled for you if [uv](https://docs.astral.sh/uv/) is installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`); otherwise any Python 3.10+ works.

**1. Set up (once):**

```bash
git clone https://github.com/vasanthsreeram/rhofold-mac.git
cd rhofold-mac
./setup.sh
```

It downloads about 0.5 GB of weights plus the Python packages. On a fast connection it took 25 s here (fresh clone, empty package cache); expect a few minutes on a slower one. It ends with a self-check that should say `All good`.

**2. Fold the bundled sample:**

```bash
./run.sh
```

With no arguments it folds `examples/sample.fasta`, a 52-nt RNA (chain A of PDB entry [6TB7](https://www.rcsb.org/structure/6TB7), copied from RhoFold's own example inputs), on the Apple GPU. Expected output on an M4 MacBook (16 GB):

```
RhoFold+ | sample.fasta | 52 nt | single-sequence | device=mps | fast=True
Model load 1.2 s | prediction 2.2 s | mean pLDDT 0.544
Wrote out/sample/unrelaxed_model.pdb (+ ss.ct, results.npz, run_info.json)
View it: drag out/sample/unrelaxed_model.pdb onto https://molstar.org/viewer, or open it in PyMOL / ChimeraX.
Total wall time (Python start-up + model load + prediction): 8 s
```

The first run after setup took 8 s in total on this Mac, while macOS loads the Python libraries for the first time; later runs took 4 s. Prediction itself is 2.2–2.7 s.

**3. Look at the result.** Everything lands in `out/sample/`. Drag `out/sample/unrelaxed_model.pdb` onto [molstar.org/viewer](https://molstar.org/viewer) in your browser (nothing to install), or open it in PyMOL or ChimeraX. Colour by B-factor to see per-residue confidence (pLDDT, 0–100).

**Your own RNA:** put one sequence (A, U, G, C) in a FASTA file and run `./run.sh my_rna.fasta`. Results go to `out/my_rna/`; add a second argument to choose the folder (`./run.sh my_rna.fasta results/`).

**Something wrong?** Run `./doctor.sh`. It checks the Mac, Python, PyTorch and the Apple GPU, the RhoFold code and patch, and the weights' checksum, and says what to do for anything that fails. See also [Troubleshooting](#troubleshooting).

## What this is

[RhoFold+](https://github.com/ml4bio/RhoFold) (Shen et al., *Nature Methods* 2024) predicts RNA 3D structure from sequence. Upstream supports Linux and says *"MacOS is currently not supported"*. This repo sets it up on an M-series Mac in one command and runs it on the Apple GPU through PyTorch's MPS (Metal) backend, with a CPU fallback. To our knowledge it is the first public Mac-friendly RhoFold+ runner.

It is not a re-implementation. `setup.sh` creates `.venv` with pinned packages (`requirements.txt`), clones `ml4bio/RhoFold` at commit `6bdfbda` into `RhoFold/`, applies a small patch (`rhofold-mac-fast.patch`), downloads `rhofold_pretrained_params.pt` from the authors' [Hugging Face repo](https://huggingface.co/cuhkaih/rhofold) and checks its SHA-256. It is safe to re-run, and a broken download resumes. The model, weights and predictions are the authors'.

Not affiliated with the RhoFold+ authors or CUHK.

**Memory:** an 8 GB Mac is fine for short RNAs. Peak memory on the M4 was 1.9 GB at 23–52 nt, 2.7 GB at 96 nt, 7.0 GB at 188 nt and 10.7 GB at 387 nt (see [Benchmarks](#benchmarks)).

## Outputs

| File | What |
|---|---|
| `unrelaxed_model.pdb` | Predicted 3D structure; per-residue pLDDT (0–100) in the B-factor column |
| `ss.ct` | Predicted secondary structure (CT format) |
| `results.npz` | Distance distributions (`dist_n`, `dist_p`, `dist_c`), base-pair probability map, pLDDT |
| `run_info.json` | Length, device, timings, mean pLDDT, torch version |

## Options

```
./run.sh [input.fasta] [out_dir] [--a3m MSA.a3m] [--device auto|mps|cpu] [--official]
```

- No `input.fasta`: the bundled sample. No `out_dir`: `out/<input name>/`.
- **Single-sequence mode by default.** Upstream builds an MSA with BLAST and Infernal against RNAcentral and NCBI nt (hundreds of GB). That is skipped here: the sequence itself is used as the MSA, as in upstream's `--single_seq_pred True`. This keeps the sample to a few seconds but is usually less accurate than an MSA. If you have an MSA (e.g. from the [RhoFold+ web server](https://proj.cse.cuhk.edu.hk/aihlab/RhoFold/) or your own search), pass it with `--a3m`.
- **No Amber relaxation.** Upstream's optional relaxation needs OpenMM; the output is the unrelaxed model (the same file upstream writes before relaxing).
- `--device` defaults to `auto`: the Apple GPU (MPS) if available, otherwise the CPU (it prints a note when it falls back).
- `--official` turns off the speed patch and runs upstream's code path unchanged.

## What the patch changes

All changes are behind `RHOFOLD_FAST=1` (set by `predict.py` unless you pass `--official`) and are exact rewrites; the final structure is the same (on every benchmark RNA below, mean pLDDT is identical with the patch on and off).

1. The MSA / RNA-FM embedding is computed once and reused across the 10 recycles (its inputs do not change between recycles).
2. The EGNN refinement net runs only on the last recycle; earlier refined coordinates are never read.
3. The EGNN edge MLP's first linear layer is split per node (`W·[f_i; f_j; d] = W_i f_i + W_j f_j + w_d d`), and the edge tensor is chunked so long RNAs fit in GPU memory.

## Benchmarks

Measured on an Apple M4 MacBook (16 GB, 2026-10-14, PyTorch 2.14.1, MPS). Network forward pass only (10 recycles, single-sequence), median of 5 runs after 1 warm-up. Raw files: `benchmarks/results/*.json`; script: `benchmarks/bench.py`, matrix: `benchmarks/run_bench.sh`.

| RNA (PDB chain) | Length | Official code, MPS | Patched, MPS | Peak memory (patched) |
|---|---:|---:|---:|---:|
| 3MEI A | 23 nt | 1.19 s | 0.87 s | 1.9 GB |
| 6TB7 A (the sample) | 52 nt | 2.87 s | 1.50 s | 1.9 GB |
| 4L81 A | 96 nt | 7.57 s | 3.54 s | 2.7 GB |
| 4P8Z | 188 nt | 29.0 s | 16.3 s | 7.0 GB |
| 7EZ0 | 387 nt | did not complete (no result recorded) | 83.4 s | 10.7 GB |

The quick-start "prediction" time for the sample is a single cold run including moving inputs to the GPU, so it is a bit above the warmed-up 1.50 s in the table.

## Troubleshooting

Start with `./doctor.sh`: it names the problem and the fix. The common ones:

- **`Note: MPS (Apple GPU) not available, running on CPU instead`**: the run still works, just slower. Causes: an Intel Mac (not supported), macOS older than 12.3, or a terminal running under Rosetta (x86 emulation). For Rosetta: Finder > Applications > Utilities > Terminal > Get Info > untick "Open using Rosetta", then delete `.venv` and re-run `./setup.sh`.
- **`need Python 3.10+`**: macOS's built-in `python3` is 3.9. Install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`), open a new terminal and re-run `./setup.sh`.
- **`weights download failed`**: network, proxy or Hugging Face trouble. Re-run `./setup.sh` to resume; the message also prints the URL to download by hand and where to save it.
- **`downloaded weights are damaged (SHA-256 mismatch)`** or **`could not load the weights`**: the file is deleted or flagged; re-run `./setup.sh` to fetch it again.
- **`sequence contains 'T'`** (or other letters): RhoFold+ takes RNA letters A, U, G, C only. Replace T with U.
- **Out of memory on the GPU** (`MPS backend out of memory`): close other apps, or use `--device cpu` (slower, uses normal RAM and swap). RNAs of 300+ nt need 10+ GB.
- **`patch did not apply`** / **`RhoFold/ is not at 6bdfbda`**: files in `RhoFold/` were edited. Move `RhoFold/pretrained/RhoFold_pretrained.pt` aside, delete `RhoFold/`, re-run `./setup.sh`, and put the weights back (or let it download them again).
- **Torch warnings** about deprecated functions come from upstream code and are harmless.

## Experimental: MLX trunk (off by default)

Not installed or used unless you ask for it: `./setup.sh` and `./run.sh` are PyTorch only. `experimental/mlx_e2eformer.py` is an [MLX](https://github.com/ml-explore/mlx) port of RhoFold+'s E2Eformer trunk only (the 12 attention blocks), for comparing numerics and speed against PyTorch. It reads the PyTorch checkpoint at runtime; there are no converted MLX weights, and it does not produce structures.

**It does not match PyTorch yet.** Its built-in check compares the trunk outputs against the official PyTorch trunk on the same inputs. On 2026-10-17 (fresh clone, M4) the relative error was 0.27–0.41 on the 52-nt sample and 0.36–0.65 on 4P8Z (188 nt), so there is a bug in the port and its timings mean nothing until that is fixed. Use the PyTorch MPS runner above for predictions.

```bash
./setup.sh --with-mlx
.venv/bin/python experimental/mlx_e2eformer.py --fasta examples/sample.fasta
```

## Credit, citation and license

RhoFold+ is by Tao Shen, Zhihang Hu, Siqi Sun et al. (CUHK and collaborators). If you use it, cite their paper (BibTeX in the upstream README):

> Shen, T., Hu, Z., Sun, S. et al. Accurate RNA 3D structure prediction using a language model-based deep learning approach. *Nature Methods* 21, 2287–2298 (2024). https://doi.org/10.1038/s41592-024-02487-0

Upstream code: [github.com/ml4bio/RhoFold](https://github.com/ml4bio/RhoFold), Apache License 2.0. Weights: [huggingface.co/cuhkaih/rhofold](https://huggingface.co/cuhkaih/rhofold). Neither is redistributed here; `setup.sh` fetches them from those sources. `rhofold-mac-fast.patch` modifies upstream files and is distributed under the same Apache 2.0 license.

This repo's own code (setup and run scripts, `predict.py`, benchmarks, patch, MLX experiment) is Apache License 2.0; see `LICENSE` and `NOTICE`.
