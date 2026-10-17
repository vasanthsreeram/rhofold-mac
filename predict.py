"""RhoFold+ inference on Apple Silicon (PyTorch, MPS GPU with CPU fallback).

Same outputs as upstream inference.py (unrelaxed_model.pdb, ss.ct, results.npz) but:
  - picks the Apple GPU (MPS) automatically, falls back to CPU;
  - single-sequence mode by default (no BLAST/Infernal MSA search, no databases to download);
  - no Amber relaxation (needs OpenMM; upstream marks macOS unsupported);
  - RHOFOLD_FAST=1 speedups on by default (exact rewrites, see rhofold-mac-fast.patch); --official turns them off.
Usage: python predict.py input.fasta out_dir [--a3m msa.a3m] [--device auto|mps|cpu] [--official]
"""
import argparse, json, os, sys, time, warnings

warnings.filterwarnings('ignore', message='Using torch.cross without specifying the dim')

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('fasta', help='input FASTA, one RNA sequence (A, U, G, C)')
ap.add_argument('out_dir', help='output folder (created if missing)')
ap.add_argument('--a3m', default=None, help='optional precomputed MSA (.a3m); default: single-sequence mode')
ap.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu'])
ap.add_argument('--official', action='store_true', help='disable the RHOFOLD_FAST speedups (upstream code path)')
ap.add_argument('--ckpt', default=os.path.join(HERE, 'RhoFold/pretrained/RhoFold_pretrained.pt'))
a = ap.parse_args()

# both read at import time by the patched rhofold modules / torch
os.environ['RHOFOLD_FAST'] = '0' if a.official else '1'
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
sys.path.insert(0, os.path.join(HERE, 'RhoFold'))

import numpy as np, torch
from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils import save_ss2ct
from rhofold.utils.alphabet import get_features

if not os.path.isfile(a.fasta):
    sys.exit(f'Input not found: {a.fasta}')
if not os.path.isfile(a.ckpt):
    sys.exit(f'Weights not found at {a.ckpt}. Run ./setup.sh first.')

dev = a.device
if dev == 'auto':
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
elif dev == 'mps' and not torch.backends.mps.is_available():
    sys.exit('MPS is not available here (needs Apple Silicon, macOS 12.3+, arm64 Python). Use --device cpu.')
if dev == 'cpu' and a.device == 'auto':
    print('MPS (Apple GPU) not available, running on CPU. This is slower but gives the same structure.')

os.makedirs(a.out_dir, exist_ok=True)
t0 = time.perf_counter()
model = RhoFold(rhofold_config)
model.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=False)['model'])
model.eval().to(dev)
t_load = time.perf_counter() - t0

d = get_features(a.fasta, a.a3m or a.fasta)  # single-sequence mode: the MSA is the query itself (as upstream does)
L = len(d['seq'])
mode = 'msa' if a.a3m else 'single-sequence'
print(f'RhoFold+ | {os.path.basename(a.fasta)} | {L} nt | {mode} | device={dev} | fast={not a.official}')

t1 = time.perf_counter()
try:
    with torch.no_grad():
        out = model(tokens=d['tokens'].to(dev), rna_fm_tokens=d['rna_fm_tokens'].to(dev), seq=d['seq'])[-1]
        if dev == 'mps':
            torch.mps.synchronize()
except RuntimeError as e:
    if dev == 'mps' and ('out of memory' in str(e).lower() or 'MPS backend' in str(e)):
        sys.exit(f'{e}\n\nThe Apple GPU ran out of memory for {L} nt. Close other apps, or re-run with --device cpu '
                 '(slower, uses ordinary RAM). See README "Troubleshooting".')
    raise
t_pred = time.perf_counter() - t1

plddt = out['plddt'][0].float().cpu().numpy()
xyz = out['cord_tns_pred'][-1].squeeze(0).float().cpu().numpy()
ss_prob = torch.sigmoid(out['ss'][0, 0]).float().cpu().numpy()

pdb = os.path.join(a.out_dir, 'unrelaxed_model.pdb')
model.structure_module.converter.export_pdb_file(d['seq'], xyz, path=pdb, chain_id=None, confidence=plddt)
save_ss2ct(ss_prob, d['seq'], os.path.join(a.out_dir, 'ss.ct'), threshold=0.5)
np.savez_compressed(os.path.join(a.out_dir, 'results.npz'),
                    dist_n=torch.softmax(out['n'].squeeze(0), dim=0).float().cpu().numpy(),
                    dist_p=torch.softmax(out['p'].squeeze(0), dim=0).float().cpu().numpy(),
                    dist_c=torch.softmax(out['c4_'].squeeze(0), dim=0).float().cpu().numpy(),
                    ss_prob_map=ss_prob, plddt=plddt)
info = dict(input=os.path.abspath(a.fasta), length=L, mode=mode, device=dev, fast=not a.official,
            model_load_s=round(t_load, 2), predict_s=round(t_pred, 2), mean_plddt=round(float(plddt.mean()), 4),
            torch=torch.__version__)
json.dump(info, open(os.path.join(a.out_dir, 'run_info.json'), 'w'), indent=1)
print(f'Model load {t_load:.1f} s | prediction {t_pred:.1f} s | mean pLDDT {plddt.mean():.3f}')
print(f'Wrote {pdb} (+ ss.ct, results.npz, run_info.json)')
