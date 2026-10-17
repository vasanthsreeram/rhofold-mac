"""RhoFold+ forward-pass benchmark on Apple Silicon (CPU / MPS).
Times only the network forward pass (10 recycles, as in the official config),
excluding MSA search, feature building, PDB writing and Amber relaxation.
Usage: bench.py --device mps --fasta X.fasta [--a3m X.a3m] --runs 5 --out out.json --pdb out.pdb
"""
import argparse, json, os, resource, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, os.path.join(HERE, 'RhoFold'))
import numpy as np, torch
from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils.alphabet import get_features

import ctypes
def peak_footprint_gb():
    # macOS proc_pid_rusage(RUSAGE_INFO_V4).ri_lifetime_max_phys_footprint: process peak incl. Metal (unified) buffers
    libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    buf = (ctypes.c_uint64 * 64)()
    assert libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) == 0
    return buf[2 + 28] / 1e9  # 16-byte uuid = 2 uint64 slots

def sync(dev):
    if dev == 'mps': torch.mps.synchronize()

ap = argparse.ArgumentParser()
ap.add_argument('--device', default='cpu'); ap.add_argument('--fasta', required=True)
ap.add_argument('--a3m', default=None); ap.add_argument('--runs', type=int, default=5)
ap.add_argument('--warmup', type=int, default=1); ap.add_argument('--out', required=True)
ap.add_argument('--pdb', default=None); ap.add_argument('--compile', action='store_true'); ap.add_argument('--threads', type=int, default=0)
a = ap.parse_args()
if a.threads: torch.set_num_threads(a.threads)
dev = a.device
model = RhoFold(rhofold_config)
model.load_state_dict(torch.load(os.path.join(HERE, 'RhoFold/pretrained/RhoFold_pretrained.pt'), map_location='cpu', weights_only=False)['model'])
model.eval().to(dev)
if a.compile:  # torch.compile (inductor) on each E2Eformer block; compile time lands in the warmup run
    for i, b in enumerate(model.e2eformer.blocks): model.e2eformer.blocks[i] = torch.compile(b, dynamic=False)
d = get_features(a.fasta, a.a3m or a.fasta)  # single-seq mode == a3m is the fasta itself (as inference.py does)
tok, fm = d['tokens'].to(dev), d['rna_fm_tokens'].to(dev)
L = len(d['seq']); depth = tok.shape[1]
times = []; warm = None
with torch.no_grad():
    for i in range(a.warmup + a.runs):
        sync(dev); t0 = time.perf_counter()
        outs = model(tokens=tok, rna_fm_tokens=fm, seq=d['seq'])
        sync(dev); dt = time.perf_counter() - t0
        if i >= a.warmup: times.append(dt)
        elif i == 0: warm = dt
    o = outs[-1]
    xyz = o['cord_tns_pred'][-1].squeeze(0).float().cpu().numpy()
    plddt = o['plddt'][0].float().cpu().numpy()
res = dict(device=dev, fasta=os.path.basename(a.fasta), L=L, msa_depth=int(depth),
           mode='msa' if a.a3m else 'single_seq', times=times, median=float(np.median(times)),
           min=float(min(times)), threads=torch.get_num_threads(),
           fast=os.environ.get('RHOFOLD_FAST', '0') == '1', compile=a.compile,
           peak_footprint_gb=peak_footprint_gb(), warmup_s=warm,
           peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,  # bytes on macOS
           mean_plddt=float(plddt.mean()))
if dev == 'mps':
    res['mps_driver_alloc_gb'] = torch.mps.driver_allocated_memory() / 1e9
np.save(a.out.replace('.json', '.xyz.npy'), xyz)
if a.pdb:
    model.structure_module.converter.export_pdb_file(d['seq'], xyz, path=a.pdb, chain_id=None, confidence=plddt)
json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps({k: v for k, v in res.items() if k != 'times'}))
