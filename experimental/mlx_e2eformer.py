"""MLX port of RhoFold+'s E2Eformer trunk (12 blocks + single projection) — the hot path.

Loads the official weights from the PyTorch checkpoint and reproduces
rhofold.model.e2eformer.E2EformerStack.forward (inference path, inplace_safe=False,
chunk_size=None) for batch 1 with no padding (all masks = 1, which is what inference.py uses).

Usage: as a library (MLXE2Eformer), or `python mlx_e2eformer.py check|bench ...` (see __main__).
"""
import math, os, sys, time
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
CKPT = os.path.join(HERE, 'RhoFold/pretrained/RhoFold_pretrained.pt')

C_M, C_Z, H_MSA, C_MSA, H_PAIR, C_PAIR, C_OPM, C_MUL, N_BLOCKS = 256, 128, 8, 32, 4, 32, 32, 128, 12
OPM_EPS = 1e-3


def load_weights(ckpt=CKPT):
    import torch
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
    return {k[len('e2eformer.'):]: mx.array(v.float().numpy()) for k, v in sd.items() if k.startswith('e2eformer.')}


def ln(x, W, p, eps=1e-5):
    return mx.fast.layer_norm(x, W[p + '.weight'], W[p + '.bias'], eps)


def lin(x, W, p):
    y = x @ W[p + '.weight'].T
    b = W.get(p + '.bias')
    return y + b if b is not None else y


def attention(x, W, p, bias, H, C):
    """openfold Attention with gating; x [B, Q, c]; bias broadcastable to [B, H, Q, K]."""
    B, Q, _ = x.shape
    q = lin(x, W, p + '.linear_q').reshape(B, Q, H, C).transpose(0, 2, 1, 3) / math.sqrt(C)
    k = lin(x, W, p + '.linear_k').reshape(B, Q, H, C).transpose(0, 2, 1, 3)
    v = lin(x, W, p + '.linear_v').reshape(B, Q, H, C).transpose(0, 2, 1, 3)
    a = q @ k.transpose(0, 1, 3, 2)
    if bias is not None:
        a = a + bias
    a = mx.softmax(a, axis=-1, precise=True)
    o = (a @ v).transpose(0, 2, 1, 3)                           # B Q H C
    g = mx.sigmoid(lin(x, W, p + '.linear_g')).reshape(B, Q, H, C)
    return lin((o * g).reshape(B, Q, H * C), W, p + '.linear_o')


def msa_row_att(m, z, W, p):
    # m [S, L, c_m], z [L, L, c_z]
    zb = lin(ln(z, W, p + '.layer_norm_z'), W, p + '.linear_z')  # L L H
    zb = zb.transpose(2, 0, 1)[None]                              # 1 H L L
    return attention(ln(m, W, p + '.layer_norm_m'), W, p + '.mha', zb, H_MSA, C_MSA)


def msa_col_att(m, W, p):
    mt = m.transpose(1, 0, 2)                                     # L S c
    o = attention(ln(mt, W, p + '._msa_att.layer_norm_m'), W, p + '._msa_att.mha', None, H_MSA, C_MSA)
    return o.transpose(1, 0, 2)


def transition(x, W, p):
    return lin(mx.maximum(lin(ln(x, W, p + '.layer_norm'), W, p + '.linear_1'), 0), W, p + '.linear_2')


def opm(m, W, p):
    S, L, _ = m.shape
    x = ln(m, W, p + '.layer_norm')
    a = lin(x, W, p + '.linear_1')                                # S L c
    b = lin(x, W, p + '.linear_2')
    # outer[i, j, c, e] = sum_s a[s, i, c] b[s, j, e]
    outer = (a.reshape(S, L * C_OPM).T @ b.reshape(S, L * C_OPM))  # (L c) (L e)
    outer = outer.reshape(L, C_OPM, L, C_OPM).transpose(0, 2, 1, 3).reshape(L, L, C_OPM * C_OPM)
    return lin(outer, W, p + '.linear_out') / (S + OPM_EPS)


def tri_mul(z, W, p, outgoing):
    z = ln(z, W, p + '.layer_norm_in')
    a = mx.sigmoid(lin(z, W, p + '.linear_a_g')) * lin(z, W, p + '.linear_a_p')   # i k c
    b = mx.sigmoid(lin(z, W, p + '.linear_b_g')) * lin(z, W, p + '.linear_b_p')
    if outgoing:   # x_ij = sum_k a_ik b_jk
        x = a.transpose(2, 0, 1) @ b.transpose(2, 1, 0)
    else:          # x_ij = sum_k a_ki b_kj
        x = a.transpose(2, 1, 0) @ b.transpose(2, 0, 1)
    x = x.transpose(1, 2, 0)
    x = lin(ln(x, W, p + '.layer_norm_out'), W, p + '.linear_z')
    return x * mx.sigmoid(lin(z, W, p + '.linear_g'))


def tri_att(z, W, p, starting):
    if not starting:
        z = z.transpose(1, 0, 2)
    x = ln(z, W, p + '.layer_norm')
    tb = lin(x, W, p + '.linear').transpose(2, 0, 1)[None]       # 1 H L L
    o = attention(x, W, p + '.mha', tb, H_PAIR, C_PAIR)
    return o if starting else o.transpose(1, 0, 2)


def block(m, z, W, i):
    p = f'blocks.{i}'
    m = m + msa_row_att(m, z, W, p + '.msa_att_row')
    m = m + msa_col_att(m, W, p + '.msa_att_col')
    c = p + '.core'
    m = m + transition(m, W, c + '.msa_transition')
    z = z + opm(m, W, c + '.outer_product_mean')
    z = z + tri_mul(z, W, c + '.tri_mul_out', True)
    z = z + tri_mul(z, W, c + '.tri_mul_in', False)
    z = z + tri_att(z, W, c + '.tri_att_start', True)
    z = z + tri_att(z, W, c + '.tri_att_end', False)
    z = z + transition(z, W, c + '.pair_transition')
    return m, z


class MLXE2Eformer:
    def __init__(self, W=None, compile=True):
        self.W = W if W is not None else load_weights()
        f = lambda m, z, W, i: block(m, z, W, i)
        self._blocks = [mx.compile(lambda m, z, i=i: block(m, z, self.W, i)) if compile else
                        (lambda m, z, i=i: block(m, z, self.W, i)) for i in range(N_BLOCKS)]

    def __call__(self, m, z):
        """m [S, L, 256], z [L, L, 128] (batch dim dropped). Returns m, z, s."""
        for b in self._blocks:
            m, z = b(m, z)
        s = lin(m[0], self.W, 'linear')
        return m, z, s


if __name__ == '__main__':
    # check: compare against the official torch E2Eformer on real embeddings; bench: time both.
    import argparse, torch
    sys.path.insert(0, os.path.join(HERE, 'RhoFold'))
    from rhofold.rhofold import RhoFold
    from rhofold.config import rhofold_config
    from rhofold.utils.alphabet import get_features
    ap = argparse.ArgumentParser()
    ap.add_argument('--fasta', required=True); ap.add_argument('--a3m', default=None)
    ap.add_argument('--runs', type=int, default=5); ap.add_argument('--torch_device', default='mps')
    ap.add_argument('--no_compile', action='store_true'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    model = RhoFold(rhofold_config)
    model.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=False)['model']); model.eval()
    d = get_features(a.fasta, a.a3m or a.fasta)
    with torch.no_grad():
        msa_fea, pair_fea = model.msa_embedder(tokens=d['tokens'][:, :128], rna_fm_tokens=d['rna_fm_tokens'], is_BKL=True)
    S, L = msa_fea.shape[1], msa_fea.shape[2]
    res = {'L': L, 'msa_depth': S}
    # reference: torch fp32 on CPU (and timing on torch_device)
    def run_torch(dev):
        e = model.e2eformer.to(dev); m, z = msa_fea.to(dev), pair_fea.to(dev)
        with torch.no_grad():
            return e(m=m.clone(), z=z.clone(), msa_mask=torch.ones(m.shape[:3], device=dev),
                     pair_mask=torch.ones(z.shape[:3], device=dev), chunk_size=None)
    ref = [t.cpu().numpy() for t in run_torch('cpu')]
    mlx_trunk = MLXE2Eformer(compile=not a.no_compile)
    m0, z0 = mx.array(msa_fea[0].numpy()), mx.array(pair_fea[0].numpy())
    out = mlx_trunk(m0, z0); mx.eval(out)
    for name, r, o in zip(['m', 'z', 's'], ref, out):
        o = np.array(o); r = r[0]
        res[f'maxabs_{name}'] = float(np.abs(o - r).max()); res[f'relerr_{name}'] = float(np.linalg.norm(o - r) / np.linalg.norm(r))
    # timing: one trunk pass (= one recycle's E2Eformer), warmup excluded
    sync = (lambda: torch.mps.synchronize()) if a.torch_device == 'mps' else (lambda: None)
    run_torch(a.torch_device); tt = []
    for _ in range(a.runs):
        sync(); t = time.perf_counter(); run_torch(a.torch_device); sync(); tt.append(time.perf_counter() - t)
    tm = []
    for _ in range(a.runs):
        t = time.perf_counter(); o = mlx_trunk(m0, z0); mx.eval(o); tm.append(time.perf_counter() - t)
    res.update(torch_device=a.torch_device, torch_trunk_median_s=float(np.median(tt)), mlx_trunk_median_s=float(np.median(tm)),
               mlx_compiled=not a.no_compile, mlx_peak_gb=mx.get_peak_memory() / 1e9 if hasattr(mx, 'get_peak_memory') else None)
    import json; print(json.dumps(res))
    if a.out: json.dump(res, open(a.out, 'w'), indent=1)
