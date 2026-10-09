#!/usr/bin/env python
# ============================================================================
# TADDY_STAGE1.py
#
# TADDY: data adapter and environment check (run once before the TADDY
# training scripts).
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, sys, time, gzip, pickle, subprocess, urllib.request, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

DATASET   = 'btc_otc'
SNAP_SIZE = 2000        # TADDY's published value for btc_otc
SMOKE_EPOCHS = 5

T0 = time.time()
print('=' * 78); print('TADDY STAGE 1  --  real labels, bitcoin_otc'); print('=' * 78)

# ---------------------------------------------------------------- bootstrap
REPO_E = '/content/EVIDENT_repo'
if not os.path.isdir(REPO_E):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/Iyad-Assaad-Nekka/'
                    'EVIDENT-A-Sparse-Evidence-Transformer-for-Explainable-Anomaly-Detection-in-Real-World-Dynamic-Graphs.git',
                    REPO_E], check=True)
exec(compile(open(os.path.join(REPO_E, 'model.py')).read(), 'model.py', 'exec'),
     globals())
MASTER = os.path.join(BENCH, 'evident_master')
os.makedirs(MASTER, exist_ok=True); os.makedirs(DATA, exist_ok=True)
if not os.path.abspath(BENCH).startswith('/content/drive'):
    print('!! DRIVE NOT MOUNTED -- caches and results will be lost !!', flush=True)

REPO_T = '/content/TADDY'
if not os.path.isdir(REPO_T):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/yixinliu233/TADDY_pytorch.git',
                    REPO_T], check=True)
print(f'[taddy] {REPO_T}', flush=True)

import torch
print(f'[gpu] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')


# ------------------------------------------------------------------ data ----
def load_otc():
    p = os.path.join(DATA, 'bitcoinotc.csv')
    if not os.path.exists(p):
        open(p, 'wb').write(gzip.decompress(urllib.request.urlopen(
            'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz',
            timeout=300).read()))
    d = pd.read_csv(p, header=None, names=['u', 'v', 'r', 't'])
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    d['y'] = (d.r <= -5).astype(np.float32); d['r'] = d.r.astype(np.float32) / 10.0
    ids = pd.unique(pd.concat([d.u, d.v])); m = {int(x): i for i, x in enumerate(ids)}
    d['u'] = d.u.map(m).astype(np.int64); d['v'] = d.v.map(m).astype(np.int64)
    return d[['u', 'v', 't', 'y', 'r']].reset_index(drop=True), len(m)


df, n_nodes = load_otc()
print(f'[data] edges={len(df)} nodes={n_nodes} pos={df.y.mean():.4f}')

# ---- reuse the EXACT EVIDENT split ----------------------------------------
poolf = os.path.join(MASTER, 'poolplus_bitcoin_otc.npz')
KEEPF = os.path.join(MASTER, 'keepmask_bitcoin_otc.npy')
if os.path.exists(poolf):
    keep = np.load(poolf)['keep'].astype(bool); print('[split] keep-mask from EVIDENT pool')
elif os.path.exists(KEEPF):
    keep = np.load(KEEPF); print('[split] keep-mask from cache')
else:
    print('[split] recomputing EVIDENT keep-mask...', flush=True)
    _pb = PoolBuilder(df, n_nodes, CFG); _W = CFG['W']
    keep = np.zeros(len(df), bool)
    for _j in range(len(df)):
        _a, _b = int(_pb.u[_j]), int(_pb.v[_j]); _ts = _pb.t[_j]
        _ev = sorted(set(_pb.inc[_a][-_W:] + _pb.inc[_b][-_W:]))
        keep[_j] = len([e for e in _ev if _pb.t[e] < _ts and e != _j]) >= 1
        _pb.absorb(_j)
    np.save(KEEPF, keep)
tr_idx, va_idx, te_idx = chrono_split(keep, CFG)
print(f'[split] EVIDENT tr={len(tr_idx)} va={len(va_idx)} te={len(te_idx)}')

# TADDY has no val split: train = EVIDENT train+val, test = EVIDENT test
taddy_train = np.sort(np.concatenate([tr_idx, va_idx]))
taddy_test = te_idx
u, v, y = df.u.to_numpy(), df.v.to_numpy(), df.y.to_numpy()
print(f'[taddy] train={len(taddy_train)} test={len(taddy_test)} '
      f'test_pos={int(y[taddy_test].sum())} ({y[taddy_test].mean():.4f})')
if y[taddy_test].sum() < 50:
    print('[STOP] too few test positives'); sys.exit(1)

# ---- write TADDY's pickle directly, with REAL labels ----------------------
from scipy import sparse

train_edges = np.stack([u[taddy_train], v[taddy_train]], 1)
test_edges = np.stack([u[taddy_test], v[taddy_test]], 1)
test_lab = y[taddy_test].astype(np.int32)

train_mat = sparse.csr_matrix(
    (np.ones(len(train_edges)), (train_edges[:, 0], train_edges[:, 1])),
    shape=(n_nodes, n_nodes))
train_mat = (train_mat + train_mat.transpose() + sparse.eye(n_nodes)).tolil()
headtail = train_mat.rows
del train_mat

train_size = int(len(train_edges) / SNAP_SIZE + 0.5)
test_size = int(len(test_edges) / SNAP_SIZE + 0.5)
rows, cols, weis, labs = [], [], [], []
for ii in range(train_size):
    sl = slice(ii * SNAP_SIZE, (ii + 1) * SNAP_SIZE)
    r_ = np.array(train_edges[sl, 0], dtype=np.int32)
    rows.append(r_); cols.append(np.array(train_edges[sl, 1], dtype=np.int32))
    labs.append(np.zeros_like(r_, dtype=np.int32))   # TADDY ignores train labels
    weis.append(np.ones_like(r_, dtype=np.int32))
for ii in range(test_size):
    sl = slice(ii * SNAP_SIZE, (ii + 1) * SNAP_SIZE)
    r_ = np.array(test_edges[sl, 0], dtype=np.int32)
    rows.append(r_); cols.append(np.array(test_edges[sl, 1], dtype=np.int32))
    labs.append(np.array(test_lab[sl], dtype=np.int32))   # REAL labels
    weis.append(np.ones_like(r_, dtype=np.int32))

# the repo ships without these dirs and the loader does not create them
for _d in ('percent', 'eigen', 'interim', 'raw'):
    os.makedirs(f'{REPO_T}/data/{_d}', exist_ok=True)
# DynamicDatasetLoader opens 'data/percent/{name}_{train_per}_{anomaly_per}.pkl'
TRAIN_PER, ANOM_PER = 0.5, 0.0
PKL = f'{REPO_T}/data/percent/{DATASET}_real_{TRAIN_PER}_{ANOM_PER}.pkl'
with open(PKL, 'wb') as fh:
    pickle.dump((rows, cols, labs, weis, headtail, train_size, test_size,
                 n_nodes, len(train_edges) + len(test_edges)), fh,
                protocol=pickle.HIGHEST_PROTOCOL)
print(f'[taddy] wrote {PKL}')
print(f'[taddy] {train_size} train snapshots + {test_size} test snapshots '
      f'(snap_size={SNAP_SIZE})')
print(f'[taddy] test positives per snapshot: '
      f'{[int(l.sum()) for l in labs[train_size:]]}')

# ------------------------------------------------- PPR memory / time probe --
sys.path.insert(0, REPO_T)
os.chdir(REPO_T)
from codes.DynamicDatasetLoader import DynamicDatasetLoader

print(f'\n[probe] dense PPR inverse: n={n_nodes} -> '
      f'{n_nodes**2*8/1e9:.2f} GB per snapshot, '
      f'{(train_size+test_size)*n_nodes**2*8/1e9:.1f} GB for all '
      f'{train_size+test_size}', flush=True)
print('[probe] building (this is the expensive step)...', flush=True)

import resource
t1 = time.time()
ld = DynamicDatasetLoader()
ld.dataset_name = DATASET + '_real'
ld.k = 5
ld.window_size = 2
ld.anomaly_per = ANOM_PER
ld.train_per = TRAIN_PER
ld.load_all_tag = False
ld.compute_s = True
data = ld.load()
el = time.time() - t1
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'[probe] loader finished in {el/60:.1f} min, peak RSS {rss:.1f} GB')
print(f'[probe] keys: {list(data.keys())}')
print(f'[probe] snapshots: {len(data["X"]) if "X" in data else "?"}')

print('\n' + '=' * 78)
print('FEASIBILITY')
print('=' * 78)
print(f'  PPR precompute: {el/60:.1f} min, peak RSS {rss:.1f} GB')
if rss > 11.0:
    print('  *** RSS close to the Colab limit. Stage 2 should raise SNAP_SIZE')
    print('      (fewer, larger snapshots) or use a high-RAM runtime.')
else:
    print(f'  fits comfortably. Stage 2 can train with snap_size={SNAP_SIZE}.')
print(f'  200 epochs at TADDY defaults is the remaining unknown; stage 2')
print(f'  prints test AUC every 10 epochs so it can be stopped early.')
print(f'\nSTAGE 1 DONE in {(time.time()-T0)/60:.1f} min')
