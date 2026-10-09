#!/usr/bin/env python
# ============================================================================
# SAD_STAGE1_ADAPTER.py
#
# SAD: environment check and data adapter (converts the Bitcoin event
# streams to SAD's input format).
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, sys, gzip, time, json, subprocess, urllib.request, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

DS       = 'btcotc'          # SAD dataset name; files become ml_btcotc.*
T0 = time.time()
print('=' * 78); print('SAD STAGE 1 -- environment, data adapter, source dump')
print('=' * 78)

REPO_S = '/content/SAD'
if not os.path.isdir(REPO_S):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/D10Andy/SAD.git', REPO_S], check=True)
os.makedirs(f'{REPO_S}/dataset', exist_ok=True)

# ---------------------------------------------------------------- 1. deps --
print('\n' + '=' * 78); print('1. torch_scatter  (the one real install risk)')
print('=' * 78)
import torch
print(f'[env] torch {torch.__version__}  cuda={torch.cuda.is_available()}')
try:
    import torch_scatter  # noqa
    print(f'[ok] torch_scatter {torch_scatter.__version__} already present')
except Exception:
    tv = torch.__version__.split('+')[0]
    cu = ('cu' + torch.version.cuda.replace('.', '')) if torch.version.cuda else 'cpu'
    url = f'https://data.pyg.org/whl/torch-{tv}+{cu}.html'
    print(f'[pip] installing torch_scatter from {url}', flush=True)
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        '--no-input', 'torch_scatter', '-f', url],
                       capture_output=True, text=True)
    try:
        import torch_scatter  # noqa
        print(f'[ok] torch_scatter {torch_scatter.__version__} installed')
    except Exception as e:
        print('[FAIL] torch_scatter unavailable:', e)
        print(r.stdout[-1500:]); print(r.stderr[-2500:])
        print('\n  FALLBACK: scatter ops are a handful of index_add_ calls and')
        print('  gets written. Do NOT skip and hope -- SAD will import-fail.')

# ------------------------------------------------------------ 2. adapter ---
print('\n' + '=' * 78); print('2. DATA ADAPTER -- bitcoin_otc into SAD format, OUR split')
print('=' * 78)

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
poolf = os.path.join(MASTER, 'poolplus_bitcoin_otc.npz')
KEEPF = os.path.join(MASTER, 'keepmask_bitcoin_otc.npy')
if os.path.exists(poolf):
    keep = np.load(poolf)['keep'].astype(bool); src = 'EVIDENT pool'
elif os.path.exists(KEEPF):
    keep = np.load(KEEPF); src = 'cache'
else:
    print('[split] recomputing the identical keep-mask...', flush=True)
    _pb = PoolBuilder(df, n_nodes, CFG); _W = CFG['W']
    keep = np.zeros(len(df), bool)
    for _j in range(len(df)):
        _a, _b = int(_pb.u[_j]), int(_pb.v[_j]); _ts = _pb.t[_j]
        _ev = sorted(set(_pb.inc[_a][-_W:] + _pb.inc[_b][-_W:]))
        keep[_j] = len([e for e in _ev if _pb.t[e] < _ts and e != _j]) >= 1
        _pb.absorb(_j)
    np.save(KEEPF, keep); src = 'recomputed'
tr_idx, va_idx, te_idx = chrono_split(keep, CFG)
print(f'[split] from {src}: tr={len(tr_idx)} va={len(va_idx)} te={len(te_idx)} '
      f'test_pos={int(df.y.to_numpy()[te_idx].sum())}')

# ---- SAD's on-disk format -------------------------------------------------
# ml_<name>.csv : u, i, ts, label, idx    (1-indexed nodes, idx from 1)
# ml_<name>.npy : edge features, row 0 is a zero pad row
# ml_<name>_node.npy : node features, zeros
#
# Node ids: SAD indexes adjacency lists by raw node id, so ids must be
# contiguous from 1. We do NOT use build_dataset_graph.reindex -- its
# bipartite branch asserts a structure bitcoin does not have, and its
# non-bipartite branch is just +1, which we do here explicitly.
FEAT_DIM = 4          # must match --input_dim at training time
keep_all = np.sort(np.concatenate([tr_idx, va_idx, te_idx]))
sub = df.iloc[keep_all].reset_index(drop=True)

ml = pd.DataFrame({
    'u':     sub.u.to_numpy() + 1,
    'i':     sub.v.to_numpy() + 1,
    'ts':    sub.t.to_numpy().astype(np.float64),
    'label': sub.y.to_numpy().astype(np.float64),
    'idx':   np.arange(1, len(sub) + 1, dtype=np.int64),
})
max_idx = int(max(ml.u.max(), ml.i.max()))

# Edge features. The current edge's OWN rating must never be a feature: it is
# what defines the label (r <= -5), so including it leaks the answer outright.
# Zeros keep SAD structural+temporal, which is what it is, and matches the
# README's own advice for datasets without edge features.
edge_feat = np.zeros((len(ml) + 1, FEAT_DIM), dtype=np.float32)
node_feat = np.zeros((max_idx + 1, FEAT_DIM), dtype=np.float32)

ml.to_csv(f'{REPO_S}/dataset/ml_{DS}.csv', index=False)
np.save(f'{REPO_S}/dataset/ml_{DS}.npy', edge_feat)
np.save(f'{REPO_S}/dataset/ml_{DS}_node.npy', node_feat)
print(f'[write] ml_{DS}.csv  {len(ml)} rows, nodes 1..{max_idx}')
print(f'[write] ml_{DS}.npy  {edge_feat.shape}   ml_{DS}_node.npy {node_feat.shape}')
print(f'[write] label rate: {ml.label.mean():.4f}  positives {int(ml.label.sum())}')

# ---- reproduce OUR split as quantiles of ts --------------------------------
# SAD computes: val_time, test_time = quantile(ts, [q0, q0+q1]) and then
#   train = ts <= val_time ; val = val_time < ts <= test_time ; test = ts > test_time
# We solve for the (q0, q1) that reproduce our chronological boundaries, then
# VERIFY edge-for-edge. If verification fails, the training cell must bypass
# get_data entirely rather than tune quantiles.
pos_in_sub = {int(g): k for k, g in enumerate(keep_all)}
tr_pos = np.array([pos_in_sub[int(i)] for i in tr_idx])
va_pos = np.array([pos_in_sub[int(i)] for i in va_idx])
te_pos = np.array([pos_in_sub[int(i)] for i in te_idx])
ts = ml.ts.to_numpy()

val_time = float(ts[tr_pos].max())
test_time = float(ts[va_pos].max())
q0 = float((ts <= val_time).mean())
q1 = float((ts <= test_time).mean()) - q0
sad_train = np.where(ts <= val_time)[0]
sad_val = np.where((ts > val_time) & (ts <= test_time))[0]
sad_test = np.where(ts > test_time)[0]

ok_tr = np.array_equal(np.sort(sad_train), np.sort(tr_pos))
ok_va = np.array_equal(np.sort(sad_val), np.sort(va_pos))
ok_te = np.array_equal(np.sort(sad_test), np.sort(te_pos))
print(f'\n[split->quantile] val_time={val_time:.1f} test_time={test_time:.1f}')
print(f'[split->quantile] q0={q0:.6f}  q1={q1:.6f}  '
      f'(pass --split_list {q0:.6f} {q1:.6f} {1-q0-q1:.6f})')
print(f'[verify] train {len(sad_train)} vs {len(tr_pos)}  exact={ok_tr}')
print(f'[verify] val   {len(sad_val)} vs {len(va_pos)}  exact={ok_va}')
print(f'[verify] test  {len(sad_test)} vs {len(te_pos)}  exact={ok_te}')
print(f'[verify] test positives {int(ml.label.to_numpy()[sad_test].sum())} '
      f'(EVIDENT/StrGNN/TADDY see 419)')
if ok_tr and ok_va and ok_te:
    print('\n  SPLIT IS IDENTICAL. SAD will be scored on exactly the edges the')
    print('  other three methods were scored on.')
else:
    print('\n  !! SPLIT MISMATCH -- ties in ts straddle a boundary. Stage 2 must')
    print('  OVERRIDE DygDataset.get_data with explicit index arrays instead of')
    print('  quantiles. Do not proceed on approximate boundaries: a SAD number')
    print('  measured on different edges is not comparable and is worse than')
    print('  no SAD row at all.')
json.dump(dict(dataset=DS, q0=q0, q1=q1, val_time=val_time, test_time=test_time,
               n=len(ml), feat_dim=FEAT_DIM, max_idx=max_idx,
               train_idx=sad_train.tolist()[:0], exact=bool(ok_tr and ok_va and ok_te)),
          open(f'{REPO_S}/dataset/{DS}_split.json', 'w'))
np.savez(f'{REPO_S}/dataset/{DS}_split_idx.npz',
         train=tr_pos, val=va_pos, test=te_pos)
print(f'[write] split index arrays -> dataset/{DS}_split_idx.npz '
      f'(Stage 2 uses these directly if the quantile route is unsafe)')

# ------------------------------------------------------ 3. source dump -----
print('\n' + '=' * 78); print('3. SOURCE DUMP for writing Stage 2')
print('=' * 78)
for f, lo, hi in [('datasets.py', 60, 320), ('train.py', 0, 300),
                  ('model/tgat.py', 0, 120), ('model/gdn.py', 0, 120)]:
    p = os.path.join(REPO_S, f)
    print(f'\n{"="*78}\n--- {f}  (lines {lo}-{hi}) ---\n{"="*78}')
    try:
        L = open(p, encoding='utf-8', errors='replace').read().split('\n')
        for i, ln in enumerate(L[lo:hi], lo + 1):
            print(f'{i:>4} {ln}')
    except Exception as e:
        print('<unreadable>', e)

print('\n' + '=' * 78)
print(f'DONE in {(time.time()-T0)/60:.1f} min.  NOTHING WAS TRAINED.')
print('against this source, with --input_dim 4, bipartite forced off, and the')
print('split pinned to the arrays written above.')
print('=' * 78)
