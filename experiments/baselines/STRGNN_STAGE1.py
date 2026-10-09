#!/usr/bin/env python
# ============================================================================
# STRGNN_STAGE1.py
#
# StrGNN: enclosing-subgraph extraction and throughput check (run once
# before the StrGNN training scripts).
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, sys, time, gzip, math, subprocess, urllib.request, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

DATASET   = 'bitcoin_otc'      # or 'bitcoin_alpha'
N_SNAP    = 60                 # snapshots the stream is cut into
WINDOW    = 5                  # StrGNN default
SMOKE_TR  = 300                # smoke-test train edges
SMOKE_TE  = 200                # smoke-test test edges

T0 = time.time()
print('=' * 76); print('StrGNN STAGE 1'); print('=' * 76)

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
print(f'[evident] bootstrap ok, MASTER={MASTER}')

# ---------------------------------------------------------------- StrGNN ---
REPO_S = '/content/StrGNN'
if not os.path.isdir(REPO_S):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/KnowledgeDiscovery/StrGNN.git',
                    REPO_S], check=True)

print('[strgnn] compiling C++ lib (ctypes only, no torch linkage)...')
r = subprocess.run('cd %s/pytorch_DGCNN/lib && make clean >/dev/null 2>&1; '
                   'make -j4' % REPO_S, shell=True, capture_output=True, text=True)
so = f'{REPO_S}/pytorch_DGCNN/lib/build/dll/libgnn.so'
print(f'[strgnn] libgnn.so present: {os.path.exists(so)}', flush=True)
if not os.path.exists(so):
    print(r.stdout[-2000:]); print(r.stderr[-2000:]); sys.exit(1)

# ---- deps: only install what is genuinely missing --------------------------
# Colab already ships networkx / scikit-learn / tqdm. An unconditional
# `pip install` makes the resolver walk the whole torch dependency tree and
# can stall for 10+ minutes, so we check first and usually do nothing.
_missing = []
for _mod, _pkg in [('networkx', 'networkx'), ('sklearn', 'scikit-learn'),
                   ('tqdm', 'tqdm')]:
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)
if _missing:
    print(f'[deps] installing {_missing} ...', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    '--no-input'] + _missing, check=False, timeout=600)
else:
    print('[deps] networkx / scikit-learn / tqdm already present, '
          'nothing to install', flush=True)

# ---- patch 0: stub the dead node2vec path ---------------------------------
import types


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _dead(*a, **k):
    raise RuntimeError('node2vec/gensim path is disabled: '
                       'StrGNN runs with use_embedding=False')


if 'gensim' not in sys.modules:
    _stub('gensim')
    _stub('gensim.models', Word2Vec=_dead)
    sys.modules['gensim'].models = sys.modules['gensim.models']
if 'node2vec' not in sys.modules:
    _stub('node2vec', Graph=_dead)
print('[patch] stubbed gensim + node2vec (dead code: use_embedding=False)', flush=True)

# ---- patch 1: networkx 3.x ------------------------------------------------
import networkx as nx
if not hasattr(nx, 'from_scipy_sparse_matrix'):
    nx.from_scipy_sparse_matrix = lambda A, **kw: nx.from_scipy_sparse_array(A, **kw)
    print('[patch] nx.from_scipy_sparse_matrix -> from_scipy_sparse_array', flush=True)

# NOTE, not patched on purpose: node_label() casts inf distances to int on
# disconnected subgraphs, which emits a RuntimeWarning. StrGNN then clamps
# |labels| > 1e6 back to 0, so it is handled. We leave their code untouched --
# a faithful baseline matters more than a tidy log -- and only silence the
# warning. Disclose this in the reproducibility note.

# ---- patch 2: gpu id ------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("[patch] CUDA_VISIBLE_DEVICES = '0'  (repo hardcodes '1')", flush=True)

sys.path.insert(0, f'{REPO_S}/pytorch_DGCNN')
sys.path.insert(0, f'{REPO_S}/detection')
_argv = sys.argv; sys.argv = ['main.py']      # pytorch_DGCNN/main.py parses argv
import util_functions as UF
from util_functions import subgraph_extraction_labeling, GNNGraph
sys.argv = _argv
print('[strgnn] util_functions imported', flush=True)

import scipy.sparse as ssp
from sklearn.metrics import roc_auc_score, average_precision_score


# =============================================================================
#  DATA -- identical stream, identical split, identical test set
# =============================================================================
def load_bitcoin(which):
    fn = 'bitcoinotc.csv' if which == 'otc' else 'bitcoinalpha.csv'
    url = f'https://snap.stanford.edu/data/soc-sign-bitcoin{which}.csv.gz'
    p = os.path.join(DATA, fn)
    if not os.path.exists(p):
        open(p, 'wb').write(gzip.decompress(
            urllib.request.urlopen(url, timeout=300).read()))
    d = pd.read_csv(p, header=None, names=['u', 'v', 'r', 't'])
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    d['y'] = (d.r <= -5).astype(np.float32); d['r'] = d.r.astype(np.float32) / 10.0
    ids = pd.unique(pd.concat([d.u, d.v])); m = {int(x): i for i, x in enumerate(ids)}
    d['u'] = d.u.map(m).astype(np.int64); d['v'] = d.v.map(m).astype(np.int64)
    return d[['u', 'v', 't', 'y', 'r']].reset_index(drop=True), len(m)


which = 'otc' if DATASET == 'bitcoin_otc' else 'alpha'
df, n_nodes = load_bitcoin(which)
print(f'\n[data] {DATASET}: edges={len(df)} nodes={n_nodes} pos={df.y.mean():.4f}')

# ---- reuse the EXACT EVIDENT split ----------------------------------------
CACHE = os.path.join(MASTER, f'poolplus_{DATASET}.npz')
if os.path.exists(CACHE):
    z = np.load(CACHE); keep = z['keep'].astype(bool)
    print('[split] using cached EVIDENT pool keep-mask -> identical test set')
else:
    keep = np.ones(len(df), bool)
    print('[split] WARNING: EVIDENT pool not on this Drive; keep-mask = all.')
    print('        Test sets will NOT match EVIDENT exactly. Run the EVIDENT')
    print('        cell first on this account for a like-for-like comparison.')
tr_idx, va_idx, te_idx = chrono_split(keep, CFG)
print(f'[split] tr={len(tr_idx)} va={len(va_idx)} te={len(te_idx)} '
      f'test_pos={df.y.to_numpy()[te_idx].mean():.4f}')

# ---- snapshots: accumulated sparse adjacency ------------------------------
u, v = df.u.to_numpy(), df.v.to_numpy()
bounds = np.linspace(0, len(df), N_SNAP + 1).astype(int)
snap_of = np.zeros(len(df), np.int64)
for s in range(N_SNAP):
    snap_of[bounds[s]:bounds[s + 1]] = s

print(f'[snap] building {N_SNAP} accumulated sparse snapshots...')
nets, t1 = [], time.time()
for s in range(N_SNAP):
    e = bounds[s + 1]
    A = ssp.csr_matrix((np.ones(e, np.float32), (u[:e], v[:e])),
                       shape=(n_nodes, n_nodes))
    A = A + A.T
    A.data[:] = 1.0
    A.setdiag(0); A.eliminate_zeros()
    nets.append(A.tocsr())
mb = sum(a.data.nbytes + a.indices.nbytes + a.indptr.nbytes for a in nets) / 1e6
print(f'[snap] done in {time.time()-t1:.0f}s, total {mb:.0f} MB '
      f'(dense would be {N_SNAP*n_nodes**2*4/1e9:.1f} GB)')


def build_graphs(idx, tag, limit=None):
    """One GNNGraph per (edge, snapshot-in-window). Label = real fraud label."""
    y = df.y.to_numpy()
    sel = idx if limit is None else idx[:limit]
    out, labels, maxlab, t = [], [], 0, time.time()
    skipped = 0
    for c, j in enumerate(sel):
        n = int(snap_of[j])
        if n - WINDOW + 1 < 0:
            skipped += 1; continue
        dl = []
        for g in range(n - WINDOW + 1, n + 1):
            gg, nl, nf = subgraph_extraction_labeling(
                (int(u[j]), int(v[j])), nets[g], 1, None, None)
            maxlab = max(maxlab, max(nl))
            dl.append(GNNGraph(gg, int(y[j]), nl, nf))
        out.append(dl); labels.append(int(y[j]))
        if (c + 1) % 100 == 0:
            el = time.time() - t
            print(f'   {tag}: {c+1}/{len(sel)}  {el:.0f}s  '
                  f'({(c+1)/el:.1f} edges/s)', flush=True)
    return out, np.array(labels), maxlab, skipped


print(f'\n[smoke] extracting {SMOKE_TR} train + {SMOKE_TE} test edges '
      f'({WINDOW} subgraphs each)...')
rng = np.random.default_rng(0)
tr_s = np.sort(rng.choice(tr_idx, min(SMOKE_TR, len(tr_idx)), replace=False))
te_s = np.sort(rng.choice(te_idx, min(SMOKE_TE, len(te_idx)), replace=False))

t2 = time.time()
gtr, ytr, ml1, sk1 = build_graphs(tr_s, 'train')
gte, yte, ml2, sk2 = build_graphs(te_s, 'test')
extract_s = time.time() - t2
per_edge = extract_s / max(len(gtr) + len(gte), 1)

print(f'\n[smoke] extraction: {len(gtr)} train + {len(gte)} test graphs '
      f'in {extract_s:.0f}s  ({per_edge*1000:.0f} ms/edge)')
print(f'[smoke] skipped (snapshot < window): train {sk1}, test {sk2}')
print(f'[smoke] max node label = {max(ml1, ml2)}')
print(f'[smoke] train pos rate {ytr.mean():.4f}, test pos rate {yte.mean():.4f}')

# ---------------------------------------------------------------- projection
full_tr, full_te = len(tr_idx), len(te_idx)
proj = (full_tr + full_te) * per_edge / 60
print('\n' + '=' * 76)
print('PROJECTION FOR THE FULL RUN')
print('=' * 76)
print(f'  extraction, all {full_tr} train + {full_te} test edges: '
      f'{proj:.0f} min')
print(f'  per seed, 50 epochs DGCNN+GRU on {full_tr*WINDOW} subgraphs: '
      f'unknown until stage 2')
print(f'  -> if extraction > 40 min, stage 2 must subsample TRAIN negatives')
print(f'     (StrGNN has --max-train-num for exactly this). TEST stays whole,')
print(f'     otherwise the comparison with EVIDENT is not like-for-like.')
print(f'\n  suggested: cache extracted graphs to Drive so seeds reuse them')
print(f'             (extraction is seed-independent)')
print(f'\nSTAGE 1 DONE in {(time.time()-T0)/60:.1f} min')
