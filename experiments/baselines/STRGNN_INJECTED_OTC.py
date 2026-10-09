#!/usr/bin/env python
# ============================================================================
# STRGNN_INJECTED_OTC.py
#
# StrGNN under the injected-anomaly protocol on Bitcoin-OTC.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, sys, time, gzip, math, pickle, subprocess, urllib.request, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

DATASET  = 'bitcoin_otc'          # or 'bitcoin_alpha'
SEEDS    = [0, 1]                 # 2 seeds per rate; ~20 min each
REDO     = False
INJ_RATES = [0.05]                # start with 5%; add 0.01 / 0.10 later
PATIENCE = 5                      # stop when val AUC has not improved for 5
# Resume-safe: any seed already present in the CSV is skipped, so this can be
# re-run after a disconnect and will pick up where it stopped.
N_SNAP   = 60
WINDOW   = 5
EPOCHS   = 40                     # HARD CAP only; early stopping decides
# EARLY STOPPING, NOT A FIXED BUDGET. A fixed cap silently truncates any seed
# whose validation is still improving, which is a way of handicapping the
# baseline. Seed 2 selected the LAST epoch (val still rising 0.8081 -> 0.8135)
# and seed 1 peaked at ep10 of 12 -- neither had settled. We now train until
# validation AUC has not improved for PATIENCE epochs, cap 40. Reportable as:
# "early stopping on validation AUC, patience 5, maximum 40 epochs", applied
# identically to every seed.
#
# ORIGINAL NOTE -- seed 0's validation AUC went 0.635 -> 0.800 (ep5) ->
# 0.791 (ep10) -> 0.744 -> 0.683 -> 0.629 -> 0.583, selecting epoch 8. It
# peaks before epoch 10 and declines monotonically after, so epochs 13-30
# only produced models that validation rejected. With validation selection
# the chosen model is unchanged, at ~40% of the cost. A TRUNCATION GUARD
# below flags any seed whose best epoch lands on the last epoch -- if that
# fires, that seed must be re-run with more epochs before it is reported.
#
# StrGNN's published protocol is a fixed 50 epochs with no model selection.
# EVIDENT+ gets validation best-epoch selection, so scoring StrGNN at its
# final (over-fitted) epoch is not a like-for-like comparison. We therefore
# give StrGNN the SAME validation-based selection. Observed val AUC peaks
# near epoch 10 and declines, so 30 epochs is ample; both the selected-epoch
# selected one with the final one in a footnote.
BATCH    = 32
LR       = 1e-4

T0 = time.time()
print('=' * 78); print(f'StrGNN STAGE 2  --  {DATASET}'); print('=' * 78)

# ============================== PHASE A : setup =============================
REPO_E = '/content/EVIDENT_repo'
if not os.path.isdir(REPO_E):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/Iyad-Assaad-Nekka/'
                    'EVIDENT-Explainable-Dynamic-Graph-anomaly-Detection.git',
                    REPO_E], check=True)
exec(compile(open(os.path.join(REPO_E, 'model.py')).read(), 'model.py', 'exec'),
     globals())
MASTER = os.path.join(BENCH, 'evident_master')
os.makedirs(MASTER, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
if not os.path.abspath(BENCH).startswith('/content/drive'):
    print('=' * 72)
    print('!! GOOGLE DRIVE IS NOT MOUNTED !!')
    print(f'   BENCH resolved to a local path: {os.path.abspath(BENCH)}')
    print('   Extraction caches and result CSVs will be written INSIDE the')
    print('   container and LOST when the runtime disconnects.')
    print('   Run this first, in its own cell, then re-run this one:')
    print('       from google.colab import drive')
    print("       drive.mount('/content/drive')")
    print('   Continuing anyway -- download the CSV before disconnecting.')
    print('=' * 72, flush=True)

REPO_S = '/content/StrGNN'
if not os.path.isdir(REPO_S):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/KnowledgeDiscovery/StrGNN.git',
                    REPO_S], check=True)
so = f'{REPO_S}/pytorch_DGCNN/lib/build/dll/libgnn.so'
if not os.path.exists(so):
    subprocess.run('cd %s/pytorch_DGCNN/lib && make clean >/dev/null 2>&1; '
                   'make -j4 >/dev/null 2>&1' % REPO_S, shell=True)
print(f'[strgnn] libgnn.so: {os.path.exists(so)}', flush=True)

import types


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _dead(*a, **k):
    raise RuntimeError('node2vec/gensim disabled: use_embedding=False')


if 'gensim' not in sys.modules:
    _stub('gensim'); _stub('gensim.models', Word2Vec=_dead)
    sys.modules['gensim'].models = sys.modules['gensim.models']
if 'node2vec' not in sys.modules:
    _stub('node2vec', Graph=_dead)

import networkx as nx
if not hasattr(nx, 'from_scipy_sparse_matrix'):
    nx.from_scipy_sparse_matrix = lambda A, **kw: nx.from_scipy_sparse_array(A, **kw)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.insert(0, f'{REPO_S}/pytorch_DGCNN')
sys.path.insert(0, f'{REPO_S}/detection')
_argv = sys.argv; sys.argv = ['main.py']
from util_functions import subgraph_extraction_labeling, GNNGraph
from util import cmd_args
import main as SM                                   # Classifier lives here
sys.argv = _argv
import torch, scipy.sparse as ssp
from sklearn.metrics import roc_auc_score
print('[strgnn] model code imported', flush=True)


# ============================== PHASE B : data ==============================
def load_bitcoin(which):
    fn = 'bitcoinotc.csv' if which == 'otc' else 'bitcoinalpha.csv'
    p = os.path.join(DATA, fn)
    if not os.path.exists(p):
        open(p, 'wb').write(gzip.decompress(urllib.request.urlopen(
            f'https://snap.stanford.edu/data/soc-sign-bitcoin{which}.csv.gz',
            timeout=300).read()))
    d = pd.read_csv(p, header=None, names=['u', 'v', 'r', 't'])
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    d['y'] = (d.r <= -5).astype(np.float32); d['r'] = d.r.astype(np.float32) / 10.0
    ids = pd.unique(pd.concat([d.u, d.v])); m = {int(x): i for i, x in enumerate(ids)}
    d['u'] = d.u.map(m).astype(np.int64); d['v'] = d.v.map(m).astype(np.int64)
    return d[['u', 'v', 't', 'y', 'r']].reset_index(drop=True), len(m)


which = 'otc' if DATASET == 'bitcoin_otc' else 'alpha'


def make_injected(rate, seed=0):
    """Literature convention: real edges all benign, anomalies are random
    node pairs spliced into the stream at the given rate."""
    fn = 'bitcoinotc.csv' if which == 'otc' else 'bitcoinalpha.csv'
    p = os.path.join(DATA, fn)
    if not os.path.exists(p):
        open(p, 'wb').write(gzip.decompress(urllib.request.urlopen(
            f'https://snap.stanford.edu/data/soc-sign-bitcoin{which}.csv.gz',
            timeout=300).read()))
    d = pd.read_csv(p, header=None, names=['u', 'v', 'r', 't'])
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    d['r'] = d.r.astype(np.float32) / 10.0
    d['y'] = 0.0
    rg = np.random.default_rng(seed)
    nodes = pd.unique(pd.concat([d.u, d.v])).astype(np.int64)
    n_inj = int(len(d) * rate)
    pos = rg.choice(len(d), n_inj, replace=False)
    inj = pd.DataFrame(dict(u=rg.choice(nodes, n_inj), v=rg.choice(nodes, n_inj),
                            t=d.t.to_numpy()[pos], y=1.0, r=0.0))
    inj = inj[inj.u != inj.v]
    d = pd.concat([d, inj], ignore_index=True)
    d = d.sort_values('t', kind='mergesort').reset_index(drop=True)
    ids = pd.unique(pd.concat([d.u, d.v])); m = {int(x): i for i, x in enumerate(ids)}
    d['u'] = d.u.map(m).astype(np.int64); d['v'] = d.v.map(m).astype(np.int64)
    return d[['u', 'v', 't', 'y', 'r']].reset_index(drop=True), len(m)


RATE = INJ_RATES[0]
df, n_nodes = make_injected(RATE, seed=0)
TAG = f'inj{int(RATE*100):02d}'
print(f'[inject] rate={RATE:.0%}  edges={len(df)}  pos={df.y.mean():.4f}',
      flush=True)
# ---- keep-mask: identical to EVIDENT's, recomputed if the pool is absent --
# EVIDENT marks an edge usable when it has at least one prior incident event.
# On a fresh account the pool .npz will not exist, so we recompute the SAME
# mask directly (cheap: no features, just the usability test), which keeps the
# chronological split -- and therefore the test set -- byte-identical.
poolf = '/nonexistent'   # injected stream has its own keep-mask
KEEPF = os.path.join(MASTER, f'keepmask_{DATASET}_{TAG}.npy')
if os.path.exists(poolf):
    keep = np.load(poolf)['keep'].astype(bool)
    print('[split] keep-mask from cached EVIDENT pool', flush=True)
elif os.path.exists(KEEPF):
    keep = np.load(KEEPF)
    print('[split] keep-mask from cache', flush=True)
else:
    print('[split] EVIDENT pool absent; recomputing the identical keep-mask...',
          flush=True)
    _pb = PoolBuilder(df, n_nodes, CFG)
    _W = CFG['W']
    keep = np.zeros(len(df), bool)
    for _j in range(len(df)):
        _a, _b = int(_pb.u[_j]), int(_pb.v[_j]); _ts = _pb.t[_j]
        _ev = sorted(set(_pb.inc[_a][-_W:] + _pb.inc[_b][-_W:]))
        _ev = [_e for _e in _ev if _pb.t[_e] < _ts and _e != _j][-_W:]
        keep[_j] = len(_ev) >= 1
        _pb.absorb(_j)
        if _j % 10000 == 0 and _j:
            print(f'   keep-mask {_j}/{len(df)}', flush=True)
    np.save(KEEPF, keep)
    print(f'[split] keep-mask cached -> {KEEPF}', flush=True)
tr_idx, va_idx, te_idx = chrono_split(keep, CFG)
y_all = df.y.to_numpy()
print(f'[data] {DATASET} edges={len(df)} nodes={n_nodes}')
print(f'[split] tr={len(tr_idx)} te={len(te_idx)} test_pos={y_all[te_idx].mean():.4f}',
      flush=True)

GCACHE = os.path.join(MASTER, f'strgnn_graphs_{DATASET}_{TAG}_s{N_SNAP}w{WINDOW}_val.pkl')
if os.path.exists(GCACHE):
    print('[cache] loading extracted subgraphs from Drive...', flush=True)
    with open(GCACHE, 'rb') as f:
        gtr, ytr, gva, yva, gte, yte, max_n_label = pickle.load(f)
    print(f'[cache] train={len(gtr)} val={len(gva)} test={len(gte)} '
          f'max_label={max_n_label}')
else:
    u, v = df.u.to_numpy(), df.v.to_numpy()
    bounds = np.linspace(0, len(df), N_SNAP + 1).astype(int)
    snap_of = np.zeros(len(df), np.int64)
    for s in range(N_SNAP):
        snap_of[bounds[s]:bounds[s + 1]] = s
    nets = []
    for s in range(N_SNAP):
        e = bounds[s + 1]
        A = ssp.csr_matrix((np.ones(e, np.float32), (u[:e], v[:e])),
                           shape=(n_nodes, n_nodes))
        A = A + A.T; A.data[:] = 1.0; A.setdiag(0); A.eliminate_zeros()
        nets.append(A.tocsr())
    print(f'[snap] {N_SNAP} sparse snapshots built', flush=True)

    def extract(idx, tag):
        out, lab, ml, sk, t = [], [], 0, 0, time.time()
        for c, j in enumerate(idx):
            n = int(snap_of[j])
            if n - WINDOW + 1 < 0:
                sk += 1; continue
            dl = []
            for g in range(n - WINDOW + 1, n + 1):
                gg, nl, nf = subgraph_extraction_labeling(
                    (int(u[j]), int(v[j])), nets[g], 1, None, None)
                ml = max(ml, max(nl))
                dl.append(GNNGraph(gg, int(y_all[j]), nl, nf))
            out.append(dl); lab.append(int(y_all[j]))
            if (c + 1) % 2000 == 0:
                el = time.time() - t
                print(f'   {tag} {c+1}/{len(idx)}  {el/60:.1f} min  '
                      f'eta {(len(idx)-c-1)*el/(c+1)/60:.0f} min', flush=True)
        print(f'   {tag}: {len(out)} kept, {sk} skipped (no full window)')
        return out, np.array(lab), ml

    print('[extract] this runs ONCE and is cached; ~35-45 min', flush=True)
    OLD = '/nonexistent'   # no pre-injection cache to reuse
    if os.path.exists(OLD):
        print('[cache] reusing train/test from the previous run; '
              'only validation needs extracting', flush=True)
        with open(OLD, 'rb') as f:
            gtr, ytr, gte, yte, m1 = pickle.load(f)
        gva, yva, m2 = extract(va_idx, 'val')
    else:
        gtr, ytr, m1 = extract(tr_idx, 'train')
        gva, yva, mv = extract(va_idx, 'val')
        gte, yte, m2 = extract(te_idx, 'test')
        m1 = max(m1, mv)
    max_n_label = max(m1, m2)
    with open(GCACHE, 'wb') as f:
        pickle.dump([gtr, ytr, gva, yva, gte, yte, max_n_label], f, protocol=4)
    print(f'[cache] saved -> {GCACHE}', flush=True)

print(f'[data] train {len(gtr)} (pos {ytr.mean():.4f}) | '
      f'test {len(gte)} (pos {yte.mean():.4f}) | max_label {max_n_label}',
      flush=True)


# ============================ PHASE C : train ===============================
cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu'
cmd_args.num_epochs = EPOCHS
cmd_args.learning_rate = LR
cmd_args.batch_size = BATCH
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = 0
cmd_args.edge_feat_dim = 0
cmd_args.conv1d_activation = 'ReLU'
cmd_args.window = WINDOW

# ---- sortpooling_k is a PERCENTILE and must be converted to a node count ---
# Verbatim from StrGNN Main.py lines 142-153: take the LAST snapshot of each
# edge, sort by node count, pick the 60th percentile, floor at 10. Skipping
# this leaves k=0.6, which makes dense_dim negative and Conv1d explodes.
if cmd_args.sortpooling_k <= 1:
    A = [i[-1] for i in gtr] + [i[-1] for i in gva] + [i[-1] for i in gte]
    num_nodes_list = sorted([g.num_nodes for g in A])
    cmd_args.sortpooling_k = num_nodes_list[
        int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
    cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
    print(f'[cfg] k used in SortPooling is: {cmd_args.sortpooling_k}')

print(f'[cfg] StrGNN published hyperparameters, feat_dim={cmd_args.feat_dim}')

CSV = os.path.join(MASTER, f'strgnn_{DATASET}_{TAG}.csv')
rows = pd.read_csv(CSV).to_dict('records') if os.path.exists(CSV) else []
if REDO:
    _before = len(rows)
    rows = [r for r in rows if not (r.get('dataset') == DATASET
                                    and int(r['seed']) in SEEDS)]
    if _before != len(rows):
        print(f'[redo] dropped {_before-len(rows)} stale row(s) for seeds {SEEDS}')
done = {int(r['seed']) for r in rows}
if done:
    print(f'[resume] seeds already done: {sorted(done)}')


@torch.no_grad()
def predict(clf, glist, bs=BATCH):
    clf.eval(); out = []
    for i in range(0, len(glist), bs):
        b = glist[i:i + bs]
        logits, _, _ = clf(b)
        out.append(logits[:, 1].detach().cpu().numpy())
    return np.concatenate(out)


for sd in SEEDS:
    if sd in done:
        print(f'[skip] seed {sd}'); continue
    print(f'\n--- StrGNN seed {sd} ---', flush=True)
    import random as _rnd
    _rnd.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)
    cmd_args.seed = sd

    clf = SM.Classifier().cuda()
    opt = torch.optim.Adam(clf.parameters(), lr=LR)
    order = np.arange(len(gtr))
    best = dict(vauc=-1, ep=-1, state=None)
    since = 0
    t1 = time.time()
    for ep in range(EPOCHS):
        clf.train(); np.random.shuffle(order); tot, nb = 0.0, 0
        for i in range(0, len(order), BATCH):
            sel = order[i:i + BATCH]
            if len(sel) < 2:
                continue
            _, loss, _ = clf([gtr[k] for k in sel])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1

        # ---- validation-based selection, same as EVIDENT+ ----------------
        vs = predict(clf, gva)
        try:
            vauc = float(roc_auc_score(yva, vs))
        except ValueError:
            vauc = float('nan')
        if np.isfinite(vauc) and vauc > best['vauc']:
            best = dict(vauc=vauc, ep=ep + 1,
                        state={k: v.detach().cpu().clone()
                               for k, v in clf.state_dict().items()})
            since = 0
        else:
            since += 1
        # test is NOT evaluated during training: it is not used for anything
        # and scoring 5327 graphs every few epochs was pure overhead.
        print(f'   ep{ep+1:>3d} loss={tot/max(nb,1):.4f} valAUC={vauc:.4f} '
              f'(best ep{best["ep"]} {best["vauc"]:.4f})  '
              f'{(time.time()-t1)/60:.1f} min  [patience {since}/{PATIENCE}]',
              flush=True)
        if since >= PATIENCE:
            print(f'   early stop at ep{ep+1}: no val improvement for '
                  f'{PATIENCE} epochs (best ep{best["ep"]})', flush=True)
            break

    m_final = metrics(yte, predict(clf, gte))          # StrGNN's own protocol
    if best['state'] is not None:
        clf.load_state_dict(best['state'])
    m = metrics(yte, predict(clf, gte))                # matched protocol

    rows.append(dict(dataset=DATASET, labels='injected', inj_rate=RATE,
                     method='StrGNN', seed=sd,
                     auc=m['auc'], ap=m['ap'], p100=m['p100'],
                     auc_final_epoch=m_final['auc'], ap_final_epoch=m_final['ap'],
                     best_epoch=best['ep'], val_auc=best['vauc'],
                     train_min=(time.time() - t1) / 60,
                     n_train=len(gtr), n_test=len(gte), epochs=EPOCHS, patience=PATIENCE))
    pd.DataFrame(rows).to_csv(CSV, index=False)
    if best['ep'] >= EPOCHS:
        print(f'   *** TRUNCATION WARNING: seed {sd} hit the {EPOCHS}-epoch '
              f'hard cap without early stopping. Raise EPOCHS. ***', flush=True)
    print(f'   FINAL seed{sd}  selected ep{best["ep"]}  AUC={m["auc"]:.4f}  '
          f'AP={m["ap"]:.4f}  P@100={m["p100"]:.4f}   '
          f'[final-epoch AUC={m_final["auc"]:.4f}]  '
          f'({(time.time()-t1)/60:.1f} min)', flush=True)

# ================================ summary ===================================
nd = pd.DataFrame(rows)

print('\n' + '=' * 80)
print(f'THE CONTROL  --  {DATASET}, StrGNN, injected {RATE:.0%} vs real labels')
print('=' * 80)
print(f"{'setting':<34}{'val AUC':>10}{'test AUC':>11}{'gap':>9}{'AP':>9}{'n':>4}")
print(f"{'StrGNN  REAL labels':<34}{0.8259:>10.4f}{0.5464:>11.4f}"
      f"{0.2795:>9.4f}{0.0903:>9.4f}{3:>4d}")
if len(nd):
    print(f"{'StrGNN  INJECTED ' + f'{RATE:.0%}':<34}{nd.val_auc.mean():>10.4f}"
          f"{nd.auc.mean():>11.4f}{nd.val_auc.mean()-nd.auc.mean():>9.4f}"
          f"{nd.ap.mean():>9.4f}{len(nd):>4d}")
print(f"\n{'EVIDENT+  REAL labels':<34}{0.9480:>10.4f}{0.8623:>11.4f}"
      f"{0.0857:>9.4f}{0.5489:>9.4f}{8:>4d}")
print(f"{'degree heuristic REAL':<34}{'--':>10}{0.4920:>11.4f}{'--':>9}{'--':>9}{1:>4d}")

if len(nd):
    a = nd.auc.mean()
    if a >= 0.80:
        print(f'  StrGNN reaches {a:.4f} on injected but {0.5464:.4f} on real')
        print('  labels. The harness is SOUND -- the method learns and')
        print('  generalises when anomalies are topologically separable, and')
        print('  collapses when they are not. The real-label number stands.')
    elif a >= 0.65:
        print(f'  StrGNN reaches {a:.4f} on injected -- better than real but')
        print('  below its published ~0.9. Partial validation of the harness;')
        print('  report both and be explicit about the shortfall.')
    else:
        print(f'  StrGNN only reaches {a:.4f} even on injected, where its own')
        print('  paper reports ~0.9. The HARNESS IS SUSPECT. Do NOT publish')
        print('  the real-label comparison until this is resolved.')

print('\n' + '=' * 80); print('RESULTS (CSV)'); print('=' * 80)
print(nd.to_csv(index=False))
print(f'DONE in {(time.time()-T0)/60:.1f} min')
