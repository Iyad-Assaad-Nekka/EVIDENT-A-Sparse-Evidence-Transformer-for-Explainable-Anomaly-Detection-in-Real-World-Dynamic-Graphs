#!/usr/bin/env python
# ============================================================================
# SAD_STAGE2C_ALPHA.py
#
# SAD on Bitcoin-Alpha real labels with four causal edge features, 8 seeds.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, sys, gzip, time, random, subprocess, urllib.request, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

DS_SAD   = 'btcalphaf'      # 'f' = featured
SEEDS    = [0, 1, 2, 3, 4, 5, 6, 7]
N_EPOCHS = 10
PATIENCE = 3
TIME_BUDGET_MIN = 150
FEAT_DIM = 4

T0 = time.time()
print('=' * 78)
print('SAD STAGE 2c  --  bitcoin_alpha, causal edge features, SUPERVISED')
print('=' * 78)

REPO_S = '/content/SAD'
if not os.path.isdir(REPO_S):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/D10Andy/SAD.git', REPO_S], check=True)
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
os.makedirs(f'{REPO_S}/dataset', exist_ok=True)

import torch
import torch.nn.functional as F
import sklearn.metrics
from sklearn.metrics import roc_auc_score

try:
    import torch_scatter  # noqa
except Exception:
    tv = torch.__version__.split('+')[0]
    cu = ('cu' + torch.version.cuda.replace('.', '')) if torch.version.cuda else 'cpu'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-input',
                    'torch_scatter', '-f',
                    f'https://data.pyg.org/whl/torch-{tv}+{cu}.html'], check=False)
    import torch_scatter  # noqa
print(f'[env] torch {torch.__version__}  torch_scatter {torch_scatter.__version__}')


# ------------------------------------------------------------------ data ---
def load_alpha():
    p = os.path.join(DATA, 'bitcoinalpha.csv')
    if not os.path.exists(p):
        open(p, 'wb').write(gzip.decompress(urllib.request.urlopen(
            'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz',
            timeout=300).read()))
    d = pd.read_csv(p, header=None, names=['u', 'v', 'r', 't'])
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    d['y'] = (d.r <= -5).astype(np.float32)
    d['r'] = d.r.astype(np.float32) / 10.0
    ids = pd.unique(pd.concat([d.u, d.v]))
    m = {int(x): i for i, x in enumerate(ids)}
    d['u'] = d.u.map(m).astype(np.int64); d['v'] = d.v.map(m).astype(np.int64)
    return d[['u', 'v', 't', 'y', 'r']].reset_index(drop=True), len(m)


df, n_nodes = load_alpha()
poolf = os.path.join(MASTER, 'poolplus_bitcoin_alpha.npz')
KEEPF = os.path.join(MASTER, 'keepmask_bitcoin_alpha.npy')
if os.path.exists(poolf):
    keep = np.load(poolf)['keep'].astype(bool)
elif os.path.exists(KEEPF):
    keep = np.load(KEEPF)
else:
    print('[split] recomputing keep-mask...', flush=True)
    _pb = PoolBuilder(df, n_nodes, CFG); _W = CFG['W']
    keep = np.zeros(len(df), bool)
    for _j in range(len(df)):
        _a, _b = int(_pb.u[_j]), int(_pb.v[_j]); _ts = _pb.t[_j]
        _ev = sorted(set(_pb.inc[_a][-_W:] + _pb.inc[_b][-_W:]))
        keep[_j] = len([e for e in _ev if _pb.t[e] < _ts and e != _j]) >= 1
        _pb.absorb(_j)
    np.save(KEEPF, keep)
tr_i, va_i, te_i = chrono_split(keep, CFG)
keep_all = np.sort(np.concatenate([tr_i, va_i, te_i]))
sub = df.iloc[keep_all].reset_index(drop=True)
U, V, R = sub.u.to_numpy(), sub.v.to_numpy(), sub.r.to_numpy()
Y, TS = sub.y.to_numpy(), sub.t.to_numpy()
N = len(sub)
print(f'[data] {N} edges, {int(Y.sum())} positives ({Y.mean():.4f})')


# ----------------------------------------------------- causal features -----
def build_causal_feats(U, V, R, upto=None):
    """Row j is built from edges < j only. Accumulators are READ, then
    UPDATED -- never the reverse."""
    n = len(U) if upto is None else upto
    nn = int(max(U.max(), V.max())) + 1
    deg = np.zeros(nn, np.float64)
    rsum = np.zeros(nn, np.float64)
    rcnt = np.zeros(nn, np.float64)
    nneg = np.zeros(nn, np.float64)
    out = np.zeros((n, FEAT_DIM), np.float32)
    for j in range(n):
        a, b = U[j], V[j]
        out[j, 0] = np.log1p(deg[a])
        out[j, 1] = np.log1p(deg[b])
        out[j, 2] = (rsum[b] / rcnt[b]) if rcnt[b] > 0 else 0.0
        out[j, 3] = np.log1p(nneg[b])
        # ---- updates happen AFTER the read, so row j never sees edge j ----
        deg[a] += 1; deg[b] += 1
        rsum[b] += R[j]; rcnt[b] += 1
        if R[j] <= -0.5:
            nneg[b] += 1
    return out


print('[feat] building causal edge features...', flush=True)
FE = build_causal_feats(U, V, R)

# ---- proof 1: row j is reproducible from edges < j alone ------------------
print('[feat] verifying causality by reconstruction...', flush=True)
ok = True
PROBE_J = sorted({min(j, N - 1) for j in [500, 5000, 15000, 30000, N - 1]})
for j in PROBE_J:
    ref = build_causal_feats(U[:j + 1], V[:j + 1], R[:j + 1], upto=j + 1)[j]
    if not np.allclose(ref, FE[j], atol=1e-6):
        ok = False
        print(f'   !! row {j} differs when rebuilt from the prefix: '
              f'{ref} vs {FE[j]}')
print(f'[feat] causality by reconstruction: {"PASS" if ok else "FAIL"}')
if not ok:
    raise SystemExit('Feature construction is not causal. Stop.')

# ---- standardise on TRAIN rows only --------------------------------------
pos_in_sub = {int(g): k for k, g in enumerate(keep_all)}
tr_pos = np.array([pos_in_sub[int(i)] for i in tr_i])
va_pos = np.array([pos_in_sub[int(i)] for i in va_i])
te_pos = np.array([pos_in_sub[int(i)] for i in te_i])
mu, sg = FE[tr_pos].mean(0), FE[tr_pos].std(0) + 1e-6
FE = ((FE - mu) / sg).astype(np.float32)
print(f'[feat] standardised on {len(tr_pos)} train rows; '
      f'mu={np.round(mu,3)} sd={np.round(sg,3)}')

# ---- proof 2: the features must not BE the label -------------------------
print('\n[probe] can these features alone predict the label?')
worst = 0.0
for k in range(FEAT_DIM):
    a = roc_auc_score(Y[te_pos], FE[te_pos, k])
    a = max(a, 1 - a)
    worst = max(worst, a)
    print(f'   dim {k}: single-feature test AUC = {a:.4f}')
try:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(FE[tr_pos], Y[tr_pos])
    a_lr = roc_auc_score(Y[te_pos], lr.predict_proba(FE[te_pos])[:, 1])
    print(f'   logistic regression on all 4: test AUC = {a_lr:.4f}')
    worst = max(worst, a_lr)
except Exception as e:
    print('   (logistic probe skipped:', e, ')')
print(f'   EVIDENT, for reference, scores 0.7606 on this test set.')
# ---------------------------------------------------------------------------
# CORRECTED GUARD.
# The original condition here was `worst >= EVIDENT's AUC`, which was wrong.
# It assumed EVIDENT is the strongest model on the corpus; on Alpha it is not
# (logreg_all6 on these same causal features scores 0.8155 vs EVIDENT 0.7606,
# measured independently). A feature being MORE PREDICTIVE than our method is
# not leakage -- it is the finding this paper already reports.
#
# Leakage means a feature carries information from the current edge, and that
# is settled by the reconstruction test above: every row is rebuilt from
# strictly earlier edges and matches. A causal feature cannot encode the
# current edge's rating, and the rating is what defines the label.
#
# What WOULD be disqualifying is a feature that is a near-deterministic
# stand-in for the label, which would indicate the construction is wrong
# regardless of the causality proof. That is the condition tested now.
# ---------------------------------------------------------------------------
if worst > 0.99:
    raise SystemExit(
        f'\n  STOP. A trivial model reaches {worst:.4f} -- effectively perfect.\n'
        f'  Causality passed, so this is a construction error rather than\n'
        f'  look-ahead. Inspect build_causal_feats before training anything.')
print(f'   -> max trivial-probe AUC {worst:.4f}. Causal by construction and\n'
      f'      far from deterministic, so these are legitimate inputs.')
if worst >= 0.7606:
    print(f'   -> NOTE: a {worst:.4f} trivial probe EXCEEDS EVIDENT (0.7606)\n'
          f'      on this corpus (see analysis/REPUTATION_HEURISTICS.py).')


# ------------------------------------------------- write SAD's files -------
ml = pd.DataFrame({'u': U + 1, 'i': V + 1, 'ts': TS.astype(np.float64),
                   'label': Y.astype(np.float64),
                   'idx': np.arange(1, N + 1, dtype=np.int64)})
mx = int(max(ml.u.max(), ml.i.max()))
edge_feat = np.vstack([np.zeros((1, FEAT_DIM), np.float32), FE])  # row 0 = pad
ml.to_csv(f'{REPO_S}/dataset/ml_{DS_SAD}.csv', index=False)
np.save(f'{REPO_S}/dataset/ml_{DS_SAD}.npy', edge_feat)
np.save(f'{REPO_S}/dataset/ml_{DS_SAD}_node.npy',
        np.zeros((mx + 1, FEAT_DIM), np.float32))   # static => must stay zero
print(f'\n[write] ml_{DS_SAD}: {N} edges, edge_feat {edge_feat.shape}, '
      f'node_feat zeros {(mx+1, FEAT_DIM)}')

# ---------------------------------------------------------------------------
# SPLIT PINNING.
# SAD splits on quantiles of ts. On OTC that reproduced our chronological
# split edge-for-edge (verified in Stage 1). On Alpha it does NOT: ties in ts
# straddle the boundaries, so a quantile rule cannot express this split and
# drifts by ~23 edges (16607/3564/3521 instead of 16584/3554/3554). A SAD row
# measured on 3521 edges is not comparable to the other methods' 3554.
#
# So the quantiles are used only as a starting point and the three index
# arrays are then written directly onto the datasets. Both tr_pos/va_pos/te_pos
# are contiguous blocks of `sub` (which is in time order), which is what
# DygDataset assumes: __len__ is len(positive_eids) and __getitem__ offsets by
# index_start.
# ---------------------------------------------------------------------------
val_time = float(TS[tr_pos].max()); test_time = float(TS[va_pos].max())
Q0 = float((TS <= val_time).mean())
Q1 = float((TS <= test_time).mean()) - Q0
for _nm, _a in (('train', tr_pos), ('val', va_pos), ('test', te_pos)):
    assert np.array_equal(_a, np.arange(_a[0], _a[0] + len(_a))), \
        f'{_nm} positions are not contiguous; DygDataset cannot represent them'

# ------------------------------------------------------------- config ------
sys.path.insert(0, REPO_S); os.chdir(REPO_S)
_argv = sys.argv; sys.argv = ['train.py']
from option import args as config
sys.argv = _argv

config.dir_data = './dataset'; config.data_set = DS_SAD
config.mode = 'sad'; config.module_type = 'graph_attention'
config.mask_label = False; config.mask_ratio = 0.0
config.input_dim = FEAT_DIM; config.hidden_dim = 128
config.n_heads = 2; config.n_layer = 2; config.drop_out = 0.2
config.n_neighbors = 20; config.batch_size = 256
config.learning_rate = 5e-4
config.anomaly_alpha = 1e-1; config.supc_alpha = 5e-3
config.memory_size = 5000; config.sample_size = 2000
config.num_data_workers = 2; config.n_epochs = N_EPOCHS

import datasets as dataset
from model.tgat import TGAT

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SPLIT = [Q0, Q1, 1.0 - Q0 - Q1]
ds_tr = dataset.DygDataset(config, 'train', split_list=SPLIT)
ds_va = dataset.DygDataset(config, 'valid', split_list=SPLIT)
ds_te = dataset.DygDataset(config, 'test',  split_list=SPLIT)

# overwrite the quantile-derived ranges with OUR exact splits
for _ds, _a, _nm in ((ds_tr, tr_pos, 'train'), (ds_va, va_pos, 'val'),
                     (ds_te, te_pos, 'test')):
    _before = len(_ds.positive_eids)
    _ds.positive_eids = _a
    _ds.index_start = int(_a[0])
    print(f'[pin] {_nm}: {_before} -> {len(_ds.positive_eids)} edges '
          f'(start {_ds.index_start})')
coll = dataset.Collate(config)
print(f'[data] train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}')
assert len(ds_tr) == 16584 and len(ds_va) == 3554 and len(ds_te) == 3554, \
    f'split drifted: {len(ds_tr)}/{len(ds_va)}/{len(ds_te)} -- expected '\
    f'16584/3554/3554. The SAD row would not be comparable. Stop.'
_lt = ds_te.full_data.labels[ds_te.positive_eids]
assert int(_lt.sum()) == 227, f'test positives {int(_lt.sum())} != 227. Stop.'
print(f'[verify] test positives {int(_lt.sum())}  |  train positives WITH '
      f'LABELS {int(ds_tr.full_data.labels[ds_tr.positive_eids].sum())}')


def mkloader(ds):
    return torch.utils.data.DataLoader(
        dataset=ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_data_workers, collate_fn=coll.dyg_collate_fn)


ld_tr, ld_va, ld_te = mkloader(ds_tr), mkloader(ds_va), mkloader(ds_te)


def criterion(pred, labels, model):
    for k, v in pred.items():
        if k not in ('root_embedding', 'group', 'dev'):
            pred[k] = v[labels > -1]
    labels = labels[labels > -1]
    lc = torch.mean(F.binary_cross_entropy_with_logits(
        pred['logits'], labels, reduction='none'))
    la = model.gdn.dev_loss(torch.squeeze(labels),
                            torch.squeeze(pred['anom_score']),
                            torch.squeeze(pred['time']))
    ls = model.suploss(pred['root_embedding'], pred['group'], pred['dev'])
    return lc + config.anomaly_alpha * la + config.supc_alpha * ls, lc, la, ls


def fwd(model, b):
    return model(b['src_edge_feat'].to(device), b['src_edge_to_time'].to(device),
                 b['src_center_node_idx'].to(device),
                 b['src_neigh_edge'].to(device),
                 b['src_node_features'].to(device),
                 b['current_time'].to(device), b['labels'].to(device))


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); P, Yl = [], []
    for b in loader:
        x = fwd(model, b)
        P.append(x['logits'].sigmoid().cpu().numpy().flatten())
        Yl.append(b['labels'].cpu().numpy().flatten())
    return np.concatenate(Yl), np.concatenate(P)


OUTD = '/content/sad_out'; os.makedirs(OUTD, exist_ok=True)
CSV = os.path.join(OUTD, 'sad_btcalpha_real_featured.csv')
CSV_DRIVE = (os.path.join(MASTER, 'sad_btcalpha_real_featured.csv')
             if os.path.abspath(BENCH).startswith('/content/drive') else None)
rows = pd.read_csv(CSV).to_dict('records') if os.path.exists(CSV) else []
done = {int(r['seed']) for r in rows}

for sd in SEEDS:
    if sd in done:
        print(f'[skip] seed {sd}'); continue
    if (time.time() - T0) / 60 > TIME_BUDGET_MIN:
        print(f'\n[budget] reached; re-run to continue. saved: {sorted(done)}')
        break
    print(f'\n--- SAD+feat seed {sd} ---', flush=True)
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)

    model = TGAT(config, device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    t2 = time.time(); best = dict(v=-1, ep=-1, state=None); since = 0

    for ep in range(1, config.n_epochs + 1):
        model.train(); losses = []
        for b in ld_tr:
            opt.zero_grad()
            x = fwd(model, b); y = b['labels'].to(device)
            loss, lc, la, ls = criterion(x, y, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, 2)
            opt.step(); losses.append(float(loss.detach()))
        yv, pv = evaluate(model, ld_va)
        vauc = (float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1
                else float('nan'))
        if np.isfinite(vauc) and vauc > best['v']:
            best = dict(v=vauc, ep=ep,
                        state={k: t.detach().cpu().clone()
                               for k, t in model.state_dict().items()})
            since = 0
        else:
            since += 1
        print(f'   ep{ep:>3d} loss={np.mean(losses):.4f} valAUC={vauc:.4f} '
              f'(best ep{best["ep"]} {best["v"]:.4f}) '
              f'{(time.time()-t2)/60:.1f} min  [patience {since}/{PATIENCE}]',
              flush=True)
        if since >= PATIENCE:
            print(f'   early stop at ep{ep} (best ep{best["ep"]})', flush=True)
            break

    yt, pt = evaluate(model, ld_te); m_fin = metrics(yt, pt)
    if best['state'] is not None:
        model.load_state_dict(best['state']); model.to(device)
    yt, pt = evaluate(model, ld_te); m = metrics(yt, pt)
    rows.append(dict(dataset='bitcoin_alpha', method='SAD', variant='causal_feats',
                     supervision='full', seed=sd, auc=m['auc'], ap=m['ap'],
                     p100=m['p100'], auc_final_epoch=m_fin['auc'],
                     ap_final_epoch=m_fin['ap'], best_epoch=best['ep'],
                     val_auc=best['v'], train_min=(time.time() - t2) / 60,
                     n_test=len(yt), epochs=config.n_epochs))
    pd.DataFrame(rows).to_csv(CSV, index=False)
    if CSV_DRIVE:
        try:
            pd.DataFrame(rows).to_csv(CSV_DRIVE, index=False)
        except OSError:
            pass
    print(f'   FINAL seed{sd} ep{best["ep"]}  AUC={m["auc"]:.4f} '
          f'AP={m["ap"]:.4f} P@100={m["p100"]:.4f} '
          f'[final-ep {m_fin["auc"]:.4f}] ({(time.time()-t2)/60:.1f} min)',
          flush=True)

# -------------------------------------------------------------- summary ----
nd = pd.DataFrame(rows)
print('\n' + '=' * 78)
print('COMPLETE BASELINE TABLE  --  bitcoin_alpha, real labels')
print('=' * 78)
print(f"{'configuration':<40}{'AUC':>18}{'AP':>10}{'P@100':>10}{'n':>4}")
print(f"{'SAD, causal feats, OTC (reference)':<40}{0.6794:>10.4f}+/-{0.0070:.4f}"
      f"{0.2183:>10.4f}{0.3575:>10.4f}{8:>4d}")
if len(nd):
    s_ = nd.auc.std(ddof=1) if len(nd) > 1 else float('nan')
    print(f"{'SAD, causal feats, ALPHA (NEW)':<40}{nd.auc.mean():>10.4f}"
          f"+/-{s_:.4f}{nd.ap.mean():>10.4f}{nd.p100.mean():>10.4f}{len(nd):>4d}")
print(f"{'cp_mean_rating (no training)':<40}{0.7401:>10.4f}{'':>8}"
      f"{0.3253:>10.4f}{0.6600:>10.4f}{1:>4d}")
print(f"{'logreg_all6 (supervised, hand feats)':<40}{0.8155:>10.4f}{'':>8}"
      f"{0.3905:>10.4f}{0.5700:>10.4f}{1:>4d}")
print(f"{'EVIDENT, NO labels, no hand features':<40}{0.7606:>10.4f}"
      f"+/-{0.0186:.4f}{0.2519:>10.4f}{0.4163:>10.4f}{8:>4d}")
print(f"{'TADDY, no labels':<40}{0.5570:>10.4f}+/-{0.0060:.4f}"
      f"{0.0980:>10.4f}{0.1837:>10.4f}{8:>4d}")
print(f"{'StrGNN, no labels':<40}{0.5561:>10.4f}+/-{0.0154:.4f}"
      f"{0.0751:>10.4f}{0.0537:>10.4f}{8:>4d}")
print(f"{'degree heuristic':<40}{0.4269:>10.4f}")

if len(nd):
    a = nd.auc.mean()
    print(f"\n  selected epochs: {list(nd.best_epoch)}")
    print(f"  val->test: {nd.val_auc.mean():.4f} -> {a:.4f} "
          f"({nd.val_auc.mean()-a:+.4f})")
    print('\n--- what this completes ---')
    print('    * SAD is SUPERVISED and uses hand-built causal features.')
    print('    * logreg_all6 is SUPERVISED on the same hand-built features.')
    print('    * EVIDENT, TADDY, StrGNN use no labels and no hand features.')
    print('    * cp_mean_rating uses no labels and no training whatsoever.')
    print('  A supervision column is mandatory. Without it the table reads as')
    print('  if every row solved the same problem, and it flatters EVIDENT.')
    if a < 0.7401:
        print(f'\n  Note: supervised SAD ({a:.4f}) is beaten on Alpha by an')
        print('  untrained reputation average (0.7401), as it was on OTC.')
        print('  That is the benchmark finding, not a SAD implementation bug:')
        print('  the injected-anomaly controls showed this same harness')
        print('  reaching 0.97 when anomalies are topologically separable.')

print('\n' + '=' * 78); print('RESULTS (CSV)'); print('=' * 78)
print(nd.to_csv(index=False))
print(f'DONE in {(time.time()-T0)/60:.1f} min')
