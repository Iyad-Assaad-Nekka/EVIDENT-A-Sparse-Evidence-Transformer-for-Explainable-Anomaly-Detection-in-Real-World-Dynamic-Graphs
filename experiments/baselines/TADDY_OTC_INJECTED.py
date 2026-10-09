#!/usr/bin/env python
# ============================================================================
# TADDY_OTC_INJECTED.py
#
# TADDY under the injected-anomaly protocol on Bitcoin-OTC.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, sys, time, gzip, types, pickle, subprocess, urllib.request, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

DATASET    = 'btc_otc'
TAG        = 'inj05'
INJ_RATE   = 0.05
INJ_SEED   = 0          # same seed as the EVIDENT / StrGNN injected pools
SNAP_SIZE  = 2000
SEEDS      = [0, 1]
MAX_EPOCH  = 20
EVAL_EVERY = 2
PATIENCE   = 5
TRAIN_PER, ANOM_PER = 0.5, 0.0

T0 = time.time()
print('=' * 78)
print(f'TADDY CONTROL  --  INJECTED {INJ_RATE:.0%}, bitcoin_otc')
print('=' * 78)

REPO_E = '/content/EVIDENT_repo'
if not os.path.isdir(REPO_E):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/Iyad-Assaad-Nekka/'
                    'EVIDENT-Explainable-Dynamic-Graph-anomaly-Detection.git',
                    REPO_E], check=True)
exec(compile(open(os.path.join(REPO_E, 'model.py')).read(), 'model.py', 'exec'),
     globals())
MASTER = os.path.join(BENCH, 'evident_master')
os.makedirs(MASTER, exist_ok=True); os.makedirs(DATA, exist_ok=True)

REPO_T = '/content/TADDY'
if not os.path.isdir(REPO_T):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/yuetan031/TADDY_pytorch.git',
                    REPO_T], check=True)
for _d in ('percent', 'eigen', 'interim', 'raw'):
    os.makedirs(f'{REPO_T}/data/{_d}', exist_ok=True)

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# ---- transformers shim: pre-4.0 module paths -> modern ones ---------------
try:
    import transformers  # noqa
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    '--no-input', 'transformers==4.44.2'], check=False)
    import transformers  # noqa
import transformers
if not hasattr(transformers, 'modeling_bert'):
    from transformers.models.bert import modeling_bert as _mb
    from transformers.models.bert import configuration_bert as _cb
    sys.modules['transformers.modeling_bert'] = _mb
    sys.modules['transformers.configuration_bert'] = _cb
    transformers.modeling_bert = _mb
    transformers.configuration_bert = _cb
    print(f'[shim] transformers {transformers.__version__}: aliased '
          'modeling_bert / configuration_bert', flush=True)


# ------------------------------------------------------------------ data ----
def _reindex(d):
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    ids = pd.unique(pd.concat([d.u, d.v]))
    m = {int(x): i for i, x in enumerate(ids)}
    d['u'] = d.u.map(m).astype(np.int64); d['v'] = d.v.map(m).astype(np.int64)
    return d[['u', 'v', 't', 'y', 'r']].reset_index(drop=True), len(m)


def load_otc_raw():
    p = os.path.join(DATA, 'bitcoinotc.csv')
    if not os.path.exists(p):
        open(p, 'wb').write(gzip.decompress(urllib.request.urlopen(
            'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz',
            timeout=300).read()))
    d = pd.read_csv(p, header=None, names=['u', 'v', 'r', 't'])
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    d['y'] = (d.r <= -5).astype(np.float32)
    d['r'] = d.r.astype(np.float32) / 10.0
    return d[['u', 'v', 't', 'y', 'r']]


def load_otc_injected(rate=INJ_RATE, seed=INJ_SEED):
    """Literature convention, identical to the EVIDENT and StrGNN injected
    pools: real edges are ALL benign; anomalies are random node pairs at the
    given rate, timestamped from randomly chosen real edges, rating 0."""
    d = load_otc_raw().copy(); d['y'] = 0.0
    rg = np.random.default_rng(seed)
    nodes = pd.unique(pd.concat([d.u, d.v])).astype(np.int64)
    n_inj = int(len(d) * rate)
    pos = rg.choice(len(d), n_inj, replace=False)
    inj = pd.DataFrame(dict(u=rg.choice(nodes, n_inj), v=rg.choice(nodes, n_inj),
                            t=d.t.to_numpy()[pos], y=1.0, r=0.0))
    inj = inj[inj.u != inj.v]
    d = pd.concat([d, inj], ignore_index=True)
    return _reindex(d)


df, n_nodes = load_otc_injected()
print(f'[inject] rate={INJ_RATE:.0%}  edges={len(df)}  pos={df.y.mean():.4f}',
      flush=True)

poolf = os.path.join(MASTER, f'poolplus_bitcoin_otc_{TAG}.npz')
KEEPF = os.path.join(MASTER, f'keepmask_bitcoin_otc_{TAG}.npy')
if os.path.exists(poolf):
    keep = np.load(poolf)['keep'].astype(bool)
    print('[split] keep-mask from the cached EVIDENT injected pool', flush=True)
elif os.path.exists(KEEPF):
    keep = np.load(KEEPF)
    print('[split] keep-mask from cache', flush=True)
else:
    print('[split] recomputing the identical keep-mask...', flush=True)
    _pb = PoolBuilder(df, n_nodes, CFG); _W = CFG['W']
    keep = np.zeros(len(df), bool)
    for _j in range(len(df)):
        _a, _b = int(_pb.u[_j]), int(_pb.v[_j]); _ts = _pb.t[_j]
        _ev = sorted(set(_pb.inc[_a][-_W:] + _pb.inc[_b][-_W:]))
        keep[_j] = len([e for e in _ev if _pb.t[e] < _ts and e != _j]) >= 1
        _pb.absorb(_j)
    np.save(KEEPF, keep)
    print(f'[split] keep-mask cached -> {KEEPF}', flush=True)
tr_idx, va_idx, te_idx = chrono_split(keep, CFG)
u, v, y = df.u.to_numpy(), df.v.to_numpy(), df.y.to_numpy()
print(f'[split] tr={len(tr_idx)} va={len(va_idx)} te={len(te_idx)} '
      f'test_pos={int(y[te_idx].sum())}')

# ---- three-group pickle: train(0) | val(injected) | test(injected) --------
from scipy import sparse

PKL = f'{REPO_T}/data/percent/{DATASET}_{TAG}3_{TRAIN_PER}_{ANOM_PER}.pkl'
groups = [('train', tr_idx, False), ('val', va_idx, True), ('test', te_idx, True)]
rows, cols, weis, labs, gsize = [], [], [], [], {}
tr_edges = np.stack([u[tr_idx], v[tr_idx]], 1)
for name, idx, real in groups:
    e = np.stack([u[idx], v[idx]], 1); lb = y[idx].astype(np.int32)
    ns = int(len(e) / SNAP_SIZE + 0.5); gsize[name] = ns
    for ii in range(ns):
        sl = slice(ii * SNAP_SIZE, (ii + 1) * SNAP_SIZE)
        r_ = np.array(e[sl, 0], dtype=np.int32)
        rows.append(r_); cols.append(np.array(e[sl, 1], dtype=np.int32))
        labs.append(np.array(lb[sl], dtype=np.int32) if real
                    else np.zeros_like(r_, dtype=np.int32))
        weis.append(np.ones_like(r_, dtype=np.int32))
tm = sparse.csr_matrix((np.ones(len(tr_edges)), (tr_edges[:, 0], tr_edges[:, 1])),
                       shape=(n_nodes, n_nodes))
headtail = (tm + tm.transpose() + sparse.eye(n_nodes)).tolil().rows
del tm
if not os.path.exists(PKL):
    with open(PKL, 'wb') as fh:
        pickle.dump((rows, cols, labs, weis, headtail, gsize['train'],
                     gsize['val'] + gsize['test'], n_nodes, len(df)),
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
print(f'[taddy] snapshots: train={gsize["train"]} val={gsize["val"]} '
      f'test={gsize["test"]}')

sys.path.insert(0, REPO_T); os.chdir(REPO_T)
from codes.DynamicDatasetLoader import DynamicDatasetLoader
from codes.DynADModel import DynADModel
from codes.BaseModel import BaseModel as _TaddyBase

# transformers >=5 added a tied-weights registry that PreTrainedModel.init_weights
# now requires. TADDY predates it and ties nothing, so the mechanism is
# inapplicable -- no-op it rather than pinning an ancient transformers.
for _cls in (_TaddyBase, DynADModel):
    _cls.all_tied_weights_keys = {}
    _cls._tied_weights_keys = []
    _cls.tie_weights = lambda self, *a, **k: None
print(f'[shim] transformers {transformers.__version__}: tie_weights disabled '
      '(TADDY ties no weights)', flush=True)
from transformers.configuration_utils import PretrainedConfig

print('[load] building/loading PPR (cached after the first run)...', flush=True)
t1 = time.time()
ld = DynamicDatasetLoader()
ld.dataset_name = f'{DATASET}_{TAG}3'
ld.k = 5; ld.window_size = 2
ld.anomaly_per = ANOM_PER; ld.train_per = TRAIN_PER
ld.load_all_tag = False; ld.compute_s = True
data = ld.load()
print(f'[load] {(time.time()-t1)/60:.1f} min', flush=True)

n_tr, n_va = gsize['train'], gsize['val']
snap_train = list(range(n_tr))
snap_val = list(range(n_tr, n_tr + n_va))
snap_test = list(range(n_tr + n_va, n_tr + n_va + gsize['test']))
data['snap_train'] = snap_train
data['snap_test'] = snap_val + snap_test
print(f'[idx] train {snap_train}  val {snap_val}  test {snap_test}')


class Args:
    pass


# absolute output path: we chdir into the TADDY repo above, and a relative
# MASTER is what destroyed a 134-minute seed-0 CSV write once already.
OUTD = '/content/taddy_out'
os.makedirs(OUTD, exist_ok=True)
CSV = os.path.join(OUTD, f'taddy_{DATASET}_{TAG}.csv')
CSV_DRIVE = (os.path.join(MASTER, f'taddy_{DATASET}_{TAG}.csv')
             if os.path.abspath(BENCH).startswith('/content/drive') else None)
out_rows = pd.read_csv(CSV).to_dict('records') if os.path.exists(CSV) else []
done = {int(r['seed']) for r in out_rows}


def auc_of(model, emb, snaps):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for s in snaps:
            o = torch.sigmoid(model.forward(emb['int'][s], emb['hop'][s],
                                            emb['time'][s], None))
            ps.append(o.squeeze().cpu().numpy())
            ys.append(data['y'][s].cpu().numpy())
    yy, pp = np.concatenate(ys), np.concatenate(ps)
    return (float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1
            else float('nan')), yy, pp


for sd in SEEDS:
    if sd in done:
        print(f'[skip] seed {sd}'); continue
    print(f'\n--- TADDY injected seed {sd} ---', flush=True)
    np.random.seed(sd); torch.manual_seed(sd)

    args = Args()
    args.dataset = DATASET; args.neighbor_num = 5; args.window_size = 2
    args.embedding_dim = 32; args.num_hidden_layers = 2
    args.num_attention_heads = 2; args.max_epoch = MAX_EPOCH
    args.lr = 1e-3; args.weight_decay = 5e-4; args.seed = sd
    args.print_feq = EVAL_EVERY; args.anomaly_per = ANOM_PER
    args.train_per = TRAIN_PER

    cfg = PretrainedConfig()
    cfg.k = args.neighbor_num; cfg.window_size = args.window_size
    cfg.hidden_size = args.embedding_dim
    cfg.num_hidden_layers = args.num_hidden_layers
    cfg.num_attention_heads = args.num_attention_heads
    cfg.max_hop_dis_index = 100; cfg.max_inti_pos_index = 100
    cfg.intermediate_size = args.embedding_dim * 2
    cfg.hidden_act = 'gelu'; cfg.hidden_dropout_prob = 0.5
    cfg.attention_probs_dropout_prob = 0.3
    cfg.initializer_range = 0.02; cfg.layer_norm_eps = 1e-12
    cfg.is_decoder = False; cfg.batch_size = 256
    cfg.weight_decay = args.weight_decay

    model = DynADModel(cfg, args)
    model.data = data
    model.max_epoch = MAX_EPOCH; model.lr = args.lr
    model.weight_decay = args.weight_decay
    model.spy_tag = True

    t2 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=model.lr,
                           weight_decay=model.weight_decay)
    raw, wl, hop, inte, tme = model.generate_embedding(data['edges'])
    data['raw_embeddings'] = None
    emb = dict(int=inte, hop=hop, time=tme, wl=wl)

    best = dict(v=-1, ep=-1, state=None); since = 0
    for ep in range(MAX_EPOCH):
        negs = model.negative_sampling(data['edges'][:max(snap_train) + 1])
        _, wln, hopn, inten, tmen = model.generate_embedding(negs)
        model.train(); tot = 0.0
        for s in snap_train:
            if wl[s] is None:
                continue
            ip, hp, tp = inte[s], hop[s], tme[s]
            yp = data['y'][s].float()
            ineg, hneg, tneg = inten[s], hopn[s], tmen[s]
            yneg = torch.ones(ineg.size()[0])
            out = model.forward(torch.vstack((ip, ineg)),
                                torch.vstack((hp, hneg)),
                                torch.vstack((tp, tneg))).squeeze()
            loss = F.binary_cross_entropy_with_logits(
                out, torch.hstack((yp, yneg)))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        if (ep + 1) % EVAL_EVERY == 0 or ep == 0:
            vauc, _, _ = auc_of(model, emb, snap_val)
            if np.isfinite(vauc) and vauc > best['v']:
                best = dict(v=vauc, ep=ep + 1,
                            state={k: t.detach().cpu().clone()
                                   for k, t in model.state_dict().items()})
                since = 0
            else:
                since += 1
            print(f'   ep{ep+1:>4d} loss={tot/max(len(snap_train),1):.4f} '
                  f'valAUC={vauc:.4f} (best ep{best["ep"]} {best["v"]:.4f}) '
                  f'{(time.time()-t2)/60:.1f} min  [patience {since}/{PATIENCE}]',
                  flush=True)
            if since >= PATIENCE:
                print(f'   early stop at ep{ep+1} (best ep{best["ep"]})',
                      flush=True)
                break

    _, y_fin, p_fin = auc_of(model, emb, snap_test)
    m_fin = metrics(y_fin, p_fin)
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    _, y_sel, p_sel = auc_of(model, emb, snap_test)
    m = metrics(y_sel, p_sel)

    out_rows.append(dict(dataset='bitcoin_otc', labels='injected',
                         inj_rate=INJ_RATE, method='TADDY', seed=sd,
                         auc=m['auc'], ap=m['ap'], p100=m['p100'],
                         auc_final_epoch=m_fin['auc'], ap_final_epoch=m_fin['ap'],
                         best_epoch=best['ep'], val_auc=best['v'],
                         train_min=(time.time() - t2) / 60,
                         n_test=len(y_sel), epochs=MAX_EPOCH))
    pd.DataFrame(out_rows).to_csv(CSV, index=False)
    if CSV_DRIVE:
        try:
            pd.DataFrame(out_rows).to_csv(CSV_DRIVE, index=False)
        except OSError:
            pass
    print(f'   FINAL seed{sd} selected ep{best["ep"]}  AUC={m["auc"]:.4f}  '
          f'AP={m["ap"]:.4f}  P@100={m["p100"]:.4f}  '
          f'[final-epoch {m_fin["auc"]:.4f}]  '
          f'({(time.time()-t2)/60:.1f} min)', flush=True)

# -------------------------------------------------------------- verdict ----
nd = pd.DataFrame(out_rows)
print('\n' + '=' * 78)
print('THE CONTROL  --  bitcoin_otc, TADDY, injected 5% vs real labels')
print('=' * 78)
print(f"{'setting':<30}{'val AUC':>10}{'test AUC':>11}{'gap':>9}{'AP':>9}{'n':>4}")
print(f"{'TADDY  REAL labels':<30}{0.5980:>10.4f}{0.5390:>11.4f}"
      f"{0.0590:>9.4f}{0.0909:>9.4f}{8:>4d}")
if len(nd):
    print(f"{'TADDY  INJECTED 5%':<30}{nd.val_auc.mean():>10.4f}"
          f"{nd.auc.mean():>11.4f}{nd.val_auc.mean()-nd.auc.mean():>9.4f}"
          f"{nd.ap.mean():>9.4f}{len(nd):>4d}")
print(f"{'StrGNN INJECTED 5%':<30}{0.9792:>10.4f}{0.9819:>11.4f}"
      f"{-0.0028:>9.4f}{0.7973:>9.4f}{2:>4d}")
print(f"{'EVIDENT INJECTED 5%':<30}{'':>10}{0.9614:>11.4f}{'':>9}{'':>9}{4:>4d}")
print(f"{'degree heuristic INJ 5%':<30}{'':>10}{0.8841:>11.4f}")

if len(nd):
    a = nd.auc.mean()
    print('\n--- verdict, thresholds fixed before the run ---')
    if a >= 0.80:
        print(f'  PASS. TADDY reaches {a:.4f} on injected and {0.5390:.4f} on')
        print('  real labels with the same harness, same corpus, same splits.')
        print('  The harness is SOUND and the real-label TADDY row stands.')
        print('  The 2x2 is complete: both SOTA baselines collapse on real')
        print('  labels while beating or matching EVIDENT on injections.')
    elif a >= 0.65:
        print(f'  PARTIAL. {a:.4f} injected. Report both numbers and state')
        print('  plainly that the TADDY harness is not fully validated.')
    else:
        print(f'  FAIL. {a:.4f} injected -- the harness does not recover even')
        print('  on a topologically separable task. DO NOT publish the')
        print('  real-label TADDY row. Debug the pickle/PPR path or drop TADDY.')

print('\n' + '=' * 78); print('RESULTS (CSV)'); print('=' * 78)
print(nd.to_csv(index=False))
print(f'DONE in {(time.time()-T0)/60:.1f} min')
