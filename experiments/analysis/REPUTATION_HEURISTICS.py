#!/usr/bin/env python
# ============================================================================
# REPUTATION_HEURISTICS.py
#
# Non-learned reference rules on real labels: causal running statistics of
# each node's rating history (mean received rating, logistic regression on
# six statistics).
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, gzip, time, subprocess, urllib.request, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

T0 = time.time()
print('=' * 78)
print('CAUSAL REPUTATION HEURISTICS -- both corpora, real labels')
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

URLS = {'bitcoin_otc':   ('bitcoinotc.csv',
                          'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'),
        'bitcoin_alpha': ('bitcoinalpha.csv',
                          'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz')}
# published numbers, for the comparison block
REF = {
    'bitcoin_otc':   dict(EVIDENT=(0.8623, 0.5489, 0.8650),
                          SAD=(0.6794, 0.2183, 0.3575),
                          TADDY=(0.5390, 0.0909, 0.1175),
                          StrGNN=(0.5373, 0.0874, 0.0862)),
    'bitcoin_alpha': dict(EVIDENT=(0.7606, 0.2519, 0.4163),
                          SAD=(None, None, None),
                          TADDY=(0.5570, 0.0980, 0.1837),
                          StrGNN=(0.5561, 0.0751, 0.0537)),
}


def load(name):
    fn, url = URLS[name]
    p = os.path.join(DATA, fn)
    if not os.path.exists(p):
        open(p, 'wb').write(gzip.decompress(
            urllib.request.urlopen(url, timeout=300).read()))
    d = pd.read_csv(p, header=None, names=['u', 'v', 'r', 't'])
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    d['y'] = (d.r <= -5).astype(np.float32)
    d['r'] = d.r.astype(np.float32) / 10.0
    ids = pd.unique(pd.concat([d.u, d.v]))
    m = {int(x): i for i, x in enumerate(ids)}
    d['u'] = d.u.map(m).astype(np.int64); d['v'] = d.v.map(m).astype(np.int64)
    return d[['u', 'v', 't', 'y', 'r']].reset_index(drop=True), len(m)


def causal_features(U, V, R, upto=None):
    """Read, THEN update. Row j is a function of edges < j only."""
    n = len(U) if upto is None else upto
    nn = int(max(U.max(), V.max())) + 1
    deg = np.zeros(nn); rsum_in = np.zeros(nn); rcnt_in = np.zeros(nn)
    nneg_in = np.zeros(nn); rsum_out = np.zeros(nn); rcnt_out = np.zeros(nn)
    F = np.zeros((n, 6), np.float64)
    for j in range(n):
        a, b = U[j], V[j]
        F[j, 0] = -(deg[a] + deg[b])
        F[j, 1] = float(deg[a] == 0) + float(deg[b] == 0)
        F[j, 2] = -(rsum_in[b] / rcnt_in[b]) if rcnt_in[b] > 0 else 0.0
        F[j, 3] = np.log1p(nneg_in[b])
        F[j, 4] = (nneg_in[b] / rcnt_in[b]) if rcnt_in[b] > 0 else 0.0
        F[j, 5] = -(rsum_out[a] / rcnt_out[a]) if rcnt_out[a] > 0 else 0.0
        deg[a] += 1; deg[b] += 1
        rsum_in[b] += R[j]; rcnt_in[b] += 1
        rsum_out[a] += R[j]; rcnt_out[a] += 1
        if R[j] <= -0.5:
            nneg_in[b] += 1
    return F


NAMES = ['deg_sum', 'newness', 'cp_mean_rating', 'cp_neg_count',
         'cp_neg_frac', 'src_mean_given']
allrows = []

for ds in ['bitcoin_otc', 'bitcoin_alpha']:
    print('\n' + '=' * 78); print(ds.upper()); print('=' * 78)
    df, n_nodes = load(ds)
    tag = ds.replace('bitcoin_', 'bitcoin_')
    poolf = os.path.join(MASTER, f'poolplus_{ds}.npz')
    KEEPF = os.path.join(MASTER, f'keepmask_{ds}.npy')
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
    U, V, R, Y = (sub.u.to_numpy(), sub.v.to_numpy(),
                  sub.r.to_numpy(), sub.y.to_numpy())
    pos = {int(g): k for k, g in enumerate(keep_all)}
    trp = np.array([pos[int(i)] for i in tr_i])
    tep = np.array([pos[int(i)] for i in te_i])
    print(f'[data] {len(sub)} edges | test {len(tep)} edges, '
          f'{int(Y[tep].sum())} positives')

    F = causal_features(U, V, R)

    # causality proof, same as Stage 2b
    ok = all(np.allclose(causal_features(U[:j+1], V[:j+1], R[:j+1], j+1)[j],
                         F[j], atol=1e-6)
             for j in [500, 5000, min(20000, len(sub)-1), len(sub)-1])
    print(f'[verify] causal by prefix reconstruction: {"PASS" if ok else "FAIL"}')
    if not ok:
        raise SystemExit('features are not causal; stop')

    yte = Y[tep]
    for k, nm in enumerate(NAMES):
        s = F[tep, k]
        allrows.append(dict(dataset=ds, method=nm, auc=roc_auc_score(yte, s),
                            ap=average_precision_score(yte, s),
                            p100=float(yte[np.argsort(-s)[:100]].mean())))
    mu, sg = F[trp].mean(0), F[trp].std(0) + 1e-9
    Z = (F - mu) / sg
    lr = LogisticRegression(max_iter=3000, class_weight='balanced')
    lr.fit(Z[trp], Y[trp])
    s = lr.predict_proba(Z[tep])[:, 1]
    allrows.append(dict(dataset=ds, method='logreg_all6',
                        auc=roc_auc_score(yte, s),
                        ap=average_precision_score(yte, s),
                        p100=float(yte[np.argsort(-s)[:100]].mean())))
    print('   logreg coefficients (standardised):')
    for nm, c in zip(NAMES, lr.coef_[0]):
        print(f'      {nm:<18}{c:+.3f}')

    d = pd.DataFrame([r for r in allrows if r['dataset'] == ds])
    print(f"\n{'baseline':<22}{'AUC':>9}{'AP':>9}{'P@100':>9}")
    for _, r in d.sort_values('auc', ascending=False).iterrows():
        print(f"{r['method']:<22}{r['auc']:>9.4f}{r['ap']:>9.4f}{r['p100']:>9.4f}")
    print(f"{'-'*49}")
    for nm, (a, p, pk) in REF[ds].items():
        if a is None:
            print(f"{nm + ' (pending)':<22}{'--':>9}{'--':>9}{'--':>9}")
        else:
            print(f"{nm:<22}{a:>9.4f}{p:>9.4f}{pk:>9.4f}")

    best = d.loc[d.auc.idxmax()]
    ev = REF[ds]['EVIDENT']
    print(f"\n  strongest heuristic: {best['method']}  AUC {best['auc']:.4f}")
    print(f"  EVIDENT over it:  AUC {ev[0]-best['auc']:+.4f}   "
          f"AP {ev[1]-best['ap']:+.4f}   P@100 {ev[2]-best['p100']:+.4f}")
    if best['ap'] > 0:
        print(f"  relative: AP x{ev[1]/best['ap']:.2f}, "
              f"P@100 x{ev[2]/max(best['p100'],1e-9):.2f}")
    for nm in ('SAD', 'TADDY', 'StrGNN'):
        a = REF[ds][nm][0]
        if a is not None and best['auc'] > a:
            print(f"  !! {nm} ({a:.4f}) is BEATEN by a hand-built rule "
                  f"({best['auc']:.4f}).")

out = pd.DataFrame(allrows)

print('=' * 78); print('RESULTS (CSV)'); print('=' * 78)
print(out.to_csv(index=False))
print(f'DONE in {(time.time()-T0)/60:.1f} min')
