#!/usr/bin/env python
# ============================================================================
# EVIDENT_ALPHA_ORACLE.py
#
# EVIDENT diagnostics on Bitcoin-Alpha: oracle variants of window and
# budget.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, gzip, time, subprocess, urllib.request, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

T0 = time.time()
print('=' * 78)
print('ORACLE DIAGNOSIS -- what can EVIDENT\'s evidence pool actually see?')
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

W = CFG['W']
print(f'[cfg] evidence window W = {W} incident edges per endpoint')

URLS = {'bitcoin_otc':   ('bitcoinotc.csv',
                          'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'),
        'bitcoin_alpha': ('bitcoinalpha.csv',
                          'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz')}
EV = {'bitcoin_otc': (0.8623, 0.5489, 0.8650),
      'bitcoin_alpha': (0.7606, 0.2519, 0.4163)}
HEUR = {'bitcoin_otc': (0.7747, 0.3976, 0.8100),
        'bitcoin_alpha': (0.7401, 0.3253, 0.6600)}
K_BUDGET = {'bitcoin_otc': 4.7, 'bitcoin_alpha': 4.6}   # measured distinct k


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


def sc(y, s):
    return (roc_auc_score(y, s), average_precision_score(y, s),
            float(y[np.argsort(-s)[:100]].mean()))


for ds in ['bitcoin_otc', 'bitcoin_alpha']:
    print('\n' + '=' * 78); print(ds.upper()); print('=' * 78)
    df, n_nodes = load(ds)
    poolf = os.path.join(MASTER, f'poolplus_{ds}.npz')
    KEEPF = os.path.join(MASTER, f'keepmask_{ds}.npy')
    if os.path.exists(poolf):
        keep = np.load(poolf)['keep'].astype(bool)
    elif os.path.exists(KEEPF):
        keep = np.load(KEEPF)
    else:
        print('[split] recomputing keep-mask...', flush=True)
        _pb = PoolBuilder(df, n_nodes, CFG)
        keep = np.zeros(len(df), bool)
        for _j in range(len(df)):
            _a, _b = int(_pb.u[_j]), int(_pb.v[_j]); _ts = _pb.t[_j]
            _ev = sorted(set(_pb.inc[_a][-W:] + _pb.inc[_b][-W:]))
            keep[_j] = len([e for e in _ev if _pb.t[e] < _ts and e != _j]) >= 1
            _pb.absorb(_j)
        np.save(KEEPF, keep)
    tr_i, va_i, te_i = chrono_split(keep, CFG)
    te_set = set(int(x) for x in te_i)
    U, V, R, Y, TS = (df.u.to_numpy(), df.v.to_numpy(), df.r.to_numpy(),
                      df.y.to_numpy(), df.t.to_numpy())
    k_budget = int(round(K_BUDGET[ds]))
    print(f'[data] test edges {len(te_i)}, positives {int(Y[te_i].sum())}, '
          f'budget k ~ {k_budget} distinct tokens')

    # one causal pass: maintain full history AND the W-window, read then update
    inc = [[] for _ in range(n_nodes)]          # incident edge ids, in time order
    rsum_in = np.zeros(n_nodes); rcnt_in = np.zeros(n_nodes)
    rows = []
    for j in range(len(df)):
        a, b = int(U[j]), int(V[j])
        if j in te_set:
            # O1: v's full received-rating history (unbounded)
            o1 = -(rsum_in[b] / rcnt_in[b]) if rcnt_in[b] > 0 else 0.0
            # the pool EVIDENT actually builds
            pool = sorted(set(inc[a][-W:] + inc[b][-W:]))
            # tokens that describe v specifically (ratings v received)
            vr = [R[e] for e in pool if int(V[e]) == b]
            o2 = -(float(np.mean(vr)) if vr else 0.0)
            if vr:
                worst = np.sort(np.asarray(vr))[:k_budget]   # most negative
                o3 = -float(np.mean(worst))
            else:
                o3 = 0.0
            allr = [R[e] for e in pool]
            o4 = -float(np.min(allr)) if allr else 0.0
            rows.append((j, Y[j], o1, o2, o3, o4,
                         len(pool), len(vr), int(rcnt_in[b])))
        inc[a].append(j); inc[b].append(j)
        rsum_in[b] += R[j]; rcnt_in[b] += 1

    A = pd.DataFrame(rows, columns=['j', 'y', 'O1_full', 'O2_window',
                                    'O3_budget', 'O4_poolmin', 'n_pool',
                                    'n_v_in_pool', 'n_v_full'])
    y = A.y.to_numpy()
    print(f'\n{"oracle":<34}{"AUC":>9}{"AP":>9}{"P@100":>9}')
    res = {}
    for col, lab in [('O1_full', 'O1 v full history (=heuristic)'),
                     ('O2_window', f'O2 v within pool (last {W})'),
                     ('O3_budget', f'O3 pool + budget k={k_budget}'),
                     ('O4_poolmin', 'O4 most negative token in pool')]:
        a, p, pk = sc(y, A[col].to_numpy())
        res[col] = (a, p, pk)
        print(f'{lab:<34}{a:>9.4f}{p:>9.4f}{pk:>9.4f}')
    print(f'{"-"*61}')
    print(f'{"EVIDENT (trained)":<34}{EV[ds][0]:>9.4f}{EV[ds][1]:>9.4f}'
          f'{EV[ds][2]:>9.4f}')
    print(f'{"cp_mean_rating (reported)":<34}{HEUR[ds][0]:>9.4f}'
          f'{HEUR[ds][1]:>9.4f}{HEUR[ds][2]:>9.4f}')

    print(f'\n  coverage: v has prior history for '
          f'{(A.n_v_full > 0).mean():.1%} of test edges; of that history, the '
          f'pool retains {A[A.n_v_full>0].eval("n_v_in_pool/n_v_full").mean():.1%} '
          f'on average')
    print(f'  pool size: median {A.n_pool.median():.0f} tokens, '
          f'of which {A.n_v_in_pool.median():.0f} describe v')

    d_win = res['O1_full'][1] - res['O2_window'][1]
    d_bud = res['O2_window'][1] - res['O3_budget'][1]
    print(f'\n  AP lost to the WINDOW   (O1 -> O2): {d_win:+.4f}')
    print(f'  AP lost to the BUDGET   (O2 -> O3): {d_bud:+.4f}')
    print(f'  AP EVIDENT is below O3            : '
          f'{EV[ds][1] - res["O3_budget"][1]:+.4f}')

    print('\n  --- verdict ---')
    if d_win > 0.05:
        print(f'  THE WINDOW IS THE BOTTLENECK. The pool discards reputation')
        print(f'  worth {d_win:.4f} AP before training starts. No budget or')
        print(f'  capacity setting recovers it. The fix is architectural:')
        print(f'  widen W, or carry a causal running-reputation summary AS A')
        print(f'  TOKEN (never in the target -- that would break empty-mask).')
    elif res['O3_budget'][1] - EV[ds][1] > 0.05:
        print(f'  THE SIGNAL IS REACHABLE AND EVIDENT IS NOT USING IT.')
        print(f'  An oracle under the SAME window and budget reaches AP '
              f'{res["O3_budget"][1]:.4f}')
        print(f'  vs EVIDENT {EV[ds][1]:.4f}. That gap is optimisation or')
        print(f'  capacity, so a bounded sweep is justified.')
    else:
        print(f'  EVIDENT is close to its own oracle ceiling under this')
        print(f'  window and budget. Tuning will not move it much; the')


print(f'DONE in {(time.time()-T0)/60:.1f} min')
