#!/usr/bin/env python
# ============================================================================
# YELP_STAGE0_RECON.py
#
# Dataset screening: Yelp review graphs.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

# >>> point this at the metadata file you received <<<
POINTED_AT = '/content/YelpNYC/metadata'
DS_NAME    = 'YelpNYC'
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
W_POOL = 16

T0 = time.time()
print('=' * 78)
print(f'{DS_NAME}  --  STAGE 0 RECON (no training)')
print('=' * 78)
if not os.path.exists(POINTED_AT):
    raise SystemExit(
        f'\n  {POINTED_AT} not found.\n'
        f'  Upload the metadata file (or set POINTED_AT) and re-run.\n'
        f'  The datasets come by email from srayana@cs.stonybrook.edu.')

# ------------------------------------------------------------------ parse ---
raw = pd.read_csv(POINTED_AT, sep=None, engine='python', header=None,
                  comment=None, skip_blank_lines=True)
print(f'[parse] {len(raw)} rows, {raw.shape[1]} columns')
print(raw.head(3).to_string())

# auto-detect: label column holds exactly two values in {-1,1} or {0,1};
# rating holds 1..5; date parses as a date; the two id columns are the rest.
cols = list(raw.columns)
lab_c = rat_c = date_c = None
for c in cols:
    v = raw[c]
    u = pd.unique(v.dropna())
    if lab_c is None and len(u) == 2 and set(np.asarray(u, dtype=object)) in (
            {-1, 1}, {0, 1}, {'-1', '1'}, {'0', '1'}):
        lab_c = c; continue
    if rat_c is None and len(u) <= 10 and pd.api.types.is_numeric_dtype(v) \
            and float(np.nanmin(v)) >= 1 and float(np.nanmax(v)) <= 5:
        rat_c = c; continue
    if date_c is None:
        d = pd.to_datetime(v, errors='coerce')
        if d.notna().mean() > 0.95:
            date_c = c
id_c = [c for c in cols if c not in (lab_c, rat_c, date_c)]
print(f'[parse] detected: user={id_c[0] if id_c else "?"} '
      f'prod={id_c[1] if len(id_c)>1 else "?"} rating={rat_c} '
      f'label={lab_c} date={date_c}')
if lab_c is None or rat_c is None or date_c is None or len(id_c) < 2:
    raise SystemExit('  Could not detect the columns. Print raw.head(20) and '
                     'set them by hand.')

df = pd.DataFrame({
    'src': raw[id_c[0]].astype(str),
    'dst': raw[id_c[1]].astype(str),
    'r': pd.to_numeric(raw[rat_c], errors='coerce'),
    'lab': pd.to_numeric(raw[lab_c], errors='coerce'),
    'dt': pd.to_datetime(raw[date_c], errors='coerce')})
df = df.dropna().reset_index(drop=True)
# Rayana's convention: 1 = recommended (genuine), -1 = filtered (fake).
df['y'] = (df.lab < 0).astype(np.float32) if df.lab.min() < 0 \
    else (df.lab == 0).astype(np.float32)
df['t'] = df.dt.view('int64') // 10 ** 9
df = df.sort_values(['t'], kind='mergesort').reset_index(drop=True)
ids = pd.unique(pd.concat([df.src, df.dst]))
m = {x: i for i, x in enumerate(ids)}
df['u'] = df.src.map(m).astype(np.int64)
df['v'] = df.dst.map(m).astype(np.int64)
n_nodes = len(m)
print(f'[data] {len(df)} reviews, {df.src.nunique()} reviewers, '
      f'{df.dst.nunique()} products, {n_nodes} nodes')
print(f'[data] {df.dt.min().date()} -> {df.dt.max().date()}')
print(f'[data] rating mix: ' + '  '.join(
    f'{int(k)}star {v:.1%}' for k, v in df.r.value_counts(normalize=True)
    .sort_index().items()))

# --------------------------------------------- REQUIREMENT 1: positive rate --
rate = float(df.y.mean())
print('\n' + '-' * 78)
print('REQUIREMENT 1 -- is this an imbalanced detection task?')
print('-' * 78)
print(f'  filtered (positive) rate   {rate:.4f}   '
      f'(Bitcoin-OTC 0.0749, Alpha 0.0405; wiki-RfA failed at 0.2109)')
lab_verdict = 'PASS' if rate <= 0.12 else ('MARGINAL' if rate <= 0.20 else 'FAIL')
print(f'  verdict                    {lab_verdict}')

# ------------------------------------------------ TIME RESOLUTION (the risk) --
print('\n' + '-' * 78)
print('TIME RESOLUTION -- Yelp dates are day-granular. Does "before" survive?')
print('-' * 78)
vc = df.t.value_counts()
print(f'  distinct timestamps        {len(vc)} for {len(df)} events')
print(f'  events sharing their stamp {1.0 - len(vc)/len(df):.1%}')
print(f'  largest single day         {int(vc.max())} events')

# ------------------------------------------ REQUIREMENT 2 + the pool, causal --
print('\n' + '-' * 78)
print('REQUIREMENT 2 -- recurring nodes, and what the pool looks like')
print('-' * 78)
U, V, TS = df.u.to_numpy(), df.v.to_numpy(), df.t.to_numpy()
inc = [[] for _ in range(n_nodes)]
usable = np.zeros(len(df), bool)
psz = np.zeros(len(df), np.int32); psame = np.zeros(len(df), np.int32)
for j in range(len(df)):
    a, b = int(U[j]), int(V[j]); ts = TS[j]
    ev = set(inc[a][-W_POOL:]) | set(inc[b][-W_POOL:])
    ev = [e for e in ev if TS[e] <= ts and e != j]
    strict = [e for e in ev if TS[e] < ts]
    psz[j] = len(strict); psame[j] = len(ev) - len(strict)
    usable[j] = len(strict) >= 1
    inc[a].append(j); inc[b].append(j)
print(f'  usable events (strictly-earlier history >= 1): '
      f'{int(usable.sum())} / {len(df)} ({usable.mean():.1%})')
print(f'  pool size (strictly earlier) median={np.median(psz[usable]):.0f} '
      f'mean={psz[usable].mean():.1f}')
share_same = float(psame.sum()) / max(float(psame.sum() + psz.sum()), 1)
print(f'  share of candidate pool events that are SAME-DAY: {share_same:.1%}')
rec_verdict = 'PASS' if usable.mean() >= 0.80 else (
    'MARGINAL' if usable.mean() >= 0.60 else 'FAIL')
time_verdict = 'PASS' if share_same <= 0.20 else (
    'MARGINAL' if share_same <= 0.50 else 'FAIL')
print(f'  recurrence verdict         {rec_verdict}')
print(f'  time-resolution verdict    {time_verdict}')

# ------------------------------------------ REQUIREMENT 3: anomalies in time --
print('\n' + '-' * 78)
print('REQUIREMENT 3 -- anomalies present in all three periods?')
print('-' * 78)
idx = np.where(usable)[0]
n = len(idx); n_tr = int(TRAIN_FRAC * n); n_va = int(VAL_FRAC * n)
tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
Y = df.y.to_numpy()
for nm, s_ in (('train', tr), ('val', va), ('test', te)):
    print(f'  {nm:<6} {len(s_):>7} events   {int(Y[s_].sum()):>6} positives '
          f'({Y[s_].mean():.4f})')
split_verdict = ('PASS' if min(Y[tr].sum(), Y[va].sum(), Y[te].sum()) >= 100
                 else 'FAIL')
print(f'  verdict                    {split_verdict}')

# ------------------------------------------------------ is there a signal? --
print('\n' + '-' * 78)
print('REFERENCE RULES (test period, strictly causal)')
print('-' * 78)
R = df.r.to_numpy().astype(np.float64)
deg = np.zeros(n_nodes); rsum = np.zeros(n_nodes); rcnt = np.zeros(n_nodes)
s_deg = np.zeros(len(df)); s_new = np.zeros(len(df))
s_dev = np.zeros(len(df)); s_ext = np.zeros(len(df))
for j in range(len(df)):
    a, b = int(U[j]), int(V[j])
    s_deg[j] = -(deg[a] + deg[b])
    s_new[j] = float(deg[a] == 0)
    mu_b = (rsum[b] / rcnt[b]) if rcnt[b] > 0 else 3.0
    s_dev[j] = abs(R[j] - mu_b)          # how far this review is from the norm
    s_ext[j] = float(R[j] in (1.0, 5.0))  # extreme rating
    deg[a] += 1; deg[b] += 1
    rsum[b] += R[j]; rcnt[b] += 1
yte = Y[te]; best_rule = 0.0
for nm, sc in (('degree heuristic', s_deg),
               ('reviewer is new', s_new),
               ('|rating - product mean so far|', s_dev),
               ('rating is 1 or 5 stars', s_ext)):
    try:
        a_ = float(roc_auc_score(yte, sc[te]))
    except ValueError:
        a_ = float('nan')
    if np.isfinite(a_):
        best_rule = max(best_rule, max(a_, 1 - a_))
    print(f'  {nm:<34} AUC {a_:.4f}')
print(f'\n  Bitcoin-OTC for reference: mean-rating rule 0.7747, '
      f'degree 0.4920, EVIDENT 0.8622.')

# ------------------------------------------------------------- the verdict --
print('\n' + '=' * 78)
print('GO / NO-GO')
print('=' * 78)
vs = dict(label=lab_verdict, recurrence=rec_verdict, split=split_verdict,
          time=time_verdict)
print('  ' + '   '.join(f'{k}={v}' for k, v in vs.items()))
if 'FAIL' in vs.values():
    bad = [k for k, v in vs.items() if v == 'FAIL']
    print(f'\n  NO-GO on: {", ".join(bad)}.')
    print('  Add one sentence to Section V-B naming this dataset and the')
    print('  number that disqualified it. Rejections on stated criteria are')
    print('  worth more than a fourth table on a dataset that does not fit.')
elif best_rule >= 0.90:
    print(f'\n  NO-GO (too easy). A one-line counter reaches {best_rule:.4f}.')
elif best_rule <= 0.55:
    print(f'\n  CAUTION. No reference rule beats {best_rule:.4f}. The signal')
    print('  EVIDENT reads may not be present, and it could land near chance.')
else:
    print('\n  GO. Proceed to Stage 1 (adapter), then EVIDENT and the three')
    print('  baselines on identical splits and identical metric code.')
    print('   - the label is Yelp\'s production filter, a real-world proxy,')
    print('     not a per-review human verdict;')
    print(f'   - dates are day-granular, and {share_same:.0%} of candidate '
          f'pool events share the target\'s day; we drop those.')
print(f'\n  usable={int(usable.sum())}  positives={int(Y[usable].sum())}  '
      f'test positives={int(Y[te].sum())}')

OUT = f'/content/{DS_NAME.lower()}_stream.csv'
df.loc[usable, ['u', 'v', 't', 'y', 'r']].to_csv(OUT, index=False)
print(f'\n[write] usable stream -> {OUT}')
print(f'DONE in {(time.time()-T0)/60:.1f} min')
