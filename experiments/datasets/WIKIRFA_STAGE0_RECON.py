#!/usr/bin/env python
# ============================================================================
# WIKIRFA_STAGE0_RECON.py
#
# Dataset screening: wiki-RfA signed voting network.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, re, gzip, time, io, urllib.request, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

URL   = 'https://snap.stanford.edu/data/wiki-RfA.txt.gz'
RAW   = '/content/wiki-RfA.txt'
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
W_POOL = 16          # same pool window as EVIDENT's CFG, for the recurrence test

T0 = time.time()
print('=' * 78)
print('WIKI-RfA  --  STAGE 0 RECON (no training)')
print('=' * 78)

# ------------------------------------------------------------------ fetch ---
if not os.path.exists(RAW):
    print('[get] downloading wiki-RfA.txt.gz ...', flush=True)
    blob = urllib.request.urlopen(URL, timeout=300).read()
    open(RAW, 'wb').write(gzip.decompress(blob))
print(f'[get] {os.path.getsize(RAW)/1e6:.1f} MB on disk')

# ------------------------------------------------------------------ parse ---
# Records are blank-line separated blocks of "KEY:value" lines. TXT may itself
# contain colons and newlines, so we key on the prefix at the start of a line
# and treat anything else as a continuation of the current field.
MONTHS = {m: i + 1 for i, m in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July',
     'August', 'September', 'October', 'November', 'December'])}
DATE_RE = re.compile(r'^\s*(\d{1,2}):(\d{2}),\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})')


def parse_dat(s):
    m = DATE_RE.match(s or '')
    if not m:
        return None
    hh, mm, dd, mon, yy = m.groups()
    if mon not in MONTHS:
        return None
    try:
        return pd.Timestamp(int(yy), MONTHS[mon], int(dd),
                            int(hh), int(mm)).value // 10 ** 9
    except ValueError:
        return None


KEYS = ('SRC:', 'TGT:', 'VOT:', 'RES:', 'YEA:', 'DAT:', 'TXT:')
recs, cur, cur_key = [], {}, None
bad_date = bad_vote = 0
with io.open(RAW, 'r', encoding='utf-8', errors='replace') as fh:
    for line in fh:
        ls = line.rstrip('\n')
        if not ls.strip():
            if cur:
                recs.append(cur); cur, cur_key = {}, None
            continue
        hit = next((k for k in KEYS if ls.startswith(k)), None)
        if hit:
            cur_key = hit[:-1]; cur[cur_key] = ls[len(hit):]
        elif cur_key == 'TXT':
            cur['TXT'] = cur.get('TXT', '') + ' ' + ls
if cur:
    recs.append(cur)
print(f'[parse] {len(recs)} raw records')

rows = []
for r in recs:
    src, tgt = (r.get('SRC') or '').strip(), (r.get('TGT') or '').strip()
    if not src or not tgt or src == tgt:
        continue
    try:
        vot = int((r.get('VOT') or '').strip())
    except ValueError:
        bad_vote += 1; continue
    if vot not in (-1, 0, 1):
        bad_vote += 1; continue
    ts = parse_dat(r.get('DAT'))
    if ts is None:
        bad_date += 1; continue
    rows.append((src, tgt, ts, vot, (r.get('RES') or '').strip()))

df = pd.DataFrame(rows, columns=['src', 'tgt', 't', 'vot', 'res'])
df = df.sort_values('t', kind='mergesort').reset_index(drop=True)
print(f'[parse] usable votes {len(df)}   dropped: {bad_date} unparsable dates, '
      f'{bad_vote} bad vote values, '
      f'{len(recs)-len(df)-bad_date-bad_vote} self/empty')
if len(df) < 50000:
    print('  !! far fewer votes than the 198,275 the dataset card states. '
          'Check the parser before trusting anything below.')

ids = pd.unique(pd.concat([df.src, df.tgt]))
m = {x: i for i, x in enumerate(ids)}
df['u'] = df.src.map(m).astype(np.int64)
df['v'] = df.tgt.map(m).astype(np.int64)
n_nodes = len(m)
span = (pd.Timestamp(df.t.min(), unit='s'), pd.Timestamp(df.t.max(), unit='s'))
print(f'[data] {len(df)} votes, {n_nodes} users, '
      f'{span[0].date()} -> {span[1].date()}')
print(f'[data] vote mix: +1 {int((df.vot==1).sum())} '
      f'({(df.vot==1).mean():.1%})   '
      f'0 {int((df.vot==0).sum())} ({(df.vot==0).mean():.1%})   '
      f'-1 {int((df.vot==-1).sum())} ({(df.vot==-1).mean():.1%})')

# ------------------------------------------------- REQUIREMENT 1: the label --
df['y'] = (df.vot == -1).astype(np.float32)
rate = float(df.y.mean())
print('\n' + '-' * 78)
print('REQUIREMENT 1 -- is "oppose" an ANOMALY, or just the other side of a vote?')
print('-' * 78)
print(f'  oppose rate            {rate:.4f}   (Bitcoin-OTC 0.0749, Alpha 0.0405)')
if rate <= 0.12:
    lab_verdict = 'PASS'
elif rate <= 0.20:
    lab_verdict = 'MARGINAL'
else:
    lab_verdict = 'FAIL'
print(f'  verdict                {lab_verdict}  '
      f'(thresholds fixed before the run: <=0.12 pass, <=0.20 marginal)')

# a stricter fallback label, in case the plain one is too common:
# oppose votes cast against a candidate who was nevertheless PROMOTED, i.e.
# the minority position. Reported for information only.
try:
    promoted = pd.to_numeric(df.res, errors='coerce')
    strict = ((df.vot == -1) & (promoted == 1)).astype(np.float32)
    print(f'  stricter label (oppose a candidate who still won): '
          f'{float(strict.mean()):.4f}')
except Exception:
    strict = None

# ------------------------------------------ REQUIREMENT 2: recurring nodes --
print('\n' + '-' * 78)
print('REQUIREMENT 2 -- do nodes come back? (same usable-event test as Bitcoin)')
print('-' * 78)
U, V, TS = df.u.to_numpy(), df.v.to_numpy(), df.t.to_numpy()
inc = [[] for _ in range(n_nodes)]
usable = np.zeros(len(df), bool)
pool_sizes = np.zeros(len(df), np.int32)
for j in range(len(df)):
    a, b = int(U[j]), int(V[j]); ts = TS[j]
    ev = set(inc[a][-W_POOL:]) | set(inc[b][-W_POOL:])
    ev = [e for e in ev if TS[e] < ts and e != j]
    pool_sizes[j] = len(ev)
    usable[j] = len(ev) >= 1
    inc[a].append(j); inc[b].append(j)
print(f'  usable events          {int(usable.sum())} / {len(df)} '
      f'({usable.mean():.1%})   [Bitcoin-OTC 99.8%]')
print(f'  pool size, usable only median={np.median(pool_sizes[usable]):.0f} '
      f'mean={pool_sizes[usable].mean():.1f} (cap {W_POOL})')
rec_verdict = 'PASS' if usable.mean() >= 0.80 else (
    'MARGINAL' if usable.mean() >= 0.60 else 'FAIL')
print(f'  verdict                {rec_verdict}  (>=80% pass, >=60% marginal)')

# -------------------------------------- REQUIREMENT 3: anomalies over time --
print('\n' + '-' * 78)
print('REQUIREMENT 3 -- are anomalies present in all three periods?')
print('-' * 78)
idx = np.where(usable)[0]
n = len(idx); n_tr = int(TRAIN_FRAC * n); n_va = int(VAL_FRAC * n)
tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
Y = df.y.to_numpy()
for nm, s_ in (('train', tr), ('val', va), ('test', te)):
    print(f'  {nm:<6} {len(s_):>7} events   {int(Y[s_].sum()):>6} positives '
          f'({Y[s_].mean():.4f})')
split_verdict = ('PASS' if min(Y[tr].sum(), Y[va].sum(), Y[te].sum()) >= 100
                 else 'MARGINAL' if min(Y[tr].sum(), Y[va].sum(),
                                        Y[te].sum()) >= 30 else 'FAIL')
print(f'  verdict                {split_verdict}  '
      f'(>=100 positives in every period passes; test has '
      f'{int(Y[te].sum())}, Bitcoin-OTC has 419)')

# ------------------------------------------------------ is there a signal? --
print('\n' + '-' * 78)
print('IS THE SIGNAL REPUTATIONAL? (reference rules, test period, causal)')
print('-' * 78)
VOT = df.vot.to_numpy().astype(np.float64)
deg = np.zeros(n_nodes); rsum = np.zeros(n_nodes)
rcnt = np.zeros(n_nodes); rneg = np.zeros(n_nodes)
s_deg = np.zeros(len(df)); s_mean = np.zeros(len(df)); s_neg = np.zeros(len(df))
for j in range(len(df)):
    a, b = int(U[j]), int(V[j])
    s_deg[j] = -(deg[a] + deg[b])
    s_mean[j] = -(rsum[b] / rcnt[b]) if rcnt[b] > 0 else 0.0
    s_neg[j] = np.log1p(rneg[b])
    deg[a] += 1; deg[b] += 1
    rsum[b] += VOT[j]; rcnt[b] += 1
    if VOT[j] < 0:
        rneg[b] += 1
yte = Y[te]
rules = {}
for nm, sc in (('degree heuristic', s_deg),
               ('mean vote received so far (negated)', s_mean),
               ('log1p(# opposes received so far)', s_neg)):
    try:
        a_ = float(roc_auc_score(yte, sc[te]))
    except ValueError:
        a_ = float('nan')
    rules[nm] = a_
    print(f'  {nm:<38} AUC {a_:.4f}')
best_rule = max(v for v in rules.values() if np.isfinite(v))
print(f'\n  For reference on Bitcoin-OTC: mean-rating rule 0.7747, '
      f'degree 0.4920, EVIDENT 0.8622.')

# ------------------------------------------------------------- the verdict --
print('\n' + '=' * 78)
print('GO / NO-GO')
print('=' * 78)
fails = [v for v in (lab_verdict, rec_verdict, split_verdict) if v == 'FAIL']
margs = [v for v in (lab_verdict, rec_verdict, split_verdict) if v == 'MARGINAL']
print(f'  label={lab_verdict}  recurrence={rec_verdict}  split={split_verdict}')
if fails:
    print('  to in Section V-B. Using it anyway would contradict our own')
    print('  selection criteria, which is the one thing this paper cannot')
    print('  afford to do. Report it in Section V-B as a dataset we examined')
    print('  and rejected, with the number above as the reason -- that costs')
    print('  one sentence and buys credibility.')
elif best_rule >= 0.95:
    print(f'\n  NO-GO (too easy). A running counter already reaches '
          f'{best_rule:.4f}.')
    pass
elif best_rule <= 0.55:
    print(f'\n  CAUTION. No reference rule beats {best_rule:.4f}, so the')
    print('  reputational signal EVIDENT is designed to read may not be there.')
    print('  EVIDENT may land near chance. That is still reportable -- it')
    print('  would bound where the method applies -- but decide BEFORE running')
elif margs:
    print('\n  GO, WITH ONE CAVEAT. Requirements are met but not comfortably.')
    print('  Proceed to Stage 1 (adapter) and state the caveat in Section V-B.')
else:
    print('\n  GO. All three requirements met, and the signal is real but not')
    print('  trivial. Proceed to Stage 1: the data adapter, then EVIDENT and')
    print('  the three baselines on identical splits.')
print(f'\n  usable={int(usable.sum())}  positives={int(Y[usable].sum())}  '
      f'test positives={int(Y[te].sum())}')

OUT = '/content/wikirfa_stream.csv'
df.loc[usable, ['u', 'v', 't', 'y', 'vot']].to_csv(OUT, index=False)
print(f'\n[write] usable stream -> {OUT} (feeds Stage 1)')
print(f'DONE in {(time.time()-T0)/60:.1f} min')
