#!/usr/bin/env python
# ============================================================================
# CTU13_STAGE0_RECON.py
#
# Dataset screening: CTU-13 botnet capture (source-host concentration of
# malicious flows).
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, time, urllib.request, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

SCENARIO   = 43
FILES      = {42: 'capture20110810.binetflow.2format',
              43: 'capture20110811.binetflow.2format',
              51: 'capture20110818.binetflow.2format'}
BASE       = ('https://mcfp.felk.cvut.cz/publicDatasets/'
              f'CTU-Malware-Capture-Botnet-{SCENARIO}/')
MAX_FLOWS  = 400_000      # chronological head; keeps the recon to ~2 minutes
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
W_POOL     = 16

T0 = time.time()
print('=' * 78)
print(f'CTU-13 scenario {SCENARIO}  --  STAGE 0 RECON (no training)')
print('=' * 78)

# ------------------------------------------------------------------ fetch ---
fn = FILES.get(SCENARIO)
if fn is None:
    raise SystemExit(f'  No filename known for scenario {SCENARIO}. Open '
                     f'{BASE} in a browser and set FILES[{SCENARIO}].')
LOCAL = f'/content/ctu{SCENARIO}.binetflow'
if not os.path.exists(LOCAL):
    print(f'[get] downloading {fn} (a few hundred MB)...', flush=True)
    urllib.request.urlretrieve(BASE + fn, LOCAL)
print(f'[get] {os.path.getsize(LOCAL)/1e6:.0f} MB on disk')

with open(LOCAL, 'r', errors='replace') as fh:
    head = [next(fh) for _ in range(3)]
print('[head] ' + head[0].strip()[:160])
if 'SrcAddr' not in head[0] or 'Label' not in head[0]:
    print('\n'.join(head))
    raise SystemExit('  Unexpected header. Set the column names by hand.')

df = pd.read_csv(LOCAL, nrows=MAX_FLOWS, low_memory=False)
df.columns = [c.strip() for c in df.columns]
need = ['StartTime', 'SrcAddr', 'DstAddr', 'Label']
miss = [c for c in need if c not in df.columns]
if miss:
    raise SystemExit(f'  Missing columns {miss}. Header is: {list(df.columns)}')
print(f'[parse] {len(df)} flows read (head of the capture)')

df['t'] = (pd.to_datetime(df.StartTime, errors='coerce')
           .view('int64') // 10 ** 9)
df = df.dropna(subset=['t', 'SrcAddr', 'DstAddr'])
df = df[df.SrcAddr != df.DstAddr]
lab = df.Label.astype(str)
df['y'] = lab.str.contains('Botnet', case=False, na=False).astype(np.float32)
df['is_bg'] = lab.str.contains('Background', case=False, na=False)
df = df.sort_values('t', kind='mergesort').reset_index(drop=True)

ids = pd.unique(pd.concat([df.SrcAddr, df.DstAddr]))
m = {x: i for i, x in enumerate(ids)}
df['u'] = df.SrcAddr.map(m).astype(np.int64)
df['v'] = df.DstAddr.map(m).astype(np.int64)
n_nodes = len(m)
print(f'[data] {len(df)} flows, {n_nodes} hosts, '
      f'{pd.Timestamp(df.t.min(), unit="s")} -> '
      f'{pd.Timestamp(df.t.max(), unit="s")}')
print(f'[data] labels: Botnet {float(df.y.mean()):.4f}   '
      f'Background {float(df.is_bg.mean()):.4f}   '
      f'other/legitimate {float((~df.is_bg & (df.y == 0)).mean()):.4f}')

# ================= THE CHECK THAT DECIDES IT: is the label the host? ========
print('\n' + '=' * 78)
print('LEAK CHECK -- is "botnet" just the identity of one infected host?')
print('=' * 78)
U, V, TS, Y = (df.u.to_numpy(), df.v.to_numpy(), df.t.to_numpy(),
               df.y.to_numpy())
src_bot = df.groupby('u').y.sum()
n_bot_src = int((src_bot > 0).sum())
print(f'  distinct sources that ever emit a botnet flow: {n_bot_src} '
      f'of {df.u.nunique()}')
top = src_bot.sort_values(ascending=False).head(5)
print(f'  botnet flows from the top 5 sources: {[int(x) for x in top]} '
      f'(total botnet flows {int(df.y.sum())})')
if int(df.y.sum()) > 0:
    print(f'  share of all botnet flows from the single worst source: '
          f'{float(top.iloc[0])/float(df.y.sum()):.1%}')

prior_bot = np.zeros(len(df), np.float64)
acc = np.zeros(n_nodes, np.float64)
for j in range(len(df)):
    prior_bot[j] = acc[U[j]]
    acc[U[j]] += Y[j]

idx_all = np.arange(len(df))
n = len(idx_all); n_tr = int(TRAIN_FRAC * n); n_va = int(VAL_FRAC * n)
te_all = idx_all[n_tr + n_va:]
try:
    leak_auc = float(roc_auc_score(Y[te_all], prior_bot[te_all]))
except ValueError:
    leak_auc = float('nan')
print(f'\n  rule "number of prior botnet flows from this source": '
      f'AUC {leak_auc:.4f}')
leak_verdict = ('FAIL' if leak_auc >= 0.95 else
                'MARGINAL' if leak_auc >= 0.80 else 'PASS')
print(f'  verdict {leak_verdict}   '
      f'(>=0.95 reject, 0.80-0.95 only with a host-disjoint control)')

# ------------------------------------------ REQUIREMENT 1: imbalanced task --
rate = float(df.y.mean())
print('\n' + '-' * 78)
print('REQUIREMENT 1 -- imbalanced detection task?')
print('-' * 78)
print(f'  botnet rate   {rate:.4f}   (Bitcoin-OTC 0.0749, Alpha 0.0405; '
      f'wiki-RfA failed at 0.2109)')
lab_verdict = ('PASS' if 0.002 <= rate <= 0.12 else
               'MARGINAL' if rate <= 0.20 else 'FAIL')
if rate < 0.002:
    lab_verdict = 'MARGINAL'
    print('  (very rare: check there are enough positives in the test period)')
print(f'  verdict       {lab_verdict}')

# ------------------------------------------------ REQUIREMENT 2: recurrence --
print('\n' + '-' * 78)
print('REQUIREMENT 2 -- recurring nodes (same usable-event test as Bitcoin)')
print('-' * 78)
inc = [[] for _ in range(n_nodes)]
usable = np.zeros(len(df), bool); psz = np.zeros(len(df), np.int32)
for j in range(len(df)):
    a, b = int(U[j]), int(V[j]); ts = TS[j]
    ev = set(inc[a][-W_POOL:]) | set(inc[b][-W_POOL:])
    ev = [e for e in ev if TS[e] < ts and e != j]
    psz[j] = len(ev); usable[j] = len(ev) >= 1
    inc[a].append(j); inc[b].append(j)
print(f'  usable events  {int(usable.sum())} / {len(df)} ({usable.mean():.1%})'
      f'   [Bitcoin-OTC 99.8%]')
print(f'  pool size      median={np.median(psz[usable]):.0f} '
      f'mean={psz[usable].mean():.1f}')
rec_verdict = ('PASS' if usable.mean() >= 0.80 else
               'MARGINAL' if usable.mean() >= 0.60 else 'FAIL')
print(f'  verdict        {rec_verdict}')

# ------------------------------------------- REQUIREMENT 3: spread in time --
print('\n' + '-' * 78)
print('REQUIREMENT 3 -- anomalies in all three periods?')
print('-' * 78)
idx = np.where(usable)[0]
n = len(idx); n_tr = int(TRAIN_FRAC * n); n_va = int(VAL_FRAC * n)
tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
for nm, s_ in (('train', tr), ('val', va), ('test', te)):
    print(f'  {nm:<6} {len(s_):>7} flows   {int(Y[s_].sum()):>6} botnet '
          f'({Y[s_].mean():.4f})')
split_verdict = ('PASS' if min(Y[tr].sum(), Y[va].sum(), Y[te].sum()) >= 100
                 else 'MARGINAL' if min(Y[tr].sum(), Y[va].sum(),
                                        Y[te].sum()) >= 30 else 'FAIL')
print(f'  verdict        {split_verdict}  (Bitcoin-OTC test has 419)')

# ------------------------------------------------------- reference rules ----
print('\n' + '-' * 78)
print('REFERENCE RULES (test period, strictly causal, no label history)')
print('-' * 78)
deg = np.zeros(n_nodes)
s_deg = np.zeros(len(df)); s_new = np.zeros(len(df))
for j in range(len(df)):
    a, b = int(U[j]), int(V[j])
    s_deg[j] = -(deg[a] + deg[b]); s_new[j] = float(deg[b] == 0)
    deg[a] += 1; deg[b] += 1
best_rule = 0.0
rules = [('degree heuristic', s_deg), ('destination is new', s_new)]
for c, nm in (('Dur', 'flow duration'), ('TotBytes', 'total bytes'),
              ('TotPkts', 'total packets')):
    if c in df.columns:
        rules.append((nm, pd.to_numeric(df[c], errors='coerce')
                      .fillna(0).to_numpy()))
for nm, sc in rules:
    try:
        a_ = float(roc_auc_score(Y[te], np.asarray(sc)[te]))
    except ValueError:
        a_ = float('nan')
    if np.isfinite(a_):
        best_rule = max(best_rule, max(a_, 1 - a_))
    print(f'  {nm:<22} AUC {a_:.4f}')

# ------------------------------------------------------------- the verdict --
print('\n' + '=' * 78)
print('GO / NO-GO')
print('=' * 78)
vs = dict(leak=leak_verdict, label=lab_verdict, recurrence=rec_verdict,
          split=split_verdict)
print('  ' + '   '.join(f'{k}={v}' for k, v in vs.items()))
if leak_verdict == 'FAIL':
    print(f'\n  NO-GO. Knowing which host misbehaved earlier already gives')
    print(f'  {leak_auc:.4f} AUC. On this capture the anomaly is a host, not')
    print('  an event, so a history-reading detector would be scoring its own')
    print('  memory of the bot\'s address. Reporting EVIDENT here would repeat,')
    print('  in a new form, the mistake this paper exists to expose.')
    print('  Try another scenario (SCENARIO = 42, 51, ...) -- captures with')
    print('  several infected hosts and more legitimate traffic behave')
    print('  differently -- or drop CTU-13 and name it in Section V-B.')
elif 'FAIL' in vs.values():
    bad = [k for k, v in vs.items() if v == 'FAIL']
    print(f'\n  NO-GO on: {", ".join(bad)}. One sentence in Section V-B.')
elif best_rule >= 0.95:
    print(f'\n  NO-GO (too easy). A single flow attribute reaches '
          f'{best_rule:.4f}.')
else:
    print(f'\n  GO. Prior-botnet-history rule AUC: {leak_auc:.4f}.')
print(f'\n  usable={int(usable.sum())}  positives={int(Y[usable].sum())}  '
      f'test positives={int(Y[te].sum())}')

OUT = f'/content/ctu{SCENARIO}_stream.csv'
keep_cols = ['u', 'v', 't', 'y'] + [c for c in ('Dur', 'TotBytes', 'TotPkts',
                                                'Proto', 'Dport')
                                    if c in df.columns]
df.loc[usable, keep_cols].to_csv(OUT, index=False)
print(f'\n[write] usable stream -> {OUT}')
print(f'DONE in {(time.time()-T0)/60:.1f} min')
