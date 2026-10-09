#!/usr/bin/env python
# ============================================================================
# CICIDS_PROBE.py
#
# Dataset screening: CICIDS2017 (temporal distribution of attacks under a
# chronological split).
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, sys, time, zipfile, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

ZIP_CANDIDATES = [
    '/content/drive/MyDrive/dgad_bench/data/GeneratedLabelledFlows.zip',
    '/content/GeneratedLabelledFlows.zip',
]
EXTRACT_TO = '/content/cicids'
NAT_IP = '172.16.0.1'

T0 = time.time()
print('=' * 78); print('CICIDS2017 PROBE'); print('=' * 78)

zp = next((z for z in ZIP_CANDIDATES if os.path.exists(z)), None)
if zp is None:
    print('[stop] GeneratedLabelledFlows.zip not found. Looked in:')
    for z in ZIP_CANDIDATES:
        print('   ', z)
    sys.exit(1)
print(f'[zip] {zp}  ({os.path.getsize(zp)/1e6:.0f} MB)')

if not os.path.isdir(EXTRACT_TO):
    os.makedirs(EXTRACT_TO, exist_ok=True)
    print('[zip] extracting...', flush=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(EXTRACT_TO)
csvs = sorted(p for p in
              (os.path.join(r, f) for r, _, fs in os.walk(EXTRACT_TO) for f in fs)
              if p.lower().endswith('.csv'))
print(f'[zip] {len(csvs)} csv files:')
for c in csvs:
    print(f'   {os.path.basename(c)}  ({os.path.getsize(c)/1e6:.0f} MB)')

# ------------------------------------------------------------- 1. columns --
probe = pd.read_csv(csvs[0], nrows=5, encoding='latin-1', low_memory=False)
# CICIDS column names carry LEADING SPACES (' Source IP', ' Label'). Match on
# the normalised name but keep the ORIGINAL string for usecols, or pandas
# rejects it.
raw_cols = list(probe.columns)
print(f'\n[cols] {len(raw_cols)} columns. First 12: '
      f'{[c.strip() for c in raw_cols[:12]]}')


def find(*cands):
    for c in cands:
        want = c.lower().replace(' ', '')
        for k in raw_cols:
            if k.strip().lower().replace(' ', '') == want:
                return k          # ORIGINAL name, spaces intact
    return None


SRC = find('Source IP', 'Src IP', 'SourceIP')
DST = find('Destination IP', 'Dst IP', 'DestinationIP')
TS = find('Timestamp')
LAB = find('Label')
print(f'[cols] source={SRC!r} dest={DST!r} time={TS!r} label={LAB!r}')
print('[cols] (leading spaces above are in the file, not a typo)')
if not all([SRC, DST, TS, LAB]):
    print('\n[STOP] IP columns missing -- this is the MachineLearningCSV build,')
    print('       which strips them. There is no graph to construct. Download')
    print('       GeneratedLabelledFlows.zip instead.')
    sys.exit(1)
print('[cols] OK -- a graph can be built.')

# --------------------------------------------------- 2..5 per-day analysis --
use = [SRC, DST, TS, LAB]
rows = []
print('\n' + '=' * 78)
print('PER-DAY DIAGNOSTICS')
print('=' * 78)
print(f"{'file':<34}{'flows':>9}{'attack%':>9}{'nodes':>8}"
      f"{'recur%':>8}{'NATatk%':>9}{'degAUC':>8}")

for c in csvs:
    try:
        hdr = list(pd.read_csv(c, nrows=0, encoding='latin-1').columns)

        def pick(*cands):
            for cc in cands:
                w = cc.lower().replace(' ', '')
                for k in hdr:
                    if k.strip().lower().replace(' ', '') == w:
                        return k
            return None

        s_, d_, t_, l_ = (pick('Source IP', 'Src IP'),
                          pick('Destination IP', 'Dst IP'),
                          pick('Timestamp'), pick('Label'))
        if not all([s_, d_, t_, l_]):
            print(f'  {os.path.basename(c)[:32]:<34} SKIP: missing columns')
            continue
        d = pd.read_csv(c, usecols=[s_, d_, t_, l_], encoding='latin-1',
                        low_memory=False)
        d = d[[s_, d_, t_, l_]]
    except Exception as e:
        print(f'  {os.path.basename(c)[:32]:<34} READ FAILED: {e}')
        continue
    d.columns = ['u', 'v', 't', 'y']
    d = d.dropna(subset=['u', 'v', 't', 'y'])
    d['y'] = (~d.y.astype(str).str.strip().str.upper().eq('BENIGN')).astype(int)
    d['t'] = pd.to_datetime(d.t, errors='coerce', dayfirst=True)
    d = d.dropna(subset=['t']).sort_values('t', kind='mergesort').reset_index(drop=True)
    if len(d) == 0:
        continue

    atk = d.y.mean()
    nodes = pd.unique(pd.concat([d.u, d.v]))
    # 3. node recurrence: fraction of flows whose SOURCE was seen earlier
    seen, recur = set(), 0
    for s in d.u.to_numpy():
        if s in seen:
            recur += 1
        seen.add(s)
    recur /= len(d)
    # 4. how much attack traffic hangs off the single NAT node
    natatk = (d[d.y == 1].u.eq(NAT_IP) | d[d.y == 1].v.eq(NAT_IP)).mean() \
        if d.y.sum() else 0.0
    # 5. degree heuristic, streaming, same definition used on Bitcoin
    deg = {}
    sc = np.empty(len(d))
    for i, (a, b) in enumerate(zip(d.u.to_numpy(), d.v.to_numpy())):
        sc[i] = -(deg.get(a, 0) + deg.get(b, 0))
        deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    try:
        from sklearn.metrics import roc_auc_score
        dauc = roc_auc_score(d.y, sc) if 0 < d.y.mean() < 1 else float('nan')
    except Exception:
        dauc = float('nan')

    rows.append(dict(file=os.path.basename(c), flows=len(d), attack=atk,
                     nodes=len(nodes), recur=recur, nat_atk=natatk,
                     deg_auc=dauc))
    print(f"{os.path.basename(c)[:32]:<34}{len(d):>9d}{100*atk:>8.1f}%"
          f"{len(nodes):>8d}{100*recur:>7.1f}%{100*natatk:>8.1f}%{dauc:>8.3f}",
          flush=True)

R = pd.DataFrame(rows)

print('\n' + '=' * 78)
print('VERDICT  (criteria fixed before looking at the numbers)')
print('=' * 78)
print('  a file QUALIFIES if:  2% <= attack rate <= 25%   (rare-event regime,')
print('                        comparable to Bitcoin 4-8%)')
print('                        node recurrence >= 50%     (EVIDENT needs history)')
print('                        degree AUC <= 0.75         (not already trivial)')
print()
if len(R):
    R['qualifies'] = ((R.attack.between(0.02, 0.25)) & (R.recur >= 0.50)
                      & (R.deg_auc.fillna(1) <= 0.75))
    for _, r in R.iterrows():
        why = []
        if not 0.02 <= r.attack <= 0.25:
            why.append(f'attack {100*r.attack:.1f}% out of range')
        if r.recur < 0.50:
            why.append(f'recurrence {100*r.recur:.0f}% too low')
        if not (r.deg_auc <= 0.75):
            why.append(f'degree AUC {r.deg_auc:.3f} -- topologically trivial')
        print(f"  {'PASS' if r.qualifies else 'FAIL'}  {r.file[:40]:<42}"
              + ('' if r.qualifies else '  <- ' + '; '.join(why)))
    ok = R[R.qualifies]
    print()
    if len(ok):
        best = ok.sort_values('flows', ascending=False).iloc[0]
        print(f'  -> USE: {best.file}')
        print(f'     {int(best.flows)} flows, {100*best.attack:.1f}% attacks, '
              f'{int(best.nodes)} nodes, {100*best.recur:.0f}% recurrence')
        print(f'     NAT node carries {100*best.nat_atk:.1f}% of attack traffic')
        if best.nat_atk > 0.5:
            print('     *** the single NAT node carries most attacks. Disclose it,')
            print('         and consider excluding 172.16.0.1 from the graph.')
    else:
        print('  -> NO file qualifies. CICIDS2017 is not a fit; do not build')
        print('     pools. Report it alongside Wikipedia/MOOC/Elliptic as a')
        print('     documented exclusion.')

print('\n' + '=' * 78); print('COPY THIS BLOCK'); print('=' * 78)
print(R.round(4).to_csv(index=False))
print(f'DONE in {(time.time()-T0)/60:.1f} min')
