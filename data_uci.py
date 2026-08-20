# =====================================================================
# EVIDENT on UCI-Messages (injected) -- CELL 1 : ARTEFACT + POOLS
# Run AFTER cell 0.
#
# ARTEFACT PROVENANCE MATTERS. Table III's ten rows were measured on ONE
# realisation of the injection protocol. A fresh rebuild with a different
# seed is a DIFFERENT benchmark, and EVIDENT's row would not be
# comparable to the rows beside it. Sources are tried in this order:
#   [A] local Drive copy          -> exact, preferred
#   [B] raw GitHub, your repo     -> exact, if committed
#   [C] rebuild from CollegeMsg   -> LAST RESORT, needs a caption note
# The source actually used is printed in capitals. Read it.
# =====================================================================
import os, io, gzip, urllib.request
import numpy as np, pandas as pd

ARTEF_NAME = 'ml_uci_testinj.csv'
ARTEF      = os.path.join(BENCH, ARTEF_NAME)

REPO = ('https://raw.githubusercontent.com/inekka-esi/'
        'Deep-Learning-for-Anomaly-Detection-in-Dynamic-Graphs-'
        'A-Verified-Survey-Taxonomy-and-benchmarking/main/')
CANDIDATE_PATHS = [ARTEF_NAME, f'data/{ARTEF_NAME}',
                   f'artefacts/{ARTEF_NAME}', f'tools/{ARTEF_NAME}']

COLLEGEMSG = 'https://snap.stanford.edu/data/CollegeMsg.txt.gz'
INJ_RATE   = 0.05      # documented protocol parameter
INJ_SEED   = 0         # documented protocol parameter

SOURCE = None

# ---- [A] local -------------------------------------------------------
if os.path.exists(ARTEF):
    SOURCE = 'A: LOCAL DRIVE COPY (exact)'

# ---- [B] github ------------------------------------------------------
if SOURCE is None:
    for rel in CANDIDATE_PATHS:
        try:
            raw = urllib.request.urlopen(REPO + rel, timeout=60).read()
            if len(raw) < 1000:
                continue
            open(ARTEF, 'wb').write(raw)
            SOURCE = f'B: GITHUB {rel} (exact)'
            break
        except Exception:
            continue

# ---- [C] rebuild -----------------------------------------------------
if SOURCE is None:
    print('[warn] artefact not found locally or on GitHub -- REBUILDING.')
    cm = os.path.join(DATA, 'CollegeMsg.txt')
    if not os.path.exists(cm):
        txt = gzip.decompress(
            urllib.request.urlopen(COLLEGEMSG, timeout=180).read()).decode()
        open(cm, 'w').write(txt)
        print(f'[ok  ] fetched CollegeMsg ({os.path.getsize(cm)/1e6:.2f} MB)')
    d = pd.read_csv(cm, sep=r'\s+', header=None, names=['u', 'v', 't'])
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    d['y'] = 0
    print(f'[uci] real stream: {len(d)} events, '
          f'{len(pd.unique(pd.concat([d.u, d.v])))} nodes')

    # Injection protocol: uniform endpoint resampling at a fixed rate,
    # timestamps inherited from a uniformly chosen host event, unseen
    # pairs only. This is the standard edge-level injection used by the
    # DTDG literature and reproduced in the unified benchmark.
    rng = np.random.default_rng(INJ_SEED)
    nodes = np.sort(pd.unique(pd.concat([d.u, d.v]))).astype(np.int64)
    seen = set(map(tuple, np.sort(d[['u', 'v']].to_numpy(), axis=1)))
    n_inj = int(round(INJ_RATE * len(d) / (1 - INJ_RATE)))
    iu, iv, it = [], [], []
    host = rng.integers(0, len(d), size=n_inj * 3)
    ta = d.t.to_numpy()
    tries = 0
    while len(iu) < n_inj and tries < n_inj * 50:
        a, b = rng.choice(nodes, 2, replace=False)
        key = (min(a, b), max(a, b))
        tries += 1
        if key in seen:
            continue
        iu.append(int(a)); iv.append(int(b))
        it.append(float(ta[host[len(iu) % len(host)]]))
    inj = pd.DataFrame(dict(u=iu, v=iv, t=it, y=1))
    d = pd.concat([d, inj], ignore_index=True)
    d = d.sort_values('t', kind='mergesort').reset_index(drop=True)
    d.to_csv(ARTEF, index=False)
    SOURCE = (f'C: REBUILT from CollegeMsg '
              f'(rate={INJ_RATE}, seed={INJ_SEED}) -- NOT the original '
              f'realisation, caption must say so')

print('=' * 70)
print(f'[ARTEFACT SOURCE] {SOURCE}')
print('=' * 70)


def load_uci(path):
    d = pd.read_csv(path)
    print('[schema]', list(d.columns))

    def pick(cands, what):
        for c in cands:
            if c in d.columns:
                return c
        raise KeyError(f'no {what} column in {list(d.columns)}')

    cu = pick(['u', 'src', 'source', 'from', 'i_src'], 'source')
    cv = pick(['v', 'i', 'dst', 'target', 'to', 'i_dst'], 'target')
    ct = pick(['t', 'ts', 'time', 'timestamp'], 'time')
    cy = pick(['y', 'label', 'is_anomaly', 'anomaly'], 'label')
    print(f'[cols] u={cu} v={cv} t={ct} y={cy}')

    o = d[[cu, cv, ct, cy]].copy()
    o.columns = ['u', 'v', 't', 'y']
    for c in o.columns:
        o[c] = pd.to_numeric(o[c], errors='coerce')
    o = o.dropna()
    o = o[o.u != o.v].copy()
    o['y'] = o.y.astype(np.int64)
    o = o.sort_values('t', kind='mergesort').reset_index(drop=True)

    nd = np.sort(pd.unique(pd.concat([o.u, o.v])))
    rm = {n: i for i, n in enumerate(nd)}
    o['u'] = o.u.map(rm).astype(np.int64)
    o['v'] = o.v.map(rm).astype(np.int64)

    print(f'[uci] events={len(o)} nodes={len(rm)} '
          f'anomalies={int(o.y.sum())} ({o.y.mean()*100:.2f}%)')
    assert o.y.sum() >= 30, 'too few anomalies'
    return o[['u', 'v', 't', 'y']].reset_index(drop=True), len(rm)


df_uci, n_uci = load_uci(ARTEF)

CFG_UCI = dict(CFG)
CFG_UCI['TRAIN_FRAC'] = 0.70
CFG_UCI['VAL_FRAC']   = 0.15

POOL_UCI = build_all_fixed(df_uci, n_uci, CFG_UCI)
TRu, VAu, TEu = chrono_split(POOL_UCI['keep'], CFG_UCI)
print(f'[split] train={len(TRu)} val={len(VAu)} test={len(TEu)}')
for nm, s in [('train', TRu), ('val', VAu), ('test', TEu)]:
    print(f'  {nm}: pos_rate={POOL_UCI["Y"][s].mean():.4f} '
          f'({int(POOL_UCI["Y"][s].sum())} positives)')

# ---------------------------------------------------------------------
# CARDINALITY AUDIT -- this is the paper's structural claim, so verify
# it rather than assert it. Deg-Sum reaches AUC 0.9861 on this benchmark
# through pool size alone; fixed resampling removes that channel.
# ---------------------------------------------------------------------
occ = POOL_UCI['TM'][TEu].sum(1)
yte = POOL_UCI['Y'][TEu]
print('\n[audit] token slots per row: '
      f'min={occ.min():.0f} max={occ.max():.0f} mean={occ.mean():.2f}')
print(f'[audit] normal mean={occ[yte==0].mean():.4f}  '
      f'anomalous mean={occ[yte==1].mean():.4f}')
assert occ.min() == occ.max(), 'cardinality NOT constant -- leak channel open'
print('[audit] PASS: cardinality is constant, carries zero label information')

np.savez_compressed(os.path.join(OUT, 'pool_uci.npz'),
                    TR=TRu, VA=VAu, TE=TEu, **POOL_UCI)
with open(os.path.join(OUT, 'artefact_source.txt'), 'w') as f:
    f.write(SOURCE + '\n')
print(f'\n[done] saved {os.path.join(OUT, "pool_uci.npz")}')
print('Next: CELL 2 (5-seed run).')
