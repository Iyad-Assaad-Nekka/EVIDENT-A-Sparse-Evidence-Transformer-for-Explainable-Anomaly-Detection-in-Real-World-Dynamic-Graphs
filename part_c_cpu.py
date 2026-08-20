# =====================================================================
# EVIDENT on UCI-Messages -- CELL 3 : PART C, CPU, NO TRAINING
#
# Self-bootstrapping. If the probe ladder is already cached this needs
# NOTHING else -- no cell 0, no cell 1, no torch, no pools.
# It only asks for cells 0 and 1 if it must recompute from checkpoints,
# and even then it does inference only: forward passes over the test
# split under no_grad. No backward pass, no optimiser, no training.
# =====================================================================
import os, json, math
import numpy as np

# ---- paths, independent of cell 0 -----------------------------------
if 'OUT' not in globals():
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        _ROOT = '/content/drive/MyDrive'
    except Exception:
        _ROOT = os.environ.get('EVIDENT_ROOT', '.')
    OUT = os.path.join(_ROOT, 'dgad_bench', 'evident_uci')
print(f'[paths] OUT = {OUT}')

RES = os.path.join(OUT, 'evident_uci_results.json')
if not os.path.exists(RES):
    raise SystemExit(
        f'No cached results at {RES}\n'
        'Nothing was persisted, so Part C cannot be produced. '
        'Check that OUT points at the same Drive as the GPU run.')

state = json.load(open(RES))
SEEDS = [0, 1, 2, 3, 4]

def save():
    json.dump(state, open(RES, 'w'), indent=2, default=float)

print(f'[cache] blocks present: {sorted(state.keys())}')

# ---------------------------------------------------------------------
# Detection numbers -- cache only, zero compute
# ---------------------------------------------------------------------
def col(block, key):
    return np.array([state[block][str(s)][key] for s in SEEDS
                     if str(s) in state.get(block, {})])

bA, bP, bK = col('baseline','auc'), col('baseline','ap'), col('baseline','p100')
eA, eP, eK = col('evident','auc'),  col('evident','ap'),  col('evident','p100')
print(f'[cache] baseline seeds={len(bA)}  evident seeds={len(eA)}')
print(f'  BASELINE AUC={bA.mean():.4f}+/-{bA.std():.4f} '
      f'AP={bP.mean():.4f}+/-{bP.std():.4f} P@100={bK.mean():.4f}+/-{bK.std():.4f}')
print(f'  EVIDENT  AUC={eA.mean():.4f}+/-{eA.std():.4f} '
      f'AP={eP.mean():.4f}+/-{eP.std():.4f} P@100={eK.mean():.4f}+/-{eK.std():.4f}')

# ---- duplicate-seed audit -------------------------------------------
for nm, arr in [('evident AP', eP), ('evident AUC', eA)]:
    u, c = np.unique(np.round(arr, 6), return_counts=True)
    dup = u[c > 1]
    if len(dup):
        print(f'[audit] {nm}: repeated value(s) {dup} across seeds -- verify '
              'the seed actually varied before this reaches the paper.')

# ---------------------------------------------------------------------
# Mask ladder -- cache, else inference-only recompute
# ---------------------------------------------------------------------
state.setdefault('probe', {})
have = [s for s in SEEDS if str(s) in state['probe']]
missing = [s for s in SEEDS if str(s) not in state['probe']]
print(f'\n[probe] cached seeds={have}  missing={missing}')

if missing:
    need = ['EVIDENT', 'matched_probe', 'POOL_UCI', 'TEu', 'CFG_UCI']
    absent = [n for n in need if n not in globals()]
    if absent:
        print('\n' + '=' * 66)
        print('Probe ladder is NOT fully cached, so it must be recomputed.')
        print('That needs the model and pools in memory. Run, in this order:')
        print('    %run src/evident/model.py       (cpu is fine)')
        print('    %run src/evident/data_uci.py    (cpu is fine, ~1-2 min)')
        print('then re-run this cell. It does INFERENCE ONLY -- no training.')
        print(f'(missing from memory: {", ".join(absent)})')
        print('=' * 66)
        if not have:
            raise SystemExit('No cached probe results either. Stopping.')
        print('\nContinuing with the cached seeds only.\n')
    else:
        import torch
        for sd in missing:
            ck = os.path.join(OUT, f'evident_uci_s{sd}.pt')
            if not os.path.exists(ck):
                print(f'  seed {sd}: no checkpoint -- SKIPPED'); continue
            mdl = EVIDENT(CFG_UCI).to('cpu')
            mdl.load_state_dict(torch.load(ck, map_location='cpu'))
            mdl.eval()
            with torch.no_grad():
                pr = matched_probe(mdl, POOL_UCI, TEu, 'cpu',
                                   CFG_UCI['BUDGET_P'], seed=sd)
            state['probe'][str(sd)] = pr; save()
            print(f'  seed {sd}: empty={pr["empty"]["auc"]:.4f} '
                  f'top_k={pr["top_k"]["auc"]:.4f}')
            del mdl
        have = [s for s in SEEDS if str(s) in state['probe']]

if not have:
    raise SystemExit('No probe results available. Part C cannot be produced.')
rows = [state['probe'][str(s)] for s in have]
n = len(rows)
if n < len(SEEDS):
    print(f'[warn] ladder covers {n} seeds, not {len(SEEDS)}. '
          'Report n explicitly in the paper.')

# ---------------------------------------------------------------------
# Integrity, ladder, claims
# ---------------------------------------------------------------------
em = np.array([r['empty']['auc'] for r in rows])
print(f'\n[integrity] empty-mask AUC = {em.mean():.4f} +/- {em.std():.4f}')
print('  PASS -- no information bypasses the gate.'
      if abs(em.mean() - 0.5) <= 0.05 else
      '  FAIL -- tokens leak past the gate; the ladder below is void.')

print(f"\n{'mask':>14s} {'AUC':>18s} {'AP':>18s}")
agg = {}
for kk in ['empty', 'bottom_k', 'disjoint_rnd', 'random', 'top_k']:
    a = np.array([r[kk]['auc'] for r in rows])
    q = np.array([r[kk]['ap'] for r in rows])
    agg[kk] = dict(auc=[a.mean(), a.std()], ap=[q.mean(), q.std()])
    print(f'{kk:>14s} {a.mean():9.4f}+/-{a.std():.4f} '
          f'{q.mean():9.4f}+/-{q.std():.4f}')

def sig(x):
    x = np.asarray(x, float)
    return x.mean() / x.std() if x.std() > 1e-9 else float('inf')

d1 = np.array([r['top_k']['auc'] - r['bottom_k']['auc'] for r in rows])
d2 = np.array([r['top_k']['auc'] - r['disjoint_rnd']['auc'] for r in rows])
d3 = np.array([r['top_k']['auc'] - r['random']['auc'] for r in rows])

print(f'\n--- claims (n={n} seeds) ---')
for nm, d in [('C1 ranking    top_k - bottom_k    ', d1),
              ('C2 necessity  top_k - disjoint_rnd', d2),
              ('C3 vs random  top_k - random      ', d3)]:
    print(f'  {nm} = {d.mean():+.4f}+/-{d.std():.4f}  ({sig(d):.2f} sigma)')

rAUC = eA.mean() / bA.mean(); rAP = eP.mean() / bP.mean()
rK = eK.mean() / bK.mean()
print(f'\n  retention  AUC={rAUC:.3f}  AP={rAP:.3f}  P@100={rK:.3f}')
print(f'  AUC is nearly blind to a change costing {100*(1-rAP):.0f}% of '
      'average precision.')

state['claims'] = dict(C1=[d1.mean(), d1.std(), sig(d1)],
                       C2=[d2.mean(), d2.std(), sig(d2)],
                       C3=[d3.mean(), d3.std(), sig(d3)],
                       retention=dict(auc=rAUC, ap=rAP, p100=rK),
                       masks=agg, n_seeds=n)
save()

print('\n' + '=' * 70)
print('TABLE III ROW')
print('=' * 70)
print(r'\rowcolor{oursgray}')
print(rf'\ours{{EVIDENT-S}} & \ours{{{eA.mean():.4f} $\pm$ {eA.std():.4f}}} '
      rf'& \ours{{{eP.mean():.4f} $\pm$ {eP.std():.4f}}} '
      rf'& \ours{{{eK.mean():.4f} $\pm$ {eK.std():.4f}}} \\')
print('\nMASK LADDER')
_bs = chr(92)
for kk in ['empty', 'bottom_k', 'disjoint_rnd', 'random', 'top_k']:
    a, s = agg[kk]['auc']
    lbl = kk.replace('_', _bs + '_')
    print(rf'{lbl:>18s} & ${a:.4f} \pm {s:.4f}$ \\')

src = os.path.join(OUT, 'artefact_source.txt')
print(f'\nartefact source: '
      f'{open(src).read().strip() if os.path.exists(src) else "UNKNOWN"}')
print(f'saved -> {RES}')
