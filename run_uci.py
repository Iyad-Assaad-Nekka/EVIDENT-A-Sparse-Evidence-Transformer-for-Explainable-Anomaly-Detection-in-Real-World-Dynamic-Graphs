# =====================================================================
# EVIDENT on UCI-Messages (injected) -- CELL 2 : RUN + CLAIMS
# Run AFTER cells 0 and 1. Eleven trainings; budget the GPU time.
# Results are written to Drive after EVERY seed, so a disconnect costs
# one seed and not the run.
# =====================================================================
import os, json, math, numpy as np, torch
from sklearn.metrics import roc_auc_score

P_OP  = CFG_UCI['BUDGET_P']      # 0.20
SEEDS = [0, 1, 2, 3, 4]
RES   = os.path.join(OUT, 'evident_uci_results.json')
state = json.load(open(RES)) if os.path.exists(RES) else {}

def save():
    json.dump(state, open(RES, 'w'), indent=2, default=float)

# ---------------------------------------------------------------------
# PART 0 -- INTEGRITY CHECK. One cheap model, empty mask. If this is
# not at chance, tokens are leaking past the gate and every number
# below is void. This check is what exposed the 0.7975 defect.
# ---------------------------------------------------------------------
if 'integrity' not in state:
    print('=== PART 0 : integrity (empty mask must sit at chance) ===')
    cfg = dict(CFG_UCI); cfg['EPOCHS'] = 10
    mdl, _, _ = train(POOL_UCI, cfg, DEV, seed=0,
                      tr=TRu, va=VAu, te=TEu, verbose=False)
    pr = matched_probe(mdl, POOL_UCI, TEu, DEV, P_OP)
    e = pr['empty']['auc']
    print(f'  empty-mask AUC = {e:.4f}')
    if abs(e - 0.5) > 0.05:
        raise RuntimeError(f'INTEGRITY FAIL: empty mask at {e:.4f}, not 0.5. '
                           'Tokens are leaking past the gate. STOP.')
    print('  PASS -- no information bypasses the gate.')
    state['integrity'] = e; save()
    del mdl; torch.cuda.empty_cache()

# ---------------------------------------------------------------------
# PART A -- UNGATED BASELINE. Budget 1.0, all rationale losses off, so
# it is IN-distribution on the full pool. This is the only honest
# measure of what the bottleneck costs. Evaluating a p=0.20 model at
# density 1.0 is out-of-distribution and produces nonsense retention.
# ---------------------------------------------------------------------
print('\n=== PART A : ungated baseline (no bottleneck, density 1.0) ===')
state.setdefault('baseline', {})
for sd in SEEDS:
    if str(sd) in state['baseline']:
        print(f'  seed {sd}: cached'); continue
    cfg = dict(CFG_UCI)
    cfg.update(BUDGET_P=1.0, LAM_NEC=0.0, LAM_BUD=0.0,
               LAM_TV=0.0, LAM_STAB=0.0, LAM_ENT=0.0)
    _, m, _ = train(POOL_UCI, cfg, DEV, seed=sd,
                    tr=TRu, va=VAu, te=TEu, verbose=False)
    state['baseline'][str(sd)] = m; save()
    print(f'  seed {sd}: AUC={m["auc"]:.4f} AP={m["ap"]:.4f} '
          f'P@100={m["p100"]:.4f}')
    torch.cuda.empty_cache()

bA = np.array([state['baseline'][str(s)]['auc'] for s in SEEDS])
bP = np.array([state['baseline'][str(s)]['ap'] for s in SEEDS])
bK = np.array([state['baseline'][str(s)]['p100'] for s in SEEDS])
print(f'  BASELINE AUC={bA.mean():.4f}+/-{bA.std():.4f} '
      f'AP={bP.mean():.4f}+/-{bP.std():.4f} P@100={bK.mean():.4f}+/-{bK.std():.4f}')

# ---------------------------------------------------------------------
# PART B -- EVIDENT at the operating point, five seeds
# ---------------------------------------------------------------------
print(f'\n=== PART B : EVIDENT at p={P_OP}, {len(SEEDS)} seeds ===')
state.setdefault('evident', {}); state.setdefault('probe', {})
for sd in SEEDS:
    if str(sd) in state['evident']:
        print(f'  seed {sd}: cached'); continue
    mdl, m, _ = train(POOL_UCI, CFG_UCI, DEV, seed=sd,
                      tr=TRu, va=VAu, te=TEu, verbose=False)
    pr = matched_probe(mdl, POOL_UCI, TEu, DEV, P_OP, seed=sd)
    torch.save(mdl.state_dict(), os.path.join(OUT, f'evident_uci_s{sd}.pt'))
    state['evident'][str(sd)] = m; state['probe'][str(sd)] = pr; save()
    print(f'  seed {sd}: AUC={m["auc"]:.4f} AP={m["ap"]:.4f} '
          f'P@100={m["p100"]:.4f}')
    del mdl; torch.cuda.empty_cache()

eA = np.array([state['evident'][str(s)]['auc'] for s in SEEDS])
eP = np.array([state['evident'][str(s)]['ap'] for s in SEEDS])
eK = np.array([state['evident'][str(s)]['p100'] for s in SEEDS])
print(f'  EVIDENT  AUC={eA.mean():.4f}+/-{eA.std():.4f} '
      f'AP={eP.mean():.4f}+/-{eP.std():.4f} P@100={eK.mean():.4f}+/-{eK.std():.4f}')

# ---------------------------------------------------------------------
# PART C -- the three claims, each with its own significance
# ---------------------------------------------------------------------
rows = [state['probe'][str(s)] for s in SEEDS]
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
ret = eA.mean() / max(bA.mean(), 1e-9)

print('\n--- claims ---')
print(f'  C1 ranking    top_k - bottom_k     = {d1.mean():+.4f}+/-{d1.std():.4f}'
      f'  ({sig(d1):.2f} sigma)')
print(f'  C2 necessity  top_k - disjoint_rnd = {d2.mean():+.4f}+/-{d2.std():.4f}'
      f'  ({sig(d2):.2f} sigma)')
print(f'  C3 vs random  top_k - random       = {d3.mean():+.4f}+/-{d3.std():.4f}'
      f'  ({sig(d3):.2f} sigma)')
print(f'\n  bottleneck cost: {eA.mean():.4f} - {bA.mean():.4f} '
      f'= {eA.mean()-bA.mean():+.4f} AUC    retention = {ret:.3f}')

state['claims'] = dict(C1=[d1.mean(), d1.std(), sig(d1)],
                       C2=[d2.mean(), d2.std(), sig(d2)],
                       C3=[d3.mean(), d3.std(), sig(d3)],
                       retention=ret, masks=agg)
save()

# ---------------------------------------------------------------------
# TABLE III ROW
# ---------------------------------------------------------------------
print('\n' + '=' * 70)
print('TABLE III ROW (paste into the LaTeX table)')
print('=' * 70)
print(r'\rowcolor{oursgray}')
print(rf'\ours{{EVIDENT-S}} & \ours{{{eA.mean():.4f} $\pm$ {eA.std():.4f}}} '
      rf'& \ours{{{eP.mean():.4f}}} & \ours{{{eK.mean():.4f} $\pm$ {eK.std():.4f}}} \\')
print('\nCAPTION SENTENCE TO ADD:')
print('EVIDENT is evaluated under fixed-cardinality pools and therefore '
      'has no access to the neighbourhood-cardinality channel that reaches '
      'AUC 0.9861 unaided on this benchmark (Section VII); every other row '
      'retains it.')
print(f'\nartefact source: {open(os.path.join(OUT,"artefact_source.txt")).read().strip()}')
print(f'saved -> {RES}')
