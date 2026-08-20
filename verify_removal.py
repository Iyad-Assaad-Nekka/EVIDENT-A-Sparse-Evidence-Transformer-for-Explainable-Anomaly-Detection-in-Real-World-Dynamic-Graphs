"""
Standalone integrity check for gated architectures.

Flooring a gate at a small epsilon is NOT removal. Every token still
contributes at O(eps), and because AUC is rank-based and hence
scale-invariant, a signal attenuated by nine orders of magnitude ranks
identically to an unattenuated one. Every mask then behaves as the full
pool, and the resulting numbers are void.

The check: score every test event with an EMPTY mask. A model that truly
consumes only its rationale has no information left and must sit at
chance. Anything materially away from 0.5 means tokens are leaking.

This is how we found a detector reporting 0.7935 +/- 0.0050 whose empty
mask scored 0.7975.

Usage (after model.py and data_uci.py have been run):

    python scripts/verify_removal.py

or, inside a notebook:

    exec(open('scripts/verify_removal.py').read())
"""
import sys
import numpy as np

TOL = 0.05


def verify(model, pool, te, dev, p):
    from sklearn.metrics import roc_auc_score
    probe = matched_probe(model, pool, te, dev, p)          # noqa: F821
    empty = probe['empty']['auc']
    topk = probe['top_k']['auc']

    print(f'  empty-mask AUC : {empty:.4f}   (must be ~0.5)')
    print(f'  top-k     AUC : {topk:.4f}')

    if abs(empty - 0.5) > TOL:
        print(f'\n  FAIL: empty mask sits at {empty:.4f}, not chance.')
        print('  Tokens are bypassing the gate. Every downstream number '
              'is void.')
        print('  Check that closed gates set the attention bias to -inf at '
              'evaluation\n  rather than to log(eps).')
        return False

    print('\n  PASS: no information bypasses the gate.')
    return True


if __name__ == '__main__':
    required = ['matched_probe', 'POOL_UCI', 'TEu', 'DEV', 'CFG_UCI']
    missing = [r for r in required if r not in globals()]
    if missing:
        print('This script runs inside the notebook session that already '
              'holds\nthe model and pools. Missing: ' + ', '.join(missing))
        print('Run src/evident/model.py and src/evident/data_uci.py first.')
        sys.exit(1)
    import glob
    import torch
    ckpts = sorted(glob.glob(f'{OUT}/evident_uci_s*.pt'))            # noqa: F821
    if not ckpts:
        print(f'No checkpoints in {OUT}. Run run_uci.py first.')     # noqa: F821
        sys.exit(1)
    ok = True
    for c in ckpts:
        print(f'\n{c}')
        m = EVIDENT(CFG_UCI).to(DEV)                                 # noqa: F821
        m.load_state_dict(torch.load(c, map_location=DEV))           # noqa: F821
        ok &= verify(m, POOL_UCI, TEu, DEV, CFG_UCI['BUDGET_P'])     # noqa: F821
    sys.exit(0 if ok else 1)
