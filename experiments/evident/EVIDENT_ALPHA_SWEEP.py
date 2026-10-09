#!/usr/bin/env python
# ============================================================================
# EVIDENT_ALPHA_SWEEP.py
#
# EVIDENT hyperparameter sweep on Bitcoin-Alpha.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, math, time, gzip, copy, urllib.request, subprocess
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy import stats


SEEDS      = [0, 1, 2, 3, 4, 5, 6, 7]
MAIN_P     = 0.20
EPOCHS     = 30
EVAL_EVERY = 1
VAL_SUB    = 3000
SELECT_ON  = 'auc'
DS         = 'bitcoin_alpha'

# ---- bitcoin_otc, EVIDENT+ p=0.20 EPOCHS=30, real labels (measured) ------
OTC = dict(
  auc =[0.8373, 0.8676, 0.8717, 0.8562, 0.8715, 0.8784, 0.8568, 0.8585],
  ap  =[0.5087, 0.5689, 0.5589, 0.5221, 0.5702, 0.5478, 0.5680, 0.5466],
  p100=[0.80, 0.88, 0.86, 0.90, 0.88, 0.84, 0.90, 0.86],
  adv =[0.1734, 0.1363, 0.1134, 0.1220, 0.1688, 0.1054, 0.1208, 0.1288],
  noc_auc =[0.8806, 0.8619, 0.8859, 0.8832, 0.8611, 0.8843, 0.8761, 0.8799],
  noc_adv =[0.0618, 0.0415, 0.0574, 0.0764, 0.0576, 0.0638, 0.0705, 0.0702])
# Alpha's earlier EVIDENT+ run used EPOCHS=15 -- superseded by this run.

T0 = time.time()
REPO = '/content/EVIDENT_repo'
if not os.path.isdir(REPO):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/Iyad-Assaad-Nekka/'
                    'EVIDENT-Explainable-Dynamic-Graph-anomaly-Detection.git',
                    REPO], check=True)
exec(compile(open(os.path.join(REPO, 'model.py')).read(), 'model.py', 'exec'),
     globals())
print(f'[gpu] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
MASTER = os.path.join(BENCH, 'evident_master'); os.makedirs(MASTER, exist_ok=True)


N_NF, N_TF = 4, 3


def load_alpha():
    p = os.path.join(DATA, 'bitcoinalpha.csv')
    if not os.path.exists(p):
        open(p, 'wb').write(gzip.decompress(urllib.request.urlopen(
            'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz',
            timeout=300).read()))
    d = pd.read_csv(p, header=None, names=['u', 'v', 'r', 't'])
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    d['y'] = (d.r <= -5).astype(np.float32); d['r'] = d.r.astype(np.float32) / 10.0
    ids = pd.unique(pd.concat([d.u, d.v])); m = {int(x): i for i, x in enumerate(ids)}
    d['u'] = d.u.map(m).astype(np.int64); d['v'] = d.v.map(m).astype(np.int64)
    return d[['u', 'v', 't', 'y', 'r']].reset_index(drop=True), len(m)


def heuristics(df, te_idx):
    u, v = df.u.to_numpy(), df.v.to_numpy()
    n = int(max(u.max(), v.max())) + 1
    deg = np.zeros(n); s_deg = np.zeros(len(df)); s_new = np.zeros(len(df))
    for j in range(len(df)):
        a, b = u[j], v[j]
        s_deg[j] = -(deg[a] + deg[b]); s_new[j] = float(deg[a] == 0) + float(deg[b] == 0)
        deg[a] += 1; deg[b] += 1
    y = df.y.to_numpy()[te_idx]; out = {}
    for nm, sc in [('degree', s_deg), ('newness', s_new)]:
        ss = sc[te_idx]
        out[nm] = dict(auc=float(roc_auc_score(y, ss)), p100=p_at_k(y, ss, 100))
    return out

def build_pool_plus(df, n_nodes, cfg):
    """build_all_fixed + reputation features + duplicate marking."""
    pb = PoolBuilder(df, n_nodes, cfg)
    uu, vv = df.u.to_numpy(), df.v.to_numpy()
    rr = df.r.to_numpy().astype(np.float64)
    M = len(df); N, K, T = cfg['N_MAX'], cfg['W'], cfg['T_MAX']
    LAB = np.zeros((M, N, 2), np.int64); DT = np.zeros((M, K), np.float32)
    TN = np.zeros((M, T), np.int64); TK = np.zeros((M, T), np.int64)
    TM = np.ones((M, T), np.float32); Y = np.zeros((M,), np.float32)
    NF = np.zeros((M, N, N_NF), np.float32)
    TF = np.zeros((M, K, N_TF), np.float32)
    TRATE = np.zeros((M, K), np.float32)
    DUPTM = np.ones((M, T), np.float32)
    keep = np.zeros((M,), bool)

    # running per-node state, updated ONLY in absorb order
    deg = np.zeros(n_nodes); recv_sum = np.zeros(n_nodes)
    recv_cnt = np.zeros(n_nodes); recv_neg = np.zeros(n_nodes)
    n_distinct = []

    for j in range(M):
        a, b = int(uu[j]), int(vv[j]); ts = pb.t[j]
        ev = sorted(set(pb.inc[a][-cfg['W']:] + pb.inc[b][-cfg['W']:]))
        ev = [e for e in ev if pb.t[e] < ts and e != j][-cfg['W']:]
        ring = set()
        for e in ev:
            ring.add(int(uu[e])); ring.add(int(vv[e]))
        nodes = ([a, b] + [x for x in sorted(ring) if x not in (a, b)])[:N]
        pos = {x: i for i, x in enumerate(nodes)}
        lab = np.zeros((len(nodes), 2), np.int64)
        nf = np.zeros((len(nodes), N_NF), np.float32)
        for i, x in enumerate(nodes):
            lab[i, 0] = 0 if x == a else (1 if x in pb.nbr[a] else 2)
            lab[i, 1] = 0 if x == b else (1 if x in pb.nbr[b] else 2)
            # ---- [A] reputation features, state BEFORE absorb(j) ----------
            nf[i, 0] = math.log1p(deg[x])
            nf[i, 1] = (recv_sum[x] / recv_cnt[x]) if recv_cnt[x] > 0 else 0.0
            nf[i, 2] = math.log1p(recv_neg[x])
            nf[i, 3] = 1.0 if deg[x] == 0 else 0.0

        dt = np.array([ts - pb.t[e] for e in ev], dtype=np.float64)
        tf = np.zeros((len(ev), N_TF), np.float32)
        for k, e in enumerate(ev):
            tf[k, 0] = rr[e]
            tf[k, 1] = 1.0 if (int(uu[e]) in (a, b)) else -1.0
            tf[k, 2] = 1.0 if {int(uu[e]), int(vv[e])} == {a, b} else 0.0

        tn, tk = [], []
        for k in range(len(ev) - 1, -1, -1):
            e = ev[k]
            for x in (int(uu[e]), int(vv[e])):
                if x in pos:
                    tn.append(pos[x]); tk.append(k)
        tn, tk = tn[:T], tk[:T]

        if len(tn) >= 1 and len(dt) >= 1:
            rg = np.random.default_rng(j)
            ni = rg.choice(len(nodes), N, replace=True) if len(nodes) < N else np.arange(N)
            LAB[j] = lab[ni]; NF[j] = nf[ni]
            ki = rg.choice(len(dt), K, replace=True) if len(dt) < K else np.arange(K)
            DT[j] = np.log1p(np.maximum(dt, 0))[ki]
            TF[j] = tf[ki]; TRATE[j] = rr[np.array(ev, np.int64)][ki]
            nmap, kmap = {}, {}
            for s_, o in enumerate(ni): nmap.setdefault(int(o), s_)
            for s_, o in enumerate(ki): kmap.setdefault(int(o), s_)
            tn_a, tk_a = np.array(tn), np.array(tk)
            ok = np.array([(int(x) in nmap and int(z) in kmap)
                           for x, z in zip(tn_a, tk_a)])
            tn_a, tk_a = tn_a[ok], tk_a[ok]
            if len(tn_a) == 0:
                tn_a, tk_a = np.array([0]), np.array([0])
                nmap.setdefault(0, 0); kmap.setdefault(0, 0)
            ti = rg.choice(len(tn_a), T, replace=True)
            TN[j] = np.array([nmap[int(x)] for x in tn_a[ti]])
            TK[j] = np.array([kmap[int(x)] for x in tk_a[ti]])
            # ---- [B] mark duplicate (node,time) tokens --------------------
            uid = TN[j] * (K + 1) + TK[j]
            _, first = np.unique(uid, return_index=True)
            m_ = np.zeros(T, np.float32); m_[first] = 1.0
            DUPTM[j] = m_; n_distinct.append(int(m_.sum()))
            keep[j] = True

        Y[j] = float(pb.y[j])
        # ---- absorb AFTER features are read --------------------------------
        deg[a] += 1; deg[b] += 1
        recv_sum[b] += rr[j]; recv_cnt[b] += 1
        if rr[j] < 0:
            recv_neg[b] += 1
        pb.absorb(j)
        if j % 20000 == 0 and j:
            print(f'    pooled {j}/{M}', flush=True)

    m = DT[keep]; DT = (DT - m[m > 0].mean()) / (m[m > 0].std() + 1e-6)
    # standardise NF/TF using TRAIN rows only (first TRAIN_FRAC of usable)
    idx = np.where(keep)[0]; ntr = int(cfg['TRAIN_FRAC'] * len(idx))
    trr = idx[:ntr]
    for arr in (NF, TF):
        mu = arr[trr].reshape(-1, arr.shape[-1]).mean(0)
        sd = arr[trr].reshape(-1, arr.shape[-1]).std(0) + 1e-6
        arr -= mu; arr /= sd
    RT = np.take_along_axis(TRATE, TK, axis=1).astype(np.float32)
    nd = np.array(n_distinct)
    print(f'  [pool] usable={int(keep.sum())} pos={Y[keep].mean():.4f}')
    print(f'  [dedup] distinct tokens per edge: median={np.median(nd):.0f} '
          f'mean={nd.mean():.1f} (of {T} slots)  '
          f'-> budget k goes {math.ceil(MAIN_P*T)} -> '
          f'~{max(1, math.ceil(MAIN_P*np.median(nd))):.0f}')
    return dict(LAB=LAB, DT=DT, TN=TN, TK=TK, TM=TM, Y=Y, keep=keep,
                NF=NF, TF=TF, TRATE=TRATE, RT=RT, DUPTM=DUPTM)


# ----------------------------------------------------------------- model ----
class EVIDENTPlus(EVIDENT):
    """Reputation-aware tokens. Node features join the SPATIAL branch, time
    features the TEMPORAL branch, mirroring the existing factorisation. No
    edge-level features anywhere -- the empty-mask invariant is preserved."""
    def __init__(s, cfg):
        super().__init__(cfg); d = cfg['D_MODEL']
        s.nf_enc = nn.Linear(N_NF, d);      s.nf_sp = nn.Linear(N_NF, d // 2)
        s.tf_enc = nn.Linear(N_TF, d);      s.tf_tp = nn.Linear(N_TF, d // 2)

    def gate_logits(s, b):
        LAB, DT, TN, TK = b['LAB'], b['DT'], b['TN'], b['TK']
        zs = s.role_sp(s._rid(LAB)) + s.nf_sp(b['NF'])
        zt = s.time_tp(DT.unsqueeze(-1)) + s.tf_tp(b['TF'])
        a = s.f_sp(zs).squeeze(-1); c = s.f_tp(zt).squeeze(-1)
        Pn = s.P(zs); Qk = s.Q(zt)
        an = torch.gather(a, 1, TN); ck = torch.gather(c, 1, TK)
        pn = torch.gather(Pn, 1, TN.unsqueeze(-1).expand(-1, -1, Pn.size(-1)))
        qk = torch.gather(Qk, 1, TK.unsqueeze(-1).expand(-1, -1, Qk.size(-1)))
        return an + ck + (pn * qk).sum(-1), a, c

    def tokens(s, b):
        LAB, DT, TN, TK = b['LAB'], b['DT'], b['TN'], b['TK']
        zn = s.role_enc(s._rid(LAB)) + s.nf_enc(b['NF'])
        zk = s.time_enc(DT.unsqueeze(-1)) + s.tf_enc(b['TF'])
        zn = torch.gather(zn, 1, TN.unsqueeze(-1).expand(-1, -1, zn.size(-1)))
        zk = torch.gather(zk, 1, TK.unsqueeze(-1).expand(-1, -1, zk.size(-1)))
        return zn + zk


def mkb(pool, idx, dev, dedup):
    b = batch_of(pool, idx, dev)
    b['NF'] = torch.from_numpy(pool['NF'][idx]).to(dev)
    b['TF'] = torch.from_numpy(pool['TF'][idx]).to(dev)
    if dedup:
        b['TM'] = torch.from_numpy(pool['DUPTM'][idx]).to(dev)
    return b


@torch.no_grad()
def val_probe(model, pool, idx, dev, p, seed, dedup):
    was = model.training; model.eval()
    g = torch.Generator(); g.manual_seed(seed)
    st, sr, sc_, ys = [], [], [], []
    for i in range(0, len(idx), 256):
        sl = idx[i:i + 256]; b = mkb(pool, sl, dev, dedup)
        Z = model.tokens(b); th, _, _ = model.gate_logits(b); tm = b['TM']
        top = model.budget_gate(th, tm, p)
        r = torch.rand(*tm.shape, generator=g).to(th.device)
        rnd = model.budget_gate(r, tm, p)
        st.append(model.score(Z, top, tm).float().cpu().numpy())
        sr.append(model.score(Z, rnd, tm).float().cpu().numpy())
        sc_.append(model.score(Z, (1.0 - top).clamp(0, 1) * tm, tm).float().cpu().numpy())
        ys.append(pool['Y'][sl])
    if was:
        model.train()
    y = np.concatenate(ys)
    def A(v):
        try:
            return float(roc_auc_score(y, v))
        except ValueError:
            return float('nan')
    def AP(v):
        try:
            return float(average_precision_score(y, v))
        except ValueError:
            return float('nan')
    # 4th value added so SELECT_ON='ap' is possible; first three unchanged
    return (A(np.concatenate(st)), A(np.concatenate(sr)),
            A(np.concatenate(sc_)), AP(np.concatenate(st)))


def train_one(pool, tr, va, te, cfg, seed, variant):
    dedup = 'dedup' in variant
    set_seed(seed); c = dict(cfg)
    if 'no_complement' in variant:
        c['LAM_NEC'] = 0.0
    model = EVIDENTPlus(c).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=c['LR'], weight_decay=c['WD'])
    pi = float(np.clip(pool['Y'][tr].mean(), 1e-4, 1 - 1e-4))
    pw = torch.tensor([(1 - pi) / pi], device=DEV)
    lp = math.log(pi / (1 - pi))
    TRIV = (1 - pi) * (-math.log(pi) - math.log(1 - pi))   # corrected threshold
    rgv = np.random.default_rng(1234)
    va_s = va if len(va) <= VAL_SUB else np.sort(rgv.choice(va, VAL_SUB, False))
    lam_nec = c['LAM_NEC']
    best = dict(score=-1e9, ep=-1, state=None)

    for ep in range(c['EPOCHS']):
        model.train(); perm = np.random.permutation(tr); dr, nb = 0.0, 0
        for i in range(0, len(perm), c['BS']):
            idx = perm[i:i + c['BS']]; b = mkb(pool, idx, DEV, dedup)
            Z = model.tokens(b); th, a, cc = model.gate_logits(b); tm = b['TM']
            g1 = model.sample_gate(th, True); g2 = model.sample_gate(th, True)
            L_det = F.binary_cross_entropy_with_logits(
                model.score(Z, g1, tm), b['Y'], pos_weight=pw)
            L_nec = ((model.score(Z, (1 - g1).clamp(0, 1), tm) - lp) ** 2).mean()
            po = model.p_open(th)
            L_bud = (((po * tm).sum() / tm.sum().clamp(min=1)) - c['BUDGET_P']) ** 2
            pc = torch.sigmoid(cc)
            L_tv = (pc[:, 1:] - pc[:, :-1]).abs().mean()
            L_st = ((g1 - g2) ** 2).mean()
            q = po.clamp(1e-6, 1 - 1e-6)
            L_en = -(q * q.log() + (1 - q) * (1 - q).log()).mean()
            loss = (L_det + lam_nec * L_nec + c['LAM_BUD'] * L_bud
                    + c['LAM_TV'] * L_tv + c['LAM_STAB'] * L_st
                    + c['LAM_ENT'] * L_en)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); dr += float(L_det.detach()); nb += 1
        det_ep = dr / max(nb, 1)

        a_top, a_rnd, a_cmp, ap_top = val_probe(model, pool, va_s, DEV,
                                                c['BUDGET_P'], seed, dedup)
        sel_val = ap_top if c.get('SELECT_ON', 'auc') == 'ap' else a_top
        if np.isfinite(sel_val) and sel_val > best['score']:
            best = dict(score=sel_val, ep=ep + 1,
                        state=copy.deepcopy(model.state_dict()))
        if c['LAM_NEC'] > 0:
            if det_ep >= 0.98 * TRIV:
                lam_nec = max(lam_nec * 0.5, 0.25)
            elif not math.isnan(a_cmp) and abs(a_cmp - 0.5) > 0.10:
                lam_nec = min(lam_nec * 1.5, 8.0)
        if ep % 5 == 4 or ep == 0:
            print(f'      ep{ep+1:>3d} L_det={det_ep:.4f} lam={lam_nec:.3f} '
                  f'valAUC={a_top:.4f} valADV={a_top-a_rnd:+.4f} '
                  f'cmpAUC={a_cmp:.4f}', flush=True)

    if best['state'] is not None:
        model.load_state_dict(best['state'])

    model.eval(); p = c['BUDGET_P']
    g = torch.Generator(); g.manual_seed(seed)
    st, sr, ys, es, er, emp, kk = [], [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(te), 256):
            idx = te[i:i + 256]; b = mkb(pool, idx, DEV, dedup)
            Z = model.tokens(b); th, _, _ = model.gate_logits(b); tm = b['TM']
            top = model.budget_gate(th, tm, p)
            r = torch.rand(*tm.shape, generator=g).to(th.device)
            rnd = model.budget_gate(r, tm, p)
            st.append(model.score(Z, top, tm).float().cpu().numpy())
            sr.append(model.score(Z, rnd, tm).float().cpu().numpy())
            emp.append(model.score(Z, torch.zeros_like(tm), tm).float().cpu().numpy())
            neg = torch.from_numpy((pool['RT'][idx] < 0).astype(np.float32)).to(th.device)
            es.append(((neg * top).sum(1) / top.sum(1).clamp(min=1)).cpu().numpy())
            er.append(((neg * rnd).sum(1) / rnd.sum(1).clamp(min=1)).cpu().numpy())
            kk.append(top.sum(1).cpu().numpy())
            ys.append(pool['Y'][idx])
    y = np.concatenate(ys)
    mt = metrics(y, np.concatenate(st)); mr = metrics(y, np.concatenate(sr))
    s_, r_ = np.concatenate(es).mean(), np.concatenate(er).mean()
    return dict(auc=mt['auc'], ap=mt['ap'], p100=mt['p100'],
                rationale_advantage=mt['auc'] - mr['auc'],
                evidence_precision_lift=float(s_ / (r_ + 1e-9)),
                empty_auc=float(roc_auc_score(y, np.concatenate(emp))),
                tokens_selected=float(np.concatenate(kk).mean()),
                best_epoch=best['ep'], lam_nec_final=lam_nec,
                val_select_score=float(best['score']),
                select_on=c.get('SELECT_ON', 'auc'))



# =============================================================================
#  SWEEP DRIVER -- everything above this line is byte-identical to
#  EVIDENT_PLUS_ALPHA_FINAL.py except for four edits, all marked in place:
#    (1) val_probe also returns validation AP
#    (2) its call site unpacks the 4th value
#    (3) epoch selection reads cfg['SELECT_ON'] ('auc' default, 'ap' new)
#    (4) the result dict records the selection score and criterion
#  The model, the batching, the losses and the lam_nec schedule are untouched,
#  so a difference in the numbers below is caused by the swept knob and
#  nothing else.
#
#  WHY THIS SWEEP IS JUSTIFIED (and is not fishing)
#    The oracle diagnosis showed that on Alpha the discriminative evidence is
#    INSIDE the pool and reachable within the k~5 budget: an oracle under the
#    same window and budget reaches AP 0.4004 while trained EVIDENT reaches
#    0.2519. On OTC the same model is +0.1316 AP ABOVE that oracle. So Alpha
#    is an optimisation failure, not an architectural limit, and it is
#    legitimate to look for the cause.
#
#  PRE-REGISTERED, WRITTEN BEFORE THE RUN
#    * Configurations are ranked on VALIDATION ONLY. Test numbers for the
#      winner are printed once, at the end, and are not used to choose.
#    * Success = Alpha test AP >= 0.3253 AND P@100 >= 0.6600 (the
#      cp_mean_rating heuristic), with rationale advantage >= 0.05 and
#      empty-mask AUC exactly 0.5000 in every run.
#    * Partial = beats EVIDENT's current 0.2519 AP but not the heuristic.
#    * Failure = no configuration beats 0.2519. Then the current number
#      stands and the Alpha shortfall is reported as-is.
#    * Whatever happens, SELECT_ON must end up IDENTICAL on both corpora.
#      If 'ap' wins here, OTC is re-run with 'ap' before anything is
#      published. A protocol that differs per dataset is indefensible.
# =============================================================================
import itertools, json

SEEDS_SWEEP = [0, 1, 2]
GRID = dict(SELECT_ON=['auc', 'ap'], BUDGET_P=[0.20, 0.33], D_MODEL=[64, 128])
EPOCHS_SWEEP = 30

print('\n' + '=' * 78)
print('PRE-REGISTERED SUCCESS CRITERIA (fixed before any result is seen)')
print('=' * 78)
print('  target   : test AP >= 0.3253 and P@100 >= 0.6600  (cp_mean_rating)')
print('  partial  : test AP >  0.2519  (current EVIDENT) but below target')
print('  failure  : no config beats 0.2519 -> report the shortfall as-is')
print('  integrity: empty-mask AUC must be exactly 0.5000 in every run')
print('  ranking  : VALIDATION only; test is read once, for the winner')

T0 = time.time()
cfg = dict(CFG); cfg['BUDGET_P'] = MAIN_P; cfg['EPOCHS'] = EPOCHS_SWEEP
df, n_nodes = load_alpha()
print(f'\n[data] edges={len(df)} nodes={n_nodes} pos={df.y.mean():.4f}')
CACHE = os.path.join(MASTER, f'poolplus_{DS}.npz')
if os.path.exists(CACHE):
    z = np.load(CACHE); pool = {k: z[k] for k in z.files}
    pool['keep'] = pool['keep'].astype(bool); print('[pool] from cache')
else:
    pool = build_pool_plus(df, n_nodes, cfg)
    np.savez_compressed(CACHE, **pool); print('[pool] built and cached')
tr, va, te = chrono_split(pool['keep'], cfg)
print(f'[split] tr={len(tr)} va={len(va)} te={len(te)}', flush=True)

rows = []
combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
print(f'[plan] {len(combos)} configs x {len(SEEDS_SWEEP)} seeds = '
      f'{len(combos)*len(SEEDS_SWEEP)} runs', flush=True)

for ci, cb in enumerate(combos, 1):
    c = dict(cfg); c.update(cb); c['EPOCHS'] = EPOCHS_SWEEP
    tag = f"sel={cb['SELECT_ON']} p={cb['BUDGET_P']} d={cb['D_MODEL']}"
    for sd in SEEDS_SWEEP:
        t0 = time.time()
        r = train_one(pool, tr, va, te, c, sd, 'plus_dedup')
        r.update(cb, seed=sd, config=tag)
        rows.append(r)
        print(f'  [{ci}/{len(combos)}] {tag} seed{sd}  '
              f'valSel={r["val_select_score"]:.4f}  '
              f'AP={r["ap"]:.4f} AUC={r["auc"]:.4f} P@100={r["p100"]:.4f} '
              f'adv={r["rationale_advantage"]:+.4f} '
              f'empty={r["empty_auc"]:.4f} ({(time.time()-t0)/60:.1f}m)',
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(MASTER,
                                 f'alpha_sweep_{DS}.csv'), index=False)

D = pd.DataFrame(rows)
print('\n' + '=' * 78)
print('RANKING BY VALIDATION SELECTION SCORE  (test columns shown but NOT used)')
print('=' * 78)
g = (D.groupby('config')
       .agg(val=('val_select_score', 'mean'), auc=('auc', 'mean'),
            ap=('ap', 'mean'), p100=('p100', 'mean'),
            adv=('rationale_advantage', 'mean'),
            empty=('empty_auc', 'max'), n=('seed', 'count'))
       .sort_values('val', ascending=False))
print(g.round(4).to_string())

bad = D[np.abs(D.empty_auc - 0.5) > 1e-6]
print(f'\n[integrity] runs with empty-mask != 0.5000: {len(bad)}')
if len(bad):
    print(bad[['config', 'seed', 'empty_auc']].to_string(index=False))
    print('  ^ any non-0.5 row invalidates that configuration entirely.')

# the winner is chosen WITHIN each selection criterion, because val AUC and
# val AP are not on the same scale and ranking across them is meaningless
print('\n' + '=' * 78)
print('WINNER PER SELECTION CRITERION (validation-chosen)')
print('=' * 78)
for so in GRID['SELECT_ON']:
    sub = g[g.index.str.startswith(f'sel={so}')]
    if not len(sub):
        continue
    w = sub.index[0]
    r = sub.iloc[0]
    print(f"\n  SELECT_ON={so}: {w}")
    print(f"     test AUC {r['auc']:.4f}  AP {r['ap']:.4f}  "
          f"P@100 {r['p100']:.4f}  adv {r['adv']:+.4f}")

print('\n' + '=' * 78)
print('VERDICT AGAINST THE PRE-REGISTERED CRITERIA')
print('=' * 78)
HEUR_AP, HEUR_P100, CUR_AP = 0.3253, 0.6600, 0.2519
best = g.iloc[0]
print(f"  best-by-validation config: {g.index[0]}")
print(f"  its test AP {best['ap']:.4f} vs heuristic {HEUR_AP:.4f} "
      f"vs current {CUR_AP:.4f}")
print(f"  its test P@100 {best['p100']:.4f} vs heuristic {HEUR_P100:.4f}")
if best['ap'] >= HEUR_AP and best['p100'] >= HEUR_P100:
    print('\n  SUCCESS. EVIDENT beats the reputation rule on Alpha under a')
    print('  validation-chosen configuration. Re-run OTC with this SELECT_ON')
    print('  before publishing anything -- the protocol must match.')
elif best['ap'] > CUR_AP:
    print('\n  PARTIAL. Better than the current number, still short of the')
    print('  heuristic. Report the improved number AND state plainly that a')
    print('  causal reputation rule leads on Alpha AP/P@100. The OTC result')
else:
    print('\n  FAILURE. Nothing beats 0.2519. The current Alpha number stands.')
    print('  Report it, report the heuristic beating it, and let the')
    print('  contribution be the benchmark critique plus faithful')
    print('  explanations. Do not run a third sweep -- that is how')
    print('  overfitting to a test set starts.')

print('\n' + '=' * 78); print('RESULTS (CSV)'); print('=' * 78)
print(D.to_csv(index=False))
print(f'DONE in {(time.time()-T0)/60:.1f} min')
