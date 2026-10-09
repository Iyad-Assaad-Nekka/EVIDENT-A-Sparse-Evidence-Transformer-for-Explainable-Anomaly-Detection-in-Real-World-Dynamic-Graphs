#!/usr/bin/env python
# ============================================================================
# EVIDENT_NO_RATINGS.py
#
# EVIDENT ablation: all rating-derived inputs removed, real labels, both
# datasets.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, math, time, gzip, copy, urllib.request, subprocess
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy import stats

SEEDS      = [0, 1, 2, 3]       # raise to [0..7] if you want 8-seed parity
MAIN_P     = 0.20
EPOCHS     = 30
VAL_SUB    = 3000
TIME_BUDGET_MIN = 170
DATASETS   = ['bitcoin_otc', 'bitcoin_alpha']
N_NF, N_TF = 4, 3

# dims that encode ratings; everything else is structure or time
NF_RATING_DIMS = [1, 2]    # mean rating received, log1p(#negatives received)
TF_RATING_DIMS = [0]       # the event's own rating

# measured reference values (full EVIDENT, same protocol)
REF = {
 'bitcoin_otc':   dict(ev=0.8622, ap=0.5489, p100=0.8650, adv=0.1336,
                       strgnn=0.5373, taddy=0.5390, sad=0.6794, degree=0.4920),
 'bitcoin_alpha': dict(ev=0.7606, ap=0.2519, p100=0.4163, adv=0.0762,
                       strgnn=0.5561, taddy=0.5570, sad=0.5974, degree=0.4269),
}

T0 = time.time()

# --- mount Drive if it is there: the main runs cached their pools on Drive,
#     and reusing that exact file is better than rebuilding one.
try:
    from google.colab import drive
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
except Exception as _e:
    print(f'[drive] not mounted ({_e.__class__.__name__}); '
          f'the pool will be rebuilt locally if needed')

REPO = '/content/EVIDENT_repo'
if not os.path.isdir(REPO):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/Iyad-Assaad-Nekka/'
                    'EVIDENT-Explainable-Dynamic-Graph-anomaly-Detection.git',
                    REPO], check=True)
exec(compile(open(os.path.join(REPO, 'model.py')).read(), 'model.py', 'exec'),
     globals())
print('=' * 80)
print('EVIDENT WITHOUT RATING FEATURES  --  real labels, both datasets')
print('=' * 80)
print(f'[gpu] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
MASTER = os.path.join(BENCH, 'evident_master'); os.makedirs(MASTER, exist_ok=True)


# ------------------------------------------------------------------ data ---
def load_bitcoin(which):
    url = ('https://snap.stanford.edu/data/soc-sign-bitcoin'
           + ('otc' if which == 'bitcoin_otc' else 'alpha') + '.csv.gz')
    p = os.path.join(DATA, 'bitcoin' + ('otc' if which == 'bitcoin_otc'
                                        else 'alpha') + '.csv')
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
    """Identical to the model used for every EVIDENT number in the paper."""
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


def mkb(pool, idx, dev, dedup, strip):
    b = batch_of(pool, idx, dev)
    nf = pool['NF'][idx].copy(); tf = pool['TF'][idx].copy()
    if strip:
        nf[..., NF_RATING_DIMS] = 0.0     # already standardised => 0 is the mean
        tf[..., TF_RATING_DIMS] = 0.0
    b['NF'] = torch.from_numpy(nf).to(dev)
    b['TF'] = torch.from_numpy(tf).to(dev)
    if dedup:
        b['TM'] = torch.from_numpy(pool['DUPTM'][idx]).to(dev)
    return b


@torch.no_grad()
def val_probe(model, pool, idx, dev, p, seed, dedup, strip):
    was = model.training; model.eval()
    g = torch.Generator(); g.manual_seed(seed)
    st, sr, sc_, ys = [], [], [], []
    for i in range(0, len(idx), 256):
        sl = idx[i:i + 256]; b = mkb(pool, sl, dev, dedup, strip)
        Z = model.tokens(b); th, _, _ = model.gate_logits(b); tm = b['TM']
        top = model.budget_gate(th, tm, p)
        r = torch.rand(*tm.shape, generator=g).to(th.device)
        rnd = model.budget_gate(r, tm, p)
        st.append(model.score(Z, top, tm).float().cpu().numpy())
        sr.append(model.score(Z, rnd, tm).float().cpu().numpy())
        sc_.append(model.score(Z, (1.0 - top).clamp(0, 1) * tm,
                               tm).float().cpu().numpy())
        ys.append(pool['Y'][sl])
    if was:
        model.train()
    y = np.concatenate(ys)
    def A(v):
        try:
            return float(roc_auc_score(y, v))
        except ValueError:
            return float('nan')
    return A(np.concatenate(st)), A(np.concatenate(sr)), A(np.concatenate(sc_))


def train_one(pool, tr, va, te, cfg, seed, strip):
    dedup = True
    set_seed(seed); c = dict(cfg)
    model = EVIDENTPlus(c).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=c['LR'], weight_decay=c['WD'])
    pi = float(np.clip(pool['Y'][tr].mean(), 1e-4, 1 - 1e-4))
    pw = torch.tensor([(1 - pi) / pi], device=DEV)
    lp = math.log(pi / (1 - pi))
    TRIV = (1 - pi) * (-math.log(pi) - math.log(1 - pi))
    rgv = np.random.default_rng(1234)
    va_s = va if len(va) <= VAL_SUB else np.sort(rgv.choice(va, VAL_SUB, False))
    lam_nec = c['LAM_NEC']
    best = dict(score=-1e9, ep=-1, state=None)

    for ep in range(c['EPOCHS']):
        model.train(); perm = np.random.permutation(tr); dr, nb = 0.0, 0
        for i in range(0, len(perm), c['BS']):
            idx = perm[i:i + c['BS']]; b = mkb(pool, idx, DEV, dedup, strip)
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

        a_top, a_rnd, a_cmp = val_probe(model, pool, va_s, DEV,
                                        c['BUDGET_P'], seed, dedup, strip)
        if np.isfinite(a_top) and a_top > best['score']:
            best = dict(score=a_top, ep=ep + 1,
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
            idx = te[i:i + 256]; b = mkb(pool, idx, DEV, dedup, strip)
            Z = model.tokens(b); th, _, _ = model.gate_logits(b); tm = b['TM']
            top = model.budget_gate(th, tm, p)
            r = torch.rand(*tm.shape, generator=g).to(th.device)
            rnd = model.budget_gate(r, tm, p)
            st.append(model.score(Z, top, tm).float().cpu().numpy())
            sr.append(model.score(Z, rnd, tm).float().cpu().numpy())
            emp.append(model.score(Z, torch.zeros_like(tm),
                                   tm).float().cpu().numpy())
            # NOTE: RT is used only to MEASURE whether the chosen evidence is
            # negative-rated. It is never an input, so it stays even here.
            neg = torch.from_numpy(
                (pool['RT'][idx] < 0).astype(np.float32)).to(th.device)
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
                best_epoch=best['ep'], lam_nec_final=lam_nec)


# -------------------------------------------------------------------- run --
cfg = dict(CFG); cfg['BUDGET_P'] = MAIN_P; cfg['EPOCHS'] = EPOCHS
CSV = os.path.join(MASTER, 'evident_no_ratings.csv')
CSV_LOCAL = '/content/evident_no_ratings.csv'
rows = (pd.read_csv(CSV).to_dict('records') if os.path.exists(CSV)
        else (pd.read_csv(CSV_LOCAL).to_dict('records')
              if os.path.exists(CSV_LOCAL) else []))
done = {(str(r['dataset']), int(r['seed'])) for r in rows}
if done:
    print(f'[resume] {len(done)} trainings already done')

stopped = False
for DS in DATASETS:
    if stopped:
        break
    df, n_nodes = load_bitcoin(DS)
    CACHE = os.path.join(MASTER, f'poolplus_{DS}.npz')
    if os.path.exists(CACHE):
        z = np.load(CACHE); pool = {k: z[k] for k in z.files}
        pool['keep'] = pool['keep'].astype(bool)
        print(f'\n[{DS}] pool from cache')
    else:
        # Not a different pool: build_pool_plus is the SAME function the main
        # runs used, and it is deterministic (per-edge rng seeded by the edge
        # index), so the rebuilt pool is identical to the cached one.
        print(f'\n[{DS}] no cached pool -- rebuilding it with the same '
              f'deterministic builder (~2-5 min)...', flush=True)
        pool = build_pool_plus(df, n_nodes, cfg)
        try:
            np.savez_compressed(CACHE, **pool); print(f'[{DS}] pool cached')
        except OSError:
            print(f'[{DS}] pool built (could not cache it; '
                  f'it will be rebuilt next run)')
    tr, va, te = chrono_split(pool['keep'], cfg)
    print(f'[{DS}] tr={len(tr)} va={len(va)} te={len(te)} '
          f'pos_test={int(pool["Y"][te].sum())}')

    # --- proof that the stripped dims really are gone ----------------------
    b0 = mkb(pool, te[:256], DEV, True, False)
    b1 = mkb(pool, te[:256], DEV, True, True)
    kept_nf = [d for d in range(N_NF) if d not in NF_RATING_DIMS]
    kept_tf = [d for d in range(N_TF) if d not in TF_RATING_DIMS]
    assert float(b1['NF'][..., NF_RATING_DIMS].abs().max()) == 0.0
    assert float(b1['TF'][..., TF_RATING_DIMS].abs().max()) == 0.0
    assert torch.allclose(b0['NF'][..., kept_nf], b1['NF'][..., kept_nf])
    assert torch.allclose(b0['TF'][..., kept_tf], b1['TF'][..., kept_tf])
    print(f'[{DS}] verified: rating dims zeroed, structural dims untouched')

    for sd in SEEDS:
        if (DS, sd) in done:
            print(f'  [skip] {DS} seed{sd}'); continue
        if (time.time() - T0) / 60 > TIME_BUDGET_MIN:
            print(f'\n[budget] reached; re-run the cell to continue.')
            stopped = True; break
        print(f'  --- {DS} NO-RATINGS seed{sd} ---'); t1 = time.time()
        r = train_one(pool, tr, va, te, cfg, sd, strip=True)
        rows.append(dict(dataset=DS, variant='no_ratings', seed=sd,
                         budget_p=MAIN_P, epochs=EPOCHS, **r))
        print(f'    AUC={r["auc"]:.4f} AP={r["ap"]:.4f} P@100={r["p100"]:.4f} '
              f'adv={r["rationale_advantage"]:+.4f} '
              f'empty={r["empty_auc"]:.4f} ({time.time()-t1:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(CSV_LOCAL, index=False)
        try:
            pd.DataFrame(rows).to_csv(CSV, index=False)
        except OSError:
            pass

# -------------------------------------------------------------- summary ----
nd = pd.DataFrame(rows)
print('\n' + '=' * 80)
print('EVIDENT WITH vs WITHOUT RATING FEATURES  --  real labels')
print('=' * 80)
for DS in DATASETS:
    g = nd[nd.dataset == DS]
    if not len(g):
        continue
    R = REF[DS]
    sd_ = g.auc.std(ddof=1) if len(g) > 1 else float('nan')
    print(f'\n--- {DS} ---')
    print(f"{'configuration':<40}{'AUC':>18}{'AP':>9}{'P@100':>9}{'adv':>9}")
    print(f"{'EVIDENT, full inputs':<40}{R['ev']:>10.4f}{'':>8}"
          f"{R['ap']:>9.4f}{R['p100']:>9.4f}{R['adv']:>9.4f}")
    print(f"{'EVIDENT, NO rating features':<40}{g.auc.mean():>10.4f}"
          f"+/-{sd_:.4f}{g.ap.mean():>9.4f}{g.p100.mean():>9.4f}"
          f"{g.rationale_advantage.mean():>9.4f}")
    print(f"{'SAD (labels + ratings)':<40}{R['sad']:>10.4f}")
    print(f"{'TADDY (structure + time)':<40}{R['taddy']:>10.4f}")
    print(f"{'StrGNN (structure + time)':<40}{R['strgnn']:>10.4f}")
    print(f"{'degree heuristic':<40}{R['degree']:>10.4f}")
    a = g.auc.mean(); best_struct = max(R['strgnn'], R['taddy'])
    print(f"\n  cost of removing ratings : {R['ev']-a:+.4f} AUC")
    print(f"  still ahead of StrGNN/TADDY by: {a-best_struct:+.4f} AUC")
    print(f"  empty-evidence AUC       : max deviation from 0.5 = "
          f"{float((g.empty_auc-0.5).abs().max()):.4f}")
    print(f"  evidence precision lift  : {g.evidence_precision_lift.mean():.2f}x")

print('\n--- WHAT TO WRITE ---')
ok = True
for DS in DATASETS:
    g = nd[nd.dataset == DS]
    if not len(g):
        print(f'  {DS}: not finished, re-run the cell.'); ok = False; continue
    R = REF[DS]; a = g.auc.mean(); best_struct = max(R['strgnn'], R['taddy'])
    emax = float((g.empty_auc - 0.5).abs().max())
    if emax > 1e-6:
        print(f'  !! {DS}: empty-evidence AUC moved ({emax:.4f}). The '
              f'evidence-only property must hold regardless of inputs. '
              f'Investigate before reporting.')
    print(f'  {DS}: without ratings {a:.4f}; best structural detector '
          f'{best_struct:.4f}; full model {R["ev"]:.4f}.')
if ok:
    g0 = nd[nd.dataset == 'bitcoin_otc']
    if len(g0) > 1:
        t, p = stats.ttest_1samp(g0.auc, REF['bitcoin_otc']['ev'])
        print(f"\n  OTC: no-ratings vs the full model's 0.8622, "
              f"one-sample t-test p={p:.2e}")

print('\n' + '=' * 80); print('RESULTS (CSV)'); print('=' * 80)
print(nd.to_csv(index=False))
print(f'DONE in {(time.time()-T0)/60:.1f} min')
