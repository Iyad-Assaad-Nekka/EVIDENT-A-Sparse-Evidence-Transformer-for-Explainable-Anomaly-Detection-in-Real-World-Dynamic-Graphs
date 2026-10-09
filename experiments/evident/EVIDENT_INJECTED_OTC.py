#!/usr/bin/env python
# ============================================================================
# EVIDENT_INJECTED_OTC.py
#
# EVIDENT under the injected-anomaly protocol on Bitcoin-OTC.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, math, time, gzip, copy, urllib.request, subprocess
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy import stats


SEEDS      = [0, 1, 2, 3]
MAIN_P     = 0.20
EPOCHS     = 30
EVAL_EVERY = 1
VAL_SUB    = 3000
SELECT_ON  = 'auc'
DS         = 'bitcoin_otc'
INJ_RATES  = [0.05]        # matches the StrGNN control; add others later

# EVIDENT+ on Alpha, REAL labels, p=0.20 (8 seeds, already measured)
REAL_PLUS = dict(
    auc=[0.8373, 0.8676, 0.8717, 0.8562, 0.8715, 0.8784, 0.8568, 0.8585],
    ap =[0.5087, 0.5689, 0.5589, 0.5221, 0.5702, 0.5478, 0.5680, 0.5466],
    p100=[0.80, 0.88, 0.86, 0.90, 0.88, 0.84, 0.90, 0.86],
    adv=[0.1734, 0.1363, 0.1134, 0.1220, 0.1688, 0.1054, 0.1208, 0.1288])

T0 = time.time()
REPO = '/content/EVIDENT_repo'
if not os.path.isdir(REPO):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/Iyad-Assaad-Nekka/'
                    'EVIDENT-A-Sparse-Evidence-Transformer-for-Explainable-Anomaly-Detection-in-Real-World-Dynamic-Graphs.git',
                    REPO], check=True)
exec(compile(open(os.path.join(REPO, 'model.py')).read(), 'model.py', 'exec'),
     globals())
print(f'[gpu] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
MASTER = os.path.join(BENCH, 'evident_master')
os.makedirs(MASTER, exist_ok=True); os.makedirs(DATA, exist_ok=True)
if not os.path.abspath(BENCH).startswith('/content/drive'):
    print('=' * 72)
    print('!! GOOGLE DRIVE IS NOT MOUNTED -- caches and CSVs will be LOST !!')
    print("   run  from google.colab import drive; drive.mount('/content/drive')")
    print('=' * 72, flush=True)
CACHE = os.path.join(MASTER, f'poolplus_{DS}.npz')

N_NF, N_TF = 4, 3


def load_alpha_raw():
    p = os.path.join(DATA, 'bitcoinotc.csv')
    if not os.path.exists(p):
        open(p, 'wb').write(gzip.decompress(urllib.request.urlopen(
            'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz',
            timeout=300).read()))
    d = pd.read_csv(p, header=None, names=['u', 'v', 'r', 't'])
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    d['r'] = d.r.astype(np.float32) / 10.0
    return d


def reindex(d):
    ids = pd.unique(pd.concat([d.u, d.v])); m = {int(x): i for i, x in enumerate(ids)}
    d = d.copy()
    d['u'] = d.u.map(m).astype(np.int64); d['v'] = d.v.map(m).astype(np.int64)
    return d[['u', 'v', 't', 'y', 'r']].reset_index(drop=True), len(m)


def make_real():
    d = load_alpha_raw(); d['y'] = (d.r * 10 <= -5).astype(np.float32)
    return reindex(d)


def make_injected(rate, seed=0):
    """Literature convention: real edges are ALL benign; anomalies are random
    node pairs inserted at the given rate. The injected edge carries no rating
    (r=0); the current edge's own rating is never a model feature, so this
    does not leak."""
    d = load_alpha_raw(); d['y'] = 0.0
    rg = np.random.default_rng(seed)
    nodes = pd.unique(pd.concat([d.u, d.v])).astype(np.int64)
    n_inj = int(len(d) * rate)
    pos = rg.choice(len(d), n_inj, replace=False)
    inj = pd.DataFrame(dict(u=rg.choice(nodes, n_inj), v=rg.choice(nodes, n_inj),
                            t=d.t.to_numpy()[pos], y=1.0, r=0.0))
    inj = inj[inj.u != inj.v]
    d = pd.concat([d, inj], ignore_index=True)
    d = d.sort_values('t', kind='mergesort').reset_index(drop=True)
    return reindex(d)


def heuristics(df, te_idx):
    """Parameter-free baselines. No training, no features."""
    u, v, t = df.u.to_numpy(), df.v.to_numpy(), df.t.to_numpy()
    n = int(max(u.max(), v.max())) + 1
    deg = np.zeros(n); s_deg = np.zeros(len(df)); s_new = np.zeros(len(df))
    for j in range(len(df)):
        a, b = u[j], v[j]
        s_deg[j] = -(deg[a] + deg[b])
        s_new[j] = float(deg[a] == 0) + float(deg[b] == 0)
        deg[a] += 1; deg[b] += 1
    y = df.y.to_numpy()[te_idx]
    out = {}
    for nm, sc in [('degree', s_deg), ('newness', s_new)]:
        ss = sc[te_idx]
        try:
            out[nm] = dict(auc=float(roc_auc_score(y, ss)),
                           ap=float(average_precision_score(y, ss)),
                           p100=p_at_k(y, ss, 100))
        except ValueError:
            out[nm] = dict(auc=float('nan'), ap=float('nan'), p100=float('nan'))
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
    return A(np.concatenate(st)), A(np.concatenate(sr)), A(np.concatenate(sc_))


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

        a_top, a_rnd, a_cmp = val_probe(model, pool, va_s, DEV,
                                        c['BUDGET_P'], seed, dedup)
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
                best_epoch=best['ep'], lam_nec_final=lam_nec)



# -------------------------------------------------------------------- run --
from sklearn.metrics import average_precision_score
cfg = dict(CFG); cfg['BUDGET_P'] = MAIN_P; cfg['EPOCHS'] = EPOCHS
CSV = os.path.join(MASTER, f'injected_vs_real_{DS}.csv')
HCSV = os.path.join(MASTER, f'injected_vs_real_heuristics_{DS}.csv')
rows = pd.read_csv(CSV).to_dict('records') if os.path.exists(CSV) else []
hrows = pd.read_csv(HCSV).to_dict('records') if os.path.exists(HCSV) else []
_done = {(round(float(r['inj_rate']), 4), int(r['seed'])) for r in rows}
_hdone = {(str(r['labels']), str(r['method']),
           round(float(r['inj_rate']), 4) if pd.notna(r['inj_rate']) else None)
          for r in hrows}
if _done:
    print(f'[resume] {len(_done)} trainings already done, skipping them')

# ---- heuristics under REAL labels ----------------------------------------
def cached_pool(tag, mk):
    pth = os.path.join(MASTER, f'poolinj_{DS}_{tag}.npz')
    if os.path.exists(pth):
        z = np.load(pth); pl = {k: z[k] for k in z.files}
        pl['keep'] = pl['keep'].astype(bool)
        print(f'  [pool:{tag}] from cache')
        return pl
    d_, n_ = mk()
    pl = build_pool_plus(d_, n_, cfg)
    np.savez_compressed(pth, **pl)
    print(f'  [pool:{tag}] built and cached')
    return pl, d_


df_r, nn_r = make_real()
_pr = cached_pool('real', make_real)
pool_r = _pr[0] if isinstance(_pr, tuple) else _pr
tr_r, va_r, te_r = chrono_split(pool_r['keep'], cfg)
for nm, m in ({} if all(('real', x, None) in _hdone for x in ['degree', 'newness'])
              else heuristics(df_r, te_r)).items():
    hrows.append(dict(dataset=DS, labels='real', inj_rate=np.nan, method=nm, **m))
    print(f'  [heur/real ] {nm:<8} AUC={m["auc"]:.4f} AP={m["ap"]:.4f} P@100={m["p100"]:.4f}')
pd.DataFrame(hrows).to_csv(HCSV, index=False)

# ---- injected ------------------------------------------------------------
for rate in INJ_RATES:
    print(f'\n=== injection rate {rate:.0%} ===', flush=True)
    df_i, nn_i = make_injected(rate, seed=0)
    tag = f'inj{int(rate*100):02d}'
    _pi = cached_pool(tag, lambda: make_injected(rate, seed=0))
    pool_i = _pi[0] if isinstance(_pi, tuple) else _pi
    tr_i, va_i, te_i = chrono_split(pool_i['keep'], cfg)
    print(f'  [split] tr={len(tr_i)} va={len(va_i)} te={len(te_i)} '
          f'pos={pool_i["Y"][te_i].mean():.4f}')
    for nm, m in ({} if all(('injected', x, round(rate, 4)) in _hdone
                            for x in ['degree', 'newness'])
                  else heuristics(df_i, te_i)).items():
        hrows.append(dict(dataset=DS, labels='injected', inj_rate=rate,
                          method=nm, **m))
        print(f'  [heur/inj ] {nm:<8} AUC={m["auc"]:.4f} AP={m["ap"]:.4f}')
    pd.DataFrame(hrows).to_csv(HCSV, index=False)
    for sd in SEEDS:
        if (round(rate, 4), sd) in _done:
            print(f'  [skip] rate={rate:.0%} seed{sd} already done')
            continue
        t1 = time.time()
        r = train_one(pool_i, tr_i, va_i, te_i, cfg, sd, 'plus_dedup')
        rows.append(dict(dataset=DS, labels='injected', inj_rate=rate,
                         seed=sd, budget_p=MAIN_P, **r))
        print(f'  [EVIDENT+] rate={rate:.0%} seed{sd} AUC={r["auc"]:.4f} '
              f'AP={r["ap"]:.4f} P@100={r["p100"]:.4f} '
              f'adv={r["rationale_advantage"]:+.4f} empty={r["empty_auc"]:.4f} '
              f'({time.time()-t1:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)

# ------------------------------------------------------------- the table --
nd = pd.DataFrame(rows); hd = pd.DataFrame(hrows)
ra, rp, rq = (np.array(REAL_PLUS['auc']), np.array(REAL_PLUS['ap']),
              np.array(REAL_PLUS['p100']))
hr = hd[(hd.labels == 'real') & (hd.method == 'degree')].iloc[0]
hn = hd[(hd.labels == 'real') & (hd.method == 'newness')].iloc[0]

print('\n' + '=' * 86)
print(f'THE 2x2  --  {DS}, identical splits, identical metric code')
print('=' * 86)
print(f"{'':<26}{'injected 5%':>16}{'real labels':>16}{'drop':>10}")
ia = nd[nd.inj_rate == 0.05].auc if len(nd) else pd.Series(dtype=float)
print(f"{'StrGNN':<26}{0.9819:>16.4f}{0.5464:>16.4f}{0.5464-0.9819:>+10.4f}")
if len(ia):
    print(f"{'EVIDENT+':<26}{ia.mean():>16.4f}{ra.mean():>16.4f}"
          f"{ra.mean()-ia.mean():>+10.4f}")
for rate in INJ_RATES:
    q = hd[(hd.labels == 'injected') & (hd.inj_rate == rate)
           & (hd.method == 'degree')]
    qr = hd[(hd.labels == 'real') & (hd.method == 'degree')]
    if len(q) and len(qr):
        print(f"{'degree heuristic':<26}{q.iloc[0].auc:>16.4f}"
              f"{qr.iloc[0].auc:>16.4f}{qr.iloc[0].auc-q.iloc[0].auc:>+10.4f}")

print('\n--- P@100, lift over base rate ---')
if len(ia):
    g5 = nd[nd.inj_rate == 0.05]
    print(f"{'EVIDENT+ injected':<26}{g5.p100.mean():>10.4f}")
    print(f"{'EVIDENT+ real':<26}{rq.mean():>10.4f}  ({rq.mean()/0.0787:.1f}x)")
print(f"{'StrGNN injected':<26}{0.8750:>10.4f}")
print(f"{'StrGNN real':<26}{0.0967:>10.4f}  ({0.0967/0.0787:.2f}x)")

print('\n--- read ---')
if len(ia):
    if ia.mean() < 0.9819:
        print('  StrGNN BEATS EVIDENT+ on the injected benchmark and collapses')
        print('  on real labels. Report it exactly that way: the benchmark, not')
        print('  the method, is what changes the ranking.')
    else:
        pass

print('\n' + '=' * 86); print('RESULTS (CSV)'); print('=' * 86)
print(nd.to_csv(index=False)); print(); print(hd.to_csv(index=False))
print(f'DONE in {(time.time()-T0)/60:.1f} min')
