#!/usr/bin/env python
# ============================================================================
# EVIDENT_CASE_STUDY.py
#
# EVIDENT case study: evidence selected for individual test events and the
# 50-event inspection.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, sys, math, time, gzip, copy, urllib.request, subprocess, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

DS       = 'bitcoin_otc'
SEED     = 0
MAIN_P   = 0.20
EPOCHS   = 30
VAL_SUB  = 3000
N_ANOM   = 25        # anomalous test edges to dump
N_BENIGN = 25        # benign test edges to dump
N_NF, N_TF = 4, 3

T0 = time.time()
REPO = '/content/EVIDENT_repo'
if not os.path.isdir(REPO):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/Iyad-Assaad-Nekka/'
                    'EVIDENT-Explainable-Dynamic-Graph-anomaly-Detection.git',
                    REPO], check=True)
exec(compile(open(os.path.join(REPO, 'model.py')).read(), 'model.py', 'exec'),
     globals())
MASTER = os.path.join(BENCH, 'evident_master')
os.makedirs(MASTER, exist_ok=True); os.makedirs(DATA, exist_ok=True)
if not os.path.abspath(BENCH).startswith('/content/drive'):
    print('!! DRIVE NOT MOUNTED -- save the CSVs before disconnecting !!',
          flush=True)
print(f'[gpu] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')


def load_otc():
    p = os.path.join(DATA, 'bitcoinotc.csv')
    if not os.path.exists(p):
        open(p, 'wb').write(gzip.decompress(urllib.request.urlopen(
            'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz',
            timeout=300).read()))
    d = pd.read_csv(p, header=None, names=['u', 'v', 'r', 't'])
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    d['y'] = (d.r <= -5).astype(np.float32)
    d['r_raw'] = d.r.astype(np.float32)
    d['r'] = d.r.astype(np.float32) / 10.0
    ids = pd.unique(pd.concat([d.u, d.v])); m = {int(x): i for i, x in enumerate(ids)}
    d['u'] = d.u.map(m).astype(np.int64); d['v'] = d.v.map(m).astype(np.int64)
    return d[['u', 'v', 't', 'y', 'r', 'r_raw']].reset_index(drop=True), len(m)


def build_pool(df, n_nodes, cfg):
    """Same builder as the published runs, plus provenance: which real node and
    which real interaction each token slot points at."""
    pb = PoolBuilder(df, n_nodes, cfg)
    uu, vv = df.u.to_numpy(), df.v.to_numpy()
    rr = df.r.to_numpy().astype(np.float64)
    tt = df.t.to_numpy()
    M = len(df); N, K, T = cfg['N_MAX'], cfg['W'], cfg['T_MAX']
    LAB = np.zeros((M, N, 2), np.int64); DT = np.zeros((M, K), np.float32)
    TN = np.zeros((M, T), np.int64); TK = np.zeros((M, T), np.int64)
    TM = np.ones((M, T), np.float32); Y = np.zeros((M,), np.float32)
    NF = np.zeros((M, N, N_NF), np.float32); TF = np.zeros((M, K, N_TF), np.float32)
    TRATE = np.zeros((M, K), np.float32); DUPTM = np.ones((M, T), np.float32)
    NODEID = np.full((M, N), -1, np.int64)
    TSTAMP = np.zeros((M, K), np.float64)
    EVID = np.full((M, K), -1, np.int64)
    ANCH = np.zeros((M, 2), np.int64)
    keep = np.zeros((M,), bool)
    deg = np.zeros(n_nodes); rs = np.zeros(n_nodes)
    rc = np.zeros(n_nodes); rn = np.zeros(n_nodes)
    for j in range(M):
        a, b = int(uu[j]), int(vv[j]); ts = pb.t[j]
        ANCH[j] = (a, b)
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
            nf[i, 0] = math.log1p(deg[x])
            nf[i, 1] = (rs[x] / rc[x]) if rc[x] > 0 else 0.0
            nf[i, 2] = math.log1p(rn[x]); nf[i, 3] = 1.0 if deg[x] == 0 else 0.0
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
            NODEID[j] = np.array(nodes, np.int64)[ni]
            ki = rg.choice(len(dt), K, replace=True) if len(dt) < K else np.arange(K)
            DT[j] = np.log1p(np.maximum(dt, 0))[ki]
            ev_arr = np.array(ev, np.int64)
            TF[j] = tf[ki]; TRATE[j] = rr[ev_arr][ki]
            TSTAMP[j] = tt[ev_arr][ki]; EVID[j] = ev_arr[ki]
            nmap, kmap = {}, {}
            for s_, o in enumerate(ni): nmap.setdefault(int(o), s_)
            for s_, o in enumerate(ki): kmap.setdefault(int(o), s_)
            ta, tb = np.array(tn), np.array(tk)
            ok = np.array([(int(x) in nmap and int(z) in kmap) for x, z in zip(ta, tb)])
            ta, tb = ta[ok], tb[ok]
            if len(ta) == 0:
                ta, tb = np.array([0]), np.array([0])
                nmap.setdefault(0, 0); kmap.setdefault(0, 0)
            ti = rg.choice(len(ta), T, replace=True)
            TN[j] = np.array([nmap[int(x)] for x in ta[ti]])
            TK[j] = np.array([kmap[int(x)] for x in tb[ti]])
            uid = TN[j] * (K + 1) + TK[j]
            _, first = np.unique(uid, return_index=True)
            m_ = np.zeros(T, np.float32); m_[first] = 1.0
            DUPTM[j] = m_; keep[j] = True
        Y[j] = float(pb.y[j])
        deg[a] += 1; deg[b] += 1; rs[b] += rr[j]; rc[b] += 1
        if rr[j] < 0:
            rn[b] += 1
        pb.absorb(j)
        if j % 20000 == 0 and j:
            print(f'    pooled {j}/{M}', flush=True)
    m = DT[keep]; DT = (DT - m[m > 0].mean()) / (m[m > 0].std() + 1e-6)
    idx = np.where(keep)[0]; trr = idx[:int(cfg['TRAIN_FRAC'] * len(idx))]
    for arr in (NF, TF):
        mu = arr[trr].reshape(-1, arr.shape[-1]).mean(0)
        sd = arr[trr].reshape(-1, arr.shape[-1]).std(0) + 1e-6
        arr -= mu; arr /= sd
    RT = np.take_along_axis(TRATE, TK, axis=1).astype(np.float32)
    return dict(LAB=LAB, DT=DT, TN=TN, TK=TK, TM=TM, Y=Y, keep=keep, NF=NF,
                TF=TF, TRATE=TRATE, RT=RT, DUPTM=DUPTM, NODEID=NODEID,
                TSTAMP=TSTAMP, EVID=EVID, ANCH=ANCH)


class EVIDENTPlus(EVIDENT):
    def __init__(s, cfg):
        super().__init__(cfg); d = cfg['D_MODEL']
        s.nf_enc = nn.Linear(N_NF, d); s.nf_sp = nn.Linear(N_NF, d // 2)
        s.tf_enc = nn.Linear(N_TF, d); s.tf_tp = nn.Linear(N_TF, d // 2)

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


def mkb(pool, idx, dev):
    b = batch_of(pool, idx, dev)
    b['NF'] = torch.from_numpy(pool['NF'][idx]).to(dev)
    b['TF'] = torch.from_numpy(pool['TF'][idx]).to(dev)
    b['TM'] = torch.from_numpy(pool['DUPTM'][idx]).to(dev)
    return b


@torch.no_grad()
def val_auc(model, pool, idx, dev, p):
    was = model.training; model.eval(); out = []
    for i in range(0, len(idx), 256):
        sl = idx[i:i + 256]; b = mkb(pool, sl, dev)
        sc, _, _, _, _ = model(b, p=p)
        out.append(sc.float().cpu().numpy())
    if was:
        model.train()
    try:
        return float(roc_auc_score(pool['Y'][idx], np.concatenate(out)))
    except ValueError:
        return float('nan')


# --------------------------------------------------------------- train ----
cfg = dict(CFG); cfg['BUDGET_P'] = MAIN_P; cfg['EPOCHS'] = EPOCHS
df, n_nodes = load_otc()
CACHE = os.path.join(MASTER, f'casestudy_pool_{DS}.npz')
if os.path.exists(CACHE):
    z = np.load(CACHE); pool = {k: z[k] for k in z.files}
    pool['keep'] = pool['keep'].astype(bool); print('[pool] cached')
else:
    pool = build_pool(df, n_nodes, cfg)
    np.savez_compressed(CACHE, **pool); print('[pool] built')
tr, va, te = chrono_split(pool['keep'], cfg)
print(f'[split] tr={len(tr)} va={len(va)} te={len(te)}', flush=True)

set_seed(SEED)
model = EVIDENTPlus(cfg).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=cfg['LR'], weight_decay=cfg['WD'])
pi = float(np.clip(pool['Y'][tr].mean(), 1e-4, 1 - 1e-4))
pw = torch.tensor([(1 - pi) / pi], device=DEV); lp = math.log(pi / (1 - pi))
TRIV = (1 - pi) * (-math.log(pi) - math.log(1 - pi))
rgv = np.random.default_rng(1234)
va_s = va if len(va) <= VAL_SUB else np.sort(rgv.choice(va, VAL_SUB, False))
lam = cfg['LAM_NEC']; best = dict(v=-1, ep=-1, state=None)
for ep in range(EPOCHS):
    model.train(); perm = np.random.permutation(tr); dr, nb = 0.0, 0
    for i in range(0, len(perm), cfg['BS']):
        idx = perm[i:i + cfg['BS']]; b = mkb(pool, idx, DEV)
        Z = model.tokens(b); th, a, cc = model.gate_logits(b); tm = b['TM']
        g1 = model.sample_gate(th, True); g2 = model.sample_gate(th, True)
        L = F.binary_cross_entropy_with_logits(
            model.score(Z, g1, tm), b['Y'], pos_weight=pw)
        Ln = ((model.score(Z, (1 - g1).clamp(0, 1), tm) - lp) ** 2).mean()
        po = model.p_open(th); pc = torch.sigmoid(cc)
        q = po.clamp(1e-6, 1 - 1e-6)
        loss = (L + lam * Ln
                + cfg['LAM_BUD'] * (((po * tm).sum() / tm.sum().clamp(min=1)) - MAIN_P) ** 2
                + cfg['LAM_TV'] * (pc[:, 1:] - pc[:, :-1]).abs().mean()
                + cfg['LAM_STAB'] * ((g1 - g2) ** 2).mean()
                + cfg['LAM_ENT'] * (-(q * q.log() + (1 - q) * (1 - q).log()).mean()))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); dr += float(L.detach()); nb += 1
    v = val_auc(model, pool, va_s, DEV, MAIN_P)
    if np.isfinite(v) and v > best['v']:
        best = dict(v=v, ep=ep + 1,
                    state={k: t.detach().cpu().clone()
                           for k, t in model.state_dict().items()})
    if cfg['LAM_NEC'] > 0:
        lam = (max(lam * 0.5, 0.25) if dr / max(nb, 1) >= 0.98 * TRIV
               else min(lam * 1.5, 8.0))
    if ep % 10 == 9:
        print(f'   ep{ep+1} valAUC={v:.4f} (best ep{best["ep"]})', flush=True)
model.load_state_dict(best['state'])
print(f'[train] best epoch {best["ep"]}, val {best["v"]:.4f}', flush=True)

# ---------------------------------------------------------------- dump ----
raw = df.r_raw.to_numpy(); tt = df.t.to_numpy()
uu, vv = df.u.to_numpy(), df.v.to_numpy()
y = pool['Y'][te]
anom = te[np.where(y > 0.5)[0]][:N_ANOM]
ben = te[np.where(y < 0.5)[0]][:N_BENIGN]
sel_idx = np.concatenate([anom, ben])

rows = []
model.eval()
with torch.no_grad():
    for i in range(0, len(sel_idx), 32):
        idx = sel_idx[i:i + 32]; b = mkb(pool, idx, DEV)
        th, _, _ = model.gate_logits(b); tm = b['TM']
        gate = model.budget_gate(th, tm, MAIN_P)
        thn = th.cpu().numpy(); gn = gate.cpu().numpy(); tmn = tm.cpu().numpy()
        rank = (-thn).argsort(1).argsort(1)
        for r_, j in enumerate(idx):
            for t_ in range(thn.shape[1]):
                if tmn[r_, t_] < 0.5:
                    continue                      # duplicate slot, not in budget
                ns = int(pool['TN'][j, t_]); ks = int(pool['TK'][j, t_])
                e = int(pool['EVID'][j, ks])
                rows.append(dict(
                    edge_id=int(j), is_anomaly=int(pool['Y'][j] > 0.5),
                    anchor_u=int(pool['ANCH'][j, 0]), anchor_v=int(pool['ANCH'][j, 1]),
                    edge_time=float(tt[j]), edge_rating_raw=float(raw[j]),
                    token_slot=t_, token_rank=int(rank[r_, t_]),
                    gate_logit=float(thn[r_, t_]),
                    selected=int(gn[r_, t_] > 0.5),
                    counterparty=int(pool['NODEID'][j, ns]),
                    role_vs_u=int(pool['LAB'][j, ns, 0]),
                    role_vs_v=int(pool['LAB'][j, ns, 1]),
                    evidence_edge=e,
                    evidence_u=int(uu[e]) if e >= 0 else -1,
                    evidence_v=int(vv[e]) if e >= 0 else -1,
                    evidence_time=float(pool['TSTAMP'][j, ks]),
                    evidence_rating_raw=float(raw[e]) if e >= 0 else 0.0,
                    hours_before=float((tt[j] - pool['TSTAMP'][j, ks]) / 3600.0)))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(MASTER, 'case_study_tokens.csv'), index=False)

S = T.groupby(['edge_id', 'is_anomaly']).agg(
    tokens=('token_slot', 'size'), selected=('selected', 'sum'),
    sel_negative=('evidence_rating_raw', lambda s: 0),
).reset_index()
sel = T[T.selected == 1]; rej = T[T.selected == 0]
S = T.groupby(['edge_id', 'is_anomaly']).apply(
    lambda g: pd.Series(dict(
        n_tokens=len(g), n_selected=int(g.selected.sum()),
        sel_distinct_cp=g[g.selected == 1].counterparty.nunique(),
        sel_mean_rating=g[g.selected == 1].evidence_rating_raw.mean(),
        rej_mean_rating=g[g.selected == 0].evidence_rating_raw.mean(),
        sel_frac_neg=(g[g.selected == 1].evidence_rating_raw < 0).mean(),
        rej_frac_neg=(g[g.selected == 0].evidence_rating_raw < 0).mean(),
        sel_median_hours=g[g.selected == 1].hours_before.median())),
    include_groups=False).reset_index()
S.to_csv(os.path.join(MASTER, 'case_study_summary.csv'), index=False)

print('\n' + '=' * 76)
print('CASE STUDY -- selected vs rejected evidence')
print('=' * 76)
print(f'  edges inspected: {T.edge_id.nunique()}  '
      f'({int(S.is_anomaly.sum())} anomalous / {int((1-S.is_anomaly).sum())} benign)')
print(f'  tokens: {len(T)}  selected: {int(T.selected.sum())} '
      f'({100*T.selected.mean():.1f}%)')
print()
print(f"{'':<22}{'selected':>12}{'rejected':>12}{'lift':>8}")
for lab, g in T.groupby('is_anomaly'):
    s_, r_ = g[g.selected == 1], g[g.selected == 0]
    nm = 'ANOMALOUS' if lab else 'benign'
    fs, fr = (s_.evidence_rating_raw < 0).mean(), (r_.evidence_rating_raw < 0).mean()
    print(f"{nm + ' frac negative':<22}{fs:>12.4f}{fr:>12.4f}{fs/max(fr,1e-9):>8.2f}x")
    print(f"{nm + ' mean rating':<22}{s_.evidence_rating_raw.mean():>12.4f}"
          f"{r_.evidence_rating_raw.mean():>12.4f}{'':>8}")
    print(f"{nm + ' median hrs':<22}{s_.hours_before.median():>12.1f}"
          f"{r_.hours_before.median():>12.1f}{'':>8}")
print()
top = S[S.is_anomaly == 1].sort_values('sel_frac_neg', ascending=False)
print('best figure candidates (anomalous, most negative selected evidence):')
print(top.head(6)[['edge_id', 'n_selected', 'sel_distinct_cp',
                   'sel_frac_neg', 'sel_mean_rating']].round(3).to_string(index=False))
print('\n' + '=' * 76); print('COPY BOTH BLOCKS'); print('=' * 76)
print(S.round(4).to_csv(index=False))
print()
print(T[T.edge_id.isin(top.head(3).edge_id)].round(4).to_csv(index=False))
print(f'DONE in {(time.time()-T0)/60:.1f} min')
