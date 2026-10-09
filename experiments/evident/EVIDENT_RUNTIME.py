#!/usr/bin/env python
# ============================================================================
# EVIDENT_RUNTIME.py
#
# EVIDENT runtime and memory measurements (inference time with and without
# explanation extraction).
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
BUDGETS  = [0.05, 0.10, 0.20, 0.33, 0.50, 1.00]
N_TIME   = 4000
REPEATS  = 3
N_NF, N_TF = 4, 3

T0 = time.time()
REPO = '/content/EVIDENT_repo'
if not os.path.isdir(REPO):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/Iyad-Assaad-Nekka/'
                    'EVIDENT-A-Sparse-Evidence-Transformer-for-Explainable-Anomaly-Detection-in-Real-World-Dynamic-Graphs.git',
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

# ------------------------------------------------------------- runtime ----
n_par = sum(p.numel() for p in model.parameters())
print(f'\n[model] parameters: {n_par:,}')
te_s = te[:min(N_TIME, len(te))]
rows = []


@torch.no_grad()
def timed(dev, p, gated, bs=256):
    model.to(dev); model.eval()
    if dev == 'cuda':
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for i in range(0, min(512, len(te_s)), bs):
        b = mkb(pool, te_s[i:i + bs], dev)
        if gated:
            model(b, p=p)
        else:
            model.score(model.tokens(b), b['TM'], b['TM'])
    if dev == 'cuda':
        torch.cuda.synchronize()
    best_t = 1e9
    for _ in range(REPEATS):
        t0 = time.time()
        for i in range(0, len(te_s), bs):
            b = mkb(pool, te_s[i:i + bs], dev)
            if gated:
                model(b, p=p)
            else:
                model.score(model.tokens(b), b['TM'], b['TM'])
        if dev == 'cuda':
            torch.cuda.synchronize()
        best_t = min(best_t, time.time() - t0)
    peak = (torch.cuda.max_memory_allocated() / 1e6 if dev == 'cuda'
            else float('nan'))
    return best_t * 1000 / len(te_s), peak


@torch.no_grad()
def mean_k(p):
    tot, n = 0.0, 0
    for i in range(0, len(te_s), 256):
        b = mkb(pool, te_s[i:i + 256], DEV)
        th, _, _ = model.gate_logits(b)
        tot += float(model.budget_gate(th, b['TM'], p).sum()); n += len(b['TM'])
    return tot / max(n, 1)


devs = ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']
ks = {p: mean_k(p) for p in BUDGETS}
for dev in devs:
    for p in BUDGETS:
        ms, peak = timed(dev, p, True)
        rows.append(dict(device=dev, budget_p=p, gated=True, ms_per_edge=ms,
                         peak_mem_mb=peak, tokens_selected=ks[p]))
        print(f'   [{dev}] p={p:.2f} gated  {ms:.4f} ms/edge  k={ks[p]:.1f}',
              flush=True)
    ms, peak = timed(dev, 1.0, False)
    rows.append(dict(device=dev, budget_p=np.nan, gated=False, ms_per_edge=ms,
                     peak_mem_mb=peak, tokens_selected=np.nan))
    print(f'   [{dev}] score only      {ms:.4f} ms/edge', flush=True)
model.to(DEV)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(MASTER, 'runtime_bitcoin_otc.csv'), index=False)

print('\n' + '=' * 76)
print('RUNTIME -- bitcoin_otc, EVIDENT, real labels')
print('=' * 76)
for dev in devs:
    g = R[(R.device == dev) & R.gated]
    u = R[(R.device == dev) & (~R.gated)]
    print(f'\n  {dev.upper()}')
    print(f"{'   budget p':<14}{'ms/edge':>12}{'edges/s':>12}{'tokens':>9}")
    for _, r in g.iterrows():
        print(f"{'   ' + format(r.budget_p, '.2f'):<14}{r.ms_per_edge:>12.4f}"
              f"{1000/r.ms_per_edge:>12.0f}{r.tokens_selected:>9.1f}")
    if len(u):
        uu_ = u.iloc[0].ms_per_edge
        print(f"{'   score only':<14}{uu_:>12.4f}{1000/uu_:>12.0f}{'--':>9}")
        ov = 100 * (g.ms_per_edge.mean() - uu_) / uu_
        print(f'\n   explanation overhead: {ov:+.1f}%')
    if dev == 'cuda' and np.isfinite(g.peak_mem_mb).any():
        print(f'   peak GPU memory: {g.peak_mem_mb.max():.1f} MB')

g0 = R[(R.device == devs[0]) & R.gated]
spread = g0.ms_per_edge.max() / g0.ms_per_edge.min()
print(f'\n  latency spread across budgets: {spread:.2f}x')
print('  -> FLAT: the gate masks rather than prunes. Sparsity buys'
      if spread < 1.2 else '  -> varies with budget; report the curve.')
if spread < 1.2:
    pass
print(f'\n  parameters: {n_par:,}   best epoch: {best["ep"]}')
print('\n' + '=' * 76); print('RESULTS (CSV)'); print('=' * 76)
print(R.round(5).to_csv(index=False))
print(f'DONE in {(time.time()-T0)/60:.1f} min')
