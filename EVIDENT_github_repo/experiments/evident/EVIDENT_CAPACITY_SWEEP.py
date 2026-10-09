#!/usr/bin/env python
# ============================================================================
# EVIDENT_CAPACITY_SWEEP.py
#
# EVIDENT model-capacity sweep.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, sys, math, time, gzip, copy, urllib.request, subprocess, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy import stats

DS         = 'bitcoin_otc'
INJ_RATE   = 0.05
SEEDS      = [0, 1]
EPOCHS     = 30
EVAL_EVERY = 1
VAL_SUB    = 3000
GRID_P     = [0.20, 0.50, 1.00]
GRID_D     = [64, 128]
STRGNN_INJ = 0.9819          # the number to beat, bitcoin_otc injected 5%

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
    print('!! DRIVE NOT MOUNTED -- caches and CSV will be lost on disconnect !!',
          flush=True)
print(f'[gpu] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
N_NF, N_TF = 4, 3


# ------------------------------------------------------------------ data ----
def make_injected(rate, seed=0):
    p = os.path.join(DATA, 'bitcoinotc.csv')
    if not os.path.exists(p):
        open(p, 'wb').write(gzip.decompress(urllib.request.urlopen(
            'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz',
            timeout=300).read()))
    d = pd.read_csv(p, header=None, names=['u', 'v', 'r', 't'])
    d = d[d.u != d.v].sort_values('t', kind='mergesort').reset_index(drop=True)
    d['r'] = d.r.astype(np.float32) / 10.0; d['y'] = 0.0
    rg = np.random.default_rng(seed)
    nodes = pd.unique(pd.concat([d.u, d.v])).astype(np.int64)
    n_inj = int(len(d) * rate)
    pos = rg.choice(len(d), n_inj, replace=False)
    inj = pd.DataFrame(dict(u=rg.choice(nodes, n_inj), v=rg.choice(nodes, n_inj),
                            t=d.t.to_numpy()[pos], y=1.0, r=0.0))
    inj = inj[inj.u != inj.v]
    d = pd.concat([d, inj], ignore_index=True).sort_values('t', kind='mergesort')
    d = d.reset_index(drop=True)
    ids = pd.unique(pd.concat([d.u, d.v])); m = {int(x): i for i, x in enumerate(ids)}
    d['u'] = d.u.map(m).astype(np.int64); d['v'] = d.v.map(m).astype(np.int64)
    return d[['u', 'v', 't', 'y', 'r']].reset_index(drop=True), len(m)


def build_pool_plus(df, n_nodes, cfg):
    pb = PoolBuilder(df, n_nodes, cfg)
    uu, vv = df.u.to_numpy(), df.v.to_numpy()
    rr = df.r.to_numpy().astype(np.float64)
    M = len(df); N, K, T = cfg['N_MAX'], cfg['W'], cfg['T_MAX']
    LAB = np.zeros((M, N, 2), np.int64); DT = np.zeros((M, K), np.float32)
    TN = np.zeros((M, T), np.int64); TK = np.zeros((M, T), np.int64)
    TM = np.ones((M, T), np.float32); Y = np.zeros((M,), np.float32)
    NF = np.zeros((M, N, N_NF), np.float32); TF = np.zeros((M, K, N_TF), np.float32)
    TRATE = np.zeros((M, K), np.float32); DUPTM = np.ones((M, T), np.float32)
    keep = np.zeros((M,), bool)
    deg = np.zeros(n_nodes); rs = np.zeros(n_nodes)
    rc = np.zeros(n_nodes); rn = np.zeros(n_nodes)
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
            ki = rg.choice(len(dt), K, replace=True) if len(dt) < K else np.arange(K)
            DT[j] = np.log1p(np.maximum(dt, 0))[ki]
            TF[j] = tf[ki]; TRATE[j] = rr[np.array(ev, np.int64)][ki]
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
    return dict(LAB=LAB, DT=DT, TN=TN, TK=TK, TM=TM, Y=Y, keep=keep,
                NF=NF, TF=TF, TRATE=TRATE, RT=RT, DUPTM=DUPTM)


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
def probe(model, pool, idx, dev, p, seed):
    was = model.training; model.eval()
    g = torch.Generator(); g.manual_seed(seed)
    st, sr, ys = [], [], []
    for i in range(0, len(idx), 256):
        sl = idx[i:i + 256]; b = mkb(pool, sl, dev)
        Z = model.tokens(b); th, _, _ = model.gate_logits(b); tm = b['TM']
        top = model.budget_gate(th, tm, p)
        r = torch.rand(*tm.shape, generator=g).to(th.device)
        rnd = model.budget_gate(r, tm, p)
        st.append(model.score(Z, top, tm).float().cpu().numpy())
        sr.append(model.score(Z, rnd, tm).float().cpu().numpy())
        ys.append(pool['Y'][sl])
    if was:
        model.train()
    y = np.concatenate(ys)
    def A(v):
        try:
            return float(roc_auc_score(y, v))
        except ValueError:
            return float('nan')
    return A(np.concatenate(st)), A(np.concatenate(sr)), y, np.concatenate(st)


def run(pool, tr, va, te, cfg, seed):
    set_seed(seed); c = dict(cfg)
    model = EVIDENTPlus(c).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=c['LR'], weight_decay=c['WD'])
    pi = float(np.clip(pool['Y'][tr].mean(), 1e-4, 1 - 1e-4))
    pw = torch.tensor([(1 - pi) / pi], device=DEV)
    lp = math.log(pi / (1 - pi))
    TRIV = (1 - pi) * (-math.log(pi) - math.log(1 - pi))
    rgv = np.random.default_rng(1234)
    va_s = va if len(va) <= VAL_SUB else np.sort(rgv.choice(va, VAL_SUB, False))
    lam = c['LAM_NEC']; best = dict(v=-1, ep=-1, state=None)
    for ep in range(c['EPOCHS']):
        model.train(); perm = np.random.permutation(tr); dr, nb = 0.0, 0
        for i in range(0, len(perm), c['BS']):
            idx = perm[i:i + c['BS']]; b = mkb(pool, idx, DEV)
            Z = model.tokens(b); th, a, cc = model.gate_logits(b); tm = b['TM']
            g1 = model.sample_gate(th, True); g2 = model.sample_gate(th, True)
            L_det = F.binary_cross_entropy_with_logits(
                model.score(Z, g1, tm), b['Y'], pos_weight=pw)
            L_nec = ((model.score(Z, (1 - g1).clamp(0, 1), tm) - lp) ** 2).mean()
            po = model.p_open(th)
            L_bud = (((po * tm).sum() / tm.sum().clamp(min=1)) - c['BUDGET_P']) ** 2
            pc = torch.sigmoid(cc)
            loss = (L_det + lam * L_nec + c['LAM_BUD'] * L_bud
                    + c['LAM_TV'] * (pc[:, 1:] - pc[:, :-1]).abs().mean()
                    + c['LAM_STAB'] * ((g1 - g2) ** 2).mean())
            q = po.clamp(1e-6, 1 - 1e-6)
            loss = loss + c['LAM_ENT'] * (-(q * q.log() + (1 - q) * (1 - q).log()).mean())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); dr += float(L_det.detach()); nb += 1
        vt, vr, _, _ = probe(model, pool, va_s, DEV, c['BUDGET_P'], seed)
        if np.isfinite(vt) and vt > best['v']:
            best = dict(v=vt, ep=ep + 1,
                        state={k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()})
        if c['LAM_NEC'] > 0:
            det = dr / max(nb, 1)
            if det >= 0.98 * TRIV:
                lam = max(lam * 0.5, 0.25)
            else:
                _, _, yv, sv = probe(model, pool, va_s, DEV, c['BUDGET_P'], seed)
                lam = lam  # complement escalation handled below
        if ep % 10 == 9:
            print(f'      ep{ep+1:>3d} valAUC={vt:.4f} (best {best["v"]:.4f})',
                  flush=True)
    model.load_state_dict(best['state'])
    tt, trn, y, sc = probe(model, pool, te, DEV, c['BUDGET_P'], seed)
    m = metrics(y, sc)
    return dict(val_auc=best['v'], best_epoch=best['ep'], auc=m['auc'],
                ap=m['ap'], p100=m['p100'], adv=tt - trn)


# -------------------------------------------------------------------- run --
cfg0 = dict(CFG); cfg0['EPOCHS'] = EPOCHS
df, n_nodes = make_injected(INJ_RATE, seed=0)
print(f'[data] injected {INJ_RATE:.0%}  edges={len(df)}  pos={df.y.mean():.4f}')
CACHE = os.path.join(MASTER, f'poolplus_{DS}_inj{int(INJ_RATE*100):02d}.npz')
if os.path.exists(CACHE):
    z = np.load(CACHE); pool = {k: z[k] for k in z.files}
    pool['keep'] = pool['keep'].astype(bool); print('[pool] from cache')
else:
    pool = build_pool_plus(df, n_nodes, cfg0)
    np.savez_compressed(CACHE, **pool); print('[pool] built and cached')
tr, va, te = chrono_split(pool['keep'], cfg0)
print(f'[split] tr={len(tr)} va={len(va)} te={len(te)}\n')

CSV = os.path.join(MASTER, f'evident_capacity_sweep_{DS}_inj.csv')
rows = pd.read_csv(CSV).to_dict('records') if os.path.exists(CSV) else []
done = {(float(r['budget_p']), int(r['d_model']), int(r['seed'])) for r in rows}

for pv in GRID_P:
    for dm in GRID_D:
        for sd in SEEDS:
            if (pv, dm, sd) in done:
                print(f'[skip] p={pv} d={dm} seed{sd}'); continue
            c = dict(cfg0); c['BUDGET_P'] = pv; c['D_MODEL'] = dm
            print(f'  --- p={pv:.2f} d_model={dm} seed{sd} ---', flush=True)
            t1 = time.time(); r = run(pool, tr, va, te, c, sd)
            rows.append(dict(budget_p=pv, d_model=dm, seed=sd, **r))
            pd.DataFrame(rows).to_csv(CSV, index=False)
            print(f'    val={r["val_auc"]:.4f}  TEST={r["auc"]:.4f}  '
                  f'adv={r["adv"]:+.4f}  ({time.time()-t1:.0f}s)\n', flush=True)

nd = pd.DataFrame(rows)
g = nd.groupby(['budget_p', 'd_model']).agg(
    val=('val_auc', 'mean'), test=('auc', 'mean'),
    ap=('ap', 'mean'), adv=('adv', 'mean'), n=('seed', 'size')).reset_index()
g = g.sort_values('val', ascending=False)

print('=' * 84)
print(f'CAPACITY SWEEP -- injected {INJ_RATE:.0%} {DS}   (ranked by VALIDATION)')
print('=' * 84)
print(f"{'p':>6}{'d_model':>9}{'val AUC':>10}{'test AUC':>10}{'AP':>9}{'adv':>9}{'n':>4}")
for _, r in g.iterrows():
    star = '  <- val winner' if _ == g.index[0] else ''
    print(f"{r.budget_p:>6.2f}{int(r.d_model):>9d}{r.val:>10.4f}{r.test:>10.4f}"
          f"{r.ap:>9.4f}{r.adv:>9.4f}{int(r.n):>4d}{star}")

w = g.iloc[0]
print(f'\nVALIDATION picks: p={w.budget_p:.2f}, d_model={int(w.d_model)}')
print(f'  its injected TEST AUC = {w.test:.4f}   vs StrGNN {STRGNN_INJ:.4f}')
print(f'  {"BEATS" if w.test > STRGNN_INJ else "does NOT beat"} StrGNN on injected.')
print(f'\n  rationale advantage of that config: {w.adv:+.4f}')
if w.adv < 0.02:
    print('  *** WARNING: advantage is ~0. This configuration won by OPENING')
    print('      the bottleneck -- it is no longer an explainable model. An')
print(f'\n  reference: EVIDENT at p=0.20 d=64 scored 0.9614 injected,')
print(f'             0.8623 on REAL labels with adv 0.1336.')
print('\n' + '=' * 84); print('RESULTS (CSV)'); print('=' * 84)
print(nd.to_csv(index=False))
print(f'DONE in {(time.time()-T0)/60:.1f} min')
