# ============================================================================
# model.py
#
# Core library: configuration, causal evidence-pool builder, chronological
# split, metrics and training utilities shared by all experiment scripts.
#
# ============================================================================

import os, sys, math, json, random, gzip, urllib.request
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    ROOT = '/content/drive/MyDrive'
except Exception:
    ROOT = os.environ.get('EVIDENT_ROOT', '.')

BENCH = os.path.join(ROOT, 'dgad_bench')
DATA  = os.path.join(BENCH, 'data')
OUT   = os.path.join(BENCH, 'evident_uci')
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[env] device={DEV}  torch={torch.__version__}')
print(f'[env] OUT={OUT}')

# ---------------------------------------------------------------------
# Configuration. Architecture block is IDENTICAL to the Bitcoin runs --
# do not change D_MODEL / N_HEADS / N_LAYERS / RANK / W / N_MAX / T_MAX,
# ---------------------------------------------------------------------
CFG = dict(W=16, N_MAX=16, T_MAX=64, H_HOPS=1,
           TRAIN_FRAC=0.70, VAL_FRAC=0.15,
           D_MODEL=64, N_HEADS=4, N_LAYERS=2, RANK=4,
           BUDGET_P=0.20, LAM_NEC=1.0, LAM_BUD=2.0, LAM_TV=0.05,
           LAM_STAB=0.10, LAM_ENT=0.01,
           BETA0=2.0/3.0, BETA1=0.25, GAMMA=-0.1, ZETA=1.1,
           EPOCHS=40, BS=64, LR=5e-4, WD=1e-5)

# =====================================================================
# POOL CONSTRUCTION -- fixed cardinality by construction
# =====================================================================
class PoolBuilder:
    def __init__(s, df, n, cfg):
        s.u = df.u.to_numpy(); s.v = df.v.to_numpy()
        s.t = df.t.to_numpy(); s.y = df.y.to_numpy()
        s.n = n; s.c = cfg
        s.inc = [[] for _ in range(n)]
        s.nbr = [set() for _ in range(n)]

    def absorb(s, j):
        a, b = int(s.u[j]), int(s.v[j])
        s.inc[a].append(j); s.inc[b].append(j)
        s.nbr[a].add(b); s.nbr[b].add(a)

    def build(s, j):
        c = s.c; W, NM = c['W'], c['N_MAX']
        a, b = int(s.u[j]), int(s.v[j]); ts = s.t[j]
        ev = sorted(set(s.inc[a][-W:] + s.inc[b][-W:]))
        ev = [e for e in ev if s.t[e] < ts and e != j][-W:]
        ring = set()
        for e in ev:
            ring.add(int(s.u[e])); ring.add(int(s.v[e]))
        nodes = [a, b] + [x for x in sorted(ring) if x not in (a, b)]
        nodes = nodes[:NM]; pos = {x: i for i, x in enumerate(nodes)}
        lab = np.zeros((len(nodes), 2), np.int64)
        for i, x in enumerate(nodes):
            lab[i, 0] = 0 if x == a else (1 if x in s.nbr[a] else 2)
            lab[i, 1] = 0 if x == b else (1 if x in s.nbr[b] else 2)
        dt = np.array([ts - s.t[e] for e in ev], dtype=np.float64)
        tn, tk = [], []
        for k in range(len(ev) - 1, -1, -1):
            e = ev[k]
            for x in (int(s.u[e]), int(s.v[e])):
                if x in pos:
                    tn.append(pos[x]); tk.append(k)
        return dict(nodes=nodes, lab=lab, dt=dt,
                    tn=tn[:c['T_MAX']], tk=tk[:c['T_MAX']], y=int(s.y[j]))


def build_all_fixed(df, n_nodes, cfg, verbose=True):
    pb = PoolBuilder(df, n_nodes, cfg); M = len(df)
    N, K, T = cfg['N_MAX'], cfg['W'], cfg['T_MAX']
    LAB = np.zeros((M, N, 2), np.int64); DT = np.zeros((M, K), np.float32)
    TN  = np.zeros((M, T), np.int64);    TK = np.zeros((M, T), np.int64)
    TM  = np.ones((M, T), np.float32)
    Y   = np.zeros((M,), np.float32);    keep = np.zeros((M,), bool)
    for j in range(M):
        p = pb.build(j)
        nn_, kk, tt = len(p['nodes']), len(p['dt']), len(p['tn'])
        if tt >= 1 and kk >= 1:
            rg = np.random.default_rng(j)
            ni = rg.choice(nn_, N, replace=True) if nn_ < N else np.arange(N)
            LAB[j] = p['lab'][ni]
            ki = rg.choice(kk, K, replace=True) if kk < K else np.arange(K)
            DT[j] = np.log1p(np.maximum(p['dt'], 0))[ki]
            nmap, kmap = {}, {}
            for s_, o in enumerate(ni): nmap.setdefault(int(o), s_)
            for s_, o in enumerate(ki): kmap.setdefault(int(o), s_)
            tn = np.array(p['tn']); tk = np.array(p['tk'])
            ok = np.array([(int(x) in nmap and int(z) in kmap)
                           for x, z in zip(tn, tk)])
            tn, tk = tn[ok], tk[ok]
            if len(tn) == 0:
                tn, tk = np.array([0]), np.array([0])
                nmap.setdefault(0, 0); kmap.setdefault(0, 0)
            ti = rg.choice(len(tn), T, replace=True)
            TN[j] = np.array([nmap[int(x)] for x in tn[ti]])
            TK[j] = np.array([kmap[int(x)] for x in tk[ti]])
            keep[j] = True
        Y[j] = p['y']; pb.absorb(j)
        if verbose and j % 20000 == 0 and j:
            print(f'  pooled {j}/{M}', flush=True)
    m = DT[keep]; mu, sd = m[m > 0].mean(), m[m > 0].std() + 1e-6
    DT = (DT - mu) / sd
    if verbose:
        print(f'[pool] M={M} usable={keep.sum()} ({keep.mean()*100:.1f}%) '
              f'tokens=CONST({T}) nodes=CONST({N}) pos={Y[keep].mean():.4f}')
    return dict(LAB=LAB, DT=DT, TN=TN, TK=TK, TM=TM, Y=Y, keep=keep)


def chrono_split(keep, cfg):
    idx = np.where(keep)[0]; n = len(idx)
    a = int(cfg['TRAIN_FRAC'] * n)
    b = int((cfg['TRAIN_FRAC'] + cfg['VAL_FRAC']) * n)
    return idx[:a], idx[a:b], idx[b:]


def batch_of(pool, idx, dev):
    return dict(
        LAB=torch.from_numpy(pool['LAB'][idx]).to(dev),
        DT =torch.from_numpy(pool['DT'][idx]).to(dev),
        TN =torch.from_numpy(pool['TN'][idx]).to(dev),
        TK =torch.from_numpy(pool['TK'][idx]).to(dev),
        TM =torch.from_numpy(pool['TM'][idx]).to(dev),
        Y  =torch.from_numpy(pool['Y'][idx]).to(dev))

# =====================================================================
# MODEL
# =====================================================================
class GatedAttention(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.dk = d // h
        s.q = nn.Linear(d, d); s.k = nn.Linear(d, d)
        s.v = nn.Linear(d, d); s.o = nn.Linear(d, d)
    def forward(s, x, lb):
        B, L, D = x.shape; H = s.h
        q = s.q(x).view(B, L, H, s.dk).transpose(1, 2)
        k = s.k(x).view(B, L, H, s.dk).transpose(1, 2)
        v = s.v(x).view(B, L, H, s.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(s.dk) + lb[:, None, None, :]
        return s.o((att.softmax(-1) @ v).transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.n1 = nn.LayerNorm(d); s.a = GatedAttention(d, h)
        s.n2 = nn.LayerNorm(d)
        s.f = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
    def forward(s, x, lb):
        x = x + s.a(s.n1(x), lb)
        return x + s.f(s.n2(x))


class EVIDENT(nn.Module):
    def __init__(s, cfg):
        super().__init__(); d = cfg['D_MODEL']; s.c = cfg
        s.role_enc = nn.Embedding(9, d);  s.time_enc = nn.Linear(1, d)
        s.role_sp  = nn.Embedding(9, d // 2); s.time_tp = nn.Linear(1, d // 2)
        s.tgt = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        s.f_sp = nn.Sequential(nn.Linear(d // 2, d // 2), nn.GELU(),
                               nn.Linear(d // 2, 1))
        s.f_tp = nn.Sequential(nn.Linear(d // 2, d // 2), nn.GELU(),
                               nn.Linear(d // 2, 1))
        r = cfg['RANK']
        s.P = nn.Linear(d // 2, r, bias=False)
        s.Q = nn.Linear(d // 2, r, bias=False)
        s.blocks = nn.ModuleList([Block(d, cfg['N_HEADS'])
                                  for _ in range(cfg['N_LAYERS'])])
        s.norm = nn.LayerNorm(d)
        s.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def _rid(s, lab): return lab[..., 0] * 3 + lab[..., 1]

    def gate_logits(s, b):
        LAB, DT, TN, TK = b['LAB'], b['DT'], b['TN'], b['TK']
        zs = s.role_sp(s._rid(LAB)); zt = s.time_tp(DT.unsqueeze(-1))
        a = s.f_sp(zs).squeeze(-1); c = s.f_tp(zt).squeeze(-1)
        Pn = s.P(zs); Qk = s.Q(zt)
        an = torch.gather(a, 1, TN); ck = torch.gather(c, 1, TK)
        pn = torch.gather(Pn, 1, TN.unsqueeze(-1).expand(-1, -1, Pn.size(-1)))
        qk = torch.gather(Qk, 1, TK.unsqueeze(-1).expand(-1, -1, Qk.size(-1)))
        return an + ck + (pn * qk).sum(-1), a, c

    def sample_gate(s, th, train):
        g0, z0, be = s.c['GAMMA'], s.c['ZETA'], s.c['BETA1']
        if train:
            u = torch.rand_like(th).clamp(1e-6, 1 - 1e-6)
            x = torch.sigmoid((torch.log(u) - torch.log(1 - u) + th) / be)
        else:
            x = torch.sigmoid(th)
        return (x * (z0 - g0) + g0).clamp(0, 1)

    def p_open(s, th):
        return torch.sigmoid(th - s.c['BETA1'] *
                             math.log(-s.c['GAMMA'] / s.c['ZETA']))

    def budget_gate(s, th, tm, p=None):
        p = s.c['BUDGET_P'] if p is None else p
        t2 = th.masked_fill(tm < 0.5, -1e9)
        k = (tm.sum(1, keepdim=True) * p).ceil().clamp(min=1)
        rank = t2.argsort(1, descending=True).argsort(1).float()
        return (rank < k).float() * tm

    def tokens(s, b):
        LAB, DT, TN, TK = b['LAB'], b['DT'], b['TN'], b['TK']
        zn = s.role_enc(s._rid(LAB)); zk = s.time_enc(DT.unsqueeze(-1))
        zn = torch.gather(zn, 1, TN.unsqueeze(-1).expand(-1, -1, zn.size(-1)))
        zk = torch.gather(zk, 1, TK.unsqueeze(-1).expand(-1, -1, zk.size(-1)))
        return zn + zk

    def score(s, Z, gate, tm):
        """TRUE REMOVAL. At eval a closed gate zeroes the value vector AND
        sets the attention bias to -inf, so the token leaves the softmax
        denominator entirely. Flooring instead of removing is the defect
        that produced empty-mask AUC 0.7975 -- see integrity check."""
        B = Z.size(0); g = (gate * tm).clamp(0.0, 1.0)
        x = torch.cat([s.tgt.expand(B, -1, -1), Z * g.unsqueeze(-1)], 1)
        ones = torch.zeros(B, 1, device=Z.device)
        if s.training:
            lb = torch.cat([ones, g.clamp_min(1e-6).log()], 1)
        else:
            lg = torch.where(g > 0, g.clamp_min(1e-30).log(),
                             torch.full_like(g, float('-inf')))
            lb = torch.cat([ones, lg], 1)
        for blk in s.blocks:
            x = blk(x, lb)
        return s.head(s.norm(x[:, 0])).squeeze(-1)

    def forward(s, b, gate=None, p=None):
        Z = s.tokens(b); th, a, c = s.gate_logits(b); tm = b['TM']
        if gate is None:
            gate = s.sample_gate(th, True) if s.training \
                   else s.budget_gate(th, tm, p)
        return s.score(Z, gate, tm), th, gate, a, c

# =====================================================================
# METRICS
# =====================================================================
def p_at_k(y, sc, k=100):
    k = min(k, len(y))
    return float(y[np.argsort(-sc)[:k]].mean())

def metrics(y, sc):
    return dict(auc=float(roc_auc_score(y, sc)),
                ap=float(average_precision_score(y, sc)),
                p100=p_at_k(y, sc, 100))

@torch.no_grad()
def infer(model, pool, idx, dev, gate_fn=None, p=None, bs=256):
    model.eval(); out = []
    for i in range(0, len(idx), bs):
        b = batch_of(pool, idx[i:i + bs], dev)
        if gate_fn is None:
            sc, _, _, _, _ = model(b, p=p)
        else:
            _, th, _, _, _ = model(b, gate=torch.ones_like(b['TM']))
            sc = model.score(model.tokens(b), gate_fn(th, b['TM']), b['TM'])
        out.append(sc.float().cpu().numpy())
    return np.concatenate(out), pool['Y'][idx]

# =====================================================================
# TRAINING
# =====================================================================
def train(pool, cfg, dev, seed=0, tr=None, va=None, te=None, verbose=True):
    set_seed(seed)
    if tr is None:
        tr, va, te = chrono_split(pool['keep'], cfg)
    model = EVIDENT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['LR'],
                            weight_decay=cfg['WD'])
    pi = float(pool['Y'][tr].mean())
    pi = min(max(pi, 1e-4), 1 - 1e-4)
    H_pi = -(pi * math.log(pi) + (1 - pi) * math.log(1 - pi))
    logit_pi = math.log(pi / (1 - pi))
    pw = torch.tensor([(1 - pi) / pi], device=dev)
    lam_nec = cfg['LAM_NEC']
    if verbose:
        print(f'  [train] pi={pi:.4f} H(pi)={H_pi:.4f} pos_weight={pw.item():.1f}')

    for ep in range(cfg['EPOCHS']):
        model.train()
        perm = np.random.permutation(tr)
        det_run, n_bt = 0.0, 0
        for i in range(0, len(perm), cfg['BS']):
            b = batch_of(pool, perm[i:i + cfg['BS']], dev)
            Z = model.tokens(b); th, a, c = model.gate_logits(b); tm = b['TM']
            g1 = model.sample_gate(th, True)
            g2 = model.sample_gate(th, True)
            sr = model.score(Z, g1, tm)
            sc_ = model.score(Z, (1.0 - g1).clamp(0, 1), tm)

            L_det = F.binary_cross_entropy_with_logits(sr, b['Y'],
                                                       pos_weight=pw)
            L_nec = ((sc_ - logit_pi) ** 2).mean()
            po = model.p_open(th)
            L_bud = (((po * tm).sum() / tm.sum()) - cfg['BUDGET_P']) ** 2
            pc = torch.sigmoid(c)
            L_tv = (pc[:, 1:] - pc[:, :-1]).abs().mean()
            L_stab = ((g1 - g2) ** 2).mean()
            q = model.p_open(th).clamp(1e-6, 1 - 1e-6)
            L_ent = -(q * q.log() + (1 - q) * (1 - q).log()).mean()

            loss = (L_det + lam_nec * L_nec + cfg['LAM_BUD'] * L_bud
                    + cfg['LAM_TV'] * L_tv + cfg['LAM_STAB'] * L_stab
                    + cfg['LAM_ENT'] * L_ent)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            det_run += float(L_det.detach()); n_bt += 1

        # --- degeneracy control ---------------------------------------
        # Trivial-predictor loss is the ENTROPY of the prior. Using
        # -log(1-pi) instead fires every epoch on a low-rate artefact and
        # silently switches the necessity term off for the whole run.
        det_ep = det_run / max(n_bt, 1)
        if cfg['LAM_NEC'] > 0:
            if det_ep >= 0.98 * H_pi:
                lam_nec = max(lam_nec * 0.5, 0.25)
            else:
                scc, yv = infer(model, pool, va, dev,
                                gate_fn=lambda th, tm: 1.0 -
                                model.budget_gate(th, tm))
                try:
                    cauc = roc_auc_score(yv, scc)
                    if abs(cauc - 0.5) > 0.10:
                        lam_nec = min(lam_nec * 1.5, 8.0)
                except ValueError:
                    cauc = float('nan')
        if verbose and (ep % 10 == 9 or ep == 0):
            print(f'    ep{ep+1:>3d} L_det={det_ep:.4f} lam_nec={lam_nec:.3f}',
                  flush=True)

    sc, y = infer(model, pool, te, dev)
    m = metrics(y, sc)
    if verbose:
        print(f'  [test] AUC={m["auc"]:.4f} AP={m["ap"]:.4f} '
              f'P@100={m["p100"]:.4f}')
    return model, m, dict(tr=tr, va=va, te=te, pi=pi)

# =====================================================================
# MATCHED-DENSITY PROBE  (C1 / C2 / C3)
# =====================================================================
@torch.no_grad()
def matched_probe(model, pool, te, dev, p, bs=256, seed=0):
    """Every compared mask has IDENTICAL cardinality, so differences are
    information content and not density. Masks:
      empty        -- integrity check, must sit at chance
      bottom_k     -- lowest-theta tokens          (C1 vs top_k)
      disjoint_rnd -- random k drawn OUTSIDE top_k (C2 vs top_k)
      random       -- random k from the full pool  (C3 vs top_k)
      top_k        -- the rationale
    """
    model.eval(); g = torch.Generator(device='cpu'); g.manual_seed(seed)
    keys = ['empty', 'bottom_k', 'disjoint_rnd', 'random', 'top_k']
    acc = {k: [] for k in keys}; ys = []
    for i in range(0, len(te), bs):
        idx = te[i:i + bs]; b = batch_of(pool, idx, dev)
        Z = model.tokens(b); th, _, _ = model.gate_logits(b); tm = b['TM']
        B, T = tm.shape
        k = int(math.ceil(p * T))
        top = model.budget_gate(th, tm, p)
        low = model.budget_gate(-th, tm, p)
        r = torch.rand(B, T, generator=g).to(th.device)
        rnd = model.budget_gate(r, tm, p)
        r2 = r.masked_fill(top > 0.5, -1e9)
        dis = model.budget_gate(r2, tm, p)
        masks = dict(empty=torch.zeros_like(tm), bottom_k=low,
                     disjoint_rnd=dis, random=rnd, top_k=top)
        for kk, mk in masks.items():
            acc[kk].append(model.score(Z, mk, tm).float().cpu().numpy())
        ys.append(pool['Y'][idx])
    y = np.concatenate(ys)
    out = {}
    for kk in keys:
        sc = np.concatenate(acc[kk])
        out[kk] = metrics(y, sc) if kk != 'empty' else \
            dict(auc=float(roc_auc_score(y, sc)),
                 ap=float(average_precision_score(y, sc)), p100=0.0)
    return out

pass
