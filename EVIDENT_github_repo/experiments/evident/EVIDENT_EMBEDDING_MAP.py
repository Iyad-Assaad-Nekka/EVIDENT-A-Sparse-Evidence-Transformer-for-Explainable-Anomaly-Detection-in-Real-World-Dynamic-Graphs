#!/usr/bin/env python
# ============================================================================
# EVIDENT_EMBEDDING_MAP.py
#
# EVIDENT: t-SNE map of the representation the anomaly score is read from.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, math, time, gzip, copy, urllib.request, subprocess
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from scipy import stats

SEEDS      = [0]
MAIN_P     = 0.20
EPOCHS     = 30
VAL_SUB    = 3000
DATASETS   = ['bitcoin_otc', 'bitcoin_alpha']
N_NF, N_TF = 4, 3
NF_RATING_DIMS, TF_RATING_DIMS = [1, 2], [0]     # unused here (strip=False)
PAPER_SEED0 = {'bitcoin_otc': 0.8373, 'bitcoin_alpha': 0.7414}
MAX_NORMAL = 900          # normals kept in the figure; all anomalies are kept
TSNE_PERP  = 30

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
print('EVIDENT EMBEDDING MAP  --  t-SNE of the evidence-only representation')
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
def last_scalar_linear(model):
    """The last nn.Linear with a single output: the layer that emits the
    score. Its input is the representation we want."""
    cand = [m for m in model.modules()
            if isinstance(m, nn.Linear) and m.out_features == 1]
    if not cand:
        raise SystemExit('  No scalar Linear found in the scoring head. '
                         'Print the model and pick the layer by hand.')
    return cand[-1]


CSV = os.path.join(MASTER, 'evident_embedding_map.csv')
CSV_LOCAL = '/content/evident_embedding_map.csv'
rows = (pd.read_csv(CSV).to_dict('records') if os.path.exists(CSV)
        else (pd.read_csv(CSV_LOCAL).to_dict('records')
              if os.path.exists(CSV_LOCAL) else []))
done = {str(r['dataset']) for r in rows}
if done:
    print(f'[resume] already mapped: {sorted(done)}')

for DS in DATASETS:
    if DS in done:
        print(f'[skip] {DS}'); continue
    cfg = dict(CFG); cfg['BUDGET_P'] = MAIN_P; cfg['EPOCHS'] = EPOCHS
    df, n_nodes = load_bitcoin(DS)
    CACHE = os.path.join(MASTER, f'poolplus_{DS}.npz')
    if os.path.exists(CACHE):
        z = np.load(CACHE); pool = {k: z[k] for k in z.files}
        pool['keep'] = pool['keep'].astype(bool)
        print(f'\n[{DS}] pool from cache')
    else:
        print(f'\n[{DS}] no cached pool -- rebuilding (~2-5 min)...', flush=True)
        pool = build_pool_plus(df, n_nodes, cfg)
        try:
            np.savez_compressed(CACHE, **pool)
        except OSError:
            pass
    tr, va, te = chrono_split(pool['keep'], cfg)
    print(f'[{DS}] tr={len(tr)} va={len(va)} te={len(te)}')

    #      train_one() does not hand back the model, so the loop is inlined
    #      here with the same seed, same data and same schedule.
    print(f'[{DS}] training seed 0 (~3-4 min) ...', flush=True)
    t1 = time.time()
    set_seed(0)
    model = EVIDENTPlus(cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['LR'],
                            weight_decay=cfg['WD'])
    pi = float(np.clip(pool['Y'][tr].mean(), 1e-4, 1 - 1e-4))
    pw = torch.tensor([(1 - pi) / pi], device=DEV)
    lp = math.log(pi / (1 - pi))
    TRIV = (1 - pi) * (-math.log(pi) - math.log(1 - pi))
    rgv = np.random.default_rng(1234)
    va_s = va if len(va) <= VAL_SUB else np.sort(rgv.choice(va, VAL_SUB, False))
    lam_nec = cfg['LAM_NEC']; best = dict(score=-1e9, ep=-1, state=None)
    for ep in range(cfg['EPOCHS']):
        model.train(); perm = np.random.permutation(tr); dr, nb = 0.0, 0
        for i in range(0, len(perm), cfg['BS']):
            idx = perm[i:i + cfg['BS']]; b = mkb(pool, idx, DEV, True, False)
            Z = model.tokens(b); th, a, cc = model.gate_logits(b); tm = b['TM']
            g1 = model.sample_gate(th, True); g2 = model.sample_gate(th, True)
            L_det = F.binary_cross_entropy_with_logits(
                model.score(Z, g1, tm), b['Y'], pos_weight=pw)
            L_nec = ((model.score(Z, (1 - g1).clamp(0, 1), tm) - lp) ** 2).mean()
            po = model.p_open(th)
            L_bud = (((po * tm).sum() / tm.sum().clamp(min=1))
                     - cfg['BUDGET_P']) ** 2
            pc = torch.sigmoid(cc)
            L_tv = (pc[:, 1:] - pc[:, :-1]).abs().mean()
            L_st = ((g1 - g2) ** 2).mean()
            q = po.clamp(1e-6, 1 - 1e-6)
            L_en = -(q * q.log() + (1 - q) * (1 - q).log()).mean()
            loss = (L_det + lam_nec * L_nec + cfg['LAM_BUD'] * L_bud
                    + cfg['LAM_TV'] * L_tv + cfg['LAM_STAB'] * L_st
                    + cfg['LAM_ENT'] * L_en)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); dr += float(L_det.detach()); nb += 1
        det_ep = dr / max(nb, 1)
        a_top, a_rnd, a_cmp = val_probe(model, pool, va_s, DEV,
                                        cfg['BUDGET_P'], 0, True, False)
        if np.isfinite(a_top) and a_top > best['score']:
            best = dict(score=a_top, ep=ep + 1,
                        state=copy.deepcopy(model.state_dict()))
        if cfg['LAM_NEC'] > 0:
            if det_ep >= 0.98 * TRIV:
                lam_nec = max(lam_nec * 0.5, 0.25)
            elif not math.isnan(a_cmp) and abs(a_cmp - 0.5) > 0.10:
                lam_nec = min(lam_nec * 1.5, 8.0)
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    model.eval()

    # ---- capture the representation the score is read from ----------------
    grab = {}
    def _hook(_m, _in, _out):
        grab['h'] = _in[0].detach().float().cpu().numpy()
    h = last_scalar_linear(model).register_forward_hook(_hook)

    H, Y, S = [], [], []
    with torch.no_grad():
        for i in range(0, len(te), 256):
            idx = te[i:i + 256]; b = mkb(pool, idx, DEV, True, False)
            Z = model.tokens(b); th, _, _ = model.gate_logits(b); tm = b['TM']
            top = model.budget_gate(th, tm, cfg['BUDGET_P'])
            s_ = model.score(Z, top, tm)
            H.append(grab['h'].reshape(len(idx), -1))
            S.append(s_.float().cpu().numpy()); Y.append(pool['Y'][idx])
    h.remove()
    H = np.concatenate(H); Y = np.concatenate(Y); S = np.concatenate(S)
    auc = float(roc_auc_score(Y, S))
    print(f'[{DS}] trained in {(time.time()-t1)/60:.1f} min; '
          f'representation {H.shape}; test AUC {auc:.4f} '
          f'(paper seed 0: {PAPER_SEED0[DS]:.4f})')
    gap = abs(auc - PAPER_SEED0[DS])
    if gap > 0.03:
        raise SystemExit(
            f'\n  STOP. This run ({auc:.4f}) is {gap:.4f} away from the '
            f'paper seed-0 value.\n  The map would be of a different model.')
    if gap > 0.01:
        print('  [sanity] within seed noise but not identical (GPU '
              'nondeterminism). Usable.')

    # ---- subsample for a readable figure ---------------------------------
    rg = np.random.default_rng(0)
    pos = np.where(Y > 0.5)[0]
    neg = np.where(Y <= 0.5)[0]
    if len(neg) > MAX_NORMAL:
        neg = np.sort(rg.choice(neg, MAX_NORMAL, replace=False))
    sel = np.sort(np.concatenate([pos, neg]))
    print(f'[{DS}] plotting {len(pos)} anomalies and {len(neg)} normals')

    emb = TSNE(n_components=2, perplexity=TSNE_PERP, init='pca',
               learning_rate='auto', random_state=0).fit_transform(H[sel])
    emb = (emb - emb.min(0)) / (emb.max(0) - emb.min(0) + 1e-9)   # -> [0,1]
    for (x_, y_), lab in zip(emb, Y[sel]):
        rows.append(dict(dataset=DS, x=round(float(x_), 4),
                         y=round(float(y_), 4), label=int(lab)))
    pd.DataFrame(rows).to_csv(CSV_LOCAL, index=False)
    try:
        pd.DataFrame(rows).to_csv(CSV, index=False)
    except OSError:
        pass

# -------------------------------------------------------------- summary ----
nd = pd.DataFrame(rows)
print('\n' + '=' * 78)
print('EMBEDDING MAP')
print('=' * 78)
for DS in DATASETS:
    g = nd[nd.dataset == DS]
    if len(g):
        print(f'  {DS}: {len(g)} points, {int(g.label.sum())} anomalous')
print('\n  The map is of EVIDENT only. In the caption, say that the')
print('  representation is built from the selected evidence alone, and do')
print('  NOT imply a comparison with the baselines: none was drawn.')

print('\n' + '=' * 78)
print('COPY THE TWO BLOCKS BELOW (pgfplots coordinates)')
print('=' * 78)
for DS in DATASETS:
    g = nd[nd.dataset == DS]
    if not len(g):
        continue
    for lab, nm in ((0, 'normal'), (1, 'anomalous')):
        gg = g[g.label == lab]
        print(f'\n%%%% {DS} -- {nm} ({len(gg)} points)')
        print(' '.join(f'({x:.4f},{y:.4f})' for x, y in zip(gg.x, gg.y)))
print(f'\nDONE in {(time.time()-T0)/60:.1f} min')
