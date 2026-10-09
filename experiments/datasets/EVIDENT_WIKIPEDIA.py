#!/usr/bin/env python
# ============================================================================
# EVIDENT_WIKIPEDIA.py
#
# Dataset screening: EVIDENT on Wikipedia (JODIE) state-change labels.
#
# Run as a single cell in a Colab GPU runtime. Results are cached on Google
# Drive when it is mounted; re-running resumes at the next unfinished seed.
# ============================================================================

import os, sys, math, time, gzip, copy, urllib.request, subprocess
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
DS         = 'wikipedia'
MAX_EDGES  = 60000        # cap; keeps the most recent window

# EVIDENT, real labels, p=0.20, EPOCHS=30, 8 seeds (measured, for the table)
BTC = dict(
  otc=dict(auc=0.8623, sd=0.0129, ap=0.5489, p100=0.8650, adv=0.1336,
           noc_adv=0.0624, base=0.0787, abl_p=0.0003),
  alpha=dict(auc=0.7606, sd=0.0186, ap=0.2519, p100=0.4163, adv=0.0762,
             noc_adv=0.0269, base=0.0398, abl_p=0.0005))

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


def load_wikipedia():
    """JODIE wikipedia.csv: user_id, item_id, timestamp, state_label, feats.
    state_label == 1 marks an edit after which the editor was BANNED -- a real
    moderation decision, not an injection. r = 0 everywhere, so the
    rating-derived features are inert by construction."""
    MIN_BYTES = 100_000_000
    drive_p = os.path.join(DATA, 'wikipedia.csv')
    local_p = '/content/wikipedia.csv'
    # Prefer a valid copy already on Drive; otherwise keep the 560 MB file in
    # the CONTAINER. Writing it to Drive is slow, and os.replace cannot move
    # across filesystems (Errno 18), which is what kept failing at 524/560 MB.
    if os.path.exists(drive_p) and os.path.getsize(drive_p) >= MIN_BYTES:
        p = drive_p
    elif os.path.exists(local_p) and os.path.getsize(local_p) >= MIN_BYTES:
        p = local_p
    else:
        p = local_p
    # The file is ~560 MB (172-dim edge features), not 50 MB. A single read()
    # truncates, so stream it in chunks, write to a .part file, verify the
    # length against Content-Length, and only then move it into place. A
    # truncated download must never be left where a rerun would load it.
    urls = ['https://snap.stanford.edu/jodie/wikipedia.csv',
            'http://snap.stanford.edu/jodie/wikipedia.csv']
    # A too-small file counts as MISSING, not as an error. Checking size only
    # AFTER the existence check made a 0-byte leftover skip the download and
    # then abort -- an infinite loop, since Drive does not always propagate the
    # delete before the next run.
    if os.path.exists(p) and os.path.getsize(p) < MIN_BYTES:
        print(f'  discarding truncated {os.path.getsize(p)/1e6:.0f} MB file',
              flush=True)
        try:
            os.remove(p)
        except OSError:
            pass
    if (not os.path.exists(p)) or os.path.getsize(p) < MIN_BYTES:
        ok = False
        for attempt in range(3):
            for url in urls:
                tmp = p + '.part'   # same filesystem as p
                try:
                    print(f'  downloading {url} (attempt {attempt+1})...',
                          flush=True)
                    req = urllib.request.Request(
                        url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=120) as r, \
                            open(tmp, 'wb') as fh:
                        exp = int(r.headers.get('Content-Length', 0))
                        got, t0 = 0, time.time()
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            fh.write(chunk); got += len(chunk)
                            if got % (50 << 20) < (1 << 20):
                                print(f'    {got/1e6:.0f} MB'
                                      + (f' / {exp/1e6:.0f} MB' if exp else '')
                                      + f'  ({got/1e6/max(time.time()-t0,1):.1f} MB/s)',
                                      flush=True)
                    if exp and got < exp:
                        raise IOError(f'truncated: {got} of {exp} bytes')
                    os.replace(tmp, p)
                    print(f'  downloaded {got/1e6:.0f} MB', flush=True)
                    ok = True; break
                except Exception as e:
                    print(f'  FAILED: {e}', flush=True)
                    if os.path.exists(tmp):
                        os.remove(tmp)
            if ok:
                break
        if not ok:
            print('\n  Could not fetch wikipedia.csv. Download it manually and')
            print(f'  place it at {p}, then re-run. Direct link:')
            print('    https://snap.stanford.edu/jodie/wikipedia.csv')
            sys.exit(1)
    print(f'  using {p}  ({os.path.getsize(p)/1e6:.0f} MB)', flush=True)
    # only the first 4 columns are needed; the 172 feature columns are skipped
    d = pd.read_csv(p, skiprows=1, header=None, usecols=[0, 1, 2, 3],
                    names=['u', 'v', 't', 'y'])
    d['v'] = d.v + d.u.max() + 1
    d['y'] = d.y.astype(np.float32); d['r'] = 0.0
    d = d.sort_values('t', kind='mergesort').reset_index(drop=True)
    if len(d) > MAX_EDGES:
        d = d.iloc[-MAX_EDGES:].reset_index(drop=True)
        print(f'  capped to the most recent {MAX_EDGES} interactions')
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
cfg = dict(CFG); cfg['BUDGET_P'] = MAIN_P; cfg['EPOCHS'] = EPOCHS
df, n_nodes = load_wikipedia()
print(f'[data] edges={len(df)} nodes={n_nodes} pos={df.y.mean():.4f}')
CACHE = os.path.join(MASTER, f'poolplus_{DS}.npz')
if os.path.exists(CACHE):
    z = np.load(CACHE); pool = {k: z[k] for k in z.files}
    pool['keep'] = pool['keep'].astype(bool); print('[pool] from cache')
else:
    pool = build_pool_plus(df, n_nodes, cfg)
    np.savez_compressed(CACHE, **pool); print('[pool] built and cached')
tr, va, te = chrono_split(pool['keep'], cfg)
print(f'[split] tr={len(tr)} va={len(va)} te={len(te)}')
print(f'[prot] select_on={SELECT_ON} epochs={EPOCHS} p={MAIN_P} seeds={len(SEEDS)}')

CSV = os.path.join(MASTER, f'evidentplus_real_{DS}.csv')
rows = pd.read_csv(CSV).to_dict('records') if os.path.exists(CSV) else []
done = {(str(r['variant']), int(r['seed'])) for r in rows}
if done:
    print(f'[resume] {len(done)} trainings already done')

for nm, m in heuristics(df, te).items():
    print(f'  [heur] {nm:<8} AUC={m["auc"]:.4f} P@100={m["p100"]:.4f}')

VARIANTS = ['plus_dedup', 'plus_dedup_no_complement']
for variant in VARIANTS:
    for sd in SEEDS:
        if (variant, sd) in done:
            print(f'  [skip] {variant} seed{sd}'); continue
        print(f'  --- {variant} seed{sd} ---'); t1 = time.time()
        r = train_one(pool, tr, va, te, cfg, sd, variant)
        rows.append(dict(dataset=DS, variant=variant, seed=sd,
                         budget_p=MAIN_P, **r))
        print(f'    AUC={r["auc"]:.4f} AP={r["ap"]:.4f} P@100={r["p100"]:.4f} '
              f'adv={r["rationale_advantage"]:+.4f} k={r["tokens_selected"]:.1f} '
              f'empty={r["empty_auc"]:.4f} ({time.time()-t1:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)

nd = pd.DataFrame(rows)
f = nd[nd.variant == 'plus_dedup'].sort_values('seed')
n_ = nd[nd.variant == 'plus_dedup_no_complement'].sort_values('seed')
base = float(df.y.to_numpy()[te].mean())

print('\n' + '=' * 88)
print('EVIDENT, REAL LABELS, THREE DATASETS -- same method, same protocol')
print('=' * 88)
print(f"{'dataset':<18}{'AUC':>18}{'AP':>9}{'P@100':>9}{'lift':>8}{'adv':>9}{'base':>9}")
for nm, d in [('bitcoin_otc', BTC['otc']), ('bitcoin_alpha', BTC['alpha'])]:
    print(f"{nm:<18}{d['auc']:>10.4f}+/-{d['sd']:.4f}{d['ap']:>9.4f}"
          f"{d['p100']:>9.4f}{d['p100']/d['base']:>7.1f}x{d['adv']:>9.4f}{d['base']:>9.4f}")
if len(f):
    print(f"{'wikipedia (NEW)':<18}{f.auc.mean():>10.4f}+/-{f.auc.std(ddof=1):.4f}"
          f"{f.ap.mean():>9.4f}{f.p100.mean():>9.4f}"
          f"{f.p100.mean()/max(base,1e-9):>7.1f}x"
          f"{f.rationale_advantage.mean():>9.4f}{base:>9.4f}")

print(f"\nempty-mask: max dev {float((nd.empty_auc-0.5).abs().max()):.4f}, "
      f"exact {int((nd.empty_auc==0.5).sum())}/{len(nd)}"
      + (f" | distinct tokens {f.tokens_selected.mean():.1f}" if len(f) else ""))

print('\n--- NECESSITY ABLATION, all three datasets ---')
print(f"{'':<18}{'full adv':>11}{'no_compl':>11}{'diff':>10}{'t-p':>10}{'w-p':>9}")
for nm, d in [('bitcoin_otc', BTC['otc']), ('bitcoin_alpha', BTC['alpha'])]:
    print(f"{nm:<18}{d['adv']:>11.4f}{d['noc_adv']:>11.4f}"
          f"{d['adv']-d['noc_adv']:>+10.4f}{d['abl_p']:>10.4f}{0.0078:>9.4f}")
if len(f) == len(n_) == len(SEEDS):
    a, b = f.rationale_advantage.values, n_.rationale_advantage.values
    t, p = stats.ttest_rel(a, b); w = stats.wilcoxon(a, b).pvalue
    print(f"{'wikipedia (NEW)':<18}{a.mean():>11.4f}{b.mean():>11.4f}"
          f"{a.mean()-b.mean():>+10.4f}{p:>10.4f}{w:>9.4f}")
    pa = stats.ttest_rel(f.auc.values, n_.auc.values).pvalue
    print(f"\n  AUC cost of the necessity term: {f.auc.mean()-n_.auc.mean():+.4f}"
          f"  (p={pa:.4f})")
    print('\n--- read ---')
    print('  Ratings are absent, so this isolates EVIDENT on topology and')
    print('  timing alone, in a bipartite domain with real moderation labels.')
    print(f"  Necessity ablation {'REPLICATES' if p < 0.05 else 'does NOT replicate'}"
          f" (p={p:.4f})")
    print('  -> the causal claim is ' +
          ('domain-independent, not an artefact of signed trust graphs.'
           if p < 0.05 else 'Bitcoin-specific.'))

print('\n' + '=' * 88); print('RESULTS (CSV)'); print('=' * 88)
print(nd.to_csv(index=False))
print(f'DONE in {(time.time()-T0)/60:.1f} min')
