# EVIDENT

**E**vidence-**V**erified **I**ntrinsic **D**etection of **E**vents in dy**N**amic graphs with **T**emporal rationales.

EVIDENT is an *ante-hoc intrinsically explainable* anomaly detector for dynamic
graphs. Its anomaly score is computed **exclusively** from a sparse, dually
factorised evidence set selected by the model itself. The explanation is not an
approximation of the decision — it *is* the causal input to the decision.
Sufficiency is therefore guaranteed by construction rather than estimated, and
the explanation costs no additional forward pass.

This repository accompanies the paper *EVIDENT: Intrinsically Explainable
Anomaly Detection in Dynamic Graphs*, and is the ante-hoc counterpart to the
post-hoc X-series (X-TADDY, X-AddGraph, X-CoLA, X-StrGNN, X-SAD).

---

## What is new

**Dual factorised rationale.** The gate logit factorises over *who* and *when*:

```
theta = a (+) c + P Q^T
```

with `a` a score per counterparty slot, `c` a score per temporal slot, and a
rank-`r` interaction term capturing counterparty-time pairs that neither
marginal explains. Selection therefore answers *which counterparties* and *at
what times* separately, and their interaction jointly.

**True removal, not attenuation.** A closed gate zeroes the value vector *and*
sets the attention bias to `-inf`, so the token leaves the softmax denominator
entirely. Flooring the gate at a small epsilon leaves every token contributing
at `O(eps)`; because AUC is rank-based and scale-invariant, a signal attenuated
by nine orders of magnitude ranks identically to an unattenuated one, and every
mask behaves as the full pool. We recommend reporting **empty-mask AUC** as a
standard integrity check for any gated architecture. `src/evident/run_uci.py`
enforces it and halts the run on failure.

**Fixed-cardinality evidence pools.** Every pool is resampled to a constant
number of node slots and token slots. This closes the neighbourhood-cardinality
channel described below.

---

## Reproducing the UCI-Messages results

Three cells, in order, in a fresh Colab notebook:

```python
%run src/evident/model.py      # bootstrap: model, pools, training, probes
%run src/evident/data_uci.py   # artefact resolution + pool construction
%run src/evident/run_uci.py    # integrity check, baseline, 5 seeds, claims
```

`run_uci.py` checkpoints to Drive after every seed, so a disconnect costs one
seed rather than the run; re-running the cell resumes.

### Artefact provenance

`data_uci.py` resolves `ml_uci_testinj.csv` from three sources in order —
local Drive, this repository, then a documented rebuild from CollegeMsg — and
prints in capitals which one it used. **This matters.** The benchmark's other
rows were measured on one realisation of the injection protocol; a rebuild with
a different seed is a different benchmark, and any comparison must say so.

---

## Verified results

### UCI-Messages (injected anomalies), 4 seeds

| model | AUC | AP | P@100 |
|---|---|---|---|
| ungated baseline (density 1.0, n=5) | 0.9881 ± 0.0008 | 0.7892 ± 0.0166 | 0.8720 ± 0.0325 |
| **EVIDENT-S (p = 0.20, n=4)** | **0.9766 ± 0.0050** | 0.6715 ± 0.0582 | 0.7425 ± 0.0567 |
| retention | 0.988 | 0.851 | 0.851 |

Reference rows from the unified benchmark: StrGNN 0.9408, Deg-Sum 0.8112.

EVIDENT is evaluated under fixed-cardinality pools and therefore has **no
access** to the neighbourhood-cardinality channel that reaches AUC 0.9861
unaided on this benchmark; the rows it is compared against retain that channel.

Note the divergence between retentions: the bottleneck costs 1.2% of AUC but
15% of average precision. AUC is nearly blind to a change that removes a seventh
of the precision at the top of the ranking — an independent instance of the
metric-divergence finding in the unified benchmark.

### Bitcoin (real labels)

| model | corpus | AUC |
|---|---|---|
| EVIDENT-S | Bitcoin-OTC | 0.7947 ± 0.0069 |
| EVIDENT-H | Bitcoin-OTC | 0.9179 ± 0.0022 |
| EVIDENT-S | Bitcoin-Alpha | 0.6679 |

Necessity on Bitcoin-OTC under the density-matched protocol: C3 = +0.2145
(6.6 sigma). EVIDENT-H is not comparable with unsupervised detectors, since it
consumes ratings of past observed edges.

---

## Leakage mechanisms documented

**Neighbourhood cardinality under edge injection.** On a standard injected
benchmark, node count alone attains AUC 0.9861. Genuine endpoints interact
repeatedly and share partners, so their histories overlap and the neighbourhood
union is small; injected endpoints are unrelated and produce a large union. Any
enclosing-subgraph method with variable pool size can read the label from
cardinality alone. On real labels the same statistic is uninformative and
Deg-Sum falls to 0.4958.

**Incomplete removal in gated attention.** Flooring gate values instead of
removing tokens lets the full pool leak through at attenuated magnitude, which
rank-based metrics cannot distinguish from no attenuation. Before correcting
this we measured an apparently strong detector at 0.7935 ± 0.0050 whose *empty*
mask scored 0.7975 — an impossibility that exposed the defect.

---

## Token heterogeneity governs explainability

In a preliminary design where tokens carried only a role class and a timestamp,
tokens were near-exchangeable and the necessity gap was indistinguishable from
zero (+0.0006, 0.04 sigma) despite an identical architecture. Enriching token
features raised it to +0.2145 (6.6 sigma). Whether a rationale can be meaningful
is a property of the representation, not only of the selection mechanism.

UCI-Messages is unsigned and carries no ratings, so it necessarily runs in the
role-plus-timestamp regime. There the necessity gap is +0.0056 (1.03 sigma)
against +0.2145 (6.6 sigma) on Bitcoin-OTC, with `bottom_k` still reaching
0.9417 — almost any 20% of the pool is sufficient evidence. The finding now
holds across two corpora and two anomaly types.

---

## Layout

```
src/evident/model.py      model, pool construction, training, matched probe
src/evident/data_uci.py   artefact resolution, injection protocol, pools
src/evident/run_uci.py    integrity check, ungated baseline, 5 seeds, claims
scripts/verify_removal.py standalone empty-mask integrity check
results/                  JSON emitted by run_uci.py
paper/                    LaTeX sources
```

## Requirements

`pip install -r requirements.txt`. A GPU is recommended; the eleven trainings in
`run_uci.py` are the bulk of the cost.

## Citation

See `CITATION.cff`.

## License

MIT. See `LICENSE`.
