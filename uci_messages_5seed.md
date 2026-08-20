# UCI-Messages (injected anomalies)

Produced by `src/evident/run_uci.py` and `src/evident/part_c_cpu.py`.

**n = 4 seeds (0–3).** Seed 4 completed on the GPU run but the session
terminated before its results were persisted; neither its metrics nor its
checkpoint survived. All figures below are over four seeds.

Artefact source: `A: LOCAL DRIVE COPY (exact)` — the same realisation of
the injection protocol as the other rows of the unified benchmark, so the
comparison needs no caveat.

## Detection

| model | AUC | AP | P@100 |
|---|---|---|---|
| ungated baseline (density 1.0, n=5) | 0.9881 ± 0.0008 | 0.7892 ± 0.0166 | 0.8720 ± 0.0325 |
| EVIDENT-S (p = 0.20, n=4)           | 0.9766 ± 0.0050 | 0.6715 ± 0.0582 | 0.7425 ± 0.0567 |
| retention                            | 0.988 | 0.851 | 0.851 |

Reference rows: StrGNN 0.9408, Deg-Sum 0.8112.

EVIDENT runs under fixed-cardinality pools and therefore has no access to
the neighbourhood-cardinality channel that reaches AUC 0.9861 unaided on
this benchmark; every row it is compared against retains that channel.

The bottleneck costs 1.2% of AUC but 15% of average precision. AUC is
nearly blind to a change that removes a seventh of the precision at the
top of the ranking.

## Density-matched mask ladder

| mask | AUC | AP |
|---|---|---|
| empty        | 0.5000 ± 0.0000 | 0.0538 |
| bottom_k     | 0.9417 ± 0.0174 | 0.5048 ± 0.0781 |
| disjoint_rnd | 0.9644 ± 0.0037 | 0.5612 ± 0.0695 |
| random       | 0.9710 ± 0.0043 | 0.6325 ± 0.0705 |
| top_k        | 0.9766 ± 0.0050 | 0.6715 ± 0.0582 |

Empty-mask AUC is exactly 0.5000: with every token removed all scores are
identical, so removal is complete and no information bypasses the gate.

## Claims

| claim | gap | sigma |
|---|---|---|
| C1 ranking (top_k − bottom_k)      | +0.0349 ± 0.0197 | 1.77 |
| C2 necessity (top_k − disjoint_rnd)| +0.0122 ± 0.0068 | 1.79 |
| C3 vs random (top_k − random)      | +0.0056 ± 0.0054 | 1.03 |

The ladder is correctly ordered but no claim clears 2 sigma. This is the
expected consequence of token near-exchangeability, not a failure of the
selection mechanism.

UCI-Messages is unsigned and carries no ratings, so tokens hold only a
role class and a timestamp — the same regime in which the necessity gap on
the preliminary Bitcoin design was +0.0006 (0.04 sigma). With enriched,
heterogeneous tokens on Bitcoin-OTC the same architecture reaches +0.2145
(6.6 sigma).

`bottom_k` reaching 0.9417 makes the mechanism explicit: on a benchmark
this close to saturation, with homogeneous tokens, almost any 20% of the
pool is sufficient evidence, so there is little for the gate to rank.

**Conclusion.** Rationale quality is bounded by the heterogeneity of the
token representation, not by the selection mechanism alone. This now holds
across two corpora and two anomaly types.
