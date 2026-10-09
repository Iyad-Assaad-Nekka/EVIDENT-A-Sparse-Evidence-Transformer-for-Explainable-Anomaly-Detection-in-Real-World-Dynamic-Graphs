[README.md](https://github.com/user-attachments/files/33262539/README.md)
# EVIDENT

**A sparse-evidence transformer for intrinsically explainable anomaly detection in real-world dynamic graphs.**

This repository contains the code, protocol and per-seed results of the paper
*EVIDENT: A Sparse-Evidence Transformer for Intrinsically Explainable Anomaly
Detection in Real-World Dynamic Graphs*. Every number in the paper can be
reproduced from the scripts in `experiments/`.

---

## Overview

State-of-the-art dynamic-graph anomaly detectors are evaluated almost entirely
on *injected* anomalies: random node pairs added to a real edge stream. We run
the three strongest of them (StrGNN, TADDY, SAD) on two Bitcoin trust networks
with **real, human-assigned** labels, using identical splits, identical test
events and identical metric code, with the injected protocol as a control.

**AUC on injected anomalies vs. real labels**

| Method | Bitcoin-OTC injected | Bitcoin-OTC real | Bitcoin-Alpha injected | Bitcoin-Alpha real |
|---|---|---|---|---|
| Degree heuristic | 0.8841 | 0.4920 | 0.8855 | 0.4269 |
| StrGNN | 0.9819 | 0.5373 | 0.9775 | 0.5561 |
| TADDY | 0.9726 | 0.5390 | 0.9666 | 0.5570 |
| SAD | 0.9779 | 0.6794 | 0.9443 | 0.5974 |
| **EVIDENT** | 0.9614 | **0.8622** | 0.9271 | **0.7606** |

Real-label results are means over eight seeds. Average precision, P@100 and the
explanation metrics are in `results/` and in the paper.

EVIDENT selects about five past interactions as *evidence* and computes the
anomaly score from that evidence **alone**. The explanation is therefore the
basis of the decision, not an approximation fitted afterward, and it can be
tested: with every evidence gate closed the score is constant and the AUC is
exactly 0.5.

---

## Repository layout

```
model.py        core library: configuration, causal evidence-pool builder,
                chronological split, metrics, training utilities
experiments/
  evident/      EVIDENT: main runs, injected control, ablations, case study,
                runtime, sensitivity grid
  baselines/    StrGNN, TADDY and SAD: data adapters, real-label runs,
                injected controls
  analysis/     non-learned reference rules
  datasets/     screening scripts for the datasets examined in Section 5.2
paper/          LaTeX source and figures of the paper
results/        per-seed numbers behind every table and figure
```

Each script is a self-contained notebook cell. Run it in a Colab GPU runtime:
it clones what it needs, downloads the data, trains and prints a summary. When
Google Drive is mounted, intermediate results are cached there and re-running a
script resumes at the next unfinished seed.

## Reproducing the main results

```
experiments/evident/EVIDENT_PLUS_OTC.py            # EVIDENT, Bitcoin-OTC, 8 seeds
experiments/evident/EVIDENT_PLUS_ALPHA_FINAL.py    # EVIDENT, Bitcoin-Alpha, 8 seeds
experiments/baselines/STRGNN_STAGE1.py             # StrGNN: subgraph extraction (run first)
experiments/baselines/STRGNN_SEEDS12_FAST.py       # StrGNN, Bitcoin-OTC
experiments/baselines/STRGNN_SEEDS_ALPHA.py        # StrGNN, Bitcoin-Alpha
experiments/baselines/TADDY_STAGE1.py              # TADDY: data adapter (run first)
experiments/baselines/TADDY_OTC_REAL_SEEDS.py      # TADDY, Bitcoin-OTC
experiments/baselines/TADDY_ALPHA_REAL_SEEDS.py    # TADDY, Bitcoin-Alpha
experiments/baselines/SAD_STAGE1_ADAPTER.py        # SAD: data adapter (run first)
experiments/baselines/SAD_STAGE2B_OTC_FEATURES.py  # SAD, Bitcoin-OTC
experiments/baselines/SAD_STAGE2C_ALPHA.py         # SAD, Bitcoin-Alpha
```

The injected control for each method is the `*_INJECTED*.py` script beside it.

## Protocol

- **Datasets.** Bitcoin-OTC and Bitcoin-Alpha (SNAP). An event is anomalous
  when its rating is at most −5, the strong-distrust half of the negative
  scale. Bitcoin-OTC: 5,881 nodes, 35,592 events. Bitcoin-Alpha: 3,783 nodes,
  24,186 events.
- **Split.** Chronological 70/15/15 over the usable events (those whose
  endpoints have at least one earlier event). Every method is trained, selected
  and scored on exactly the same events.
- **Causality.** Every feature is computed from events strictly earlier than
  the one being scored. The scripts check this by rebuilding feature rows from
  a prefix of the stream and comparing.
- **Model selection.** The epoch is chosen on validation AUC. Real-label
  results are means over eight seeds.
- **Metrics.** AUC, average precision and precision among the 100
  highest-scored test events, from one shared implementation.

## Inputs of each method

EVIDENT, SAD and (in the real-label setting) StrGNN are trained on the anomaly
labels of the training period; TADDY follows its label-free protocol. EVIDENT
and SAD use information from past ratings; StrGNN and TADDY use topology and
time only, as designed.

## Additional analyses

Two analyses beyond the paper's tables are provided for completeness:

- `results/no_ratings_ablation.csv`: EVIDENT with all rating-derived inputs
  removed (script: `experiments/evident/EVIDENT_NO_RATINGS.py`).
- `results/reference_rules.csv`: non-learned and simple learned reference rules
  computed from causal running statistics of each node's rating history
  (script: `experiments/analysis/REPUTATION_HEURISTICS.py`).

## Requirements

Python 3.10+, PyTorch with CUDA, and the packages in `requirements.txt`. All
runs were made on a single NVIDIA T4 GPU. Training EVIDENT on Bitcoin-OTC takes
about four minutes per seed.

## Citation

```bibtex
@article{nekka2026evident,
  title   = {EVIDENT: A Sparse-Evidence Transformer for Intrinsically Explainable
             Anomaly Detection in Real-World Dynamic Graphs},
  author  = {Nekka, Iyad Assaad and Hidouci, Walid Khaled and Seba, Hamida
             and Amrouche, Karima},
  year    = {2026},
  note    = {Under review}
}
```

## Acknowledgements

The baseline implementations are those released by the authors of
[StrGNN](https://github.com/LeiCaiwsu/StrGNN),
[TADDY](https://github.com/yixinliu233/TADDY_pytorch) and
[SAD](https://github.com/D10Andy/SAD). The Bitcoin datasets are from
[SNAP](https://snap.stanford.edu/data/).

## License

MIT. See `LICENSE`.
