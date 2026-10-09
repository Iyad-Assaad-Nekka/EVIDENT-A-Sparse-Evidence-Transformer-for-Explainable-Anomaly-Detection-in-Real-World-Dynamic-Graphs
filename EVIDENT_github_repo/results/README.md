# Results

| File | Contents |
|---|---|
| `main_results.csv` | Table 3: every method, both datasets, injected and real labels |
| `per_seed_auc_real_labels.csv` | The eight seeds behind every real-label AUC |
| `sad_injected_bitcoin_alpha_per_seed.csv` | SAD under injection on Bitcoin-Alpha, four seeds |
| `explainability_and_ablation.csv` | Tables 4 and 5: explanation metrics and the necessity ablation |
| `sensitivity_grid.csv` | Fig. 5: budget × width grid on Bitcoin-OTC, one seed per cell |
| `datasets_rejected.csv` | The datasets examined in Section 5.2 and the measurement that disqualified each |
| `no_ratings_ablation.csv` | Additional analysis: EVIDENT with all rating-derived inputs removed |
| `reference_rules.csv` | Additional analysis: non-learned and simple learned reference rules |

`empty_evidence_auc` is the integrity check: with every evidence gate closed the
score is constant, so the AUC must be exactly 0.5.
