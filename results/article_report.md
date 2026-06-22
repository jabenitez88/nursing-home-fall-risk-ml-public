# Article-ready analysis package

This package rebuilds a longitudinal person-year dataset and evaluates prospective prediction tasks using grouped validation by patient.

## Cohort

| person_years | patients | first_predictor_year | last_predictor_year | next_year_any_fall_n | next_year_any_fall_rate | next_year_recurrent_fall_n | next_year_high_burden_fall_n | next_year_death_after_fall_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1158 | 242 | 2009 | 2020 | 472 | 0.408 | 288 | 134 | 73 |

## Baseline characteristics by next-year fall outcome

| FALL_NEXT_YEAR | n | age_median | age_q1 | age_q3 | female_rate | same_year_fall_rate | median_cumulative_falls | functional_available_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 686 | 89.000 | 83.000 | 93.000 | 0.836 | 0.265 | 1.000 | 0.112 |
| 1 | 472 | 88.000 | 84.000 | 91.000 | 0.873 | 0.638 | 3.000 | 0.280 |

## Best non-dummy model per outcome

| outcome | outcome_label | feature_set | model | n_total | n_groups_total | target_positive | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | sensitivity_mean | specificity_mean | ppv_mean | roc_auc_mean | pr_auc_mean | brier_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| death_after_fall_next_year | Death after fall in next year | functional_no_availability | random_forest_balanced | 1158 | 242 | 73 | 0.530 | 0.036 | 0.529 | 0.093 | 0.964 | 0.155 | 0.577 | 0.121 | 0.096 |
| fall_next_year | Any fall in next year | functional_lcf | random_forest_balanced | 1158 | 242 | 472 | 0.683 | 0.032 | 0.687 | 0.676 | 0.699 | 0.607 | 0.748 | 0.641 | 0.202 |
| high_burden_fall_next_year | Four or more falls in next year | core_history_with_ayuda | random_forest_balanced | 1158 | 242 | 134 | 0.660 | 0.038 | 0.688 | 0.491 | 0.884 | 0.362 | 0.764 | 0.353 | 0.127 |
| recurrent_fall_next_year | Two or more falls in next year | functional_no_ayuda | random_forest_balanced | 1158 | 242 | 288 | 0.667 | 0.040 | 0.688 | 0.611 | 0.765 | 0.464 | 0.753 | 0.478 | 0.182 |

## Bootstrap CIs for selected out-of-fold predictions

### death_after_fall_next_year

| metric | estimate | ci_low | ci_high |
| --- | --- | --- | --- |
| accuracy | 0.912 | 0.891 | 0.931 |
| balanced_accuracy | 0.525 | 0.495 | 0.558 |
| macro_f1 | 0.529 | 0.490 | 0.572 |
| weighted_f1 | 0.900 | 0.877 | 0.921 |
| sensitivity | 0.082 | 0.021 | 0.146 |
| specificity | 0.968 | 0.956 | 0.979 |
| ppv | 0.146 | 0.043 | 0.273 |
| npv | 0.940 | 0.925 | 0.953 |
| brier | 0.094 | 0.085 | 0.104 |
| prevalence | 0.063 | 0.050 | 0.078 |
| roc_auc | 0.578 | 0.504 | 0.648 |
| pr_auc | 0.099 | 0.073 | 0.151 |

### fall_next_year

| metric | estimate | ci_low | ci_high |
| --- | --- | --- | --- |
| accuracy | 0.687 | 0.654 | 0.718 |
| balanced_accuracy | 0.684 | 0.653 | 0.715 |
| macro_f1 | 0.680 | 0.650 | 0.712 |
| weighted_f1 | 0.689 | 0.657 | 0.720 |
| sensitivity | 0.672 | 0.624 | 0.718 |
| specificity | 0.697 | 0.654 | 0.740 |
| ppv | 0.604 | 0.552 | 0.652 |
| npv | 0.755 | 0.712 | 0.794 |
| brier | 0.201 | 0.188 | 0.212 |
| prevalence | 0.408 | 0.362 | 0.454 |
| roc_auc | 0.751 | 0.716 | 0.783 |
| pr_auc | 0.633 | 0.573 | 0.693 |

### high_burden_fall_next_year

| metric | estimate | ci_low | ci_high |
| --- | --- | --- | --- |
| accuracy | 0.847 | 0.819 | 0.875 |
| balanced_accuracy | 0.696 | 0.646 | 0.748 |
| macro_f1 | 0.671 | 0.625 | 0.716 |
| weighted_f1 | 0.856 | 0.831 | 0.883 |
| sensitivity | 0.500 | 0.394 | 0.600 |
| specificity | 0.893 | 0.868 | 0.914 |
| ppv | 0.379 | 0.288 | 0.458 |
| npv | 0.932 | 0.915 | 0.948 |
| brier | 0.125 | 0.111 | 0.139 |
| prevalence | 0.116 | 0.088 | 0.141 |
| roc_auc | 0.764 | 0.703 | 0.808 |
| pr_auc | 0.323 | 0.240 | 0.416 |

### recurrent_fall_next_year

| metric | estimate | ci_low | ci_high |
| --- | --- | --- | --- |
| accuracy | 0.730 | 0.697 | 0.760 |
| balanced_accuracy | 0.687 | 0.643 | 0.724 |
| macro_f1 | 0.668 | 0.630 | 0.705 |
| weighted_f1 | 0.740 | 0.709 | 0.771 |
| sensitivity | 0.601 | 0.521 | 0.669 |
| specificity | 0.772 | 0.737 | 0.803 |
| ppv | 0.466 | 0.401 | 0.524 |
| npv | 0.854 | 0.825 | 0.881 |
| brier | 0.181 | 0.167 | 0.195 |
| prevalence | 0.249 | 0.214 | 0.286 |
| roc_auc | 0.755 | 0.715 | 0.789 |
| pr_auc | 0.460 | 0.394 | 0.540 |

## Interpretation

- The strongest defensible signal is prospective next-year fall prediction.
- Recurrent and high-burden fall outcomes are feasible but should be framed as secondary/exploratory.
- Death after fall remains too rare for a strong standalone model in this dataset.
- Validation is grouped by patient; no row-level leakage is allowed in the primary estimates.
- The current evidence is suitable for a development/internal-validation manuscript, not for deployment claims without external validation.

## Files

- `longitudinal_article_dataset.csv`: reconstructed person-year dataset.
- `model_fold_metrics.csv`: fold-level metrics.
- `model_summary.csv`: repeated grouped CV summary.
- `oof_predictions.csv`: out-of-fold predictions.
- `figures/*.png`: ROC, PR, calibration, decision curve, yearly distribution, and permutation importance plots.