# Data Files

## `processed/person_year_dataset.csv`

Anonymized person-year dataset used by `src/article_pipeline.py`.

Each row is a resident-year. Predictors are measured at or before `YEAR_T`; outcomes refer to `YEAR_T + 1`.

Direct names and raw institutional identifiers are not included. The `PERSON_ID` column contains synthetic identifiers required for resident-grouped cross-validation.

## `derived/`

Precomputed outputs from the analysis pipeline:

- cohort and baseline summaries;
- repeated grouped cross-validation fold metrics;
- model-level summaries and best-model tables;
- bootstrap confidence intervals;
- calibration tables and calibration metrics;
- decision-curve tables;
- sensitivity analyses;
- selected aggregated out-of-fold predictions;
- permutation-importance results.

The full out-of-fold prediction table for all models and folds is not included by default because it can be regenerated with:

```powershell
python src/article_pipeline.py --include-oof-predictions
```
