# Data Availability and Use

This repository is designed to share the analysis code, processed analysis dataset, and non-identifying data products needed to reproduce the reported experiments.

## Included Data

- `data/processed/person_year_dataset.csv`: anonymized person-year analysis dataset used in the article experiments.
- `data/derived/*.csv`: aggregate or derived model outputs, including cohort summaries, cross-validation metrics, calibration tables, decision-curve data, bootstrap confidence intervals, sensitivity analyses, and selected aggregated out-of-fold predictions.

The processed dataset removes names and raw institutional identifiers. `PERSON_ID` is a synthetic resident identifier used only to preserve grouped validation.

## Not Included

- Raw institutional exports.
- Resident names.
- Original notebooks containing exploratory work.
- PDF reports, student reports, conference paper PDFs, LaTeX source, and Overleaf files.
- The full `oof_predictions.csv` table for every model/fold/repeat, because it is large and can be regenerated with `--include-oof-predictions`.

## Important Governance Note

The processed person-year dataset remains row-level health-related data from a small institutional cohort. Even after removal of direct identifiers, public release should be approved by the relevant data controller, ethics committee, and institutional governance process.

If a future repository copy is prepared in a setting where row-level release is not authorized, remove:

```text
data/processed/person_year_dataset.csv
```

and publish only:

```text
src/
data/derived/
figures/
results/article_report.md
```

Users of the data must not attempt to re-identify residents or link these records to other data sources.
