# Nursing Home Fall-Risk Prediction

This repository contains the reproducible analysis code and shareable data files for a development/internal-validation study of next-year fall-risk prediction in nursing home residents.

The repository intentionally does not include the manuscript source, LaTeX files, Overleaf assets, original PDFs, original notebooks, or raw institutional exports.

## Repository Contents

- `src/article_pipeline.py`: end-to-end analysis pipeline.
- `data/processed/person_year_dataset.csv`: anonymized person-year analysis dataset used by the public pipeline.
- `data/derived/`: precomputed tables, metrics, calibration summaries, decision-curve data, sensitivity analyses, and selected out-of-fold predictions.
- `figures/`: generated figures used to inspect model performance.
- `results/article_report.md`: generated analysis summary.
- `scripts/run_analysis.ps1`: convenience script for rerunning the analysis on Windows PowerShell.

## Data Scope

The processed dataset is a longitudinal person-year table. Each row represents one resident-year at predictor year `YEAR_T`, using variables available at or before that year to predict fall-related outcomes in year `YEAR_T + 1`.

Resident identifiers are synthetic (`R0001`, `R0002`, ...). Names and raw institutional identifiers are not included.

Before making this repository public, confirm that release of the processed row-level dataset is covered by the relevant ethics, data-governance, and institutional approvals. If row-level sharing is not authorized, remove `data/processed/person_year_dataset.csv` and share only the code plus `data/derived/` aggregate outputs.

## Reproducing the Analysis

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/article_pipeline.py --repeats 5 --folds 5 --bootstrap 500
```

The command writes regenerated outputs to `results/article/`.

To also save the full out-of-fold prediction table for every model, feature set, fold, and repeat:

```powershell
python src/article_pipeline.py --repeats 5 --folds 5 --bootstrap 500 --include-oof-predictions
```

## Main Analysis Design

- Prediction horizon: next-year fall outcomes.
- Unit of analysis: resident-year.
- Validation: repeated stratified grouped cross-validation, grouped by resident.
- Primary outcome: any fall in the next year.
- Secondary outcomes: recurrent falls, high-burden falls, and death after a fall in the next year.
- Models: stratified dummy classifier, class-weighted logistic regression, class-balanced Random Forest, and Gradient Boosting.
- Sensitivity analyses: removal of habitual aid, removal of explicit functional-assessment availability indicators, and strict historical-variable models.

## Citation

If you use this repository, please cite the associated article when available. A provisional `CITATION.cff` file is included and should be updated with the final DOI and bibliographic details after publication.

## License

Code is released under the MIT License. Data reuse is described in `DATA_AVAILABILITY.md` and should be checked against the final institutional authorization before public release.
