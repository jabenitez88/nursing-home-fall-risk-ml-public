from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
ARTICLE_DIR = Path("results") / "article"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))

    def clean_cell(value: object) -> str:
        return str(value).replace("\n", " ").replace("|", "\\|")

    headers = [clean_cell(c) for c in display.columns]
    rows = [[clean_cell(v) for v in row] for row in display.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


@dataclass(frozen=True)
class Outcome:
    name: str
    label: str
    target_column: str
    min_positive: int = 25


@dataclass(frozen=True)
class FeatureSet:
    name: str
    label: str
    columns: tuple[str, ...]
    complete_case_columns: tuple[str, ...] = ()


def suffix(year: int) -> str:
    return "09" if year == 2009 else str(year - 2000)


def valid_number(value: object) -> bool:
    return pd.notna(value) and not (isinstance(value, (int, float, np.floating)) and float(value) < 0)


def latest_prior_value(row: pd.Series, prefix: str, year: int) -> tuple[float | None, int | None]:
    for candidate in range(year, 2015, -1):
        col = f"{prefix}{suffix(candidate)}"
        if col in row.index and valid_number(row.get(col)):
            return float(row[col]), candidate
    return None, None


def build_longitudinal_dataset(wide: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    years = list(range(2009, 2022))

    for person_idx, row in wide.iterrows():
        for year in range(2009, 2021):
            s = suffix(year)
            ns = suffix(year + 1)
            required = [f"EDAD{s}", f"NUM_CAIDA{s}", f"EDAD{ns}", f"NUM_CAIDA{ns}"]
            if any(col not in wide.columns or pd.isna(row.get(col)) for col in required):
                continue

            falls_history = []
            observed_years = 0
            for hist_year in years:
                if hist_year > year:
                    break
                num_col = f"NUM_CAIDA{suffix(hist_year)}"
                edad_col = f"EDAD{suffix(hist_year)}"
                if num_col in wide.columns and edad_col in wide.columns and pd.notna(row.get(edad_col)) and pd.notna(row.get(num_col)):
                    observed_years += 1
                    falls_history.append(max(float(row[num_col]), 0.0))

            fall_t = max(float(row[f"NUM_CAIDA{s}"]), 0.0)
            falls_prev_2 = sum(falls_history[-2:]) if falls_history else 0.0
            falls_prev_3 = sum(falls_history[-3:]) if falls_history else 0.0
            cumulative_falls = sum(falls_history)
            ever_fallen = int(cumulative_falls > 0)
            mean_falls_observed = cumulative_falls / observed_years if observed_years else np.nan

            tinetti_last, tinetti_year = latest_prior_value(row, "TINETTI", year)
            berg_last, berg_year = latest_prior_value(row, "BERG", year)
            functional_years = [y for y in [tinetti_year, berg_year] if y is not None]
            last_functional_year = max(functional_years) if functional_years else None

            next_falls = max(float(row[f"NUM_CAIDA{ns}"]), 0.0)
            death_after_next_fall = row.get(f"MUERTE_TRAS_CAIDA{ns}")
            death_after_next_fall = int(death_after_next_fall == 1)

            records.append(
                {
                    "PERSON_ID": str(person_idx),
                    "YEAR_T": year,
                    "SEXO": row.get("SEXO"),
                    "EDAD_T": row.get(f"EDAD{s}"),
                    "TPO_INSTIT_T": row.get(f"TPO_INSTIT{s}"),
                    "AYUDA_HABITUAL_T": row.get(f"AYUDA_HABITUAL{s}"),
                    "NUM_CAIDA_T": fall_t,
                    "FALL_T": int(fall_t > 0),
                    "FALLS_PREV_2Y_T": falls_prev_2,
                    "FALLS_PREV_3Y_T": falls_prev_3,
                    "CAIDAS_ACUM_T": cumulative_falls,
                    "EVER_FALLEN_T": ever_fallen,
                    "OBSERVED_YEARS_T": observed_years,
                    "MEAN_FALLS_OBSERVED_T": mean_falls_observed,
                    "TINETTI_LAST_T": tinetti_last,
                    "BERG_LAST_T": berg_last,
                    "HAS_FUNCTIONAL_T": int(last_functional_year is not None),
                    "YEARS_SINCE_FUNCTIONAL_T": (year - last_functional_year) if last_functional_year is not None else np.nan,
                    "NUM_CAIDA_NEXT": next_falls,
                    "FALL_NEXT_YEAR": int(next_falls > 0),
                    "RECURRENT_FALL_NEXT_YEAR": int(next_falls >= 2),
                    "HIGH_BURDEN_FALL_NEXT_YEAR": int(next_falls >= 4),
                    "DEATH_AFTER_FALL_NEXT_YEAR": death_after_next_fall,
                }
            )
    return pd.DataFrame.from_records(records)


def outcomes() -> list[Outcome]:
    return [
        Outcome("fall_next_year", "Any fall in next year", "FALL_NEXT_YEAR"),
        Outcome("recurrent_fall_next_year", "Two or more falls in next year", "RECURRENT_FALL_NEXT_YEAR"),
        Outcome("high_burden_fall_next_year", "Four or more falls in next year", "HIGH_BURDEN_FALL_NEXT_YEAR"),
        Outcome("death_after_fall_next_year", "Death after fall in next year", "DEATH_AFTER_FALL_NEXT_YEAR", min_positive=20),
    ]


def feature_sets() -> list[FeatureSet]:
    core = (
        "SEXO",
        "EDAD_T",
        "TPO_INSTIT_T",
        "NUM_CAIDA_T",
        "FALL_T",
        "FALLS_PREV_2Y_T",
        "FALLS_PREV_3Y_T",
        "CAIDAS_ACUM_T",
        "EVER_FALLEN_T",
        "OBSERVED_YEARS_T",
        "MEAN_FALLS_OBSERVED_T",
    )
    return [
        FeatureSet(
            "core_history",
            "Core demographic and fall-history variables",
            core,
        ),
        FeatureSet(
            "core_history_with_ayuda",
            "Core variables plus habitual aid",
            core + ("AYUDA_HABITUAL_T",),
        ),
        FeatureSet(
            "functional_lcf",
            "Core variables plus latest prior Tinetti/Berg, imputed when missing",
            core
            + (
                "AYUDA_HABITUAL_T",
                "TINETTI_LAST_T",
                "BERG_LAST_T",
                "HAS_FUNCTIONAL_T",
                "YEARS_SINCE_FUNCTIONAL_T",
            ),
        ),
        FeatureSet(
            "functional_no_ayuda",
            "Functional latest-carried-forward set excluding habitual aid",
            core
            + (
                "TINETTI_LAST_T",
                "BERG_LAST_T",
                "HAS_FUNCTIONAL_T",
                "YEARS_SINCE_FUNCTIONAL_T",
            ),
        ),
        FeatureSet(
            "functional_no_availability",
            "Functional values without explicit assessment-availability indicators",
            core
            + (
                "AYUDA_HABITUAL_T",
                "TINETTI_LAST_T",
                "BERG_LAST_T",
            ),
        ),
        FeatureSet(
            "functional_complete_case",
            "Functional complete-case subset",
            core + ("AYUDA_HABITUAL_T", "TINETTI_LAST_T", "BERG_LAST_T", "YEARS_SINCE_FUNCTIONAL_T"),
            complete_case_columns=("TINETTI_LAST_T", "BERG_LAST_T"),
        ),
    ]


def categorical_columns(columns: list[str]) -> list[str]:
    categorical = []
    for col in columns:
        if col in {"SEXO", "TPO_INSTIT_T", "AYUDA_HABITUAL_T", "FALL_T", "EVER_FALLEN_T", "HAS_FUNCTIONAL_T"}:
            categorical.append(col)
    return categorical


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    cat_cols = categorical_columns(columns)
    num_cols = [col for col in columns if col not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def models() -> dict[str, object]:
    return {
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "logistic_balanced": LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def make_pipeline(columns: list[str], estimator: object) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor(columns)), ("model", estimator)])


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float | None]:
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    out: dict[str, float | None] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else None,
        "ppv": precision_score(y_true, y_pred, zero_division=0),
        "npv": tn / (tn + fn) if (tn + fn) else None,
        "brier": brier_score_loss(y_true, y_score),
        "prevalence": float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def predicted_scores(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba
    if hasattr(pipe, "decision_function"):
        raw = pipe.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    return pipe.predict(X)


def prepare_data(long: pd.DataFrame, outcome: Outcome, feature_set: FeatureSet) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    data = long.copy()
    if feature_set.complete_case_columns:
        data = data.dropna(subset=list(feature_set.complete_case_columns)).copy()
    y = data[outcome.target_column].astype(int)
    X = data.loc[:, list(feature_set.columns)].copy()
    groups = data["PERSON_ID"].astype(str)
    return X, y, groups


def run_repeated_group_cv(long: pd.DataFrame, repeats: int, n_splits: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for outcome in outcomes():
        positives = int(long[outcome.target_column].sum())
        if positives < outcome.min_positive:
            continue
        for feature_set in feature_sets():
            X, y, groups = prepare_data(long, outcome, feature_set)
            if int(y.sum()) < outcome.min_positive or y.nunique() < 2:
                continue
            for repeat in range(repeats):
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + repeat)
                for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    train_groups = set(groups.iloc[train_idx])
                    test_groups = set(groups.iloc[test_idx])
                    overlap = len(train_groups & test_groups) / max(1, len(test_groups))

                    for model_name, estimator in models().items():
                        pipe = make_pipeline(list(X.columns), estimator)
                        status = "ok"
                        error = None
                        try:
                            pipe.fit(X_train, y_train)
                            y_pred = pipe.predict(X_test).astype(int)
                            y_score = predicted_scores(pipe, X_test)
                            metrics = binary_metrics(y_test.to_numpy(), y_pred, y_score)
                            fold_predictions = pd.DataFrame(
                                {
                                    "outcome": outcome.name,
                                    "feature_set": feature_set.name,
                                    "model": model_name,
                                    "repeat": repeat,
                                    "fold": fold,
                                    "row_index": X_test.index,
                                    "person_id": groups.iloc[test_idx].to_numpy(),
                                    "y_true": y_test.to_numpy(),
                                    "y_pred": y_pred,
                                    "y_score": y_score,
                                }
                            )
                            prediction_rows.append(fold_predictions)
                        except Exception as exc:
                            metrics = {}
                            status = "error"
                            error = repr(exc)

                        row = {
                            "outcome": outcome.name,
                            "outcome_label": outcome.label,
                            "feature_set": feature_set.name,
                            "feature_set_label": feature_set.label,
                            "model": model_name,
                            "repeat": repeat,
                            "fold": fold,
                            "status": status,
                            "error": error,
                            "n_total": len(y),
                            "n_train": len(train_idx),
                            "n_test": len(test_idx),
                            "n_groups_total": groups.nunique(),
                            "n_groups_test": len(test_groups),
                            "test_group_overlap_fraction": overlap,
                            "target_positive": int(y.sum()),
                            "target_negative": int((y == 0).sum()),
                            "test_positive": int(y_test.sum()),
                            "test_negative": int((y_test == 0).sum()),
                            "features": json.dumps(list(X.columns)),
                        }
                        row.update(metrics)
                        metric_rows.append(row)

    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return pd.DataFrame(metric_rows), predictions


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    ok = metrics[metrics["status"] == "ok"].copy()
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "roc_auc",
        "pr_auc",
        "brier",
        "test_group_overlap_fraction",
    ]
    grouped = ok.groupby(["outcome", "outcome_label", "feature_set", "feature_set_label", "model"], dropna=False)
    agg = grouped[metric_cols].agg(["mean", "std", "median"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if x]) for col in agg.columns.to_flat_index()]
    counts = grouped.agg(
        n_evaluations=("fold", "count"),
        n_total=("n_total", "first"),
        n_groups_total=("n_groups_total", "first"),
        target_positive=("target_positive", "first"),
        target_negative=("target_negative", "first"),
        features=("features", "first"),
    ).reset_index()
    return counts.merge(agg, on=["outcome", "outcome_label", "feature_set", "feature_set_label", "model"], how="left")


def best_non_dummy(summary: pd.DataFrame) -> pd.DataFrame:
    non_dummy = summary[~summary["model"].str.startswith("dummy")].copy()
    return (
        non_dummy.sort_values(["outcome", "macro_f1_mean", "roc_auc_mean"], ascending=[True, False, False])
        .groupby("outcome", as_index=False)
        .head(1)
    )


def bootstrap_metric_ci(preds: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    groups = preds["person_id"].drop_duplicates().to_numpy()
    rows = []

    def metric_bundle(sample: pd.DataFrame) -> dict[str, float | None]:
        return binary_metrics(sample["y_true"].to_numpy(), sample["y_pred"].to_numpy(), sample["y_score"].to_numpy())

    point = metric_bundle(preds)
    boot_values: dict[str, list[float]] = {k: [] for k in point.keys()}
    group_frames = {group: frame for group, frame in preds.groupby("person_id")}
    for _ in range(n_boot):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([group_frames[group] for group in sampled_groups], ignore_index=True)
        vals = metric_bundle(sample)
        for key, value in vals.items():
            if value is not None and np.isfinite(value):
                boot_values[key].append(float(value))

    for metric, value in point.items():
        values = np.array(boot_values.get(metric, []), dtype=float)
        if values.size:
            low, high = np.quantile(values, [0.025, 0.975])
        else:
            low, high = np.nan, np.nan
        rows.append({"metric": metric, "estimate": value, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def aggregate_oof_predictions(predictions: pd.DataFrame, config: pd.Series) -> pd.DataFrame:
    subset = predictions[
        (predictions["outcome"] == config["outcome"])
        & (predictions["feature_set"] == config["feature_set"])
        & (predictions["model"] == config["model"])
    ].copy()
    # Each row is predicted once per repeat. Average probabilities and use majority threshold.
    agg = (
        subset.groupby(["row_index", "person_id"], as_index=False)
        .agg(y_true=("y_true", "first"), y_score=("y_score", "mean"))
    )
    agg["y_pred"] = (agg["y_score"] >= 0.5).astype(int)
    return agg


def calibration_table(preds: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    data = preds.copy()
    data["bin"] = pd.qcut(data["y_score"], q=min(n_bins, data["y_score"].nunique()), duplicates="drop")
    table = data.groupby("bin", observed=True).agg(
        n=("y_true", "size"),
        mean_predicted=("y_score", "mean"),
        observed_rate=("y_true", "mean"),
    )
    return table.reset_index(drop=True)


def calibration_metrics(preds: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    y = preds["y_true"].to_numpy(dtype=int)
    p = np.clip(preds["y_score"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    linear_predictor = np.log(p / (1 - p)).reshape(-1, 1)

    try:
        cal_model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000)
        cal_model.fit(linear_predictor, y)
    except Exception:
        cal_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
        cal_model.fit(linear_predictor, y)

    intercept = float(cal_model.intercept_[0])
    slope = float(cal_model.coef_[0][0])
    calibrated = 1.0 / (1.0 + np.exp(-(intercept + slope * linear_predictor.ravel())))
    individual_abs_error = np.abs(calibrated - p)

    table = calibration_table(preds, n_bins=n_bins)
    ece = float(np.average(np.abs(table["observed_rate"] - table["mean_predicted"]), weights=table["n"]))

    rows = [
        {"metric": "calibration_intercept", "estimate": intercept},
        {"metric": "calibration_slope", "estimate": slope},
        {"metric": "brier_score", "estimate": float(brier_score_loss(y, p))},
        {"metric": "expected_calibration_error", "estimate": ece},
        {"metric": "integrated_calibration_index", "estimate": float(np.mean(individual_abs_error))},
        {"metric": "median_absolute_calibration_error", "estimate": float(np.median(individual_abs_error))},
        {"metric": "p90_absolute_calibration_error", "estimate": float(np.quantile(individual_abs_error, 0.90))},
    ]
    return pd.DataFrame(rows)


def decision_curve(preds: pd.DataFrame) -> pd.DataFrame:
    y = preds["y_true"].to_numpy()
    score = preds["y_score"].to_numpy()
    n = len(y)
    prevalence = float(np.mean(y))
    rows = []
    for threshold in np.linspace(0.05, 0.80, 31):
        pred = score >= threshold
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        net_benefit = tp / n - fp / n * (threshold / (1 - threshold))
        treat_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        rows.append({"threshold": threshold, "model": net_benefit, "treat_all": treat_all, "treat_none": 0.0})
    return pd.DataFrame(rows)


def save_roc_pr_plots(preds: pd.DataFrame, out_prefix: Path, title: str) -> None:
    y = preds["y_true"].to_numpy()
    score = preds["y_score"].to_numpy()
    fpr, tpr, _ = roc_curve(y, score)
    precision, recall, _ = precision_recall_curve(y, score)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2)
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title} - ROC")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_roc.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
    ax.plot(recall, precision, color="#2ca02c", linewidth=2)
    ax.axhline(float(np.mean(y)), color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title} - precision-recall")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_pr.png"))
    plt.close(fig)


def save_calibration_plot(table: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", color="#d62728", linewidth=2)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title(f"{title} - calibration")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_decision_curve_plot(dca: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.5), dpi=160)
    ax.plot(dca["threshold"], dca["model"], label="Model", color="#1f77b4", linewidth=2)
    ax.plot(dca["threshold"], dca["treat_all"], label="Treat all", color="#ff7f0e", linestyle="--")
    ax.plot(dca["threshold"], dca["treat_none"], label="Treat none", color="#555555", linestyle=":")
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"{title} - decision curve")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_year_plot(long: pd.DataFrame, out_path: Path) -> None:
    by_year = long.groupby("YEAR_T").agg(n=("FALL_NEXT_YEAR", "size"), rate=("FALL_NEXT_YEAR", "mean")).reset_index()
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    ax1.bar(by_year["YEAR_T"], by_year["n"], color="#8ecae6", label="Person-years")
    ax1.set_ylabel("Person-years")
    ax1.set_xlabel("Predictor year")
    ax2 = ax1.twinx()
    ax2.plot(by_year["YEAR_T"], by_year["rate"], color="#d62728", marker="o", label="Next-year fall rate")
    ax2.set_ylabel("Next-year fall rate")
    ax2.set_ylim(0, max(0.65, by_year["rate"].max() + 0.1))
    ax1.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_performance_plot(best_by_feature: pd.DataFrame, out_path: Path, outcome_name: str) -> None:
    data = best_by_feature[best_by_feature["outcome"] == outcome_name].copy()
    data = data.sort_values("macro_f1_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=160)
    ax.barh(data["feature_set"], data["macro_f1_mean"], color="#457b9d")
    ax.set_xlabel("Mean Macro-F1 across repeated GroupKFold")
    ax.set_ylabel("Feature set")
    ax.set_xlim(0, max(0.8, data["macro_f1_mean"].max() + 0.08))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def permutation_importance_for_config(long: pd.DataFrame, config: pd.Series, out_dir: Path) -> pd.DataFrame:
    outcome = next(o for o in outcomes() if o.name == config["outcome"])
    feature_set = next(fs for fs in feature_sets() if fs.name == config["feature_set"])
    estimator = models()[config["model"]]
    X, y, groups = prepare_data(long, outcome, feature_set)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    pipe = make_pipeline(list(X.columns), estimator)
    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    result = permutation_importance(
        pipe,
        X.iloc[test_idx],
        y.iloc[test_idx],
        scoring="roc_auc",
        n_repeats=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    imp = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp.to_csv(out_dir / f"permutation_importance_{config['outcome']}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
    plot_data = imp.head(12).sort_values("importance_mean")
    ax.barh(plot_data["feature"], plot_data["importance_mean"], xerr=plot_data["importance_std"], color="#6a994e")
    ax.set_xlabel("Permutation importance, ROC-AUC decrease")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"permutation_importance_{config['outcome']}.png")
    plt.close(fig)
    return imp


def cohort_tables(long: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.DataFrame(
        [
            {
                "person_years": len(long),
                "patients": long["PERSON_ID"].nunique(),
                "first_predictor_year": int(long["YEAR_T"].min()),
                "last_predictor_year": int(long["YEAR_T"].max()),
                "next_year_any_fall_n": int(long["FALL_NEXT_YEAR"].sum()),
                "next_year_any_fall_rate": float(long["FALL_NEXT_YEAR"].mean()),
                "next_year_recurrent_fall_n": int(long["RECURRENT_FALL_NEXT_YEAR"].sum()),
                "next_year_high_burden_fall_n": int(long["HIGH_BURDEN_FALL_NEXT_YEAR"].sum()),
                "next_year_death_after_fall_n": int(long["DEATH_AFTER_FALL_NEXT_YEAR"].sum()),
            }
        ]
    )
    cohort.to_csv(out_dir / "cohort_summary.csv", index=False)

    def q1(x: pd.Series) -> float:
        return float(x.quantile(0.25))

    def q3(x: pd.Series) -> float:
        return float(x.quantile(0.75))

    by_target = long.groupby("FALL_NEXT_YEAR").agg(
        n=("FALL_NEXT_YEAR", "size"),
        age_median=("EDAD_T", "median"),
        age_q1=("EDAD_T", q1),
        age_q3=("EDAD_T", q3),
        female_rate=("SEXO", "mean"),
        same_year_fall_rate=("FALL_T", "mean"),
        median_cumulative_falls=("CAIDAS_ACUM_T", "median"),
        functional_available_rate=("HAS_FUNCTIONAL_T", "mean"),
    ).reset_index()
    by_target.to_csv(out_dir / "baseline_characteristics_by_next_year_fall.csv", index=False)
    return cohort, by_target


def write_report(
    long: pd.DataFrame,
    summary: pd.DataFrame,
    best: pd.DataFrame,
    ci_tables: dict[str, pd.DataFrame],
    cohort: pd.DataFrame,
    by_target: pd.DataFrame,
    out_dir: Path,
) -> None:
    cols = [
        "outcome",
        "outcome_label",
        "feature_set",
        "model",
        "n_total",
        "n_groups_total",
        "target_positive",
        "macro_f1_mean",
        "macro_f1_std",
        "balanced_accuracy_mean",
        "sensitivity_mean",
        "specificity_mean",
        "ppv_mean",
        "roc_auc_mean",
        "pr_auc_mean",
        "brier_mean",
    ]
    best_display = best[cols].copy()
    text = [
        "# Article-ready analysis package",
        "",
        "This package rebuilds a longitudinal person-year dataset and evaluates prospective prediction tasks using grouped validation by patient.",
        "",
        "## Cohort",
        "",
        dataframe_to_markdown(cohort),
        "",
        "## Baseline characteristics by next-year fall outcome",
        "",
        dataframe_to_markdown(by_target),
        "",
        "## Best non-dummy model per outcome",
        "",
        dataframe_to_markdown(best_display),
        "",
        "## Bootstrap CIs for selected out-of-fold predictions",
        "",
    ]
    for name, table in ci_tables.items():
        text.extend([f"### {name}", "", dataframe_to_markdown(table), ""])
    text.extend(
        [
            "## Interpretation",
            "",
            "- The strongest defensible signal is prospective next-year fall prediction.",
            "- Recurrent and high-burden fall outcomes are feasible but should be framed as secondary/exploratory.",
            "- Death after fall remains too rare for a strong standalone model in this dataset.",
            "- Validation is grouped by patient; no row-level leakage is allowed in the primary estimates.",
            "- The current evidence is suitable for a development/internal-validation manuscript, not for deployment claims without external validation.",
            "",
            "## Files",
            "",
            "- `longitudinal_article_dataset.csv`: reconstructed person-year dataset.",
            "- `model_fold_metrics.csv`: fold-level metrics.",
            "- `model_summary.csv`: repeated grouped CV summary.",
            "- `predictions_*.csv`: aggregated out-of-fold predictions for the selected models.",
            "- `oof_predictions.csv`: full out-of-fold predictions, written only when `--include-oof-predictions` is used.",
            "- `figures/*.png`: ROC, PR, calibration, decision curve, yearly distribution, and permutation importance plots.",
        ]
    )
    (out_dir / "article_report.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input-data", type=Path, default=Path("data") / "processed" / "person_year_dataset.csv")
    parser.add_argument("--raw-wide", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ARTICLE_DIR)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--include-oof-predictions", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    input_data = args.input_data if args.input_data.is_absolute() else args.root / args.input_data
    if input_data.exists():
        long = pd.read_csv(input_data, dtype={"PERSON_ID": str})
    else:
        if args.raw_wide is None:
            raise FileNotFoundError(
                f"Could not find {input_data}. Provide --input-data or pass a source wide CSV with --raw-wide."
            )
        wide_path = args.raw_wide
        wide = pd.read_csv(wide_path)
        long = build_longitudinal_dataset(wide)
        long["PERSON_ID"] = long["PERSON_ID"].astype(str)
    long.to_csv(out_dir / "longitudinal_article_dataset.csv", index=False)

    cohort, by_target = cohort_tables(long, out_dir)
    save_year_plot(long, fig_dir / "person_years_and_next_year_fall_rate.png")

    metrics, predictions = run_repeated_group_cv(long, args.repeats, args.folds)
    metrics.to_csv(out_dir / "model_fold_metrics.csv", index=False)
    if args.include_oof_predictions:
        predictions.to_csv(out_dir / "oof_predictions.csv", index=False)
    summary = summarize_metrics(metrics)
    summary.to_csv(out_dir / "model_summary.csv", index=False)

    best = best_non_dummy(summary)
    best.to_csv(out_dir / "best_models.csv", index=False)

    best_by_feature = (
        summary[~summary["model"].str.startswith("dummy")]
        .sort_values(["outcome", "feature_set", "macro_f1_mean"], ascending=[True, True, False])
        .groupby(["outcome", "feature_set"], as_index=False)
        .head(1)
    )
    save_performance_plot(best_by_feature, fig_dir / "fall_next_year_feature_set_performance.png", "fall_next_year")

    ci_tables: dict[str, pd.DataFrame] = {}
    for _, config in best.iterrows():
        agg_preds = aggregate_oof_predictions(predictions, config)
        pred_name = f"{config['outcome']}_{config['feature_set']}_{config['model']}"
        agg_preds.to_csv(out_dir / f"predictions_{pred_name}.csv", index=False)
        ci = bootstrap_metric_ci(agg_preds, n_boot=args.bootstrap)
        ci.to_csv(out_dir / f"bootstrap_ci_{pred_name}.csv", index=False)
        ci_tables[config["outcome"]] = ci

        cal = calibration_table(agg_preds)
        cal.to_csv(out_dir / f"calibration_{pred_name}.csv", index=False)
        cal_metrics = calibration_metrics(agg_preds)
        cal_metrics.to_csv(out_dir / f"calibration_metrics_{pred_name}.csv", index=False)
        dca = decision_curve(agg_preds)
        dca.to_csv(out_dir / f"decision_curve_{pred_name}.csv", index=False)

        if config["outcome"] == "fall_next_year":
            title = "Next-year fall prediction"
            save_roc_pr_plots(agg_preds, fig_dir / "fall_next_year_best", title)
            save_calibration_plot(cal, fig_dir / "fall_next_year_calibration.png", title)
            save_decision_curve_plot(dca, fig_dir / "fall_next_year_decision_curve.png", title)
            permutation_importance_for_config(long, config, out_dir)

    write_report(long, summary, best, ci_tables, cohort, by_target, out_dir)
    print(f"Wrote article package to {out_dir}")


if __name__ == "__main__":
    main()
