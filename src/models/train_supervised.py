"""Train supervised impact regression and tier classification models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.constants import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.models.build_training_table import TRAINING_PATH, build_training_table

FEATURES_PATH = PROCESSED_DIR / "player_features.parquet"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
REGRESSION_CURVE_CSV = REPORTS_DIR / "impact_regression_learning_curve.csv"
REGRESSION_CURVE_PNG = REPORTS_DIR / "impact_regression_learning_curve.png"
CLASSIFICATION_CURVE_CSV = REPORTS_DIR / "tier_classification_learning_curve.csv"
CLASSIFICATION_CURVE_PNG = REPORTS_DIR / "tier_classification_learning_curve.png"
REGRESSOR_PATH = MODELS_DIR / "impact_regressor.joblib"
CLASSIFIER_PATH = MODELS_DIR / "tier_classifier.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"

TIER_ORDER = ["bust", "marginal", "solid", "star"]


def _save_model(model: Pipeline, path: Path) -> None:
    joblib.dump(model, path)


def _training_features(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_cols = ["role"] + sorted(c for c in table.columns if c.startswith("sma_"))
    x = table[feature_cols].copy()
    categorical = ["role"]
    numeric = [c for c in feature_cols if c not in categorical]
    return x, numeric, categorical


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def _regressor(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _preprocessor(numeric, categorical)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=250,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def _classifier(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _preprocessor(numeric, categorical)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=250,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def _spearman(y_true: pd.Series, y_pred: np.ndarray) -> float:
    corr = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(corr) if pd.notna(corr) else float("nan")


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evaluate_regression(table: pd.DataFrame, x: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    groups = table["player_key"]
    y = table["y_impact"].astype(float)
    n_groups = groups.nunique()
    if n_groups < 2 or len(table) < 4:
        return {"skipped": "not enough player groups for GroupKFold"}

    n_splits = min(5, n_groups)
    preds = pd.Series(index=table.index, dtype=float)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(x, y, groups), start=1):
        model = _regressor(numeric, categorical)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[test_idx])
        preds.iloc[test_idx] = pred
        fold_metrics.append(
            {
                "fold": fold,
                "mae": float(mean_absolute_error(y.iloc[test_idx], pred)),
                "rmse": _rmse(y.iloc[test_idx], pred),
                "spearman": _spearman(y.iloc[test_idx], pred),
                "n_test": int(len(test_idx)),
            }
        )

    valid = preds.notna()
    return {
        "model": "sklearn_hist_gradient_boosting_regressor",
        "n_splits": int(n_splits),
        "mae": float(mean_absolute_error(y[valid], preds[valid])),
        "rmse": _rmse(y[valid], preds[valid]),
        "spearman": _spearman(y[valid], preds[valid].to_numpy()),
        "folds": fold_metrics,
    }


def _evaluate_classification(table: pd.DataFrame, x: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    groups = table["player_key"]
    y = table["tier"].astype(str)
    n_groups = groups.nunique()
    if n_groups < 2 or y.nunique() < 2:
        return {"skipped": "not enough player groups/classes for GroupKFold"}

    n_splits = min(5, n_groups)
    preds = pd.Series(index=table.index, dtype=object)

    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y, groups):
        model = _classifier(numeric, categorical)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[test_idx] = model.predict(x.iloc[test_idx])

    valid = preds.notna()
    labels = [tier for tier in TIER_ORDER if tier in set(y) or tier in set(preds.dropna())]
    return {
        "model": "sklearn_hist_gradient_boosting_classifier",
        "n_splits": int(n_splits),
        "accuracy": float(accuracy_score(y[valid], preds[valid])),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y[valid], preds[valid], labels=labels).tolist(),
        "classification_report": classification_report(
            y[valid],
            preds[valid],
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def _first_group_split(
    table: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    groups = table["player_key"]
    n_splits = min(5, groups.nunique())
    return next(GroupKFold(n_splits=n_splits).split(x, y, groups))


def _save_regression_learning_curve(
    table: pd.DataFrame,
    x: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> dict:
    import matplotlib.pyplot as plt

    y = table["y_impact"].astype(float)
    train_idx, test_idx = _first_group_split(table, x, y)
    preprocessor = _preprocessor(numeric, categorical)
    x_train = preprocessor.fit_transform(x.iloc[train_idx])
    x_test = preprocessor.transform(x.iloc[test_idx])
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(x_train, y_train)

    rows = []
    for iteration, (train_pred, test_pred) in enumerate(
        zip(model.staged_predict(x_train), model.staged_predict(x_test)),
        start=1,
    ):
        rows.append(
            {
                "iteration": iteration,
                "train_mae": float(mean_absolute_error(y_train, train_pred)),
                "validation_mae": float(mean_absolute_error(y_test, test_pred)),
                "train_rmse": _rmse(y_train, train_pred),
                "validation_rmse": _rmse(y_test, test_pred),
            }
        )

    curve = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(REGRESSION_CURVE_CSV, index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(curve["iteration"], curve["train_mae"], label="Train MAE")
    plt.plot(curve["iteration"], curve["validation_mae"], label="Validation MAE")
    plt.xlabel("Boosting iteration")
    plt.ylabel("MAE")
    plt.title("Impact Regression Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REGRESSION_CURVE_PNG, dpi=160)
    plt.close()

    best_idx = curve["validation_mae"].idxmin()
    return {
        "csv": str(REGRESSION_CURVE_CSV),
        "png": str(REGRESSION_CURVE_PNG),
        "best_iteration": int(curve.loc[best_idx, "iteration"]),
        "best_validation_mae": float(curve.loc[best_idx, "validation_mae"]),
        "final_validation_mae": float(curve.iloc[-1]["validation_mae"]),
    }


def _save_classification_learning_curve(
    table: pd.DataFrame,
    x: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> dict:
    import matplotlib.pyplot as plt

    y = table["tier"].astype(str)
    train_idx, test_idx = _first_group_split(table, x, y)
    preprocessor = _preprocessor(numeric, categorical)
    x_train = preprocessor.fit_transform(x.iloc[train_idx])
    x_test = preprocessor.transform(x.iloc[test_idx])
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(x_train, y_train)

    rows = []
    labels = list(model.classes_)
    for iteration, (train_proba, test_proba) in enumerate(
        zip(model.staged_predict_proba(x_train), model.staged_predict_proba(x_test)),
        start=1,
    ):
        rows.append(
            {
                "iteration": iteration,
                "train_log_loss": float(log_loss(y_train, train_proba, labels=labels)),
                "validation_log_loss": float(log_loss(y_test, test_proba, labels=labels)),
            }
        )

    curve = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(CLASSIFICATION_CURVE_CSV, index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(curve["iteration"], curve["train_log_loss"], label="Train log loss")
    plt.plot(curve["iteration"], curve["validation_log_loss"], label="Validation log loss")
    plt.xlabel("Boosting iteration")
    plt.ylabel("Log loss")
    plt.title("Tier Classification Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CLASSIFICATION_CURVE_PNG, dpi=160)
    plt.close()

    best_idx = curve["validation_log_loss"].idxmin()
    return {
        "csv": str(CLASSIFICATION_CURVE_CSV),
        "png": str(CLASSIFICATION_CURVE_PNG),
        "best_iteration": int(curve.loc[best_idx, "iteration"]),
        "best_validation_log_loss": float(curve.loc[best_idx, "validation_log_loss"]),
        "final_validation_log_loss": float(curve.iloc[-1]["validation_log_loss"]),
    }


def _load_or_build_training_table() -> pd.DataFrame:
    if TRAINING_PATH.exists():
        return pd.read_parquet(TRAINING_PATH)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features table: {FEATURES_PATH}. Run run_features.py first.")

    features = pd.read_parquet(FEATURES_PATH)
    table, _ = build_training_table(features)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(TRAINING_PATH, index=False)
    return table


def train_supervised() -> dict:
    table = _load_or_build_training_table()
    if table.empty:
        raise ValueError("Training table is empty. Check SMA-before-IPL overlap in player_features.parquet.")

    x, numeric, categorical = _training_features(table)
    metrics = {
        "training_rows": int(len(table)),
        "players": int(table["player_key"].nunique()),
        "role_counts": table["role"].value_counts().to_dict(),
        "tier_counts": table["tier"].value_counts().to_dict(),
        "feature_columns": list(x.columns),
    }

    metrics["impact_regression"] = _evaluate_regression(table, x, numeric, categorical)
    metrics["tier_classification"] = _evaluate_classification(table, x, numeric, categorical)
    metrics["learning_curves"] = {
        "impact_regression": _save_regression_learning_curve(table, x, numeric, categorical),
        "tier_classification": _save_classification_learning_curve(table, x, numeric, categorical),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    regressor = _regressor(numeric, categorical)
    regressor.fit(x, table["y_impact"].astype(float))
    _save_model(regressor, REGRESSOR_PATH)

    classifier = _classifier(numeric, categorical)
    classifier.fit(x, table["tier"].astype(str))
    _save_model(classifier, CLASSIFIER_PATH)

    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump({"numeric": numeric, "categorical": categorical, "all": list(x.columns)}, f, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    metrics = train_supervised()
    print(f"Saved training metrics to {METRICS_PATH}")
    print(f"Saved regressor to {REGRESSOR_PATH}")
    print(f"Saved classifier to {CLASSIFIER_PATH}")
    print(f"Training rows: {metrics['training_rows']}")


if __name__ == "__main__":
    main()
