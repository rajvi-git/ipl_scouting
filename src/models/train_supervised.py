"""Train role-specific supervised impact and tier models with baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.constants import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.features.impact_score import BAT_IMPACT_WEIGHTS, BOWL_IMPACT_WEIGHTS
from src.models.build_training_table import TRAINING_PATH, build_training_table

FEATURES_PATH = PROCESSED_DIR / "player_features.parquet"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"

TIER_ORDER = ["bust", "marginal", "solid", "star"]
ROLES = ("bat", "bowl")
MAX_ITER = 250
EARLY_STOPPING_ROUNDS = 10
VALIDATION_FRACTION = 0.2
TOL = 1e-4

ROLE_MODEL_PATHS = {
    "bat": {
        "regressor": MODELS_DIR / "bat_impact_regressor.joblib",
        "classifier": MODELS_DIR / "bat_tier_classifier.joblib",
    },
    "bowl": {
        "regressor": MODELS_DIR / "bowl_impact_regressor.joblib",
        "classifier": MODELS_DIR / "bowl_tier_classifier.joblib",
    },
}


def _save_model(model: Pipeline, path: Path) -> None:
    joblib.dump(model, path)


def _role_feature_columns(table: pd.DataFrame, role: str) -> list[str]:
    common = ["sma_seasons", "sma_start_season", "sma_end_season"]
    if role == "bat":
        wanted = common + [
            "sma_bat_innings",
            "sma_runs",
            "sma_balls",
            "sma_fours",
            "sma_sixes",
            "sma_bat_pp_runs",
            "sma_bat_pp_balls",
            "sma_bat_mid_runs",
            "sma_bat_mid_balls",
            "sma_bat_death_runs",
            "sma_bat_death_balls",
            "sma_strike_rate",
            "sma_runs_per_innings",
            "sma_boundary_pct",
            "sma_bat_pp_sr",
            "sma_bat_mid_sr",
            "sma_bat_death_sr",
            "sma_caf_strike_rate",
            "sma_caf_runs_per_innings",
            "sma_caf_boundary_pct",
            "sma_caf_bat_pp_sr",
            "sma_caf_bat_mid_sr",
            "sma_caf_bat_death_sr",
            "sma_shrunk_strike_rate",
            "sma_shrunk_runs_per_innings",
            "sma_shrunk_boundary_pct",
            "sma_shrunk_bat_pp_sr",
            "sma_shrunk_bat_mid_sr",
            "sma_shrunk_bat_death_sr",
        ]
    else:
        wanted = common + [
            "sma_bowl_innings",
            "sma_runs_conceded",
            "sma_balls_bowled",
            "sma_wickets",
            "sma_bowl_pp_runs",
            "sma_bowl_pp_balls",
            "sma_bowl_mid_runs",
            "sma_bowl_mid_balls",
            "sma_bowl_death_runs",
            "sma_bowl_death_balls",
            "sma_wides",
            "sma_noballs",
            "sma_economy",
            "sma_bowling_sr",
            "sma_wickets_per_innings",
            "sma_wide_rate",
            "sma_noball_rate",
            "sma_bowl_pp_economy",
            "sma_bowl_mid_economy",
            "sma_bowl_death_economy",
            "sma_caf_economy",
            "sma_caf_bowling_sr",
            "sma_caf_wickets_per_innings",
            "sma_caf_bowl_pp_economy",
            "sma_caf_bowl_mid_economy",
            "sma_caf_bowl_death_economy",
            "sma_shrunk_economy",
            "sma_shrunk_bowling_sr",
            "sma_shrunk_wickets_per_innings",
            "sma_shrunk_bowl_pp_economy",
            "sma_shrunk_bowl_mid_economy",
            "sma_shrunk_bowl_death_economy",
        ]
    return [col for col in wanted if col in table.columns]


def _numeric_preprocessor(scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps=steps)


def _regressor() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _numeric_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=MAX_ITER,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=VALIDATION_FRACTION,
                    n_iter_no_change=EARLY_STOPPING_ROUNDS,
                    tol=TOL,
                    random_state=42,
                ),
            ),
        ]
    )


def _classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _numeric_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=MAX_ITER,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=VALIDATION_FRACTION,
                    n_iter_no_change=EARLY_STOPPING_ROUNDS,
                    tol=TOL,
                    random_state=42,
                ),
            ),
        ]
    )


def _ridge() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _numeric_preprocessor(scale=True)),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def _spearman(y_true: pd.Series, y_pred: np.ndarray) -> float:
    corr = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(corr) if pd.notna(corr) else float("nan")


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _regression_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "spearman": _spearman(y_true, np.asarray(y_pred)),
    }


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict:
    labels = [tier for tier in TIER_ORDER if tier in set(y_true) or tier in set(pd.Series(y_pred))]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
    }


def _fold_splits(table: pd.DataFrame, y: pd.Series):
    groups = table["player_key"]
    n_splits = min(5, groups.nunique())
    return n_splits, GroupKFold(n_splits=n_splits).split(table, y, groups)


def _caf_formula_scores(x: pd.DataFrame, role: str) -> pd.Series:
    if role == "bat":
        weights = BAT_IMPACT_WEIGHTS
    else:
        weights = BOWL_IMPACT_WEIGHTS

    score = pd.Series(0.0, index=x.index)
    used_weight = 0.0
    for stat, weight in weights.items():
        candidates = [
            f"sma_shrunk_{stat}",
            f"sma_caf_{stat}",
            f"sma_{stat}",
        ]
        col = next((name for name in candidates if name in x.columns), None)
        if col is None:
            continue
        values = x[col]
        std = values.std(ddof=0)
        if pd.isna(std) or std == 0:
            continue
        z = (values - values.mean()) / std
        score = score.add(z.fillna(0.0) * weight, fill_value=0.0)
        used_weight += abs(weight)

    if used_weight == 0:
        return pd.Series(0.0, index=x.index)
    return score / used_weight


def _fit_thresholds(y_impact: pd.Series, y_tier: pd.Series) -> dict[str, float]:
    thresholds = {}
    for tier in ("bust", "marginal", "solid"):
        values = y_impact[y_tier == tier]
        thresholds[tier] = float(values.max()) if len(values) else float(y_impact.quantile(0.5))
    return thresholds


def _impact_to_tier(pred: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    out = pd.Series("star", index=pred.index, dtype=object)
    out[pred <= thresholds["solid"]] = "solid"
    out[pred <= thresholds["marginal"]] = "marginal"
    out[pred <= thresholds["bust"]] = "bust"
    return out


def _evaluate_regressor_cv(role_table: pd.DataFrame, feature_cols: list[str], model_factory, name: str) -> dict:
    x = role_table[feature_cols]
    y = role_table["y_impact"].astype(float)
    n_splits, splits = _fold_splits(role_table, y)
    preds = pd.Series(index=role_table.index, dtype=float)
    folds = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model = model_factory()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[test_idx])
        preds.iloc[test_idx] = pred
        folds.append({"fold": fold, **_regression_metrics(y.iloc[test_idx], pred), "n_test": int(len(test_idx))})

    valid = preds.notna()
    return {
        "model": name,
        "n_splits": int(n_splits),
        **_regression_metrics(y[valid], preds[valid]),
        "folds": folds,
    }


def _evaluate_classifier_cv(role_table: pd.DataFrame, feature_cols: list[str], model_factory, name: str) -> dict:
    x = role_table[feature_cols]
    y = role_table["tier"].astype(str)
    n_splits, splits = _fold_splits(role_table, y)
    preds = pd.Series(index=role_table.index, dtype=object)

    for train_idx, test_idx in splits:
        model = model_factory()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[test_idx] = model.predict(x.iloc[test_idx])

    valid = preds.notna()
    return {
        "model": name,
        "n_splits": int(n_splits),
        **_classification_metrics(y[valid], preds[valid]),
    }


def _evaluate_role_mean_baseline(role_table: pd.DataFrame) -> dict:
    y = role_table["y_impact"].astype(float)
    n_splits, splits = _fold_splits(role_table, y)
    preds = pd.Series(index=role_table.index, dtype=float)

    for train_idx, test_idx in splits:
        pred = float(y.iloc[train_idx].mean())
        preds.iloc[test_idx] = pred

    return {
        "model": "role_mean",
        "n_splits": int(n_splits),
        **_regression_metrics(y, preds),
    }


def _evaluate_caf_formula_baseline(role_table: pd.DataFrame, feature_cols: list[str], role: str) -> dict:
    y = role_table["y_impact"].astype(float)
    scores = _caf_formula_scores(role_table[feature_cols], role)
    return {
        "model": "caf_formula",
        **_regression_metrics(y, scores),
    }


def _evaluate_majority_baseline(role_table: pd.DataFrame) -> dict:
    y = role_table["tier"].astype(str)
    n_splits, splits = _fold_splits(role_table, y)
    preds = pd.Series(index=role_table.index, dtype=object)

    for train_idx, test_idx in splits:
        majority = y.iloc[train_idx].mode().iloc[0]
        preds.iloc[test_idx] = majority

    return {
        "model": "majority_class",
        "n_splits": int(n_splits),
        **_classification_metrics(y, preds),
    }


def _evaluate_impact_threshold_baseline(role_table: pd.DataFrame, feature_cols: list[str]) -> dict:
    x = role_table[feature_cols]
    y_impact = role_table["y_impact"].astype(float)
    y_tier = role_table["tier"].astype(str)
    n_splits, splits = _fold_splits(role_table, y_tier)
    preds = pd.Series(index=role_table.index, dtype=object)

    for train_idx, test_idx in splits:
        model = DummyRegressor(strategy="mean")
        model.fit(x.iloc[train_idx], y_impact.iloc[train_idx])
        impact_pred = pd.Series(model.predict(x.iloc[test_idx]), index=role_table.index[test_idx])
        thresholds = _fit_thresholds(y_impact.iloc[train_idx], y_tier.iloc[train_idx])
        preds.iloc[test_idx] = _impact_to_tier(impact_pred, thresholds)

    return {
        "model": "impact_threshold_from_regression_baseline",
        "n_splits": int(n_splits),
        **_classification_metrics(y_tier, preds),
    }


def _first_group_split(role_table: pd.DataFrame, y: pd.Series):
    n_splits, splits = _fold_splits(role_table, y)
    return next(splits)


def _save_regression_learning_curve(role: str, role_table: pd.DataFrame, feature_cols: list[str]) -> dict:
    import matplotlib.pyplot as plt

    x = role_table[feature_cols]
    y = role_table["y_impact"].astype(float)
    train_idx, test_idx = _first_group_split(role_table, y)

    model = _regressor()
    model.fit(x.iloc[train_idx], y.iloc[train_idx])
    trained_model = model.named_steps["model"]
    x_train = model.named_steps["preprocess"].transform(x.iloc[train_idx])
    x_test = model.named_steps["preprocess"].transform(x.iloc[test_idx])

    rows = []
    for iteration, (train_pred, test_pred) in enumerate(
        zip(trained_model.staged_predict(x_train), trained_model.staged_predict(x_test)),
        start=1,
    ):
        rows.append(
            {
                "iteration": iteration,
                "train_mae": float(mean_absolute_error(y.iloc[train_idx], train_pred)),
                "validation_mae": float(mean_absolute_error(y.iloc[test_idx], test_pred)),
                "train_rmse": _rmse(y.iloc[train_idx], train_pred),
                "validation_rmse": _rmse(y.iloc[test_idx], test_pred),
            }
        )

    curve = pd.DataFrame(rows)
    csv_path = REPORTS_DIR / f"{role}_impact_regression_learning_curve.csv"
    png_path = REPORTS_DIR / f"{role}_impact_regression_learning_curve.png"
    curve.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(curve["iteration"], curve["train_mae"], label="Train MAE")
    plt.plot(curve["iteration"], curve["validation_mae"], label="Validation MAE")
    plt.xlabel("Boosting iteration")
    plt.ylabel("MAE")
    plt.title(f"{role.title()} Impact Regression Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    best_idx = curve["validation_mae"].idxmin()
    return {
        "csv": str(csv_path),
        "png": str(png_path),
        "best_iteration": int(curve.loc[best_idx, "iteration"]),
        "best_validation_mae": float(curve.loc[best_idx, "validation_mae"]),
        "final_validation_mae": float(curve.iloc[-1]["validation_mae"]),
    }


def _save_classification_learning_curve(role: str, role_table: pd.DataFrame, feature_cols: list[str]) -> dict:
    import matplotlib.pyplot as plt

    x = role_table[feature_cols]
    y = role_table["tier"].astype(str)
    train_idx, test_idx = _first_group_split(role_table, y)

    model = _classifier()
    model.fit(x.iloc[train_idx], y.iloc[train_idx])
    trained_model = model.named_steps["model"]
    x_train = model.named_steps["preprocess"].transform(x.iloc[train_idx])
    x_test = model.named_steps["preprocess"].transform(x.iloc[test_idx])

    rows = []
    labels = list(trained_model.classes_)
    for iteration, (train_proba, test_proba) in enumerate(
        zip(trained_model.staged_predict_proba(x_train), trained_model.staged_predict_proba(x_test)),
        start=1,
    ):
        rows.append(
            {
                "iteration": iteration,
                "train_log_loss": float(log_loss(y.iloc[train_idx], train_proba, labels=labels)),
                "validation_log_loss": float(log_loss(y.iloc[test_idx], test_proba, labels=labels)),
            }
        )

    curve = pd.DataFrame(rows)
    csv_path = REPORTS_DIR / f"{role}_tier_classification_learning_curve.csv"
    png_path = REPORTS_DIR / f"{role}_tier_classification_learning_curve.png"
    curve.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(curve["iteration"], curve["train_log_loss"], label="Train log loss")
    plt.plot(curve["iteration"], curve["validation_log_loss"], label="Validation log loss")
    plt.xlabel("Boosting iteration")
    plt.ylabel("Log loss")
    plt.title(f"{role.title()} Tier Classification Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    best_idx = curve["validation_log_loss"].idxmin()
    return {
        "csv": str(csv_path),
        "png": str(png_path),
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


def _comparison(role_metrics: dict) -> dict:
    regression = role_metrics["regression"]
    classification = role_metrics["classification"]
    baselines = role_metrics["baselines"]
    return {
        "regression_mae": {
            "role_mean": baselines["regression"]["role_mean"]["mae"],
            "caf_formula": baselines["regression"]["caf_formula"]["mae"],
            "ridge": baselines["regression"]["ridge"]["mae"],
            "hist_gradient_boosting": regression["hist_gradient_boosting"]["mae"],
        },
        "classification_macro_f1": {
            "majority_class": baselines["classification"]["majority_class"]["macro_f1"],
            "impact_threshold_from_regression_baseline": baselines["classification"][
                "impact_threshold_from_regression_baseline"
            ]["macro_f1"],
            "hist_gradient_boosting": classification["hist_gradient_boosting"]["macro_f1"],
        },
    }


def _train_role(role: str, table: pd.DataFrame) -> tuple[dict, dict]:
    role_table = table[table["role"] == role].reset_index(drop=True)
    feature_cols = _role_feature_columns(role_table, role)
    if role_table.empty:
        raise ValueError(f"No training rows for role: {role}")
    if not feature_cols:
        raise ValueError(f"No feature columns for role: {role}")

    metrics = {
        "rows": int(len(role_table)),
        "players": int(role_table["player_key"].nunique()),
        "tier_counts": role_table["tier"].value_counts().to_dict(),
        "feature_columns": feature_cols,
        "baselines": {
            "regression": {
                "role_mean": _evaluate_role_mean_baseline(role_table),
                "caf_formula": _evaluate_caf_formula_baseline(role_table, feature_cols, role),
                "ridge": _evaluate_regressor_cv(role_table, feature_cols, _ridge, "ridge"),
            },
            "classification": {
                "majority_class": _evaluate_majority_baseline(role_table),
                "impact_threshold_from_regression_baseline": _evaluate_impact_threshold_baseline(
                    role_table, feature_cols
                ),
            },
        },
        "regression": {
            "hist_gradient_boosting": _evaluate_regressor_cv(
                role_table, feature_cols, _regressor, "hist_gradient_boosting_regressor"
            )
        },
        "classification": {
            "hist_gradient_boosting": _evaluate_classifier_cv(
                role_table, feature_cols, _classifier, "hist_gradient_boosting_classifier"
            )
        },
        "learning_curves": {
            "impact_regression": _save_regression_learning_curve(role, role_table, feature_cols),
            "tier_classification": _save_classification_learning_curve(role, role_table, feature_cols),
        },
    }
    metrics["comparison"] = _comparison(metrics)

    regressor = _regressor()
    regressor.fit(role_table[feature_cols], role_table["y_impact"].astype(float))
    classifier = _classifier()
    classifier.fit(role_table[feature_cols], role_table["tier"].astype(str))
    _save_model(regressor, ROLE_MODEL_PATHS[role]["regressor"])
    _save_model(classifier, ROLE_MODEL_PATHS[role]["classifier"])

    final_models = {
        "impact_regressor_n_iter": int(regressor.named_steps["model"].n_iter_),
        "tier_classifier_n_iter": int(classifier.named_steps["model"].n_iter_),
        "impact_regressor_path": str(ROLE_MODEL_PATHS[role]["regressor"]),
        "tier_classifier_path": str(ROLE_MODEL_PATHS[role]["classifier"]),
    }
    return metrics, final_models


def train_supervised() -> dict:
    table = _load_or_build_training_table()
    if table.empty:
        raise ValueError("Training table is empty. Check SMA-before-IPL overlap in player_features.parquet.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "training_rows": int(len(table)),
        "players": int(table["player_key"].nunique()),
        "role_counts": table["role"].value_counts().to_dict(),
        "tier_counts": table["tier"].value_counts().to_dict(),
        "early_stopping": {
            "enabled": True,
            "max_iter": MAX_ITER,
            "validation_fraction": VALIDATION_FRACTION,
            "n_iter_no_change": EARLY_STOPPING_ROUNDS,
            "tol": TOL,
        },
        "role_models": {},
        "baselines": {},
        "final_models": {},
    }

    feature_metadata = {}
    for role in ROLES:
        role_metrics, final_models = _train_role(role, table)
        metrics["role_models"][role] = {
            key: value
            for key, value in role_metrics.items()
            if key not in {"baselines", "feature_columns"}
        }
        metrics["baselines"][role] = role_metrics["baselines"]
        metrics["final_models"][role] = final_models
        feature_metadata[role] = role_metrics["feature_columns"]

    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_metadata, f, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    metrics = train_supervised()
    print(f"Saved training metrics to {METRICS_PATH}")
    for role, paths in ROLE_MODEL_PATHS.items():
        print(f"Saved {role} regressor to {paths['regressor']}")
        print(f"Saved {role} classifier to {paths['classifier']}")
    print(f"Training rows: {metrics['training_rows']}")


if __name__ == "__main__":
    main()
