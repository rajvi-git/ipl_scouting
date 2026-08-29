"""Generate EDA tables and charts for SMA/IPL scouting data."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.constants import PROCESSED_DIR, REPORTS_DIR
from src.features.impact_score import BAT_IMPACT_WEIGHTS, BOWL_IMPACT_WEIGHTS

FEATURES_PATH = PROCESSED_DIR / "player_features.parquet"
TRAINING_PATH = PROCESSED_DIR / "ml_training_pairs.parquet"
EDA_DIR = REPORTS_DIR / "eda"

BAT_METRICS = [
    "bat_innings",
    "runs",
    "balls",
    "strike_rate",
    "runs_per_innings",
    "boundary_pct",
    "bat_pp_sr",
    "bat_mid_sr",
    "bat_death_sr",
    "shrunk_strike_rate",
    "shrunk_runs_per_innings",
    "shrunk_boundary_pct",
]

BOWL_METRICS = [
    "bowl_innings",
    "runs_conceded",
    "balls_bowled",
    "wickets",
    "economy",
    "bowling_sr",
    "wickets_per_innings",
    "bowl_pp_economy",
    "bowl_mid_economy",
    "bowl_death_economy",
    "shrunk_economy",
    "shrunk_bowling_sr",
    "shrunk_wickets_per_innings",
]


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def _save_csv(df: pd.DataFrame, name: str) -> Path:
    path = EDA_DIR / name
    df.to_csv(path, index=False)
    return path


def _save_fig(name: str) -> Path:
    path = EDA_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))

    headers = [str(col) for col in display.columns]
    rows = display.values.tolist()
    widths = [
        max(len(headers[i]), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    divider = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def _weighted_metric_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for (competition, role), group in df.groupby(["competition", "role"], dropna=False):
        for metric in _available(group, metrics):
            values = group[metric].dropna()
            rows.append(
                {
                    "competition": competition,
                    "role": role,
                    "metric": metric,
                    "non_null": int(values.count()),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "std": float(values.std(ddof=0)) if len(values) else np.nan,
                    "min": float(values.min()) if len(values) else np.nan,
                    "q25": float(values.quantile(0.25)) if len(values) else np.nan,
                    "q75": float(values.quantile(0.75)) if len(values) else np.nan,
                    "max": float(values.max()) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _add_observed_impact(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["impact_raw"] = np.nan
    out["impact_reliability"] = np.nan
    out["impact_score"] = np.nan
    out["impact_percentile"] = np.nan

    role_specs = {
        "bat": (BAT_IMPACT_WEIGHTS, "bat_innings"),
        "bowl": (BOWL_IMPACT_WEIGHTS, "bowl_innings"),
    }
    for (competition, role), group in out.groupby(["competition", "role"], dropna=False):
        if role not in role_specs:
            continue
        weights, sample_col = role_specs[role]
        idx = group.index
        score = pd.Series(0.0, index=idx)
        used_weight = 0.0

        for metric, weight in weights.items():
            if metric not in out.columns:
                continue
            values = out.loc[idx, metric]
            if values.notna().sum() < 2:
                continue
            score = score.add(_zscore(values).fillna(0.0) * weight, fill_value=0.0)
            used_weight += abs(weight)

        if used_weight == 0:
            continue

        raw = score / used_weight
        sample = out.loc[idx, sample_col].fillna(0).astype(float) if sample_col in out.columns else 0.0
        reliability = sample / (sample + 10.0)
        role_mean = raw.mean()
        impact = reliability * raw + (1.0 - reliability) * role_mean

        out.loc[idx, "impact_raw"] = raw
        out.loc[idx, "impact_reliability"] = reliability
        out.loc[idx, "impact_score"] = impact
        out.loc[idx, "impact_percentile"] = impact.rank(pct=True) * 100.0

    return out


def _overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (competition, role), group in df.groupby(["competition", "role"], dropna=False):
        rows.append(
            {
                "competition": competition,
                "role": role,
                "player_season_rows": int(len(group)),
                "players": int(group["player_key"].nunique()),
                "seasons": int(group["season"].nunique()),
                "first_season": int(group["season"].min()),
                "last_season": int(group["season"].max()),
                "total_bat_innings": float(group["bat_innings"].sum(skipna=True)),
                "total_bowl_innings": float(group["bowl_innings"].sum(skipna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["competition", "role"])


def _missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (competition, role), group in df.groupby(["competition", "role"], dropna=False):
        for col in BAT_METRICS + BOWL_METRICS:
            if col not in group.columns:
                continue
            rows.append(
                {
                    "competition": competition,
                    "role": role,
                    "column": col,
                    "missing": int(group[col].isna().sum()),
                    "missing_pct": float(group[col].isna().mean() * 100),
                }
            )
    return pd.DataFrame(rows)


def _overlap(df: pd.DataFrame) -> pd.DataFrame:
    player_comp = (
        df.dropna(subset=["player_key"])
        .groupby(["role", "player_key"])["competition"]
        .agg(lambda x: ",".join(sorted(set(x.dropna()))))
        .reset_index()
    )
    rows = []
    for role, group in player_comp.groupby("role"):
        rows.append(
            {
                "role": role,
                "players_total": int(len(group)),
                "ipl_only": int((group["competition"] == "IPL").sum()),
                "sma_only": int((group["competition"] == "SMA").sum()),
                "played_both": int((group["competition"] == "IPL,SMA").sum()),
            }
        )
    return pd.DataFrame(rows)


def _top_players(scored: pd.DataFrame, competition: str, role: str, n: int = 20) -> pd.DataFrame:
    sample_col = "bat_innings" if role == "bat" else "bowl_innings"
    min_sample = 5
    metrics = ["strike_rate", "runs_per_innings", "boundary_pct"] if role == "bat" else [
        "economy",
        "bowling_sr",
        "wickets_per_innings",
    ]
    cols = [
        "player_name",
        "player_key",
        "competition",
        "season",
        "role",
        sample_col,
        "impact_score",
        "impact_percentile",
        "impact_reliability",
        *metrics,
    ]
    subset = scored[
        (scored["competition"] == competition)
        & (scored["role"] == role)
        & (scored[sample_col].fillna(0) >= min_sample)
        & scored["impact_score"].notna()
    ].copy()
    return subset[_available(subset, cols)].sort_values("impact_score", ascending=False).head(n)


def _plot_counts(overview: pd.DataFrame) -> None:
    pivot = overview.pivot(index="competition", columns="role", values="players").fillna(0)
    pivot.plot(kind="bar", figsize=(8, 5), color=["#2f6f73", "#c57b57"])
    plt.title("Unique Players by Competition and Role")
    plt.xlabel("Competition")
    plt.ylabel("Players")
    plt.xticks(rotation=0)
    _save_fig("players_by_competition_role.png")


def _plot_seasons(df: pd.DataFrame) -> None:
    season_counts = (
        df.groupby(["season", "competition"])["player_key"]
        .nunique()
        .reset_index(name="players")
        .pivot(index="season", columns="competition", values="players")
        .fillna(0)
    )
    season_counts.plot(kind="line", marker="o", figsize=(10, 5), color=["#2f6f73", "#c57b57"])
    plt.title("Player Coverage by Season")
    plt.xlabel("Season")
    plt.ylabel("Unique players")
    _save_fig("season_player_coverage.png")


def _plot_metric_distributions(df: pd.DataFrame, role: str, metrics: list[str], filename: str) -> None:
    role_df = df[df["role"] == role]
    metrics = _available(role_df, metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    colors = {"IPL": "#2f6f73", "SMA": "#c57b57"}
    for ax, metric in zip(axes, metrics):
        for competition, group in role_df.groupby("competition"):
            values = group[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                ax.hist(values, bins=30, alpha=0.55, label=competition, color=colors.get(competition))
        ax.set_title(metric)
        ax.set_ylabel("Player-seasons")
        ax.legend()
    _save_fig(filename)


def _plot_impact_distribution(scored: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"IPL": "#2f6f73", "SMA": "#c57b57"}
    for ax, role in zip(axes, ["bat", "bowl"]):
        role_df = scored[scored["role"] == role]
        for competition, group in role_df.groupby("competition"):
            values = group["impact_score"].dropna()
            if len(values):
                ax.hist(values, bins=30, alpha=0.55, label=competition, color=colors.get(competition))
        ax.set_title(f"{role.title()} impact score")
        ax.set_xlabel("Impact score")
        ax.set_ylabel("Player-seasons")
        ax.legend()
    _save_fig("impact_score_distributions.png")


def _plot_top_players(top: pd.DataFrame, filename: str, title: str) -> None:
    if top.empty:
        return
    plot_df = top.sort_values("impact_score", ascending=True)
    labels = plot_df["player_name"].fillna(plot_df["player_key"])
    plt.figure(figsize=(9, max(5, len(plot_df) * 0.35)))
    plt.barh(labels, plot_df["impact_score"], color="#2f6f73")
    plt.title(title)
    plt.xlabel("Impact score")
    _save_fig(filename)


def _plot_correlation(df: pd.DataFrame, role: str, metrics: list[str], filename: str) -> None:
    role_df = df[df["role"] == role]
    metrics = _available(role_df, metrics)
    corr = role_df[metrics].corr()
    _save_csv(corr.reset_index().rename(columns={"index": "metric"}), filename.replace(".png", ".csv"))

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticklabels(metrics)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            value = corr.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.title(f"{role.title()} Metric Correlations")
    _save_fig(filename)


def _training_summary() -> pd.DataFrame:
    if not TRAINING_PATH.exists():
        return pd.DataFrame()
    training = pd.read_parquet(TRAINING_PATH)
    rows = []
    for role, group in training.groupby("role"):
        rows.append(
            {
                "role": role,
                "training_rows": int(len(group)),
                "players": int(group["player_key"].nunique()),
                "mean_target_ipl_sample": float(group["target_ipl_sample"].mean()),
                "median_target_ipl_sample": float(group["target_ipl_sample"].median()),
                "mean_y_impact_reliability": float(group["y_impact_reliability"].mean()),
                "tier_counts": group["tier"].value_counts().to_dict(),
            }
        )
    return pd.DataFrame(rows)


def _write_markdown_report(
    overview: pd.DataFrame,
    overlap: pd.DataFrame,
    metric_summary: pd.DataFrame,
    training_summary: pd.DataFrame,
) -> Path:
    lines = [
        "# SMA/IPL EDA Report",
        "",
        "## What This EDA Checks",
        "",
        "- Data coverage by competition, role, and season.",
        "- Sample-size strength for batters and bowlers.",
        "- IPL vs SMA differences in core batting and bowling metrics.",
        "- Missingness in engineered features.",
        "- Observed impact-score distributions and top player-seasons.",
        "- Correlations between scouting metrics.",
        "",
        "## Coverage Summary",
        "",
        _markdown_table(overview),
        "",
        "## Player Overlap",
        "",
        _markdown_table(overlap),
        "",
    ]

    bat_key_metrics = ["strike_rate", "runs_per_innings", "boundary_pct"]
    bowl_key_metrics = ["economy", "bowling_sr", "wickets_per_innings"]
    selected = metric_summary[
        ((metric_summary["role"] == "bat") & metric_summary["metric"].isin(bat_key_metrics))
        | ((metric_summary["role"] == "bowl") & metric_summary["metric"].isin(bowl_key_metrics))
    ].copy()
    if not selected.empty:
        selected = selected[
            ["competition", "role", "metric", "non_null", "mean", "median", "std"]
        ].sort_values(["role", "metric", "competition"])
        lines.extend(["## Core Metric Summary", "", _markdown_table(selected), ""])

    if not training_summary.empty:
        lines.extend(
            [
                "## Supervised Training Context",
                "",
                _markdown_table(training_summary),
                "",
                "This section is useful for explaining why supervised prediction is hard: the model is trained only on players with prior SMA data and later IPL samples, which is a much smaller and noisier subset than the full scouting pool.",
                "",
            ]
        )

    lines.extend(
        [
            "## Generated Files",
            "",
            "- `dataset_overview.csv`",
            "- `metric_summary.csv`",
            "- `missingness.csv`",
            "- `player_overlap.csv`",
            "- `impact_scores_player_season.csv`",
            "- `top_*_impact.csv`",
            "- `*.png` charts for coverage, distributions, top players, and correlations.",
            "",
        ]
    )

    path = EDA_DIR / "EDA_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_eda() -> dict[str, Path]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature table: {FEATURES_PATH}. Run run_features.py first.")

    EDA_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(FEATURES_PATH)
    scored = _add_observed_impact(features)

    overview = _overview(features)
    metric_summary = _weighted_metric_summary(features, BAT_METRICS + BOWL_METRICS)
    missingness = _missingness(features)
    overlap = _overlap(features)
    training = _training_summary()

    outputs = {
        "overview": _save_csv(overview, "dataset_overview.csv"),
        "metric_summary": _save_csv(metric_summary, "metric_summary.csv"),
        "missingness": _save_csv(missingness, "missingness.csv"),
        "player_overlap": _save_csv(overlap, "player_overlap.csv"),
        "impact_scores": _save_csv(scored, "impact_scores_player_season.csv"),
    }
    if not training.empty:
        outputs["training_summary"] = _save_csv(training, "training_summary.csv")

    for competition in ["SMA", "IPL"]:
        for role in ["bat", "bowl"]:
            top = _top_players(scored, competition, role)
            name = f"top_{competition.lower()}_{role}ters_by_impact.csv" if role == "bat" else f"top_{competition.lower()}_bowlers_by_impact.csv"
            outputs[name] = _save_csv(top, name)
            chart_name = name.replace(".csv", ".png")
            outputs[chart_name] = EDA_DIR / chart_name
            _plot_top_players(
                top.head(15),
                chart_name,
                f"Top {competition} {'Batters' if role == 'bat' else 'Bowlers'} by Observed Impact",
            )

    _plot_counts(overview)
    _plot_seasons(features)
    _plot_metric_distributions(
        features,
        "bat",
        ["strike_rate", "runs_per_innings", "boundary_pct"],
        "batting_metric_distributions.png",
    )
    _plot_metric_distributions(
        features,
        "bowl",
        ["economy", "bowling_sr", "wickets_per_innings"],
        "bowling_metric_distributions.png",
    )
    _plot_impact_distribution(scored)
    _plot_correlation(
        features,
        "bat",
        ["strike_rate", "runs_per_innings", "boundary_pct", "bat_pp_sr", "bat_mid_sr", "bat_death_sr"],
        "batting_metric_correlations.png",
    )
    _plot_correlation(
        features,
        "bowl",
        ["economy", "bowling_sr", "wickets_per_innings", "bowl_pp_economy", "bowl_mid_economy", "bowl_death_economy"],
        "bowling_metric_correlations.png",
    )

    outputs["markdown_report"] = _write_markdown_report(overview, overlap, metric_summary, training)
    return outputs


def main() -> None:
    outputs = run_eda()
    print(f"Saved EDA outputs to {EDA_DIR}")
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
