# SMA/IPL EDA Report

## What This EDA Checks

- Data coverage by competition, role, and season.
- Sample-size strength for batters and bowlers.
- IPL vs SMA differences in core batting and bowling metrics.
- Missingness in engineered features.
- Observed impact-score distributions and top player-seasons.
- Correlations between scouting metrics.

## Coverage Summary

| competition | role | player_season_rows | players | seasons | first_season | last_season | total_bat_innings | total_bowl_innings |
| ----------- | ---- | ------------------ | ------- | ------- | ------------ | ----------- | ----------------- | ------------------ |
| IPL         | bat  | 2948               | 731     | 19      | 2008         | 2026        | 18603.000         | 0.000              |
| IPL         | bowl | 2199               | 576     | 19      | 2008         | 2026        | 0.000             | 14580.000          |
| SMA         | bat  | 2797               | 1304    | 7       | 2016         | 2024        | 10527.000         | 0.000              |
| SMA         | bowl | 2037               | 974     | 7       | 2016         | 2024        | 0.000             | 8278.000           |

## Player Overlap

| role | players_total | ipl_only | sma_only | played_both |
| ---- | ------------- | -------- | -------- | ----------- |
| bat  | 1761          | 457      | 1030     | 274         |
| bowl | 1345          | 371      | 769      | 205         |

## Core Metric Summary

| competition | role | metric              | non_null | mean    | median  | std    |
| ----------- | ---- | ------------------- | -------- | ------- | ------- | ------ |
| IPL         | bat  | boundary_pct        | 2948     | 13.377  | 14.208  | 10.210 |
| SMA         | bat  | boundary_pct        | 2797     | 11.624  | 11.765  | 9.898  |
| IPL         | bat  | runs_per_innings    | 2948     | 14.213  | 12.074  | 11.470 |
| SMA         | bat  | runs_per_innings    | 2797     | 13.850  | 10.667  | 12.194 |
| IPL         | bat  | strike_rate         | 2948     | 111.482 | 117.779 | 49.871 |
| SMA         | bat  | strike_rate         | 2797     | 101.886 | 105.000 | 50.401 |
| IPL         | bowl | bowling_sr          | 1809     | 24.892  | 21.333  | 14.045 |
| SMA         | bowl | bowling_sr          | 1582     | 23.840  | 20.000  | 15.706 |
| IPL         | bowl | economy             | 2199     | 9.094   | 8.597   | 2.622  |
| SMA         | bowl | economy             | 2036     | 8.002   | 7.625   | 2.543  |
| IPL         | bowl | wickets_per_innings | 2199     | 0.766   | 0.786   | 0.533  |
| SMA         | bowl | wickets_per_innings | 2037     | 0.808   | 0.750   | 0.679  |

## Supervised Training Context

| role | training_rows | players | mean_target_ipl_sample | median_target_ipl_sample | mean_y_impact_reliability | tier_counts                                              |
| ---- | ------------- | ------- | ---------------------- | ------------------------ | ------------------------- | -------------------------------------------------------- |
| bat  | 421           | 113     | 10.378                 | 11.000                   | 0.493                     | {'marginal': 147, 'bust': 106, 'solid': 105, 'star': 63} |
| bowl | 336           | 99      | 10.369                 | 10.000                   | 0.493                     | {'marginal': 118, 'bust': 84, 'solid': 83, 'star': 51}   |

This section is useful for explaining why supervised prediction is hard: the model is trained only on players with prior SMA data and later IPL samples, which is a much smaller and noisier subset than the full scouting pool.

## Generated Files

- `dataset_overview.csv`
- `metric_summary.csv`
- `missingness.csv`
- `player_overlap.csv`
- `impact_scores_player_season.csv`
- `top_*_impact.csv`
- `*.png` charts for coverage, distributions, top players, and correlations.
