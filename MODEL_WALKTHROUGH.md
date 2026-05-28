# IPL Scouting Model Walkthrough

This file explains what has been built so far, why each part exists, and how the pieces connect. It is written as a study guide for understanding the project, not as code documentation only.

## 1. Big Picture

The project is trying to answer this scouting question:

> Can a player's Syed Mushtaq Ali Trophy (SMA) T20 performance help predict their future IPL impact?

The current pipeline has two main stages:

1. **Feature engineering**
   - Parse ball-by-ball Cricsheet JSON.
   - Build player batting and bowling statistics.
   - Adjust domestic SMA stats toward IPL scale using CAF.
   - Stabilize noisy small-sample stats using shrinkage.

2. **Supervised learning**
   - Build training examples from players who played SMA before IPL.
   - Use their SMA features as inputs.
   - Use their observed IPL performance as labels.
   - Train:
     - an impact regression model
     - a tier classification model

No OQA, comparables, final scouting rankings, or report generation have been added yet.

## 2. Raw Data

The main raw data comes from Cricsheet:

- `data/raw/cricsheet_ipl/`
- `data/raw/cricsheet_sma/`

Each JSON file is one match. It contains:

- match information
- teams
- dates
- player registry IDs
- ball-by-ball deliveries
- runs, extras, wickets, batters, and bowlers

The project uses Cricsheet as the primary source because the same player registry format appears in both IPL and SMA.

## 3. Parsing Cricsheet

Main file:

```text
src/data/parse_cricsheet.py
```

This converts raw JSON into:

```text
data/interim/cricsheet_balls.parquet
data/interim/cricsheet_matches.parquet
```

The important output is `cricsheet_balls.parquet`, one row per delivery.

Important delivery columns:

- `runs_batter`: runs credited to the batter
- `runs_total`: total team runs on the delivery
- `extras_wides`
- `extras_noballs`
- `extras_byes`
- `extras_legbyes`
- `extras_penalty`
- `legal_ball`: whether the delivery counts as a legal ball in the over
- `batter_balls_faced`: whether the delivery counts as a ball faced by the batter
- `bowler_runs_conceded`: runs charged to the bowler
- `is_wicket`: whether any wicket happened
- `bowler_wicket`: whether the wicket is credited to the bowler

The reason `legal_ball` and `batter_balls_faced` are separate is that batting and bowling statistics count balls differently.

Example:

```text
No-ball hit for four:
runs_batter = 4
extras_noballs = 1
runs_total = 5
legal_ball = 0
batter_balls_faced = 1
bowler_runs_conceded = 5
```

Example:

```text
Wide:
runs_batter = 0
extras_wides = 1
runs_total = 1
legal_ball = 0
batter_balls_faced = 0
bowler_runs_conceded = 1
```

Byes and leg byes are not charged to the bowler:

```text
Leg bye:
runs_total = 1
extras_legbyes = 1
bowler_runs_conceded = 0
```

## 4. Building Player Innings And Seasons

Main file:

```text
src/data/build_innings.py
```

This takes ball-by-ball data and creates player-level summaries.

It writes:

```text
data/processed/player_innings_batting.parquet
data/processed/player_innings_bowling.parquet
data/processed/player_season_raw.parquet
```

### Batting Innings

For each batter in each innings:

- runs are summed from `runs_batter`
- balls are summed from `batter_balls_faced`
- fours and sixes are counted
- strike rate is calculated
- phase strike rates are calculated

Batting strike rate:

```text
strike_rate = runs / balls * 100
```

The phase splits are:

- powerplay: overs 0 to 5
- middle: overs 6 to 14
- death: overs 15 onwards

### Bowling Innings

For each bowler in each innings:

- runs conceded are summed from `bowler_runs_conceded`
- balls bowled are summed from `legal_ball`
- wickets are summed from `bowler_wicket`
- economy is calculated
- bowling strike rate is calculated
- phase economy rates are calculated

Bowling economy:

```text
economy = runs_conceded / balls_bowled * 6
```

Bowling strike rate:

```text
bowling_sr = balls_bowled / wickets
```

### Player Season Table

`player_season_raw.parquet` has one row per:

```text
player x competition x season x role
```

The two roles are:

- `bat`
- `bowl`

A player can have one batting row and one bowling row in the same season.

## 5. CAF: Competition Adjustment Factor

Main file:

```text
src/features/caf.py
```

CAF means:

```text
Competition Adjustment Factor
```

Its purpose is to translate SMA statistics onto an IPL-like scale.

For players who have played both SMA and IPL, CAF compares their IPL career stat to their SMA career stat:

```text
CAF = IPL stat / SMA stat
```

Example:

```text
If paired players have lower strike rates in IPL than SMA,
CAF for strike rate may be less than 1.

SMA strike rate = 150
CAF strike rate = 0.97
CAF-adjusted strike rate = 145.5
```

CAF is calculated separately for batting and bowling stats.

Batting CAF stats:

- `strike_rate`
- `runs_per_innings`
- `boundary_pct`
- `bat_pp_sr`
- `bat_mid_sr`
- `bat_death_sr`

Bowling CAF stats:

- `economy`
- `bowling_sr`
- `wickets_per_innings`
- `bowl_pp_economy`
- `bowl_mid_economy`
- `bowl_death_economy`

The output file is:

```text
data/processed/caf_factors.json
```

It stores:

- CAF multipliers
- sample sizes for each multiplier
- number of eligible paired players

CAF-adjusted columns are added to the feature table with a `caf_` prefix.

Example:

```text
caf_strike_rate
caf_bat_pp_sr
caf_economy
caf_bowl_death_economy
```

## 6. Shrinkage

Main file:

```text
src/features/shrinkage.py
```

Shrinkage is used because cricket stats from small samples can be misleading.

For example, a batter may have a very high strike rate after only two innings. Shrinkage pulls that value toward the IPL average.

Formula:

```text
shrunk_stat = (n * player_stat + k * league_mean) / (n + k)
```

Where:

- `n` is the player's sample size
- `k` is the shrinkage strength
- `league_mean` is the IPL mean for that stat

The current value of `k` is in:

```text
src/data/constants.py
```

The output columns use the `shrunk_` prefix:

```text
shrunk_strike_rate
shrunk_runs_per_innings
shrunk_bat_pp_sr
shrunk_economy
shrunk_bowl_death_economy
```

Final feature output:

```text
data/processed/player_features.parquet
```

## 7. Running Feature Engineering

Entry point:

```text
run_features.py
```

Run:

```powershell
python run_features.py
```

This performs:

1. Parse Cricsheet JSON
2. Build player innings and season stats
3. Estimate CAF factors
4. Apply CAF
5. Apply shrinkage
6. Save `player_features.parquet`

## 8. Supervised Learning: What It Means Here

Supervised learning means the model learns from examples where both inputs and outputs are known.

In this project:

```text
Input X = pre-IPL SMA features
Output y = observed IPL impact
```

The model does not learn from uncapped players yet because they do not have IPL outcomes.

Instead, it learns from players who:

1. played SMA first
2. later played IPL

These players become the training examples.

## 9. IPL Impact Score

Main file:

```text
src/features/impact_score.py
```

The impact score is the regression label.

Important point:

```text
The impact score is calculated from observed IPL stats only.
```

This avoids label leakage from domestic SMA features.

For batters, the impact score uses:

- strike rate
- runs per innings
- boundary percentage
- powerplay strike rate
- middle overs strike rate
- death overs strike rate

For bowlers, the impact score uses:

- economy
- bowling strike rate
- wickets per innings
- powerplay economy
- death economy

Stats are converted to z-scores before combining. This puts different metrics on a comparable scale.

For bowling economy and bowling strike rate, lower is better, so those weights are negative.

## 10. Tier Labels

Also in:

```text
src/features/impact_score.py
```

The tier classifier predicts one of:

```text
bust
marginal
solid
star
```

Tiers are created from observed IPL impact percentiles within each role.

The thresholds are saved in:

```text
data/processed/tier_thresholds.json
```

This is useful for assignment framing because it turns a regression problem into a supervised classification problem too.

## 11. Building The Training Table

Main file:

```text
src/models/build_training_table.py
```

Output:

```text
data/processed/ml_training_pairs.parquet
```

This file builds the supervised training examples.

For each player-role:

1. Find their first IPL season.
2. Collect SMA seasons before that IPL debut.
3. Aggregate those SMA seasons into model features.
4. Aggregate their IPL seasons into observed label stats.
5. Calculate `y_impact`.
6. Assign a tier label.

The final table has:

- player identifiers
- role
- SMA feature columns prefixed with `sma_`
- IPL observed label columns prefixed with `ipl_`
- `y_impact`
- `tier`

The latest run produced:

```text
161 training rows
120 players
90 batting rows
71 bowling rows
```

## 12. Model Training

Main file:

```text
src/models/train_supervised.py
```

Entry point:

```text
run_supervised.py
```

Run:

```powershell
python run_supervised.py
```

This does two things:

1. Builds `ml_training_pairs.parquet`
2. Trains the supervised models

### Model A: Impact Regression

This predicts:

```text
y_impact
```

Current model:

```text
HistGradientBoostingRegressor
```

This is a supervised regression model from scikit-learn.

Why this model?

- works with small-to-medium tabular data
- handles non-linear relationships
- does not require XGBoost installation
- is suitable as a strong first supervised model

Saved model:

```text
models/impact_regressor.joblib
```

### Model B: Tier Classification

This predicts:

```text
tier = bust / marginal / solid / star
```

Current model:

```text
HistGradientBoostingClassifier
```

Saved model:

```text
models/tier_classifier.joblib
```

## 13. Feature Preprocessing

In `train_supervised.py`, the model uses:

- numeric SMA features
- role as a categorical feature

The preprocessing pipeline:

1. fills missing numeric values with the median
2. fills missing categorical values with the most common value
3. one-hot encodes the role

This is important because machine learning models cannot directly train on missing values or text categories in a raw form.

The feature list is saved in:

```text
models/feature_columns.json
```

## 14. Validation Method

The project uses:

```text
GroupKFold by player_key
```

This means the same player cannot appear in both training and test folds.

That matters because otherwise the model might partly memorize a player and get an unrealistically good score.

## 15. Current Model Results

Metrics are saved in:

```text
reports/model_metrics.json
```

Latest run:

```text
Training rows: 161
Players: 120
Impact regression MAE: 0.516
Impact regression RMSE: 0.706
Impact regression Spearman: 0.323
Tier classifier accuracy: 0.366
Tier classifier macro-F1: 0.338
```

Interpretation:

- The model has some ranking signal, shown by positive Spearman correlation.
- The tier classifier is modest, which is expected with a small dataset and four classes.
- This is a valid first supervised baseline, not a final polished scouting system.

For an assignment, this is useful because you can discuss:

- small training sample size
- player transition uncertainty
- noisy cricket outcomes
- class imbalance
- need for future OQA and richer features

## 16. Important Files Created

Feature engineering:

```text
src/data/parse_cricsheet.py
src/data/build_innings.py
src/features/caf.py
src/features/shrinkage.py
src/features/pipeline.py
run_features.py
```

Supervised learning:

```text
src/features/impact_score.py
src/models/build_training_table.py
src/models/train_supervised.py
run_supervised.py
```

Generated data:

```text
data/interim/cricsheet_balls.parquet
data/interim/cricsheet_matches.parquet
data/processed/player_innings_batting.parquet
data/processed/player_innings_bowling.parquet
data/processed/player_season_raw.parquet
data/processed/player_features.parquet
data/processed/ml_training_pairs.parquet
data/processed/tier_thresholds.json
```

Generated model outputs:

```text
models/impact_regressor.joblib
models/tier_classifier.joblib
models/feature_columns.json
reports/model_metrics.json
```

## 17. What Has Not Been Done Yet

These are planned future steps:

- OQA opponent-quality adjustment
- prediction for uncapped players
- final scouting rankings
- cosine k-NN comparables
- uncertainty intervals
- scouting report export
- richer dissertation visualizations

## 18. How To Explain This In An Assignment

A concise explanation:

> I built a hybrid cricket analytics and supervised learning pipeline. First, I parsed IPL and SMA ball-by-ball data from Cricsheet and engineered role-specific batting and bowling features, including phase splits. I then calculated Competition Adjustment Factors from players who played both SMA and IPL, so domestic statistics could be translated onto an IPL-like scale. To reduce small-sample noise, I applied Bayesian shrinkage toward IPL league averages. For supervised learning, I built training rows from players with pre-IPL SMA data and post-debut IPL outcomes. The regression target was an observed IPL-only impact score, and the classification target was a percentile-based IPL tier. Models were evaluated with GroupKFold by player to avoid leakage.

The key methodological point:

```text
SMA data is used as input.
Observed IPL data is used as the label.
The same player is not split across train and test folds.
```

That is the core supervised learning argument.
