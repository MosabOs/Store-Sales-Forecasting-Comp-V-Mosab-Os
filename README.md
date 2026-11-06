# Store Sales – Time Series Forecasting

A Machine Learning project developed using Python and XGBoost to forecast store sales as part of the **Store Sales – Time Series Forecasting** competition on Kaggle.

The project explores the end-to-end workflow of a Machine Learning competition, from data exploration and feature engineering to model optimization and competition submission.

## Project Overview

The objective is to predict daily sales for different product families across multiple stores using historical sales data and additional information such as promotions, store characteristics, holidays, and oil prices.

The project focuses on understanding time-dependent patterns, engineering meaningful features, and evaluating their impact on forecasting performance.

## Competition & Dataset

- **Platform:** Kaggle
- **Competition:** [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Dataset:** Provided by the competition.
- **Prediction Target:** `sales`
- **Evaluation Metric:** Root Mean Squared Logarithmic Error (RMSLE)

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Optuna
- Kaggle Notebooks

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Examined dataset structure, dimensions, and data types.
- Investigated missing values and duplicate records.
- Analyzed sales distributions and statistical characteristics.
- Explored sales patterns across stores and product families.
- Investigated the effects of promotions, holidays, and transactions.
- Examined temporal trends and seasonality.

### 2. Baseline Model

Developed an initial forecasting model using XGBoost to establish a performance benchmark.

The baseline included:

- Calendar-based features.
- Store and product family information.
- Promotion data.
- Historical sales features.

### 3. Feature Engineering

Experimented with several feature engineering techniques:

- **Lag Features:** Historical sales values at different time intervals.
- **Rolling Statistics:** Rolling averages and standard deviations.
- **Calendar Features:** Year, month, day, day of week, week of year, and quarter.
- **Cyclical Encoding:** Sine and cosine transformations for calendar features.
- **Store Metadata:** Store location and characteristics.

Additional experiments were conducted with promotion transformations, oil prices, transaction-based features, and alternative historical windows.

Features were retained or excluded based on their measured impact on validation performance.

### 4. Model Analysis

- Analyzed feature importance.
- Evaluated prediction errors.
- Compared model performance across stores and product families.
- Investigated the impact of individual features and feature groups.

### 5. Hyperparameter Optimization

Used **Optuna** to optimize XGBoost hyperparameters.

The optimization process explored parameters such as:

- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`
- `reg_alpha`
- `reg_lambda`

Early stopping was also used during model training.

## Model Evaluation & Results

Model performance was evaluated using the Root Mean Squared Logarithmic Error (RMSLE).

The following results were obtained during the experiments conducted in this project:

| Experiment | Validation RMSLE |
|---|---:|
| Initial Baseline Model | 0.55231 |
| Baseline + Store State Feature | 0.51674 |
| Optimized XGBoost using Optuna | **0.46011** |

The optimized model achieved a lower validation error compared with the initial baseline through feature evaluation and hyperparameter optimization.

### Kaggle Public Leaderboard

The generated predictions were submitted to the Kaggle competition, resulting in the following Public Score:

| Evaluation | RMSLE |
|---|---:|
| Kaggle Public Score | 0.51750 |

The Public Score differs from the local validation result. This is expected because the two scores are calculated using different data subsets.

This difference may reflect changes in data distribution, temporal patterns, and the model's ability to generalize to unseen data.

**Note:** These values represent the results obtained during the current experiments and are not fixed or guaranteed results. They may change with future modifications to the dataset, feature engineering process, validation strategy, model configuration, hyperparameters, or training process. Re-running the experiments may also produce different results.

## Final Feature Set

The optimized model used the following 20 features:

```python
final_features = [
    'store_nbr',
    'family',
    'onpromotion',
    'year',
    'month',
    'day',
    'day_of_week',
    'day_of_year',
    'week_of_year',
    'is_weekend',
    'quarter',
    'month_sin',
    'month_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'lag_16',
    'lag_30',
    'lag_365',
    'rolling_mean_16',
    'state'
]
```

## Competition Submission

The optimized XGBoost model was used to generate predictions for the competition test dataset.

The predictions were exported into the required Kaggle submission format:

`submission.csv`

- **Submission Status:** Successfully submitted.
- **Number of Predictions:** 28,512.
- **Evaluation Metric:** RMSLE.
- **Public Score:** 0.51750 (at the time of submission).

The Public Leaderboard score provides an additional evaluation of the model on unseen competition data and should be considered alongside the local validation results.

## Key Learnings

Throughout this project, I gained practical experience in:

- Working with large-scale tabular datasets.
- Understanding time-series forecasting problems.
- Engineering historical and calendar-based features.
- Evaluating feature contributions through controlled experiments.
- Analyzing prediction errors and model behavior.
- Optimizing XGBoost models using Optuna.
- Understanding the importance of time-based validation and data leakage prevention.
- Evaluating the difference between local validation performance and public leaderboard performance.
- Following an end-to-end Kaggle competition workflow.

## Future Improvements

Potential directions for further development include:

- Independent time-based validation.
- Further investigation of temporal patterns.
- Experimenting with alternative forecasting models.
- Exploring ensemble techniques.
- Improving generalization across stores and product families.

## Project Status

**Core modeling, optimization, and competition submission completed.**

Further experiments and improvements may be introduced as part of the ongoing learning process.

---

**Author:** Mosab Osama
**Platform:** Kaggle  
**Competition:** Store Sales – Time Series Forecasting
