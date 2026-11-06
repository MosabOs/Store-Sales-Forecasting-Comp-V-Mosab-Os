# Store Sales – Time Series Forecasting

A Machine Learning project developed using Python and XGBoost to forecast store sales as part of the **Store Sales – Time Series Forecasting** competition on Kaggle.

The project explores the complete workflow of a Machine Learning competition, from data exploration and feature engineering to model optimization and prediction generation.

## Project Overview

The objective is to predict daily sales for different product families across multiple stores using historical sales data and additional information such as promotions, store characteristics, holidays, and oil prices.

This project focuses on understanding time-dependent patterns, engineering meaningful features, and evaluating their impact on forecasting performance.

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

- **Lag Features:** `lag_16`, `lag_30`, and `lag_365`.
- **Rolling Statistics:** `rolling_mean_16`.
- **Calendar Features:** Year, month, day, day of week, week of year, and quarter.
- **Cyclical Encoding:** Sine and cosine transformations for calendar features.
- **Store Metadata:** Store location and characteristics.

Additional experiments were conducted with longer lag periods, rolling statistics, promotion transformations, oil prices, and transaction-based features.

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

## Results

| Model / Experiment | Validation RMSLE |
|---|---:|
| Initial Baseline | 0.55231 |
| Baseline + Store State | 0.51674 |
| Optimized XGBoost + Optuna | **0.46743** |

The optimized model improved the validation RMSLE compared with the initial baseline.

**Important:** The Optuna optimization and final model evaluation used the same validation period. Therefore, the reported result should not be interpreted as an independent estimate of generalization performance.

## Final Feature Set

The final model used 20 features:

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

The optimized model was used to generate predictions for the competition test dataset.

The final submission file was prepared in the required Kaggle format:

```text
submission.csv
```

Expected structure:

| Column | Description |
|---|---|
| `id` | Unique row identifier |
| `sales` | Predicted sales value |

The submission contains 28,512 predictions.

## Key Learnings

Throughout this project, I gained practical experience in:

- Working with large-scale tabular datasets.
- Understanding time-series forecasting problems.
- Engineering historical and calendar-based features.
- Evaluating feature contributions through controlled experiments.
- Analyzing prediction errors and model behavior.
- Optimizing XGBoost models using Optuna.
- Understanding the importance of time-based validation and data leakage prevention.
- Following an end-to-end Kaggle competition workflow.

## Future Improvements

Potential directions for further development include:

- Independent time-based validation.
- Spatial and temporal feature exploration.
- Experimenting with alternative forecasting models.
- Exploring ensemble techniques.
- Further investigating generalization across stores and product families.

## Project Status

**Core modeling and submission preparation completed.**

This project is part of my ongoing practical learning journey in Machine Learning and Data Science.

---

**Author:** Mosab Osama
**Platform:** Kaggle  
**Competition:** Store Sales – Time Series Forecasting
