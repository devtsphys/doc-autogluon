# AutoGluon Time Series Forecasting - Complete Reference Card

**Version**: AutoGluon 1.1+ | **Focus**: Single Time Series Forecasting  
**Last Updated**: February 2026

-----

## Table of Contents

1. [Quick Start](#quick-start)
1. [Installation](#installation)
1. [Core Classes](#core-classes)
1. [TimeSeriesDataFrame](#timeseriesdataframe)
1. [TimeSeriesPredictor](#timeseriespredictor)
1. [Configuration & Hyperparameters](#configuration--hyperparameters)
1. [Available Models](#available-models)
1. [Methods Reference Table](#methods-reference-table)
1. [Feature Engineering](#feature-engineering)
1. [Evaluation Metrics](#evaluation-metrics)
1. [Advanced Techniques](#advanced-techniques)
1. [Code Examples](#code-examples)
1. [Best Practices](#best-practices)
1. [Troubleshooting](#troubleshooting)

-----

## Quick Start

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# 1. Prepare data
train_data = TimeSeriesDataFrame.from_path("train.csv")

# 2. Train predictor
predictor = TimeSeriesPredictor(
    prediction_length=24,
    target="value",
    eval_metric="MASE"
)
predictor.fit(train_data)

# 3. Make predictions
predictions = predictor.predict(train_data)

# 4. Evaluate
leaderboard = predictor.leaderboard(train_data)
```

-----

## Installation

```bash
# Basic installation
pip install autogluon.timeseries

# Full installation with deep learning models
pip install autogluon.timeseries[all]

# Specific backends
pip install autogluon.timeseries[torch]      # PyTorch models
pip install autogluon.timeseries[mxnet]      # MXNet models
pip install autogluon.timeseries[lightgbm]   # LightGBM

# From source (latest)
pip install git+https://github.com/autogluon/autogluon.git#subdirectory=timeseries
```

-----

## Core Classes

### 1. `TimeSeriesDataFrame`

Container for time series data with metadata handling.

### 2. `TimeSeriesPredictor`

Main class for training and prediction.

-----

## TimeSeriesDataFrame

### Constructor

```python
TimeSeriesDataFrame(
    data,                    # pd.DataFrame with time series data
    id_column=None,         # Column name for item IDs
    timestamp_column=None   # Column name for timestamps
)
```

### Creation Methods

#### From DataFrame

```python
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

df = pd.DataFrame({
    'timestamp': pd.date_range('2020-01-01', periods=100, freq='D'),
    'value': range(100)
})

ts_df = TimeSeriesDataFrame(df)
```

#### From CSV/Parquet

```python
# From CSV
ts_df = TimeSeriesDataFrame.from_path("data.csv")

# From Parquet
ts_df = TimeSeriesDataFrame.from_path("data.parquet")

# With custom columns
ts_df = TimeSeriesDataFrame.from_path(
    "data.csv",
    id_column="item_id",
    timestamp_column="date"
)
```

#### From Iterable

```python
# Multiple time series
data_list = [
    pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=50), 
                  'value': range(50)}),
    pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=60), 
                  'value': range(60)})
]

ts_df = TimeSeriesDataFrame.from_iterable(
    data_list,
    id_column="item_id",
    timestamp_column="timestamp"
)
```

#### From Data Frame with MultiIndex

```python
# Create multi-indexed DataFrame
df = pd.DataFrame({
    'value': range(200),
    'covariate': range(200)
})
df.index = pd.MultiIndex.from_product(
    [['A', 'B'], pd.date_range('2020-01-01', periods=100)],
    names=['item_id', 'timestamp']
)

ts_df = TimeSeriesDataFrame(df)
```

### Key Attributes

```python
ts_df.item_ids          # List of unique item IDs
ts_df.num_items         # Number of time series
ts_df.freq              # Frequency of time series
ts_df.columns           # Column names
ts_df.index             # MultiIndex (item_id, timestamp)
```

### Indexing & Slicing

```python
# Get single time series
single_ts = ts_df.loc["item_1"]

# Get multiple time series
multi_ts = ts_df.loc[["item_1", "item_2"]]

# Time range slicing
ts_slice = ts_df.slice_by_timestep(0, 50)  # First 50 timesteps

# Get last n values
recent = ts_df.slice_by_timestep(-24, None)  # Last 24 timesteps
```

### Data Manipulation

```python
# Fill missing values
ts_df_filled = ts_df.fill_missing_values(method="ffill")

# Convert frequency
ts_df_hourly = ts_df.convert_frequency("H")

# Split train/test
train, test = ts_df.train_test_split(test_size=24)

# Add static features
ts_df["category"] = ["A", "B", "A"]  # One value per item

# Add time-varying features
ts_df["day_of_week"] = ts_df.index.get_level_values('timestamp').dayofweek
```

-----

## TimeSeriesPredictor

### Initialization

```python
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    prediction_length=24,           # Required: forecast horizon
    target="value",                 # Target column name
    eval_metric="MASE",            # Evaluation metric
    path="./autogluon_models",     # Save directory
    verbosity=2,                   # 0-4, higher = more output
    freq=None,                     # Auto-detect from data
    quantile_levels=[0.1, 0.5, 0.9],  # Quantile forecasts
    known_covariates_names=None,   # Future-known features
    cache_predictions=True,        # Cache intermediate results
    random_state=42               # For reproducibility
)
```

### Parameters Explained

|Parameter               |Type|Description                          |Default             |
|------------------------|----|-------------------------------------|--------------------|
|`prediction_length`     |int |Number of steps to forecast          |Required            |
|`target`                |str |Name of target column                |“target”            |
|`eval_metric`           |str |Metric for model selection           |“WQL”               |
|`path`                  |str |Directory to save models             |“./AutogluonModels/”|
|`verbosity`             |int |Logging level (0-4)                  |2                   |
|`freq`                  |str |Pandas frequency string              |Auto-detected       |
|`quantile_levels`       |list|Quantiles for probabilistic forecasts|[0.1, 0.2, …, 0.9]  |
|`known_covariates_names`|list|Columns with future values           |None                |
|`random_state`          |int |Random seed                          |None                |

### Training

#### Basic Training

```python
predictor.fit(
    train_data,                    # TimeSeriesDataFrame
    time_limit=3600,              # Seconds (None = no limit)
    presets="medium_quality",     # Preset configuration
    hyperparameters=None,         # Model-specific hyperparameters
    num_val_windows=None,         # Validation windows
    val_step_size=None,           # Validation step size
    enable_ensemble=True,         # Use ensemble models
    skip_model_selection=False,   # Skip model selection
    excluded_model_types=None,    # Models to exclude
    random_state=None            # Random seed
)
```

#### Presets

```python
# Quality presets (speed vs accuracy trade-off)
presets = [
    "fast_training",              # Fastest, least accurate
    "medium_quality",             # Balanced (default)
    "high_quality",              # Slower, more accurate
    "best_quality"               # Slowest, most accurate
]

# Example
predictor.fit(train_data, presets="best_quality")
```

### Prediction

#### Basic Prediction

```python
# Point forecasts
predictions = predictor.predict(data)

# Returns TimeSeriesDataFrame with forecasts
# Index: (item_id, timestamp)
# Columns: mean, 0.1, 0.5, 0.9 (quantiles)
```

#### Advanced Prediction Options

```python
predictions = predictor.predict(
    data,
    model=None,                   # Specific model name
    use_cache=True,              # Use cached predictions
    random_seed=None,            # For stochastic models
    known_covariates=None        # Future covariate values
)
```

#### Quantile Predictions

```python
# Specify custom quantiles
predictor_custom = TimeSeriesPredictor(
    prediction_length=24,
    quantile_levels=[0.05, 0.25, 0.5, 0.75, 0.95]
)

predictions = predictor_custom.predict(data)

# Access quantiles
mean_forecast = predictions["mean"]
lower_bound = predictions["0.05"]
median = predictions["0.5"]
upper_bound = predictions["0.95"]
```

### Evaluation

#### Leaderboard

```python
leaderboard = predictor.leaderboard(
    data=test_data,
    display=True,               # Print to console
    use_cache=True             # Use cached predictions
)

# Returns DataFrame with:
# - model: Model name
# - score_test: Test score
# - score_val: Validation score
# - pred_time_test: Prediction time
# - fit_time: Training time
# - ... other metrics
```

#### Model Performance

```python
# Detailed scores for all models
scores = predictor.score(test_data, metric="MASE")

# Single model score
model_score = predictor.score(test_data, model="SeasonalNaive")
```

#### Get Best Model

```python
best_model = predictor.get_model_best()
print(f"Best model: {best_model}")
```

### Model Information

```python
# List all trained models
model_names = predictor.model_names()

# Model summary
info = predictor.info()

# Fitted hyperparameters
params = predictor.get_model_params("DeepAR")

# Feature importance (for tree-based models)
importance = predictor.feature_importance(test_data)
```

### Saving & Loading

```python
# Save predictor
predictor.save()

# Load predictor
from autogluon.timeseries import TimeSeriesPredictor
loaded_predictor = TimeSeriesPredictor.load("./autogluon_models")

# Save predictions
predictions.to_csv("forecasts.csv")
```

-----

## Configuration & Hyperparameters

### Model-Specific Hyperparameters

```python
hyperparameters = {
    "DeepAR": {
        "epochs": 100,
        "num_batches_per_epoch": 50,
        "context_length": 48,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout_rate": 0.1
    },
    "SimpleFeedForward": {
        "epochs": 100,
        "hidden_size": 128,
        "num_layers": 3
    },
    "ETS": {
        "seasonal": "add",
        "seasonal_periods": None
    },
    "AutoARIMA": {
        "maxiter": 100,
        "seasonal": True,
        "stepwise": True
    },
    "Theta": {
        "decomposition_type": "multiplicative"
    },
    "RecursiveTabular": {
        "model": "GBM",  # or "CAT", "RF", "XT"
        "max_depth": 10,
        "num_boost_round": 500
    }
}

predictor.fit(train_data, hyperparameters=hyperparameters)
```

### Global Configuration

```python
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    prediction_length=24,
    eval_metric="MASE",
    
    # Ensemble settings
    enable_ensemble=True,
    num_ensemble_models=10,
    
    # Validation settings
    num_val_windows=3,           # Cross-validation windows
    val_step_size=12,            # Steps between windows
    
    # Cache settings
    cache_predictions=True,
    refit_full=True,            # Refit on full data after validation
    
    # Advanced settings
    skip_model_selection=False,
    enable_meta_learning=True
)
```

-----

## Available Models

### Statistical Models

#### 1. **SeasonalNaive**

Simple baseline using seasonal patterns.

```python
hyperparameters = {
    "SeasonalNaive": {
        "season_length": 24  # Seasonal period
    }
}
```

**Strengths**: Fast, interpretable, good baseline  
**Use case**: Quick baseline, seasonal data

#### 2. **Naive**

Uses last observed value.

```python
hyperparameters = {
    "Naive": {}
}
```

**Strengths**: Simplest baseline  
**Use case**: Random walk processes

#### 3. **ETS (Error, Trend, Seasonal)**

Exponential smoothing state space model.

```python
hyperparameters = {
    "ETS": {
        "seasonal": "add",        # "add", "mul", or None
        "seasonal_periods": 24,   # Seasonal period
        "error": "add",           # Error type
        "trend": "add",           # Trend type
        "damped_trend": False     # Damped trend
    }
}
```

**Strengths**: Classical, interpretable, fast  
**Use case**: Smooth trends, clear seasonality

#### 4. **AutoARIMA**

Automatic ARIMA model selection.

```python
hyperparameters = {
    "AutoARIMA": {
        "seasonal": True,
        "stepwise": True,         # Stepwise search
        "maxiter": 100,
        "seasonal_periods": 24,
        "max_p": 5,              # Max AR order
        "max_q": 5,              # Max MA order
        "max_P": 2,              # Max seasonal AR
        "max_Q": 2,              # Max seasonal MA
        "max_d": 2,              # Max differencing
        "max_D": 1               # Max seasonal differencing
    }
}
```

**Strengths**: Classical, automatic order selection  
**Use case**: Linear patterns, trend + seasonality

#### 5. **Theta**

Theta method for forecasting.

```python
hyperparameters = {
    "Theta": {
        "decomposition_type": "multiplicative",  # or "additive"
        "theta": 2.0            # Theta parameter
    }
}
```

**Strengths**: Simple, effective for monthly data  
**Use case**: Business forecasting, M-competitions

#### 6. **AutoCES (Complex Exponential Smoothing)**

```python
hyperparameters = {
    "AutoCES": {
        "seasonal": True,
        "seasonal_periods": 24
    }
}
```

**Strengths**: Handles complex seasonality  
**Use case**: Multiple seasonal patterns

-----

### Deep Learning Models

#### 7. **DeepAR**

Autoregressive RNN for probabilistic forecasting.

```python
hyperparameters = {
    "DeepAR": {
        "epochs": 100,
        "num_batches_per_epoch": 50,
        "context_length": 48,     # Historical lookback
        "hidden_size": 64,        # LSTM hidden units
        "num_layers": 2,          # LSTM layers
        "dropout_rate": 0.1,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "cell_type": "lstm"       # "lstm" or "gru"
    }
}
```

**Strengths**: Probabilistic, handles multiple series  
**Use case**: Large datasets, complex patterns

#### 8. **SimpleFeedForward**

Simple feedforward neural network.

```python
hyperparameters = {
    "SimpleFeedForward": {
        "epochs": 100,
        "context_length": 48,
        "hidden_size": 128,
        "num_layers": 3,
        "dropout_rate": 0.1,
        "learning_rate": 1e-3
    }
}
```

**Strengths**: Simple, fast to train  
**Use case**: Non-linear patterns, moderate data

#### 9. **TemporalFusionTransformer (TFT)**

Advanced transformer for time series.

```python
hyperparameters = {
    "TemporalFusionTransformer": {
        "epochs": 100,
        "context_length": 48,
        "hidden_size": 64,
        "num_heads": 4,           # Attention heads
        "dropout_rate": 0.1,
        "learning_rate": 1e-3,
        "batch_size": 32
    }
}
```

**Strengths**: State-of-art, interpretable attention  
**Use case**: Complex patterns, covariates

#### 10. **PatchTST**

Transformer using patch-based inputs.

```python
hyperparameters = {
    "PatchTST": {
        "epochs": 100,
        "context_length": 96,
        "patch_len": 16,          # Patch length
        "stride": 8,              # Patch stride
        "num_layers": 3,
        "d_model": 128,
        "num_heads": 8,
        "dropout_rate": 0.2
    }
}
```

**Strengths**: Recent SOTA, efficient  
**Use case**: Long sequences, recent research

-----

### Tree-Based Models

#### 11. **RecursiveTabular**

Recursive forecasting with tree models.

```python
hyperparameters = {
    "RecursiveTabular": {
        "model": "GBM",          # "GBM", "CAT", "RF", "XT"
        "max_depth": 10,
        "num_boost_round": 500,
        "learning_rate": 0.05,
        "context_length": 48,     # Lag features
        "feature_generator_kwargs": {
            "datetime_features": ["year", "month", "day", "dayofweek"],
            "lag_features": [1, 2, 3, 7, 14, 21],
            "rolling_features": {
                "mean": [7, 14, 28],
                "std": [7, 14, 28]
            }
        }
    }
}
```

**Strengths**: Feature-rich, handles covariates well  
**Use case**: Many covariates, feature engineering

#### 12. **DirectTabular**

Direct multi-step forecasting.

```python
hyperparameters = {
    "DirectTabular": {
        "model": "GBM",
        "max_depth": 8,
        "num_boost_round": 300,
        "context_length": 48
    }
}
```

**Strengths**: Separate model per horizon  
**Use case**: Different patterns at different horizons

-----

### Ensemble Models

#### 13. **WeightedEnsemble**

Automatically created weighted ensemble.

```python
# Automatically enabled with enable_ensemble=True
predictor.fit(
    train_data,
    enable_ensemble=True,
    num_ensemble_models=10  # Top-k models to ensemble
)
```

**Strengths**: Combines best models  
**Use case**: Maximum accuracy

-----

## Methods Reference Table

### TimeSeriesDataFrame Methods

|Method                 |Description                   |Example                                     |
|-----------------------|------------------------------|--------------------------------------------|
|`from_path()`          |Load from file                |`TimeSeriesDataFrame.from_path("data.csv")` |
|`from_iterable()`      |Create from list of DataFrames|`TimeSeriesDataFrame.from_iterable(df_list)`|
|`fill_missing_values()`|Fill NaN values               |`ts_df.fill_missing_values(method="ffill")` |
|`convert_frequency()`  |Change frequency              |`ts_df.convert_frequency("H")`              |
|`train_test_split()`   |Split data                    |`train, test = ts_df.train_test_split(0.2)` |
|`slice_by_timestep()`  |Slice by position             |`ts_df.slice_by_timestep(0, 100)`           |
|`to_regular_index()`   |Convert to regular DataFrame  |`df = ts_df.to_regular_index()`             |
|`num_items`            |Number of time series         |`n = ts_df.num_items`                       |
|`item_ids`             |List of item IDs              |`ids = ts_df.item_ids`                      |

### TimeSeriesPredictor Methods

|Method                |Description           |Returns            |Example                                        |
|----------------------|----------------------|-------------------|-----------------------------------------------|
|`fit()`               |Train models          |self               |`predictor.fit(train_data)`                    |
|`predict()`           |Generate forecasts    |TimeSeriesDataFrame|`pred = predictor.predict(data)`               |
|`leaderboard()`       |Model rankings        |pd.DataFrame       |`lb = predictor.leaderboard(test_data)`        |
|`score()`             |Evaluate metric       |float/dict         |`score = predictor.score(test_data)`           |
|`save()`              |Save predictor        |None               |`predictor.save()`                             |
|`load()`              |Load predictor        |TimeSeriesPredictor|`TimeSeriesPredictor.load(path)`               |
|`model_names()`       |List models           |list               |`models = predictor.model_names()`             |
|`get_model_best()`    |Best model name       |str                |`best = predictor.get_model_best()`            |
|`info()`              |Predictor summary     |dict               |`info = predictor.info()`                      |
|`feature_importance()`|Feature rankings      |pd.DataFrame       |`fi = predictor.feature_importance(data)`      |
|`get_model_params()`  |Model hyperparameters |dict               |`params = predictor.get_model_params("DeepAR")`|
|`refit_full()`        |Retrain on full data  |self               |`predictor.refit_full(full_data)`              |
|`plot()`              |Visualize forecasts   |matplotlib figure  |`predictor.plot(test_data)`                    |
|`persist()`           |Cache models in memory|None               |`predictor.persist()`                          |
|`unpersist()`         |Remove from memory    |None               |`predictor.unpersist()`                        |

-----

## Feature Engineering

### Datetime Features

```python
# AutoGluon automatically creates:
# - year, month, day
# - hour, minute (if applicable)
# - dayofweek, dayofyear
# - week, quarter

# Custom datetime features
ts_df["is_weekend"] = ts_df.index.get_level_values('timestamp').dayofweek >= 5
ts_df["is_month_end"] = ts_df.index.get_level_values('timestamp').is_month_end
ts_df["hour"] = ts_df.index.get_level_values('timestamp').hour
```

### Lag Features

```python
# Configure in RecursiveTabular
hyperparameters = {
    "RecursiveTabular": {
        "feature_generator_kwargs": {
            "lag_features": [1, 2, 3, 7, 14, 21, 28]  # Lag values
        }
    }
}
```

### Rolling Window Features

```python
hyperparameters = {
    "RecursiveTabular": {
        "feature_generator_kwargs": {
            "rolling_features": {
                "mean": [7, 14, 28],      # Rolling mean windows
                "std": [7, 14, 28],       # Rolling std
                "min": [7, 14],           # Rolling min
                "max": [7, 14],           # Rolling max
                "sum": [7, 14]            # Rolling sum
            }
        }
    }
}
```

### Static Covariates

```python
# Add features that don't change over time
ts_df["store_type"] = ["A", "B", "A"]  # One value per item
ts_df["region"] = ["North", "South", "North"]
```

### Known Future Covariates

```python
# Features known in advance (holidays, promotions, etc.)
predictor = TimeSeriesPredictor(
    prediction_length=24,
    known_covariates_names=["is_holiday", "promotion", "price"]
)

# Must provide future values during prediction
future_covariates = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
    'is_holiday': [0, 0, 1, 1, 0, ...],
    'promotion': [0.1, 0.1, 0.2, ...],
    'price': [10, 10, 9.5, ...]
})

predictions = predictor.predict(
    data,
    known_covariates=TimeSeriesDataFrame(future_covariates)
)
```

-----

## Evaluation Metrics

### Available Metrics

|Metric |Description                       |Best Value|Use Case                 |
|-------|----------------------------------|----------|-------------------------|
|`MASE` |Mean Absolute Scaled Error        |0         |General, robust to scale |
|`MAPE` |Mean Absolute Percentage Error    |0         |Interpretable (%)        |
|`SMAPE`|Symmetric MAPE                    |0         |Symmetric version of MAPE|
|`MSE`  |Mean Squared Error                |0         |Penalizes large errors   |
|`RMSE` |Root Mean Squared Error           |0         |Same scale as target     |
|`MAE`  |Mean Absolute Error               |0         |Robust to outliers       |
|`WAPE` |Weighted Absolute Percentage Error|0         |Business metrics         |
|`WQL`  |Weighted Quantile Loss            |0         |Probabilistic forecasts  |
|`QL`   |Quantile Loss                     |0         |Single quantile          |
|`RMSSE`|Root Mean Squared Scaled Error    |0         |M5 competition           |

### Setting Evaluation Metric

```python
# During initialization
predictor = TimeSeriesPredictor(
    prediction_length=24,
    eval_metric="MASE"  # Primary metric for model selection
)

# Evaluate with different metric
score_mape = predictor.score(test_data, metric="MAPE")
score_rmse = predictor.score(test_data, metric="RMSE")
```

### Custom Metrics

```python
from autogluon.timeseries.metrics import TimeSeriesScorer

def custom_metric(y_true, y_pred):
    """Custom metric function"""
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1))

# Create scorer
custom_scorer = TimeSeriesScorer(
    name="custom_metric",
    score_func=custom_metric,
    greater_is_better=False
)

# Use in evaluation
score = predictor.score(test_data, metric=custom_scorer)
```

-----

## Advanced Techniques

### 1. Cross-Validation

```python
# Time series cross-validation with multiple windows
predictor.fit(
    train_data,
    num_val_windows=5,      # Number of validation windows
    val_step_size=12        # Steps between windows
)

# Sliding window validation ensures temporal ordering
```

### 2. Handling Multiple Time Series

```python
# Stack multiple time series
df1 = pd.DataFrame({
    'item_id': 'A',
    'timestamp': pd.date_range('2020-01-01', periods=100),
    'value': range(100)
})

df2 = pd.DataFrame({
    'item_id': 'B',
    'timestamp': pd.date_range('2020-01-01', periods=120),
    'value': range(120)
})

ts_df = TimeSeriesDataFrame(pd.concat([df1, df2]))

# AutoGluon handles different lengths automatically
predictor.fit(ts_df)
predictions = predictor.predict(ts_df)
```

### 3. Probabilistic Forecasting

```python
# Configure quantiles
predictor = TimeSeriesPredictor(
    prediction_length=24,
    quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9]
)

predictor.fit(train_data)
predictions = predictor.predict(test_data)

# Extract prediction intervals
lower_90 = predictions["0.1"]
upper_90 = predictions["0.9"]
median = predictions["0.5"]
```

### 4. Model Selection & Ensembling

```python
# Exclude slow models
predictor.fit(
    train_data,
    excluded_model_types=["TemporalFusionTransformer", "PatchTST"]
)

# Custom model selection
hyperparameters = {
    "DeepAR": {},
    "AutoARIMA": {},
    "RecursiveTabular": {"model": "GBM"}
}

predictor.fit(
    train_data,
    hyperparameters=hyperparameters,
    enable_ensemble=True
)
```

### 5. Transfer Learning

```python
# Train on multiple series, predict on new ones
predictor.fit(multi_series_train)

# Predict on new series with same structure
new_predictions = predictor.predict(new_series)
```

### 6. Handling Irregular Time Series

```python
# Resample to regular frequency
ts_df_regular = ts_df.convert_frequency("D")

# Fill gaps
ts_df_filled = ts_df.fill_missing_values(method="linear")
```

### 7. Hyperparameter Tuning

```python
# Define search space
hyperparameters = {
    "DeepAR": {
        "epochs": 100,
        "hidden_size": ag.space.Categorical(32, 64, 128),
        "num_layers": ag.space.Int(1, 3),
        "dropout_rate": ag.space.Real(0.0, 0.3)
    }
}

# Run hyperparameter search
predictor.fit(
    train_data,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs={
        "num_trials": 20,
        "scheduler": "local",
        "searcher": "bayes"
    }
)
```

### 8. Feature Importance Analysis

```python
# Get feature importance (for tree models)
importance = predictor.feature_importance(test_data)

# Plot
import matplotlib.pyplot as plt
importance.plot(kind='barh', figsize=(10, 6))
plt.title("Feature Importance")
plt.show()
```

### 9. Handling Covariates

```python
# Time-varying covariates (past values only)
ts_df["temperature"] = temperature_data
ts_df["holiday"] = holiday_indicator

# Known future covariates
predictor = TimeSeriesPredictor(
    prediction_length=24,
    known_covariates_names=["holiday", "day_of_week"]
)

# Provide future covariate values
future_covariates = create_future_covariates(24)
predictions = predictor.predict(test_data, known_covariates=future_covariates)
```

### 10. Retraining on Full Data

```python
# Initial training with validation
predictor.fit(train_data)

# After model selection, retrain on all data
predictor.refit_full(full_data)
```

-----

## Code Examples

### Example 1: Basic Single Time Series Forecast

```python
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Create sample data
dates = pd.date_range('2020-01-01', periods=365, freq='D')
values = np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.randn(365) * 0.1

df = pd.DataFrame({
    'timestamp': dates,
    'value': values
})

# Convert to TimeSeriesDataFrame
ts_df = TimeSeriesDataFrame(df)

# Split train/test
train = ts_df.slice_by_timestep(0, 300)
test = ts_df.slice_by_timestep(300, None)

# Train predictor
predictor = TimeSeriesPredictor(
    prediction_length=30,
    eval_metric="MASE"
)

predictor.fit(
    train,
    presets="medium_quality",
    time_limit=600
)

# Predict
predictions = predictor.predict(train)

# Evaluate
leaderboard = predictor.leaderboard(test)
print(leaderboard)
```

### Example 2: Sales Forecasting with Covariates

```python
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load sales data
df = pd.read_csv("sales.csv", parse_dates=['date'])

# Add features
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Create TimeSeriesDataFrame
ts_df = TimeSeriesDataFrame(df, timestamp_column='date')

# Split
train = ts_df.slice_by_timestep(0, -30)
test = ts_df.slice_by_timestep(-30, None)

# Train with known future covariates
predictor = TimeSeriesPredictor(
    prediction_length=30,
    target='sales',
    known_covariates_names=['day_of_week', 'month', 'is_weekend'],
    eval_metric="MAPE"
)

predictor.fit(train, presets="best_quality")

# Create future covariates
future_dates = pd.date_range(test.index.get_level_values('timestamp')[0], 
                             periods=30, freq='D')
future_cov = pd.DataFrame({
    'timestamp': future_dates,
    'day_of_week': future_dates.dayofweek,
    'month': future_dates.month,
    'is_weekend': (future_dates.dayofweek >= 5).astype(int)
})

# Predict with future covariates
predictions = predictor.predict(
    train,
    known_covariates=TimeSeriesDataFrame(future_cov)
)

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.plot(test.to_regular_index()['sales'], label='Actual', linewidth=2)
plt.plot(predictions.to_regular_index()['mean'], label='Forecast', linewidth=2)
plt.fill_between(
    predictions.to_regular_index().index,
    predictions.to_regular_index()['0.1'],
    predictions.to_regular_index()['0.9'],
    alpha=0.3,
    label='90% Interval'
)
plt.legend()
plt.title("Sales Forecast")
plt.show()
```

### Example 3: Energy Demand Forecasting

```python
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load hourly energy data
df = pd.read_csv("energy.csv", parse_dates=['timestamp'])

# Add weather and calendar features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & 
                          (df['day_of_week'] < 5)).astype(int)

ts_df = TimeSeriesDataFrame(df, timestamp_column='timestamp')

# Split
train = ts_df.slice_by_timestep(0, -168)  # Hold out 1 week
test = ts_df.slice_by_timestep(-168, None)

# Configure models for hourly data with strong patterns
hyperparameters = {
    "DeepAR": {
        "epochs": 50,
        "context_length": 168,  # 1 week lookback
        "num_batches_per_epoch": 50
    },
    "RecursiveTabular": {
        "model": "GBM",
        "context_length": 168,
        "feature_generator_kwargs": {
            "lag_features": [1, 24, 48, 168],  # 1h, 1d, 2d, 1w
            "rolling_features": {
                "mean": [24, 168],
                "std": [24, 168]
            }
        }
    }
}

# Train
predictor = TimeSeriesPredictor(
    prediction_length=24,  # Forecast 24 hours
    target='demand',
    known_covariates_names=['hour', 'day_of_week', 'month', 'is_business_hour'],
    eval_metric="MASE"
)

predictor.fit(
    train,
    hyperparameters=hyperparameters,
    time_limit=1800
)

# Evaluate
leaderboard = predictor.leaderboard(test)
print("\nModel Leaderboard:")
print(leaderboard)

# Get best model
best_model = predictor.get_model_best()
print(f"\nBest model: {best_model}")

# Feature importance
if 'RecursiveTabular' in predictor.model_names():
    fi = predictor.feature_importance(test, model='RecursiveTabular')
    print("\nTop 10 Features:")
    print(fi.head(10))
```

### Example 4: Stock Price Prediction

```python
import pandas as pd
import yfinance as yf
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Download stock data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="2y", interval="1d")

# Feature engineering
df['returns'] = df['Close'].pct_change()
df['volatility'] = df['returns'].rolling(20).std()
df['ma_5'] = df['Close'].rolling(5).mean()
df['ma_20'] = df['Close'].rolling(20).mean()
df['rsi'] = compute_rsi(df['Close'], 14)  # Implement RSI

df = df.dropna()
df['timestamp'] = df.index

# Create TimeSeriesDataFrame
ts_df = TimeSeriesDataFrame(df[['timestamp', 'Close', 'Volume', 
                                 'volatility', 'ma_5', 'ma_20', 'rsi']])

# Split
train = ts_df.slice_by_timestep(0, -30)
test = ts_df.slice_by_timestep(-30, None)

# Predict
predictor = TimeSeriesPredictor(
    prediction_length=5,  # 5-day forecast
    target='Close',
    eval_metric="MAPE",
    quantile_levels=[0.1, 0.5, 0.9]
)

predictor.fit(train, presets="best_quality", time_limit=1200)

# Generate forecasts
predictions = predictor.predict(train)

# Extract confidence intervals
forecast_df = predictions.to_regular_index()
print("\n5-Day Forecast:")
print(forecast_df[['mean', '0.1', '0.9']])
```

### Example 5: Website Traffic Prediction

```python
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load traffic data
df = pd.read_csv("website_traffic.csv", parse_dates=['date'])

# Add temporal features
df['day_of_week'] = df['date'].dt.dayofweek
df['week_of_year'] = df['date'].dt.isocalendar().week
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Add marketing campaign indicator (known future)
# Assume we know campaign schedule in advance
df['is_campaign'] = 0  # Set to 1 for campaign days

ts_df = TimeSeriesDataFrame(df, timestamp_column='date')

# Train-test split
train = ts_df.slice_by_timestep(0, -14)
test = ts_df.slice_by_timestep(-14, None)

# Configure
predictor = TimeSeriesPredictor(
    prediction_length=14,
    target='visits',
    known_covariates_names=['day_of_week', 'is_weekend', 'is_campaign'],
    eval_metric="WAPE"
)

# Multiple validation windows for robust evaluation
predictor.fit(
    train,
    num_val_windows=4,
    val_step_size=7,
    presets="high_quality"
)

# Leaderboard
lb = predictor.leaderboard(test)
print(lb[['model', 'score_test', 'pred_time_test']])

# Predictions
predictions = predictor.predict(train)
```

-----

## Best Practices

### 1. Data Preparation

- **Regular frequency**: Ensure consistent time intervals
- **Handle missing values**: Use `fill_missing_values()` before training
- **Sufficient history**: At least 2-3x prediction_length recommended
- **Stationarity**: Check for trends and seasonality

### 2. Model Selection

- **Start simple**: Begin with statistical baselines (Naive, ETS, ARIMA)
- **Use presets**: Start with “medium_quality”, upgrade if needed
- **Enable ensemble**: Almost always improves performance
- **Cross-validation**: Use multiple validation windows

### 3. Feature Engineering

- **Datetime features**: Always useful for most models
- **Lag features**: Critical for tree-based models
- **Domain knowledge**: Add business-specific features
- **Future covariates**: Only use truly known future values

### 4. Hyperparameter Tuning

- **Default first**: AutoGluon defaults are well-tuned
- **Context length**: Match your seasonal period
- **Validation**: Use appropriate num_val_windows
- **Time budget**: Allow sufficient time_limit for deep learning

### 5. Evaluation

- **Multiple metrics**: Don’t rely on single metric
- **Hold-out set**: Always evaluate on unseen data
- **Leaderboard**: Check multiple models for robustness
- **Residual analysis**: Plot errors to find patterns

### 6. Production Deployment

- **Refit on full data**: Use `refit_full()` before deployment
- **Save models**: Use `predictor.save()` for persistence
- **Monitor performance**: Track forecast accuracy over time
- **Regular retraining**: Update models with new data

### 7. Common Pitfalls to Avoid

- Don’t use data leakage (future info in features)
- Don’t ignore validation scores (overfit risk)
- Don’t skip baseline models (always compare)
- Don’t use test data for model selection
- Don’t forget to handle seasonality appropriately

-----

## Troubleshooting

### Common Issues

#### 1. “Frequency cannot be inferred”

```python
# Solution: Explicitly set frequency
ts_df = TimeSeriesDataFrame(df, freq='D')  # Daily
# or
predictor = TimeSeriesPredictor(prediction_length=24, freq='H')  # Hourly
```

#### 2. “Not enough data”

```python
# Solution: Increase history or reduce prediction_length
# Recommended: history >= 2-3x prediction_length
predictor = TimeSeriesPredictor(prediction_length=12)  # Reduced from 24
```

#### 3. “Model training failed”

```python
# Solution: Check for:
# - Missing values
# - Irregular timestamps
# - Insufficient data

# Fill missing values
ts_df = ts_df.fill_missing_values(method='ffill')

# Check data quality
print(ts_df.isnull().sum())
print(f"Unique timestamps: {len(ts_df.index.get_level_values('timestamp').unique())}")
```

#### 4. “Poor forecast accuracy”

```python
# Solutions:
# 1. Add more historical data
# 2. Include relevant features
# 3. Try different models
# 4. Increase time_limit
# 5. Use better presets

predictor.fit(
    train_data,
    presets="best_quality",
    time_limit=3600,  # 1 hour
    hyperparameters={
        "DeepAR": {"epochs": 100},
        "RecursiveTabular": {"model": "GBM"}
    }
)
```

#### 5. “Memory error”

```python
# Solutions:
# 1. Reduce num_val_windows
# 2. Use smaller models
# 3. Reduce context_length
# 4. Disable caching

predictor.fit(
    train_data,
    num_val_windows=2,  # Reduced from 5
    cache_predictions=False,
    excluded_model_types=["TemporalFusionTransformer"]
)
```

#### 6. “Slow training”

```python
# Solutions:
# 1. Use faster preset
# 2. Reduce time_limit
# 3. Exclude slow models
# 4. Use GPU for deep learning

predictor.fit(
    train_data,
    presets="fast_training",
    excluded_model_types=["TemporalFusionTransformer", "PatchTST"]
)
```

-----

## Quick Reference Card Summary

### Installation

```bash
pip install autogluon.timeseries[all]
```

### Minimal Working Example

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load data
train = TimeSeriesDataFrame.from_path("train.csv")

# Train
predictor = TimeSeriesPredictor(prediction_length=24)
predictor.fit(train)

# Predict
predictions = predictor.predict(train)
```

### Key Parameters

- `prediction_length`: Forecast horizon (required)
- `target`: Target column name
- `eval_metric`: MASE, MAPE, WQL, RMSE, etc.
- `presets`: fast_training, medium_quality, high_quality, best_quality

### Top Models for Single Series

1. **DeepAR**: Complex patterns, probabilistic
1. **AutoARIMA**: Classical, linear trends
1. **RecursiveTabular**: With covariates, interpretable
1. **ETS**: Simple, seasonal data
1. **TemporalFusionTransformer**: State-of-art, slow

### Essential Methods

```python
predictor.fit(train_data)                    # Train
predictor.predict(data)                      # Forecast
predictor.leaderboard(test_data)            # Rankings
predictor.score(test_data, metric="MASE")   # Evaluate
predictor.save()                            # Save
TimeSeriesPredictor.load(path)              # Load
```

-----

## Additional Resources

- **Official Documentation**: https://auto.gluon.ai/stable/tutorials/timeseries/
- **GitHub**: https://github.com/autogluon/autogluon
- **Tutorials**: https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html
- **API Reference**: https://auto.gluon.ai/stable/api/autogluon.timeseries.html

-----

**End of Reference Card**
