# UnifiedMetrics.jl

A comprehensive Julia package for evaluating machine learning models. Provides **97+ metrics** across regression, classification, binary classification, information retrieval, and time series forecasting.

## Features

- **Time Series Forecasting**: 13 specialized metrics including MASE, RMSSE, tracking signal, Winkler score - with comprehensive guidance on scale-independent evaluation, bias detection, and probabilistic forecasting
- **Regression**: 32 metrics including MAE, RMSE, MAPE, R², Huber loss, and more
- **Classification**: 14 metrics including accuracy, balanced accuracy, Cohen's Kappa, MCC
- **Binary Classification**: 23 metrics including AUC, precision, recall, F-score, Brier score
- **Information Retrieval**: 15 metrics including NDCG, MRR, MAP@K, hit rate

## Installation

```julia
using Pkg
Pkg.add("UnifiedMetrics")
```

Or in the Julia REPL package mode (press `]`):

```
add UnifiedMetrics
```

## Quick Example

### Time Series Forecasting

```julia
using UnifiedMetrics

actual = [100.0, 110.0, 105.0, 115.0, 120.0, 125.0, 130.0, 128.0]
predicted = [98.0, 108.0, 107.0, 113.0, 118.0, 123.0, 128.0, 126.0]

# Scale-independent metrics (compare to naive forecast)
mase(actual, predicted)           # < 1 means better than naive
mase(actual, predicted, m=12)     # For seasonal data (monthly with yearly pattern)

# Bias detection
tracking_signal(actual, predicted)  # |TS| > 4 indicates systematic bias

# Prediction intervals
lower = [90.0, 100.0, 99.0, 105.0, 110.0, 115.0, 120.0, 118.0]
upper = [106.0, 116.0, 115.0, 121.0, 126.0, 131.0, 136.0, 134.0]
coverage_probability(actual, lower, upper)  # Should match confidence level
winkler_score(actual, lower, upper, alpha=0.05)  # Lower is better
```

### Regression

```julia
using UnifiedMetrics

actual = [1.0, 2.0, 3.0, 4.0, 5.0]
predicted = [1.1, 2.1, 2.9, 4.2, 4.8]

mae(actual, predicted)      # Mean Absolute Error
rmse(actual, predicted)     # Root Mean Squared Error
mape(actual, predicted)     # Mean Absolute Percentage Error
```

### Classification

```julia
using UnifiedMetrics

actual = [1, 1, 0, 0, 1, 0]
predicted = [1, 0, 0, 1, 1, 0]

accuracy(actual, predicted)   # Classification Accuracy
precision(actual, predicted)  # Precision
recall(actual, predicted)     # Recall
fbeta_score(actual, predicted) # F1 Score
```

## Documentation Overview

- **[Getting Started](@ref)**: Installation and basic usage
- **[Choosing the Right Metric](@ref)**: Comprehensive guide on which metric to use for your problem
- **[Time Series Forecasting](@ref)**: In-depth guide to evaluating forecasting models - scale-independent metrics, bias detection, prediction intervals, and more
- **Other Metrics**: Detailed documentation for regression, classification, binary classification, and information retrieval
- **[API Reference](@ref)**: Complete function reference

## Why UnifiedMetrics.jl?

1. **Comprehensive**: One package for all evaluation needs
2. **Consistent API**: All metrics follow the same `metric(actual, predicted)` pattern
3. **Well-documented**: Every function includes docstrings with examples
4. **Pure Julia**: No external dependencies beyond StatsBase
5. **Production-ready**: Handles edge cases gracefully
6. **Forecasting-focused**: Special emphasis on time series evaluation with M-competition recommended metrics

## Contents

```@contents
Pages = [
    "getting_started.md",
    "choosing_metrics.md",
    "time_series.md",
    "regression.md",
    "classification.md",
    "binary_classification.md",
    "information_retrieval.md",
    "api.md",
]
Depth = 2
```
