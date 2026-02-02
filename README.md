<div align="center">
<img src="docs/src/assets/logo.svg"/>
</div>

# UnifiedMetrics.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://taf-society.github.io/UnifiedMetrics.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://taf-society.github.io/UnifiedMetrics.jl/dev/) [![Build Status](https://github.com/taf-society/UnifiedMetrics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/taf-society/UnifiedMetrics.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/taf-society/UnifiedMetrics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/taf-society/UnifiedMetrics.jl)

A comprehensive Julia package for evaluating machine learning models. Provides **97 metrics** across regression, classification, binary classification, information retrieval, and time series forecasting.

## Installation

```julia
using Pkg
Pkg.add("UnifiedMetrics")
```

Or in the Julia REPL package mode (press `]`):

```
add UnifiedMetrics
```

## Quick Start

```julia
using UnifiedMetrics

# Regression
actual = [1.0, 2.0, 3.0, 4.0, 5.0]
predicted = [1.1, 2.1, 2.9, 4.2, 4.8]

mae(actual, predicted)      # Mean Absolute Error
rmse(actual, predicted)     # Root Mean Squared Error
mape(actual, predicted)     # Mean Absolute Percentage Error

# Classification
actual = [1, 1, 0, 0, 1, 0]
predicted = [1, 0, 0, 1, 1, 0]

accuracy(actual, predicted)   # Classification Accuracy
precision(actual, predicted)  # Precision
recall(actual, predicted)     # Recall
fbeta_score(actual, predicted) # F1 Score
```

## Metrics Overview

### Regression Metrics (32 functions)

| Function | Description |
|----------|-------------|
| `ae(actual, predicted)` | Absolute Error (elementwise) |
| `mae(actual, predicted)` | Mean Absolute Error |
| `mdae(actual, predicted)` | Median Absolute Error |
| `se(actual, predicted)` | Squared Error (elementwise) |
| `sse(actual, predicted)` | Sum of Squared Errors |
| `mse(actual, predicted)` | Mean Squared Error |
| `rmse(actual, predicted)` | Root Mean Squared Error |
| `nrmse(actual, predicted; normalization=:range)` | Normalized RMSE |
| `bias(actual, predicted)` | Bias (mean error) |
| `percent_bias(actual, predicted)` | Percent Bias |
| `mpe(actual, predicted)` | Mean Percentage Error |
| `ape(actual, predicted)` | Absolute Percentage Error (elementwise) |
| `mape(actual, predicted)` | Mean Absolute Percentage Error |
| `smape(actual, predicted)` | Symmetric MAPE |
| `wmape(actual, predicted)` | Weighted MAPE |
| `sle(actual, predicted)` | Squared Log Error (elementwise) |
| `msle(actual, predicted)` | Mean Squared Log Error |
| `rmsle(actual, predicted)` | Root Mean Squared Log Error |
| `rse(actual, predicted)` | Relative Squared Error |
| `rrse(actual, predicted)` | Root Relative Squared Error |
| `rae(actual, predicted)` | Relative Absolute Error |
| `explained_variation(actual, predicted)` | R² (Coefficient of Determination) |
| `adjusted_r2(actual, predicted, n_features)` | Adjusted R² |
| `max_error(actual, predicted)` | Maximum Absolute Error |
| `huber_loss(actual, predicted; delta=1.0)` | Huber Loss |
| `log_cosh_loss(actual, predicted)` | Log-Cosh Loss |
| `quantile_loss(actual, predicted; quantile=0.5)` | Quantile (Pinball) Loss |
| `tweedie_deviance(actual, predicted; power=1.5)` | Tweedie Deviance |
| `mean_poisson_deviance(actual, predicted)` | Mean Poisson Deviance |
| `mean_gamma_deviance(actual, predicted)` | Mean Gamma Deviance |
| `d2_tweedie_score(actual, predicted; power=1.5)` | D² Tweedie Score |

### Classification Metrics (14 functions)

| Function | Description |
|----------|-------------|
| `ce(actual, predicted)` | Classification Error |
| `accuracy(actual, predicted)` | Classification Accuracy |
| `balanced_accuracy(actual, predicted)` | Balanced Accuracy (macro-averaged recall) |
| `cohens_kappa(actual, predicted)` | Cohen's Kappa |
| `matthews_corrcoef(actual, predicted)` | Matthews Correlation Coefficient |
| `mcc(actual, predicted)` | Alias for Matthews Correlation Coefficient |
| `confusion_matrix(actual, predicted)` | Confusion Matrix |
| `top_k_accuracy(actual, predicted_probs, k)` | Top-K Accuracy |
| `hamming_loss(actual, predicted)` | Hamming Loss |
| `zero_one_loss(actual, predicted)` | Zero-One Loss |
| `hinge_loss(actual, predicted)` | Hinge Loss (for SVMs) |
| `squared_hinge_loss(actual, predicted)` | Squared Hinge Loss |
| `ScoreQuadraticWeightedKappa(...)` | Quadratic Weighted Kappa |
| `MeanQuadraticWeightedKappa(...)` | Mean Quadratic Weighted Kappa |

### Binary Classification Metrics (23 functions)

| Function | Description |
|----------|-------------|
| `auc(actual, predicted)` | Area Under ROC Curve |
| `ll(actual, predicted)` | Log Loss (elementwise) |
| `logloss(actual, predicted)` | Mean Log Loss |
| `precision(actual, predicted)` | Precision (PPV) |
| `recall(actual, predicted)` | Recall (Sensitivity, TPR) |
| `sensitivity(actual, predicted)` | Alias for Recall |
| `specificity(actual, predicted)` | Specificity (TNR) |
| `npv(actual, predicted)` | Negative Predictive Value |
| `fpr(actual, predicted)` | False Positive Rate |
| `fnr(actual, predicted)` | False Negative Rate |
| `fbeta_score(actual, predicted; beta=1.0)` | F-beta Score |
| `brier_score(actual, predicted)` | Brier Score |
| `gini_coefficient(actual, predicted)` | Gini Coefficient |
| `ks_statistic(actual, predicted)` | Kolmogorov-Smirnov Statistic |
| `lift(actual, predicted; percentile=0.1)` | Lift |
| `gain(actual, predicted; percentile=0.1)` | Cumulative Gain |
| `youden_j(actual, predicted)` | Youden's J (Informedness) |
| `markedness(actual, predicted)` | Markedness |
| `fowlkes_mallows_index(actual, predicted)` | Fowlkes-Mallows Index |
| `positive_likelihood_ratio(actual, predicted)` | Positive Likelihood Ratio (LR+) |
| `negative_likelihood_ratio(actual, predicted)` | Negative Likelihood Ratio (LR-) |
| `diagnostic_odds_ratio(actual, predicted)` | Diagnostic Odds Ratio |

### Information Retrieval Metrics (15 functions)

| Function | Description |
|----------|-------------|
| `f1(actual, predicted)` | F1 Score (IR context) |
| `apk(k, actual, predicted)` | Average Precision at K |
| `mapk(k, actual, predicted)` | Mean Average Precision at K |
| `dcg(relevance; k=nothing)` | Discounted Cumulative Gain |
| `idcg(relevance; k=nothing)` | Ideal DCG |
| `ndcg(relevance; k=nothing)` | Normalized DCG |
| `mean_ndcg(relevances; k=nothing)` | Mean NDCG |
| `mrr(actual, predicted)` | Mean Reciprocal Rank |
| `reciprocal_rank(actual, predicted)` | Reciprocal Rank |
| `hit_rate(actual, predicted; k=10)` | Hit Rate |
| `recall_at_k(actual, predicted; k=10)` | Recall at K |
| `precision_at_k(actual, predicted; k=10)` | Precision at K |
| `f1_at_k(actual, predicted; k=10)` | F1 at K |
| `coverage(predicted, catalog)` | Catalog Coverage |
| `novelty(predicted, item_popularity)` | Novelty |

### Time Series Metrics (13 functions)

| Function | Description |
|----------|-------------|
| `mase(actual, predicted; m=1)` | Mean Absolute Scaled Error |
| `msse(actual, predicted; m=1)` | Mean Squared Scaled Error |
| `rmsse(actual, predicted; m=1)` | Root Mean Squared Scaled Error |
| `tracking_signal(actual, predicted)` | Tracking Signal |
| `forecast_bias(actual, predicted)` | Forecast Bias |
| `theil_u1(actual, predicted)` | Theil's U1 Statistic |
| `theil_u2(actual, predicted; m=1)` | Theil's U2 Statistic |
| `wape(actual, predicted)` | Weighted Absolute Percentage Error |
| `directional_accuracy(actual, predicted)` | Directional Accuracy |
| `coverage_probability(actual, lower, upper)` | Coverage Probability |
| `pinball_loss_series(actual, predicted; quantile=0.5)` | Pinball Loss |
| `winkler_score(actual, lower, upper; alpha=0.05)` | Winkler Score |
| `autocorrelation_error(actual, predicted; max_lag=10)` | Autocorrelation Error |

## Detailed Usage Examples

### Regression

```julia
using UnifiedMetrics

actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]

# Basic error metrics
mae(actual, predicted)    # 0.25
rmse(actual, predicted)   # 0.334
mape(actual, predicted)   # 0.0658 (6.58%)

# R-squared (coefficient of determination)
explained_variation(actual, predicted)  # 0.958

# Robust loss functions (less sensitive to outliers)
huber_loss(actual, predicted, delta=1.0)
log_cosh_loss(actual, predicted)

# Quantile regression loss
quantile_loss(actual, predicted, quantile=0.9)  # Penalize under-prediction more

# Normalized RMSE for comparing across different scales
nrmse(actual, predicted, normalization=:range)  # Normalized by range
nrmse(actual, predicted, normalization=:mean)   # CV(RMSE)

# For count data or GLMs
mean_poisson_deviance(actual, predicted)
mean_gamma_deviance(actual, predicted)
```

### Classification

```julia
using UnifiedMetrics

actual = ["cat", "dog", "cat", "cat", "dog", "bird"]
predicted = ["cat", "dog", "dog", "cat", "dog", "cat"]

# Basic metrics
accuracy(actual, predicted)           # 0.667
ce(actual, predicted)                 # 0.333 (classification error)

# Cohen's Kappa (accounts for chance agreement)
cohens_kappa(actual, predicted)

# Balanced accuracy (useful for imbalanced datasets)
balanced_accuracy(actual, predicted)

# Confusion matrix
cm = confusion_matrix(actual, predicted)
cm[:matrix]   # The confusion matrix
cm[:labels]   # Class labels

# For binary classification with integer labels
actual_bin = [1, 1, 0, 0, 1, 0]
predicted_bin = [1, 0, 0, 1, 1, 0]

matthews_corrcoef(actual_bin, predicted_bin)  # MCC: best single metric for binary

# Top-K accuracy for multi-class with probabilities
actual_idx = [1, 2, 3, 1]  # True class indices (1-indexed)
predicted_probs = [0.8 0.1 0.1;   # Class probabilities per sample
                   0.2 0.5 0.3;
                   0.1 0.3 0.6;
                   0.3 0.4 0.3]
top_k_accuracy(actual_idx, predicted_probs, 2)  # Top-2 accuracy
```

### Binary Classification

```julia
using UnifiedMetrics

actual = [1, 1, 1, 0, 0, 0]
predicted_scores = [0.9, 0.8, 0.4, 0.5, 0.3, 0.2]  # Probabilities
predicted_labels = [1, 1, 0, 1, 0, 0]               # Binary predictions

# ROC-based metrics (use probability scores)
auc(actual, predicted_scores)              # Area Under ROC Curve
gini_coefficient(actual, predicted_scores) # Gini = 2*AUC - 1
ks_statistic(actual, predicted_scores)     # Max separation between classes

# Probability calibration
brier_score(actual, predicted_scores)      # Lower is better
logloss(actual, predicted_scores)          # Cross-entropy loss

# Threshold-based metrics (use binary predictions)
precision(actual, predicted_labels)    # PPV
recall(actual, predicted_labels)       # Sensitivity, TPR
specificity(actual, predicted_labels)  # TNR
npv(actual, predicted_labels)          # Negative Predictive Value

# F-scores
fbeta_score(actual, predicted_labels)              # F1 (beta=1)
fbeta_score(actual, predicted_labels, beta=0.5)   # F0.5 (precision-weighted)
fbeta_score(actual, predicted_labels, beta=2.0)   # F2 (recall-weighted)

# Error rates
fpr(actual, predicted_labels)  # False Positive Rate
fnr(actual, predicted_labels)  # False Negative Rate

# Combined metrics
youden_j(actual, predicted_labels)     # Sensitivity + Specificity - 1
markedness(actual, predicted_labels)   # Precision + NPV - 1
fowlkes_mallows_index(actual, predicted_labels)  # sqrt(Precision * Recall)

# Lift and gain analysis
lift(actual, predicted_scores, percentile=0.1)  # Lift in top 10%
gain(actual, predicted_scores, percentile=0.1)  # % of positives in top 10%

# Likelihood ratios
positive_likelihood_ratio(actual, predicted_labels)  # LR+
negative_likelihood_ratio(actual, predicted_labels)  # LR-
diagnostic_odds_ratio(actual, predicted_labels)      # DOR
```

### Information Retrieval

```julia
using UnifiedMetrics

# Ranking evaluation with relevance scores
relevance = [3, 2, 3, 0, 1, 2]  # Relevance scores in ranked order

dcg(relevance)         # Discounted Cumulative Gain
ndcg(relevance)        # Normalized DCG (0-1 scale)
ndcg(relevance, k=3)   # NDCG at position 3

# Multiple queries
relevances = [[3, 2, 1, 0], [2, 1, 2, 1], [1, 1, 0, 0]]
mean_ndcg(relevances, k=3)

# Set-based retrieval metrics
actual_docs = ["a", "c", "d"]      # Relevant documents
predicted_docs = ["d", "e", "a"]   # Retrieved documents (ranked)

f1(actual_docs, predicted_docs)            # F1 score
precision_at_k(actual_docs, predicted_docs, k=2)
recall_at_k(actual_docs, predicted_docs, k=2)

# Average Precision
apk(3, actual_docs, predicted_docs)

# Mean Average Precision over multiple queries
actual_list = [["a", "b"], ["c"], ["d", "e"]]
predicted_list = [["a", "c", "d"], ["x", "c"], ["e", "f"]]
mapk(2, actual_list, predicted_list)

# Mean Reciprocal Rank
mrr(actual_list, predicted_list)

# Hit rate (at least one relevant in top-k)
hit_rate(actual_list, predicted_list, k=2)

# Recommendation system metrics
catalog = ["a", "b", "c", "d", "e", "f"]
recommendations = [["a", "b"], ["a", "c"], ["b", "d"]]
coverage(recommendations, catalog)  # What % of catalog was recommended

# Novelty (recommending less popular items)
popularity = Dict("a" => 0.9, "b" => 0.5, "c" => 0.1, "d" => 0.05)
novelty(recommendations, popularity)
```

### Time Series Forecasting

```julia
using UnifiedMetrics

actual = [100.0, 110.0, 105.0, 115.0, 120.0, 125.0, 130.0, 128.0]
predicted = [98.0, 108.0, 110.0, 112.0, 118.0, 127.0, 128.0, 130.0]

# Scaled error metrics (compare to naive forecast)
mase(actual, predicted)        # m=1: compare to random walk
mase(actual, predicted, m=4)   # m=4: compare to seasonal naive (quarterly)
mase(actual, predicted, m=12)  # m=12: compare to seasonal naive (monthly)

# The 'm' parameter represents the seasonal period:
# - m=1:  Non-seasonal (naive = previous value)
# - m=4:  Quarterly seasonality
# - m=7:  Weekly seasonality (daily data)
# - m=12: Monthly seasonality
# - m=52: Weekly seasonality (weekly data)

msse(actual, predicted, m=1)   # Mean Squared Scaled Error
rmsse(actual, predicted, m=1)  # Root Mean Squared Scaled Error

# Bias detection
forecast_bias(actual, predicted)    # Positive = under-forecasting
tracking_signal(actual, predicted)  # Outside [-4, 4] indicates bias

# Theil's inequality coefficients
theil_u1(actual, predicted)         # 0 = perfect, 1 = worst
theil_u2(actual, predicted)         # <1 = better than naive

# Directional accuracy
directional_accuracy(actual, predicted)  # % correct direction predictions

# WAPE (handles zeros better than MAPE)
wape(actual, predicted)

# Prediction interval evaluation
lower = [95.0, 105.0, 100.0, 108.0, 112.0, 120.0, 122.0, 123.0]
upper = [105.0, 115.0, 115.0, 122.0, 128.0, 135.0, 138.0, 137.0]

coverage_probability(actual, lower, upper)  # Should match confidence level
winkler_score(actual, lower, upper, alpha=0.05)  # Lower is better

# Quantile forecast evaluation
pinball_loss_series(actual, predicted, quantile=0.5)  # Median forecast
pinball_loss_series(actual, predicted, quantile=0.9)  # 90th percentile

# Autocorrelation structure preservation
autocorrelation_error(actual, predicted, max_lag=5)
```

## Interpreting Common Metrics

### Regression Metrics Guide

| Metric | Range | Interpretation |
|--------|-------|----------------|
| MAE | [0, ∞) | Average absolute error; same units as target |
| RMSE | [0, ∞) | Penalizes large errors more; same units as target |
| MAPE | [0, ∞) | Percentage error; undefined when actual=0 |
| SMAPE | [0, 2] | Symmetric percentage error; handles zeros better |
| R² | (-∞, 1] | 1=perfect, 0=mean baseline, <0=worse than mean |
| MASE | [0, ∞) | <1 better than naive, >1 worse than naive |

### Classification Metrics Guide

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Accuracy | [0, 1] | Proportion correct; misleading for imbalanced data |
| Balanced Accuracy | [0, 1] | Macro-averaged recall; better for imbalanced |
| MCC | [-1, 1] | Best single metric for binary; 0=random, 1=perfect |
| Cohen's Kappa | [-1, 1] | Agreement beyond chance; 0=chance, 1=perfect |
| AUC | [0, 1] | 0.5=random, 1=perfect ranking |
| F1 | [0, 1] | Harmonic mean of precision and recall |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

