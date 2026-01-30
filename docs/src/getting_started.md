# Getting Started

This guide will help you get started with UnifiedMetrics.jl quickly.

## Installation

Install UnifiedMetrics.jl using Julia's package manager:

```julia
using Pkg
Pkg.add("UnifiedMetrics")
```

Or in the REPL package mode (press `]`):

```
pkg> add UnifiedMetrics
```

## Basic Usage

All metrics in UnifiedMetrics.jl follow a consistent API pattern:

```julia
metric(actual, predicted)
```

Where:
- `actual` is the ground truth (what actually happened)
- `predicted` is your model's prediction

### Regression Example

```julia
using UnifiedMetrics

# Your actual values and predictions
actual = [1.0, 2.0, 3.0, 4.0, 5.0]
predicted = [1.1, 2.0, 2.8, 4.2, 4.9]

# Common metrics
mae(actual, predicted)      # Mean Absolute Error: 0.12
rmse(actual, predicted)     # Root Mean Squared Error: 0.14
mape(actual, predicted)     # Mean Absolute Percentage Error
explained_variation(actual, predicted)  # R²: 0.99
```

### Classification Example

```julia
using UnifiedMetrics

# Multi-class classification
actual = ["cat", "dog", "cat", "bird", "dog"]
predicted = ["cat", "cat", "cat", "bird", "dog"]

accuracy(actual, predicted)  # 0.8
ce(actual, predicted)        # Classification Error: 0.2
balanced_accuracy(actual, predicted)
cohens_kappa(actual, predicted)
```

### Binary Classification Example

```julia
using UnifiedMetrics

# Binary labels
actual = [1, 1, 1, 0, 0, 0]
predicted_labels = [1, 0, 1, 1, 0, 0]
predicted_probs = [0.9, 0.4, 0.8, 0.6, 0.3, 0.2]

# Threshold-based metrics (use labels)
precision(actual, predicted_labels)
recall(actual, predicted_labels)
fbeta_score(actual, predicted_labels)
mcc(actual, predicted_labels)

# Probability-based metrics (use probabilities)
auc(actual, predicted_probs)
brier_score(actual, predicted_probs)
logloss(actual, predicted_probs)
```

### Information Retrieval Example

```julia
using UnifiedMetrics

# Ranking evaluation with relevance scores
relevance = [3, 2, 3, 0, 1, 2]  # Relevance of items in your ranking
ndcg(relevance)        # Normalized DCG
ndcg(relevance, k=3)   # NDCG at position 3

# Set-based retrieval
actual_relevant = ["doc1", "doc3", "doc5"]
retrieved = ["doc1", "doc2", "doc3", "doc4"]

precision_at_k(actual_relevant, retrieved, k=3)
recall_at_k(actual_relevant, retrieved, k=3)
```

### Time Series Example

```julia
using UnifiedMetrics

# Time series forecasting
actual = [100.0, 110.0, 105.0, 115.0, 120.0, 125.0]
predicted = [98.0, 108.0, 110.0, 112.0, 118.0, 127.0]

# Scale-independent metrics
mase(actual, predicted)          # Non-seasonal
mase(actual, predicted, m=4)     # Quarterly seasonality

# Bias detection
tracking_signal(actual, predicted)
forecast_bias(actual, predicted)

# Prediction intervals
lower = [95.0, 105.0, 100.0, 108.0, 112.0, 120.0]
upper = [105.0, 115.0, 115.0, 122.0, 128.0, 135.0]
coverage_probability(actual, lower, upper)
```

## Understanding Metric Outputs

### Metrics Where Lower is Better

Most error metrics: MAE, RMSE, MSE, MAPE, Brier score, log loss, Hamming loss, etc.

### Metrics Where Higher is Better

- Accuracy, balanced accuracy
- R² (explained variation)
- AUC, Gini coefficient
- Precision, recall, F-score
- NDCG, MRR, hit rate

### Metrics with Specific Interpretations

| Metric | Range | Perfect Score | Random/Baseline |
|--------|-------|---------------|-----------------|
| R² | (-∞, 1] | 1 | 0 |
| AUC | [0, 1] | 1 | 0.5 |
| MCC | [-1, 1] | 1 | 0 |
| Cohen's Kappa | [-1, 1] | 1 | 0 |
| MASE | [0, ∞) | 0 | 1 |

## Tips for Effective Evaluation

1. **Use multiple metrics**: No single metric tells the whole story
2. **Match metric to objective**: Choose metrics that reflect your business goals
3. **Consider data characteristics**: Imbalanced data needs appropriate metrics
4. **Report confidence intervals**: Single numbers can be misleading

## Next Steps

- Read [Choosing the Right Metric](@ref) for guidance on metric selection
- Explore domain-specific pages for detailed metric documentation
- Check the [API Reference](@ref) for complete function signatures
