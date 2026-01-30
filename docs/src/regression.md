# Regression Metrics

Metrics for evaluating models that predict continuous values.

## Overview

Regression metrics measure the difference between predicted and actual continuous values. UnifiedMetrics.jl provides 32 regression metrics grouped into several categories:

- **Basic error metrics**: MAE, RMSE, MSE
- **Percentage-based metrics**: MAPE, SMAPE, WMAPE
- **Relative metrics**: R², RAE, RSE
- **Robust metrics**: Huber loss, MdAE
- **GLM-specific metrics**: Tweedie deviance, Poisson/Gamma deviance

## Quick Reference

| Metric | Formula | Best For |
|--------|---------|----------|
| `mae` | mean(\|y - ŷ\|) | General purpose, interpretable |
| `rmse` | √mean((y - ŷ)²) | Penalizing large errors |
| `mape` | mean(\|y - ŷ\| / \|y\|) | Percentage reporting |
| `smape` | Symmetric MAPE | When actuals have zeros |
| `explained_variation` | 1 - RSS/TSS | Model explanation power |

## Basic Error Metrics

### Absolute Error Family

```julia
ae(actual, predicted)   # Elementwise absolute error
mae(actual, predicted)  # Mean absolute error
mdae(actual, predicted) # Median absolute error
```

**When to use**:
- `mae`: Standard choice when all errors are equally important
- `mdae`: When you want robustness to outliers (median instead of mean)

### Squared Error Family

```julia
se(actual, predicted)   # Elementwise squared error
sse(actual, predicted)  # Sum of squared errors
mse(actual, predicted)  # Mean squared error
rmse(actual, predicted) # Root mean squared error
```

**When to use**:
- `rmse`: When large errors are particularly costly
- `mse`: As a loss function for optimization (differentiable)

### Normalized RMSE

```julia
nrmse(actual, predicted; normalization=:range)
```

**When to use**: Comparing RMSE across datasets with different scales.

## Percentage-Based Metrics

### Basic Percentage Errors

```julia
ape(actual, predicted)  # Elementwise absolute percentage error
mape(actual, predicted) # Mean absolute percentage error
```

**Warning**: MAPE is undefined when actual values are zero.

### Symmetric and Weighted Alternatives

```julia
smape(actual, predicted) # Symmetric MAPE
wmape(actual, predicted) # Weighted MAPE
mpe(actual, predicted)   # Mean percentage error (signed)
```

**When to use**:
- `smape`: When actuals may be zero; bounded between 0 and 2
- `wmape`: When larger values should have more influence
- `mpe`: When you want to detect systematic over/under-prediction

## Bias Metrics

```julia
bias(actual, predicted)         # Average bias
percent_bias(actual, predicted) # Percentage bias
```

**Interpretation**:
- Positive bias: Model under-predicts on average
- Negative bias: Model over-predicts on average

## Logarithmic Error Metrics

```julia
sle(actual, predicted)   # Elementwise squared log error
msle(actual, predicted)  # Mean squared log error
rmsle(actual, predicted) # Root mean squared log error
```

**When to use**:
- When targets span multiple orders of magnitude (prices, populations)
- When under-prediction is worse than over-prediction
- **Note**: Only for positive values

## Relative Error Metrics

```julia
rse(actual, predicted)  # Relative squared error
rrse(actual, predicted) # Root relative squared error
rae(actual, predicted)  # Relative absolute error
```

**Interpretation**: Error relative to a naive model that predicts the mean.

## Model Explanation Metrics

```julia
explained_variation(actual, predicted)      # R² (coefficient of determination)
adjusted_r2(actual, predicted, n_features)  # Adjusted R²
```

**Interpretation for R²**:
- R² = 1: Perfect predictions
- R² = 0: Model is no better than predicting the mean
- R² < 0: Model is worse than predicting the mean

## Extreme Error Metrics

```julia
max_error(actual, predicted) # Maximum absolute error
max_ae(actual, predicted)    # Alias for max_error
```

**When to use**: When worst-case error matters (safety-critical applications).

## Robust Loss Functions

```julia
huber_loss(actual, predicted; delta=1.0)
log_cosh_loss(actual, predicted)
```

**When to use**:
- When data has outliers
- As training loss functions for neural networks
- `huber_loss`: Quadratic for small errors, linear for large
- `log_cosh_loss`: Similar to Huber but twice differentiable

## Quantile Loss

```julia
quantile_loss(actual, predicted; quantile=0.5)
pinball_loss(actual, predicted; quantile=0.5)  # Alias
```

**When to use**:
- Quantile regression
- When asymmetric errors matter
- `quantile=0.9`: Penalize under-prediction more
- `quantile=0.1`: Penalize over-prediction more

## GLM Deviance Metrics

```julia
tweedie_deviance(actual, predicted; power=1.5)
mean_poisson_deviance(actual, predicted)
mean_gamma_deviance(actual, predicted)
d2_tweedie_score(actual, predicted; power=1.5)
```

**When to use**:
- Evaluating Generalized Linear Models
- `mean_poisson_deviance`: Count data (visitors, purchases)
- `mean_gamma_deviance`: Positive continuous with variance ∝ mean² (insurance claims)

## Usage Examples

### Comparing Multiple Models

```julia
using UnifiedMetrics

actual = [10.0, 20.0, 30.0, 40.0, 50.0]
model1_pred = [12.0, 18.0, 31.0, 38.0, 52.0]
model2_pred = [11.0, 21.0, 29.0, 41.0, 48.0]

# Compare using multiple metrics
for (name, pred) in [("Model 1", model1_pred), ("Model 2", model2_pred)]
    println("$name:")
    println("  MAE:  $(round(mae(actual, pred), digits=2))")
    println("  RMSE: $(round(rmse(actual, pred), digits=2))")
    println("  R²:   $(round(explained_variation(actual, pred), digits=3))")
end
```

### Handling Data with Zeros

```julia
actual = [0.0, 10.0, 20.0, 0.0, 30.0]
predicted = [1.0, 9.0, 21.0, 2.0, 28.0]

# MAPE will be Inf due to zeros
# mape(actual, predicted)  # Don't use!

# Use SMAPE or WMAPE instead
smape(actual, predicted)  # Bounded, handles zeros
wmape(actual, predicted)  # Weighted by actuals
```

### Robust Evaluation with Outliers

```julia
actual = [1.0, 2.0, 3.0, 4.0, 100.0]  # 100 is an outlier
predicted = [1.1, 2.1, 2.9, 4.1, 5.0]

# Standard metrics are heavily influenced by outlier
rmse(actual, predicted)  # ~42.5
mae(actual, predicted)   # ~19.0

# Robust alternatives
mdae(actual, predicted)  # ~0.1 (median)
huber_loss(actual, predicted, delta=1.0)  # Less sensitive
```

See the [API Reference](@ref) for complete function documentation.
