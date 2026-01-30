# Time Series Forecasting Metrics

A comprehensive guide to evaluating time series forecasting models.

## Why Time Series Metrics Are Different

Time series evaluation has unique challenges that standard regression metrics don't address:

1. **Scale Dependence**: MAE of 10 means nothing without context - is that good or bad?
2. **Benchmark Comparison**: How does your model compare to simple baselines (naive, seasonal naive)?
3. **Temporal Structure**: Errors may be autocorrelated, biased, or directionally wrong
4. **Probabilistic Forecasts**: Modern forecasting produces prediction intervals, not just point forecasts
5. **Multiple Horizons**: Accuracy often degrades as forecast horizon increases

UnifiedMetrics.jl provides 13 specialized metrics to address these challenges.

## Metrics at a Glance

| Metric | Category | Range | Key Insight |
|--------|----------|-------|-------------|
| `mase` | Scaled Error | [0, ∞) | Is model better than naive? |
| `msse` | Scaled Error | [0, ∞) | Squared version of MASE |
| `rmsse` | Scaled Error | [0, ∞) | Same scale as data |
| `tracking_signal` | Bias | (-∞, ∞) | Is forecast systematically off? |
| `forecast_bias` | Bias | (-∞, ∞) | Average over/under prediction |
| `theil_u1` | Benchmark | [0, 1] | Normalized inequality |
| `theil_u2` | Benchmark | [0, ∞) | Comparison to naive |
| `wape` | Percentage | [0, ∞) | Weighted percentage error |
| `directional_accuracy` | Direction | [0, 1] | Up/down prediction accuracy |
| `coverage_probability` | Intervals | [0, 1] | Interval calibration |
| `winkler_score` | Intervals | [0, ∞) | Interval sharpness + calibration |
| `pinball_loss_series` | Quantile | [0, ∞) | Quantile forecast accuracy |
| `autocorrelation_error` | Structure | [0, ∞) | Temporal pattern preservation |

---

## Choosing the Right Time Series Metric

### Decision Flowchart

```
What do you need to evaluate?
│
├─► Point Forecast Accuracy
│   │
│   ├─► Need to compare across different series/scales?
│   │   │
│   │   ├─► YES ──► mase() or rmsse()
│   │   │
│   │   └─► NO ──► mae() or rmse() from regression metrics
│   │
│   └─► Need percentage-based reporting?
│       │
│       ├─► Data has zeros? ──► wape()
│       │
│       └─► No zeros ──► mape() from regression metrics
│
├─► Forecast Bias Detection
│   │
│   ├─► Real-time monitoring ──► tracking_signal()
│   │
│   └─► One-time evaluation ──► forecast_bias()
│
├─► Benchmark Comparison
│   │
│   └─► Is model better than naive forecast? ──► theil_u2() or mase()
│
├─► Direction Prediction (Trading)
│   │
│   └─► directional_accuracy()
│
└─► Probabilistic Forecasts
    │
    ├─► Prediction intervals ──► coverage_probability() + winkler_score()
    │
    └─► Quantile forecasts ──► pinball_loss_series()
```

### Metric Selection by Use Case

| Use Case | Primary Metric | Secondary Metrics |
|----------|---------------|-------------------|
| M-competition style evaluation | `mase` | `rmsse`, `mape` |
| Supply chain forecasting | `wape` | `mase`, `forecast_bias` |
| Demand forecasting | `mase` | `tracking_signal`, `coverage_probability` |
| Financial/trading | `directional_accuracy` | `theil_u2` |
| Weather forecasting | `rmsse` | `coverage_probability`, `winkler_score` |
| Real-time monitoring | `tracking_signal` | `forecast_bias` |
| Model selection | `mase` | `theil_u2`, `winkler_score` |

---

## Scaled Error Metrics

The most important innovation in time series evaluation. These metrics compare your forecast error to the error of a naive benchmark, making them **scale-independent** and **interpretable**.

### MASE (Mean Absolute Scaled Error)

```julia
mase(actual, predicted; m=1)
```

Compute the Mean Absolute Scaled Error. See [API Reference](@ref) for full documentation.

#### Why MASE is the Gold Standard

1. **Scale-independent**: Compare forecasts across products, regions, or time periods with different scales
2. **Interpretable threshold**: MASE < 1 means better than naive, MASE > 1 means worse
3. **Handles zeros**: Unlike MAPE, works with intermittent demand
4. **Symmetric**: Treats over- and under-forecasting equally
5. **Recommended**: Official metric of M3 and M4 forecasting competitions

#### Understanding the Seasonal Period `m`

The `m` parameter defines what "naive forecast" means:

| Your Data | Seasonality | m Value | Naive Forecast |
|-----------|-------------|---------|----------------|
| Daily sales | Weekly pattern | 7 | Same day last week |
| Daily sales | No clear pattern | 1 | Yesterday's value |
| Weekly data | Yearly pattern | 52 | Same week last year |
| Monthly data | Yearly pattern | 12 | Same month last year |
| Quarterly data | Yearly pattern | 4 | Same quarter last year |
| Hourly data | Daily pattern | 24 | Same hour yesterday |

**Example:**
```julia
# Monthly retail sales with yearly seasonality
actual = [100, 95, 110, 120, 140, 160, 155, 150, 130, 115, 105, 180,  # Year 1
          105, 98, 115, 125, 145, 165, 160, 155, 135, 120, 110, 190]  # Year 2

predicted = [102, 97, 112, 118, 142, 158, 157, 152, 132, 117, 107, 178,
             107, 100, 117, 123, 147, 163, 162, 157, 137, 122, 112, 188]

# Compare to seasonal naive (same month last year)
mase(actual, predicted, m=12)  # Yearly seasonality

# Compare to simple naive (previous month)
mase(actual, predicted, m=1)   # Usually higher - seasonal naive is a tougher benchmark
```

#### MASE Interpretation Guide

| MASE Value | Interpretation | Action |
|------------|----------------|--------|
| < 0.5 | Excellent | Model is production-ready |
| 0.5 - 0.8 | Good | Model adds significant value |
| 0.8 - 1.0 | Acceptable | Model slightly beats naive |
| 1.0 | Break-even | Model equals naive benchmark |
| 1.0 - 1.5 | Poor | Model worse than naive |
| > 1.5 | Very Poor | Investigate model issues |

### MSSE and RMSSE

```julia
msse(actual, predicted; m=1)
rmsse(actual, predicted; m=1)
```

Squared scaled error metrics. See [API Reference](@ref) for full documentation.

#### When to Use RMSSE vs MASE

- **RMSSE**: Penalizes large errors more heavily (like RMSE vs MAE)
- **MASE**: More robust to outliers
- **M5 competition** used RMSSE as the primary metric

```julia
actual = [100, 110, 105, 200, 120]  # Note: 200 is an outlier
predicted = [102, 108, 107, 150, 118]

mase(actual, predicted)   # Less affected by the large error at position 4
rmsse(actual, predicted)  # More affected by the large error
```

---

## Bias Detection Metrics

Systematic bias is a common problem in forecasting. A model might have good overall accuracy but consistently over- or under-predict.

### Tracking Signal

```julia
tracking_signal(actual, predicted)
```

Monitor forecast bias over time. See [API Reference](@ref) for full documentation.

#### Real-Time Bias Monitoring

The tracking signal is designed for **continuous monitoring** of forecast performance:

```julia
# Monitor forecast bias over time
function monitor_forecast(actual_stream, predicted_stream)
    for t in eachindex(actual_stream)
        actual_so_far = actual_stream[1:t]
        predicted_so_far = predicted_stream[1:t]

        ts = tracking_signal(actual_so_far, predicted_so_far)

        if abs(ts) > 4
            println("⚠️  Period $t: Tracking signal = $(round(ts, digits=2))")
            if ts > 0
                println("   Model is under-forecasting. Consider adjusting upward.")
            else
                println("   Model is over-forecasting. Consider adjusting downward.")
            end
        end
    end
end
```

#### Control Chart Interpretation

| Tracking Signal | Status | Action |
|-----------------|--------|--------|
| -4 to +4 | In control | Continue monitoring |
| ±4 to ±6 | Warning | Investigate recent forecasts |
| Beyond ±6 | Out of control | Recalibrate model immediately |

### Forecast Bias

```julia
forecast_bias(actual, predicted)
```

Compute the average forecast error. See [API Reference](@ref) for full documentation.

#### Bias vs Tracking Signal

| Metric | Use Case | Output |
|--------|----------|--------|
| `forecast_bias` | One-time evaluation | Average error (in original units) |
| `tracking_signal` | Continuous monitoring | Normalized ratio (unitless) |

```julia
actual = [100, 110, 105, 115, 120]
predicted = [95, 105, 100, 110, 115]  # Consistently under-predicting by ~5

forecast_bias(actual, predicted)    # Returns 5.0 (average under-prediction)
tracking_signal(actual, predicted)  # Returns ~5.0 (normalized, indicates bias)
```

---

## Benchmark Comparison Metrics

### Theil's U Statistics

```julia
theil_u1(actual, predicted)
theil_u2(actual, predicted; m=1)
```

Benchmark comparison metrics. See [API Reference](@ref) for full documentation.

#### Understanding Theil's U1 vs U2

| Statistic | Range | Interpretation |
|-----------|-------|----------------|
| **U1** | [0, 1] | 0 = perfect, 1 = worst possible |
| **U2** | [0, ∞) | < 1 = better than naive, > 1 = worse than naive |

**U2 is more commonly used** because it directly answers: "Is my model better than just using the last value?"

```julia
actual = [100, 110, 105, 115, 120, 125]
predicted = [98, 108, 107, 113, 118, 123]

# Is this forecast better than naive?
u2 = theil_u2(actual, predicted)
println("Theil U2: $u2")
println(u2 < 1 ? "Model beats naive forecast" : "Naive forecast is better")
```

---

## Percentage-Based Metrics

### WAPE (Weighted Absolute Percentage Error)

```julia
wape(actual, predicted)
```

Weighted percentage error metric. See [API Reference](@ref) for full documentation.

#### WAPE vs MAPE

| Metric | Formula | Handles Zeros? | Weighting |
|--------|---------|----------------|-----------|
| MAPE | mean(\|error\| / \|actual\|) | No (undefined) | Equal weight |
| WAPE | sum(\|error\|) / sum(\|actual\|) | Yes | Weighted by actual |

**WAPE is preferred for:**
- Intermittent demand (many zeros)
- Aggregated reporting (total error as % of total actual)
- Supply chain metrics

```julia
# Intermittent demand with zeros
actual = [0, 10, 0, 0, 20, 5, 0, 15]
predicted = [1, 8, 2, 0, 18, 6, 1, 14]

# MAPE would be undefined due to zeros
# mape(actual, predicted)  # Don't use!

# WAPE works fine
wape(actual, predicted)  # Returns meaningful percentage
```

---

## Directional Accuracy

```julia
directional_accuracy(actual, predicted)
```

Measures how often the model predicts the correct direction of change. See [API Reference](@ref) for full documentation.

#### When Direction Matters More Than Magnitude

In many applications, predicting the **direction** of change is more valuable than predicting the exact value:

- **Trading**: Buy/sell signals depend on up/down prediction
- **Inventory**: Increase/decrease stock based on demand direction
- **Capacity planning**: Scale up/down based on trend direction

```julia
# Stock price forecasting
actual_prices = [100.0, 102.0, 101.0, 103.0, 102.5, 104.0, 103.0, 105.0]
predicted_prices = [99.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5]

# MAE might look good...
mae(actual_prices, predicted_prices)  # ~1.0

# But what about direction?
da = directional_accuracy(actual_prices, predicted_prices)
println("Directional Accuracy: $(round(da * 100, digits=1))%")
println(da > 0.5 ? "Model has predictive value for direction" : "Model fails to predict direction")
```

#### Directional Accuracy Benchmarks

| DA Value | Interpretation |
|----------|----------------|
| > 60% | Good directional forecasting |
| 50-60% | Marginal predictive value |
| ~50% | No better than coin flip |
| < 50% | Worse than random (consider inverting) |

---

## Prediction Interval Metrics

Modern forecasting produces **probabilistic forecasts** with prediction intervals, not just point predictions. These metrics evaluate interval quality.

### Coverage Probability

```julia
coverage_probability(actual, lower, upper)
```

Compute the proportion of actual values within prediction intervals. See [API Reference](@ref) for full documentation.

#### Calibration Assessment

A well-calibrated 95% prediction interval should contain the actual value ~95% of the time:

```julia
actual = [100, 110, 105, 115, 120, 125, 130, 128, 135, 140]

# Your model's 95% prediction intervals
lower_95 = [92, 102, 97, 107, 112, 117, 122, 120, 127, 132]
upper_95 = [108, 118, 113, 123, 128, 133, 138, 136, 143, 148]

coverage = coverage_probability(actual, lower_95, upper_95)
println("95% Interval Coverage: $(round(coverage * 100, digits=1))%")

if coverage < 0.90
    println("⚠️  Under-coverage: Intervals too narrow")
elseif coverage > 0.99
    println("⚠️  Over-coverage: Intervals too wide (but valid)")
else
    println("✓ Well-calibrated")
end
```

### Winkler Score

```julia
winkler_score(actual, lower, upper; alpha=0.05)
```

Evaluate prediction intervals for sharpness and calibration. See [API Reference](@ref) for full documentation.

#### Why Winkler Score?

Coverage alone doesn't tell the whole story. Two models can have the same coverage but different interval widths:

- **Model A**: 95% coverage with wide intervals (less useful)
- **Model B**: 95% coverage with narrow intervals (more useful)

Winkler score rewards **sharp** (narrow) intervals while penalizing **miscoverage**:

```julia
actual = [100, 110, 105]

# Model A: Wide intervals (always covers, but not useful)
lower_a = [80, 90, 85]
upper_a = [120, 130, 125]

# Model B: Narrow intervals (same coverage, more useful)
lower_b = [95, 105, 100]
upper_b = [105, 115, 110]

# Both have 100% coverage
coverage_probability(actual, lower_a, upper_a)  # 1.0
coverage_probability(actual, lower_b, upper_b)  # 1.0

# But Winkler score prefers narrower intervals
winkler_score(actual, lower_a, upper_a, alpha=0.05)  # Higher (worse)
winkler_score(actual, lower_b, upper_b, alpha=0.05)  # Lower (better)
```

### Pinball Loss (Quantile Loss)

```julia
pinball_loss_series(actual, predicted; quantile=0.5)
```

Evaluate quantile forecasts. See [API Reference](@ref) for full documentation.

#### Evaluating Quantile Forecasts

For probabilistic forecasts that output multiple quantiles:

```julia
actual = [100, 110, 105, 115, 120]

# Forecasts at different quantiles
forecast_p10 = [85, 95, 90, 100, 105]    # 10th percentile
forecast_p50 = [98, 108, 103, 113, 118]  # Median
forecast_p90 = [112, 122, 117, 127, 132] # 90th percentile

# Evaluate each quantile
for (q, forecast) in [(0.1, forecast_p10), (0.5, forecast_p50), (0.9, forecast_p90)]
    loss = pinball_loss_series(actual, forecast, quantile=q)
    println("P$(Int(q*100)) Pinball Loss: $(round(loss, digits=3))")
end
```

---

## Autocorrelation Preservation

```julia
autocorrelation_error(actual, predicted; max_lag=10)
```

Measure how well the forecast preserves the temporal structure. See [API Reference](@ref) for full documentation.

#### When Temporal Structure Matters

Some applications require forecasts that preserve the **statistical properties** of the original series:

- Simulation and scenario generation
- Synthetic data for testing
- Risk modeling (preserving volatility clustering)

```julia
# Original series has strong autocorrelation
actual = cumsum(randn(100))  # Random walk

# Good forecast preserves autocorrelation structure
good_forecast = actual .+ randn(100) * 0.5  # Small noise

# Bad forecast destroys autocorrelation
bad_forecast = shuffle(actual)  # Shuffled - no temporal structure

autocorrelation_error(actual, good_forecast, max_lag=10)  # Low
autocorrelation_error(actual, bad_forecast, max_lag=10)   # High
```

---

## Complete Evaluation Framework

### Recommended Evaluation Protocol

For comprehensive time series model evaluation, use this framework:

```julia
using UnifiedMetrics

function evaluate_forecast(actual, predicted, lower, upper; m=1, alpha=0.05)
    println("=" ^ 60)
    println("TIME SERIES FORECAST EVALUATION REPORT")
    println("=" ^ 60)

    # 1. Point Forecast Accuracy
    println("\n📊 POINT FORECAST ACCURACY")
    println("-" ^ 40)
    println("MAE:  $(round(mae(actual, predicted), digits=3))")
    println("RMSE: $(round(rmse(actual, predicted), digits=3))")
    println("MAPE: $(round(mape(actual, predicted) * 100, digits=2))%")
    println("WAPE: $(round(wape(actual, predicted) * 100, digits=2))%")

    # 2. Scale-Independent Metrics
    println("\n📏 SCALE-INDEPENDENT METRICS")
    println("-" ^ 40)
    m_val = mase(actual, predicted, m=m)
    println("MASE (m=$m):  $(round(m_val, digits=3))")
    println("RMSSE (m=$m): $(round(rmsse(actual, predicted, m=m), digits=3))")
    println("Theil U2:     $(round(theil_u2(actual, predicted, m=m), digits=3))")

    if m_val < 1
        println("✓ Model outperforms naive forecast")
    else
        println("⚠ Model underperforms naive forecast")
    end

    # 3. Bias Analysis
    println("\n🎯 BIAS ANALYSIS")
    println("-" ^ 40)
    fb = forecast_bias(actual, predicted)
    ts = tracking_signal(actual, predicted)
    println("Forecast Bias:    $(round(fb, digits=3))")
    println("Tracking Signal:  $(round(ts, digits=3))")

    if abs(ts) > 4
        println("⚠ Systematic bias detected!")
    else
        println("✓ No significant bias")
    end

    # 4. Directional Accuracy
    println("\n↗️ DIRECTIONAL ACCURACY")
    println("-" ^ 40)
    da = directional_accuracy(actual, predicted)
    println("Direction Accuracy: $(round(da * 100, digits=1))%")

    # 5. Prediction Intervals (if provided)
    if !isnothing(lower) && !isnothing(upper)
        println("\n📈 PREDICTION INTERVAL QUALITY")
        println("-" ^ 40)
        cov = coverage_probability(actual, lower, upper)
        wink = winkler_score(actual, lower, upper, alpha=alpha)
        expected_cov = 1 - alpha

        println("Expected Coverage: $(round(expected_cov * 100, digits=1))%")
        println("Actual Coverage:   $(round(cov * 100, digits=1))%")
        println("Winkler Score:     $(round(wink, digits=3))")

        if abs(cov - expected_cov) < 0.05
            println("✓ Intervals well-calibrated")
        elseif cov < expected_cov
            println("⚠ Under-coverage: intervals too narrow")
        else
            println("⚠ Over-coverage: intervals too wide")
        end
    end

    println("\n" * "=" ^ 60)
end

# Example usage
actual = [100.0, 110.0, 105.0, 115.0, 120.0, 125.0, 130.0, 128.0]
predicted = [98.0, 108.0, 107.0, 113.0, 118.0, 123.0, 128.0, 126.0]
lower = [90.0, 100.0, 99.0, 105.0, 110.0, 115.0, 120.0, 118.0]
upper = [106.0, 116.0, 115.0, 121.0, 126.0, 131.0, 136.0, 134.0]

evaluate_forecast(actual, predicted, lower, upper, m=1, alpha=0.05)
```

### Multi-Series Comparison

When comparing forecasts across multiple time series:

```julia
function compare_models_across_series(series_data, models)
    results = Dict{String, Vector{Float64}}()

    for model_name in keys(models)
        results[model_name] = Float64[]
    end

    for (actual, model_forecasts) in series_data
        for (model_name, predicted) in model_forecasts
            push!(results[model_name], mase(actual, predicted))
        end
    end

    println("Model Comparison (MASE)")
    println("-" ^ 40)
    for (model_name, mase_values) in results
        avg_mase = mean(mase_values)
        println("$model_name: $(round(avg_mase, digits=3)) (avg across $(length(mase_values)) series)")
    end
end
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Using MAPE with Zeros

**Problem**: MAPE is undefined when actual values are zero (common in intermittent demand).

**Solution**: Use WAPE or MASE instead.

```julia
actual = [0, 10, 0, 5, 0, 20]  # Intermittent demand
predicted = [1, 9, 1, 4, 1, 19]

# Don't do this:
# mape(actual, predicted)  # Returns Inf or NaN

# Do this instead:
wape(actual, predicted)
mase(actual, predicted)
```

### Pitfall 2: Ignoring Seasonality in MASE

**Problem**: Using m=1 when data has seasonality makes the benchmark too easy to beat.

**Solution**: Set m to match your data's seasonal period.

```julia
# Monthly data with yearly seasonality
actual = repeat([100, 80, 90, 110, 130, 150, 160, 155, 140, 120, 100, 180], 2)
predicted = actual .+ randn(24) * 5

# This makes naive look bad (comparing to previous month)
mase(actual, predicted, m=1)  # Artificially low

# This is the correct comparison (same month last year)
mase(actual, predicted, m=12)  # More realistic assessment
```

### Pitfall 3: Only Evaluating Point Forecasts

**Problem**: Ignoring prediction intervals misses important information about forecast uncertainty.

**Solution**: Always evaluate both point accuracy and interval quality.

```julia
# A model with great point accuracy but terrible intervals
actual = [100, 110, 105, 115, 120]
predicted = [100, 110, 105, 115, 120]  # Perfect point forecast!
lower = [99, 109, 104, 114, 119]       # Intervals way too narrow
upper = [101, 111, 106, 116, 121]

mae(actual, predicted)  # 0.0 - Perfect!
coverage_probability(actual, lower, upper)  # May be < 0.95 - Problem!
```

### Pitfall 4: Not Monitoring for Bias

**Problem**: A model may have good overall accuracy but develop systematic bias over time.

**Solution**: Use tracking signal for ongoing monitoring.

```julia
# Model starts good but develops bias
actual = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
predicted = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]  # Increasing under-forecast

# Overall MAE looks okay
mae(actual, predicted)  # ~4.5

# But tracking signal reveals the problem
tracking_signal(actual, predicted)  # High positive value - systematic under-forecasting
```

---

## References and Further Reading

### Academic References

- Hyndman, R.J., & Koehler, A.B. (2006). "Another look at measures of forecast accuracy." *International Journal of Forecasting*, 22(4), 679-688. (Introduced MASE)

- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). "The M4 Competition: 100,000 time series and 61 forecasting methods." *International Journal of Forecasting*, 36(1), 54-74.

- Gneiting, T., & Raftery, A.E. (2007). "Strictly proper scoring rules, prediction, and estimation." *Journal of the American Statistical Association*, 102(477), 359-378. (Theory behind proper scoring rules)

### Metric Selection Guidelines

- **M-competitions**: Use MASE, sMAPE (symmetric MAPE), and RMSSE
- **Supply chain**: Use WAPE, MASE, and tracking signal
- **Finance**: Use directional accuracy, Theil's U2
- **Probabilistic forecasting**: Use coverage probability, Winkler score, CRPS
