"""
    mase(actual, predicted; m=1)

Compute the Mean Absolute Scaled Error for time series data.

MASE compares the prediction error to the error of a naive forecast. The naive forecast
predicts the value from `m` periods ago (seasonal naive for m > 1).

MASE < 1 indicates the model outperforms the naive forecast.
MASE = 1 indicates performance equal to naive forecast.
MASE > 1 indicates the model underperforms the naive forecast.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series (ordered by time)
- `predicted::AbstractVector{<:Real}`: Predicted time series
- `m::Integer`: Seasonal period / frequency (default: 1)
  - `m=1`: Non-seasonal naive forecast (random walk)
  - `m=4`: Quarterly seasonality (compare with same quarter last year)
  - `m=7`: Weekly seasonality for daily data
  - `m=12`: Monthly seasonality (compare with same month last year)
  - `m=52`: Weekly seasonality for weekly data

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
mase(actual, predicted)  # Non-seasonal (m=1)
mase(actual, predicted, m=2)  # With seasonal period 2
```
"""
function mase(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; m::Integer=1)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    @assert m >= 1 "Seasonal period m must be at least 1"
    @assert length(actual) > m "Length of actual must be greater than seasonal period m"

    n = length(actual)
    naive_start = m + 1
    naive_end = n - m

    sum_errors = sum(ae(actual, predicted))
    naive_errors = sum(ae(actual[naive_start:n], actual[1:naive_end]))

    return sum_errors / (n * naive_errors / naive_end)
end

"""
    msse(actual, predicted; m=1)

Compute the Mean Squared Scaled Error for time series data.

Similar to MASE but uses squared errors, making it more sensitive to large errors.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series (ordered by time)
- `predicted::AbstractVector{<:Real}`: Predicted time series
- `m::Integer`: Seasonal period / frequency (default: 1)

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
msse(actual, predicted)
```
"""
function msse(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; m::Integer=1)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    @assert m >= 1 "Seasonal period m must be at least 1"
    @assert length(actual) > m "Length of actual must be greater than seasonal period m"

    n = length(actual)
    naive_start = m + 1
    naive_end = n - m

    sum_sq_errors = sum(se(actual, predicted))
    naive_sq_errors = sum(se(actual[naive_start:n], actual[1:naive_end]))

    return sum_sq_errors / (n * naive_sq_errors / naive_end)
end

"""
    rmsse(actual, predicted; m=1)

Compute the Root Mean Squared Scaled Error for time series data.

Square root of MSSE, on the same scale as the original data.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series (ordered by time)
- `predicted::AbstractVector{<:Real}`: Predicted time series
- `m::Integer`: Seasonal period / frequency (default: 1)

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
rmsse(actual, predicted)
```
"""
function rmsse(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; m::Integer=1)
    return sqrt(msse(actual, predicted, m=m))
end

"""
    tracking_signal(actual, predicted)

Compute the tracking signal for forecast monitoring.

Tracking signal = Cumulative Forecast Error / MAD

Used to detect forecast bias. Values outside [-4, 4] typically indicate bias.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series
- `predicted::AbstractVector{<:Real}`: Predicted time series

# Examples
```julia
actual = [100, 110, 105, 115, 120]
predicted = [98, 108, 110, 112, 125]
tracking_signal(actual, predicted)
```
"""
function tracking_signal(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    errors = actual .- predicted
    cfe = sum(errors)  # Cumulative Forecast Error
    mad = mean(abs.(errors))  # Mean Absolute Deviation

    if mad == 0
        return cfe == 0 ? 0.0 : Inf
    end
    return cfe / mad
end

"""
    forecast_bias(actual, predicted)

Compute the forecast bias (cumulative forecast error normalized by number of periods).

Positive bias indicates systematic under-forecasting.
Negative bias indicates systematic over-forecasting.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series
- `predicted::AbstractVector{<:Real}`: Predicted time series

# Examples
```julia
actual = [100, 110, 105, 115, 120]
predicted = [98, 108, 110, 112, 125]
forecast_bias(actual, predicted)
```
"""
function forecast_bias(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return mean(actual .- predicted)
end

"""
    theil_u1(actual, predicted)

Compute Theil's U1 statistic (inequality coefficient).

U1 ranges from 0 to 1:
- 0: Perfect forecast
- 1: Worst possible forecast

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series
- `predicted::AbstractVector{<:Real}`: Predicted time series

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
theil_u1(actual, predicted)
```
"""
function theil_u1(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    numerator = sqrt(mean((actual .- predicted).^2))
    denominator = sqrt(mean(actual.^2)) + sqrt(mean(predicted.^2))

    return denominator == 0 ? NaN : numerator / denominator
end

"""
    theil_u2(actual, predicted; m=1)

Compute Theil's U2 statistic (compares to naive forecast).

U2 < 1: Model outperforms naive forecast
U2 = 1: Model equals naive forecast
U2 > 1: Model underperforms naive forecast

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series
- `predicted::AbstractVector{<:Real}`: Predicted time series
- `m::Integer`: Seasonal period for naive forecast (default: 1)

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
theil_u2(actual, predicted)
```
"""
function theil_u2(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; m::Integer=1)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    @assert length(actual) > m "Length of actual must be greater than m"

    # Forecast errors (for periods where we have both forecast and naive)
    n = length(actual)
    actual_subset = actual[(m+1):n]
    predicted_subset = predicted[(m+1):n]
    naive_forecast = actual[1:(n-m)]

    forecast_mse = mean((actual_subset .- predicted_subset).^2)
    naive_mse = mean((actual_subset .- naive_forecast).^2)

    return naive_mse == 0 ? Inf : sqrt(forecast_mse / naive_mse)
end

"""
    wape(actual, predicted)

Compute the Weighted Absolute Percentage Error for time series.

WAPE = Σ|actual - predicted| / Σ|actual|

Unlike MAPE, WAPE is well-defined when actual values are zero and gives more
weight to larger values.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series
- `predicted::AbstractVector{<:Real}`: Predicted time series

# Examples
```julia
actual = [100, 200, 150, 300, 250]
predicted = [110, 190, 160, 290, 260]
wape(actual, predicted)
```
"""
function wape(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    total_actual = sum(abs.(actual))
    return total_actual == 0 ? Inf : sum(abs.(actual .- predicted)) / total_actual
end

"""
    directional_accuracy(actual, predicted)

Compute the directional accuracy (hit rate for direction of change).

Measures how often the forecast correctly predicts whether the value goes up or down.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series
- `predicted::AbstractVector{<:Real}`: Predicted time series

# Examples
```julia
actual = [100, 110, 105, 115, 120]  # Changes: +10, -5, +10, +5
predicted = [100, 108, 106, 112, 118]  # Changes: +8, -2, +6, +6
directional_accuracy(actual, predicted)  # All directions match = 1.0
```
"""
function directional_accuracy(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    @assert length(actual) >= 2 "Need at least 2 observations"

    actual_changes = diff(actual)
    predicted_changes = diff(predicted)

    # Count correct direction predictions (both positive, both negative, or both zero)
    correct = sum(sign.(actual_changes) .== sign.(predicted_changes))

    return correct / length(actual_changes)
end

"""
    coverage_probability(actual, lower, upper)

Compute the coverage probability of prediction intervals.

Measures what fraction of actual values fall within the prediction intervals.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series
- `lower::AbstractVector{<:Real}`: Lower bounds of prediction intervals
- `upper::AbstractVector{<:Real}`: Upper bounds of prediction intervals

# Examples
```julia
actual = [100, 110, 105, 115, 120]
lower = [95, 105, 100, 108, 112]  # 95% lower bounds
upper = [105, 115, 112, 122, 128]  # 95% upper bounds
coverage_probability(actual, lower, upper)  # Should be ≈ 0.95 if well-calibrated
```
"""
function coverage_probability(actual::AbstractVector{<:Real}, lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real})
    @assert length(actual) == length(lower) == length(upper) "All vectors must have the same length"
    return mean((actual .>= lower) .& (actual .<= upper))
end

"""
    pinball_loss_series(actual, predicted; quantile=0.5)

Compute the pinball (quantile) loss for time series probabilistic forecasts.

Same as `quantile_loss` but named to be consistent with time series literature.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series
- `predicted::AbstractVector{<:Real}`: Quantile forecast
- `quantile::Real`: Target quantile in (0, 1) (default: 0.5 = median)

# Examples
```julia
actual = [100, 110, 105, 115, 120]
predicted_median = [98, 108, 110, 112, 118]  # Median forecasts
pinball_loss_series(actual, predicted_median, quantile=0.5)
```
"""
function pinball_loss_series(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; quantile::Real=0.5)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    @assert 0 < quantile < 1 "quantile must be in (0, 1)"

    errors = actual .- predicted
    loss = [e >= 0 ? quantile * e : (quantile - 1) * e for e in errors]
    return mean(loss)
end

"""
    winkler_score(actual, lower, upper; alpha=0.05)

Compute the Winkler score for prediction interval evaluation.

The Winkler score rewards narrow intervals and penalizes intervals that don't
contain the actual value.

Lower scores are better.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series
- `lower::AbstractVector{<:Real}`: Lower bounds of prediction intervals
- `upper::AbstractVector{<:Real}`: Upper bounds of prediction intervals
- `alpha::Real`: Significance level (default: 0.05 for 95% intervals)

# Examples
```julia
actual = [100, 110, 105, 115, 120]
lower = [95, 105, 100, 108, 112]
upper = [105, 115, 112, 122, 128]
winkler_score(actual, lower, upper, alpha=0.05)
```
"""
function winkler_score(actual::AbstractVector{<:Real}, lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real}; alpha::Real=0.05)
    @assert length(actual) == length(lower) == length(upper) "All vectors must have the same length"
    @assert 0 < alpha < 1 "alpha must be in (0, 1)"

    scores = Float64[]
    for (y, l, u) in zip(actual, lower, upper)
        width = u - l
        if y < l
            score = width + (2 / alpha) * (l - y)
        elseif y > u
            score = width + (2 / alpha) * (y - u)
        else
            score = width
        end
        push!(scores, score)
    end

    return mean(scores)
end

"""
    autocorrelation_error(actual, predicted; max_lag=10)

Compute the error in autocorrelation structure.

Measures how well the forecast preserves the autocorrelation structure of the
actual series. Lower values indicate better preservation of temporal patterns.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth time series
- `predicted::AbstractVector{<:Real}`: Predicted time series
- `max_lag::Integer`: Maximum lag to consider (default: 10)

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6, 6.2, 7.1, 8.0, 9.2]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2, 6.0, 7.0, 8.1, 9.0]
autocorrelation_error(actual, predicted)
```
"""
function autocorrelation_error(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; max_lag::Integer=10)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    @assert max_lag >= 1 "max_lag must be at least 1"

    n = length(actual)
    effective_max_lag = min(max_lag, n - 2)

    if effective_max_lag < 1
        return NaN  # Series too short for autocorrelation analysis
    end

    function acf(x, lag)
        x_centered = x .- mean(x)
        n = length(x)
        denom = sum(x_centered.^2)
        if denom == 0
            return 0.0  # Constant series has no autocorrelation
        end
        return sum(x_centered[1:n-lag] .* x_centered[lag+1:n]) / denom
    end

    errors = Float64[]
    for lag in 1:effective_max_lag
        acf_actual = acf(actual, lag)
        acf_predicted = acf(predicted, lag)
        push!(errors, (acf_actual - acf_predicted)^2)
    end

    return sqrt(mean(errors))
end
