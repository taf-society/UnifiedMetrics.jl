"""
    bias(actual, predicted)

Compute the average amount by which `actual` is greater than `predicted`.

If a model is unbiased, `bias(actual, predicted)` should be close to zero.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
bias(actual, predicted)
```
"""
function bias(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return mean(actual .- predicted)
end

"""
    percent_bias(actual, predicted)

Compute the average amount that `actual` is greater than `predicted` as a percentage
of the absolute value of `actual`.

Returns `-Inf`, `Inf`, or `NaN` if any elements of `actual` are `0`.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
percent_bias(actual, predicted)
```
"""
function percent_bias(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return mean((actual .- predicted) ./ abs.(actual))
end

"""
    se(actual, predicted)

Compute the elementwise squared error between two numeric vectors.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
se(actual, predicted)
```
"""
function se(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return (actual .- predicted) .^ 2
end

"""
    sse(actual, predicted)

Compute the sum of squared errors between two numeric vectors.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
sse(actual, predicted)
```
"""
function sse(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return sum(se(actual, predicted))
end

"""
    mse(actual, predicted)

Compute the mean squared error between two numeric vectors.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
mse(actual, predicted)
```
"""
function mse(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return mean(se(actual, predicted))
end

"""
    rmse(actual, predicted)

Compute the root mean squared error between two numeric vectors.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
rmse(actual, predicted)
```
"""
function rmse(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return sqrt(mse(actual, predicted))
end

"""
    ae(actual, predicted)

Compute the elementwise absolute error between two numeric vectors.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
ae(actual, predicted)
```
"""
function ae(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return abs.(actual .- predicted)
end

"""
    mae(actual, predicted)

Compute the mean absolute error between two numeric vectors.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
mae(actual, predicted)
```
"""
function mae(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return mean(ae(actual, predicted))
end

"""
    mdae(actual, predicted)

Compute the median absolute error between two numeric vectors.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
mdae(actual, predicted)
```
"""
function mdae(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return median(ae(actual, predicted))
end

"""
    ape(actual, predicted)

Compute the elementwise absolute percent error between two numeric vectors.

Returns `-Inf`, `Inf`, or `NaN` if `actual` contains zeros.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
ape(actual, predicted)
```
"""
function ape(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return ae(actual, predicted) ./ abs.(actual)
end

"""
    mape(actual, predicted)

Compute the mean absolute percent error between two numeric vectors.

Returns `-Inf`, `Inf`, or `NaN` if `actual` contains zeros. Due to instability at
or near zero, `smape` or `mase` are often used as alternatives.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
mape(actual, predicted)
```
"""
function mape(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return mean(ape(actual, predicted))
end

"""
    smape(actual, predicted)

Compute the symmetric mean absolute percentage error between two numeric vectors.

Defined as `2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted)))`.
Returns `NaN` only if both `actual` and `predicted` are zero at the same position.
Has an upper bound of 2.

`smape` is symmetric: `smape(x, y) == smape(y, x)`.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
smape(actual, predicted)
```
"""
function smape(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return 2 * mean(ae(actual, predicted) ./ (abs.(actual) .+ abs.(predicted)))
end

"""
    sle(actual, predicted)

Compute the elementwise squared log error between two numeric vectors.

Adds one to both `actual` and `predicted` before taking the natural logarithm
to avoid taking the log of zero. Not appropriate for negative values.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth non-negative vector
- `predicted::AbstractVector{<:Real}`: Predicted non-negative vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
sle(actual, predicted)
```
"""
function sle(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return (log.(1 .+ actual) .- log.(1 .+ predicted)) .^ 2
end

"""
    msle(actual, predicted)

Compute the mean squared log error between two numeric vectors.

Adds one to both `actual` and `predicted` before taking the natural logarithm
to avoid taking the log of zero. Not appropriate for negative values.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth non-negative vector
- `predicted::AbstractVector{<:Real}`: Predicted non-negative vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
msle(actual, predicted)
```
"""
function msle(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return mean(sle(actual, predicted))
end

"""
    rmsle(actual, predicted)

Compute the root mean squared log error between two numeric vectors.

Adds one to both `actual` and `predicted` before taking the natural logarithm
to avoid taking the log of zero. Not appropriate for negative values.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth non-negative vector
- `predicted::AbstractVector{<:Real}`: Predicted non-negative vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
rmsle(actual, predicted)
```
"""
function rmsle(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return sqrt(msle(actual, predicted))
end

"""
    rse(actual, predicted)

Compute the relative squared error between two numeric vectors.

Divides `sse(actual, predicted)` by `sse(actual, mean(actual))`, providing the
squared error relative to a naive model that predicts the mean for every data point.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
rse(actual, predicted)
```
"""
function rse(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    actual_mean = fill(mean(actual), length(actual))
    return sse(actual, predicted) / sse(actual, actual_mean)
end

"""
    rrse(actual, predicted)

Compute the root relative squared error between two numeric vectors.

Takes the square root of `rse(actual, predicted)`.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
rrse(actual, predicted)
```
"""
function rrse(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return sqrt(rse(actual, predicted))
end

"""
    rae(actual, predicted)

Compute the relative absolute error between two numeric vectors.

Divides `sum(ae(actual, predicted))` by `sum(ae(actual, mean(actual)))`, providing
the absolute error relative to a naive model that predicts the mean for every data point.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
rae(actual, predicted)
```
"""
function rae(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    actual_mean = fill(mean(actual), length(actual))
    return sum(ae(actual, predicted)) / sum(ae(actual, actual_mean))
end

"""
    explained_variation(actual, predicted)

Compute the explained variation (coefficient of determination, R²) between two numeric vectors.

Subtracts `rse(actual, predicted)` from 1. Can return negative values if predictions
are worse than predicting the mean.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
explained_variation(actual, predicted)
```
"""
function explained_variation(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return 1 - rse(actual, predicted)
end

"""
    max_error(actual, predicted)

Compute the maximum absolute error between two numeric vectors.

Useful for understanding the worst-case prediction error.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
max_error(actual, predicted)
```
"""
function max_error(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return maximum(ae(actual, predicted))
end

"""
    max_ae(actual, predicted)

Alias for `max_error`. Compute the maximum absolute error.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector
"""
max_ae(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}) = max_error(actual, predicted)

"""
    huber_loss(actual, predicted; delta=1.0)

Compute the Huber loss, which is quadratic for small errors and linear for large errors.

Huber loss is less sensitive to outliers than MSE. For errors smaller than `delta`,
it behaves like MSE; for larger errors, it behaves like MAE.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector
- `delta::Real`: Threshold where loss transitions from quadratic to linear (default: 1.0)

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
huber_loss(actual, predicted)
huber_loss(actual, predicted, delta=0.5)
```
"""
function huber_loss(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; delta::Real=1.0)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    @assert delta > 0 "delta must be positive"

    errors = actual .- predicted
    abs_errors = abs.(errors)

    loss = similar(errors, Float64)
    for i in eachindex(errors)
        if abs_errors[i] <= delta
            loss[i] = 0.5 * errors[i]^2
        else
            loss[i] = delta * (abs_errors[i] - 0.5 * delta)
        end
    end

    return mean(loss)
end

"""
    log_cosh_loss(actual, predicted)

Compute the log-cosh loss, a smooth approximation of MAE.

Log-cosh is approximately equal to `(x^2)/2` for small x and `abs(x) - log(2)` for large x.
It has the advantage of being twice differentiable everywhere.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
log_cosh_loss(actual, predicted)
```
"""
function log_cosh_loss(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    errors = actual .- predicted
    return mean(log.(cosh.(errors)))
end

"""
    quantile_loss(actual, predicted; quantile=0.5)

Compute the quantile (pinball) loss for quantile regression.

The quantile loss asymmetrically penalizes over-prediction and under-prediction.
At quantile=0.5, this is equivalent to MAE. For quantile<0.5, under-prediction is
penalized more; for quantile>0.5, over-prediction is penalized more.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector
- `quantile::Real`: Target quantile in (0, 1) (default: 0.5)

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
quantile_loss(actual, predicted, quantile=0.5)  # Equivalent to MAE
quantile_loss(actual, predicted, quantile=0.9)  # Penalize under-prediction more
```
"""
function quantile_loss(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; quantile::Real=0.5)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    @assert 0 < quantile < 1 "quantile must be in (0, 1)"

    errors = actual .- predicted
    loss = [e >= 0 ? quantile * e : (quantile - 1) * e for e in errors]
    return mean(loss)
end

"""
    pinball_loss(actual, predicted; quantile=0.5)

Alias for `quantile_loss`. Compute the pinball loss for quantile regression.
"""
pinball_loss(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; quantile::Real=0.5) =
    quantile_loss(actual, predicted, quantile=quantile)

"""
    nrmse(actual, predicted; normalization=:range)

Compute the Normalized Root Mean Squared Error.

Normalizes RMSE to make it comparable across different scales.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector
- `normalization::Symbol`: Normalization method (default: `:range`)
  - `:range` - Normalize by range (max - min) of actual values
  - `:mean` - Normalize by mean of actual values (coefficient of variation of RMSE)
  - `:std` - Normalize by standard deviation of actual values
  - `:iqr` - Normalize by interquartile range of actual values

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
nrmse(actual, predicted)  # Normalized by range
nrmse(actual, predicted, normalization=:mean)  # CV(RMSE)
```
"""
function nrmse(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; normalization::Symbol=:range)
    rmse_val = rmse(actual, predicted)

    if normalization == :range
        denom = maximum(actual) - minimum(actual)
    elseif normalization == :mean
        denom = mean(actual)
    elseif normalization == :std
        denom = std(actual)
    elseif normalization == :iqr
        sorted = sort(actual)
        n = length(sorted)
        q1 = sorted[max(1, floor(Int, n * 0.25))]
        q3 = sorted[min(n, ceil(Int, n * 0.75))]
        denom = q3 - q1
    else
        error("Unknown normalization method: $normalization. Use :range, :mean, :std, or :iqr")
    end

    return denom == 0 ? Inf : rmse_val / denom
end

"""
    adjusted_r2(actual, predicted, n_features)

Compute the Adjusted R² (coefficient of determination adjusted for number of predictors).

Adjusted R² penalizes the addition of irrelevant features to a model.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector
- `n_features::Integer`: Number of features/predictors in the model

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
adjusted_r2(actual, predicted, 2)  # Model with 2 features
```
"""
function adjusted_r2(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}, n_features::Integer)
    n = length(actual)
    @assert n > n_features + 1 "Need more samples than features + 1"

    r2 = explained_variation(actual, predicted)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
end

"""
    mpe(actual, predicted)

Compute the Mean Percentage Error (signed).

Unlike MAPE, MPE can indicate systematic bias: positive values indicate under-prediction
on average, negative values indicate over-prediction.

Returns `Inf`, `-Inf`, or `NaN` if `actual` contains zeros.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
mpe(actual, predicted)
```
"""
function mpe(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return mean((actual .- predicted) ./ actual) * 100
end

"""
    wmape(actual, predicted)

Compute the Weighted Mean Absolute Percentage Error.

WMAPE weights errors by the magnitude of actual values, making it more robust
than MAPE when actual values vary significantly in magnitude.

WMAPE = sum(|actual - predicted|) / sum(|actual|)

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth numeric vector
- `predicted::AbstractVector{<:Real}`: Predicted numeric vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
wmape(actual, predicted)
```
"""
function wmape(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return sum(ae(actual, predicted)) / sum(abs.(actual))
end

"""
    tweedie_deviance(actual, predicted; power=1.5)

Compute the Tweedie deviance for generalized linear models.

The power parameter controls the distribution assumption:
- power=0: Normal distribution (equivalent to MSE)
- power=1: Poisson distribution
- power=2: Gamma distribution
- power=3: Inverse Gaussian distribution
- 1 < power < 2: Compound Poisson-Gamma (common for insurance claims)

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth non-negative vector
- `predicted::AbstractVector{<:Real}`: Predicted non-negative vector
- `power::Real`: Tweedie power parameter (default: 1.5)

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
tweedie_deviance(actual, predicted, power=1.5)
```
"""
function tweedie_deviance(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; power::Real=1.5)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    @assert all(predicted .> 0) "Predicted values must be positive"

    if power == 0
        # Normal distribution - equivalent to MSE
        return mean((actual .- predicted).^2)
    elseif power == 1
        # Poisson distribution
        deviance = 2 .* (actual .* log.(max.(actual, 1e-10) ./ predicted) .- (actual .- predicted))
        return mean(deviance)
    elseif power == 2
        # Gamma distribution
        deviance = 2 .* (log.(predicted ./ max.(actual, 1e-10)) .+ actual ./ predicted .- 1)
        return mean(deviance)
    else
        # General Tweedie
        term1 = max.(actual, 1e-10).^(2 - power) / ((1 - power) * (2 - power))
        term2 = actual .* predicted.^(1 - power) / (1 - power)
        term3 = predicted.^(2 - power) / (2 - power)
        deviance = 2 .* (term1 .- term2 .+ term3)
        return mean(deviance)
    end
end

"""
    mean_gamma_deviance(actual, predicted)

Compute the mean Gamma deviance.

Equivalent to `tweedie_deviance` with power=2. Appropriate for positive continuous
targets with variance proportional to the square of the mean.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth positive vector
- `predicted::AbstractVector{<:Real}`: Predicted positive vector

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
mean_gamma_deviance(actual, predicted)
```
"""
mean_gamma_deviance(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}) =
    tweedie_deviance(actual, predicted, power=2)

"""
    mean_poisson_deviance(actual, predicted)

Compute the mean Poisson deviance.

Equivalent to `tweedie_deviance` with power=1. Appropriate for count data.

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth non-negative vector (counts)
- `predicted::AbstractVector{<:Real}`: Predicted positive vector

# Examples
```julia
actual = [1, 2, 3, 4, 5, 6]
predicted = [1.1, 1.9, 3.1, 3.9, 5.1, 5.9]
mean_poisson_deviance(actual, predicted)
```
"""
mean_poisson_deviance(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}) =
    tweedie_deviance(actual, predicted, power=1)

"""
    d2_tweedie_score(actual, predicted; power=1.5)

Compute the D² (deviance explained) score using Tweedie deviance.

Similar to R² but uses Tweedie deviance instead of squared error.
D² = 1 - deviance(actual, predicted) / deviance(actual, mean(actual))

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth non-negative vector
- `predicted::AbstractVector{<:Real}`: Predicted non-negative vector
- `power::Real`: Tweedie power parameter (default: 1.5)

# Examples
```julia
actual = [1.1, 1.9, 3.0, 4.4, 5.0, 5.6]
predicted = [0.9, 1.8, 2.5, 4.5, 5.0, 6.2]
d2_tweedie_score(actual, predicted, power=1.5)
```
"""
function d2_tweedie_score(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; power::Real=1.5)
    dev_pred = tweedie_deviance(actual, predicted, power=power)
    mean_actual = fill(mean(actual), length(actual))
    dev_null = tweedie_deviance(actual, mean_actual, power=power)
    return 1 - dev_pred / dev_null
end
