"""
    auc(actual, predicted)

Compute the area under the ROC curve (AUC).

Uses the Mann-Whitney U statistic for fast computation without building the ROC curve.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Predicted scores (higher = more likely positive)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [0.9, 0.8, 0.4, 0.5, 0.3, 0.2]
auc(actual, predicted)
```
"""
function auc(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    r = _ordinalrank(predicted)
    n_pos = sum(actual .== 1)
    n_neg = length(actual) - n_pos

    return (sum(r[actual .== 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
end

"""
    ll(actual, predicted)

Compute the elementwise log loss (cross-entropy loss).

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Predicted probabilities for positive class

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [0.9, 0.8, 0.4, 0.5, 0.3, 0.2]
ll(actual, predicted)
```
"""
function ll(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    score = -(actual .* log.(predicted) .+ (1 .- actual) .* log.(1 .- predicted))

    # Handle perfect predictions
    for i in eachindex(score)
        if actual[i] == predicted[i]
            score[i] = 0.0
        elseif isnan(score[i])
            score[i] = Inf
        end
    end

    return score
end

"""
    logloss(actual, predicted)

Compute the mean log loss (cross-entropy loss).

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Predicted probabilities for positive class

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [0.9, 0.8, 0.4, 0.5, 0.3, 0.2]
logloss(actual, predicted)
```
"""
function logloss(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return mean(ll(actual, predicted))
end

"""
    precision(actual, predicted)

Compute precision (positive predictive value).

Proportion of positive predictions that are actually positive.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 1, 1, 1, 1, 1]
precision(actual, predicted)
```
"""
function precision(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    positive_preds = predicted .== 1
    if !any(positive_preds)
        return NaN
    end
    return mean(actual[positive_preds])
end

"""
    recall(actual, predicted)

Compute recall (sensitivity, true positive rate).

Proportion of actual positives that are correctly predicted.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 1, 1]
recall(actual, predicted)
```
"""
function recall(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    actual_positives = actual .== 1
    if !any(actual_positives)
        return NaN
    end
    return mean(predicted[actual_positives])
end

"""
    fbeta_score(actual, predicted; beta=1.0)

Compute the F-beta score, a weighted harmonic mean of precision and recall.

When beta=1, this is the F1 score (equal weight to precision and recall).
When beta<1, precision is weighted more heavily.
When beta>1, recall is weighted more heavily.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)
- `beta::Real`: Weight parameter (default: 1.0)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 1, 1]
fbeta_score(actual, predicted)  # F1 score
fbeta_score(actual, predicted, beta=0.5)  # F0.5 score (precision-weighted)
fbeta_score(actual, predicted, beta=2.0)  # F2 score (recall-weighted)
```
"""
function fbeta_score(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; beta::Real=1.0)
    prec = precision(actual, predicted)
    rec = recall(actual, predicted)

    if isnan(prec) || isnan(rec) || (prec == 0 && rec == 0)
        return NaN
    end

    return (1 + beta^2) * prec * rec / ((beta^2 * prec) + rec)
end

"""
    sensitivity(actual, predicted)

Alias for `recall`. Compute sensitivity (true positive rate).

Proportion of actual positives that are correctly predicted.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)
"""
sensitivity(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}) = recall(actual, predicted)

"""
    specificity(actual, predicted)

Compute specificity (true negative rate, selectivity).

Proportion of actual negatives that are correctly predicted.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]
specificity(actual, predicted)  # 2/3 = 0.667
```
"""
function specificity(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    actual_negatives = actual .== 0
    if !any(actual_negatives)
        return NaN
    end
    return mean(predicted[actual_negatives] .== 0)
end

"""
    npv(actual, predicted)

Compute the Negative Predictive Value.

Proportion of negative predictions that are actually negative.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]
npv(actual, predicted)  # 2/3 = 0.667 (of negative predictions, 2 of 3 are correct)
```
"""
function npv(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    negative_preds = predicted .== 0
    if !any(negative_preds)
        return NaN
    end
    return mean(actual[negative_preds] .== 0)
end

"""
    fpr(actual, predicted)

Compute the False Positive Rate (fall-out, 1 - specificity).

Proportion of actual negatives that are incorrectly predicted as positive.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]
fpr(actual, predicted)  # 1/3 = 0.333
```
"""
function fpr(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return 1 - specificity(actual, predicted)
end

"""
    fnr(actual, predicted)

Compute the False Negative Rate (miss rate, 1 - recall).

Proportion of actual positives that are incorrectly predicted as negative.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]
fnr(actual, predicted)  # 1/3 = 0.333
```
"""
function fnr(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return 1 - recall(actual, predicted)
end

"""
    brier_score(actual, predicted)

Compute the Brier score for probability predictions.

The Brier score is the mean squared error of predicted probabilities compared to
binary outcomes. Lower is better (0 is perfect, 1 is worst).

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Predicted probabilities for positive class [0, 1]

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [0.9, 0.8, 0.4, 0.5, 0.3, 0.2]
brier_score(actual, predicted)
```
"""
function brier_score(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return mean((predicted .- actual).^2)
end

"""
    gini_coefficient(actual, predicted)

Compute the Gini coefficient from AUC.

Gini = 2 * AUC - 1

The Gini coefficient ranges from -1 to 1:
- 1: Perfect prediction
- 0: Random prediction
- -1: Perfectly wrong prediction

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Predicted scores (higher = more likely positive)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [0.9, 0.8, 0.4, 0.5, 0.3, 0.2]
gini_coefficient(actual, predicted)
```
"""
function gini_coefficient(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return 2 * auc(actual, predicted) - 1
end

"""
    ks_statistic(actual, predicted)

Compute the Kolmogorov-Smirnov statistic for binary classification.

KS statistic is the maximum difference between the cumulative distribution functions
of positive and negative classes. Higher values indicate better discrimination.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Predicted scores (higher = more likely positive)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [0.9, 0.8, 0.4, 0.5, 0.3, 0.2]
ks_statistic(actual, predicted)
```
"""
function ks_statistic(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    # Sort by predicted values
    sorted_indices = sortperm(predicted, rev=true)
    sorted_actual = actual[sorted_indices]

    n_pos = sum(actual .== 1)
    n_neg = sum(actual .== 0)

    if n_pos == 0 || n_neg == 0
        return NaN
    end

    # Cumulative proportions
    cum_pos = cumsum(sorted_actual .== 1) ./ n_pos
    cum_neg = cumsum(sorted_actual .== 0) ./ n_neg

    return maximum(abs.(cum_pos .- cum_neg))
end

"""
    lift(actual, predicted; percentile=0.1)

Compute the lift at a given percentile.

Lift measures how much better the model is at identifying positives compared to
random selection. Lift = (% positives in top X%) / (% positives overall).

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Predicted scores (higher = more likely positive)
- `percentile::Real`: Top fraction to consider (default: 0.1 = top 10%)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
predicted = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
lift(actual, predicted, percentile=0.3)  # Lift in top 30%
```
"""
function lift(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; percentile::Real=0.1)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    @assert 0 < percentile <= 1 "percentile must be in (0, 1]"

    n = length(actual)
    n_top = max(1, floor(Int, n * percentile))

    sorted_indices = sortperm(predicted, rev=true)
    top_actual = actual[sorted_indices[1:n_top]]

    rate_in_top = mean(top_actual .== 1)
    rate_overall = mean(actual .== 1)

    return rate_overall == 0 ? Inf : rate_in_top / rate_overall
end

"""
    gain(actual, predicted; percentile=0.1)

Compute the cumulative gain at a given percentile.

Gain measures what percentage of total positives are captured in the top X%.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Predicted scores (higher = more likely positive)
- `percentile::Real`: Top fraction to consider (default: 0.1 = top 10%)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
predicted = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
gain(actual, predicted, percentile=0.3)  # What % of positives in top 30%
```
"""
function gain(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}; percentile::Real=0.1)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    @assert 0 < percentile <= 1 "percentile must be in (0, 1]"

    n = length(actual)
    n_top = max(1, floor(Int, n * percentile))

    sorted_indices = sortperm(predicted, rev=true)
    top_actual = actual[sorted_indices[1:n_top]]

    n_pos = sum(actual .== 1)
    return n_pos == 0 ? NaN : sum(top_actual .== 1) / n_pos
end

"""
    youden_j(actual, predicted)

Compute Youden's J statistic (informedness).

J = sensitivity + specificity - 1 = TPR - FPR

Ranges from -1 to 1, where 1 indicates perfect prediction.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]
youden_j(actual, predicted)
```
"""
function youden_j(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return sensitivity(actual, predicted) + specificity(actual, predicted) - 1
end

"""
    markedness(actual, predicted)

Compute markedness (deltaP, Δp).

Markedness = PPV + NPV - 1 = precision + NPV - 1

The counterpart to informedness (Youden's J) in the prediction direction.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]
markedness(actual, predicted)
```
"""
function markedness(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    return precision(actual, predicted) + npv(actual, predicted) - 1
end

"""
    fowlkes_mallows_index(actual, predicted)

Compute the Fowlkes-Mallows index.

FM = sqrt(PPV × TPR) = sqrt(precision × recall)

Geometric mean of precision and recall, ranges from 0 to 1.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]
fowlkes_mallows_index(actual, predicted)
```
"""
function fowlkes_mallows_index(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    prec = precision(actual, predicted)
    rec = recall(actual, predicted)
    return sqrt(prec * rec)
end

"""
    positive_likelihood_ratio(actual, predicted)

Compute the Positive Likelihood Ratio (LR+).

LR+ = TPR / FPR = sensitivity / (1 - specificity)

Indicates how much more likely a positive prediction is for actual positives.
Higher values are better.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]
positive_likelihood_ratio(actual, predicted)
```
"""
function positive_likelihood_ratio(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    sens = sensitivity(actual, predicted)
    spec = specificity(actual, predicted)
    return spec == 1.0 ? Inf : sens / (1 - spec)
end

"""
    negative_likelihood_ratio(actual, predicted)

Compute the Negative Likelihood Ratio (LR-).

LR- = FNR / TNR = (1 - sensitivity) / specificity

Indicates how much more likely a negative prediction is for actual positives.
Lower values are better.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]
negative_likelihood_ratio(actual, predicted)
```
"""
function negative_likelihood_ratio(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    sens = sensitivity(actual, predicted)
    spec = specificity(actual, predicted)
    return spec == 0.0 ? Inf : (1 - sens) / spec
end

"""
    diagnostic_odds_ratio(actual, predicted)

Compute the Diagnostic Odds Ratio (DOR).

DOR = LR+ / LR- = (TP × TN) / (FP × FN)

The ratio of the odds of a positive prediction in actual positives to the odds
in actual negatives. Higher values indicate better discrimination.

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]
diagnostic_odds_ratio(actual, predicted)
```
"""
function diagnostic_odds_ratio(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    tp = sum((actual .== 1) .& (predicted .== 1))
    tn = sum((actual .== 0) .& (predicted .== 0))
    fp = sum((actual .== 0) .& (predicted .== 1))
    fn = sum((actual .== 1) .& (predicted .== 0))

    if fp * fn == 0
        return Inf
    end
    return (tp * tn) / (fp * fn)
end
