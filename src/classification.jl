"""
    ce(actual, predicted)

Compute the classification error (proportion of misclassified observations).

# Arguments
- `actual::AbstractVector`: Ground truth vector
- `predicted::AbstractVector`: Predicted vector

# Examples
```julia
actual = ['a', 'a', 'c', 'b', 'c']
predicted = ['a', 'b', 'c', 'b', 'a']
ce(actual, predicted)
```
"""
function ce(actual::AbstractVector, predicted::AbstractVector)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return mean(actual .!= predicted)
end

"""
    accuracy(actual, predicted)

Compute the classification accuracy (proportion of correctly classified observations).

# Arguments
- `actual::AbstractVector`: Ground truth vector
- `predicted::AbstractVector`: Predicted vector

# Examples
```julia
actual = ['a', 'a', 'c', 'b', 'c']
predicted = ['a', 'b', 'c', 'b', 'a']
accuracy(actual, predicted)
```
"""
function accuracy(actual::AbstractVector, predicted::AbstractVector)
    return 1 - ce(actual, predicted)
end

"""
    ScoreQuadraticWeightedKappa(rater_a, rater_b; min_rating=nothing, max_rating=nothing)

Compute the quadratic weighted kappa between two vectors of integer ratings.

# Arguments
- `rater_a::AbstractVector{<:Integer}`: First rater's ratings
- `rater_b::AbstractVector{<:Integer}`: Second rater's ratings
- `min_rating::Union{Integer,Nothing}`: Minimum possible rating (default: minimum of both vectors)
- `max_rating::Union{Integer,Nothing}`: Maximum possible rating (default: maximum of both vectors)

# Examples
```julia
rater_a = [1, 4, 5, 5, 2, 1]
rater_b = [2, 2, 4, 5, 3, 3]
ScoreQuadraticWeightedKappa(rater_a, rater_b, min_rating=1, max_rating=5)
```
"""
function ScoreQuadraticWeightedKappa(rater_a::AbstractVector{<:Integer},
                                      rater_b::AbstractVector{<:Integer};
                                      min_rating::Union{Integer,Nothing}=nothing,
                                      max_rating::Union{Integer,Nothing}=nothing)
    @assert length(rater_a) == length(rater_b) "Length of rater_a and rater_b must be the same"

    min_r = isnothing(min_rating) ? min(minimum(rater_a), minimum(rater_b)) : min_rating
    max_r = isnothing(max_rating) ? max(maximum(rater_a), maximum(rater_b)) : max_rating

    levels = min_r:max_r
    n_levels = length(levels)

    # Build confusion matrix
    confusion_mat = zeros(n_levels, n_levels)
    for (a, b) in zip(rater_a, rater_b)
        i = a - min_r + 1
        j = b - min_r + 1
        confusion_mat[i, j] += 1
    end
    confusion_mat ./= sum(confusion_mat)

    # Get expected matrix under independence
    hist_a = zeros(n_levels)
    hist_b = zeros(n_levels)
    for a in rater_a
        hist_a[a - min_r + 1] += 1
    end
    for b in rater_b
        hist_b[b - min_r + 1] += 1
    end
    hist_a ./= length(rater_a)
    hist_b ./= length(rater_b)

    expected_mat = hist_a * hist_b'
    expected_mat ./= sum(expected_mat)

    # Build weight matrix
    labels = collect(levels)
    weights = [(labels[i] - labels[j])^2 for i in 1:n_levels, j in 1:n_levels]

    # Calculate kappa
    return 1 - sum(weights .* confusion_mat) / sum(weights .* expected_mat)
end

"""
    MeanQuadraticWeightedKappa(kappas; weights=nothing)

Compute the mean quadratic weighted kappa, optionally weighted.

Uses Fisher's z-transformation for averaging.

# Arguments
- `kappas::AbstractVector{<:Real}`: Vector of kappa values
- `weights::Union{AbstractVector{<:Real},Nothing}`: Optional weights (default: equal weights)

# Examples
```julia
kappas = [0.3, 0.2, 0.2, 0.5, 0.1, 0.2]
weights = [1.0, 2.5, 1.0, 1.0, 2.0, 3.0]
MeanQuadraticWeightedKappa(kappas, weights=weights)
```
"""
function MeanQuadraticWeightedKappa(kappas::AbstractVector{<:Real};
                                     weights::Union{AbstractVector{<:Real},Nothing}=nothing)
    w = isnothing(weights) ? ones(length(kappas)) : copy(weights)
    w ./= mean(w)

    # Clamp kappas to avoid singularities
    max999(x) = sign(x) * min(0.999, abs(x))
    min001(x) = sign(x) * max(0.001, abs(x))

    clamped_kappas = [min001(max999(k)) for k in kappas]

    # Fisher's z-transformation
    r2z(x) = 0.5 * log((1 + x) / (1 - x))
    z2r(x) = (exp(2x) - 1) / (exp(2x) + 1)

    z_kappas = [r2z(k) for k in clamped_kappas]
    weighted_mean = mean(z_kappas .* w)

    return z2r(weighted_mean)
end

"""
    balanced_accuracy(actual, predicted)

Compute balanced accuracy, which accounts for imbalanced datasets.

Balanced accuracy is the macro-averaged recall: the average of recall scores
for each class. It gives equal weight to each class regardless of its frequency.

# Arguments
- `actual::AbstractVector`: Ground truth vector
- `predicted::AbstractVector`: Predicted vector

# Examples
```julia
actual = [1, 1, 1, 1, 0, 0]  # Imbalanced: 4 positives, 2 negatives
predicted = [1, 1, 1, 0, 0, 0]
balanced_accuracy(actual, predicted)  # (0.75 + 1.0) / 2 = 0.875
```
"""
function balanced_accuracy(actual::AbstractVector, predicted::AbstractVector)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    classes = unique(actual)
    recalls = Float64[]

    for c in classes
        mask = actual .== c
        if any(mask)
            push!(recalls, mean(predicted[mask] .== c))
        end
    end

    return isempty(recalls) ? 0.0 : mean(recalls)
end

"""
    cohens_kappa(actual, predicted)

Compute Cohen's Kappa coefficient for inter-rater agreement.

Kappa measures agreement between two raters, accounting for agreement by chance.
- κ = 1: Perfect agreement
- κ = 0: Agreement equivalent to chance
- κ < 0: Less agreement than chance

# Arguments
- `actual::AbstractVector`: Ground truth vector
- `predicted::AbstractVector`: Predicted vector

# Examples
```julia
actual = ['a', 'a', 'b', 'b', 'c', 'c']
predicted = ['a', 'b', 'b', 'b', 'c', 'a']
cohens_kappa(actual, predicted)
```
"""
function cohens_kappa(actual::AbstractVector, predicted::AbstractVector)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    n = length(actual)
    classes = unique(vcat(actual, predicted))

    # Observed agreement
    po = mean(actual .== predicted)

    # Expected agreement by chance
    pe = 0.0
    for c in classes
        p_actual = sum(actual .== c) / n
        p_pred = sum(predicted .== c) / n
        pe += p_actual * p_pred
    end

    return pe == 1.0 ? 1.0 : (po - pe) / (1 - pe)
end

"""
    matthews_corrcoef(actual, predicted)

Compute the Matthews Correlation Coefficient (MCC) for binary classification.

MCC is considered one of the best metrics for binary classification, especially
for imbalanced datasets. It returns a value in [-1, 1]:
- +1: Perfect prediction
-  0: Random prediction
- -1: Total disagreement

# Arguments
- `actual::AbstractVector{<:Real}`: Binary ground truth (1 for positive, 0 for negative)
- `predicted::AbstractVector{<:Real}`: Binary predictions (1 for positive, 0 for negative)

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]
matthews_corrcoef(actual, predicted)
```
"""
function matthews_corrcoef(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    tp = sum((actual .== 1) .& (predicted .== 1))
    tn = sum((actual .== 0) .& (predicted .== 0))
    fp = sum((actual .== 0) .& (predicted .== 1))
    fn = sum((actual .== 1) .& (predicted .== 0))

    denom = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return denom == 0 ? 0.0 : (tp * tn - fp * fn) / denom
end

"""
    mcc(actual, predicted)

Alias for `matthews_corrcoef`. Compute the Matthews Correlation Coefficient.
"""
mcc(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real}) = matthews_corrcoef(actual, predicted)

"""
    confusion_matrix(actual, predicted)

Compute the confusion matrix for classification.

Returns a dictionary with keys:
- `:matrix` - The confusion matrix as a 2D array
- `:labels` - The class labels in order

For binary classification with classes 0 and 1:
- matrix[1,1] = TN (true negatives)
- matrix[1,2] = FP (false positives)
- matrix[2,1] = FN (false negatives)
- matrix[2,2] = TP (true positives)

# Arguments
- `actual::AbstractVector`: Ground truth vector
- `predicted::AbstractVector`: Predicted vector

# Examples
```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]
cm = confusion_matrix(actual, predicted)
cm[:matrix]  # 2x2 confusion matrix
cm[:labels]  # [0, 1]
```
"""
function confusion_matrix(actual::AbstractVector, predicted::AbstractVector)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    labels = sort(unique(vcat(actual, predicted)))
    n_classes = length(labels)
    label_to_idx = Dict(l => i for (i, l) in enumerate(labels))

    matrix = zeros(Int, n_classes, n_classes)
    for (a, p) in zip(actual, predicted)
        i = label_to_idx[a]
        j = label_to_idx[p]
        matrix[i, j] += 1
    end

    return Dict(:matrix => matrix, :labels => labels)
end

"""
    top_k_accuracy(actual, predicted_probs, k)

Compute top-k accuracy for multi-class classification.

Returns the fraction of samples where the true class is among the top k predictions.

# Arguments
- `actual::AbstractVector{<:Integer}`: Ground truth class indices (1-indexed)
- `predicted_probs::AbstractMatrix{<:Real}`: Matrix of predicted probabilities (samples × classes)
- `k::Integer`: Number of top predictions to consider

# Examples
```julia
actual = [1, 2, 3, 1]
predicted_probs = [0.8 0.1 0.1;   # Sample 1: class 1 most likely
                   0.2 0.5 0.3;   # Sample 2: class 2 most likely
                   0.1 0.3 0.6;   # Sample 3: class 3 most likely
                   0.3 0.4 0.3]   # Sample 4: class 2 most likely (actual is 1)
top_k_accuracy(actual, predicted_probs, 1)  # Standard accuracy
top_k_accuracy(actual, predicted_probs, 2)  # Top-2 accuracy
```
"""
function top_k_accuracy(actual::AbstractVector{<:Integer}, predicted_probs::AbstractMatrix{<:Real}, k::Integer)
    n_samples = length(actual)
    @assert size(predicted_probs, 1) == n_samples "Number of rows must match length of actual"
    @assert k >= 1 "k must be at least 1"
    @assert k <= size(predicted_probs, 2) "k cannot exceed number of classes"

    correct = 0
    for i in 1:n_samples
        # Get indices of top k predictions (sorted by probability descending)
        probs = predicted_probs[i, :]
        top_k_indices = partialsortperm(probs, 1:k, rev=true)
        if actual[i] in top_k_indices
            correct += 1
        end
    end

    return correct / n_samples
end

"""
    hamming_loss(actual, predicted)

Compute the Hamming loss (fraction of misclassified labels).

For single-label classification, this equals the classification error (1 - accuracy).
For multi-label classification, it measures the fraction of incorrect labels.

# Arguments
- `actual::AbstractVector`: Ground truth vector
- `predicted::AbstractVector`: Predicted vector

# Examples
```julia
actual = ['a', 'a', 'b', 'b', 'c', 'c']
predicted = ['a', 'b', 'b', 'b', 'c', 'a']
hamming_loss(actual, predicted)
```
"""
function hamming_loss(actual::AbstractVector, predicted::AbstractVector)
    return ce(actual, predicted)  # Same as classification error for single-label
end

"""
    hamming_loss(actual, predicted)

Compute Hamming loss for multi-label classification.

# Arguments
- `actual::AbstractMatrix{Bool}`: Ground truth binary matrix (samples × labels)
- `predicted::AbstractMatrix{Bool}`: Predicted binary matrix (samples × labels)
"""
function hamming_loss(actual::AbstractMatrix{Bool}, predicted::AbstractMatrix{Bool})
    @assert size(actual) == size(predicted) "Dimensions must match"
    return mean(actual .!= predicted)
end

"""
    zero_one_loss(actual, predicted)

Compute the zero-one loss (fraction of samples with any incorrect prediction).

For single-label classification, this equals the classification error.

# Arguments
- `actual::AbstractVector`: Ground truth vector
- `predicted::AbstractVector`: Predicted vector

# Examples
```julia
actual = ['a', 'a', 'b', 'b']
predicted = ['a', 'b', 'b', 'b']
zero_one_loss(actual, predicted)
```
"""
function zero_one_loss(actual::AbstractVector, predicted::AbstractVector)
    return ce(actual, predicted)
end

"""
    hinge_loss(actual, predicted)

Compute the hinge loss for binary classification (used in SVMs).

Actual values should be -1 or 1. Predicted values are the decision function values
(not probabilities).

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth (-1 or 1)
- `predicted::AbstractVector{<:Real}`: Decision function values (raw scores)

# Examples
```julia
actual = [1, 1, -1, -1]
predicted = [0.8, 0.3, -0.5, 0.1]  # Decision function values
hinge_loss(actual, predicted)
```
"""
function hinge_loss(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return mean(max.(0, 1 .- actual .* predicted))
end

"""
    squared_hinge_loss(actual, predicted)

Compute the squared hinge loss (used in some SVMs).

# Arguments
- `actual::AbstractVector{<:Real}`: Ground truth (-1 or 1)
- `predicted::AbstractVector{<:Real}`: Decision function values (raw scores)

# Examples
```julia
actual = [1, 1, -1, -1]
predicted = [0.8, 0.3, -0.5, 0.1]
squared_hinge_loss(actual, predicted)
```
"""
function squared_hinge_loss(actual::AbstractVector{<:Real}, predicted::AbstractVector{<:Real})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"
    return mean(max.(0, 1 .- actual .* predicted).^2)
end
