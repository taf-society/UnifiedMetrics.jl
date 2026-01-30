"""
    f1(actual, predicted)

Compute the F1 score in the context of information retrieval.

Computes `2 * precision * recall / (precision + recall)` where precision is the
proportion of retrieved documents that are relevant, and recall is the proportion
of relevant documents that are retrieved.

Returns 0 if there are no true positives.

# Arguments
- `actual::AbstractVector`: Ground truth relevant documents (order doesn't matter)
- `predicted::AbstractVector`: Retrieved documents (order doesn't matter)

# Examples
```julia
actual = ['a', 'c', 'd']
predicted = ['d', 'e']
f1(actual, predicted)
```
"""
function f1(actual::AbstractVector, predicted::AbstractVector)
    act = unique(actual)
    pred = unique(predicted)

    tp = length(intersect(act, pred))
    fp = length(setdiff(pred, act))
    fn = length(setdiff(act, pred))

    if tp == 0
        return 0.0
    end

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)
end

"""
    apk(k, actual, predicted)

Compute the average precision at k.

Loops over the first k values of `predicted`. For each value that is in `actual`
and hasn't been predicted before, increments the score by (number of hits so far) / position.
Returns the final score divided by min(length(actual), k).

Returns `NaN` if `actual` is empty.

# Arguments
- `k::Integer`: Number of predictions to consider
- `actual::AbstractVector`: Ground truth relevant documents (order doesn't matter)
- `predicted::AbstractVector`: Retrieved documents in ranked order (most relevant first)

# Examples
```julia
actual = ['a', 'b', 'd']
predicted = ['b', 'c', 'a', 'e', 'f']
apk(3, actual, predicted)
```
"""
function apk(k::Integer, actual::AbstractVector, predicted::AbstractVector)
    if isempty(actual)
        return NaN
    end

    actual_set = Set(actual)  # O(1) lookup instead of O(n)
    score = 0.0
    cnt = 0.0
    seen = Set{eltype(predicted)}()

    for i in 1:min(k, length(predicted))
        if predicted[i] in actual_set && !(predicted[i] in seen)
            cnt += 1
            score += cnt / i
        end
        push!(seen, predicted[i])
    end

    return score / min(length(actual), k)
end

"""
    mapk(k, actual, predicted)

Compute the mean average precision at k.

Evaluates `apk` for each pair of elements from `actual` and `predicted` lists,
then returns the mean.

# Arguments
- `k::Integer`: Number of predictions to consider for each query
- `actual::AbstractVector{<:AbstractVector}`: List of ground truth vectors
- `predicted::AbstractVector{<:AbstractVector}`: List of prediction vectors

# Examples
```julia
actual = [['a', 'b'], ['a'], ['x', 'y', 'b']]
predicted = [['a', 'c', 'd'], ['x', 'b', 'a', 'b'], ['y']]
mapk(2, actual, predicted)
```
"""
function mapk(k::Integer, actual::AbstractVector{<:AbstractVector},
              predicted::AbstractVector{<:AbstractVector})
    if isempty(actual) || isempty(predicted)
        return 0.0
    end

    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    scores = [apk(k, actual[i], predicted[i]) for i in eachindex(actual)]
    return mean(scores)
end

"""
    dcg(relevance; k=nothing)

Compute the Discounted Cumulative Gain at position k.

DCG measures the usefulness of a ranking based on relevance scores, with a
logarithmic discount to penalize relevant items appearing lower in the ranking.

DCG = Σ (2^rel_i - 1) / log2(i + 1)

# Arguments
- `relevance::AbstractVector{<:Real}`: Relevance scores in ranked order (highest rank first)
- `k::Union{Integer,Nothing}`: Number of positions to consider (default: all)

# Examples
```julia
relevance = [3, 2, 3, 0, 1, 2]  # Relevance scores for ranked items
dcg(relevance)  # DCG for all positions
dcg(relevance, k=3)  # DCG@3
```
"""
function dcg(relevance::AbstractVector{<:Real}; k::Union{Integer,Nothing}=nothing)
    n = isnothing(k) ? length(relevance) : min(k, length(relevance))
    if n == 0
        return 0.0
    end

    gains = (2.0 .^ relevance[1:n]) .- 1.0
    discounts = log2.(2:(n+1))

    return sum(gains ./ discounts)
end

"""
    idcg(relevance; k=nothing)

Compute the Ideal Discounted Cumulative Gain at position k.

IDCG is the DCG of the best possible ranking (relevance scores sorted descending).

# Arguments
- `relevance::AbstractVector{<:Real}`: Relevance scores (order doesn't matter)
- `k::Union{Integer,Nothing}`: Number of positions to consider (default: all)

# Examples
```julia
relevance = [3, 2, 3, 0, 1, 2]
idcg(relevance)  # Ideal DCG for all positions
idcg(relevance, k=3)  # Ideal DCG@3
```
"""
function idcg(relevance::AbstractVector{<:Real}; k::Union{Integer,Nothing}=nothing)
    sorted_rel = sort(relevance, rev=true)
    return dcg(sorted_rel, k=k)
end

"""
    ndcg(relevance; k=nothing)

Compute the Normalized Discounted Cumulative Gain at position k.

NDCG = DCG / IDCG

Normalizes DCG to [0, 1] by dividing by the ideal DCG.

# Arguments
- `relevance::AbstractVector{<:Real}`: Relevance scores in ranked order (highest rank first)
- `k::Union{Integer,Nothing}`: Number of positions to consider (default: all)

# Examples
```julia
relevance = [3, 2, 3, 0, 1, 2]  # Actual ranking
ndcg(relevance)  # NDCG for all positions
ndcg(relevance, k=3)  # NDCG@3
```
"""
function ndcg(relevance::AbstractVector{<:Real}; k::Union{Integer,Nothing}=nothing)
    ideal = idcg(relevance, k=k)
    return ideal == 0 ? 0.0 : dcg(relevance, k=k) / ideal
end

"""
    mrr(actual, predicted)

Compute the Mean Reciprocal Rank.

MRR is the average of reciprocal ranks of the first relevant item for each query.

# Arguments
- `actual::AbstractVector{<:AbstractVector}`: List of ground truth relevant items for each query
- `predicted::AbstractVector{<:AbstractVector}`: List of ranked predictions for each query

# Examples
```julia
actual = [["a", "b"], ["c"], ["d", "e"]]
predicted = [["b", "a", "c"], ["a", "c", "d"], ["e", "d", "f"]]
mrr(actual, predicted)  # (1/2 + 1/2 + 1/1) / 3 = 0.667
```
"""
function mrr(actual::AbstractVector{<:AbstractVector}, predicted::AbstractVector{<:AbstractVector})
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    reciprocal_ranks = Float64[]
    for (act, pred) in zip(actual, predicted)
        act_set = Set(act)
        rr = 0.0
        for (i, item) in enumerate(pred)
            if item in act_set
                rr = 1.0 / i
                break
            end
        end
        push!(reciprocal_ranks, rr)
    end

    return mean(reciprocal_ranks)
end

"""
    reciprocal_rank(actual, predicted)

Compute the Reciprocal Rank for a single query.

# Arguments
- `actual::AbstractVector`: Ground truth relevant items
- `predicted::AbstractVector`: Ranked predictions (highest rank first)

# Examples
```julia
actual = ["a", "b"]
predicted = ["c", "a", "b", "d"]
reciprocal_rank(actual, predicted)  # 1/2 = 0.5 (first relevant at position 2)
```
"""
function reciprocal_rank(actual::AbstractVector, predicted::AbstractVector)
    act_set = Set(actual)
    for (i, item) in enumerate(predicted)
        if item in act_set
            return 1.0 / i
        end
    end
    return 0.0
end

"""
    hit_rate(actual, predicted; k=10)

Compute the hit rate (recall@k) for recommendation systems.

Hit rate is the fraction of queries where at least one relevant item appears
in the top k predictions.

# Arguments
- `actual::AbstractVector{<:AbstractVector}`: List of ground truth relevant items for each query
- `predicted::AbstractVector{<:AbstractVector}`: List of ranked predictions for each query
- `k::Integer`: Number of top predictions to consider (default: 10)

# Examples
```julia
actual = [["a", "b"], ["c"], ["d", "e"]]
predicted = [["a", "x", "y"], ["x", "y", "z"], ["e", "f", "g"]]
hit_rate(actual, predicted, k=3)  # 2/3 queries have a hit in top 3
```
"""
function hit_rate(actual::AbstractVector{<:AbstractVector}, predicted::AbstractVector{<:AbstractVector}; k::Integer=10)
    @assert length(actual) == length(predicted) "Length of actual and predicted must be the same"

    hits = 0
    for (act, pred) in zip(actual, predicted)
        act_set = Set(act)
        top_k = pred[1:min(k, length(pred))]
        if any(item in act_set for item in top_k)
            hits += 1
        end
    end

    return hits / length(actual)
end

"""
    recall_at_k(actual, predicted; k=10)

Compute recall@k for information retrieval.

The fraction of relevant items that appear in the top k predictions.

# Arguments
- `actual::AbstractVector`: Ground truth relevant items
- `predicted::AbstractVector`: Ranked predictions (highest rank first)
- `k::Integer`: Number of top predictions to consider (default: 10)

# Examples
```julia
actual = ["a", "b", "c", "d"]
predicted = ["a", "x", "b", "y", "z"]
recall_at_k(actual, predicted, k=3)  # 2/4 = 0.5 (found "a" and "b" in top 3)
```
"""
function recall_at_k(actual::AbstractVector, predicted::AbstractVector; k::Integer=10)
    if isempty(actual)
        return NaN
    end

    act_set = Set(actual)
    top_k = Set(predicted[1:min(k, length(predicted))])
    n_hits = length(intersect(act_set, top_k))

    return n_hits / length(actual)
end

"""
    precision_at_k(actual, predicted; k=10)

Compute precision@k for information retrieval.

The fraction of top k predictions that are relevant.

# Arguments
- `actual::AbstractVector`: Ground truth relevant items
- `predicted::AbstractVector`: Ranked predictions (highest rank first)
- `k::Integer`: Number of top predictions to consider (default: 10)

# Examples
```julia
actual = ["a", "b", "c", "d"]
predicted = ["a", "x", "b", "y", "z"]
precision_at_k(actual, predicted, k=3)  # 2/3 (2 of top 3 are relevant)
```
"""
function precision_at_k(actual::AbstractVector, predicted::AbstractVector; k::Integer=10)
    actual_k = min(k, length(predicted))
    if actual_k == 0
        return NaN
    end

    act_set = Set(actual)
    top_k = predicted[1:actual_k]
    n_hits = count(item in act_set for item in top_k)

    return n_hits / actual_k
end

"""
    f1_at_k(actual, predicted; k=10)

Compute F1@k for information retrieval.

Harmonic mean of precision@k and recall@k.

# Arguments
- `actual::AbstractVector`: Ground truth relevant items
- `predicted::AbstractVector`: Ranked predictions (highest rank first)
- `k::Integer`: Number of top predictions to consider (default: 10)

# Examples
```julia
actual = ["a", "b", "c", "d"]
predicted = ["a", "x", "b", "y", "z"]
f1_at_k(actual, predicted, k=3)
```
"""
function f1_at_k(actual::AbstractVector, predicted::AbstractVector; k::Integer=10)
    prec = precision_at_k(actual, predicted, k=k)
    rec = recall_at_k(actual, predicted, k=k)

    if isnan(prec) || isnan(rec)
        return NaN
    end

    if prec == 0 && rec == 0
        return 0.0
    end

    return 2 * prec * rec / (prec + rec)
end

"""
    mean_ndcg(relevances; k=nothing)

Compute the mean NDCG over multiple queries.

# Arguments
- `relevances::AbstractVector{<:AbstractVector{<:Real}}`: List of relevance score vectors
- `k::Union{Integer,Nothing}`: Number of positions to consider (default: all)

# Examples
```julia
relevances = [[3, 2, 1, 0], [2, 1, 2, 1], [1, 1, 0, 0]]
mean_ndcg(relevances)  # Mean NDCG across queries
mean_ndcg(relevances, k=2)  # Mean NDCG@2
```
"""
function mean_ndcg(relevances::AbstractVector{<:AbstractVector{<:Real}}; k::Union{Integer,Nothing}=nothing)
    return mean([ndcg(rel, k=k) for rel in relevances])
end

"""
    coverage(predicted, catalog)

Compute the catalog coverage of recommendations.

Coverage measures what fraction of items in the catalog have been recommended
at least once across all predictions.

# Arguments
- `predicted::AbstractVector{<:AbstractVector}`: List of predictions for all queries
- `catalog::AbstractVector`: Full catalog of items

# Examples
```julia
catalog = ["a", "b", "c", "d", "e", "f"]
predicted = [["a", "b"], ["a", "c"], ["b", "d"]]
coverage(predicted, catalog)  # 4/6 = 0.667 (recommended: a, b, c, d)
```
"""
function coverage(predicted::AbstractVector{<:AbstractVector}, catalog::AbstractVector)
    if isempty(catalog)
        return NaN
    end

    recommended = Set{eltype(catalog)}()
    for preds in predicted
        union!(recommended, preds)
    end

    return length(intersect(recommended, Set(catalog))) / length(catalog)
end

"""
    novelty(predicted, item_popularity)

Compute the novelty of recommendations.

Novelty measures how unexpected/surprising the recommendations are, based on
the popularity of recommended items. Higher novelty means recommending
less popular (long-tail) items.

# Arguments
- `predicted::AbstractVector{<:AbstractVector}`: List of predictions for all queries
- `item_popularity::Dict`: Dictionary mapping items to their popularity (0-1)

# Examples
```julia
popularity = Dict("a" => 0.9, "b" => 0.5, "c" => 0.1, "d" => 0.05)
predicted = [["a", "b"], ["c", "d"]]
novelty(predicted, popularity)  # Average -log2(popularity)
```
"""
function novelty(predicted::AbstractVector{<:AbstractVector}, item_popularity::Dict)
    novelty_scores = Float64[]

    for preds in predicted
        for item in preds
            if haskey(item_popularity, item)
                pop = item_popularity[item]
                if pop > 0
                    push!(novelty_scores, -log2(pop))
                end
            end
        end
    end

    return isempty(novelty_scores) ? NaN : mean(novelty_scores)
end
