# Information Retrieval Metrics

Metrics for evaluating search engines, recommendation systems, and ranking models.

## Overview

Information retrieval metrics evaluate how well a system ranks or retrieves relevant items. They're essential for:

- Search engines
- Recommendation systems
- Question answering
- Document retrieval

## Quick Reference

| Metric | Input | Range | Best For |
|--------|-------|-------|----------|
| `ndcg` | Graded relevance | [0, 1] | Search ranking |
| `mrr` | Binary relevance | [0, 1] | Finding first result |
| `mapk` | Binary relevance | [0, 1] | Overall ranking quality |
| `hit_rate` | Binary relevance | [0, 1] | Recommendations |
| `precision_at_k` | Binary relevance | [0, 1] | Top-k quality |

## Relevance Types

### Binary Relevance
Items are either relevant (1) or not (0).
```julia
actual_relevant = ["doc1", "doc3", "doc5"]  # Relevant documents
retrieved = ["doc1", "doc2", "doc3"]         # What the system returned
```

### Graded Relevance
Items have relevance scores (e.g., 0-5).
```julia
relevance = [3, 2, 1, 0, 2]  # Scores for items in ranked order
```

## Discounted Cumulative Gain (DCG) Family

### DCG

```@docs
dcg
```

**How it works**: Sums relevance scores with logarithmic discount by position.
- Items at top positions contribute more
- Formula: Σ (2^rel - 1) / log₂(i + 1)

### IDCG (Ideal DCG)

```@docs
idcg
```

**How it works**: DCG of the best possible ranking (sorted by relevance descending).

### NDCG (Normalized DCG)

```@docs
ndcg
```

**Interpretation**:
- NDCG = 1: Perfect ranking
- NDCG = 0: No relevant items retrieved

**When to use**:
- Search engine evaluation
- When relevance is graded (not binary)
- When position matters

### Mean NDCG

```@docs
mean_ndcg
```

**When to use**: Evaluating across multiple queries.

## Reciprocal Rank Metrics

### Reciprocal Rank

```@docs
reciprocal_rank
```

**How it works**: 1/position of first relevant item.

### Mean Reciprocal Rank (MRR)

```@docs
mrr
```

**When to use**:
- Question answering systems
- When only the first relevant result matters
- Voice assistants, "I'm Feeling Lucky" searches

**Interpretation**:
- MRR = 1: First result always relevant
- MRR = 0.5: First relevant result is typically at position 2

## Average Precision

### AP@K

```@docs
apk
```

**How it works**: Average precision at each position where a relevant item is found.

### MAP@K (Mean Average Precision)

```@docs
mapk
```

**When to use**:
- Standard metric for document retrieval
- When both precision and recall matter
- Benchmark datasets (TREC, MS MARCO)

## Set-Based Retrieval Metrics

### F1 Score (IR Context)

```@docs
f1
```

### Precision@K

```@docs
precision_at_k
```

**Interpretation**: Of the top K results, what fraction are relevant?

### Recall@K

```@docs
recall_at_k
```

**Interpretation**: Of all relevant items, what fraction appear in top K?

### F1@K

```@docs
f1_at_k
```

## Hit Rate

```@docs
hit_rate
```

**When to use**:
- Recommendation systems
- When showing at least one good item is success
- E-commerce, media streaming

## Recommendation System Metrics

### Coverage

```@docs
coverage
```

**Interpretation**: What fraction of the catalog gets recommended?
- High coverage: Diverse recommendations
- Low coverage: Recommendations focus on popular items

### Novelty

```@docs
novelty
```

**Interpretation**: Are we recommending non-obvious items?
- High novelty: Recommending less popular ("long-tail") items
- Low novelty: Recommending already-popular items

## Usage Examples

### Search Engine Evaluation

```julia
using UnifiedMetrics

# Graded relevance scores for top 6 results
# 3 = highly relevant, 2 = relevant, 1 = marginally relevant, 0 = not relevant
relevance = [3, 2, 1, 0, 2, 1]

println("DCG: ", round(dcg(relevance), digits=3))
println("NDCG: ", round(ndcg(relevance), digits=3))
println("NDCG@3: ", round(ndcg(relevance, k=3), digits=3))

# Multiple queries
relevances = [
    [3, 2, 1, 0],    # Query 1
    [0, 1, 2, 3],    # Query 2 (poor ranking)
    [3, 3, 2, 1],    # Query 3 (good ranking)
]
println("Mean NDCG: ", round(mean_ndcg(relevances), digits=3))
println("Mean NDCG@2: ", round(mean_ndcg(relevances, k=2), digits=3))
```

### Document Retrieval Evaluation

```julia
using UnifiedMetrics

# Multiple queries
actual_relevant = [
    ["doc1", "doc5", "doc7"],           # Relevant docs for query 1
    ["doc2", "doc3"],                    # Relevant docs for query 2
    ["doc4", "doc6", "doc8", "doc9"],   # Relevant docs for query 3
]

retrieved = [
    ["doc1", "doc2", "doc5", "doc3"],   # Retrieved for query 1
    ["doc1", "doc2", "doc4"],           # Retrieved for query 2
    ["doc4", "doc5", "doc6", "doc7"],   # Retrieved for query 3
]

println("MAP@3: ", round(mapk(3, actual_relevant, retrieved), digits=3))
println("MRR: ", round(mrr(actual_relevant, retrieved), digits=3))
```

### Recommendation System Evaluation

```julia
using UnifiedMetrics

# User-item recommendations
actual_liked = [
    ["item_a", "item_c"],              # User 1's liked items
    ["item_b", "item_d", "item_e"],    # User 2's liked items
    ["item_a", "item_f"],              # User 3's liked items
]

recommended = [
    ["item_a", "item_b", "item_g", "item_c", "item_h"],
    ["item_x", "item_d", "item_y", "item_b", "item_z"],
    ["item_f", "item_a", "item_m", "item_n", "item_o"],
]

println("=== Recommendation Quality ===")
println("Hit Rate@3: ", round(hit_rate(actual_liked, recommended, k=3), digits=3))
println("Hit Rate@5: ", round(hit_rate(actual_liked, recommended, k=5), digits=3))
println("MRR: ", round(mrr(actual_liked, recommended), digits=3))
println("MAP@5: ", round(mapk(5, actual_liked, recommended), digits=3))

# Per-user metrics
for (i, (act, rec)) in enumerate(zip(actual_liked, recommended))
    println("User $i - P@3: $(round(precision_at_k(act, rec, k=3), digits=2)), ",
            "R@3: $(round(recall_at_k(act, rec, k=3), digits=2))")
end
```

### Evaluating Recommendation Diversity

```julia
using UnifiedMetrics

# Full catalog of items
catalog = ["item_" * string(i) for i in 1:100]

# Recommendations for 50 users
recommendations = [["item_1", "item_2", "item_3", "item_5", "item_10"],
                   ["item_1", "item_3", "item_7", "item_12", "item_15"],
                   # ... more users
                  ]

# What fraction of catalog was recommended?
cov = coverage(recommendations, catalog)
println("Catalog Coverage: $(round(cov*100, digits=1))%")

# Novelty (recommending less popular items)
popularity = Dict("item_$i" => 1.0/i for i in 1:100)  # Power law popularity
nov = novelty(recommendations, popularity)
println("Novelty: ", round(nov, digits=2))
```

### Comparing Ranking Models

```julia
using UnifiedMetrics

# Ground truth relevance for 3 queries
actual = [
    ["a", "b", "c"],
    ["d", "e"],
    ["f", "g", "h", "i"],
]

# Model A's rankings
model_a = [
    ["a", "x", "b", "y", "c"],
    ["d", "z", "e", "w"],
    ["f", "g", "x", "h", "i"],
]

# Model B's rankings
model_b = [
    ["x", "a", "b", "c", "y"],
    ["e", "d", "z", "w"],
    ["x", "y", "f", "g", "h"],
]

println("=== Model Comparison ===")
for (name, model) in [("Model A", model_a), ("Model B", model_b)]
    println("$name:")
    println("  MAP@3: $(round(mapk(3, actual, model), digits=3))")
    println("  MRR: $(round(mrr(actual, model), digits=3))")
    println("  Hit Rate@3: $(round(hit_rate(actual, model, k=3), digits=3))")
end
```
