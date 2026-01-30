# Classification Metrics

Metrics for evaluating multi-class classification models.

## Overview

Classification metrics evaluate how well a model assigns items to discrete categories. These metrics work with any number of classes and support both integer and string labels.

## Quick Reference

| Metric | Range | Best For |
|--------|-------|----------|
| `accuracy` | [0, 1] | Balanced datasets |
| `balanced_accuracy` | [0, 1] | Imbalanced datasets |
| `cohens_kappa` | [-1, 1] | Accounting for chance |
| `mcc` | [-1, 1] | Binary, imbalanced data |
| `hamming_loss` | [0, 1] | Multi-label problems |

## Basic Accuracy Metrics

```julia
accuracy(actual, predicted)  # Classification accuracy
ce(actual, predicted)        # Classification error (1 - accuracy)
```

**When to use**:
- `accuracy`: Quick evaluation on balanced datasets
- `ce` (classification error): When you want error rate instead of accuracy

**Important**: Accuracy can be misleading on imbalanced datasets. If 95% of samples are class A, a model that always predicts A achieves 95% accuracy.

## Balanced Accuracy

```julia
balanced_accuracy(actual, predicted)
```

**When to use**:
- Imbalanced datasets
- When each class is equally important regardless of frequency

**How it works**: Computes recall for each class and averages them.

## Agreement Metrics

```julia
cohens_kappa(actual, predicted)
```

**Interpretation**:
- κ = 1: Perfect agreement
- κ = 0: Agreement equals chance
- κ < 0: Less agreement than chance

**Guidelines** (Landis & Koch, 1977):
- 0.81-1.00: Almost perfect
- 0.61-0.80: Substantial
- 0.41-0.60: Moderate
- 0.21-0.40: Fair
- 0.00-0.20: Slight

## Matthews Correlation Coefficient

```julia
matthews_corrcoef(actual, predicted)
mcc(actual, predicted)  # Alias
```

**When to use**: The recommended single metric for binary classification, especially with imbalanced data.

**Interpretation**:
- MCC = 1: Perfect predictions
- MCC = 0: Random predictions
- MCC = -1: Complete disagreement

## Confusion Matrix

```julia
confusion_matrix(actual, predicted)
```

**How to use**:

```julia
actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 0, 1, 1, 0, 0]

cm = confusion_matrix(actual, predicted)
cm[:matrix]   # The confusion matrix
cm[:labels]   # [0, 1]

# For binary classification [0, 1]:
# matrix[1,1] = TN, matrix[1,2] = FP
# matrix[2,1] = FN, matrix[2,2] = TP
```

## Top-K Accuracy

```julia
top_k_accuracy(actual, predicted_probs, k)
```

**When to use**:
- Multi-class problems with many classes
- When partial credit for "close" predictions matters
- Image classification, recommendation systems

**Example**:
```julia
actual = [1, 2, 3]  # True classes (1-indexed)
probs = [0.7 0.2 0.1;   # Sample 1: class 1 most likely (correct)
         0.3 0.4 0.3;   # Sample 2: class 2 most likely (correct)
         0.4 0.4 0.2]   # Sample 3: class 3 not in top-2 (incorrect for top-2)

top_k_accuracy(actual, probs, 1)  # Standard accuracy
top_k_accuracy(actual, probs, 2)  # Correct if true class in top 2
```

## Quadratic Weighted Kappa

```julia
ScoreQuadraticWeightedKappa(rater_a, rater_b; min_rating, max_rating)
MeanQuadraticWeightedKappa(kappas; weights=nothing)
```

**When to use**:
- Ordinal classification (ratings, severity levels)
- When distance between classes matters
- Medical/educational assessments

**Example**:
```julia
# Rating predictions (1-5 scale)
actual_ratings = [1, 2, 3, 4, 5, 3, 2]
predicted_ratings = [1, 2, 2, 4, 4, 3, 3]

# Predicting 2 when actual is 3 is penalized less than predicting 1
ScoreQuadraticWeightedKappa(actual_ratings, predicted_ratings,
                            min_rating=1, max_rating=5)
```

## Loss Functions

### Hamming Loss

```julia
hamming_loss(actual, predicted)
```

**When to use**:
- Multi-label classification
- When you want to penalize each label error equally

### Zero-One Loss

```julia
zero_one_loss(actual, predicted)
```

### Hinge Loss

```julia
hinge_loss(actual, predicted)
squared_hinge_loss(actual, predicted)
```

**When to use**:
- Support Vector Machine evaluation
- Labels should be -1 and 1 (not 0 and 1)
- `predicted` should be decision function values, not probabilities

## Usage Examples

### Evaluating a Multi-Class Model

```julia
using UnifiedMetrics

actual = ["cat", "dog", "bird", "cat", "dog", "bird"]
predicted = ["cat", "cat", "bird", "dog", "dog", "cat"]

println("Accuracy: ", accuracy(actual, predicted))
println("Balanced Accuracy: ", balanced_accuracy(actual, predicted))
println("Cohen's Kappa: ", cohens_kappa(actual, predicted))

cm = confusion_matrix(actual, predicted)
println("Confusion Matrix:")
println(cm[:matrix])
println("Labels: ", cm[:labels])
```

### Handling Imbalanced Data

```julia
# Highly imbalanced: 90% class A, 10% class B
actual = vcat(fill("A", 90), fill("B", 10))
predicted = fill("A", 100)  # Naive model always predicts A

accuracy(actual, predicted)           # 0.9 - looks good!
balanced_accuracy(actual, predicted)  # 0.5 - reveals the problem
```

### Ordinal Classification

```julia
# Patient pain levels: 1 (none) to 5 (severe)
actual = [1, 2, 3, 4, 5, 3, 2, 4]
predicted = [1, 2, 2, 4, 4, 3, 3, 5]

# Standard accuracy doesn't account for "closeness"
accuracy(actual, predicted)  # 0.625

# QWK penalizes predictions far from actual more
ScoreQuadraticWeightedKappa(actual, predicted, min_rating=1, max_rating=5)
```

### Multi-Label Classification

```julia
# Each sample can have multiple labels (e.g., image tags)
actual = Bool[1 0 1; 0 1 1; 1 1 0]      # 3 samples, 3 labels
predicted = Bool[1 1 1; 0 1 0; 1 0 0]   # Predictions

# Fraction of incorrectly predicted labels
hamming_loss(actual, predicted)  # 0.444 (4 of 9 labels wrong)
```

See the [API Reference](@ref) for complete function documentation.
