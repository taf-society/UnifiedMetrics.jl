# Binary Classification Metrics

Metrics for evaluating models that predict between two classes (positive/negative, yes/no, 1/0).

## Overview

Binary classification metrics fall into two categories:

1. **Threshold-dependent**: Require binary predictions (0/1)
   - Precision, Recall, F-score, Specificity
2. **Threshold-independent**: Use probability scores
   - AUC, Brier score, Log loss

## Quick Reference

| Metric | Input Type | Range | Best For |
|--------|-----------|-------|----------|
| `auc` | Probabilities | [0, 1] | Model comparison |
| `precision` | Labels | [0, 1] | Minimizing FP |
| `recall` | Labels | [0, 1] | Minimizing FN |
| `fbeta_score` | Labels | [0, 1] | Balanced evaluation |
| `mcc` | Labels | [-1, 1] | Imbalanced data |
| `brier_score` | Probabilities | [0, 1] | Calibration |

## ROC-Based Metrics

### Area Under ROC Curve

```julia
auc(actual, predicted_probs)
```

**Interpretation**:
- AUC = 1.0: Perfect ranking
- AUC = 0.5: Random guessing
- AUC < 0.5: Worse than random (flip predictions!)

**Guidelines**:
- 0.9-1.0: Excellent
- 0.8-0.9: Good
- 0.7-0.8: Fair
- 0.6-0.7: Poor
- 0.5-0.6: Fail

### Gini Coefficient

```julia
gini_coefficient(actual, predicted_probs)
```

**Relationship**: Gini = 2 × AUC - 1

### KS Statistic

```julia
ks_statistic(actual, predicted_probs)
```

**When to use**: Credit scoring, marketing response modeling.

## Probability Calibration Metrics

### Log Loss

```julia
ll(actual, predicted_probs)      # Elementwise
logloss(actual, predicted_probs) # Mean
```

**When to use**:
- Training neural networks (cross-entropy loss)
- When probability values matter, not just ranking

### Brier Score

```julia
brier_score(actual, predicted_probs)
```

**Interpretation**:
- 0: Perfect calibration
- 0.25: Random guessing for balanced data
- 1: Complete miscalibration

**When to use**: Weather forecasting, medical prognosis - anywhere probability calibration matters.

## Precision and Recall

### Precision (Positive Predictive Value)

```julia
precision(actual, predicted_labels)
```

**Interpretation**: Of all samples predicted positive, what fraction are actually positive?

**Optimize for precision when**: False positives are costly
- Spam detection (don't mark good email as spam)
- Legal discovery (don't flag innocent documents)

### Recall (Sensitivity, True Positive Rate)

```julia
recall(actual, predicted_labels)
sensitivity(actual, predicted_labels)  # Alias
```

**Interpretation**: Of all actual positives, what fraction did we detect?

**Optimize for recall when**: False negatives are costly
- Disease screening (don't miss sick patients)
- Fraud detection (don't miss fraudulent transactions)
- Security threats (don't miss actual threats)

## F-Score

```julia
fbeta_score(actual, predicted_labels; beta=1.0)
```

**Choosing beta**:
- β = 1: Equal weight to precision and recall (F1)
- β = 0.5: Precision weighted 2× more than recall
- β = 2: Recall weighted 2× more than precision

**Formula**: F_β = (1 + β²) × (precision × recall) / (β² × precision + recall)

## Specificity and NPV

```julia
specificity(actual, predicted_labels)
npv(actual, predicted_labels)
```

**Relationships**:
- Sensitivity (recall) + FNR = 1
- Specificity + FPR = 1
- Precision + FDR = 1
- NPV + FOR = 1

## Error Rates

```julia
fpr(actual, predicted_labels)  # False Positive Rate
fnr(actual, predicted_labels)  # False Negative Rate
```

## Combined Metrics

### Youden's J (Informedness)

```julia
youden_j(actual, predicted_labels)
```

**Use case**: Finding optimal threshold that maximizes sensitivity + specificity.

### Markedness

```julia
markedness(actual, predicted_labels)
```

**Interpretation**: How marked (informative) are positive and negative predictions?

### Fowlkes-Mallows Index

```julia
fowlkes_mallows_index(actual, predicted_labels)
```

## Likelihood Ratios (Medical/Diagnostic)

```julia
positive_likelihood_ratio(actual, predicted_labels)
negative_likelihood_ratio(actual, predicted_labels)
diagnostic_odds_ratio(actual, predicted_labels)
```

**Interpretation of LR+**:
- LR+ > 10: Strong evidence for positive
- LR+ = 5-10: Moderate evidence
- LR+ = 2-5: Weak evidence
- LR+ = 1: Useless test

**Interpretation of LR-**:
- LR- < 0.1: Strong evidence for negative
- LR- = 0.1-0.2: Moderate evidence
- LR- = 0.2-0.5: Weak evidence
- LR- = 1: Useless test

## Business Metrics

### Lift

```julia
lift(actual, predicted_probs; percentile=0.1)
```

**Interpretation**: How many times better than random in the top X%?
- Lift = 3 in top 10%: 3× more positives than random

### Gain

```julia
gain(actual, predicted_probs; percentile=0.1)
```

**Interpretation**: What percentage of all positives are captured in top X%?

## Usage Examples

### Complete Binary Classification Evaluation

```julia
using UnifiedMetrics

actual = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
predicted_probs = [0.9, 0.8, 0.7, 0.3, 0.6, 0.4, 0.35, 0.2, 0.1, 0.05]
predicted_labels = predicted_probs .>= 0.5

println("=== Threshold-Independent ===")
println("AUC: ", round(auc(actual, predicted_probs), digits=3))
println("Gini: ", round(gini_coefficient(actual, predicted_probs), digits=3))
println("KS Statistic: ", round(ks_statistic(actual, predicted_probs), digits=3))
println("Brier Score: ", round(brier_score(actual, predicted_probs), digits=3))
println("Log Loss: ", round(logloss(actual, predicted_probs), digits=3))

println("\n=== Threshold-Dependent (threshold=0.5) ===")
println("Precision: ", round(precision(actual, predicted_labels), digits=3))
println("Recall: ", round(recall(actual, predicted_labels), digits=3))
println("F1 Score: ", round(fbeta_score(actual, predicted_labels), digits=3))
println("Specificity: ", round(specificity(actual, predicted_labels), digits=3))
println("MCC: ", round(mcc(actual, predicted_labels), digits=3))
```

### Comparing Different Thresholds

```julia
using UnifiedMetrics

actual = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
probs = [0.9, 0.8, 0.7, 0.3, 0.6, 0.4, 0.35, 0.2, 0.1, 0.05]

for threshold in [0.3, 0.5, 0.7]
    labels = probs .>= threshold
    println("Threshold: $threshold")
    println("  Precision: $(round(precision(actual, labels), digits=2))")
    println("  Recall: $(round(recall(actual, labels), digits=2))")
    println("  F1: $(round(fbeta_score(actual, labels), digits=2))")
    println("  Youden's J: $(round(youden_j(actual, labels), digits=2))")
end
```

### Medical Diagnostic Evaluation

```julia
using UnifiedMetrics

# Disease screening results
actual = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 = has disease
predicted = [1, 1, 1, 0, 0, 0, 0, 0, 1, 0]  # Test results

println("=== Diagnostic Performance ===")
println("Sensitivity: ", round(sensitivity(actual, predicted), digits=3))
println("Specificity: ", round(specificity(actual, predicted), digits=3))
println("PPV (Precision): ", round(precision(actual, predicted), digits=3))
println("NPV: ", round(npv(actual, predicted), digits=3))
println("LR+: ", round(positive_likelihood_ratio(actual, predicted), digits=2))
println("LR-: ", round(negative_likelihood_ratio(actual, predicted), digits=2))
println("DOR: ", round(diagnostic_odds_ratio(actual, predicted), digits=1))
```

### Marketing/Business Application

```julia
using UnifiedMetrics

# Customer response prediction
actual_responded = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
predicted_scores = [0.9, 0.8, 0.3, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1, 0.05]

println("=== Campaign Targeting ===")
for pct in [0.1, 0.2, 0.3, 0.5]
    println("Top $(Int(pct*100))%:")
    println("  Lift: $(round(lift(actual_responded, predicted_scores, percentile=pct), digits=2))x")
    println("  Gain: $(round(gain(actual_responded, predicted_scores, percentile=pct)*100, digits=1))%")
end
```

### Handling Imbalanced Data

```julia
using UnifiedMetrics

# Highly imbalanced: 95% negative, 5% positive
actual = vcat(fill(0, 95), fill(1, 5))
predicted = vcat(fill(0, 100))  # Naive: always predict negative

println("=== Naive Model on Imbalanced Data ===")
println("Accuracy: ", accuracy(actual, predicted))  # 0.95 - misleading!
println("Recall: ", recall(actual, predicted))       # 0.0 - reveals the problem
println("MCC: ", mcc(actual, predicted))             # 0.0 - correctly shows failure

# A better model
predicted_better = vcat(fill(0, 90), fill(1, 5), fill(0, 3), fill(1, 2))
println("\n=== Better Model ===")
println("Accuracy: ", accuracy(actual, predicted_better))
println("Recall: ", recall(actual, predicted_better))
println("Precision: ", precision(actual, predicted_better))
println("MCC: ", round(mcc(actual, predicted_better), digits=3))
```

See the [API Reference](@ref) for complete function documentation.
