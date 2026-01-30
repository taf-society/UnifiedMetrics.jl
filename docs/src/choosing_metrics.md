# Choosing the Right Metric

This guide helps you select the appropriate metric for your machine learning task. The right metric depends on your problem type, data characteristics, and business requirements.

## Quick Decision Guide

### What type of problem are you solving?

| Problem Type | Go to Section |
|-------------|---------------|
| Predicting continuous values (prices, temperatures, etc.) | [Regression Metrics](@ref regression-guide) |
| Predicting categories (multi-class) | [Classification Metrics](@ref classification-guide) |
| Predicting yes/no outcomes | [Binary Classification Metrics](@ref binary-guide) |
| Ranking items (search, recommendations) | [Information Retrieval Metrics](@ref ir-guide) |
| Predicting future values in a sequence | [Time Series Metrics](@ref ts-guide) |

---

## [Regression Metrics](@id regression-guide)

### Decision Flowchart

```
START: Regression Problem
    |
    v
Do you need interpretable units?
    |
    +-- YES --> Do you want to penalize large errors more?
    |               |
    |               +-- YES --> Use RMSE
    |               |
    |               +-- NO --> Use MAE
    |
    +-- NO --> Do you need scale-independent comparison?
                    |
                    +-- YES --> Are there zeros in actual values?
                    |               |
                    |               +-- YES --> Use SMAPE or WMAPE
                    |               |
                    |               +-- NO --> Use MAPE
                    |
                    +-- NO --> Use R² (explained_variation)
```

### When to Use Each Metric

| Metric | Use When | Avoid When |
|--------|----------|------------|
| **MAE** | You want average error in original units; outliers should not dominate | You need to heavily penalize large errors |
| **RMSE** | Large errors are particularly bad; you want same units as target | Outliers are present and acceptable |
| **MAPE** | You need percentage errors for stakeholder communication | Actual values contain zeros or near-zeros |
| **SMAPE** | You need percentage errors and have zeros | You need asymmetric error treatment |
| **R²** | You want to know proportion of variance explained | Comparing models on different datasets |
| **MASE** | Comparing forecasts across different scales | Non-time-series data |

### Detailed Recommendations

#### For General Regression Tasks

**Primary metric**: `rmse` or `mae`
- Use `rmse` when large errors are costly (e.g., predicting house prices where a \$100K error is much worse than ten \$10K errors)
- Use `mae` when all errors matter equally (e.g., predicting delivery times)

```julia
# Standard evaluation
rmse(actual, predicted)  # Penalizes large errors
mae(actual, predicted)   # Treats all errors equally
```

#### For Percentage-Based Reporting

**Primary metric**: `mape`, `smape`, or `wmape`

```julia
# When actual values are always positive and non-zero
mape(actual, predicted)

# When actual values may be zero
smape(actual, predicted)  # Symmetric, bounded [0, 2]
wmape(actual, predicted)  # Weighted by actuals

# To detect systematic bias
mpe(actual, predicted)  # Positive = under-prediction
```

#### For Model Comparison

**Primary metric**: `explained_variation` (R²) or `adjusted_r2`

```julia
# Basic R²
explained_variation(actual, predicted)  # 1 = perfect, 0 = mean baseline

# When comparing models with different numbers of features
adjusted_r2(actual, predicted, n_features)
```

#### For Robust Models (Outlier-Resistant)

**Primary metric**: `huber_loss` or `mdae`

```julia
# Huber loss: quadratic for small errors, linear for large
huber_loss(actual, predicted, delta=1.0)

# Median Absolute Error: robust to outliers
mdae(actual, predicted)
```

#### For Skewed Target Variables

**Primary metric**: `rmsle` or `msle`

```julia
# For targets spanning multiple orders of magnitude (prices, populations)
rmsle(actual, predicted)  # Penalizes under-prediction more
```

#### For Count Data or GLMs

**Primary metric**: `mean_poisson_deviance` or `mean_gamma_deviance`

```julia
# For count data (website visits, number of purchases)
mean_poisson_deviance(actual, predicted)

# For positive continuous data with variance ~ mean²
mean_gamma_deviance(actual, predicted)
```

---

## [Classification Metrics](@id classification-guide)

### Decision Flowchart

```
START: Multi-class Classification
    |
    v
Is your dataset balanced?
    |
    +-- YES --> Use accuracy() or ce()
    |
    +-- NO --> Use balanced_accuracy() or cohens_kappa()
                    |
                    v
               Do you need a single summary metric?
                    |
                    +-- YES --> For binary: mcc()
                    |           For ordinal: ScoreQuadraticWeightedKappa()
                    |
                    +-- NO --> Use confusion_matrix() for detailed analysis
```

### When to Use Each Metric

| Metric | Use When | Avoid When |
|--------|----------|------------|
| **accuracy** | Classes are balanced; simple reporting needed | Imbalanced datasets |
| **balanced_accuracy** | Classes are imbalanced | You need per-class details |
| **cohens_kappa** | You want to account for chance agreement | N/A |
| **mcc** | Binary classification; best single metric | Multi-class (use macro-averaged) |
| **confusion_matrix** | You need detailed error analysis | Simple summary is sufficient |

### Detailed Recommendations

#### For Balanced Datasets

```julia
accuracy(actual, predicted)  # Simple and interpretable
```

#### For Imbalanced Datasets

```julia
# Macro-averaged recall across classes
balanced_accuracy(actual, predicted)

# Accounts for chance agreement
cohens_kappa(actual, predicted)
```

#### For Ordinal Classification

When classes have a natural order (e.g., ratings 1-5):

```julia
# Penalizes predictions farther from true class
ScoreQuadraticWeightedKappa(actual, predicted, min_rating=1, max_rating=5)
```

#### For Multi-Label Classification

```julia
# Fraction of incorrect labels
hamming_loss(actual_matrix, predicted_matrix)
```

---

## [Binary Classification Metrics](@id binary-guide)

### Decision Flowchart

```
START: Binary Classification
    |
    v
What type of predictions do you have?
    |
    +-- Probabilities (0-1) --> Do you need threshold-independent evaluation?
    |                               |
    |                               +-- YES --> Use auc() or gini_coefficient()
    |                               |
    |                               +-- NO --> What matters more?
    |                                               |
    |                                               +-- Calibration --> brier_score() or logloss()
    |                                               |
    |                                               +-- Ranking --> ks_statistic()
    |
    +-- Binary Labels (0/1) --> What is your priority?
                                    |
                                    +-- Balance precision/recall --> fbeta_score()
                                    |
                                    +-- Minimize false positives --> precision()
                                    |
                                    +-- Minimize false negatives --> recall()
                                    |
                                    +-- Single best metric --> mcc()
```

### When to Use Each Metric

| Metric | Use When | Avoid When |
|--------|----------|------------|
| **auc** | Comparing models; threshold hasn't been chosen | You need a specific operating point |
| **precision** | False positives are costly (spam detection) | Missing positives is worse |
| **recall** | False negatives are costly (disease detection) | False alarms are problematic |
| **fbeta_score** | You need to balance precision and recall | Clear priority for one over other |
| **mcc** | Imbalanced data; need single summary metric | You need threshold-independent metric |
| **brier_score** | Probability calibration matters | Ranking is more important |

### Detailed Recommendations

#### For Model Selection (Before Choosing Threshold)

```julia
# Area Under ROC Curve - threshold independent
auc(actual, predicted_scores)

# Gini coefficient (= 2*AUC - 1)
gini_coefficient(actual, predicted_scores)

# Maximum separation between classes
ks_statistic(actual, predicted_scores)
```

#### For Probability Calibration

When you need well-calibrated probabilities:

```julia
# Mean squared error of probabilities
brier_score(actual, predicted_probs)  # Lower is better

# Cross-entropy loss
logloss(actual, predicted_probs)  # Lower is better
```

#### For Threshold-Based Evaluation

After choosing a classification threshold:

```julia
# Convert probabilities to labels
predicted_labels = predicted_probs .>= threshold

# When false positives are costly (spam filter, fraud detection)
precision(actual, predicted_labels)

# When false negatives are costly (disease screening, security threats)
recall(actual, predicted_labels)
sensitivity(actual, predicted_labels)  # Same as recall

# Balanced metric
fbeta_score(actual, predicted_labels)         # F1: equal weight
fbeta_score(actual, predicted_labels, beta=0.5)  # Favor precision
fbeta_score(actual, predicted_labels, beta=2.0)  # Favor recall
```

#### For Medical/Diagnostic Applications

```julia
# Sensitivity (true positive rate)
sensitivity(actual, predicted_labels)

# Specificity (true negative rate)
specificity(actual, predicted_labels)

# Youden's J (optimal threshold criterion)
youden_j(actual, predicted_labels)

# Likelihood ratios for clinical decision making
positive_likelihood_ratio(actual, predicted_labels)
negative_likelihood_ratio(actual, predicted_labels)
diagnostic_odds_ratio(actual, predicted_labels)
```

#### For Imbalanced Data

The single best metric for binary classification with imbalanced data:

```julia
# Matthews Correlation Coefficient: accounts for all quadrants of confusion matrix
mcc(actual, predicted_labels)  # Range: [-1, 1], 0 = random
```

#### For Business Applications

```julia
# Lift: how much better than random in top X%
lift(actual, predicted_scores, percentile=0.1)

# Gain: what % of positives captured in top X%
gain(actual, predicted_scores, percentile=0.1)
```

---

## [Information Retrieval Metrics](@id ir-guide)

### Decision Flowchart

```
START: Ranking/Retrieval Problem
    |
    v
Do you have graded relevance scores?
    |
    +-- YES --> Use ndcg() or dcg()
    |
    +-- NO (binary relevance) --> What matters more?
                                      |
                                      +-- Finding first relevant item --> mrr()
                                      |
                                      +-- Finding all relevant items --> recall_at_k()
                                      |
                                      +-- Precision of top results --> precision_at_k()
                                      |
                                      +-- Balance of both --> f1_at_k() or mapk()
```

### When to Use Each Metric

| Metric | Use When | Avoid When |
|--------|----------|------------|
| **ndcg** | Relevance is graded (0-5 stars) | Binary relevance only |
| **mrr** | Only first relevant result matters | All relevant items matter |
| **map@k** | Ranking quality across positions matters | Only top-1 or top-k matters |
| **recall@k** | Coverage of relevant items is priority | Precision matters more |
| **precision@k** | Quality of top results is priority | Missing relevant items is costly |
| **hit_rate** | At least one relevant in top-k is success | Need finer granularity |

### Detailed Recommendations

#### For Search Engines

```julia
# Graded relevance (best for search)
ndcg(relevance_scores, k=10)

# Mean NDCG across queries
mean_ndcg(relevances_list, k=10)

# Mean Reciprocal Rank (how quickly users find what they want)
mrr(actual_list, predicted_list)
```

#### For Recommendation Systems

```julia
# Did we show at least one good item?
hit_rate(actual_list, predicted_list, k=10)

# How many relevant items did we show?
recall_at_k(actual, predicted, k=10)

# What fraction of shown items are relevant?
precision_at_k(actual, predicted, k=10)

# Balanced metric
f1_at_k(actual, predicted, k=10)

# Catalog coverage (diversity)
coverage(predicted_list, full_catalog)

# Novelty (recommending non-obvious items)
novelty(predicted_list, item_popularity)
```

#### For E-commerce / Product Search

```julia
# Average precision at k
apk(10, relevant_products, retrieved_products)

# Mean AP across queries
mapk(10, relevant_lists, retrieved_lists)
```

---

## [Time Series Metrics](@id ts-guide)

### Decision Flowchart

```
START: Time Series Forecasting
    |
    v
What aspect of forecast quality matters?
    |
    +-- Point forecast accuracy --> Is scale-independent comparison needed?
    |                                   |
    |                                   +-- YES --> mase() or theil_u2()
    |                                   |
    |                                   +-- NO --> rmse() or mae()
    |
    +-- Directional accuracy --> directional_accuracy()
    |
    +-- Forecast bias --> tracking_signal() or forecast_bias()
    |
    +-- Prediction intervals --> coverage_probability() or winkler_score()
```

### When to Use Each Metric

| Metric | Use When | Avoid When |
|--------|----------|------------|
| **mase** | Comparing across series with different scales | Single series evaluation |
| **rmsse** | Scale-independent; sensitive to large errors | Outliers acceptable |
| **tracking_signal** | Monitoring for systematic bias | One-time evaluation |
| **directional_accuracy** | Direction matters more than magnitude | Magnitude accuracy critical |
| **winkler_score** | Evaluating prediction intervals | Point forecasts only |
| **theil_u2** | Comparing to naive benchmark | Absolute accuracy needed |

### Detailed Recommendations

#### For Comparing Forecasts Across Different Series

The M-competition recommended metrics:

```julia
# Mean Absolute Scaled Error (most recommended)
mase(actual, predicted, m=1)     # Non-seasonal
mase(actual, predicted, m=12)    # Monthly data with yearly seasonality
mase(actual, predicted, m=7)     # Daily data with weekly seasonality

# Root Mean Squared Scaled Error
rmsse(actual, predicted, m=1)
```

**Interpretation**:
- MASE < 1: Better than naive forecast
- MASE = 1: Same as naive forecast
- MASE > 1: Worse than naive forecast

#### For Single Series Evaluation

```julia
# Standard metrics in original units
mae(actual, predicted)
rmse(actual, predicted)

# Percentage-based (avoid if zeros present)
mape(actual, predicted)
wape(actual, predicted)  # Handles zeros better
```

#### For Detecting Forecast Bias

```julia
# Normalized measure of cumulative error
tracking_signal(actual, predicted)
# Interpretation: values outside [-4, 4] indicate systematic bias

# Simple bias (positive = under-forecasting)
forecast_bias(actual, predicted)
```

#### For Comparing to Benchmark

```julia
# Theil's U2: comparison to naive forecast
theil_u2(actual, predicted, m=1)
# < 1: better than naive, > 1: worse than naive

# Theil's U1: normalized error
theil_u1(actual, predicted)
# 0 = perfect, 1 = worst
```

#### For Direction Prediction (Trading, etc.)

```julia
# What fraction of up/down movements were predicted correctly?
directional_accuracy(actual, predicted)
```

#### For Probabilistic Forecasts / Prediction Intervals

```julia
# Does the interval contain the actual value at expected rate?
coverage_probability(actual, lower, upper)
# Should match your confidence level (e.g., 0.95 for 95% intervals)

# Interval score (rewards narrow intervals, penalizes misses)
winkler_score(actual, lower, upper, alpha=0.05)

# Quantile forecast evaluation
pinball_loss_series(actual, predicted_quantile, quantile=0.9)
```

#### For Preserving Temporal Structure

```julia
# Does the forecast maintain autocorrelation patterns?
autocorrelation_error(actual, predicted, max_lag=10)
```

---

## Common Mistakes to Avoid

### Regression

1. **Using MAPE with zeros**: MAPE is undefined when actual values are zero. Use SMAPE or WMAPE instead.
2. **Ignoring scale**: When comparing models across different datasets, use scale-independent metrics (R², MAPE, MASE).
3. **Only using R²**: R² can be misleading for non-linear relationships. Always check residual plots.

### Classification

1. **Using accuracy on imbalanced data**: A model predicting the majority class always achieves high accuracy. Use balanced_accuracy, MCC, or per-class metrics.
2. **Optimizing for wrong metric**: If false negatives are costly (medical diagnosis), optimize for recall, not precision.

### Binary Classification

1. **Comparing AUC across very different datasets**: AUC can be misleading if class distributions differ significantly.
2. **Ignoring calibration**: High AUC doesn't mean probabilities are well-calibrated. Check Brier score.
3. **Using accuracy on imbalanced data**: Use MCC instead.

### Information Retrieval

1. **Using NDCG with binary relevance**: While valid, simpler metrics (MAP, MRR) may be more interpretable.
2. **Ignoring position**: Metrics like precision don't account for ranking. Use NDCG or MRR.

### Time Series

1. **Not using scaled metrics**: Raw MAE/RMSE can't be compared across series with different scales.
2. **Ignoring seasonality in MASE**: Set `m` to match your data's seasonal period.
3. **Only checking point accuracy**: Also evaluate bias (tracking_signal) and intervals (coverage_probability).

---

## Metric Selection Summary Table

| Scenario | Recommended Metric | Alternative |
|----------|-------------------|-------------|
| General regression | RMSE | MAE |
| Regression with outliers | Huber loss | MdAE |
| Stakeholder reporting | MAPE (if no zeros) | SMAPE |
| Imbalanced binary classification | MCC | Balanced accuracy |
| Medical diagnosis | Sensitivity + Specificity | Youden's J |
| Search ranking | NDCG | MRR |
| Recommendation system | Hit rate, Recall@k | MAP@k |
| Forecast comparison | MASE | RMSSE |
| Forecast monitoring | Tracking signal | Forecast bias |
| Prediction intervals | Coverage + Winkler | Pinball loss |
