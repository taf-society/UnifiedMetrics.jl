# API Reference

Complete reference for all functions in UnifiedMetrics.jl.

## Regression Metrics

### Error Metrics

```@docs
ae
mae
mdae
se
sse
mse
rmse
nrmse
max_error
max_ae
```

### Bias Metrics

```@docs
bias
percent_bias
mpe
```

### Percentage Error Metrics

```@docs
ape
mape
smape
wmape
```

### Logarithmic Error Metrics

```@docs
sle
msle
rmsle
```

### Relative Error Metrics

```@docs
rse
rrse
rae
```

### Explained Variance

```@docs
explained_variation
adjusted_r2
```

### Robust Loss Functions

```@docs
huber_loss
log_cosh_loss
```

### Quantile Loss

```@docs
quantile_loss
pinball_loss
```

### GLM Deviance Metrics

```@docs
tweedie_deviance
mean_poisson_deviance
mean_gamma_deviance
d2_tweedie_score
```

## Classification Metrics

### Accuracy Metrics

```@docs
accuracy
ce
balanced_accuracy
```

### Agreement Metrics

```@docs
cohens_kappa
ScoreQuadraticWeightedKappa
MeanQuadraticWeightedKappa
```

### Correlation Metrics

```@docs
matthews_corrcoef
mcc
```

### Confusion Matrix

```@docs
confusion_matrix
```

### Top-K Metrics

```@docs
top_k_accuracy
```

### Loss Functions

```@docs
hamming_loss
zero_one_loss
hinge_loss
squared_hinge_loss
```

## Binary Classification Metrics

### ROC-Based Metrics

```@docs
auc
gini_coefficient
ks_statistic
```

### Probability Metrics

```@docs
ll
logloss
brier_score
```

### Precision and Recall

```@docs
precision
recall
sensitivity
specificity
npv
```

### F-Score

```@docs
fbeta_score
```

### Error Rates

```@docs
fpr
fnr
```

### Combined Metrics

```@docs
youden_j
markedness
fowlkes_mallows_index
```

### Likelihood Ratios

```@docs
positive_likelihood_ratio
negative_likelihood_ratio
diagnostic_odds_ratio
```

### Business Metrics

```@docs
lift
gain
```

## Information Retrieval Metrics

### Set-Based Metrics

```@docs
f1
precision_at_k
recall_at_k
f1_at_k
```

### Ranking Metrics

```@docs
dcg
idcg
ndcg
mean_ndcg
```

### Average Precision

```@docs
apk
mapk
```

### Reciprocal Rank

```@docs
reciprocal_rank
mrr
```

### Hit Rate

```@docs
hit_rate
```

### Recommendation Metrics

```@docs
coverage
novelty
```

## Time Series Metrics

### Scaled Error Metrics

```@docs
mase
msse
rmsse
```

### Bias Metrics

```@docs
tracking_signal
forecast_bias
```

### Benchmark Comparison

```@docs
theil_u1
theil_u2
```

### Percentage Metrics

```@docs
wape
```

### Directional Metrics

```@docs
directional_accuracy
```

### Prediction Interval Metrics

```@docs
coverage_probability
pinball_loss_series
winkler_score
```

### Autocorrelation Metrics

```@docs
autocorrelation_error
```

## Index

```@index
```
