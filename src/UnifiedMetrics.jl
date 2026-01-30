module UnifiedMetrics

using Statistics
using StatsBase: ordinalrank

include("regression.jl")
include("classification.jl")
include("binary_classification.jl")
include("information_retrieval.jl")
include("time_series.jl")

export bias, percent_bias
export se, sse, mse, rmse
export ae, mae, mdae
export ape, mape, smape
export sle, msle, rmsle
export rse, rrse, rae
export explained_variation
export max_error, max_ae
export huber_loss, log_cosh_loss
export quantile_loss, pinball_loss
export nrmse, adjusted_r2
export mpe, wmape
export tweedie_deviance, mean_gamma_deviance, mean_poisson_deviance
export d2_tweedie_score

export ce, accuracy
export ScoreQuadraticWeightedKappa, MeanQuadraticWeightedKappa
export balanced_accuracy
export cohens_kappa
export matthews_corrcoef, mcc
export confusion_matrix
export top_k_accuracy
export hamming_loss, zero_one_loss
export hinge_loss, squared_hinge_loss

export auc
export ll, logloss
export precision, recall, fbeta_score
export sensitivity, specificity
export npv, fpr, fnr
export brier_score
export gini_coefficient
export ks_statistic
export lift, gain
export youden_j, markedness
export fowlkes_mallows_index
export positive_likelihood_ratio, negative_likelihood_ratio
export diagnostic_odds_ratio

export f1, apk, mapk
export dcg, idcg, ndcg, mean_ndcg
export mrr, reciprocal_rank
export hit_rate, recall_at_k, precision_at_k, f1_at_k
export coverage, novelty

export mase
export msse, rmsse
export tracking_signal, forecast_bias
export theil_u1, theil_u2
export wape
export directional_accuracy
export coverage_probability
export pinball_loss_series
export winkler_score
export autocorrelation_error

end
