using UnifiedMetrics
using Test
using Statistics

@testset "UnifiedMetrics.jl" begin

    @testset "Regression Metrics" begin
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [1.1, 2.0, 2.9, 4.2, 4.8]

        @testset "Absolute Error" begin
            @test ae(actual, predicted) ≈ [0.1, 0.0, 0.1, 0.2, 0.2]
            @test mae(actual, predicted) ≈ 0.12
            @test mdae(actual, predicted) ≈ 0.1
        end

        @testset "Squared Error" begin
            @test se(actual, predicted) ≈ [0.01, 0.0, 0.01, 0.04, 0.04]
            @test sse(actual, predicted) ≈ 0.1
            @test mse(actual, predicted) ≈ 0.02
            @test rmse(actual, predicted) ≈ sqrt(0.02)
        end

        @testset "Bias" begin
            @test bias(actual, predicted) ≈ 0.0
            # Test with biased predictions
            biased_pred = [0.9, 1.8, 2.8, 3.8, 4.8]
            @test bias(actual, biased_pred) ≈ 0.18 atol=1e-10  # Under-predicting
        end

        @testset "Percent Bias" begin
            actual_pb = [10.0, 20.0, 30.0, 40.0, 50.0]
            predicted_pb = [9.0, 19.0, 29.0, 39.0, 49.0]
            @test percent_bias(actual_pb, predicted_pb) ≈ mean([1/10, 1/20, 1/30, 1/40, 1/50])
        end

        @testset "Percentage Errors" begin
            actual_pct = [10.0, 20.0, 30.0, 40.0, 50.0]
            predicted_pct = [9.0, 22.0, 27.0, 44.0, 48.0]

            @test length(ape(actual_pct, predicted_pct)) == 5
            @test mape(actual_pct, predicted_pct) ≈ mean(abs.(actual_pct .- predicted_pct) ./ actual_pct)
            @test smape(actual_pct, predicted_pct) >= 0
            @test smape(actual_pct, predicted_pct) <= 2
        end

        @testset "SMAPE symmetry" begin
            a = [1.0, 2.0, 3.0]
            b = [2.0, 3.0, 4.0]
            @test smape(a, b) ≈ smape(b, a)
        end

        @testset "WMAPE" begin
            actual_w = [100.0, 200.0, 300.0]
            predicted_w = [110.0, 190.0, 310.0]
            expected_wmape = sum(abs.(actual_w .- predicted_w)) / sum(abs.(actual_w))
            @test wmape(actual_w, predicted_w) ≈ expected_wmape
        end

        @testset "MPE" begin
            actual_mpe = [100.0, 200.0, 300.0]
            predicted_mpe = [90.0, 210.0, 290.0]
            @test mpe(actual_mpe, predicted_mpe) ≈ mean((actual_mpe .- predicted_mpe) ./ actual_mpe) * 100
        end

        @testset "Log Errors" begin
            actual_log = [1.0, 2.0, 3.0, 4.0, 5.0]
            predicted_log = [1.1, 2.1, 2.9, 4.2, 4.8]

            @test all(sle(actual_log, predicted_log) .>= 0)
            @test msle(actual_log, predicted_log) >= 0
            @test rmsle(actual_log, predicted_log) ≈ sqrt(msle(actual_log, predicted_log))
        end

        @testset "Relative Errors" begin
            @test rse(actual, predicted) >= 0
            @test rrse(actual, predicted) ≈ sqrt(rse(actual, predicted))
            @test rae(actual, predicted) >= 0
        end

        @testset "Explained Variation (R²)" begin
            # Perfect prediction
            @test explained_variation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) ≈ 1.0

            # Mean prediction (R² = 0)
            actual_r2 = [1.0, 2.0, 3.0, 4.0, 5.0]
            mean_pred = fill(mean(actual_r2), 5)
            @test explained_variation(actual_r2, mean_pred) ≈ 0.0 atol=1e-10

            # Good prediction
            @test explained_variation(actual, predicted) > 0.9
        end

        @testset "Adjusted R²" begin
            @test adjusted_r2(actual, predicted, 1) <= explained_variation(actual, predicted)
            @test adjusted_r2(actual, predicted, 2) < adjusted_r2(actual, predicted, 1)
        end

        @testset "Max Error" begin
            @test max_error(actual, predicted) ≈ 0.2
            @test max_ae(actual, predicted) ≈ max_error(actual, predicted)
        end

        @testset "NRMSE" begin
            @test nrmse(actual, predicted, normalization=:range) ≈ rmse(actual, predicted) / (maximum(actual) - minimum(actual))
            @test nrmse(actual, predicted, normalization=:mean) ≈ rmse(actual, predicted) / mean(actual)
            @test nrmse(actual, predicted, normalization=:std) ≈ rmse(actual, predicted) / std(actual)
            @test nrmse(actual, predicted, normalization=:iqr) >= 0
        end

        @testset "Huber Loss" begin
            @test huber_loss(actual, predicted) >= 0
            @test huber_loss(actual, predicted, delta=0.5) >= 0
            @test huber_loss(actual, predicted, delta=10.0) ≈ mse(actual, predicted) / 2 atol=1e-6

            # With outlier
            actual_outlier = [1.0, 2.0, 3.0, 4.0, 100.0]
            predicted_outlier = [1.0, 2.0, 3.0, 4.0, 5.0]
            @test huber_loss(actual_outlier, predicted_outlier, delta=1.0) < mse(actual_outlier, predicted_outlier) / 2
        end

        @testset "Log-Cosh Loss" begin
            @test log_cosh_loss(actual, predicted) >= 0
            @test log_cosh_loss([1.0, 2.0], [1.0, 2.0]) ≈ 0.0 atol=1e-10
        end

        @testset "Quantile Loss" begin
            @test quantile_loss(actual, predicted, quantile=0.5) >= 0
            @test quantile_loss(actual, predicted, quantile=0.9) >= 0
            @test quantile_loss(actual, predicted, quantile=0.1) >= 0

            # Pinball is alias
            @test pinball_loss(actual, predicted, quantile=0.5) ≈ quantile_loss(actual, predicted, quantile=0.5)
        end

        @testset "Tweedie Deviance" begin
            actual_tw = [1.0, 2.0, 3.0, 4.0, 5.0]
            predicted_tw = [1.1, 2.1, 2.9, 4.2, 4.8]

            @test tweedie_deviance(actual_tw, predicted_tw, power=0) >= 0
            @test tweedie_deviance(actual_tw, predicted_tw, power=1) >= 0
            @test tweedie_deviance(actual_tw, predicted_tw, power=1.5) >= 0
            @test tweedie_deviance(actual_tw, predicted_tw, power=2) >= 0

            @test mean_poisson_deviance(actual_tw, predicted_tw) ≈ tweedie_deviance(actual_tw, predicted_tw, power=1)
            @test mean_gamma_deviance(actual_tw, predicted_tw) ≈ tweedie_deviance(actual_tw, predicted_tw, power=2)
        end

        @testset "D² Tweedie Score" begin
            actual_d2 = [1.0, 2.0, 3.0, 4.0, 5.0]
            predicted_d2 = [1.1, 2.0, 2.9, 4.1, 4.9]
            @test d2_tweedie_score(actual_d2, predicted_d2, power=1.5) <= 1.0
        end
    end

    @testset "Classification Metrics" begin
        @testset "Accuracy and Error" begin
            actual = [1, 1, 0, 0, 1, 0]
            predicted = [1, 0, 0, 1, 1, 0]

            @test accuracy(actual, predicted) ≈ 4/6
            @test ce(actual, predicted) ≈ 2/6
            @test accuracy(actual, predicted) + ce(actual, predicted) ≈ 1.0
        end

        @testset "String Labels" begin
            actual_str = ["cat", "dog", "cat", "cat", "dog"]
            predicted_str = ["cat", "dog", "dog", "cat", "cat"]

            @test accuracy(actual_str, predicted_str) ≈ 3/5
            @test ce(actual_str, predicted_str) ≈ 2/5
        end

        @testset "Balanced Accuracy" begin
            # Imbalanced dataset
            actual_imb = [1, 1, 1, 1, 0, 0]
            predicted_imb = [1, 1, 1, 0, 0, 0]

            @test balanced_accuracy(actual_imb, predicted_imb) ≈ (3/4 + 2/2) / 2
        end

        @testset "Cohen's Kappa" begin
            actual_k = [1, 1, 1, 0, 0, 0]
            predicted_k = [1, 1, 0, 0, 0, 1]

            @test cohens_kappa(actual_k, predicted_k) >= -1
            @test cohens_kappa(actual_k, predicted_k) <= 1

            # Perfect agreement
            @test cohens_kappa([1, 2, 3], [1, 2, 3]) ≈ 1.0
        end

        @testset "Matthews Correlation Coefficient" begin
            actual_mcc = [1, 1, 1, 0, 0, 0]
            predicted_mcc = [1, 0, 1, 1, 0, 0]

            @test matthews_corrcoef(actual_mcc, predicted_mcc) >= -1
            @test matthews_corrcoef(actual_mcc, predicted_mcc) <= 1
            @test mcc(actual_mcc, predicted_mcc) ≈ matthews_corrcoef(actual_mcc, predicted_mcc)

            # Perfect prediction
            @test mcc([1, 1, 0, 0], [1, 1, 0, 0]) ≈ 1.0
        end

        @testset "Confusion Matrix" begin
            actual_cm = [1, 1, 1, 0, 0, 0]
            predicted_cm = [1, 0, 1, 1, 0, 0]

            cm = confusion_matrix(actual_cm, predicted_cm)
            @test haskey(cm, :matrix)
            @test haskey(cm, :labels)
            @test size(cm[:matrix]) == (2, 2)
            @test sum(cm[:matrix]) == 6
        end

        @testset "Top-K Accuracy" begin
            actual_topk = [1, 2, 3, 1]
            predicted_probs = [0.7 0.2 0.1;
                               0.2 0.5 0.3;
                               0.1 0.3 0.6;
                               0.3 0.4 0.3]

            @test top_k_accuracy(actual_topk, predicted_probs, 1) ≈ 3/4
            @test top_k_accuracy(actual_topk, predicted_probs, 2) ≈ 1.0
            @test top_k_accuracy(actual_topk, predicted_probs, 3) ≈ 1.0
        end

        @testset "Quadratic Weighted Kappa" begin
            rater_a = [1, 2, 3, 4, 5, 3, 2, 4]
            rater_b = [1, 2, 2, 4, 4, 3, 3, 5]

            qwk = ScoreQuadraticWeightedKappa(rater_a, rater_b, min_rating=1, max_rating=5)
            @test qwk >= -1
            @test qwk <= 1
        end

        @testset "Mean Quadratic Weighted Kappa" begin
            kappas = [0.5, 0.6, 0.7, 0.8]
            mqwk = MeanQuadraticWeightedKappa(kappas)
            @test mqwk >= -1
            @test mqwk <= 1

            # With weights
            weights = [1.0, 2.0, 1.0, 1.0]
            mqwk_weighted = MeanQuadraticWeightedKappa(kappas, weights=weights)
            @test mqwk_weighted >= -1
            @test mqwk_weighted <= 1
        end

        @testset "Hamming Loss" begin
            actual_h = [1, 2, 3, 4]
            predicted_h = [1, 2, 2, 4]
            @test hamming_loss(actual_h, predicted_h) ≈ 1/4
        end

        @testset "Zero-One Loss" begin
            actual_zo = [1, 2, 3, 4]
            predicted_zo = [1, 2, 2, 4]
            @test zero_one_loss(actual_zo, predicted_zo) ≈ 1/4
        end

        @testset "Hinge Loss" begin
            actual_hl = [1, 1, -1, -1]
            predicted_hl = [0.8, 0.3, -0.5, 0.1]

            @test hinge_loss(actual_hl, predicted_hl) >= 0
            @test squared_hinge_loss(actual_hl, predicted_hl) >= 0
        end
    end

    @testset "Binary Classification Metrics" begin
        actual = [1, 1, 1, 0, 0, 0]
        predicted_scores = [0.9, 0.8, 0.4, 0.5, 0.3, 0.2]
        predicted_labels = [1, 1, 0, 1, 0, 0]

        @testset "AUC" begin
            @test auc(actual, predicted_scores) >= 0
            @test auc(actual, predicted_scores) <= 1

            # Perfect ranking
            perfect_scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
            @test auc(actual, perfect_scores) ≈ 1.0
        end

        @testset "Gini Coefficient" begin
            @test gini_coefficient(actual, predicted_scores) ≈ 2 * auc(actual, predicted_scores) - 1
        end

        @testset "KS Statistic" begin
            @test ks_statistic(actual, predicted_scores) >= 0
            @test ks_statistic(actual, predicted_scores) <= 1
        end

        @testset "Log Loss" begin
            @test all(ll(actual, predicted_scores) .>= 0)
            @test logloss(actual, predicted_scores) >= 0

            # Perfect predictions
            @test logloss([1, 0], [1.0, 0.0]) ≈ 0.0
        end

        @testset "Brier Score" begin
            @test brier_score(actual, predicted_scores) >= 0
            @test brier_score(actual, predicted_scores) <= 1

            # Perfect predictions
            @test brier_score([1, 0], [1.0, 0.0]) ≈ 0.0
        end

        @testset "Precision and Recall" begin
            # Use fully qualified names to avoid conflict with Base.precision
            @test UnifiedMetrics.precision(actual, predicted_labels) >= 0
            @test UnifiedMetrics.precision(actual, predicted_labels) <= 1
            @test recall(actual, predicted_labels) >= 0
            @test recall(actual, predicted_labels) <= 1
            @test sensitivity(actual, predicted_labels) ≈ recall(actual, predicted_labels)
        end

        @testset "Specificity and NPV" begin
            @test specificity(actual, predicted_labels) >= 0
            @test specificity(actual, predicted_labels) <= 1
            @test npv(actual, predicted_labels) >= 0
            @test npv(actual, predicted_labels) <= 1
        end

        @testset "Error Rates" begin
            @test fpr(actual, predicted_labels) ≈ 1 - specificity(actual, predicted_labels)
            @test fnr(actual, predicted_labels) ≈ 1 - recall(actual, predicted_labels)
        end

        @testset "F-beta Score" begin
            @test fbeta_score(actual, predicted_labels) >= 0
            @test fbeta_score(actual, predicted_labels) <= 1
            @test fbeta_score(actual, predicted_labels, beta=0.5) >= 0
            @test fbeta_score(actual, predicted_labels, beta=2.0) >= 0

            # F1 is harmonic mean of precision and recall
            p = UnifiedMetrics.precision(actual, predicted_labels)
            r = recall(actual, predicted_labels)
            @test fbeta_score(actual, predicted_labels, beta=1.0) ≈ 2 * p * r / (p + r)
        end

        @testset "Youden's J" begin
            @test youden_j(actual, predicted_labels) >= -1
            @test youden_j(actual, predicted_labels) <= 1
            @test youden_j(actual, predicted_labels) ≈ sensitivity(actual, predicted_labels) + specificity(actual, predicted_labels) - 1
        end

        @testset "Markedness" begin
            @test markedness(actual, predicted_labels) >= -1
            @test markedness(actual, predicted_labels) <= 1
        end

        @testset "Fowlkes-Mallows Index" begin
            @test fowlkes_mallows_index(actual, predicted_labels) >= 0
            @test fowlkes_mallows_index(actual, predicted_labels) <= 1
            @test fowlkes_mallows_index(actual, predicted_labels) ≈ sqrt(UnifiedMetrics.precision(actual, predicted_labels) * recall(actual, predicted_labels))
        end

        @testset "Likelihood Ratios" begin
            @test positive_likelihood_ratio(actual, predicted_labels) >= 0
            @test negative_likelihood_ratio(actual, predicted_labels) >= 0
        end

        @testset "Diagnostic Odds Ratio" begin
            @test diagnostic_odds_ratio(actual, predicted_labels) >= 0
        end

        @testset "Lift and Gain" begin
            @test lift(actual, predicted_scores, percentile=0.5) >= 0
            @test gain(actual, predicted_scores, percentile=0.5) >= 0
            @test gain(actual, predicted_scores, percentile=0.5) <= 1
        end
    end

    @testset "Information Retrieval Metrics" begin
        @testset "F1 (IR)" begin
            actual_docs = ["a", "c", "d"]
            predicted_docs = ["d", "e", "a"]

            @test f1(actual_docs, predicted_docs) >= 0
            @test f1(actual_docs, predicted_docs) <= 1

            # Perfect match
            @test f1(["a", "b"], ["a", "b"]) ≈ 1.0
        end

        @testset "Average Precision at K" begin
            actual_apk = ["a", "b", "d"]
            predicted_apk = ["b", "c", "a", "e", "f"]

            @test apk(3, actual_apk, predicted_apk) >= 0
            @test apk(3, actual_apk, predicted_apk) <= 1
            @test isnan(apk(3, String[], predicted_apk))
        end

        @testset "Mean Average Precision at K" begin
            actual_list = [["a", "b"], ["c"], ["d", "e"]]
            predicted_list = [["a", "c", "d"], ["x", "c"], ["e", "f"]]

            @test mapk(2, actual_list, predicted_list) >= 0
            @test mapk(2, actual_list, predicted_list) <= 1
        end

        @testset "DCG, IDCG, NDCG" begin
            relevance = [3, 2, 3, 0, 1, 2]

            @test dcg(relevance) >= 0
            @test idcg(relevance) >= dcg(relevance)
            @test ndcg(relevance) >= 0
            @test ndcg(relevance) <= 1

            # NDCG at k
            @test ndcg(relevance, k=3) >= 0
            @test ndcg(relevance, k=3) <= 1

            # Perfect ranking
            perfect_rel = [3, 3, 2, 2, 1, 0]
            @test ndcg(perfect_rel) ≈ 1.0
        end

        @testset "Mean NDCG" begin
            relevances = [[3, 2, 1, 0], [2, 1, 2, 1], [1, 1, 0, 0]]
            @test mean_ndcg(relevances) >= 0
            @test mean_ndcg(relevances) <= 1
            @test mean_ndcg(relevances, k=2) >= 0
        end

        @testset "Reciprocal Rank and MRR" begin
            actual_rr = ["a", "b"]
            predicted_rr = ["c", "a", "b", "d"]

            @test reciprocal_rank(actual_rr, predicted_rr) ≈ 0.5

            actual_list = [["a", "b"], ["c"], ["d", "e"]]
            predicted_list = [["b", "a", "c"], ["a", "c", "d"], ["e", "d", "f"]]

            @test mrr(actual_list, predicted_list) >= 0
            @test mrr(actual_list, predicted_list) <= 1
        end

        @testset "Hit Rate" begin
            actual_hr = [["a", "b"], ["c"], ["d", "e"]]
            predicted_hr = [["a", "x", "y"], ["x", "y", "z"], ["e", "f", "g"]]

            @test hit_rate(actual_hr, predicted_hr, k=3) ≈ 2/3
        end

        @testset "Precision, Recall, F1 at K" begin
            actual_k = ["a", "b", "c", "d"]
            predicted_k = ["a", "x", "b", "y", "z"]

            @test precision_at_k(actual_k, predicted_k, k=3) ≈ 2/3
            @test recall_at_k(actual_k, predicted_k, k=3) ≈ 2/4
            @test f1_at_k(actual_k, predicted_k, k=3) >= 0
        end

        @testset "Coverage" begin
            catalog = ["a", "b", "c", "d", "e", "f"]
            predicted_cov = [["a", "b"], ["a", "c"], ["b", "d"]]

            @test coverage(predicted_cov, catalog) ≈ 4/6
        end

        @testset "Novelty" begin
            popularity = Dict("a" => 0.9, "b" => 0.5, "c" => 0.1, "d" => 0.05)
            predicted_nov = [["a", "b"], ["c", "d"]]

            @test novelty(predicted_nov, popularity) >= 0
        end
    end

    @testset "Time Series Metrics" begin
        actual = [100.0, 110.0, 105.0, 115.0, 120.0, 125.0, 130.0, 128.0]
        predicted = [98.0, 108.0, 107.0, 113.0, 118.0, 123.0, 128.0, 126.0]

        @testset "MASE" begin
            @test mase(actual, predicted) >= 0
            @test mase(actual, predicted, m=1) >= 0

            # Perfect forecast
            @test mase(actual, actual) ≈ 0.0
        end

        @testset "MSSE and RMSSE" begin
            @test msse(actual, predicted) >= 0
            @test rmsse(actual, predicted) ≈ sqrt(msse(actual, predicted))
            @test rmsse(actual, predicted, m=1) >= 0
        end

        @testset "Tracking Signal" begin
            @test isfinite(tracking_signal(actual, predicted))

            # Biased forecast
            biased = predicted .- 5
            ts = tracking_signal(actual, biased)
            @test ts > 0  # Under-forecasting
        end

        @testset "Forecast Bias" begin
            @test isfinite(forecast_bias(actual, predicted))

            # Systematic under-forecast
            under_forecast = predicted .- 5
            @test forecast_bias(actual, under_forecast) > 0
        end

        @testset "Theil's U Statistics" begin
            @test theil_u1(actual, predicted) >= 0
            @test theil_u1(actual, predicted) <= 1

            @test theil_u2(actual, predicted) >= 0
        end

        @testset "WAPE" begin
            @test wape(actual, predicted) >= 0

            # Perfect forecast
            @test wape(actual, actual) ≈ 0.0
        end

        @testset "Directional Accuracy" begin
            @test directional_accuracy(actual, predicted) >= 0
            @test directional_accuracy(actual, predicted) <= 1

            # Perfect direction
            monotonic_actual = [1.0, 2.0, 3.0, 4.0, 5.0]
            monotonic_pred = [1.1, 2.1, 3.1, 4.1, 5.1]
            @test directional_accuracy(monotonic_actual, monotonic_pred) ≈ 1.0
        end

        @testset "Coverage Probability" begin
            lower = [95.0, 105.0, 100.0, 108.0, 112.0, 118.0, 122.0, 120.0]
            upper = [105.0, 115.0, 112.0, 122.0, 128.0, 132.0, 138.0, 136.0]

            @test coverage_probability(actual, lower, upper) >= 0
            @test coverage_probability(actual, lower, upper) <= 1

            # All covered
            wide_lower = actual .- 100
            wide_upper = actual .+ 100
            @test coverage_probability(actual, wide_lower, wide_upper) ≈ 1.0
        end

        @testset "Pinball Loss Series" begin
            @test pinball_loss_series(actual, predicted, quantile=0.5) >= 0
            @test pinball_loss_series(actual, predicted, quantile=0.1) >= 0
            @test pinball_loss_series(actual, predicted, quantile=0.9) >= 0
        end

        @testset "Winkler Score" begin
            lower = [95.0, 105.0, 100.0, 108.0, 112.0, 118.0, 122.0, 120.0]
            upper = [105.0, 115.0, 112.0, 122.0, 128.0, 132.0, 138.0, 136.0]

            @test winkler_score(actual, lower, upper, alpha=0.05) >= 0

            # Narrower intervals should have lower score if they cover
            narrow_lower = actual .- 1
            narrow_upper = actual .+ 1
            @test winkler_score(actual, narrow_lower, narrow_upper, alpha=0.05) < winkler_score(actual, lower, upper, alpha=0.05)
        end

        @testset "Autocorrelation Error" begin
            long_actual = cumsum(randn(50)) .+ 100
            long_predicted = long_actual .+ randn(50) * 0.5

            @test autocorrelation_error(long_actual, long_predicted, max_lag=5) >= 0
        end
    end

    @testset "Edge Cases" begin
        @testset "Perfect predictions" begin
            actual = [1.0, 2.0, 3.0, 4.0, 5.0]
            @test mae(actual, actual) ≈ 0.0
            @test rmse(actual, actual) ≈ 0.0
            @test mse(actual, actual) ≈ 0.0
        end

        @testset "Length mismatch" begin
            @test_throws AssertionError mae([1.0, 2.0], [1.0])
            @test_throws AssertionError accuracy([1, 2], [1])
            @test_throws AssertionError auc([1, 0], [0.5])
        end

        @testset "AUC edge cases" begin
            # No positives
            @test isnan(auc([0, 0, 0], [0.5, 0.3, 0.2]))
            # No negatives
            @test isnan(auc([1, 1, 1], [0.9, 0.8, 0.7]))
            # Tied scores should use average ranks
            @test auc([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]) ≈ 0.5
        end

        @testset "Lift/Gain empty inputs" begin
            @test isnan(lift(Float64[], Float64[]))
            @test isnan(gain(Float64[], Float64[]))
        end

        @testset "RSE/RAE constant actuals" begin
            constant = [5.0, 5.0, 5.0, 5.0]
            pred_match = [5.0, 5.0, 5.0, 5.0]
            pred_diff = [4.0, 5.0, 6.0, 5.0]

            # Perfect match on constant data
            @test rse(constant, pred_match) ≈ 0.0
            @test rae(constant, pred_match) ≈ 0.0

            # Non-matching predictions on constant data
            @test rse(constant, pred_diff) == Inf
            @test rae(constant, pred_diff) == Inf
        end

        @testset "Tracking signal perfect predictions" begin
            actual = [1.0, 2.0, 3.0, 4.0, 5.0]
            @test tracking_signal(actual, actual) ≈ 0.0
        end

        @testset "Autocorrelation error edge cases" begin
            # Short series
            @test isnan(autocorrelation_error([1.0, 2.0], [1.0, 2.0]))

            # Constant series
            constant = [5.0, 5.0, 5.0, 5.0, 5.0]
            @test isfinite(autocorrelation_error(constant, constant))
        end

        @testset "Tweedie deviance power=0" begin
            actual = [1.0, 2.0, 3.0]
            # Negative predictions should be allowed for power=0 (normal distribution)
            predicted_neg = [-0.5, 2.5, 2.8]
            @test isfinite(tweedie_deviance(actual, predicted_neg, power=0))
        end

        @testset "D2 Tweedie score edge cases" begin
            # Non-positive mean with power != 0
            actual_neg_mean = [-5.0, -3.0, -2.0]
            predicted = [1.0, 1.0, 1.0]
            @test isnan(d2_tweedie_score(actual_neg_mean, predicted, power=1.5))
        end

        @testset "SLE negative input validation" begin
            @test_throws AssertionError sle([-2.0, 1.0], [1.0, 2.0])
            @test_throws AssertionError sle([1.0, 2.0], [-2.0, 1.0])
        end

        @testset "F1@k zero precision and recall" begin
            # No overlap between actual and predicted
            @test f1_at_k(["a", "b"], ["x", "y", "z"], k=3) ≈ 0.0
        end

        @testset "MASE/MSSE constant series" begin
            # Constant series (naive error is zero)
            constant = [5.0, 5.0, 5.0, 5.0, 5.0]
            # Perfect forecast on constant series
            @test mase(constant, constant) ≈ 0.0
            @test msse(constant, constant) ≈ 0.0
            # Non-perfect forecast on constant series
            non_constant_pred = [5.0, 5.1, 5.0, 4.9, 5.0]
            @test mase(constant, non_constant_pred) == Inf
            @test msse(constant, non_constant_pred) == Inf
        end

        @testset "WAPE/WMAPE zero actuals" begin
            zeros_actual = [0.0, 0.0, 0.0]
            zeros_pred = [0.0, 0.0, 0.0]
            nonzero_pred = [0.1, 0.2, 0.3]
            # Perfect match on zeros
            @test wape(zeros_actual, zeros_pred) ≈ 0.0
            @test wmape(zeros_actual, zeros_pred) ≈ 0.0
            # Non-matching predictions on zeros
            @test wape(zeros_actual, nonzero_pred) == Inf
            @test wmape(zeros_actual, nonzero_pred) == Inf
        end

        @testset "Tweedie deviance actual validation" begin
            # Negative actuals should fail for power != 0
            @test_throws AssertionError tweedie_deviance([-1.0, 2.0], [1.0, 2.0], power=1)
            @test_throws AssertionError tweedie_deviance([-1.0, 2.0], [1.0, 2.0], power=2)
            @test_throws AssertionError tweedie_deviance([-1.0, 2.0], [1.0, 2.0], power=1.5)
        end

        @testset "MeanQuadraticWeightedKappa near-zero" begin
            # Near-zero kappas should not be biased
            near_zero_kappas = [0.0, 0.001, -0.001, 0.0]
            result = MeanQuadraticWeightedKappa(near_zero_kappas)
            @test abs(result) < 0.01  # Should remain near zero
        end

        @testset "Probability validation" begin
            @test_throws AssertionError ll([1, 0], [1.5, 0.5])  # probability > 1
            @test_throws AssertionError ll([1, 0], [-0.1, 0.5])  # probability < 0
            @test_throws AssertionError brier_score([1, 0], [1.5, 0.5])
        end

        @testset "ScoreQuadraticWeightedKappa single level" begin
            # Both raters use only one rating level → perfect agreement
            @test ScoreQuadraticWeightedKappa([3, 3, 3, 3], [3, 3, 3, 3]) ≈ 1.0
        end

        @testset "MRR and mean_ndcg empty inputs" begin
            empty_actual = Vector{Vector{String}}()
            empty_predicted = Vector{Vector{String}}()
            @test isnan(mrr(empty_actual, empty_predicted))

            empty_relevances = Vector{Vector{Float64}}()
            @test isnan(mean_ndcg(empty_relevances))
        end

        @testset "Theil U2 m validation" begin
            actual = [1.0, 2.0, 3.0, 4.0, 5.0]
            predicted = [1.1, 2.0, 2.9, 4.1, 4.9]
            @test_throws AssertionError theil_u2(actual, predicted, m=0)
        end

        @testset "Theil U1/U2 constant series" begin
            constant = [5.0, 5.0, 5.0, 5.0, 5.0]
            # Perfect forecast on constant series
            @test theil_u1(constant, constant) ≈ 0.0
            @test theil_u2(constant, constant) ≈ 0.0
            # Non-perfect forecast on constant series
            non_constant = [5.0, 5.1, 5.0, 4.9, 5.0]
            @test theil_u2(constant, non_constant) == Inf
            # All zeros
            zeros_vec = [0.0, 0.0, 0.0, 0.0, 0.0]
            @test theil_u1(zeros_vec, zeros_vec) ≈ 0.0
        end
    end

end
