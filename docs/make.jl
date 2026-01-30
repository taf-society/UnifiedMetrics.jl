using Documenter
using UnifiedMetrics

makedocs(
    sitename = "UnifiedMetrics.jl",
    modules = [UnifiedMetrics],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://taf-society.github.io/UnifiedMetrics.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Choosing the Right Metric" => "choosing_metrics.md",
        "Time Series Forecasting" => "time_series.md",
        "Other Metrics" => [
            "Regression" => "regression.md",
            "Classification" => "classification.md",
            "Binary Classification" => "binary_classification.md",
            "Information Retrieval" => "information_retrieval.md",
        ],
        "API Reference" => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/taf-society/UnifiedMetrics.jl.git",
    devbranch = "main",
)
