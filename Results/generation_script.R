# Paper tables using the historical benchmark filters and summary statistics.
args <- commandArgs(trailingOnly = TRUE)
input_csv <- if (length(args) >= 1L) args[[1L]] else "Results/reference/benchmark.csv"
output_dir <- if (length(args) >= 2L) args[[2L]] else "Results/generated/tables"
data <- read.csv(input_csv, stringsAsFactors = FALSE)
data$method <- trimws(data$method)
data$percent_bias <- data$percent_bias / 100

all_vars = c('iris', 'wine', 'parkinsons', 'climate_model_crashes', 'concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'ecoli', 'yeast', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'concrete_slump', 'wine_quality_red', 'wine_quality_white', 'california', 'bean', 'car','congress','tictactoe','libras')
W_vars = c('iris', 'wine', 'parkinsons', 'climate_model_crashes', 'concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'ecoli', 'yeast', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'concrete_slump', 'wine_quality_red', 'wine_quality_white', 'car','congress','tictactoe','libras')
R2_vars = c('concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'wine_quality_red', 'wine_quality_white', 'california')
F1_vars = c('iris', 'wine', 'parkinsons', 'climate_model_crashes', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'ecoli', 'yeast', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'bean', 'car','congress','tictactoe','libras')
time_vars = setdiff(all_vars, 'libras')

metric_datasets <- list(
    W_train = W_vars,
    W_test = W_vars,
    coverage_train = all_vars,
    coverage_test = all_vars,
    R2_fake = R2_vars,
    F1_fake = F1_vars,
    percent_bias = R2_vars,
    coverage_rate = R2_vars,
    class_score = all_vars,
    dist_score_train = all_vars,
    dist_score_test = all_vars,
    corr_score_train = all_vars,
    corr_score_test = all_vars,
    disc_score_train = all_vars,
    disc_score_test = all_vars,
    time = time_vars
)

methods <- c("GaussianCopula", "TVAE", "CTGAN", "CTABGAN", "stasy", "TabDDPM",
             "forest_diffusion_vp", "forest_diffusion_flow", "fm_tabular")
expected <- expand.grid(method = methods, dataset = all_vars)
pair_keys <- paste(data$method, data$dataset)
if (anyDuplicated(pair_keys) || !setequal(pair_keys, with(expected, paste(method, dataset)))) {
    stop("Expected exactly one row per paper method and dataset.")
}

format_summary <- function(metric, method) {
    rows <- data$method == method & data$dataset %in% metric_datasets[[metric]]
    values <- data[[metric]][rows]
    values <- values[!is.na(values)]
    if (length(values) < 2L) {
        stop(sprintf("Too few values for %s on %s.", metric, method))
    }
    digits <- if (metric %in% syndat_metrics) 0L else 2L
    sprintf("%.*f (%.*f)", digits, mean(values), digits, sd(values) / sqrt(length(values)))
}

quality_metrics <- c("W_train", "W_test", "coverage_train", "coverage_test")
syndat_metrics <- c("disc_score_train", "disc_score_test", "dist_score_train",
                    "dist_score_test", "corr_score_train", "corr_score_test")
utility_metrics <- c("R2_fake", "F1_fake", "class_score", "percent_bias", "coverage_rate", "time")
tables <- list(quality = quality_metrics, syndat = syndat_metrics, utility = utility_metrics)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
for (table_name in names(tables)) {
    result <- data.frame(method = methods)
    for (metric in tables[[table_name]]) {
        result[[metric]] <- vapply(methods, function(method) format_summary(metric, method), character(1))
    }
    output_csv <- file.path(output_dir, paste0("paper_", table_name, ".csv"))
    write.csv(result, output_csv, row.names = FALSE)
    message(output_csv)
}
