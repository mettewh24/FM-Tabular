"""Plot the metrics in a results CSV as a function of epsilon.

Example:
    python plot_metrics_vs_epsilon.py --input-csv iris_eps.csv

Duplicate dataset/solver/epsilon configurations raise ValueError, including
identical rows. Epsilon is ignored for exact OT and independent CFM references.

Use --ngen 1 for SE across training runs, correcting input SD from ddof=0
to sample SD (ddof=1). For pooled multiple-generation results, SE remains
an explicit approximation. Use --error-bars sd for original SD.
Use --nominal-se-n 15 --ngen 5 for nominal pooled quality bars SD/sqrt(15).
This ignores within-training dependence; runtime still uses its training count.
Time is measured in seconds per run and includes training and generation,
excluding metric evaluation. Its SD is computed across run timings.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EPSILON_PATTERN = re.compile(
    r"\bepsilon\s*=\s*"
    r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
    r"(?![0-9A-Za-z_.])"
)
EXACT_SOLVER_PATTERN = re.compile(r"\bot_solver\s*=\s*exact\b", re.IGNORECASE)
INDEPENDENT_SOLVER_PATTERN = re.compile(
    r"\bot_solver\s*=\s*independent\b", re.IGNORECASE
)
COMPLETED_RUNS_PATTERN = re.compile(r"\bcompleted_runs\s*=\s*(\d+)\s*/\s*\d+")
IDENTIFIER_COLUMNS = {
    "dataset", "method", "epsilon", "n_runs", "n_generations", "nominal_se_n"
}
EXCLUDED_METRIC_PREFIXES = ("F1", "R2")
DEFAULT_METRICS = (
    "W_test",
    "coverage_test",
    "dist_score_test",
    "corr_score_test",
    "disc_score_test",
    "time",
)
METRIC_LABELS = {
    "W_test": "Wasserstein distance (test)",
    "coverage_test": "Coverage (test)",
    "dist_score_test": "Distribution score (test)",
    "corr_score_test": "Correlation score (test)",
    "disc_score_test": "Discrimination score (test)",
    "time": "Training + generation time (s)",
}
DATASET_LABELS = {"iris": "Iris", "libras": "Libras", "connectionist_bench_sonar": "Sonar"}


def extract_epsilon(method: str) -> float:
    """Extract epsilon from a method description such as ``epsilon=0.5``."""
    match = EPSILON_PATTERN.search(str(method))
    if match is None:
        raise ValueError(f"Could not find epsilon in method: {method!r}")
    return float(match.group(1))


def is_exact_solver(method: object) -> bool:
    """Return whether a method description uses the exact OT solver."""
    return bool(EXACT_SOLVER_PATTERN.search(str(method)))


def is_independent_solver(method: object) -> bool:
    """Return whether a method description uses independent CFM."""
    return bool(INDEPENDENT_SOLVER_PATTERN.search(str(method)))


def reference_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify couplings for which epsilon is not applicable."""
    if "method" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["method"].map(is_exact_solver) | frame["method"].map(
        is_independent_solver
    )


def validate_no_duplicates(frame: pd.DataFrame) -> None:
    """Reject repeated configurations, including duplicate reference couplings."""
    keys = frame[["dataset", "epsilon"]].copy()
    keys["solver"] = "sinkhorn"
    if "method" in frame.columns:
        keys.loc[frame["method"].map(is_exact_solver), "solver"] = "exact"
        keys.loc[frame["method"].map(is_independent_solver), "solver"] = "independent"
    keys.loc[reference_mask(frame), "epsilon"] = float("nan")
    counts = keys.groupby(["dataset", "solver", "epsilon"], dropna=False).size()
    duplicates = counts[counts > 1]
    if not duplicates.empty:
        details = []
        for (dataset, solver, epsilon), count in duplicates.items():
            epsilon_label = "NA" if pd.isna(epsilon) else repr(epsilon)
            details.append(
                f"dataset={dataset!r}, solver={solver!r}, "
                f"epsilon={epsilon_label}: {count} rows"
            )
        raise ValueError("Duplicate configurations found:\n" + "\n".join(details))


def load_results(input_csv: Path) -> pd.DataFrame:
    """Load the CSV and normalize epsilon and metric columns."""
    frame = pd.read_csv(input_csv)
    required = {"dataset"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "epsilon" not in frame.columns and "method" not in frame.columns:
        raise ValueError("CSV must contain either an 'epsilon' or a 'method' column")

    frame = frame.copy()
    varying_mask = ~reference_mask(frame)
    if "epsilon" in frame.columns:
        frame["epsilon"] = pd.to_numeric(frame["epsilon"], errors="coerce")
    else:
        frame["epsilon"] = float("nan")
        frame.loc[varying_mask, "epsilon"] = frame.loc[
            varying_mask, "method"
        ].map(extract_epsilon).astype(float)
    frame.loc[~varying_mask, "epsilon"] = float("nan")
    varying_epsilon = frame.loc[varying_mask, "epsilon"]
    if varying_epsilon.isna().any() or (varying_epsilon <= 0).any():
        raise ValueError("Epsilon values for varying couplings must be positive numbers")
    validate_no_duplicates(frame)
    for column in frame.columns:
        if column not in {"dataset", "method", "epsilon"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def select_metrics(
    frame: pd.DataFrame,
    requested: list[str] | None,
    include_f1_r2: bool = False,
) -> list[str]:
    """Return numeric value columns, excluding standard-deviation columns."""
    available = [
        column
        for column in frame.columns
        if column not in IDENTIFIER_COLUMNS
        and not column.endswith(("_std", "_se", "_se_approx", "_se_nominal"))
        and pd.api.types.is_numeric_dtype(frame[column])
        and (
            include_f1_r2
            or not column.upper().startswith(EXCLUDED_METRIC_PREFIXES)
        )
    ]
    if requested is None:
        selected = [metric for metric in DEFAULT_METRICS if metric in available]
        if include_f1_r2:
            selected.extend(
                metric for metric in available
                if metric.upper().startswith(EXCLUDED_METRIC_PREFIXES)
            )
        if not selected:
            raise ValueError("No default test metrics found; select metrics with --metrics")
        return selected

    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(
            f"Unknown metric(s): {missing}. Available metrics: {', '.join(available)}"
        )
    return requested


def add_standard_errors(
    frame: pd.DataFrame, metrics: list[str], n_runs: int | None = None,
    ngen: int | None = None, std_ddof: int = 0,
) -> pd.DataFrame:
    """Compute sample SE for one generation per run, or label pooled approximation."""
    if ngen is not None and ngen < 1:
        raise ValueError("--ngen must be positive")
    if std_ddof not in (0, 1):
        raise ValueError("--std-ddof must be 0 or 1")
    frame = frame.copy()
    counts = pd.Series(float("nan"), index=frame.index)
    if "method" in frame.columns:
        counts = pd.to_numeric(
            frame["method"].str.extract(COMPLETED_RUNS_PATTERN, expand=False),
            errors="coerce",
        )
    if n_runs is not None:
        if n_runs < 2:
            raise ValueError("--n-runs must be at least 2")
        counts = counts.fillna(n_runs)
    invalid = counts.isna() | (counts < 2)
    if invalid.any():
        datasets = sorted(frame.loc[invalid, "dataset"].astype(str).unique())
        raise ValueError(
            "SE requires at least two completed training runs per row. "
            "For CSVs without completed_runs metadata, supply --n-runs. "
            f"Affected datasets: {datasets}"
        )
    frame["n_runs"] = counts
    if ngen is not None:
        frame["n_generations"] = ngen
    for metric in metrics:
        std_column = f"{metric}_std"
        if std_column not in frame.columns:
            raise ValueError(f"Cannot compute SE: missing column {std_column!r}")
        if ngen == 1:
            # Input np.std(..., ddof=0): s_sample/sqrt(N) = SD_0/sqrt(N-1).
            denominator = counts - 1 if std_ddof == 0 else counts
            frame[f"{metric}_se"] = frame[std_column] / denominator.pow(0.5)
        else:
            frame[f"{metric}_se_approx"] = frame[std_column] / counts.pow(0.5)
    if ngen != 1:
        warnings.warn(
            "Error bars are approximate SE: pooled SD / sqrt(completed training runs). "
            "Generations within each run are not counted as independent training runs. "
            "Between-run SE requires per-run metric means, or --ngen 1 for single-generation data.",
            UserWarning, stacklevel=2,
        )
    return frame


def add_nominal_standard_errors(
    frame: pd.DataFrame, metrics: list[str], nominal_n: int,
    ngen: int | None, n_runs: int | None = None, std_ddof: int = 0,
) -> pd.DataFrame:
    """Use SD/sqrt(N) for pooled quality scores, with explicit nominal labels.

    Runtime has one observation per training, so it retains sample SE across
    training timings rather than treating generations as extra timings.
    """
    if nominal_n < 2 or ngen is None or ngen < 1:
        raise ValueError("--nominal-se-n requires N >= 2 and a positive --ngen")
    timing_metrics = [metric for metric in metrics if metric == "time"]
    result = add_standard_errors(
        frame, timing_metrics, n_runs=n_runs, ngen=1, std_ddof=std_ddof
    )
    if (result["n_runs"] * ngen != nominal_n).any():
        raise ValueError("Nominal N must equal completed training runs times --ngen")
    result["n_generations"] = ngen
    result["nominal_se_n"] = nominal_n
    for metric in metrics:
        if metric == "time":
            continue
        std_column = f"{metric}_std"
        if std_column not in result:
            raise ValueError(f"Cannot compute nominal SE: missing {std_column!r}")
        result[f"{metric}_se_nominal"] = result[std_column] / nominal_n**0.5
    warnings.warn(
        f"Nominal quality SE uses SD/sqrt({nominal_n}), ignoring dependence between "
        "generations from the same training. Runtime uses sample SE across training runs.",
        UserWarning, stacklevel=2,
    )
    return result


def plot_errors(data: pd.DataFrame, metric: str, error_bars: str):
    """Return the selected error values and an explicit label for the figure."""
    if error_bars == "se" and f"{metric}_se_nominal" in data:
        values = data["nominal_se_n"].dropna().unique()
        label = f"nominal SE (N={values[0]:g})" if len(values) == 1 else "nominal SE"
        return data[f"{metric}_se_nominal"], label
    if error_bars == "se" and f"{metric}_se" in data:
        if metric == "time" and "nominal_se_n" in data:
            values = data["n_runs"].dropna().unique()
            label = f"SE (N={values[0]:g})" if len(values) == 1 else "SE"
            return data[f"{metric}_se"], label
        return data[f"{metric}_se"], "SE"
    suffix = "se_approx" if error_bars == "se" else "std"
    column = f"{metric}_{suffix}"
    label = "approx. SE" if error_bars == "se" else "SD"
    return (data[column] if column in data else None), label


def save_figure(figure, output_path: Path, dpi: int, formats: tuple[str, ...]) -> None:
    """Export the same figure in raster and/or vector formats."""
    for extension in formats:
        figure.savefig(output_path.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")


def aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Return unique epsilon-curve rows without averaging reported statistics."""
    validate_no_duplicates(frame)
    varying_results = frame.loc[~reference_mask(frame)]
    numeric_columns = frame.select_dtypes(include="number").columns.difference(
        ["dataset"], sort=False
    )
    return (
        varying_results[["dataset", *numeric_columns]]
        .sort_values(["dataset", "epsilon"])
        .reset_index(drop=True)
    )


def reference_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exact OT and independent CFM rows for horizontal reference lines."""
    return frame.loc[reference_mask(frame)]


def plot_references(
    axis, references: pd.DataFrame, dataset: str, metric: str, error_bars: str = "sd"
) -> None:
    """Draw reference means and full-width bands for the selected uncertainty."""
    data = references.loc[references["dataset"] == dataset]
    if data.empty:
        return
    for predicate, label, color, linestyle in (
        (is_exact_solver, "OT-CFM (exact)", "#c44e52", "--"),
        (is_independent_solver, "IC-CFM (independent)", "#228833", "-."),
    ):
        coupling = data.loc[data["method"].map(predicate)]
        values = coupling[metric].dropna()
        errors, error_label = plot_errors(coupling, metric, error_bars)
        for index, (row_index, value) in enumerate(values.items()):
            error = errors.loc[row_index] if errors is not None else None
            line_label = label
            if error is not None and pd.notna(error):
                axis.axhspan(
                    value - error, value + error,
                    color=color, alpha=0.12, linewidth=0, zorder=0.5,
                    label="_nolegend_",
                )
                line_label = f"{label} ± {error_label}"
            axis.axhline(
                value,
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                label=line_label if index == 0 else "_nolegend_",
            )


def plot_metric(
    aggregated: pd.DataFrame,
    references: pd.DataFrame,
    metric: str,
    output_path: Path,
    dataset: str,
    error_bars: str = "sd",
    dpi: int = 600,
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Create one epsilon curve with labeled uncertainty and reference means."""
    figure, axis = plt.subplots(figsize=(8, 5))
    data = aggregated[aggregated["dataset"] == dataset].sort_values("epsilon")
    error, error_label = plot_errors(data, metric, error_bars)
    axis.errorbar(
        data["epsilon"],
        data[metric],
        yerr=error,
        marker="o",
        linewidth=2,
        capsize=3 if error is not None else 0,
        color="#176b87",
        label=f"Entropic OT-CFM (mean ± {error_label})",
    )
    plot_references(axis, references, dataset, metric, error_bars)
    axis.legend()
    metric_label = METRIC_LABELS.get(metric, metric)
    axis.set_title(f"{metric_label} - {DATASET_LABELS.get(dataset, dataset)}")
    axis.set_xlabel(r"Entropic regularization $\varepsilon$")
    axis.set_ylabel(metric_label)
    axis.set_xscale("log")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output_path, dpi, formats)
    plt.close(figure)


def plot_dashboard(
    aggregated: pd.DataFrame,
    references: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    dataset: str,
    error_bars: str = "sd",
    dpi: int = 600,
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Create a multi-panel figure containing all selected metrics."""
    columns = min(3, len(metrics))
    rows = (len(metrics) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.4 * columns, 4.2 * rows),
        squeeze=False, constrained_layout=True,
    )
    data = aggregated[aggregated["dataset"] == dataset].sort_values("epsilon")

    for index, metric in enumerate(metrics):
        axis = axes[index // columns][index % columns]
        error, error_label = plot_errors(data, metric, error_bars)
        axis.errorbar(
            data["epsilon"],
            data[metric],
            yerr=error,
            marker="o",
            linewidth=1.7,
            capsize=2 if error is not None else 0,
            color="#176b87",
            label=f"Entropic OT-CFM (mean ± {error_label})",
        )
        plot_references(axis, references, dataset, metric, error_bars)
        axis.legend(fontsize=8)
        axis.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        axis.set_xlabel(r"Entropic regularization $\varepsilon$")
        axis.set_xscale("log")
        axis.grid(alpha=0.22)

    for index in range(len(metrics), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    figure.suptitle(
        f"Sensitivity to entropic regularization - {DATASET_LABELS.get(dataset, dataset)}",
        x=0.5, ha="center", fontsize=15, fontweight="bold",
    )
    save_figure(figure, output_path, dpi, formats)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("ablation_eps.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("ablation_epsilon_plots"))
    parser.add_argument("--dataset", default=None, help="Dataset to plot; default: all datasets.")
    parser.add_argument("--dpi", type=int, default=600, help="Raster output resolution (default: 600 dpi).")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png"])
    parser.add_argument(
        "--error-bars", choices=["se", "sd"], default="se",
        help="Error bars: SE (sample SE with --ngen 1, pooled approximation otherwise), or SD.",
    )
    parser.add_argument(
        "--ngen", type=int, default=None,
        help="Generations per training in the input CSV; 1 enables sample SE across runs.",
    )
    parser.add_argument(
        "--std-ddof", type=int, choices=[0, 1], default=0,
        help="ddof used for input SD (script_generation.py uses 0); prevents double correction.",
    )
    parser.add_argument(
        "--nominal-se-n", type=int, default=None,
        help="Explicit nominal pooled quality SE=SD/sqrt(N); requires --ngen. "
        "Runtime retains sample SE across training timings.",
    )
    parser.add_argument(
        "--n-runs", type=int, default=None,
        help="Training count for rows without completed_runs metadata; used only for SE.",
    )
    parser.add_argument(
        "--include-f1-r2",
        action="store_true",
        help="Also plot F1 and R2 metrics (excluded by default).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to plot; default: W_test, coverage_test, dist_score_test, "
        "corr_score_test, disc_score_test, time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    frame = load_results(args.input_csv)
    metrics = select_metrics(frame, args.metrics, args.include_f1_r2)
    if args.nominal_se_n is not None:
        if args.error_bars != "se":
            raise ValueError("--nominal-se-n requires --error-bars se")
        frame = add_nominal_standard_errors(
            frame, metrics, args.nominal_se_n, args.ngen, args.n_runs, args.std_ddof
        )
    elif args.error_bars == "se":
        frame = add_standard_errors(frame, metrics, args.n_runs, args.ngen, args.std_ddof)
    aggregated = aggregate_results(frame)
    references = reference_results(frame)
    available_datasets = sorted(
        set(aggregated["dataset"].unique()) | set(references["dataset"].unique())
    )
    datasets = [args.dataset] if args.dataset else available_datasets

    for dataset in datasets:
        if dataset not in available_datasets:
            raise ValueError(f"Dataset {dataset!r} not found in {args.input_csv}")
        dataset_dir = args.output_dir / str(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for metric in metrics:
            plot_metric(
                aggregated,
                references,
                metric,
                dataset_dir / f"{metric}_vs_epsilon.png",
                dataset,
                args.error_bars,
                args.dpi,
                tuple(args.formats),
            )
        plot_dashboard(
            aggregated,
            references,
            metrics,
            dataset_dir / "all_metrics_vs_epsilon.png",
            dataset,
            args.error_bars,
            args.dpi,
            tuple(args.formats),
        )

    aggregated.to_csv(args.output_dir / "metrics_vs_epsilon_aggregated.csv", index=False)
    references.to_csv(args.output_dir / "reference_metrics.csv", index=False)
    print(f"Saved plots for {len(datasets)} dataset(s) and {len(metrics)} metric(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
