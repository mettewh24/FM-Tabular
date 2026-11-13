import logging
from math import ceil, sqrt
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["singling_out", "linkability", "inference"]
METRIC_LABELS = {
    "singling_out": "Singling-out",
    "linkability": "Linkability",
    "inference": "Inference",
}


def _escape_latex(text):
    """Escape the handful of LaTeX-special characters likely to show up in model/dataset names."""
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    text = str(text)
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def _format_cell(mean, standard_error, decimals):
    """Format one (mean, standard error) pair as a LaTeX table cell: 'mean (SE)'.

    Falls back to just the mean if the standard error is unavailable (e.g. only one valid
    data point for that model/metric), and to '--' if there is no valid
    value at all (every row for that model/metric was excluded or failed).
    """
    if mean is None or pd.isna(mean):
        return "--"
    if standard_error is None or pd.isna(standard_error):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ({standard_error:.{decimals}f})"


def write_latex_table(summary, output_tex, decimals=1, caption="Privacy risk scores (mean (standard error), 0--100 scale) across all evaluated datasets.", label="tab:privacy_risks"):
    """Write a self-contained LaTeX table: one row per model, one column per risk metric."""
    header_cols = " & ".join(["Model"] + [METRIC_LABELS[m] for m in METRICS])
    col_spec = "l" + "c" * len(METRICS)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_cols + r" \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        cells = [_escape_latex(row["methods"])]
        for metric in METRICS:
            cell = _format_cell(row[f"{metric}_risk_mean"], row[f"{metric}_risk_se"], decimals)
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    with open(output_tex, "w") as f:
        f.write(tex)
    return tex


def _is_flagged(warning_series):
    """A row counts as 'flagged' for a metric if its {metric}_warning cell is non-empty.

    Empty cells round-trip through CSV as either an empty string or NaN
    depending on how they were written/read, so both are treated as "no
    warning" here.
    """
    return warning_series.notna() & (warning_series.astype(str).str.strip() != "")


def summarize_metric(group, metric, include_flagged):
    """Compute mean/standard error across dataset means for one model group."""
    risk_col = f"{metric}_risk"
    warn_col = f"{metric}_warning"

    values = group[risk_col].copy()
    n_rows = len(values)

    # Rows where the metric raised an exception entirely (see the original
    # script's per-metric try/except): {metric}_risk was left as None/NaN.
    n_failed = int(values.isna().sum())

    if warn_col in group.columns:
        flagged = _is_flagged(group[warn_col])
    else:
        flagged = pd.Series(False, index=group.index)
    n_flagged = int(flagged.sum())

    if not include_flagged:
        # Don't drop the rows, just exclude them from THIS metric's stats --
        # a row flagged for singling_out may still be perfectly usable for
        # linkability/inference.
        values = values.where(~flagged)

    valid_rows = values.notna()
    if not include_flagged:
        valid_rows &= ~flagged

    dataset_means = (
        group.loc[valid_rows, ["dataset", risk_col]]
        .groupby("dataset")[risk_col]
        .mean()
    )
    n_valid = len(dataset_means)
    print(f"number of valid datasets for {metric}: {n_valid}")

    standard_error = dataset_means.std(ddof=1) / sqrt(n_valid) if n_valid > 1 else None

    return {
        f"{metric}_risk_mean": ceil(dataset_means.mean()) if n_valid else None,
        f"{metric}_risk_se": ceil(standard_error) if standard_error is not None else None,
        f"{metric}_n_valid": n_valid,
        f"{metric}_n_flagged": n_flagged,
        f"{metric}_n_failed": n_failed,
        f"{metric}_n_rows": n_rows,
    }


def build_parser():
    results_dir = Path(__file__).resolve().parent / "generated" / "privacy"
    parser = ArgumentParser(
        description="Aggregate the per-row privacy_results.csv into one mean/standard-error risk summary per model."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(results_dir / "privacy_results.csv"),
        help="Path to the CSV produced by the privacy evaluation script.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(results_dir / "privacy_summary_by_model.csv"),
        help="Where to write the per-model summary CSV.",
    )
    parser.add_argument(
        "--include-flagged",
        action="store_true",
           help="By default, rows whose {metric}_warning is non-empty are excluded from that metric's "
               "mean/standard error -- Anonymeter itself flagged them as untrustworthy (attack no better than the "
             "random baseline, control attack saturated at 100%%, or too few singling-out queries could "
             "be generated). Pass this flag to include them anyway.",
    )
    parser.add_argument(
        "--output-tex",
        type=str,
        default=str(results_dir / "privacy_summary_by_model.tex"),
        help="Where to write the LaTeX table (rows=models, columns=risks, cells='mean (standard error)').",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=0,
        help="Number of decimal places shown in the LaTeX table cells.",
    )
    parser.add_argument(
        "--tex-caption",
        type=str,
        default="Privacy risk scores (mean (standard error), 0--100 scale) across all evaluated datasets.",
        help="Caption for the LaTeX table.",
    )
    parser.add_argument(
        "--tex-label",
        type=str,
        default="tab:privacy_risks",
        help="LaTeX \\label{} for the table.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    df = pd.read_csv(args.input_csv)

    if "methods" not in df.columns:
        raise SystemExit(f"'{args.input_csv}' does not look like a privacy_results.csv (no 'methods' column).")

    rows = []
    for model, group in df.groupby("methods"):
        row = {
            "methods": model,
            "n_rows": len(group),
            "n_datasets": group["dataset"].nunique(),
        }
        for metric in METRICS:
            row.update(summarize_metric(group, metric, args.include_flagged))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("methods").reset_index(drop=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_tex).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    logger.info(f"Wrote per-model summary for {len(summary)} model(s) to {args.output_csv}")

    if not args.include_flagged:
        for metric in METRICS:
            total_flagged = summary[f"{metric}_n_flagged"].sum()
            if total_flagged:
                logger.info(
                    f"{metric}: {total_flagged} row(s) excluded from the mean/standard error across all models "
                    f"because Anonymeter flagged them (use --include-flagged to keep them)."
                )

    print(summary.to_string(index=False))

    tex = write_latex_table(
        summary,
        args.output_tex,
        decimals=args.decimals,
        caption=args.tex_caption,
        label=args.tex_label,
    )
    logger.info(f"Wrote LaTeX table to {args.output_tex}")
    print("\n" + "=" * 70)
    print("LaTeX table (copy/paste into your paper):")
    print("=" * 70)
    print(tex)


if __name__ == "__main__":
    main()