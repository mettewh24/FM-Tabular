import logging
import math
import re
import warnings
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from anonymeter.evaluators import InferenceEvaluator, LinkabilityEvaluator, SinglingOutEvaluator
from syndat.postprocessing import assert_minmax, normalize_float_precision, normalize_scale

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


def compute_risk_score(risk_value):
    """Map an Anonymeter risk value (a float in [0, 1]) onto an integer 0-100 scale.

    We deliberately round UP (ceiling), not to the nearest integer. This
    guarantees the reported score never *understates* the risk estimated by
    Anonymeter: a risk of, say, 0.004 becomes "1" rather than being rounded
    down to "0". A risk of exactly 0 still maps to 0, and a risk of exactly 1
    maps to 100.

    Note: this is a reporting convention, not something defined by Anonymeter
    itself. The library's `PrivacyRisk.value` is already a bounded [0, 1]
    estimate (see `anonymeter.stats.confidence.bind_value`), so no clipping is
    needed here.
    """
    return math.ceil(risk_value * 100)


class _ListLogHandler(logging.Handler):
    """Collects log records into a plain list instead of printing them."""

    def __init__(self, level=logging.WARNING):
        super().__init__(level=level)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _run_with_warnings(fn, *args, **kwargs):
    """Run an Anonymeter call and capture both warnings and log messages it emits.

    Anonymeter surfaces problems through two different, unrelated channels,
    and the original script silently dropped both:
    - `warnings.warn(...)`, used e.g. when the "real" attack performs no
      better than the random-guessing baseline, or when the control attack
      succeeds 100% of the time (the residual risk cannot be trusted);
    - plain `logger.warning(...)` calls (NOT `warnings.warn`), used e.g. by
      `SinglingOutEvaluator` when it could only generate a handful (or zero)
      of the requested singling-out queries -- a case where the reported risk
      can look like "no risk" while really meaning "this attack could not be
      tested". This second channel is easy to miss because it never raises a
      `UserWarning`; here we attach a temporary handler to anonymeter's own
      logger hierarchy to catch it too.
    """
    log_handler = _ListLogHandler()
    anonymeter_logger = logging.getLogger("anonymeter")
    anonymeter_logger.addHandler(log_handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn(*args, **kwargs)
    finally:
        anonymeter_logger.removeHandler(log_handler)

    messages = [str(w.message) for w in caught] + log_handler.messages
    return result, "; ".join(messages)


def _n_attacks_for(results, kind):
    """Get the number of attack attempts used for a given attack type.

    Anonymeter's `EvaluationResults` has changed shape between releases:
    - PyPI 1.0.0 (the version `pip install anonymeter` currently gives you):
      a single shared `n_attacks` attribute, used as the denominator for the
      attack, baseline, *and* control success rates alike.
    - The unreleased `main` branch (a future 1.0.1+): separate
      `n_attacks_ori` / `n_attacks_baseline` / `n_attacks_control` attributes,
      needed because the `InferenceEvaluator` can use differently-sized
      samples for each.
    `kind` is one of "ori", "baseline", "control". This helper picks whichever
    the installed version actually provides, so the script keeps working
    whichever anonymeter release ends up installed.
    """
    versioned_attr = f"n_attacks_{kind}"
    if hasattr(results, versioned_attr):
        return getattr(results, versioned_attr)
    return results.n_attacks


def _is_regression_target(column):
    """Decide whether a column should be scored as a regression (tolerance-based)
    or classification (exact-match) target for the inference attack.

    Note: `pandas.api.types.is_numeric_dtype` returns True for boolean columns
    too, which would incorrectly route flags/booleans through the
    tolerance-based regression scoring instead of exact-match classification.
    Booleans (and anything else non-numeric) are treated as classification.
    """
    return pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)


def _pct(n_success, n_total):
    """Raw (unsmoothed) percentage of successful attacks: n_success / n_total * 100."""
    if n_success is None or not n_total:
        return None
    return round(100 * n_success / n_total, 2)


def _row_from_results(results, warn_msg, prefix):
    """Build the output columns for one metric from an Anonymeter `EvaluationResults`.

    Exposes both:
    - the final risk score (`{prefix}_risk`) and its confidence interval, which
      Anonymeter derives from a Wilson-smoothed success rate (and, when a
      control set is used, corrected against the control attack) -- NOT from
      the raw success fraction;
    - the raw, unsmoothed attack/baseline/control success counts and
      percentages (`n_success` out of `n_total` attempted attacks), i.e. the
      literal "how many attacks actually succeeded" figure.

    The two will usually be close but are not mathematically identical,
    especially when `n_attacks` is small (the Wilson correction shrinks
    extreme rates towards 0.5 to account for sampling noise).
    """
    risk = results.risk()
    n_ori = _n_attacks_for(results, "ori")
    n_baseline = _n_attacks_for(results, "baseline")
    n_control = _n_attacks_for(results, "control") if results.n_control is not None else None
    return {
        f"{prefix}_risk": compute_risk_score(risk.value),
        f"{prefix}_risk_ci_low": round(risk.ci[0] * 100, 2),
        f"{prefix}_risk_ci_high": round(risk.ci[1] * 100, 2),
        f"{prefix}_attack_n_success": results.n_success,
        f"{prefix}_attack_n_total": n_ori,
        f"{prefix}_attack_success_pct": _pct(results.n_success, n_ori),
        f"{prefix}_baseline_n_success": results.n_baseline,
        f"{prefix}_baseline_n_total": n_baseline,
        f"{prefix}_baseline_success_pct": _pct(results.n_baseline, n_baseline),
        f"{prefix}_control_n_success": results.n_control,
        f"{prefix}_control_n_total": n_control,
        f"{prefix}_control_success_pct": _pct(results.n_control, n_control) if results.n_control is not None else None,
        f"{prefix}_warning": warn_msg,
    }


# --------------------------------------------------------------------------- #
# Risk evaluators
# --------------------------------------------------------------------------- #

def get_singling_out_results(real_data, synthetic_data, control_data=None, n_attacks=100, mode="univariate", n_cols=3):
    """Singling-out evaluation, following anonymeter's documented usage pattern.

    https://github.com/statice/anonymeter#basic-usage-pattern
    Returns the raw `EvaluationResults` object (not just the final risk) so
    raw success counts/percentages can be reported alongside the risk.
    """
    n_attacks = min(n_attacks, real_data.shape[0])

    def _build_and_run():
        evaluator = SinglingOutEvaluator(
            ori=real_data,
            syn=synthetic_data,
            control=control_data,
            n_attacks=n_attacks,
            n_cols=n_cols,
            max_attempts=100000,
        )
        evaluator.evaluate(mode=mode)
        return evaluator.results()

    try:
        results, warn_msg = _run_with_warnings(_build_and_run)
    except RuntimeError as exc:
        # When `control` and `real_data` differ in size (the normal case),
        # Anonymeter fits a small curve (scipy.optimize.curve_fit) to rescale
        # the control-attack success rate. This fit can occasionally fail to
        # converge and raise a RuntimeError. Anonymeter's own example
        # notebook explicitly catches this and recommends retrying once
        # (optionally with a higher n_attacks for more stable queries).
        logger.warning(f"Singling-out evaluation failed to converge ({exc}); retrying once.")
        results, warn_msg = _run_with_warnings(_build_and_run)

    return results, warn_msg


def get_linkability_results(real_data, synthetic_data, control_data=None, n_attacks=100, n_neighbors=1):
    """Linkability evaluation, following anonymeter's documented usage pattern.

    `n_attacks` is clipped against both `real_data` and `control_data` sizes:
    `LinkabilityEvaluator` samples `n_attacks` rows *without replacement* from
    both the training data and (if provided) the control data, so it must not
    exceed either. PyPI 1.0.0 does not clip this internally; newer releases
    only clip against the training data, not the control set, for this
    evaluator -- so we do it defensively here regardless of version.
    """
    n_attacks = min(n_attacks, real_data.shape[0])
    if control_data is not None:
        n_attacks = min(n_attacks, control_data.shape[0])
    aux_cols = split_list(real_data.columns.to_list())

    def _build_and_run():
        evaluator = LinkabilityEvaluator(
            ori=real_data,
            syn=synthetic_data,
            control=control_data,
            aux_cols=aux_cols,
            n_attacks=n_attacks,
            n_neighbors=n_neighbors,
        )
        evaluator.evaluate(n_jobs=-2)
        return evaluator.results()

    results, warn_msg = _run_with_warnings(_build_and_run)
    return results, warn_msg


def get_inference_row(real_data, synthetic_data, control_data=None, n_attacks=1000):
    """Inference risk, aggregated over every column used as the 'secret'.

    IMPORTANT FIX: `InferenceEvaluator`'s `regression` parameter defaults to
    `False` (classification / exact-match scoring), *not* to automatic
    detection based on the column's dtype, despite what the docstring implies
    -- automatic detection only happens if you explicitly pass
    `regression=None`. The original script never passed `regression` at all,
    so every secret column -- including purely numeric/continuous ones -- was
    scored as if it were categorical, requiring an *exact* match to count as
    a successful guess. For continuous columns this makes the "real" attack
    success rate collapse to ~0, producing meaningless/deflated risk
    estimates. Here we explicitly set `regression` from the column's dtype.

    Aggregation across secret columns:
    - `inference_risk` is the mean of the per-column risk values (same as the
      original script);
    - the raw success counts are *summed* across all secret columns before
      computing the percentage, giving a properly-weighted "how often did
      the attack succeed, across every column tested" figure -- rather than
      an unweighted average of per-column percentages.

    `n_attacks` is clipped against both `real_data` and `control_data` sizes.
    `InferenceEvaluator` samples `n_attacks` rows *without replacement* from
    the control set too (`control.sample(n_attacks, replace=False)`). PyPI
    1.0.0 does not clip this internally, which is exactly what caused
    `Cannot take a larger sample than population when 'replace=False'` on
    datasets where the control (test) set is smaller than `n_attacks`.
    """
    n_attacks = min(n_attacks, real_data.shape[0])
    if control_data is not None:
        n_attacks = min(n_attacks, control_data.shape[0])
    columns = real_data.columns
    per_secret_results = []
    warn_msgs = []

    for secret in columns:
        aux_cols = [col for col in columns if col != secret]
        is_regression = _is_regression_target(real_data[secret])

        def _build_and_run(secret=secret, aux_cols=aux_cols, is_regression=is_regression):
            evaluator = InferenceEvaluator(
                ori=real_data,
                syn=synthetic_data,
                control=control_data,
                aux_cols=aux_cols,
                secret=secret,
                regression=is_regression,
                n_attacks=n_attacks,
            )
            evaluator.evaluate(n_jobs=-2)
            return evaluator.results()

        results, warn_msg = _run_with_warnings(_build_and_run)
        per_secret_results.append(results)
        if warn_msg:
            warn_msgs.append(f"{secret}: {warn_msg}")

    risk_values = [r.risk().value for r in per_secret_results]
    ci_lows = [r.risk().ci[0] for r in per_secret_results]
    ci_highs = [r.risk().ci[1] for r in per_secret_results]

    total_attack_success = sum(r.n_success for r in per_secret_results)
    total_attack_total = sum(_n_attacks_for(r, "ori") for r in per_secret_results)
    total_baseline_success = sum(r.n_baseline for r in per_secret_results)
    total_baseline_total = sum(_n_attacks_for(r, "baseline") for r in per_secret_results)

    has_control = all(r.n_control is not None for r in per_secret_results)
    total_control_success = sum(r.n_control for r in per_secret_results) if has_control else None
    total_control_total = sum(_n_attacks_for(r, "control") for r in per_secret_results) if has_control else None

    return {
        "inference_risk": compute_risk_score(float(np.mean(risk_values))),
        "inference_risk_ci_low": round(float(np.mean(ci_lows)) * 100, 2),
        "inference_risk_ci_high": round(float(np.mean(ci_highs)) * 100, 2),
        "inference_attack_n_success": total_attack_success,
        "inference_attack_n_total": total_attack_total,
        "inference_attack_success_pct": _pct(total_attack_success, total_attack_total),
        "inference_baseline_n_success": total_baseline_success,
        "inference_baseline_n_total": total_baseline_total,
        "inference_baseline_success_pct": _pct(total_baseline_success, total_baseline_total),
        "inference_control_n_success": total_control_success,
        "inference_control_n_total": total_control_total,
        "inference_control_success_pct": _pct(total_control_success, total_control_total) if has_control else None,
        "inference_warning": "; ".join(warn_msgs),
    }


# --------------------------------------------------------------------------- #
# File discovery (unchanged from the original script)
# --------------------------------------------------------------------------- #

def discover_train_test_files(train_test_dir):
    pattern = re.compile(r"^train_data_(?P<dataset>.+)_exp(?P<exp>\d+)\.csv$")
    discovered = {}
    for path in sorted(Path(train_test_dir).glob("train_data_*_exp*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        dataset = match.group("dataset")
        exp_id = int(match.group("exp"))
        discovered.setdefault(dataset, {})[exp_id] = {
            "train": path,
            "test": Path(train_test_dir) / f"test_data_{dataset}_exp{exp_id}.csv",
        }
    return discovered


def list_models(synthetic_root, selected_models=None):
    root = Path(synthetic_root)
    models = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if selected_models:
        selected = set(selected_models)
        models = [m for m in models if m in selected]
    return models


def list_synthetic_files(synthetic_root, model, dataset, exp_id):
    model_dir = Path(synthetic_root) / model
    regex = re.compile(
        rf"^synthetic_data_{re.escape(model)}_{re.escape(dataset)}_exp{exp_id}_gen(?P<gen>\d+)\.csv$"
    )
    found = []
    for path in model_dir.glob(f"synthetic_data_{model}_{dataset}_exp{exp_id}_gen*.csv"):
        match = regex.match(path.name)
        if match:
            found.append((int(match.group("gen")), path))
    found.sort(key=lambda x: x[0])
    return found


def align_synthetic_to_real(train_data, synthetic_data):
    synthetic_data = synthetic_data.loc[:, ~synthetic_data.columns.astype(str).str.startswith("Unnamed:")]
    if list(synthetic_data.columns) == list(train_data.columns):
        return synthetic_data, None

    if set(synthetic_data.columns) == set(train_data.columns):
        return synthetic_data[train_data.columns], None

    if synthetic_data.shape[1] == train_data.shape[1]:
        synthetic_data.columns = train_data.columns
        return synthetic_data, None

    missing = [c for c in train_data.columns if c not in synthetic_data.columns]
    if not missing:
        return synthetic_data[train_data.columns], None

    return None, f"Column mismatch. Missing columns in synthetic data: {missing}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    parser = ArgumentParser(
        description="Evaluate singling-out, linkability, and inference privacy risks "
                    "for synthetic datasets and write one CSV row per dataset, model, "
                    "experiment, and generation."
    )
    parser.add_argument(
        "--train-test-dir",
        type=str,
        default="./train_test_data",
        help="Directory containing train_data_<dataset>_exp<id>.csv and optional "
             "test_data_<dataset>_exp<id>.csv control sets. Default: %(default)s.",
    )
    parser.add_argument(
        "--synthetic-root",
        type=str,
        default="./synthetic_data_files",
        help="Directory containing one subdirectory per model, with files named "
             "synthetic_data_<model>_<dataset>_exp<id>_gen<id>.csv. Default: %(default)s.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=['iris', 'wine', 'parkinsons', 'concrete_compression', 'yacht_hydrodynamics',
        'airfoil_self_noise', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation',
        'seeds', 'glass', 'ecoli', 'yeast', 'planning_relax', 'blood_transfusion',
        'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'concrete_slump', 'tictactoe',
        'congress', 'car', 'wine_quality_red', 'wine_quality_white', 'california', 'bean', 'climate_model_crashes', 'libras'],
        help="Dataset names to select from the discovered training files. "
             "Default: %(default)s.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=['TabDDPM', 'CTGAN', 'GaussianCopula', 'TVAE', 'CTABGAN', 'stasy', 'forest_diffusion_vp', 'forest_diffusion_flow', 'fm_tabular'],
        help="Model subdirectory names to select under --synthetic-root. "
             "Default: %(default)s.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(Path(__file__).resolve().parent / "Results/generated/privacy/privacy_results.csv"),
        help="Output CSV path; parent directories are created as needed. "
             "Results are appended unless --overwrite is set. Default: %(default)s.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output CSV before writing results. "
             "By default, append to the existing file.",
    )
    parser.set_defaults(use_control_data=True)
    parser.add_argument(
        "--no-control-data",
        dest="use_control_data",
        action="store_false",
        help="Evaluate without control data. By default, use the matching test CSV "
             "as a control set when it exists.",
    )
    parser.add_argument(
        "--n-attacks-singling", type=int, default=500,
        help="Requested number of singling-out attacks, capped at the number of "
             "training rows. Default: %(default)s.",
    )
    parser.add_argument(
        "--n-attacks-linkability", type=int, default=500,
        help="Requested number of linkability attacks, capped at the number of "
             "training rows and, when used, control rows. Default: %(default)s.",
    )
    parser.add_argument(
        "--n-attacks-inference", type=int, default=1000,
        help="Requested number of inference attacks per secret column. Each column "
             "is evaluated using the remaining columns as auxiliary information. "
             "The count is capped at the number of training rows and, when used, "
             "control rows. Default: %(default)s.",
    )
    parser.add_argument(
        "--singling-out-mode",
        choices=["univariate", "multivariate"],
        default="multivariate",
        help="Singling-out query strategy: 'univariate' uses one column per query; "
             "'multivariate' uses combinations of --n-cols-singling columns. "
             "Default: %(default)s.",
    )
    parser.add_argument(
        "--n-cols-singling", type=int, default=3,
        help="Number of columns per multivariate singling-out query. "
             "Ignored in univariate mode. Default: %(default)s.",
    )
    parser.add_argument(
        "--n-neighbors-linkability",
        type=int,
        default=10, 
        help="Number of nearest synthetic neighbors considered for each record "
             "in the linkability attack. Default: %(default)s.",
    )
    return parser


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = build_parser().parse_args()
    discovered = discover_train_test_files(args.train_test_dir)

    if args.datasets:
        dataset_filter = set(args.datasets)
        discovered = {k: v for k, v in discovered.items() if k in dataset_filter}
    logger.info(f"Discovered datasets: {list(discovered.keys())}")
    if not discovered:
        logger.info("No datasets found in train/test directory. Exiting.")
        return

    models = list_models(args.synthetic_root, args.models)
    if not models:
        logger.info("No model folders found in synthetic root. Exiting.")
        return

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_file.exists():
        output_file.unlink()

    write_header = not output_file.exists() or output_file.stat().st_size == 0

    counters = {
        "rows_written": 0,
        "synthetic_missing": 0,
        "align_skipped": 0,
        "singling_out_failed": 0,
        "linkability_failed": 0,
        "inference_failed": 0,
    }

    for dataset in sorted(discovered.keys()):
        for exp_id in sorted(discovered[dataset].keys()):
            train_path = discovered[dataset][exp_id]["train"]
            test_path = discovered[dataset][exp_id]["test"]

            train_data = pd.read_csv(train_path).dropna().reset_index(drop=True)
            control_data = None
            if args.use_control_data and test_path.exists():
                control_data = pd.read_csv(test_path).dropna().reset_index(drop=True)

            for model in models:
                synthetic_files = list_synthetic_files(args.synthetic_root, model, dataset, exp_id)
                if not synthetic_files:
                    counters["synthetic_missing"] += 1
                    logger.info(f"[SKIP] No synthetic files for model={model}, dataset={dataset}, exp={exp_id}")
                    continue

                for gen_id, syn_path in synthetic_files:
                    logger.info(f"[RUN] model={model}, dataset={dataset}, exp={exp_id}, gen={gen_id}")
                    synthetic_data = pd.read_csv(syn_path)
                    synthetic_data, align_error = align_synthetic_to_real(train_data, synthetic_data)
                    if synthetic_data is None:
                        counters["align_skipped"] += 1
                        logger.info(f"[SKIP] Alignment failed for {syn_path.name}. {align_error}")
                        continue

                    synthetic_data = synthetic_data.iloc[: train_data.shape[0], :].reset_index(drop=True)
                    synthetic_data = normalize_float_precision(train_data, synthetic_data)
                    synthetic_data = normalize_scale(train_data, synthetic_data)
                    synthetic_data = assert_minmax(train_data, synthetic_data)

                    # Fixed schema: every column is initialized to None/"" up front so that
                    # the CSV header stays identical across rows even if a metric fails for
                    # a particular dataset/model/generation (pandas would otherwise silently
                    # change the column set between appended chunks).
                    row = {
                        "methods": model,
                        "dataset": dataset,
                        "exp_id": exp_id,
                        "gen_id": gen_id,
                        "control_data_used": control_data is not None,
                        "train_file": str(train_path),
                        "test_file": str(test_path) if control_data is not None else "",
                        "synthetic_file": str(syn_path),
                    }
                    for prefix in ("singling_out", "linkability", "inference"):
                        row.update({
                            f"{prefix}_risk": None,
                            f"{prefix}_risk_ci_low": None,
                            f"{prefix}_risk_ci_high": None,
                            f"{prefix}_attack_n_success": None,
                            f"{prefix}_attack_n_total": None,
                            f"{prefix}_attack_success_pct": None,
                            f"{prefix}_baseline_n_success": None,
                            f"{prefix}_baseline_n_total": None,
                            f"{prefix}_baseline_success_pct": None,
                            f"{prefix}_control_n_success": None,
                            f"{prefix}_control_n_total": None,
                            f"{prefix}_control_success_pct": None,
                            f"{prefix}_warning": "",
                        })

                    # Each metric is computed and captured independently: a failure in one
                    # (e.g. a non-converging curve fit in the singling-out evaluator) no
                    # longer discards the other two, otherwise-valid, results.
                    try:
                        so_results, so_warn = get_singling_out_results(
                            real_data=train_data,
                            synthetic_data=synthetic_data,
                            control_data=control_data,
                            n_attacks=args.n_attacks_singling,
                            mode=args.singling_out_mode,
                            n_cols=args.n_cols_singling,
                        )
                        row.update(_row_from_results(so_results, so_warn, "singling_out"))
                    except Exception as exc:
                        counters["singling_out_failed"] += 1
                        row["singling_out_warning"] = f"FAILED: {exc}"
                        logger.warning(f"[SKIP] Singling-out evaluation failed for {syn_path.name}. Error: {exc}")

                    try:
                        link_results, link_warn = get_linkability_results(
                            real_data=train_data,
                            synthetic_data=synthetic_data,
                            control_data=control_data,
                            n_attacks=args.n_attacks_linkability,
                            n_neighbors=args.n_neighbors_linkability,
                        )
                        row.update(_row_from_results(link_results, link_warn, "linkability"))
                    except Exception as exc:
                        counters["linkability_failed"] += 1
                        row["linkability_warning"] = f"FAILED: {exc}"
                        logger.warning(f"[SKIP] Linkability evaluation failed for {syn_path.name}. Error: {exc}")

                    try:
                        inference_row = get_inference_row(
                            real_data=train_data,
                            synthetic_data=synthetic_data,
                            control_data=control_data,
                            n_attacks=args.n_attacks_inference,
                        )
                        row.update(inference_row)
                    except Exception as exc:
                        counters["inference_failed"] += 1
                        row["inference_warning"] = f"FAILED: {exc}"
                        logger.warning(f"[SKIP] Inference evaluation failed for {syn_path.name}. Error: {exc}")

                    pd.DataFrame([row]).to_csv(
                        output_file,
                        mode="a",
                        header=write_header,
                        index=False,
                    )
                    write_header = False
                    counters["rows_written"] += 1

    logger.info("Done.")
    logger.info(f"Rows written: {counters['rows_written']}")
    logger.info(f"Missing synthetic groups: {counters['synthetic_missing']}")
    logger.info(f"Skipped for alignment issues: {counters['align_skipped']}")
    logger.info(f"Singling-out failures: {counters['singling_out_failed']}")
    logger.info(f"Linkability failures: {counters['linkability_failed']}")
    logger.info(f"Inference failures: {counters['inference_failed']}")


if __name__ == "__main__":
    main()
