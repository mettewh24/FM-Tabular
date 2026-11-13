"""Compare real Iris train data with four synthetic generators."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from syndat.postprocessing import assert_minmax, normalize_float_precision, normalize_scale

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "species"
SPECIES_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}
MARKER = "o"
MARKER_SIZE = 42
MARKER_ALPHA = 0.75
MODEL_DIRS = {
    "TVAE": "TVAE",
    "TABDDPM": "TabDDPM",
    "Forest Diffusion Flow": "forest_diffusion_flow",
    "Forest Diffusion VP": "forest_diffusion_vp",
    "FM Tabular": "fm_tabular",
    "CTGAN": "CTGAN",
#    "Copula GAN": "copulaGAN",
    "CTABGAN": "CTABGAN",
    "Gaussian Copula": "GaussianCopula",
    "StaSy": "stasy",
}
FOUR_MODEL_LABELS = ("TVAE", "Forest Diffusion Flow", "FM Tabular")


def format_axis_label(feature: str) -> str:
    """Convert an internal feature name into a readable axis label."""
    return feature.replace("_", " ").title()


def load_iris_csv(path: Path) -> pd.DataFrame:
    """Load a headerless or headered Iris CSV and normalize its schema."""
    frame = pd.read_csv(path)
    if list(frame.columns) != [*map(str, range(5))]:
        frame = pd.read_csv(path, header=None)

    if frame.shape[1] != 5:
        raise ValueError(f"Expected 5 columns in {path}, found {frame.shape[1]}")

    frame.columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN]).copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].round().astype(int)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].map(SPECIES_NAMES)
    frame = frame.dropna(subset=[TARGET_COLUMN])
    return frame


def postprocess_synthetic(train: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Apply the repository's Syndat post-processing to synthetic features."""
    train_features = train[FEATURE_COLUMNS].copy()
    synthetic = synthetic.copy()
    synthetic_features = normalize_float_precision(
        train_features,
        synthetic[FEATURE_COLUMNS].copy(),
    )
    synthetic_features = normalize_scale(train_features, synthetic_features)
    synthetic_features = assert_minmax(train_features, synthetic_features)
    synthetic.loc[:, FEATURE_COLUMNS] = synthetic_features
    return synthetic


def make_plot(
    train_path: Path,
    synthetic_paths: dict[str, Path],
    output_path: Path,
    x_feature: str,
    y_feature: str,
    ncols: int,
    title: str,
) -> None:
    train = load_iris_csv(train_path)
    frames = {"Real": train}
    frames.update(
        {
            label: postprocess_synthetic(train, load_iris_csv(path))
            for label, path in synthetic_paths.items()
        }
    )

    nrows = math.ceil(len(frames) / ncols)
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.4 * ncols, 3.8 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = {
        "Real": "#1f2937",
        "setosa": "#007c91",
        "versicolor": "#d97706",
        "virginica": "#b42318",
    }

    for panel_index, (label, frame) in enumerate(frames.items()):
        row_index, column_index = divmod(panel_index, ncols)
        axis = axes[row_index][column_index]
        for species in SPECIES_NAMES.values():
            subset = frame[frame[TARGET_COLUMN] == species]
            axis.scatter(
                subset[x_feature],
                subset[y_feature],
                s=MARKER_SIZE,
                alpha=MARKER_ALPHA,
                color=colors[species],
                marker=MARKER,
                edgecolors="white",
                linewidths=0.35,
                label=species,
            )
            axis.grid(alpha=0.18)
        axis.set_title(label, fontsize=11, fontweight="bold")
        axis.set_xlabel(format_axis_label(x_feature))

        axis.legend(loc="best", frameon=False, fontsize=8)

        if column_index == 0:
            axis.set_ylabel(format_axis_label(y_feature))

    for panel_index in range(len(frames), nrows * ncols):
        row_index, column_index = divmod(panel_index, ncols)
        axes[row_index][column_index].set_visible(False)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", type=int, default=0, help="Experiment/seed index.")
    parser.add_argument("--generation", type=int, default=0, help="Synthetic generation index.")
    parser.add_argument("--synthetic-root", type=Path, default=Path("synthetic_data_files"))
    parser.add_argument("--train-root", type=Path, default=Path("train_test_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("iris_plots"))
    parser.add_argument("--x", dest="x_feature", choices=FEATURE_COLUMNS, default="sepal_length")
    parser.add_argument("--y", dest="y_feature", choices=FEATURE_COLUMNS, default="petal_length")
    return parser.parse_args()


def find_synthetic_paths(
    synthetic_root: Path,
    exp: int,
    generation: int,
    model_dirs: dict[str, str],
) -> dict[str, Path]:
    paths = {}
    for label, model_dir in model_dirs.items():
        path = (
            synthetic_root
            / model_dir
            / f"synthetic_data_{model_dir}_iris_exp{exp}_gen{generation}.csv"
        )
        if path.exists():
            paths[label] = path
    return paths


def main() -> None:
    args = parse_args()
    train_path = args.train_root / f"train_data_iris_exp{args.exp}.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Real train file not found: {train_path}")

    all_paths = find_synthetic_paths(
        args.synthetic_root,
        args.exp,
        args.generation,
        MODEL_DIRS,
    )
    four_paths = {
        label: all_paths[label]
        for label in FOUR_MODEL_LABELS
        if label in all_paths
    }
    if len(four_paths) != len(FOUR_MODEL_LABELS):
        missing = [label for label in FOUR_MODEL_LABELS if label not in four_paths]
        raise FileNotFoundError(f"Missing files for four-model plot: {missing}")

    four_output = args.output_dir / "iris_scatter_four_models.png"
    make_plot(
        train_path,
        four_paths,
        four_output,
        args.x_feature,
        args.y_feature,
        ncols=2,
        title="Iris: real and selected models",
    )
    print(f"Saved {four_output}")

    all_output = args.output_dir / "iris_scatter_all_models.png"
    make_plot(
        train_path,
        all_paths,
        all_output,
        args.x_feature,
        args.y_feature,
        ncols=3,
        title="Iris: real train and all models",
    )
    print(f"Saved {all_output}")


if __name__ == "__main__":
    main()
