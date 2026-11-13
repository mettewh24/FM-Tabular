# Synthetic Tabular Data Generation via Flow Matching with Entropic Regularized Optimal Transport

Research code for the paper *Synthetic Tabular Data Generation via Flow Matching with Entropic Regularized Optimal Transport*. **FM-Tabular** generates synthetic tabular data using conditional flow matching and minibatch entropic optimal transport. The benchmark compares nine generators across 27 datasets, measuring fidelity, downstream utility, runtime, and privacy risk.

## Repository contents

| Path | Purpose |
| --- | --- |
| [`ot_flow/`](ot_flow/) | FM-Tabular model, transport couplings, and sampling |
| [`script_generation.py`](script_generation.py) | Generator training and benchmark evaluation |
| [`data_loaders.py`](data_loaders.py), [`download_data.py`](download_data.py) | Dataset acquisition and preprocessing |
| [`evaluate_privacy.py`](evaluate_privacy.py) | Singling-out, linkability, and inference attacks |
| [`Results/reference/`](Results/reference/) | Included benchmark, privacy, and Iris ablation metrics |
| [`Results/generation_script.R`](Results/generation_script.R), [`Results/aggregate_privacy_results.py`](Results/aggregate_privacy_results.py) | Summary tables |
| [`plot_metrics_vs_epsilon.py`](plot_metrics_vs_epsilon.py), [`iris_scatter_comparison.py`](iris_scatter_comparison.py) | Ablation and scatter plots |
| [`tests/`](tests/) | FM-Tabular regression tests |

Run all commands from the repository root. Generated outputs go under `Results/generated/`; downloaded data, train/test splits, and synthetic samples are excluded from version control.

## Rebuild tables and ablation plots

The three CSVs in `Results/reference/` contain the metrics needed for these analyses. No dataset downloads or model training are required. Input checksums are listed in [`SHA256SUMS`](Results/reference/SHA256SUMS).

Create an analysis environment with Python 3.11 or 3.12:

```sh
python -m venv .venv-analysis
```

Activate it with `.\.venv-analysis\Scripts\Activate.ps1` in PowerShell or `source .venv-analysis/bin/activate` on Linux/macOS, then install:

```sh
python -m pip install -r requirements-analysis.txt
```

R is also required for the benchmark tables; the script uses only base R.

```sh
Rscript Results/generation_script.R
python Results/aggregate_privacy_results.py --input-csv Results/reference/privacy.csv --output-csv Results/generated/tables/paper_privacy.csv --output-tex Results/generated/tables/paper_privacy.tex
python plot_metrics_vs_epsilon.py --input-csv Results/reference/iris_ablation.csv --dataset iris --output-dir Results/generated/iris-ablation --ngen 5 --nominal-se-n 15 --std-ddof 0 --error-bars se --dpi 600 --formats png pdf
```

| Analysis | Outputs under `Results/generated/` |
| --- | --- |
| Benchmark | `tables/paper_quality.csv`, `tables/paper_syndat.csv`, `tables/paper_utility.csv` |
| Privacy | `tables/paper_privacy.csv`, `tables/paper_privacy.tex` |
| Iris ablation | Combined and individual PNG/PDF plots in `iris-ablation/iris/` |

These commands overwrite their outputs. The LaTeX table requires `\usepackage{booktabs}`.

Benchmark summaries report an unweighted mean and sample standard error across dataset means, using the metric-specific subsets defined in the R script. Privacy risks are averaged within each dataset before summarizing across datasets; warnings exclude evaluations separately for each risk metric, and displayed means and standard errors are rounded upward on the 0–100 scale.

The Iris ablation compares ten Sinkhorn regularization values with exact OT. Quality error bars use the nominal `SD(ddof=0) / sqrt(15)` for three trainings and five generations per training. Generations from the same model are dependent, so these bars do not estimate independent between-training uncertainty. Runtime uses `SD(ddof=0) / sqrt(3 - 1)`.

## Standalone FM-Tabular example

For a small example without the full benchmark dependencies, use a separate environment:

```sh
python -m pip install "numpy<2" "pandas<3" torch torchdiffeq scikit-learn POT tqdm
python -m ot_flow.example_generation
python -m unittest discover -s tests -p test_integer_columns.py
```

The example trains on toy data and saves and reloads `Results/generated/fm_tabular_generator.pt`. The tests cover integer-column post-processing and checkpoint behavior.

## Run experiments

### Environment and data

Use a separate generation environment and install [`requirements.txt`](requirements.txt):

```sh
python -m venv .venv-benchmark
```

Activate it as above, replacing `.venv-analysis` with `.venv-benchmark`, then run:

```sh
python -m pip install -r requirements.txt
python -m pip check
python download_data.py
```

Uncached datasets require internet access. Supported dataset names and preprocessing are defined in [`data_loaders.py`](data_loaders.py).

The benchmark imports all generators at startup, including when selecting only FM-Tabular. A clean installation of the full benchmark has not been validated, and generation dependencies are not fully pinned. Check compatibility before training, particularly the bundled TabDDPM dependency constraints. TabDDPM is configured for `cuda:0` in [`config.toml`](TabDDPM/config/config.toml); FM-Tabular uses CUDA when available and otherwise CPU.

### Benchmark

The protocol uses three 80/20 train/test splits and five synthetic generations per training. Classification targets are conditioned on; regression targets are modeled jointly. `--n_tries 5` controls repeated downstream utility fits.

Use a dedicated working copy for each experiment. Before running TabDDPM, create its output parent directory:

```sh
python -c "from pathlib import Path; Path('TabDDPM/exp').mkdir(parents=True, exist_ok=True)"
```

Run the following commands sequentially to evaluate all nine generators. FM-Tabular defaults to Sinkhorn with epsilon 0.5, 2,000 epochs, and 100 Euler steps.

```sh
python script_generation.py --methods fm_tabular GaussianCopula TVAE CTGAN CTABGAN stasy TabDDPM --nexp 3 --ngen 5 --n_tries 5 --ycond True --out_path Results/generated/benchmark/metrics.csv
python script_generation.py --methods forest_diffusion --diffusion_type flow --n_t 50 --n_t_sampling 50 --nexp 3 --ngen 5 --n_tries 5 --ycond True --out_path Results/generated/benchmark/metrics.csv
python script_generation.py --methods forest_diffusion --diffusion_type vp --n_t 50 --n_t_sampling 50 --nexp 3 --ngen 5 --n_tries 5 --ycond True --out_path Results/generated/benchmark/metrics.csv
```

Add `--datasets iris` to restrict a run to Iris. For FM-Tabular ablations, use `--fm_epsilon <value>` or `--fm_ot_solver exact`.

Metrics append to the output CSV. Splits in `train_test_data/` and samples in `synthetic_data_files/` overwrite matching filenames; TabDDPM also reuses temporary directories. Start with a fresh metrics file, archive samples before another configuration, and run one job at a time per working copy. Check logs and FM's `completed_runs=3/3` labels for incomplete runs.

Before aggregating new benchmark results, manually standardize the names in the `method` column to match `Results/reference/benchmark.csv`. The scripts do not perform this renaming automatically.

| Benchmark label | Label required by the table script |
| --- | --- |
| `fm_tabular hidden_dim=512 ...` | `fm_tabular` |
| `forest_diffusion ... diffusion=flow ...` | `forest_diffusion_flow` |
| `forest_diffusion ... diffusion=vp ...` | `forest_diffusion_vp` |

Leave the other method names unchanged. If multiple configurations exist for a dataset and method, select the configuration to include before renaming. Keep exactly one row per dataset and method, and save the resulting copy as `Results/generated/benchmark-for-tables.csv`:

```sh
Rscript Results/generation_script.R Results/generated/benchmark-for-tables.csv Results/generated/new-tables
```

### Privacy evaluation

If another virtual environment is active, run `deactivate`. Using Python 3.11, create a separate privacy environment:

```sh
python -m venv .venv-privacy
```

Activate it with `.\.venv-privacy\Scripts\Activate.ps1` in PowerShell or `source .venv-privacy/bin/activate` on Linux/macOS, then install the dependencies and evaluate the generated samples:

```sh
python -m pip install -r requirements-privacy.txt
python -m pip check
python evaluate_privacy.py --train-test-dir train_test_data --synthetic-root synthetic_data_files --output-file Results/generated/privacy/privacy_results.csv
```

The defaults evaluate all nine methods using the matching test splits as controls, with up to 500 singling-out attacks, 500 linkability attacks, and 1,000 inference attacks. Results append by default; use a fresh output file for each evaluation. Check the reported missing samples and attack failures.

Aggregate the results in the analysis environment:

```sh
python Results/aggregate_privacy_results.py --input-csv Results/generated/privacy/privacy_results.csv --output-csv Results/generated/privacy/paper_privacy.csv --output-tex Results/generated/privacy/paper_privacy.tex
```

### Iris scatter plots

After generating Iris samples, run in the generation environment:

```sh
python iris_scatter_comparison.py --exp 0 --generation 0 --train-root train_test_data --synthetic-root synthetic_data_files --output-dir Results/generated/iris-scatter
```

The four-panel figure uses real data, TVAE, Forest Diffusion Flow, and FM-Tabular. The all-models figure includes available samples; check that all expected methods are present. Train/test splits and synthetic samples are not bundled with the repository.

For reproducible experiments, archive commands, code revision, resolved dependencies, hardware details, and logs alongside the results. New training runs need not reproduce the included metrics exactly.

## Attribution and license

Original contributions are distributed under the [MIT License](LICENSE). The benchmark builds on ForestDiffusion and includes third-party baseline implementations. See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and applicable licensing information.
