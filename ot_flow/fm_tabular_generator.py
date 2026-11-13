from __future__ import annotations

import copy
import numbers
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .models import TabularFlowNet
from .utils import OTDataLoader, TabularDataset, flow_matching_loss, mixed_cost, odeint_integrate, ADAPTIVE_ODE_METHODS


def _unique_in_order(values: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _resolve_names(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]],
    indices: Optional[Sequence[int]],
) -> list[str]:
    resolved: list[str] = []
    if columns is not None:
        resolved.extend([str(col) for col in columns])
    if indices is not None:
        for index in indices:
            resolved.append(str(df.columns[int(index)]))
    return _unique_in_order(resolved)


def _is_categorical_series(series: pd.Series, max_cat_unique: int) -> bool:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(series.dtype):
        return True
    if pd.api.types.is_object_dtype(series.dtype):
        return True
    if pd.api.types.is_integer_dtype(series.dtype):
        return series.nunique(dropna=True) <= max_cat_unique
    return False


def _infer_partition(
    df: pd.DataFrame,
    continuous_cols: Optional[Sequence[str]] = None,
    continuous_idx: Optional[Sequence[int]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    categorical_idx: Optional[Sequence[int]] = None,
    max_cat_unique: int = 20,
    integer_cols: Optional[Sequence[str]] = None,
    integer_idx: Optional[Sequence[int]] = None,
) -> tuple[list[str], list[str]]:
    continuous_overrides = set(_resolve_names(df, continuous_cols, continuous_idx))
    # Integer-valued numeric columns use the continuous training path.
    continuous_overrides.update(_resolve_names(df, integer_cols, integer_idx))
    categorical_overrides = set(_resolve_names(df, categorical_cols, categorical_idx))

    overlap = continuous_overrides & categorical_overrides
    if overlap:
        raise ValueError(f"Columns cannot be both continuous and categorical: {sorted(overlap)}")

    unknown = (continuous_overrides | categorical_overrides) - set(df.columns)
    if unknown:
        raise ValueError(f"Unknown column(s) in continuous/categorical overrides: {sorted(unknown)}")

    inferred_categorical = []
    for column in df.columns:
        if column in categorical_overrides:
            inferred_categorical.append(column)
            continue
        if column in continuous_overrides:
            continue
        if _is_categorical_series(df[column], max_cat_unique=max_cat_unique):
            inferred_categorical.append(column)

    categorical = _unique_in_order(inferred_categorical)
    continuous = [column for column in df.columns if column not in categorical]
    return continuous, categorical


def _preprocess_dataframe(
    df: pd.DataFrame,
    continuous_cols: list[str],
    categorical_cols: list[str],
    integer_cols: Optional[Sequence[str]] = None,
):
    integer_columns = _unique_in_order(
        (list(integer_cols) if integer_cols is not None else [])
        + [column for column in continuous_cols if pd.api.types.is_integer_dtype(df[column].dtype)]
    )
    for column in integer_columns:
        if column not in continuous_cols:
            raise ValueError(f"Integer column '{column}' must be a numeric continuous column.")
        series = df[column]
        if not (
            pd.api.types.is_integer_dtype(series.dtype)
            or pd.api.types.is_float_dtype(series.dtype)
        ):
            raise ValueError(f"Integer column '{column}' must have an integer or float dtype.")
        if series.isna().any() or not np.isfinite(series).all() or not (series == np.rint(series)).all():
            raise ValueError(f"Integer column '{column}' must contain only finite integer-valued data.")

    scaler = None
    df_cont = pd.DataFrame(index=df.index)
    if continuous_cols:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_cont = pd.DataFrame(
            scaler.fit_transform(df[continuous_cols]),
            columns=continuous_cols,
            index=df.index,
        )

    df_cat = pd.DataFrame(index=df.index)
    dummy_map: dict[str, list[str]] = {}
    categorical_levels: dict[str, list] = {}
    categorical_dtypes: dict[str, object] = {}

    for column in categorical_cols:
        series = df[column]
        categorical_dtypes[column] = series.dtype
        levels = list(pd.Series(series).dropna().unique())
        categorical_levels[column] = levels
        categorical_series = pd.Categorical(series, categories=levels)
        dummies = pd.get_dummies(
            categorical_series,
            prefix=column,
            prefix_sep="__",
            dummy_na=False,
        ).astype(np.float32)
        # pd.Categorical drops the Series index; restore it before concatenation.
        dummies.index = df.index
        # Encode as {-1, 1} rather than {0, 1} so categorical columns live on
        # the same scale as the min-max-scaled continuous columns ([-1, 1]).
        # This keeps the target velocity (x1 - x0) comparable in magnitude
        # across feature types, since x0 is standard Gaussian noise centered
        # at 0. The transform is monotonic per-column, so decoding via
        # argmax (see _inverse_preprocess_dataframe) is unaffected.
        dummies = dummies * 2.0 - 1.0
        dummy_map[column] = list(dummies.columns)
        df_cat = pd.concat([df_cat, dummies], axis=1)

    df_proc = pd.concat([df_cont, df_cat], axis=1)
    info = {
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
        "categorical_levels": categorical_levels,
        "categorical_dtypes": categorical_dtypes,
        "integer_cols": integer_columns,
        "integer_dtypes": {column: df[column].dtype for column in integer_columns},
        "dummy_map": dummy_map,
        "dummy_columns": list(df_cat.columns),
        "processed_columns": list(df_proc.columns),
        "original_columns": list(df.columns),
        "scaler": scaler,
        "preprocessor_type": "minmax_onehot_with_levels",
    }
    return df_proc, info


def _inverse_preprocess_dataframe(df_processed: pd.DataFrame, info: dict) -> pd.DataFrame:
    df_out = pd.DataFrame(index=df_processed.index)

    continuous_cols = info["continuous_cols"]
    if continuous_cols:
        cont_data = df_processed[continuous_cols].to_numpy()
        # Bound continuous values in scaled space before restoring their units.
        cont_data = np.clip(cont_data, -1.0, 1.0)
        scaler = info.get("scaler")
        if scaler is not None:
            cont_data = scaler.inverse_transform(cont_data)
        df_out[continuous_cols] = cont_data

        # Round in original units, preserving float dtypes for integer-valued
        # float inputs. Missing keys keep older checkpoints compatible.
        for column in info.get("integer_cols", []):
            df_out[column] = np.rint(df_out[column]).astype(info["integer_dtypes"][column])

    for column in info["categorical_cols"]:
        dummy_cols = info["dummy_map"].get(column, [])
        levels = info["categorical_levels"].get(column, [])
        original_dtype = info["categorical_dtypes"].get(column)

        if not dummy_cols:
            df_out[column] = pd.Series([np.nan] * len(df_processed), index=df_processed.index)
            continue

        dummy_values = df_processed[dummy_cols].to_numpy()
        if dummy_values.size == 0:
            decoded = [np.nan] * len(df_processed)
        else:
            # Use raw categorical scores: clipping can create ties and change
            # which category has the highest score.
            decoded_idx = dummy_values.argmax(axis=1)
            decoded = [levels[idx] if idx < len(levels) else None for idx in decoded_idx]

        if isinstance(original_dtype, pd.CategoricalDtype):
            df_out[column] = pd.Categorical(decoded, categories=levels)
        else:
            df_out[column] = pd.Series(decoded, index=df_processed.index)
            has_missing = df_out[column].isna().any()
            if has_missing and pd.api.types.is_integer_dtype(original_dtype):
                # Can't cast a column containing NaN/None to an integer dtype;
                # leave as object/float rather than raising.
                continue
            try:
                df_out[column] = df_out[column].astype(original_dtype)
            except (ValueError, TypeError):
                pass

    return df_out[info["original_columns"]]


class FMTabularGenerator(BaseEstimator):
    """Tabular data generator based on OT flow matching.

    A scikit-learn-style estimator (similar in spirit to ForestDiffusion):
    instantiate with hyperparameters, call :meth:`fit` on a pandas
    DataFrame, then draw synthetic rows with :meth:`generate` /
    :meth:`sample`.

    Continuous columns are min-max scaled to [-1, 1]; categorical columns
    (inferred from dtype/cardinality, or specified explicitly) are one-hot
    encoded into {-1, 1} so both feature types share the same scale. A
    velocity field (:class:`TabularFlowNet`) is trained with
    OT flow matching and sampled via an ODE solver.

    Integer-valued numeric columns follow the continuous training path and
    are rounded after inverse scaling, retaining their original dtype.

    When `y_col` is passed to :meth:`fit` and the target is categorical or
    binary, the model is additionally conditioned on the class label via a
    learned embedding injected through the FiLM layers of the velocity
    field. At generation time, pass `y` to :meth:`generate` to draw samples
    from a specific class (or a mix of classes).

    Parameters
    ----------
    device : str
        Torch device, e.g. ``"cpu"`` or ``"cuda"``.
    hidden_dim : int
        Width of the hidden layers of the velocity field.
    time_dim : int
        Dimension of the sinusoidal time embedding.
    depth : int
        Number of FiLM-conditioned residual blocks.
    dropout : float
        Dropout rate inside the velocity field (0 disables dropout).
    max_cat_unique : int
        Integer columns with at most this many distinct values are
        treated as categorical when ``categorical_cols``/``continuous_cols``
        are not given explicitly.
    class_embed_dim : int
        Dimension of the class embedding used for y-conditioning.
        The FiLM conditioning vector has size ``time_dim + class_embed_dim``
        when y-conditioning is active.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        hidden_dim: int = 256,
        time_dim: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
        max_cat_unique: int = 20,
        class_embed_dim: int = 64,
    ):
        self.device = device
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.max_cat_unique = int(max_cat_unique)
        self.class_embed_dim = int(class_embed_dim)

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.time_dim <= 0 or self.time_dim % 2 != 0:
            raise ValueError("time_dim must be a positive even integer.")
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")
        if self.max_cat_unique <= 0:
            raise ValueError("max_cat_unique must be positive.")
        if self.class_embed_dim <= 0:
            raise ValueError("class_embed_dim must be positive.")

        self.model_: Optional[TabularFlowNet] = None
        self.preprocess_info_: Optional[dict] = None
        self.is_fitted_: bool = False
        self.loss_history_: list[float] = []

        # set after fit() when y_col is provided
        self.y_col_: Optional[str] = None
        self.y_levels_: Optional[list] = None
        self.y_proportions_: Optional[np.ndarray] = None
        self.n_classes_: int = 0

    def _build_model(self, input_dim: int, n_classes: int = 0) -> TabularFlowNet:
        return TabularFlowNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            time_dim=self.time_dim,
            depth=self.depth,
            dropout=self.dropout,
            n_classes=n_classes,
            class_embed_dim=self.class_embed_dim,
        )

    def _build_cost_fn(self, info: dict):
        col_index = {column: idx for idx, column in enumerate(info["processed_columns"])}
        continuous_idx = [col_index[column] for column in info["continuous_cols"] if column in col_index]
        categorical_dummy_idx = [col_index[column] for column in info["dummy_columns"] if column in col_index]
        if continuous_idx or categorical_dummy_idx:
            return lambda x0, x1, ci=continuous_idx, di=categorical_dummy_idx: mixed_cost(x0, x1, ci, di)
        return None

    def fit(
        self,
        df: pd.DataFrame,
        *,
        y_col: Optional[str] = None,
        num_epochs: int = 50,
        batch_size: int = 256,
        epsilon: float = 0.1,
        lr: float = 1e-3,
        ot_solver: str = "sinkhorn",
        categorical_cols: Optional[Sequence[str]] = None,
        categorical_idx: Optional[Sequence[int]] = None,
        continuous_cols: Optional[Sequence[str]] = None,
        continuous_idx: Optional[Sequence[int]] = None,
        integer_cols: Optional[Sequence[str]] = None,
        integer_idx: Optional[Sequence[int]] = None,
        max_cat_unique: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> "FMTabularGenerator":
        """Fit the generator on `df`.

        Parameters
        ----------
        df : pd.DataFrame
            Training data. Must contain at least one row.
        y_col : str, optional
            Name of a categorical or binary target column to condition the
            velocity field on. When provided, the model learns a separate
            FiLM path per class via a learned embedding. The column is kept
            in the training data so joint (X, y) generation is still
            possible; pass ``y`` to :meth:`generate` to pin a specific class.
        integer_cols, integer_idx : sequence, optional
            Names or positions of integer-valued numeric columns, including
            those stored as floats. These columns are modeled numerically
            and rounded to the nearest integer after inverse scaling
            (halfway values round to even), keeping the input dtype.
            They cannot also be categorical. Integer dtypes among numeric
            columns are detected automatically; float columns require an
            explicit declaration. Training values must already be integral.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"`df` must be a pandas DataFrame, got {type(df)!r}.")
        if len(df) == 0:
            raise ValueError("`df` must contain at least one row.")
        if not isinstance(num_epochs, numbers.Integral) or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer.")
        if not isinstance(batch_size, numbers.Integral) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(epsilon, numbers.Real) or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be a positive finite number.")
        if not isinstance(lr, numbers.Real) or not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be a positive finite number.")

        missing = df.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            details = ", ".join(f"{column}: {count}" for column, count in missing.items())
            raise ValueError(
                "Input DataFrame contains missing values; impute or remove them "
                f"before fit(). Columns/counts: {details}."
            )

        numeric_data = df.select_dtypes(include=[np.number])
        if not np.isfinite(numeric_data.to_numpy()).all():
            raise ValueError("Input DataFrame contains non-finite numeric values.")

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        # ── y-conditioning setup ──────────────────────────────────────────
        label_y: Optional[np.ndarray] = None
        n_classes = 0
        if y_col is not None:
            if y_col not in df.columns:
                raise ValueError(f"y_col='{y_col}' not found in DataFrame.")
            raw_y = df[y_col].to_numpy()
            n_missing = int(pd.isna(raw_y).sum())
            if n_missing > 0:
                raise ValueError(
                    f"y_col='{y_col}' contains {n_missing} missing value(s). "
                    "Class-conditioning does not support NaN targets: drop or "
                    "impute the missing rows/values before calling fit()."
                )
            y_levels = list(pd.Series(raw_y).dropna().unique())
            level_to_idx = {v: i for i, v in enumerate(y_levels)}
            label_y = np.array([level_to_idx[v] for v in raw_y], dtype=np.int64)
            n_classes = len(y_levels)
            # Empirical class proportions — used at generate() time when no
            # explicit y is passed, to reproduce the training distribution.
            counts = np.bincount(label_y, minlength=n_classes)
            self.y_col_ = y_col
            self.y_levels_ = y_levels
            self.y_proportions_ = counts / counts.sum()   # shape (n_classes,)
            self.n_classes_ = n_classes
        else:
            self.y_col_ = None
            self.y_levels_ = None
            self.y_proportions_ = None
            self.n_classes_ = 0

        # ── preprocessing ─────────────────────────────────────────────────
        cat_max = self.max_cat_unique if max_cat_unique is None else int(max_cat_unique)
        continuous, categorical = _infer_partition(
            df,
            continuous_cols=continuous_cols,
            continuous_idx=continuous_idx,
            categorical_cols=categorical_cols,
            categorical_idx=categorical_idx,
            max_cat_unique=cat_max,
            integer_cols=integer_cols,
            integer_idx=integer_idx,
        )

        df_proc, info = _preprocess_dataframe(
            df, continuous, categorical,
            integer_cols=_resolve_names(df, integer_cols, integer_idx),
        )
        if not info["processed_columns"]:
            raise ValueError("No usable columns found after preprocessing.")
        self.preprocess_info_ = info

        # ── dataloader ────────────────────────────────────────────────────
        dataset = TabularDataset(df_proc, device=self.device)
        cost_fn = self._build_cost_fn(info)
        loader = OTDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            epsilon=epsilon,
            cost_fn=cost_fn,
            ot_solver=ot_solver,
            device=self.device,
            label_y=label_y,   # None when no y-conditioning
        )

        # ── model + optimiser ─────────────────────────────────────────────
        self.model_ = self._build_model(
            input_dim=len(info["processed_columns"]),
            n_classes=n_classes,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(num_epochs), 1),
            eta_min=lr * 0.01,
        )

        # ── training loop ─────────────────────────────────────────────────
        self.loss_history_.clear()
        epoch_iter = tqdm(range(int(num_epochs)), desc="Training", disable=not verbose)

        for _epoch in epoch_iter:
            self.model_.train()
            total_loss = 0.0
            n_batches = 0

            # The loader yields (x0, x1, y_batch) where y_batch is None
            # when label_y was not provided. y_batch is already reindexed
            # by j (the OT column index) so it always aligns with x1[j].
            for x0, x1, y_batch in loader:
                t = torch.rand(len(x0), 1, device=self.device) * 0.98 + 0.01
                loss = flow_matching_loss(self.model_, x0, x1, t, y=y_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            self.loss_history_.append(avg_loss)
            if verbose:
                epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")

        self.is_fitted_ = True
        return self

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        *,
        y: Optional[np.ndarray] = None,
        n_steps: Optional[int] = None,
        method: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        seed: Optional[int] = None,
        **odeint_kwargs,
    ) -> pd.DataFrame:
        """Draw `n_samples` synthetic rows.

        Parameters
        ----------
        n_samples : int
            Number of rows to generate.
        y : array-like of shape (n_samples,), optional
            Desired class labels for each generated row. Must contain values
            from the same set seen during ``fit(y_col=...)``. When omitted
            (or when the model was trained without ``y_col``), the ODE is
            integrated unconditionally.
        n_steps : int, optional
            Number of ODE integration steps. Only used for fixed-step
            solvers (``'euler'``, ``'midpoint'``, ``'rk4'``); ignored (with
            a warning) for adaptive solvers, which pick their own step
            sizes from `rtol`/`atol`. Defaults to 100 when a fixed-step
            method is used.
        method : str
            ODE solver (``'dopri5'``, ``'bosh3'``, ``'euler'``, ``'rk4'``, ...).
            Defaults to ``'dopri5'`` (adaptive).
        rtol, atol : float
            Relative/absolute tolerance for adaptive solvers. Ignored for
            fixed-step solvers.
        seed : int, optional
            RNG seed for the initial Gaussian noise.

        Returns
        -------
        pd.DataFrame
            Synthetic data with the same column names, ordering and
            (best-effort) dtypes as the DataFrame passed to :meth:`fit`.
        """
        if not self.is_fitted_ or self.model_ is None or self.preprocess_info_ is None:
            raise RuntimeError("Call fit() before generate().")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}.")
        if n_steps is not None and (
            not isinstance(n_steps, numbers.Integral) or n_steps <= 0
        ):
            raise ValueError("n_steps must be a positive integer when provided.")
        if not isinstance(rtol, numbers.Real) or not np.isfinite(rtol) or rtol <= 0:
            raise ValueError("rtol must be a positive finite number.")
        if not isinstance(atol, numbers.Real) or not np.isfinite(atol) or atol <= 0:
            raise ValueError("atol must be a positive finite number.")

        # ── resolve y labels ──────────────────────────────────────────────
        y_tensor: Optional[torch.Tensor] = None

        if self.y_levels_ is not None:
            level_to_idx = {v: i for i, v in enumerate(self.y_levels_)}

            if y is None:
                rng = np.random.default_rng(seed)
                y_idx = rng.choice(
                    len(self.y_levels_),
                    size=n_samples,
                    p=self.y_proportions_,
                )
            else:
                y_arr = np.asarray(y)
                if y_arr.ndim == 0:
                    scalar_val = y_arr.item()
                    if scalar_val not in level_to_idx:
                        raise ValueError(
                            f"Unknown y value {scalar_val!r}. "
                            f"Expected one of: {self.y_levels_}"
                        )
                    y_idx = np.full(n_samples, level_to_idx[scalar_val], dtype=np.int64)
                else:
                    if len(y_arr) != n_samples:
                        raise ValueError(
                            f"y has length {len(y_arr)} but n_samples={n_samples}."
                        )
                    unknown = set(y_arr.tolist()) - set(level_to_idx)
                    if unknown:
                        raise ValueError(
                            f"Unknown y values not seen during fit: {sorted(unknown)}"
                        )
                    y_idx = np.array([level_to_idx[v] for v in y_arr], dtype=np.int64)

            y_tensor = torch.tensor(y_idx, dtype=torch.long, device=self.device)

        # ── initial noise ─────────────────────────────────────────────────
        input_dim = len(self.preprocess_info_["processed_columns"])
        if seed is None:
            x_init = torch.randn(n_samples, input_dim, device=self.device)
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            x_init = torch.randn(n_samples, input_dim, device=self.device, generator=generator)

        # ── ODE integration ───────────────────────────────────────────────
        # n_steps stays None here when the caller didn't set it; adaptive
        # methods ignore it, fixed-step methods fall back to 100 inside
        # odeint_integrate.
        if method in ADAPTIVE_ODE_METHODS:
            odeint_kwargs.setdefault("rtol", rtol)
            odeint_kwargs.setdefault("atol", atol)

        x_gen = odeint_integrate(
            self.model_,
            x_init,
            n_steps=n_steps,
            method=method,
            device=self.device,
            y=y_tensor,
            **odeint_kwargs,
        )

        df_proc = pd.DataFrame(
            x_gen.detach().cpu().numpy(),
            columns=self.preprocess_info_["processed_columns"],
        )
        return _inverse_preprocess_dataframe(df_proc, self.preprocess_info_)

    # Alias for users coming from a ForestDiffusion-style API.
    sample = generate

    def save(self, path: str | Path) -> None:
        """Persist the trained model, preprocessing state and config to `path`."""
        if self.model_ is None or self.preprocess_info_ is None:
            raise RuntimeError("Nothing to save. Fit the generator first.")

        payload = {
            "version": 2,
            "config": {
                "device": self.device,
                "hidden_dim": self.hidden_dim,
                "time_dim": self.time_dim,
                "depth": self.depth,
                "dropout": self.dropout,
                "max_cat_unique": self.max_cat_unique,
                "class_embed_dim": self.class_embed_dim,
            },
            "state_dict": self.model_.state_dict(),
            "preprocess_info": copy.deepcopy(self.preprocess_info_),
            "loss_history": list(self.loss_history_),
            "is_fitted": self.is_fitted_,
            "y_col": self.y_col_,
            "y_levels": self.y_levels_,
            "y_proportions": self.y_proportions_.tolist() if self.y_proportions_ is not None else None,
            "n_classes": self.n_classes_,
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "FMTabularGenerator":
        """Load a generator previously saved with :meth:`save`.

        Pass `device` to load onto a different device than the one used
        for training (e.g. load a model trained on "cuda" onto "cpu").
        """
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        config = payload["config"]
        generator = cls(
            device=device or config["device"],
            hidden_dim=config["hidden_dim"],
            time_dim=config["time_dim"],
            depth=config["depth"],
            dropout=config["dropout"],
            max_cat_unique=config.get("max_cat_unique", 20),
            class_embed_dim=config.get("class_embed_dim", 64),
        )

        preprocess_info = payload["preprocess_info"]
        n_classes = int(payload.get("n_classes", 0))
        input_dim = len(preprocess_info["processed_columns"])
        generator.model_ = generator._build_model(
            input_dim=input_dim, n_classes=n_classes
        ).to(generator.device)
        generator.model_.load_state_dict(payload["state_dict"])
        generator.preprocess_info_ = preprocess_info
        generator.loss_history_ = list(payload.get("loss_history", []))
        generator.is_fitted_ = bool(payload.get("is_fitted", True))
        generator.y_col_ = payload.get("y_col")
        generator.y_levels_ = payload.get("y_levels")
        raw_prop = payload.get("y_proportions")
        generator.y_proportions_ = np.array(raw_prop) if raw_prop is not None else None
        generator.n_classes_ = n_classes
        return generator
