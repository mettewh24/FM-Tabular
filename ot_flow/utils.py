import math
import numbers
from typing import Optional
import warnings

import torch
import numpy as np
import ot  # POT


# ============================================================
# Dataset
# ============================================================

class TabularDataset:
    """Wraps a preprocessed (numeric) DataFrame as a tensor on `device`."""

    def __init__(self, df, device="cpu", dtype=torch.float32):
        self.device = device
        np_dtype = np.float64 if dtype == torch.float64 else np.float32
        values = df.to_numpy(dtype=np_dtype, copy=True)
        self.X = torch.tensor(values, dtype=dtype, device=device)

    def __len__(self):
        return self.X.shape[0]


# ============================================================
# Cost functions
# ============================================================

def default_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Squared L2 cost (standard OT)."""
    return torch.cdist(x0, x1, p=2) ** 2


def mixed_cost(
    x0: torch.Tensor,
    x1: torch.Tensor,
    cont_idx: list,
    disc_idx: list,
) -> torch.Tensor:
    """
    Mixed cost for tabular data:
      - Squared L2 on continuous features
      - L1 on one-hot discrete features (equivalent to Hamming distance per category)
    """
    cost = torch.zeros(x0.shape[0], x1.shape[0], device=x0.device, dtype=x0.dtype)
    if cont_idx:
        ci = torch.tensor(cont_idx, device=x0.device, dtype=torch.long)
        cost = cost + torch.cdist(x0[:, ci], x1[:, ci], p=2) ** 2
    if disc_idx:
        di = torch.tensor(disc_idx, device=x0.device, dtype=torch.long)
        cost = cost + torch.cdist(x0[:, di], x1[:, di], p=1)
    return cost


# ============================================================
# Sinkhorn / exact OT via POT
# ============================================================

def sinkhorn_pot(C: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
    """
    Compute Sinkhorn coupling via POT.

    Args:
        C: Cost matrix (B, B), PyTorch tensor.
        epsilon: Regularization strength.

    Returns:
        pi: Coupling matrix (B, B), normalized to sum to 1.
    """
    B = C.shape[0]
    a = torch.full((B,), 1.0 / B, device=C.device, dtype=C.dtype)
    b = torch.full((B,), 1.0 / B, device=C.device, dtype=C.dtype)

    C_max = C.max()
    C_norm = C / (C_max + 1e-8)

    # POT can operate directly on torch tensors (including CUDA tensors).
    pi = ot.sinkhorn(a, b, C_norm, reg=epsilon, verbose=False, stopThr=1e-7)
    if not torch.is_tensor(pi):
        pi = torch.tensor(pi, device=C.device, dtype=C.dtype)
    else:
        pi = pi.to(device=C.device, dtype=C.dtype)

    return pi


def exact_ot_pot(C: torch.Tensor) -> torch.Tensor:
    """
    Compute exact OT coupling via POT (Earth Mover's Distance).

    Note: POT's exact solver runs on CPU, so CUDA tensors are moved
    to CPU for the solve and then copied back to the original device.

    Args:
        C: Cost matrix (B, B), PyTorch tensor.

    Returns:
        pi: Coupling matrix (B, B), normalized to sum to 1.
    """
    B = C.shape[0]
    a = np.full((B,), 1.0 / B, dtype=np.float64)
    b = np.full((B,), 1.0 / B, dtype=np.float64)

    C_max = C.max()
    C_norm = C / (C_max + 1e-8)

    C_np = C_norm.detach().to(device="cpu", dtype=torch.float64).numpy()
    pi_np = ot.emd(a, b, C_np)
    return torch.tensor(pi_np, device=C.device, dtype=C.dtype)


# ============================================================
# Sampling from coupling
# ============================================================

def sample_from_pi(pi: torch.Tensor, n_samples: int):
    """Sample index pairs (i, j) from flattened coupling distribution."""
    B = pi.shape[0]
    # Clamp to avoid numerical negatives leaking out of sinkhorn
    pi_flat = pi.reshape(-1).clamp(min=0.0)
    idx = torch.multinomial(pi_flat, n_samples, replacement=True)
    i = torch.div(idx, B, rounding_mode="floor")
    j = idx % B
    return i, j


# ============================================================
# OT DataLoader
# ============================================================

class OTDataLoader:
    """
    DataLoader that uses Optimal Transport to pair samples within each batch.

    The source distribution x0 is always standard Gaussian noise, sampled
    fresh for every batch; x1 is drawn from `dataset`. Samples are paired
    according to an OT coupling computed on a (possibly mixed-type) cost
    matrix.

    When `label_y` is provided (a 1-D integer numpy array of length N),
    each batch also yields the corresponding y labels aligned to x1 after
    OT reindexing. The labels follow x1 through the coupling so that
    model(x0[i], x1[j], y[j]) is always consistent.

    Args:
        dataset: Target TabularDataset.
        batch_size: Batch size.
        epsilon: Sinkhorn regularization.
        cost_fn: Callable(x0, x1) -> cost matrix. Defaults to squared L2.
        ot_solver: "sinkhorn" (default) or "exact".
        shuffle: Whether to shuffle the dataset each epoch.
        device: Torch device string.
        label_y: Optional 1-D integer numpy array of class labels, one per
                 sample in `dataset`. When provided, each iteration yields
                 (x0, x1, y_batch); otherwise yields (x0, x1, None).
    """

    def __init__(
        self,
        dataset: TabularDataset,
        batch_size: int = 256,
        epsilon: float = 0.1,
        cost_fn=None,
        ot_solver: str = "sinkhorn",
        shuffle: bool = True,
        device: str = "cpu",
        label_y: Optional[np.ndarray] = None,
        **deprecated_kwargs,
    ):
        if deprecated_kwargs:
            warnings.warn(
                "The following OTDataLoader arguments are deprecated and "
                f"ignored: {sorted(deprecated_kwargs)}. The source "
                "distribution is always standard Gaussian noise.",
                DeprecationWarning,
                stacklevel=2,
            )

        if not isinstance(batch_size, numbers.Integral) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(epsilon, numbers.Real) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be a positive finite number.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.shuffle = shuffle
        self.device = device
        self.cost_fn = cost_fn if cost_fn is not None else default_cost

        ot_solver = str(ot_solver).lower().strip()
        if ot_solver not in ("sinkhorn", "exact"):
            raise ValueError(
                f"ot_solver must be 'sinkhorn' or 'exact', got '{ot_solver}'."
            )
        self.ot_solver = ot_solver
        self.N = len(dataset)

        if label_y is not None:
            label_y = np.asarray(label_y)
            if label_y.shape != (self.N,):
                raise ValueError(
                    f"label_y must have shape ({self.N},), got {label_y.shape}."
                )
        self.label_y = label_y

    def _solve_ot(self, C: torch.Tensor) -> torch.Tensor:
        if self.ot_solver == "exact":
            return exact_ot_pot(C)
        return sinkhorn_pot(C, self.epsilon)

    def __iter__(self):
        indices = np.arange(self.N)
        if self.shuffle:
            np.random.shuffle(indices)

        for b in range(len(self)):
            idx = indices[b * self.batch_size : (b + 1) * self.batch_size]
            if len(idx) == 0:
                continue

            x1 = self.dataset.X[idx]           # (B, D) — target batch
            x0 = torch.randn_like(x1)           # (B, D) — Gaussian source

            C = self.cost_fn(x0, x1)
            pi = self._solve_ot(C).detach()
            i, j = sample_from_pi(pi, len(x0))

            # x0[i] and x1[j] are OT-coupled pairs.
            # y must follow x1 through the j-reindexing so that the label
            # always refers to the target point x1[j].
            y_batch: Optional[torch.Tensor] = None
            if self.label_y is not None:
                y_batch = torch.tensor(
                    self.label_y[idx][j.cpu().numpy()],
                    dtype=torch.long,
                    device=self.device,
                )

            yield x0[i], x1[j], y_batch

    def __len__(self):
        return math.ceil(self.N / self.batch_size)


# ============================================================
# Flow Matching Loss
# ============================================================

def flow_matching_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor = None,
) -> torch.Tensor:
    """
    Conditional Flow Matching loss.

    Interpolates x_t = (1-t)*x0 + t*x1 and regresses the model's predicted
    velocity onto the constant target velocity ut = x1 - x0.

    If `y` is provided, it is forwarded to class-conditional models.
    """
    xt = (1.0 - t) * x0 + t * x1
    ut = x1 - x0
    t_model = t.view(-1) if t.ndim == 2 and t.shape[1] == 1 else t
    if y is None:
        vt = model(xt, t_model)
    else:
        vt = model(xt, t_model, y=y)
    return ((vt - ut) ** 2).mean()


# ============================================================
# Inference: ODE integration via torchdiffeq
# ============================================================

ADAPTIVE_ODE_METHODS = {"dopri5", "dopri8", "bosh3", "adaptive_heun"}


@torch.no_grad()
def odeint_integrate(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    n_steps: Optional[int] = None,
    method: str = "dopri5",
    device: str = "cpu",
    y: torch.Tensor = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    **odeint_kwargs,
) -> torch.Tensor:
    """
    Generate samples by integrating the learned velocity field via torchdiffeq.

    torchdiffeq expects the ODE function to have signature f(t, x) where t is
    a scalar and x is the state tensor. The model uses the opposite convention
    forward(x, t) with t as a (B, 1) batch vector, so we wrap it in an adapter.

    Args:
        model: Trained velocity-field network with signature forward(x, t).
        x_init: Starting points (B, input_dim).
        n_steps: Number of evaluation time-points in [0, 1], used to build
                 the `t` grid passed to odeint (step size is 1/n_steps).
                 Only meaningful for fixed-step solvers ('euler', 'midpoint',
                 'rk4'). Ignored (with a warning if explicitly set) for
                 adaptive solvers ('dopri5', 'dopri8', 'bosh3',
                 'adaptive_heun'), which choose their own internal step
                 sizes based on `rtol`/`atol` — for those, only the two
                 endpoints t=0 and t=1 are passed to odeint.
        method: ODE solver passed to torchdiffeq.odeint. Common choices:
                  'euler'    - explicit Euler (1st order, fixed step)
                  'midpoint' - explicit midpoint (2nd order, fixed step)
                  'rk4'      - classic Runge-Kutta (4th order, fixed step)
                  'dopri5'   - Dormand-Prince (4/5th order, adaptive step) [default]
                  'bosh3'    - Bogacki-Shampine (2/3rd order, adaptive)
        device: Torch device string.
        y: Optional class labels (shape B,) forwarded to conditional models.
        rtol, atol: Relative/absolute tolerance forwarded to odeint. Only
                    used by adaptive solvers; ignored by fixed-step ones.
        **odeint_kwargs: Additional keyword arguments forwarded to odeint.

    Returns:
        x: Generated samples (B, input_dim) at t=1.
    """
    try:
        from torchdiffeq import odeint
    except ImportError as exc:
        raise ImportError(
            "torchdiffeq is required for ODE integration. "
            "Install it with: pip install torchdiffeq"
        ) from exc

    if n_steps is not None and (
        not isinstance(n_steps, numbers.Integral) or n_steps <= 0
    ):
        raise ValueError("n_steps must be a positive integer when provided.")
    if not isinstance(rtol, numbers.Real) or not math.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be a positive finite number.")
    if not isinstance(atol, numbers.Real) or not math.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be a positive finite number.")

    model.eval()
    x_init = x_init.to(device)

    is_adaptive = method in ADAPTIVE_ODE_METHODS

    if is_adaptive:
        if n_steps is not None:
            warnings.warn(
                f"n_steps={n_steps} is ignored for adaptive method '{method}'; "
                "the solver chooses its own internal step sizes based on "
                "rtol/atol.",
                stacklevel=2,
            )
        t_grid = torch.tensor([0.0, 1.0], device=device, dtype=x_init.dtype)
        odeint_kwargs.setdefault("rtol", rtol)
        odeint_kwargs.setdefault("atol", atol)
    else:
        if n_steps is None:
            n_steps = 100
        t_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=x_init.dtype)

    # Adapter: torchdiffeq calls f(t_scalar, x) with a 0-d t tensor.
    # We expand t to (B,) so the model's time embedding receives the
    # expected shape regardless of batch size.
    def ode_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_batch = t.expand(x.shape[0])
        if y is None:
            return model(x, t_batch)
        return model(x, t_batch, y=y)

    # odeint returns (len(t_grid), B, D); we take the final time-point
    trajectory = odeint(ode_fn, x_init, t_grid, method=method, **odeint_kwargs)
    return trajectory[-1]