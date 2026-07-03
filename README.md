# APREMBO: Adaptive Partition REMBO-UCB Optimizer

APREMBO is a REMBO-style Bayesian optimization optimizer for high-dimensional black-box functions. It searches in a low-dimensional embedded space, projects embedded candidates back into the original bounded domain, fits local Gaussian-process models on tree cells, and can stop early using posterior probability rules.

The optimizer is designed for problems where the original dimension `D` is large, but the effective search dimension `d_embedding` is much smaller.

---

## Main features

- **Random embedding optimization**
  - Builds a Gaussian projection matrix `A` with shape `(D, d_embedding)`.
  - Optimizes in embedded `z`-space.
  - Projects candidates back to the normalized original domain using

    ```text
    x = p_X(Az)
    ```

- **Tree-based local search**
  - Maintains active embedded-space cells.
  - Selects nodes using UCB-style bounds.
  - Splits selected cells into `N` children.
  - Local datasets are assigned to child cells according to embedded bounds.

- **Gaussian-process surrogate models**
  - Uses BoTorch / GPyTorch `SingleTaskGP`.
  - Supports embedded kernels of the form

    ```text
    k_emb(z, z') = k(p_X(Az), p_X(Az'))
    ```

  - Supported base kernels:
    - `rbf`
    - `matern12`
    - `matern32`
    - `matern52`

- **Automatic kernel selection**
  - If `auto_select_kernel=True`, the optimizer fits candidate kernels and selects the one with the best marginal log likelihood.

- **Lengthscale optimization**
  - Includes bounded marginal-likelihood search for scalar lengthscale.
  - Can use SciPy DIRECT / optional CMA-style refinement when enabled.
  - Reoptimizes every `lengthscale_reoptimize_interval` iterations by default.

- **Lipschitz estimation**
  - `lipschitz_mode="predict"`: estimates the Lipschitz constant from the GP posterior mean gradient.
  - `lipschitz_mode="user"`: uses a fixed user-provided `L`.

- **Stopping rules**
  - Posterior probability stopping:

    ```text
    P(f(x) >= tau) >= 1 - delta
    ```

  - Epsilon-delta PAC stopping.
  - Multiple risk tolerance / threshold tracing.

---

## File structure

Use these files inside the `APREMBO/` package folder:

```text
APREMBO/
├── APREMBO.py
├── kernel.py
├── utils.py
├── Stopping_rules.py
└── __init__.py
```

Current uploaded file names may contain suffixes such as `utils(16).py` or `kernel(1).py`. Rename them before using as a package:

```bash
mv "utils(16).py" utils.py
mv "kernel(1).py" kernel.py
mv "Stopping_rules(1).py" Stopping_rules.py
mv "__init__(20).py" __init__.py
```

---

## Installation

Recommended environment: Python 3.10+.

Install the main dependencies:

```bash
pip install numpy scipy torch gpytorch botorch matplotlib
```

Optional, depending on your local setup:

```bash
pip install cma
```

---

## Basic usage

```python
import numpy as np
from APREMBO import APREMBO

# Original high-dimensional domain: D = 25, each variable in [-5, 5]
D = 25
bounds = np.array([[-5.0, 5.0]] * D)

# Example objective. APREMBO passes x in the real original domain.
def objective(x):
    x = np.asarray(x).reshape(-1, D)
    return -np.sum(x[:, :2] ** 2, axis=1)[0]

optimizer = APREMBO(
    original_boundaries=bounds,
    d_embedding=2,
    max_iter=100,
    tau=-1e-3,
    N=2,
    delta=1 / 64,
    batch_size=1,
    n_init=4,
    a=-100.0,
    b=0.0,
    R=0.01,
    kernel_name="matern52",
    auto_select_kernel=True,
    lipschitz_mode="predict",
    verbose=True,
)

result = optimizer.optimize(objective)
print(result)
print("best score:", optimizer.best_score)
print("best x:", optimizer.best_x)
print("best z:", optimizer.best_z)
```

---

## Important constructor arguments

| Argument | Meaning |
|---|---|
| `original_boundaries` | Real search box with shape `(D, 2)` |
| `d_embedding` | Low-dimensional REMBO embedding dimension |
| `max_iter` | Total objective evaluation budget |
| `tau` | Posterior stopping threshold |
| `N` | Number of children per split |
| `delta` | Risk tolerance for posterior stopping |
| `batch_size` | Number of points evaluated per BO iteration |
| `n_init` | Number of initial random/LHS evaluations |
| `a`, `b` | Lower/upper objective score bounds used in UCB confidence width |
| `R` | Observation noise standard deviation |
| `kernel_name` | Fallback kernel name: `rbf`, `matern12`, `matern32`, `matern52` |
| `auto_select_kernel` | If `True`, select kernel by marginal log likelihood |
| `kernel_candidates` | Candidate kernels for automatic kernel selection |
| `lipschitz_mode` | `predict` or `user` |
| `L` | Fixed Lipschitz constant, required only when `lipschitz_mode="user"` |
| `embedding_radius` | Embedded search radius; default is `sqrt(d_embedding)` |
| `lengthscale_lower_bound` | Lower bound for scalar lengthscale search |
| `lengthscale_upper_bound` | Upper bound for scalar lengthscale search |
| `threshold_trace_risk_tolerances` | Risk tolerances used for threshold trace logging |
| `verbose` | If `True`, print dtype/device/start value and one progress line per objective evaluation |

When `verbose=True`, APREMBO prints startup information and per-iteration progress:

```text
Using dtype = torch.float64
Using device = cpu
Starting from fbest = 64.0
iteration: 1 Best Value: 26.9 Current Value: 26.9
iteration: 2 Best Value: 25.66 Current Value: 25.66
```

---

## Kernel notes

The embedded kernel is not a plain kernel directly on `z`. It follows the REMBO-induced form:

```text
k_emb(z, z') = k(p_X(Az), p_X(Az'))
```

This means two embedded points are compared after projection into the clipped normalized original domain.

Supported base kernels are:

```python
("rbf", "matern12", "matern32", "matern52")
```

For BO, `matern52` or `matern32` is usually a safer default than `rbf` when the objective may not be extremely smooth.

---

## Lipschitz modes

### Predictive mode

```python
lipschitz_mode="predict"
```

The optimizer estimates `L` from the maximum norm of the GP posterior mean gradient inside the current embedded cell. This is useful when you do not want to pass a fixed Lipschitz constant manually.

### User-fixed mode

```python
lipschitz_mode="user",
L=10.0
```

The optimizer uses the fixed value `L=10.0` in the tree/UCB bound.

---

## Stopping behavior

The main posterior stopping rule is:

```text
P(f(x) >= tau) >= 1 - delta
```

Example:

```python
optimizer = APREMBO(
    original_boundaries=bounds,
    d_embedding=2,
    max_iter=200,
    tau=-0.01,
    N=2,
    delta=1 / 256,
)
```

A smaller `delta` means stricter stopping. For example:

```text
1/256  stricter
1/64
1/16
1/4    looser
```

You can also evaluate multiple risk tolerances:

```python
results = optimizer.optimize(
    objective,
    risk_tolerances=[1/256, 1/64, 1/16, 1/4],
)
```

Or multiple thresholds and deltas:

```python
results = optimizer.optimize(
    objective,
    threshold_values=[-1e-3, -1e-2, -1e-1],
    risk_tolerances=[1/256, 1/64, 1/16, 1/4],
)
```

---

## Acquisition functions

By default, APREMBO uses:

```python
qLogExpectedImprovement
```

Other supported acquisition choices include:

```python
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound

optimizer = APREMBO(
    original_boundaries=bounds,
    d_embedding=2,
    max_iter=100,
    tau=-1e-3,
    N=2,
    acquisition_function=qUpperConfidenceBound,
)
```

---

## Returned result

For a standard run, `optimize()` returns a tuple similar to:

```text
(stopped, success, y_raw, y_noisy, node_level, node_index, tau, iteration)
```

where:

- `stopped`: whether posterior stopping triggered before `max_iter`
- `success`: whether the final/raw objective exceeded the threshold
- `y_raw`: noiseless/raw objective value at the stopping/final point
- `y_noisy`: noisy observed objective value
- `node_level`: selected tree level
- `node_index`: selected tree node index
- `tau`: stopping threshold
- `iteration`: stopping or final iteration

The optimizer also stores:

```python
optimizer.best_score
optimizer.best_observed_score
optimizer.best_x
optimizer.best_z
optimizer.threshold_probability_trace
```

---

## Practical recommendations

For your high-dimensional BO experiments, a good default configuration is:

```python
optimizer = APREMBO(
    original_boundaries=bounds,
    d_embedding=2,
    max_iter=500,
    tau=-1e-2,
    N=2,
    delta=1 / 64,
    batch_size=1,
    n_init=4,
    a=-100.0,
    b=0.0,
    R=0.01,
    kernel_name="matern52",
    auto_select_kernel=True,
    lipschitz_mode="predict",
)
```

Use `auto_select_kernel=True` when comparing kernels by marginal likelihood. Use `lipschitz_mode="predict"` when you do not want to manually tune `L`.

---

## Notes and caveats

- The objective is evaluated in the **real original domain**, not in normalized space.
- Internally, original bounds are normalized to `[-1, 1]^D` before embedding projection.
- `a` and `b` should be chosen consistently with the expected objective-value range.
- GP fitting can become expensive as the local dataset grows.
- Kernel selection by marginal likelihood improves flexibility but increases runtime.
- If the objective is noisy, set `R` close to the true observation noise standard deviation.
