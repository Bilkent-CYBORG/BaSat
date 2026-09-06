# BayOPy

BayOPy is a typed ask–tell Bayesian-optimization library. It provides algorithms,
environment contracts, surrogate utilities, stopping rules, and crash-safe
experiment execution. It does not select benchmark functions or bundled datasets:
the application owns the objective, data, bounds, configuration, and output path.

## Installation

```bash
python -m pip install -e .
```

Optional capabilities can be installed with `optimizers`, `tree`, `mcmc`, `robust`,
`viz`, or `notebook`; `bayopy[full]` installs all runtime extras and `bayopy[dev]`
adds the test and static-analysis tools. The notebook extra includes the Jupyter
kernel and widgets required by dependency progress displays.

## Optimize a function you define

Only one BayOPy import is required:

```python
import bayopy


def my_benchmark(x):
    # BayOPy maximizes this reward.
    return -((x - 0.25).square().sum())


bounds = bayopy.Bounds(
    lower=[-5.0, -5.0],
    upper=[5.0, 5.0],
)

environment = bayopy.StochasticEnvironment(
    bounds=bounds,
    function=my_benchmark,
    noise_std=0.0,
    dtype="float64",
    device="cpu",
    seed=42,
)

algorithm = bayopy.TuRBO(
    bounds=bounds,
    config=bayopy.TuRBOConfig(
        initial_points=10,
        candidate_count=200,
        acquisition=bayopy.AcquisitionConfig(
            name="EI",
            beta=2.0,
            xi=0.0,
            mc_samples=512,
            seed=42,
        ),
        batch_size=1,
        seed=42,
    ),
    dtype="float64",
    device="cpu",
)

experiment = bayopy.Experiment(
    environment=environment,
    algorithm=algorithm,
    output_directory="./results/my_benchmark/turbo",
    resume=True,
    save_every=10,
    initial_sampling_method="paper",
    acquisition_sampling_method="paper",
)

target_rounds = 150
remaining = max(0, target_rounds - experiment.completed_rounds)
result = experiment.run(max_rounds=remaining)

print(result.recommended_action)
print(result.recommended_value)
print(result.evaluations)
```

`verbose`, `verbose_setting`, and `verbose_iter_by_iter` control the run summary,
configuration block, and per-round lines respectively. `plot=True` displays observed
reward and incumbent-reward panels when the run call finishes.

Relative output paths are resolved by Python from the current working directory—the
user's project—not from BayOPy's installation directory. `resume=True` restores a
valid `latest.json` when present and otherwise starts a new experiment. A corrupt or
incompatible existing checkpoint is never silently discarded. Checkpoints contain
observations and optimization progress, not fitted GP parameters. On resume, each
algorithm refits its surrogate from its restored observations using the current
`gp_training_steps`, `gp_maxiter`, or `fit_maxiter` setting when that setting is
exposed by the selected algorithm.

BayOPy maximizes observations by default. For a loss `loss(x)` that should be
minimized, either return `-loss(x)` or construct the algorithm with `maximize=False`.

## Benchmark and acquisition responsibilities

The benchmark is supplied only to an environment. An algorithm never imports, calls,
or otherwise knows the benchmark; `Experiment` passes proposed points to the
environment and passes the returned reward to `algorithm.tell(...)`.

Every algorithm declares its acquisition or decision policy. TuRBO accepts either
`acquisition="TS"` for the TuRBO-1 paper rule or an explicit
`bayopy.AcquisitionConfig(name="EI" | "UCB", ...)`. It fits a local Gaussian process,
generates a sparse candidate set inside the trust region with the configured
acquisition sampling method, and optimizes the selected policy over those candidates.
`candidate_count` controls that candidate set; `gp_training_steps` controls GP fitting.

Some named methods are themselves fixed decision criteria—for example CGP-UCB,
StableOPT, and satisficing Thompson sampling. Replacing those criteria would define a
different algorithm, so their configurations expose parameters for their declared
policy instead of an unrelated EI/UCB selector.

## Use a dataset you own

For an in-memory finite table, create `EnvironmentDataset` yourself:

```python
import bayopy

dataset = bayopy.EnvironmentDataset(
    actions=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
    observations=[0.1, 0.8, 0.2],
)
bounds = bayopy.Bounds(lower=[0.0, 0.0], upper=[1.0, 1.0])
environment = bayopy.StochasticEnvironment(bounds=bounds, dataset=dataset, seed=7)
algorithm = bayopy.TuRBO(
    bounds=bounds,
    config=bayopy.TuRBOConfig(initial_points=1, seed=7),
)
result = bayopy.Experiment(environment=environment, algorithm=algorithm).run(max_rounds=3)
```

`bayopy.load_dataset(path)` is an optional generic CSV/TSV/NPZ loader for user-owned
files. It contains no bundled dataset and performs only schema validation and tensor
conversion.

## Result and diagnostics

`AlgorithmResult` separates the recommended point from the best observed point and
includes `AlgorithmDiagnostics`, a read-only snapshot useful for auditing and
debugging. For TuRBO, method-specific diagnostics include the number of local training
rows, trust-region length, success/failure counters, restart count, center,
length-scale weights, and whether feedback is pending.

`status="running"` means the algorithm is initialized, has consumed feedback, and can
accept another ask–tell round. `max_rounds` bounds one `Experiment.run` call; it is not
an algorithm convergence claim. Inspect `evaluations` or use a stopping rule to express
the application's termination condition.

BayOPy adds positive diagonal jitter to posterior covariance matrices before Cholesky
factorization. Its shared Gaussian-process default is `jitter=1e-6`; users do not need
to add a TuRBO setting. Warnings are not filtered or hidden; unrecoverable
factorization failures and other numerical warnings remain visible.

## Source layout

- `bayopy/`: installed runtime library
- `docs/`: developer and user documentation; excluded from the wheel
- `results/`: user-created project output; never imported by BayOPy

The wheel configuration packages only `bayopy`.
