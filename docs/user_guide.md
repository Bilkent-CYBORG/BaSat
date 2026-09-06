# BayOPy user guide

BayOPy supplies Bayesian-optimization algorithms and ask–tell infrastructure. The
user's application supplies the objective function or data, bounds, algorithm
configuration, stopping condition, and output location. BayOPy has no benchmark
registry, experiment profile, bundled benchmark dataset, or `run_case` wrapper.

## Experiment component reference

`bayopy.Experiment` accepts typed library components rather than unstructured
objects:

| Parameter | Type | Available built-in choices |
| --- | --- | --- |
| `environment` | `bayopy.Environment` | `StochasticEnvironment`, `ContextualEnvironment`, `DistributionalEnvironment`, `AdversarialEnvironment` |
| `algorithm` | `bayopy.Algorithm` | `TuRBO`, `REMBO`, `HeSBO`, `ALEBO`, `SAASBO`, `Dropout`, `ADD_GP_UCB`, `AG_VS`, `LA_MCTS`, `MCTS_VS`, `PG`, `EG`, `GS`, `Satisficing_TS`, `CGP_UCB`, `DRBO`, `RoBOS`, `StableOPT`, `AdveRS` |
| `stopping_rule` | `bayopy.StoppingRule \| None` | `MaximumRoundsStoppingRule`, `MaximumEvaluationsStoppingRule`, `MaximumTimeStoppingRule`, or rules combined with `any_of`/`all_of` |
| `initial_sampling_method` | literal or `None` | `"paper"`, `"sobol"`, `"uniform"` |
| `acquisition_sampling_method` | literal or `None` | `"paper"`, `"sobol"`, `"uniform"` |

Users may implement a new environment, algorithm, or stopping policy by subclassing
the corresponding public base class. The environment owns the reward function or
dataset; the algorithm receives only learner-visible feedback. When `stopping_rule`
is `None`, `Experiment.run` requires a finite `max_rounds` argument.

## Function-backed optimization

```python
import bayopy


def reward(x):
    return -((x - 0.25).square().sum())


bounds = bayopy.Bounds(lower=[-5.0, -5.0], upper=[5.0, 5.0])
environment = bayopy.StochasticEnvironment(
    bounds=bounds,
    function=reward,
    noise_std=0.0,
    dtype="float64",
    device="cpu",
    seed=42,
)
algorithm = bayopy.TuRBO(
    bounds=bounds,
    config=bayopy.TuRBOConfig(
        batch_size=1,
        q_batch=None,
        initial_points=10,
        length_init=0.8,
        length_min=0.5**7,
        length_max=1.6,
        success_tolerance=3,
        failure_tolerance=None,
        candidate_count=None,
        perturbation_cap=20.0,
        acquisition=bayopy.AcquisitionConfig(
            name="EI",
            beta=2.0,
            xi=0.0,
            mc_samples=512,
            seed=42,
        ),
        success_relative_tolerance=1e-3,
        noisy_center="posterior_mean",
        max_cholesky_size=2000,
        gp_training_steps=50,
        seed=42,
    ),
    maximize=True,
    dtype="float64",
    device="cpu",
    plot=False,
    verbose=False,
    verbose_setting=False,
    verbose_iter_by_iter=False,
)

experiment = bayopy.Experiment(
    environment=environment,
    algorithm=algorithm,
    stopping_rule=None,
    output_directory="./results/my_benchmark/turbo",
    resume=True,
    save_every=10,
    constant_context=None,
    initial_sampling_method="paper",
    acquisition_sampling_method="paper",
)

target_rounds = 150
remaining = max(0, target_rounds - experiment.completed_rounds)
result = experiment.run(max_rounds=remaining)
```

The presentation flags have the following independent meanings:

- `verbose=True` prints one start line and one final run summary.
- `verbose_setting=True` prints the algorithm, acquisition, bounds, configuration,
  sampling, and persistence settings before the run.
- `verbose_iter_by_iter=True` prints one line after every completed round.
- `plot=True` displays observed reward and incumbent-reward panels after the run.

They do not change optimization mathematics or the requested evaluation budget.

The objective above already returns a reward to maximize. If a function returns a
loss, use `maximize=False` on the algorithm or negate the loss, but do not do both.

`output_directory` is interpreted relative to the process's current working directory.
Thus `./results/...` belongs to the user's project and never to the installed BayOPy
package. Omit `output_directory` to run entirely in memory.

## Resume and checkpoint archives

`resume=True` means resume if a valid `latest.json` exists; otherwise start from the
beginning. Existing corrupt or configuration-incompatible checkpoints raise an error.
The environment source, bounds, algorithm type, immutable problem configuration, and
relevant random states must match when resuming.

Configuration-mismatch errors report the differing field and its checkpoint/current
values. Fitting budgets are the exception: `gp_training_steps`, `gp_maxiter`, and
`fit_maxiter` may be changed when the selected algorithm exposes them. BayOPy restores
the algorithm's own observations and optimization progress but never restores fitted
GP parameters; the next model-based proposal refits the surrogate from those
observations with the current fitting budget. For changes to problem-defining settings
such as bounds or acquisition policy, use a new output directory so experiments remain
separate.

`save_every=10` retains a numbered historical checkpoint every ten rounds. BayOPy
still updates its alternating transactional recovery checkpoint around every `tell`,
so this option controls archive density rather than whether crash recovery works.

`Experiment.run(max_rounds=n)` executes at most `n` additional rounds. To resume toward
a total target, subtract `experiment.completed_rounds` as shown above.

## Benchmark ownership and how TuRBO chooses an action

The user-defined reward function belongs only to the environment. The algorithm does
not receive or inspect that function. `Experiment` asks the algorithm for a point,
lets the environment evaluate the reward, and returns learner-visible feedback to the
algorithm.

TuRBO supports its paper Thompson policy and explicit EI/UCB policies:

1. Until `initial_points` local observations exist, it evaluates seeded design points.
2. It normalizes actions to the unit box and standardizes observed rewards.
3. It fits a local Gaussian process to observations since the latest restart.
4. It centers a trust region on the best posterior-mean or observed point, depending on
   `noisy_center`.
5. It generates `candidate_count` candidates in that trust region. If the value is
   `None`, the count is `min(100 × dimension, 5000)`.
6. With `acquisition="TS"`, it draws posterior Thompson samples and selects their
   maximizers. With `AcquisitionConfig(name="EI" | "UCB", ...)`, it constructs that
   acquisition from the local GP and optimizes it over the same candidates.
7. Successful batches enlarge the region; repeated failures shrink it. A region below
   `length_min` is restarted.

`acquisition_sampling_method` controls how continuous acquisition candidates are
generated (`"paper"`, `"sobol"`, or `"uniform"`); it does not select the acquisition
function. Every concrete algorithm declares its decision policy in the shared
`Algorithm` contract. A method whose name defines that policy, such as CGP-UCB or
StableOPT, does not accept an unrelated replacement without becoming a different
algorithm.

## User-owned finite data

Pass an in-memory table directly:

```python
import bayopy

dataset = bayopy.EnvironmentDataset(
    actions=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
    observations=[0.1, 0.8, 0.2],
)
bounds = bayopy.Bounds(lower=[0.0, 0.0], upper=[1.0, 1.0])
environment = bayopy.StochasticEnvironment(bounds=bounds, dataset=dataset, seed=7)
```

`EnvironmentDataset` may also contain contexts and observation-noise variances for
the corresponding environment contract. `bayopy.load_dataset(path)` is a generic
loader for user-owned CSV, TSV, and NPZ files; it contains no built-in data or dataset
name registry.

## Understanding `AlgorithmResult`

The main fields are:

- `recommended_action` and `recommended_value`: the algorithm's current recommendation.
- `best_observed_action` and `best_observed_value`: the best evaluated pair visible to
  the learner.
- `evaluations`: individual observations consumed; a batch may add more than one.
- `status`: lifecycle state of the algorithm.
- `diagnostics`: a detached, read-only audit snapshot.
- `source`: publication and reference-implementation provenance.

`status="running"` does not mean a background process remains active. It means at least
one ask–tell exchange completed and the algorithm remains able to accept another one.
Stopping because `Experiment.run` reached its invocation budget is an external control
decision, not an assertion that TuRBO mathematically converged.

Common diagnostics report iteration/evaluation counts, best observed value, model
dimension, acquisition score when meaningful, warning codes, and compatibility state.
TuRBO adds local training rows, trust-region geometry, success/failure counters,
restart count, center, length-scale weights, and pending-feedback state. These values
explain the run; users do not need to feed them back into the algorithm.

## Numerical stabilization

Exact GP covariance matrices can become numerically semidefinite when evaluated points
are repeated or very close, or when fitted noise becomes extremely small. GPyTorch
then adds a tiny diagonal jitter before Cholesky factorization. This is a standard
stabilization and does not change the benchmark observations.

BayOPy proactively adds the configured positive jitter to posterior covariance
matrices before Cholesky factorization. The shared `GaussianProcessConfig.jitter`
default is `1e-6`, so no TuRBO parameter is required. Warnings are not filtered or
hidden, and an unrecoverable factorization still raises a typed error. Using
`dtype="float64"`, avoiding duplicate points where possible, and specifying realistic
observation noise generally improve conditioning.
