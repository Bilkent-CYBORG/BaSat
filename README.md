# Basat

Basat is a Python library for **Bayesian optimization** and helps you find good inputs to an expensive objective using a limited number of evaluations. It brings 19 optimization algorithms together behind a consistent interface for objective functions, tabular datasets, experiment management, and plotting.

Define what you want to optimize, choose an environment and an algorithm, and run an `Experiment`.

[Quick start](#quick-start) · [Compare algorithms](#compare-algorithms) · [Use your own data](#use-your-own-data) · [Notebook setup](#start-from-the-notebook) · [Installation](#installation)

[What you can do](#what-you-can-do) · [Algorithms](#available-algorithms) · [Environments](#environments) · [Budgets and checkpoints](#budgets-stopping-and-checkpoints)

[User guide](#user-guide) · [Project structure](#project-structure) · [License](#license)

## What you can do

- Optimize your own Python objective, simulation, or model-evaluation function.
- Work with stochastic rewards, contextual information, distributional uncertainty, or adversarial perturbations.
- Compare algorithms and configurations across independent, seeded trials.
- Control evaluation budgets, batches, early stopping, and checkpoint continuation.
- Plot optimization progress and compare measured results with paper-specific benchmarks.

## Installation

Basat requires **Python 3.11–3.14**. Install it from the project root using the following command.

```bash
python -m pip install .
```

For an editable installation with development tools, use the following command.

```bash
python -m pip install -e ".[dev]"
```

Installing Basat also installs its numerical, inference, optimization, notebook, and plotting dependencies, including NumPy, PyTorch, JAX, and NumPyro. The package requirements select compatible versions automatically. Paper reproductions with specific numerical-version requirements use the environments described in [reproduction setup](docs/reproduction.md).

### Start from the notebook

Extract the complete GitHub ZIP and open `docs/basat_user_guide.ipynb` in Jupyter or VS Code with a Python kernel. Run the first code cell before the examples. It checks the Python version and Mac architecture, finds the extracted project, and installs Basat and its dependencies into the active kernel's Python environment. The first installation needs internet access and may take several minutes. Follow any kernel restart instruction before continuing. Opening or downloading the notebook leaves installation under your control through this cell.

If Jupyter starts outside the extracted project, set `PROJECT_DIRECTORY` in the setup cell to the folder containing `pyproject.toml` and `basat/`. Run the setup cell again whenever you restart the kernel. Once setup succeeds, further calls in the same kernel skip installation.

### Choose a compatible kernel

The setup cell needs a Python 3.11–3.14 kernel. On an Apple Silicon Mac, Python must report `arm64`. Use the Python 3.11 macOS universal2 installer from [python.org](https://www.python.org/downloads/macos/) if you need a compatible interpreter. In Terminal, check the Python installation you intend to use with the following command.

```bash
python3.11 -c "import platform; print(platform.machine())"
```

If an Apple Silicon Mac reports `x86_64`, use the native interpreter to create an environment from the extracted project root.

```bash
arch -arm64 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m venv .venv-basat
source .venv-basat/bin/activate
python -m pip install ipykernel
python -m ipykernel install --user --name basat --display-name "Basat Python 3.11"
```

For other supported Python installations, create the environment with `python3.11 -m venv .venv-basat` and then use the same activation and kernel registration commands. Select **Basat Python 3.11** in your notebook and run its setup cell to install Basat automatically. If you need to install Jupyter itself, run `python -m pip install jupyterlab` followed by `python -m jupyter lab docs/basat_user_guide.ipynb` in the activated environment.

The current dependency stack requires a supported platform with available PyTorch and JAX wheels. Intel macOS lacks official wheels for the required PyTorch releases. Apple Silicon Macs can use native arm64 Python, and Intel Mac users can use a supported Linux environment. The setup cell reports incompatible kernels before attempting installation.

## Quick start

The following example maximizes a two-dimensional function.

```python
import basat

seed = basat.set_seed(42, threads=1)
bounds = basat.Bounds(lower=[0.0, 0.0], upper=[1.0, 1.0])


def objective(x):
    return -sum((value - 0.25) ** 2 for value in x)


algorithm = basat.make_algorithm("TuRBO", bounds=bounds, seed=seed)
environment = basat.StochasticEnvironment(
    bounds=bounds,
    function=objective,
    seed=seed,
)
experiment = basat.Experiment(
    environment=environment,
    algorithm=algorithm,
)
result = experiment.run(max_evaluations=30, plot=True, show=True)
print(experiment.summary())
```

Basat proposes inputs within the bounds, evaluates your function, and updates the algorithm from the observed rewards. The maximum reward in this example is `0`, at `[0.25, 0.25]`.

An objective returns one finite scalar, and Basat maximizes by default. To minimize a loss, return its negative as the reward. You can choose an evaluation budget that suits your objective, such as `max_evaluations=30` in this example.

## Compare algorithms

Pass a dictionary of labeled algorithms to run independent comparisons. For example, you can use labels to distinguish different algorithms or different configurations of one algorithm, as shown below.

```python
import basat

seed = basat.set_seed(42, threads=1)
bounds = basat.Bounds(lower=[0.0, 0.0], upper=[1.0, 1.0])


def objective(x):
    return -sum((value - 0.25) ** 2 for value in x)


algorithms = {
    "TuRBO / EI": basat.make_algorithm(
        "TuRBO",
        bounds=bounds,
        seed=seed,
        acquisition="EI",
        initial_points=8,
        candidate_count=128,
    ),
    "TuRBO / UCB": basat.make_algorithm(
        "TuRBO",
        bounds=bounds,
        seed=seed,
        acquisition=basat.AcquisitionConfig(name="UCB", beta=3.0, seed=seed),
        initial_points=8,
        candidate_count=128,
    ),
}
experiment = basat.Experiment(
    environment=basat.StochasticEnvironment(
        bounds=bounds, function=objective, seed=seed,
    ),
    algorithm=algorithms,
    output_directory=basat.new_run_directory("results/comparison"),
)
comparison = experiment.run(
    trials=3, max_evaluations=30, plot=True, show=True,
)
print(experiment.summary())
```

Here each algorithm receives up to 30 evaluations per trial, across three trials. Basat advances the configured seeds between trials and keeps each algorithm's state separate.

Use `basat.plot()` to customize plots of saved experiments. `basat.PlotSeries`, `basat.PlotPanel`, and `basat.plot_panels()` support custom metrics, panels, and explicitly configured uncertainty bars. The notebook provides examples of these plotting APIs.

## Available algorithms

| Family | Algorithms |
| --- | --- |
| Stochastic | REMBO, HeSBO, ALEBO, ADD_GP_UCB, TuRBO, Dropout, MCTS_VS, SAASBO, AG_VS, LA_MCTS, PG, EG, GS, Satisficing_TS |
| Contextual | CGP_UCB |
| Distributional | DRBO, RoBOS |
| Adversarial / robust | StableOPT, AdveRS |

List the registered names with `basat.algorithm_names()`. Construct an algorithm by name with `basat.make_algorithm()`, or use its class and configuration directly.

Settings differ by method. Embedding-based methods require an embedding dimension, satisficing methods require a threshold, and AG-VS requires a valid split count. The [user guide](docs/basat_user_guide.ipynb) explains these choices and shows expanded configurations.

## Environments

An environment defines how an action is evaluated and what information the learner receives.

| Environment | Purpose |
| --- | --- |
| `StochasticEnvironment` | Evaluate an action and return a scalar reward, optionally with observation noise. |
| `ContextualEnvironment` | Reveal context before choosing an action and evaluate the action in that context. |
| `DistributionalEnvironment` | Supply public distribution information and evaluate an action under a sampled context. |
| `AdversarialEnvironment` | Apply a configured perturbation or attack during evaluation. |

Each algorithm's theoretical guarantees depend on its stated assumptions. The guide describes supported cross-environment adaptations and explains the assumptions and limitations associated with each one.

## Use your own data

Basat supports CSV-backed experiments as well as callable objectives. The following example loads a table using the appropriate environment family.

```python
import basat

loaded = basat.load_dataset(
    basat.find_path("data/stochastic/diabetes_bandit.csv"),
    environment="stochastic",
)
print(loaded.preview(rows=3))
```

The `data/` directory contains small diabetes, wine, wind-power, and iris examples. [Dataset documentation](data/README.md) describes their sources and transformations, and the guide shows the required CSV schemas and complete environment construction. You can substitute your own data in the same format.

## Budgets, stopping, and checkpoints

`max_evaluations` limits objective evaluations, and a round may contain several evaluations when an algorithm proposes a batch. You can also use stopping rules for conditions such as a target reward or lack of improvement. For example, the following configuration combines an evaluation limit with an improvement-based stopping rule.

```python
import basat

stopping_rule = basat.any_of(
    basat.MaximumEvaluationsStoppingRule(50),
    basat.NoImprovementStoppingRule(10, min_rounds=20, min_delta=0.001),
)
```

Pass the rule as `stopping_rule` when constructing an `Experiment`.

Set `output_directory` to save results and checkpoints. To continue a run, reconstruct the same objective, environment, configuration, and output directory with `resume=True`, then call `run()` again. Evaluation limits are totals, so continuing from 20 evaluations to a limit of 50 permits up to 30 additional evaluations per trial.

The default transactional checkpoint policy preserves evaluated feedback before giving it to the learner. An explicit periodic policy can reduce disk overhead for replayable experiments, and a crash may repeat evaluations since the last checkpoint. Costly or externally stateful objectives require appropriate idempotency and recovery safeguards when exactly-once execution matters.

## User guide

Open `docs/basat_user_guide.ipynb` in your existing notebook editor and run its first setup cell, or use the following command after installing Basat from the project root.

```bash
python -m basat notebook docs/basat_user_guide.ipynb
```

The guide covers objective definitions, algorithm settings, environments, CSV data, stopping rules, checkpoint continuation, plotting, and paper-specific comparisons.

The notebook includes embedded figures that you can view independently of the original result directories. To obtain figures for different settings, rerun the experiments and refresh the notebook presentation. Full reproduction campaigns may take substantial time, download additional data, and require separate numerical environments. See [reproduction setup](docs/reproduction.md) before running them. Paper comparisons document measured behavior under the stated protocols, and exact historical or cross-platform agreement requires separate verification.

## Project structure

```text
basat/          Library source and component license notices
docs/           User-guide notebook and reproduction setup
data/           Small example datasets and provenance
pyproject.toml  Package metadata and dependency settings
```

Results, checkpoints, caches, temporary environments, and manuscript files are local research artifacts, and the ignore rules keep these files out of the repository.

## License

Basat's project code is distributed under the [Apache License 2.0](LICENSE). Source-derived components retain the applicable upstream license notices and terms alongside their implementation files.
