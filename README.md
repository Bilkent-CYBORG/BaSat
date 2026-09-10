# BayOPy

**Bayesian optimization with your own objectives, data, and experiment settings.**

[Quick start](#quick-start) · [Compare algorithms](#compare-algorithms) · [Notebook](docs/bayopy_last_ver_user_guide.ipynb) · [User guide](output/pdf/bayopy_user_guide.pdf)

BayOPy is a Python library for running and comparing Bayesian optimization algorithms. Define an objective, choose an environment and an algorithm, and run an `Experiment`. The same interface supports one algorithm or several, independent trials, early stopping, saved checkpoints, and plots.

- **19 algorithms** across stochastic, contextual, distributional, and adversarial/robust families.
- **User-defined objectives and CSV data**, with environment-specific observations supplied during optimization.
- **Configurable models and policies**, including GP kernels, acquisition functions, and sampling settings where supported by each algorithm.
- **Repeatable experiments**, with seed management, per-trial results, and checkpoint continuation.
- **Built-in plotting**, including comparisons and staggered uncertainty bars across trials.

BayOPy is under active development. The guides describe algorithm assumptions, numerical diagnostics, and cross-environment adaptations.

## Installation

Requires **Python 3.11–3.14**. From the root of this checkout, install BayOPy:

```bash
python -m pip install .
```

This installs the library and its runtime dependencies, including plotting and notebook support. The examples below use only `import bayopy` and Python built-ins.

For an editable installation when developing the library:

```bash
python -m pip install -e .
```

These instructions install the code in this checkout, including the API used by the accompanying notebook.

## Quick start

Save this example as `quickstart.py`:

```python
import bayopy

seed = bayopy.set_seed(42, threads=1)
bounds = bayopy.Bounds(lower=[0.0, 0.0], upper=[1.0, 1.0])


def objective(values):
    return -sum((x - 0.25) ** 2 for x in values)


algorithm = bayopy.make_algorithm("TuRBO", bounds=bounds, seed=seed)
environment = bayopy.StochasticEnvironment(
    bounds=bounds, function=objective, seed=seed,
)
experiment = bayopy.Experiment(environment=environment, algorithm=algorithm)
result = experiment.run(max_evaluations=50, plot=True, show=True)
print(experiment.summary())
```

Run it from your terminal:

```bash
python quickstart.py
```

BayOPy selects inputs within the bounds, calls your objective, and updates the algorithm from the returned rewards. This example has a maximum reward of `0` at `[0.25, 0.25]`. The summary reports the observed results, and the figure shows optimization progress.

An objective returns one finite scalar. **BayOPy maximizes reward by default**: return the negative of a loss when you want to minimize it. Replace the function body with your own calculation, model evaluation, or simulator.

`max_evaluations=50` is an example budget, not a library-wide restriction. An evaluation is one objective call; a round can contain several evaluations when using batches.

## Compare algorithms

Pass a list of algorithms or a dictionary of labels to `algorithm`. Each algorithm runs independently on the environment. A dictionary also lets you compare different settings of the same method.

This example can be run on its own:

```python
import bayopy

seed = bayopy.set_seed(42, threads=1)
bounds = bayopy.Bounds(lower=[0.0, 0.0], upper=[1.0, 1.0])
run_directory = bayopy.new_run_directory("results/readme/comparison")


def objective(values):
    return -sum((x - 0.25) ** 2 for x in values)


algorithms = {
    "TuRBO / EI": bayopy.make_algorithm(
        "TuRBO", bounds=bounds, seed=seed, acquisition="EI",
        initial_points=8, candidate_count=128,
    ),
    "TuRBO / UCB": bayopy.make_algorithm(
        "TuRBO", bounds=bounds, seed=seed,
        acquisition=bayopy.AcquisitionConfig(name="UCB", beta=3.0, seed=seed),
        initial_points=8, candidate_count=128,
    ),
}
experiment = bayopy.Experiment(
    environment=bayopy.StochasticEnvironment(
        bounds=bounds, function=objective, seed=seed,
    ),
    algorithm=algorithms,
    output_directory=run_directory,
)
result = experiment.run(trials=5, max_evaluations=50, plot=False)
print(experiment.summary())
figure = bayopy.plot(
    experiment,
    kind="incumbent",
    title="TuRBO: EI and UCB",
    xlabel="Objective evaluations",
    ylabel="Best observed reward",
    xlim=(1, 50),
    save_path=bayopy.join_path(run_directory, "comparison.png"),
    show=True,
)
```

`trials=5` repeats the entire optimization five times. A configured algorithm seed of `42` becomes `42, 43, 44, 45, 46`; the environment's seed advances from its own base seed. Each algorithm receives up to 50 evaluations **per trial**, or 250 across five trials. The budget and trial count are user-selected.

Repeated-trial plots show means and, by default, vertical bars spanning **mean ± 1.96 sample standard deviations**. These bars describe variability across trials; they are not confidence intervals for the mean. Bar locations are staggered between algorithms. After early stopping, only trials with observations at an evaluation contribute to that point.

For a single run, `trials` defaults to `1`. `plot=True` in `run()` creates a figure; `show=False` prevents automatic display. Use `bayopy.plot()` to customize and save plots, and `bayopy.show(figure)` to display a retained figure explicitly.

## Choose algorithms and settings

List the registered algorithm names:

```python
import bayopy

print(bayopy.algorithm_names())
```

| Family | Algorithms |
| --- | --- |
| Stochastic | REMBO, HeSBO, ALEBO, ADD_GP_UCB, TuRBO, Dropout, MCTS_VS, SAASBO, AG_VS, LA_MCTS, PG, EG, GS, Satisficing_TS |
| Contextual | CGP_UCB |
| Distributional | DRBO, RoBOS |
| Adversarial / robust | StableOPT, AdveRS |

Use `make_algorithm()` for construction by name, or instantiate an algorithm with its configuration class. For example, this constructs TuRBO with UCB and `beta=3`:

```python
import bayopy

bounds = bayopy.Bounds(lower=[0.0, 0.0], upper=[1.0, 1.0])
algorithm = bayopy.TuRBO(
    bounds=bounds,
    config=bayopy.TuRBOConfig(
        seed=42,
        acquisition=bayopy.AcquisitionConfig(name="UCB", beta=3.0, seed=42),
    ),
)
```

Pass this algorithm to an `Experiment` as in the quick start. Omitted optional settings use the algorithm's defaults. There are no parameter profiles. Required scientific choices, such as an embedding dimension or a satisficing threshold, depend on the selected method; supported settings also differ between algorithms. The notebook includes expanded configurations and kernel, acquisition, and sampling examples. For configurable action and context kernels, see the [CGP_UCB kernel comparison](overlaf/code/settings_real_kernels.py).

## Environments and real data

An environment defines how an action is evaluated and what information is available to the learner.

| Environment | Information and evaluation |
| --- | --- |
| `StochasticEnvironment` | Evaluates an action and returns a scalar reward, optionally with observation noise. |
| `ContextualEnvironment` | Reveals context before action selection and evaluates the action in that context. |
| `DistributionalEnvironment` | Provides public distribution information before selection and evaluates the action under a sampled context. |
| `AdversarialEnvironment` | Applies its configured attack and returns the resulting evaluation and feedback. |

Environment-provided information takes precedence over internal algorithm defaults. Algorithms ignore extra inputs they do not need. Where execution is technically possible, missing inputs receive documented defaults and cross-environment adaptations produce warnings. An adaptation does not establish the method's native theoretical guarantees. See the guide's **Algorithm adaptations across environments** appendix for each combination.

For CSV input, use the loader with the appropriate environment family. Run this example from the checkout, which includes the example datasets:

```python
import bayopy

loaded = bayopy.load_dataset(
    bayopy.find_path("data/distributional/wind_power_bandit.csv"),
    environment="distributional",
)
print(loaded.preview(rows=3))
```

The [notebook](docs/bayopy_last_ver_user_guide.ipynb) shows the required CSV schemas and complete environment construction. It includes four distinct real-data examples: diabetes, wine, wind power, and iris. Dataset sources and transformations are documented in [data/README.md](data/README.md). Example datasets live outside the installed library.

## Early stopping and saved runs

To stop on either an evaluation budget or lack of improvement, pass a stopping rule when constructing an experiment:

```python
import bayopy

stopping_rule = bayopy.any_of(
    bayopy.MaximumEvaluationsStoppingRule(50),
    bayopy.NoImprovementStoppingRule(10, min_rounds=20, min_delta=0.001),
)
```

Use `stopping_rule=stopping_rule` in `bayopy.Experiment(...)`. The first applicable run limit or stopping condition ends that algorithm's trial. Time limits, target values, round limits, and `all_of()` composition are also available; see the notebook's **Early Stopping** section.

Set `output_directory` to retain results and checkpoints. Multiple algorithms and trials receive separate directories under the experiment's output folder. `bayopy.new_run_directory()` creates a fresh location for a new experiment.

To continue a saved experiment, reconstruct it with the same objective, environment, base configuration, output directory, and `resume=True`, then call `run()` with the same trial count. Evaluation limits are **totals**, so continuing from 20 to `max_evaluations=50` allows up to 30 additional evaluations per trial. A completed explicit stopping rule remains stopped. The [complete resume example](overlaf/code/settings_resume.py) demonstrates checkpoint continuation.

## Documentation

Open the editable notebook from the project root:

```bash
python -m bayopy notebook docs/bayopy_last_ver_user_guide.ipynb
```

- [Notebook](docs/bayopy_last_ver_user_guide.ipynb): editable objectives, configurations, real-data experiments, and saved results.
- [User guide](output/pdf/bayopy_user_guide.pdf): explanations, code, and figures.
- [Full guide](output/pdf/bayopy_last_ver_user_guide.pdf): includes saved terminal outputs and diagnostics.
- [Overleaf project](overlaf/README.md): `main1.tex` for the full guide and `main2.tex` for the edition without terminal outputs.
- [Overleaf ZIP](overlaf.zip): upload into one project and compile either entry file with pdfLaTeX.

Notebook code can be edited and rerun. Its stored figures represent saved experiments; changing code alone does not update those results.

## License

BayOPy is distributed under the [Apache License 2.0](LICENSE).
