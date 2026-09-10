import bayopy

seed = bayopy.set_seed(410, threads=1)
run_directory = bayopy.new_run_directory("results/last_guide/defaults")
bounds = bayopy.Bounds(lower=[0.0] * 3, upper=[1.0] * 3)

def objective(values):
    return -sum((x - 0.3) ** 2 for x in values)

algorithms = [
    bayopy.TuRBO(
        bounds=bounds, config=bayopy.TuRBOConfig(seed=seed), dtype="float64",
    ),
    bayopy.AG_VS(
        bounds=bounds,
        config=bayopy.AG_VSConfig(effective_dimension=2, seed=seed),
        dtype="float64",
    ),
]
print("TuRBO defaults:", algorithms[0].config)
experiment = bayopy.Experiment(
    environment=bayopy.StochasticEnvironment(
        bounds=bounds, function=objective, dtype="float64", seed=seed,
    ), algorithm=algorithms, output_directory=run_directory,
)
result = experiment.run(trials=5, max_evaluations=50, plot=False, progress=True)
print(experiment.summary())
figure = bayopy.plot(
    experiment, kind="incumbent", title="Default settings with an algorithm list",
    xlim=(1, 50), save_path=bayopy.join_path(run_directory, "defaults.png"),
)
