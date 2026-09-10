import bayopy

seed = bayopy.set_seed(411, threads=1)
run_directory = bayopy.new_run_directory("results/last_guide/acquisitions")
bounds = bayopy.Bounds(lower=[0.0] * 3, upper=[1.0] * 3)

def objective(values):
    return -sum((x - 0.4) ** 2 for x in values)

algorithms = {
    "UCB beta=3": bayopy.TuRBO(
        bounds=bounds, config=bayopy.TuRBOConfig(
            acquisition=bayopy.AcquisitionConfig(name="UCB", beta=3.0, seed=seed),
            initial_points=8, candidate_count=128, seed=seed,
        ), dtype="float64",
    ),
    "UCB beta=9": bayopy.TuRBO(
        bounds=bounds, config=bayopy.TuRBOConfig(
            acquisition=bayopy.AcquisitionConfig(name="UCB", beta=9.0, seed=seed),
            initial_points=8, candidate_count=128, seed=seed,
        ), dtype="float64",
    ),
    "EI xi=0.01": bayopy.TuRBO(
        bounds=bounds, config=bayopy.TuRBOConfig(
            acquisition=bayopy.AcquisitionConfig(name="EI", xi=0.01, seed=seed),
            initial_points=8, candidate_count=128, seed=seed,
        ), dtype="float64",
    ),
}
experiment = bayopy.Experiment(
    environment=bayopy.StochasticEnvironment(
        bounds=bounds, function=objective, dtype="float64", seed=seed,
    ), algorithm=algorithms, output_directory=run_directory,
)
result = experiment.run(trials=5, max_evaluations=50, plot=False, progress=True)
print(experiment.summary())
figure = bayopy.plot(
    experiment, kind="incumbent", title="Acquisitions inside TuRBO",
    xlim=(1, 50), save_path=bayopy.join_path(run_directory, "acquisitions.png"),
)
