import bayopy

seed = bayopy.set_seed(310, threads=1)
run_directory = bayopy.new_run_directory("results/last_guide/early_patience")
bounds = bayopy.Bounds(lower=[0.0] * 3, upper=[1.0] * 3)

def objective(values):
    return -sum((x - 0.25) ** 2 for x in values)

def make_stopping():
    return bayopy.any_of(
        bayopy.MaximumEvaluationsStoppingRule(50),
        bayopy.NoImprovementStoppingRule(
            10, min_rounds=20, min_delta=0.001,
        ),
    )

algorithms = {
    "TuRBO / patience": bayopy.TuRBO(
        bounds=bounds, config=bayopy.TuRBOConfig(
            initial_points=8, candidate_count=128, seed=seed,
        ), dtype="float64",
    ),
    "AG_VS / patience": bayopy.AG_VS(
        bounds=bounds, config=bayopy.AG_VSConfig(
            effective_dimension=2, initial_points=8, sequence_length=2,
            acquisition_restarts=2, acquisition_raw_samples=32, seed=seed,
            stopping_threshold=None, stopping_delta=None,
        ), dtype="float64",
    ),
}
experiment = bayopy.Experiment(
    environment=bayopy.StochasticEnvironment(
        bounds=bounds, function=objective, dtype="float64", seed=seed,
    ), algorithm=algorithms, stopping_rule=make_stopping(),
    output_directory=run_directory,
)
result = experiment.run(trials=5, max_evaluations=50, plot=False, progress=True)
print(experiment.summary())
figure = bayopy.plot(
    experiment, kind="incumbent", title="Patience: independent stopping histories",
    xlim=(1, 50), save_path=bayopy.join_path(run_directory, "patience.png"),
)
