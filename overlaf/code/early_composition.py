import bayopy

seed = bayopy.set_seed(311, threads=1)
run_directory = bayopy.new_run_directory("results/last_guide/early_composition")
bounds = bayopy.Bounds(lower=[0.0] * 3, upper=[1.0] * 3)

def objective(values):
    return -sum((x - 0.5) ** 2 for x in values)

def make_stopping():
    return bayopy.any_of(
        bayopy.MaximumEvaluationsStoppingRule(50),
        bayopy.MaximumRoundsStoppingRule(35),
        bayopy.MaximumTimeStoppingRule(20.0),
        bayopy.all_of(
            bayopy.MaximumRoundsStoppingRule(12),
            bayopy.TargetValueStoppingRule(-0.05, maximize=True),
        ),
    )

algorithm = bayopy.TuRBO(
    bounds=bounds,
    config=bayopy.TuRBOConfig(
        initial_points=8, candidate_count=128,
        acquisition=bayopy.AcquisitionConfig(name="EI", xi=0.0, seed=seed),
        seed=seed,
    ), dtype="float64",
)
experiment = bayopy.Experiment(
    environment=bayopy.StochasticEnvironment(
        bounds=bounds, function=objective, dtype="float64", seed=seed,
    ), algorithm=algorithm, stopping_rule=make_stopping(),
    output_directory=run_directory,
)
result = experiment.run(trials=5, max_evaluations=50, plot=False)
print(experiment.summary())
print("Triggered conditions by trial:", [
    trial.stopping_decision for trial in experiment.trial_experiments
])
figure = bayopy.plot(
    experiment, kind="incumbent", title="Round, time, and target conditions",
    xlim=(1, 50), save_path=bayopy.join_path(run_directory, "conditions.png"),
)
