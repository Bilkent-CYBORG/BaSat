import bayopy

seed = bayopy.set_seed(414, threads=1)
run_directory = bayopy.new_run_directory("results/last_guide/resume")
bounds = bayopy.Bounds(lower=[0.0] * 3, upper=[1.0] * 3)

def objective(values):
    return -sum((x - 0.45) ** 2 for x in values)

def make_experiment(resume):
    return bayopy.Experiment(
        environment=bayopy.StochasticEnvironment(
            bounds=bounds, function=objective, dtype="float64", seed=seed,
        ),
        algorithm=bayopy.TuRBO(
            bounds=bounds, config=bayopy.TuRBOConfig(
                initial_points=8, candidate_count=128, seed=seed,
            ), dtype="float64",
        ), output_directory=run_directory, resume=resume, save_every=5,
    )

first = make_experiment(False)
first_result = first.run(trials=5, max_evaluations=10, plot=True)
resumed = make_experiment(True)
resumed_result = resumed.run(trials=5, max_evaluations=50, plot=True, show=False)
print("Before:", first_result.evaluations, "After:", resumed_result.evaluations)
print(resumed.summary())
figure = bayopy.plot(
    resumed, kind="incumbent", title="Checkpoint continuation to evaluation50",
    xlabel="Evaluation", ylabel="Best reward", xlim=(1, 50),
    grid=True, legend=True, legend_columns=1, marker="o",
    save_path=bayopy.join_path(run_directory, "resumed.png"), show=False,
)
print("show=False retained the plot. Display it explicitly next:")
bayopy.show(figure)
