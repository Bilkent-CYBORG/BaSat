import bayopy

seed = bayopy.set_seed(413, threads=1)
run_directory = bayopy.new_run_directory("results/last_guide/sampling_batches")
bounds = bayopy.Bounds(lower=[0.0] * 3, upper=[1.0] * 3)

def objective(values):
    return -sum((x - 0.6) ** 2 for x in values)

algorithms = {
    "Sobol / batch2": bayopy.TuRBO(
        bounds=bounds, config=bayopy.TuRBOConfig(
            initial_points=8, candidate_count=128, batch_size=2, seed=seed,
        ), dtype="float64",
    ),
    "Uniform / batch2": bayopy.TuRBO(
        bounds=bounds, config=bayopy.TuRBOConfig(
            initial_points=8, candidate_count=128, batch_size=2, seed=seed,
        ), dtype="float64",
    ),
    "Joint EI / q2": bayopy.TuRBO(
        bounds=bounds, config=bayopy.TuRBOConfig(
            initial_points=8, candidate_count=128, batch_size=None, q_batch=2,
            acquisition=bayopy.AcquisitionConfig(
                name="EI", mc_samples=64, seed=seed,
            ), seed=seed,
        ), dtype="float64",
    ),
}
experiment = bayopy.Experiment(
    environment=bayopy.StochasticEnvironment(
        bounds=bounds, function=objective, dtype="float64", seed=seed,
    ), algorithm=algorithms, output_directory=run_directory,
    initial_points_sampling_method={
        "Sobol / batch2": "sobol", "Uniform / batch2": "uniform",
        "Joint EI / q2": "sobol",
    },
    acquisition_sampling_method={
        "Sobol / batch2": "sobol", "Uniform / batch2": "uniform",
        "Joint EI / q2": "sobol",
    },
)
result = experiment.run(trials=5, max_evaluations=50, plot=False, progress=True)
print(experiment.summary())
figure = bayopy.plot(
    experiment, kind="incumbent", title="Sampling and two-action batches",
    xlim=(1, 50), save_path=bayopy.join_path(run_directory, "batches.png"),
)
