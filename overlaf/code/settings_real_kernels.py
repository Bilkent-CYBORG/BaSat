import bayopy

seed = bayopy.set_seed(412, threads=1)
run_directory = bayopy.new_run_directory("results/last_guide/real_kernels")
loaded = bayopy.load_dataset(
    bayopy.find_path("data/contextual/wine_bandit.csv"), environment="contextual",
)
bounds = loaded.bounds
optimizer = bayopy.AcquisitionOptimizerSpec(
    raw_samples=32, restarts=2, maxiter=50, fit_maxiter=50,
)
algorithms = {
    "CGP_UCB / RBF": bayopy.CGP_UCB(
        bounds=bounds, config=bayopy.CGPUCBConfig(
            action_bounds=bounds, action_kernel=bayopy.KernelConfig(name="rbf"),
            context_kernel=bayopy.KernelConfig(name="rbf"), beta=3.0,
            initial_points=8, acquisition_optimizer=optimizer, seed=seed,
        ), dtype="float64",
    ),
    "CGP_UCB / Matern32": bayopy.CGP_UCB(
        bounds=bounds, config=bayopy.CGPUCBConfig(
            action_bounds=bounds, action_kernel=bayopy.KernelConfig(name="matern32"),
            context_kernel=bayopy.KernelConfig(name="rbf"), beta=3.0,
            initial_points=8, acquisition_optimizer=optimizer, seed=seed,
        ), dtype="float64",
    ),
    "CGP_UCB / Matern52": bayopy.CGP_UCB(
        bounds=bounds, config=bayopy.CGPUCBConfig(
            action_bounds=bounds, action_kernel=bayopy.KernelConfig(name="matern52"),
            context_kernel=bayopy.KernelConfig(name="rbf"), beta=3.0,
            initial_points=8, acquisition_optimizer=optimizer, seed=seed,
        ), dtype="float64",
    ),
}
experiment = bayopy.Experiment(
    environment=bayopy.ContextualEnvironment(
        bounds=bounds, dataset=loaded.dataset, dtype="float64", seed=seed,
    ), algorithm=algorithms, output_directory=run_directory,
)
result = experiment.run(trials=5, max_evaluations=50, plot=False, progress=True)
print(experiment.summary())
print(experiment.diagnostics())
figure = bayopy.plot(
    experiment, kind="incumbent", title="Wine: GP kernel comparison",
    xlim=(1, 50), save_path=bayopy.join_path(run_directory, "kernels.png"),
)
