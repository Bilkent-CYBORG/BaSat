import bayopy

seed = bayopy.set_seed(212, threads=1)
data_path = bayopy.find_path("data/distributional/wind_power_bandit.csv")
loaded = bayopy.load_dataset(data_path, environment="distributional")
print(loaded.preview(rows=3))
bounds = loaded.bounds
# This named directory resumes saved trials. Choose another name for a fresh run.
run_directory = bayopy.join_path("results/last_guide/real_distributional_wind", "run_matern_context")

# Fresh, fully explicit configurations for distributional observations.
# These bounded computation settings are editable examples.
# No setting overrides are hidden behind a profile.
dimension = bounds.dimension
reduced_dimension = min(2, dimension)
constructor_options = {
    'maximize': True,
    'dtype': 'float64',
    'device': 'cpu',
    'plot': False,
    'verbose': False,
    'verbose_setting': False,
    'verbose_iter_by_iter': False,
}

# Isotropic kernels model every input; no active coordinates are given.
action_kernel = bayopy.KernelConfig(
    name='matern32',
    ard_num_dims=None,
    active_dims=None,
    use_outputscale=True,
    lengthscale_prior=None,
    outputscale_prior=None,
    kwargs={},
)
context_kernel = bayopy.KernelConfig(
    name='rbf',
    ard_num_dims=None,
    active_dims=None,
    use_outputscale=True,
    lengthscale_prior=None,
    outputscale_prior=None,
    kwargs={},
)
ei = bayopy.AcquisitionConfig(
    name='EI',
    beta=3.5,
    xi=0.0,
    mc_samples=64,
    seed=seed,
)
ucb = bayopy.AcquisitionConfig(
    name='UCB',
    beta=3.5,
    xi=0.0,
    mc_samples=64,
    seed=seed,
)
contextual_solver = bayopy.AcquisitionOptimizerSpec(
    raw_samples=32,
    restarts=2,
    maxiter=50,
    fit_maxiter=50,
    solver_tolerance=1e-06,
)
distributional_solver = bayopy.ActionOptimizerSpec(
    raw_samples=24,
    restarts=2,
    maxiter=50,
    fit_maxiter=50,
)
robust_solver = bayopy.MultiStartSolver(
    method='multistart',
)

rembo_config = bayopy.REMBOConfig(
    embedding_dimension=reduced_dimension,
    finite_candidate_mode='project',
    Y_lower=None,
    Y_upper=None,
    kernel_type='low_dimensional',
    custom_kernel=None,
    acquisition='EI',
    initial_points=10,
    initial_points_sampling_method='sobol',
    batch_size=1,
    q_batch=None,
    lengthscale_lower=0.01,
    lengthscale_upper=50.0,
    initial_lengthscale=None,
    t_sigma=0.002,
    refit_interval=20,
    exploitation_count_limit=5,
    upper_bound_multiplier=0.9,
    seed=seed + 0,
)

hesbo_config = bayopy.HeSBOConfig(
    embedding_dimension=reduced_dimension,
    finite_candidate_mode='project',
    acquisition=ei,
    initial_points=10,
    initial_points_sampling_method='sobol',
    hash_family='paper',
    batch_size=1,
    q_batch=None,
    seed=seed + 1,
)

alebo_config = bayopy.ALEBOConfig(
    embedding_dimension=reduced_dimension,
    finite_candidate_mode='project',
    initial_points=10,
    initial_points_sampling_method='sobol',
    acquisition=ei,
    metric_inference='laplace',
    metric_posterior_samples=8,
    feasibility_tolerance=1e-09,
    acquisition_solver='slsqp',
    raw_feasible_samples=64,
    num_restarts=2,
    batch_size=1,
    q_batch=None,
    seed=seed + 2,
)

add_gp_ucb_config = bayopy.AddGPUCBConfig(
    groups=tuple((i,) for i in range(dimension)),
    maximum_group_size=1,
    learn_decomposition=False,
    decomposition_candidates=None,
    refresh_cycle=25,
    beta=3.5,
    initial_points=10,
    initial_points_sampling_method='sobol',
    component_optimizer=None,
    batch_size=1,
    q_batch=None,
    seed=seed + 3,
)

turbo_config = bayopy.TuRBOConfig(
    batch_size=1,
    q_batch=None,
    initial_points=10,
    initial_points_sampling_method='sobol',
    length_init=0.8,
    length_min=0.0078125,
    length_max=1.6,
    success_tolerance=3,
    failure_tolerance=None,
    candidate_count=128,
    perturbation_cap=20.0,
    acquisition=ei,
    success_relative_tolerance=0.001,
    noisy_center='posterior_mean',
    max_cholesky_size=2000,
    gp_training_steps=50,
    seed=seed + 4,
)

dropout_config = bayopy.DropoutConfig(
    active_dimension=reduced_dimension,
    fill_strategy='mix',
    random_fill_probability=0.1,
    beta=3.5,
    acquisition_optimizer=None,
    initial_points=10,
    initial_points_sampling_method='sobol',
    batch_size=1,
    q_batch=None,
    seed=seed + 5,
)

mcts_vs_config = bayopy.MCTSVSConfig(
    Nv=2,
    Ns=2,
    archive_size=20,
    initial_points=10,
    initial_points_sampling_method='sobol',
    bad_count_limit=5,
    split_threshold=3,
    exploration_constant=0.1,
    inner_optimizer='bo',
    candidate_pool_size=64,
    kernel_lengthscale=0.2,
    acquisition_beta=3.5,
    gp_maxiter=50,
    batch_size=1,
    q_batch=None,
    seed=seed + 6,
)

saasbo_config = bayopy.SAASBOConfig(
    shrinkage_scale=0.1,
    warmup_steps=64,
    posterior_samples=64,
    thinning=4,
    num_chains=1,
    initial_points=10,
    initial_points_sampling_method='sobol',
    acquisition='EI',
    max_tree_depth=4,
    target_accept_prob=0.8,
    num_restarts=2,
    raw_samples=32,
    mc_samples=32,
    ucb_beta=3.5,
    max_r_hat=1.1,
    min_effective_sample_size=10.0,
    allow_divergences=False,
    batch_size=1,
    q_batch=None,
    seed=seed + 7,
)

ag_vs_config = bayopy.AG_VSConfig(
    effective_dimension=reduced_dimension,
    sequence_length=2,
    split_groups=2,
    initial_points=10,
    initial_points_sampling_method='sobol',
    acquisition=ucb,
    stopping_threshold=None,
    stopping_delta=None,
    stagnation_patience=None,
    gp_fit_method='mle',
    gp_maxiter=50,
    acquisition_restarts=2,
    acquisition_raw_samples=64,
    batch_size=1,
    q_batch=None,
    seed=seed + 8,
)

la_mcts_config = bayopy.LAMCTSConfig(
    initial_points=10,
    initial_points_sampling_method='sobol',
    exploration_constant=1.0,
    split_threshold=12,
    min_child_size=3,
    max_depth=20,
    classifier='svm_rbf',
    local_sampler='turbo',
    candidate_pool_size=64,
    max_resampling_attempts=4,
    kernel_lengthscale=0.2,
    acquisition_beta=3.5,
    gp_maxiter=50,
    batch_size=1,
    q_batch=None,
    seed=seed + 9,
)

pg_config = bayopy.GoodActionConfig(
    threshold=-0.1,
    candidate_domain=None,
    initial_points=10,
    initial_points_sampling_method='sobol',
    hyperparameter_refit_cycle=3,
    acquisition_restarts=2,
    reference_point=None,
    lookahead_samples=8,
    maximum_value_samples=8,
    candidate_samples=32,
    fit_maxiter=50,
    posterior_std_floor=1e-12,
    kernel=action_kernel,
    batch_size=1,
    q_batch=None,
    seed=seed + 10,
)

eg_config = bayopy.GoodActionConfig(
    threshold=-0.1,
    candidate_domain=None,
    initial_points=10,
    initial_points_sampling_method='sobol',
    hyperparameter_refit_cycle=3,
    acquisition_restarts=2,
    reference_point=None,
    lookahead_samples=8,
    maximum_value_samples=8,
    candidate_samples=32,
    fit_maxiter=50,
    posterior_std_floor=1e-12,
    kernel=action_kernel,
    batch_size=1,
    q_batch=None,
    seed=seed + 11,
)

gs_config = bayopy.GoodActionConfig(
    threshold=-0.1,
    candidate_domain=None,
    initial_points=10,
    initial_points_sampling_method='sobol',
    hyperparameter_refit_cycle=3,
    acquisition_restarts=2,
    reference_point=None,
    lookahead_samples=8,
    maximum_value_samples=8,
    candidate_samples=32,
    fit_maxiter=50,
    posterior_std_floor=1e-12,
    kernel=action_kernel,
    batch_size=1,
    q_batch=None,
    seed=seed + 12,
)

satisficing_ts_config = bayopy.GoodActionConfig(
    threshold=-0.1,
    candidate_domain=None,
    initial_points=10,
    initial_points_sampling_method='sobol',
    hyperparameter_refit_cycle=3,
    acquisition_restarts=2,
    reference_point=None,
    lookahead_samples=8,
    maximum_value_samples=8,
    candidate_samples=32,
    fit_maxiter=50,
    posterior_std_floor=1e-12,
    kernel=action_kernel,
    batch_size=1,
    q_batch=None,
    seed=seed + 13,
)

cgp_ucb_config = bayopy.CGPUCBConfig(
    action_bounds=bounds,
    action_set=None,
    action_kernel=action_kernel,
    context_kernel=context_kernel,
    kernel_combination='product',
    beta=3.5,
    delta=0.1,
    finite_domain_cardinality=None,
    acquisition_optimizer=contextual_solver,
    initial_points=10,
    batch_size=1,
    q_batch=None,
    seed=seed + 14,
)

drbo_config = bayopy.DRBOConfig(
    context_kernel=context_kernel,
    action_kernel=action_kernel,
    beta=3.5,
    ambiguity_metric='mmd',
    inner_solver='slsqp',
    action_optimizer=distributional_solver,
    distribution_grid=None,
    enumeration_resolution=20,
    initial_points=10,
    feasibility_tolerance=1e-08,
    optimizer_tolerance=1e-06,
    batch_size=1,
    q_batch=None,
    seed=seed + 15,
)

robos_config = bayopy.RoBOSConfig(
    aspiration=-0.1,
    # Matern is numerically well-conditioned on this 11-point context grid.
    context_kernel=bayopy.KernelConfig(name='matern32'),
    action_kernel=action_kernel,
    joint_kernel_combination='product',
    joint_kernel=None,
    beta=3.5,
    simplex_solver='slsqp',
    distribution_floor=0.0,
    distribution_grid=None,
    enumeration_resolution=20,
    action_raw_samples=24,
    action_restarts=2,
    action_maxiter=50,
    initial_points=10,
    fit_maxiter=50,
    feasibility_tolerance=1e-08,
    optimizer_tolerance=1e-06,
    batch_size=1,
    q_batch=None,
    seed=seed + 16,
)

stableopt_config = bayopy.StableOPTConfig(
    perturbation_radius=0.08,
    distance=bayopy.L2Distance(),
    beta=3.5,
    inner_solver=robust_solver,
    outer_solver=robust_solver,
    inner_raw_samples=16,
    outer_raw_samples=32,
    inner_restarts=2,
    outer_restarts=2,
    solver_tolerance=1e-06,
    initial_points=10,
    fit_maxiter=50,
    kernel=action_kernel,
    batch_size=1,
    q_batch=None,
    seed=seed + 17,
)

advers_config = bayopy.AdveRSConfig(
    variant='RS1',
    aspiration=-0.1,
    p=1.0,
    beta=3.5,
    perturbation_domain=bayopy.FullDomainPerturbations(norm='l2', distance_floor=1e-8),
    candidate_solver=robust_solver,
    inner_solver=robust_solver,
    decision_model='ucb',
    outer_raw_samples=32,
    inner_raw_samples=16,
    radius_bisection_steps=8,
    solver_tolerance=1e-06,
    initial_points=10,
    fit_maxiter=50,
    kernel=action_kernel,
    batch_size=1,
    q_batch=None,
    seed=seed + 18,
)

# SAASBO uses actual NUTS. Short chains can report poor ESS/divergences.
# Warnings retain finite draws; they do not certify convergence.
# For inference increase warmup/draws and use at least two chains.
algorithms = {
    'REMBO': bayopy.REMBO(
        bounds=bounds, config=rembo_config,
        **constructor_options,
    ),
    'HeSBO': bayopy.HeSBO(
        bounds=bounds, config=hesbo_config,
        **constructor_options,
    ),
    'ALEBO': bayopy.ALEBO(
        bounds=bounds, config=alebo_config,
        **constructor_options,
    ),
    'ADD_GP_UCB': bayopy.ADD_GP_UCB(
        bounds=bounds, config=add_gp_ucb_config,
        **constructor_options,
    ),
    'TuRBO': bayopy.TuRBO(
        bounds=bounds, config=turbo_config,
        **constructor_options,
    ),
    'Dropout': bayopy.Dropout(
        bounds=bounds, config=dropout_config,
        **constructor_options,
    ),
    'MCTS_VS': bayopy.MCTS_VS(
        bounds=bounds, config=mcts_vs_config,
        **constructor_options,
    ),
    'SAASBO': bayopy.SAASBO(
        bounds=bounds, config=saasbo_config,
        **constructor_options,
        diagnostic_policy='warn',
        jit_compile=True,
        mcmc_num_threads=1,
    ),
    'AG_VS': bayopy.AG_VS(
        bounds=bounds, config=ag_vs_config,
        **constructor_options,
    ),
    'LA_MCTS': bayopy.LA_MCTS(
        bounds=bounds, config=la_mcts_config,
        **constructor_options,
    ),
    'PG': bayopy.PG(
        bounds=bounds, config=pg_config,
        **constructor_options,
    ),
    'EG': bayopy.EG(
        bounds=bounds, config=eg_config,
        **constructor_options,
    ),
    'GS': bayopy.GS(
        bounds=bounds, config=gs_config,
        **constructor_options,
    ),
    'Satisficing_TS': bayopy.Satisficing_TS(
        bounds=bounds, config=satisficing_ts_config,
        **constructor_options,
    ),
    'CGP_UCB': bayopy.CGP_UCB(
        bounds=bounds, config=cgp_ucb_config,
        **constructor_options,
    ),
    'DRBO': bayopy.DRBO(
        bounds=bounds, config=drbo_config,
        **constructor_options,
    ),
    'RoBOS': bayopy.RoBOS(
        bounds=bounds, config=robos_config,
        **constructor_options,
    ),
    'StableOPT': bayopy.StableOPT(
        bounds=bounds, config=stableopt_config,
        **constructor_options,
    ),
    'AdveRS': bayopy.AdveRS(
        bounds=bounds, config=advers_config,
        **constructor_options,
    ),
}
print('Configured algorithms:', len(algorithms), list(algorithms))


# Real hourly German wind generation, from the data package cited by DRBO.
# This CSV subset has no quoted fields; column 4 is recorded capacity factor.
with open(bayopy.find_path("data/distributional/opsd_wind_2013.csv")) as stream:
    source_lines = stream.read().splitlines()
wind_profiles = [float(line.split(",")[3]) for line in source_lines[1:99]]
# Fixed 11-point support: quantize observed capacity factors to nearest 0.1.
wind_indices = tuple(min(10, max(0, int(value * 10 + 0.5))) for value in wind_profiles)
assert len(wind_indices) == 98  # 48 history hours + 50 evaluation hours


def reference_at_round(round_index):
    # Only past observations contribute to the public 48-hour reference.
    history = wind_indices[round_index:round_index + 48]
    return [history.count(j) / len(history) for j in range(11)]


def realized_wind_at_round(round_index):
    # The environment keeps this next observation private until evaluation.
    if not 0 <= round_index < 50:
        raise ValueError("This replay contains 50 evaluation hours.")
    actual_index = wind_indices[48 + round_index]
    return [float(j == actual_index) for j in range(11)]


# Every CSV reward is calculated from the paper's revenue formula:
# def objective(x, c):
#     return 0.1 * max(c - x, 0) + min(x, c) - 5 * max(x - c, 0)
# x and c are fractions of installed capacity. Revenue is modeled, not measured.
# The initial CSV probabilities are overridden by the providers above.
environment = bayopy.DistributionalEnvironment(
    bounds=bounds, dataset=loaded.dataset, function=None,
    context_support=loaded.context_support,
    reference_distribution=reference_at_round,
    sampling_distribution=realized_wind_at_round,
    margin=loaded.margin,
    previous_data=None, dtype="float64", device="cpu",
    noise_std=0.0, noise_fn=None, add_noise_to_data=False,
    allow_repeated_actions=True, data_row_selection="first", seed=seed,
)

assert set(algorithms) == set(bayopy.algorithm_names())
print("Algorithms submitted:", len(algorithms))
print("Names:", list(algorithms))
experiment = bayopy.Experiment(
    environment=environment, algorithm=algorithms,
    output_directory=run_directory, resume=True, save_every=10,
    stopping_rule=None,
)
result = experiment.run(
    trials=5, max_evaluations=50, plot=False, show=False, progress=True,
)
print(experiment.summary())
print(experiment.diagnostics())
figure = bayopy.plot(
    experiment, kind="comparison", title="Wind-power commitment / distributional: all 19 algorithms",
    titles=("Observed reward", "Best observed reward so far"),
    xlabel="Objective evaluations", ylabel="Reward (higher is better)",
    figsize=(12.0, 7.5), xlim=(1, 50), grid=True,
    legend=True, legend_columns=4, legend_fontsize=12.0,
    save_path=bayopy.join_path(run_directory, "all_algorithms.png"),
    dpi=180, show=True,
)
print("Saved results:", run_directory)
