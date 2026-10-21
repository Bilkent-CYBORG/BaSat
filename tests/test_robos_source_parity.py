"""Numerical contracts for the released RoBOS finite-context experiment.

The small GP constants below were measured with the official GPyModel wrapper
at RoBOS revision 96bf39400a0b2044e3c21e08b4fbeb8886584ee3, GPy 1.13.2,
NumPy 1.26.4 and SciPy 1.12.0. They are independent of Basat's GP backend.
The bounded external-source audit supplements these portable regression tests.
"""

import copy
import warnings

import numpy as np
import pytest
import scipy
import scipy.optimize
import torch

import basat
from basat.algorithms.Distributional.Exception.exception_RoBOS import RoBOSWarning


def _gp_config():
    return basat.GaussianProcessConfig(
        kernel=basat.FixedRBFKernel(input_dimension=2, lengthscale=[0.2, 5.0], ard=True),
        mean="zero",
        noise_mode="fixed",
        noise_variance=0.0004,
        standardize_outcomes=False,
        dtype="float64",
    )


def _learner(kind="RoBOS", *, tie="first"):
    options = dict(
        initial_points=3,
        gp_config=_gp_config(),
        gp_backend="fixed_gpy",
        gp_training_regularization=1e-8,
        beta=4.0,
        seed=123,
        context_kernel=basat.FixedRBFKernel(input_dimension=1, lengthscale=5.0),
    )
    if kind == "RoBOS":
        config = basat.RoBOSConfig(
            **options,
            aspiration=0.6,
            fit_maxiter=0,
            simplex_solver="source_cobyla",
            allow_semidefinite_mmd=True,
        )
    else:
        config = basat.DRBOConfig(
            **options,
            inner_solver="source_cvxpy",
            finite_action_tie_breaking=tie,
            action_optimizer=basat.ActionOptimizerSpec(fit_maxiter=0),
        )
    learner = getattr(basat, kind)(
        bounds=basat.Bounds([-1.0], [1.0]),
        config=config,
        dtype="float64",
        verbose=False,
        plot=False,
    )
    learner.bind_environment_metadata({"context_support": [[-1.0], [1.0]]})
    return learner


@pytest.mark.parametrize(
    "n,expected_mean,expected_variance",
    [
        (
            2,
            [0.10038960697714404, 0.501295515899514, -0.012461482405991429, 4.221186173347974e-6],
            [0.0003993761433745968, 0.03620882368611844, 0.9989954762248121, 0.9999999999546517],
        ),
        (
            4,
            [0.11017433191265624, 0.49237868030869536, -0.09581975527488228, 3.977314803616543e-6],
            [0.00019983803890233354, 0.035299642841135404, 0.6329073254303681, 0.9999999999537176],
        ),
    ],
)
def test_fixed_gpy_matches_external_source_fixture_and_replays(n, expected_mean, expected_variance):
    train = np.array([[0.0, 0.0], [0.1, 1.0], [-0.3, 5.0], [0.0, 0.0]])[:n]
    observations = np.array([0.1, 0.4, -0.2, 0.12])[:n]
    query = np.array([[0.0, 0.0], [0.15, 2.0], [-0.5, 6.0], [1.0, 12.0]])
    gp = basat.FixedKernelGaussianProcess(
        _gp_config(),
        input_dimension=2,
        conditioning="full",
        arithmetic="gpy",
        training_regularization=1e-8,
    )
    with pytest.warns(basat.BasatNumericalWarning, match="training regularization"):
        gp.set_data(train, observations)
    mean, std = gp.predict(query)
    np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=2e-12)
    np.testing.assert_allclose(std.square(), expected_variance, rtol=0, atol=2e-12)
    assert not gp.jitter_adjustments.any()
    np.testing.assert_array_equal(gp.observation_noise, np.full(n, 0.0004))
    restored = basat.FixedKernelGaussianProcess(
        _gp_config(),
        input_dimension=2,
        conditioning="full",
        arithmetic="gpy",
        training_regularization=1e-8,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored.load_state_dict(gp.state_dict())
    assert not caught
    assert all(
        torch.equal(a, b) for a, b in zip(gp.predict(query), restored.predict(query), strict=True)
    )


def test_fixed_rbf_matches_gpy_kernel_fixture():
    query = torch.tensor([[0.0, 0.0], [0.15, 2.0], [-0.5, 6.0], [1.0, 12.0]], dtype=torch.float64)
    expected = [
        [1.0, 0.6968047754960348, 0.021386401561157106, 2.091947919797089e-7],
        [0.6968047754960348, 1.0, 0.0036932442745900023, 1.6187843478185913e-5],
        [0.021386401561157106, 0.0036932442745900023, 1.0, 2.9701314435492996e-13],
        [2.091947919797089e-7, 1.6187843478185913e-5, 2.9701314435492996e-13, 1.0],
    ]
    np.testing.assert_allclose(_gp_config().kernel(query).to_dense(), expected, rtol=0, atol=2e-15)


@pytest.mark.parametrize("kind", ["RoBOS", "DRBO"])
def test_batched_context_major_ucb_and_source_confidence(kind, monkeypatch):
    learner = _learner(kind)
    received = []

    def predict(points):
        received.append(points.clone())
        return torch.arange(len(points), dtype=torch.float64), torch.ones(
            len(points), dtype=torch.float64
        )

    monkeypatch.setattr(learner.gp, "predict", predict)
    upper = learner._ucb_matrix(torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64))
    assert len(received) == 1
    assert received[0].tolist() == [
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    assert upper.tolist() == [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
    assert upper[:, 1].stride() == (3,)


def test_source_rng_consumption_and_checkpoint_are_exact(monkeypatch):
    from types import SimpleNamespace

    received = []

    def minimize(objective, initial, **options):
        received.append(initial.copy())
        return SimpleNamespace(x=initial, success=True, status=1, nfev=1, message="fixture")

    monkeypatch.setattr(scipy.optimize, "minimize", minimize)
    learner = _learner()
    stream = basat.RandomStream(123)
    expected_rng = np.random.RandomState(123)
    # Even noiseless initial responses consume normal draws in the notebook.
    np.testing.assert_array_equal(stream.randint(0, 400, 3), expected_rng.randint(0, 400, 3))
    np.testing.assert_array_equal(stream.normal(size=3), expected_rng.randn(3))
    np.testing.assert_array_equal(stream.uniform(size=201), expected_rng.rand(201))
    learner.set_source_solver_state(stream.get_state())
    arguments = ([0.3, 0.9], [0.5, 0.5], [[1.0, 0.2], [0.2, 1.0]])
    learner.evaluate_fragility(*arguments)
    np.testing.assert_array_equal(
        received[0], np.diff(np.r_[0.0, np.sort(expected_rng.rand(1)), 1.0])
    )
    learner.set_source_solver_starts([[0.2, 0.8], [0.6, 0.4]])
    learner.evaluate_fragility(*arguments)
    restored = _learner()
    restored.load_state_dict(learner.state_dict())
    before = learner.get_source_solver_state()
    a, b = learner.evaluate_fragility(*arguments), restored.evaluate_fragility(*arguments)
    assert a[0] == b[0] and torch.equal(a[1], b[1])
    assert learner.get_source_solver_state() == restored.get_source_solver_state() == before
    np.testing.assert_array_equal(received[-1], [0.6, 0.4])
    invalid = copy.deepcopy(before)
    invalid["position"] = -1
    with pytest.raises(Exception, match="Invalid source solver RNG"):
        learner.set_source_solver_state(invalid)
    assert learner.get_source_solver_state() == before


def test_source_cobyla_matches_literal_ratio_program_and_reports_backend():
    learner = _learner()
    start = np.array([0.31, 0.69])
    ref = np.array([0.45, 0.55])
    upper = np.array([0.35, 0.95])
    gram = np.array([[1.0, 0.2], [0.2, 1.0]])

    def source_ratio(w):
        delta = (ref - w).reshape(-1, 1)
        distance = float(np.sqrt(np.dot(delta.T, np.dot(gram, delta))).item())
        return float(((0.6 - np.dot(upper.reshape(1, -1), w[:, None])) / (distance + 1e-9)).item())

    source = scipy.optimize.minimize(
        lambda w: -source_ratio(w),
        start,
        method="COBYLA",
        constraints=[
            {"type": "ineq", "fun": lambda w: w},
            {"type": "ineq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: 1 - np.sum(w)},
        ],
        options={"maxiter": 1000},
    )
    learner.set_source_solver_starts(start[None, :])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RoBOSWarning)
        value, witness, _ = learner.evaluate_fragility(upper, ref, gram)
    np.testing.assert_array_equal(witness, source.x)
    assert value == source_ratio(source.x)
    report = learner._source_solver_diagnostics
    assert report["scipy_version"] == scipy.__version__
    expected = (
        "powell_fortran" if tuple(map(int, scipy.__version__.split(".")[:2])) < (1, 16) else "prima"
    )
    assert report["backend"] == expected
    assert not report["global_optimum_certified"]


def test_cold_clarabel_has_analytic_mmd_ball_value_and_no_problem_reuse(monkeypatch):
    import cvxpy as cp

    learner = _learner("DRBO")
    solve = cp.Problem.solve
    calls = []

    def record(problem, **kwargs):
        calls.append((problem, kwargs.copy()))
        return solve(problem, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", record)
    gram = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float64)
    ref = torch.tensor([0.5, 0.5], dtype=torch.float64)
    upper = torch.tensor([0.0, 1.0], dtype=torch.float64)
    for _ in range(2):
        value, witness, residuals = learner._worst_case_expected_ucb(upper, ref, 0.2, gram)
        assert value == pytest.approx(0.5 - 0.2 / np.sqrt(1.6), abs=2e-8)
        delta = witness - ref
        assert float(delta @ gram @ delta) <= 0.2**2 + 1e-9
        assert max(residuals[key] for key in ("simplex_sum", "nonnegativity", "mmd_radius")) < 1e-8
    assert len(calls) == 2 and calls[0][0] is not calls[1][0]
    assert all(
        options["solver"] == "CLARABEL" and not options["warm_start"] for _, options in calls
    )


def test_cold_source_solver_failure_does_not_change_backend(monkeypatch):
    import cvxpy as cp

    calls = []

    def unavailable(problem, **options):
        calls.append(options["solver"])
        raise RuntimeError("injected Clarabel failure")

    monkeypatch.setattr(cp.Problem, "solve", unavailable)
    learner = _learner("DRBO")
    with pytest.raises(Exception, match="cold CLARABEL solve"):
        learner._solve_cvxpy(
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.5, 0.5]),
            0.2,
            torch.tensor([[1.0, 0.2], [0.2, 1.0]]),
        )
    assert calls == ["CLARABEL"]


def test_source_argmax_does_not_group_near_ties(monkeypatch):
    learner = _learner("DRBO")
    monkeypatch.setattr(
        learner,
        "_ucb_matrix",
        lambda actions: torch.tensor([[1.0, 1.0 + 1e-12], [1.0, 1.0]], dtype=torch.float64),
    )
    monkeypatch.setattr(
        learner,
        "_worst_case_expected_ucb",
        lambda upper, reference, margin, gram: (float(upper[0]), reference, {}),
    )
    action = learner._select_robust_action(
        torch.tensor([[-1.0], [1.0]], dtype=torch.float64),
        torch.tensor([0.5, 0.5]),
        0.2,
        torch.arange(2),
    )[0]
    assert float(action[0]) == 1.0
