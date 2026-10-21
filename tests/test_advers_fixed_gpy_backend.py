"""Opt-in fixed GPy arithmetic, independent kernel references and strict state."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest
import torch

import basat
from basat.algorithms.algorithm import AlgorithmConfigurationError, AlgorithmStateError
from basat.models.GaussianProcess import InvalidModelDataError, ModelStateError
from basat.warnings import BasatNumericalWarning

torch.set_num_threads(1)


def config(kernel=None):
    return basat.GaussianProcessConfig(kernel=kernel or basat.FixedPolynomialKernel(input_dimension=2,
        scale=1.11, bias=2.84, variance=1.32, order=6), mean="zero", noise_mode="fixed",
        noise_variance=1., standardize_outcomes=False, dtype="float64", device="cpu", jitter=1e-10)


def make(method=0, **changes):
    settings = dict(beta=1., initial_points=2, fit_maxiter=0, gp_config=config(), gp_backend="fixed_gpy",
                    gp_training_regularization=1e-8, source_protocol="advers_2026_camera_ready", seed=19)
    settings.update(changes)
    common = dict(bounds=basat.Bounds(lower=[-1., -1.], upper=[1., 1.]), dtype="float64", device="cpu", plot=False, verbose=False)
    if method == 0:
        return basat.AdveRS(**common, config=basat.AdveRSConfig(variant="RS1", aspiration=0., **settings))
    return basat.StableOPT(**common, config=basat.StableOPTConfig(perturbation_radius=.5,
        query_policy="center", **settings))


def initialize(algorithm):
    data = basat.EnvironmentData(actions=basat.array([[-.8, -.6], [.1, .3], [.7, .5]], dtype="float64"),
        observations=basat.array([-1., .3, .8], dtype="float64"), observation_noise_variance=basat.array([1., 1., 1.]))
    with pytest.warns(BasatNumericalWarning, match="training regularization"):
        algorithm.initialize_from_environment(data)
    return algorithm


@pytest.fixture(scope="module")
def gpy_reference():
    """Run only a tiny independent kernel fixture, never a source campaign."""
    python = Path(__file__).resolve().parents[1] / "tmp/advers_camera_ready_runtime/bin/python"
    if not python.is_file():
        pytest.skip("Optional independent GPy reference environment is not installed.")
    source = '''import json,GPy,numpy as np
X=np.array([[.1,.7,.3],[.8,.2,.5],[.4,.6,.9],[.35,.1,.25],[.23,.54,.91]])
Z=np.array([[.11,.72,.31],[.81,.23,.51],[.45,.64,.97]])
out=[]
for order in (6,6.0,2.5):
 k=GPy.kern.Poly(3,scale=1.11,bias=2.84,variance=1.32,order=order)
 out.append(dict(order=order,kind=type(order).__name__,self_cov=k.K(X).tolist(),cross=k.K(X,Z).tolist(),diag=k.Kdiag(X).tolist()))
print(json.dumps(dict(numpy=np.__version__,X=X.tolist(),Z=Z.tolist(),cases=out)))
'''
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    return json.loads(subprocess.check_output([str(python), "-c", source], text=True, env=environment))


def test_fixed_polynomial_arrays_match_independent_gpy_not_algebraic_rewrite(gpy_reference):
    x, z = (torch.tensor(gpy_reference[name], dtype=torch.float64) for name in ("X", "Z"))
    for case in gpy_reference["cases"]:
        order = int(case["order"]) if case["kind"] == "int" else float(case["order"])
        kernel = basat.FixedPolynomialKernel(input_dimension=3, scale=1.11, bias=2.84, variance=1.32, order=order)
        assert type(kernel.order) is type(order)
        for actual, expected in ((kernel(x).to_dense(), case["self_cov"]), (kernel(x, z).to_dense(), case["cross"]),
                                 (kernel(x, diag=True), case["diag"])):
            # Source-stack runs require exact arrays; the ordinary modern
            # runtime is independently supported, not claimed bitwise GPy.
            if np.__version__ == gpy_reference["numpy"]:
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)


def test_self_covariance_slices_once_cross_twice_and_diag_uses_full_dot(monkeypatch):
    kernel = basat.FixedPolynomialKernel(input_dimension=2)
    x = torch.tensor([[.1, .2], [.3, .4], [.6, .8]], dtype=torch.float64)
    original, calls = np.dot, []
    def recording(left, right):
        calls.append((left, right))
        assert left.flags.f_contiguous and right.flags.c_contiguous
        return original(left, right)
    monkeypatch.setattr(np, "dot", recording)
    full = kernel(x).to_dense()
    assert np.shares_memory(*calls[-1])
    diagonal = kernel(x, diag=True)
    assert calls[-1][0].shape == (3, 2) and calls[-1][1].shape == (2, 3)
    assert np.shares_memory(*calls[-1])
    assert torch.equal(diagonal, full.diagonal())
    kernel(x, x.clone()).to_dense()
    assert not np.shares_memory(*calls[-1])


@pytest.mark.parametrize("corruption", ["input-gradient", "float32", "batched", "training", "nonfinite"])
def test_fixed_polynomial_rejects_unsupported_inputs_without_detaching(corruption):
    kernel = basat.FixedPolynomialKernel(input_dimension=2)
    x = torch.tensor([[.2, .4], [.1, .3]], dtype=torch.float64)
    if corruption == "input-gradient":
        x.requires_grad_(True)
    elif corruption == "float32":
        x = x.float()
    elif corruption == "batched":
        x = x.unsqueeze(0)
    elif corruption == "training":
        kernel.train()
    else:
        x[0, 0] = float("nan")
    with pytest.raises((ValueError, RuntimeError)):
        kernel(x).to_dense()


@pytest.mark.parametrize("method", [0, 3])
@pytest.mark.parametrize("rows", [1, 4])
def test_finite_confidence_shape_and_regularization_are_explicit(method, rows):
    algorithm = initialize(make(method))
    points = torch.zeros((rows, 2), dtype=torch.float64)
    upper, lower = algorithm.confidence_bounds(points)
    mean, std = algorithm.gp.predict(points)
    assert upper.shape == lower.shape == (rows,)
    assert torch.equal(upper, mean+std) and torch.equal(lower, mean-std)
    assert isinstance(algorithm.gp, basat.FixedKernelGaussianProcess)
    assert algorithm.gp._inference_settings() == {"conditioning": "full", "arithmetic": "gpy", "training_regularization": 1e-8}
    assert torch.equal(algorithm.gp.observation_noise, torch.ones(3, dtype=torch.float64))
    assert torch.equal(algorithm.gp.jitter_adjustments, torch.zeros(3, dtype=torch.float64))
    with pytest.raises(InvalidModelDataError, match="gradients"):
        algorithm.confidence_bounds(points.requires_grad_(True))
    with pytest.raises(AlgorithmConfigurationError, match="finite candidate_actions"):
        algorithm.ask(round_info={})


@pytest.mark.parametrize("method", [0, 3])
def test_state_roundtrip_and_backend_regularization_kernel_guards(method):
    algorithm = initialize(make(method))
    points = torch.tensor([[.2, .4]], dtype=torch.float64)
    expected = algorithm.confidence_bounds(points)
    state = algorithm.state_dict()
    restored = make(method)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored.load_state_dict(state)
    assert not [warning for warning in caught if issubclass(warning.category, BasatNumericalWarning)]
    actual = restored.confidence_bounds(points)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert restored.get_source_random_state() == algorithm.get_source_random_state()
    for changes in ({"gp_training_regularization": 0.}, {"gp_backend": "standard", "gp_training_regularization": 0.},
                    {"gp_config": config(basat.FixedPolynomialKernel(input_dimension=2, scale=1.12,bias=2.84,variance=1.32,order=6))},
                    {"gp_config": config(basat.FixedPolynomialKernel(input_dimension=2, scale=1.11,bias=2.84,variance=1.32,order=6.0))}):
        with pytest.raises(AlgorithmStateError, match="configuration"):
            make(method, **changes).load_state_dict(state)
    gp_state = algorithm.gp.state_dict()
    bad = basat.FixedKernelGaussianProcess(config(), input_dimension=2, conditioning="full", arithmetic="gpy", training_regularization=0.)
    with pytest.raises(ModelStateError, match="inference protocol"):
        bad.load_state_dict(gp_state)


@pytest.mark.parametrize("method", [0, 3])
@pytest.mark.parametrize("changes", [{"fit_maxiter": 1}, {"gp_config": None}, {"gp_training_regularization": -1.},
    {"gp_training_regularization": True}, {"gp_training_regularization": float("nan")},
    {"gp_backend": "polynomial_features"}, {"gp_backend": "standard"}])
def test_fixed_backend_invalid_configurations_fail_before_scientific_work(method, changes):
    with pytest.raises((AlgorithmConfigurationError, ValueError)):
        make(method, **changes)


@pytest.mark.parametrize("method", [0, 3])
def test_default_backend_and_legacy_missing_regularization_remain_compatible(method):
    configuration = basat.AdveRSConfig(variant="RS1", aspiration=0.) if method == 0 else basat.StableOPTConfig(perturbation_radius=.5)
    assert configuration.gp_backend == "standard" and configuration.gp_training_regularization == 0.
    cls = basat.AdveRS if method == 0 else basat.StableOPT
    algorithm = cls(bounds=basat.Bounds(lower=[-1., -1.], upper=[1., 1.]), config=configuration, dtype="float64", plot=False, verbose=False)
    assert type(algorithm.gp) is basat.GaussianProcess
    state = copy.deepcopy(algorithm.state_dict())
    del state["configuration_snapshot"]["algorithm_config"]["gp_training_regularization"]
    state["configuration_fingerprint"] = hashlib.sha256(json.dumps(state["configuration_snapshot"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    algorithm.load_state_dict(state)
    assert algorithm.config.gp_training_regularization == 0.
