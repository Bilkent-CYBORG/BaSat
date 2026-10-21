"""Independent source fixtures for the opt-in dense numerical protocol.

The first six fixtures also exercise the ordinary current runtime. Exact
replay of the demonstrated one-ULP tie requires the separately pinned source
numerical stack; no seed, acquisition epsilon, or expected outcome is tuned.
"""
import json
from pathlib import Path

import basat
import numpy as np
import pytest
import scipy
import sklearn

from basat.algorithms.algorithm import AlgorithmStateError

from test_advers_camera_ready_lcb_execution import (
    FIXTURE, make, metric, rng_digest, trajectory,
)


@pytest.mark.parametrize("method", range(6))
def test_dense_fixed_source_pilot_matches_independent_official_fixture(method):
    expected = json.loads(FIXTURE.read_text())["records"][method]
    experiment = make(method, gp_backend="fixed_gpy")
    result = experiment.run(max_rounds=100, regret=metric(), plot=False, show=False)
    actual = trajectory(experiment)
    assert [r[0] for r in actual] == expected["proposed"]
    assert [r[1] for r in actual] == expected["evaluated"]
    np.testing.assert_allclose([r[2] for r in actual], expected["observations"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(result.metadata["regret"]["values"],
                               expected["cumulative_rs_regret"], rtol=0, atol=1e-8)
    assert [r[4] for r in actual] == expected["rng_after_feedback_sha256"]
    assert rng_digest(experiment.algorithm.get_source_random_state()) == expected["final_rng_sha256"]


@pytest.mark.skipif(
    (np.__version__, scipy.__version__, sklearn.__version__) != ("1.26.4", "1.12.0", "1.7.2"),
    reason="Exact one-ULP acquisition tie replay requires the documented advers-source numerical runtime.",
)
def test_first_divergence_repetition020_matches_all_100_source_decisions():
    path = Path(__file__).parent / "fixtures/advers_2026_lcb_rs1_repetition020.json"
    expected = json.loads(path.read_text())
    experiment = make(0, repetition=20, gp_backend="fixed_gpy")
    assert experiment.algorithm.seed == expected["seed"] == 3989069128
    result = experiment.run(max_rounds=100, regret=metric(), plot=False, show=False)
    actual = trajectory(experiment)
    assert actual[11][0] == expected["proposed"][11] == 460
    assert [r[0] for r in actual] == expected["proposed"]
    assert [r[1] for r in actual] == expected["evaluated"]
    np.testing.assert_allclose([r[2] for r in actual], expected["observations"], rtol=0, atol=1e-9)
    np.testing.assert_allclose([r[3] for r in actual], expected["latent_observations"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(result.metadata["regret"]["values"],
                               expected["cumulative_rs_regret"], rtol=0, atol=1e-8)
    assert [r[4] for r in actual] == expected["rng_after_feedback_sha256"]
    assert rng_digest(experiment.algorithm.get_source_random_state()) == expected["final_rng_sha256"]


@pytest.mark.parametrize("method", [0, 4])
@pytest.mark.parametrize("phase", ["initial_design", "initial_observation", "initialized",
                                  "pending_proposal", "pre_tell", "post_tell"])
def test_dense_source_interrupted_resume_and_completed_noop(tmp_path, monkeypatch, method, phase):
    reference = make(method, gp_backend="fixed_gpy")
    reference_result = reference.run(max_rounds=4, regret=metric(), plot=False, show=False)
    interrupted = make(method, directory=tmp_path, gp_backend="fixed_gpy")
    save = interrupted.save_checkpoint

    def crash(**kwargs):
        result = save(**kwargs)
        if kwargs.get("phase") == phase:
            raise RuntimeError("injected dense-source interruption")
        return result

    monkeypatch.setattr(interrupted, "save_checkpoint", crash)
    with pytest.raises(RuntimeError, match="injected dense-source"):
        interrupted.run(max_rounds=4, regret=metric(), plot=False, show=False)
    resumed = make(method, directory=tmp_path, resume=True, gp_backend="fixed_gpy")
    result = resumed.run(max_rounds=4, regret=metric(), plot=False, show=False)
    assert trajectory(resumed) == trajectory(reference)
    assert resumed.algorithm.get_source_random_state() == reference.algorithm.get_source_random_state()
    assert result.metadata["regret"]["values"] == reference_result.metadata["regret"]["values"]
    monkeypatch.setattr(resumed.environment, "evaluate_function", lambda *a: pytest.fail("new query"))
    monkeypatch.setattr(resumed.algorithm, "ask", lambda *a, **kw: pytest.fail("new proposal"))
    resumed.run(max_rounds=4, regret=metric(), plot=False, show=False)
    assert trajectory(resumed) == trajectory(reference)


def test_dense_source_does_not_resume_feature_backend_checkpoint(tmp_path):
    legacy = make(directory=tmp_path)
    legacy.run(max_rounds=1, regret=metric(), plot=False, show=False)
    with pytest.raises(AlgorithmStateError, match="(?i)(configuration|fingerprint|kernel|backend|checkpoint)"):
        changed = make(directory=tmp_path, resume=True, gp_backend="fixed_gpy")
        changed.run(max_rounds=1, regret=metric(), plot=False, show=False)
