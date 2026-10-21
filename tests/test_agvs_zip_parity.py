"""Independent bounded checks against the user's unmodified reference ZIP.

The optional archive is read directly by Python's zip importer; its source is
never extracted into the package and never used by the Basat implementation.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import basat


@pytest.fixture(scope="module")
def reference_optimizer():
    path = Path(os.environ.get(
        "BASAT_AGVS_REFERENCE_ZIP",
        str(Path(__file__).resolve().parents[2] / "AISTATS2027_Anonymous_Code.zip")))
    if not path.is_file():
        pytest.skip("The optional user-provided AG-VS source ZIP is unavailable.")
    source_path = str(path) + "/AISTATS2027_Anonymous_Code"
    sys.path.insert(0, source_path)
    try:
        yield importlib.import_module("ag_vs.optimizer").AGVSOptimizer
    finally:
        sys.path.remove(source_path)


def objective_numpy(x):
    return -float(np.sum((x[:2] - .25) ** 2))


def objective_torch(x):
    return -(x[..., :2] - .25).square().sum(-1)


@pytest.mark.parametrize("backend,budget,dimension,joint,patience,absolute,merge", [
    ("bo", 24, 6, False, 12, True, True),
    ("bo", 60, 6, False, 12, True, True),
    ("turbo", 24, 6, False, 12, True, True),
    ("turbo", 24, 12, False, 12, True, True),
    ("bo", 60, 8, False, 12, True, True),
    ("bo", 24, 6, True, 12, True, True),
    ("bo", 60, 6, False, 1, True, True),
    ("bo", 60, 6, False, 12, False, True),
    ("bo", 60, 6, False, 12, True, False),
])
def test_unmodified_zip_actions_rewards_and_active_tree(
    reference_optimizer, backend, budget, dimension, joint, patience, absolute, merge,
):
    basat.set_seed(2021, threads=1)
    reference = reference_optimizer(
        objective=objective_numpy, bounds=np.tile([0., 1.], (dimension, 1)),
        eff_dim=2, n_split=3, n_init=6, max_iter=budget,
        t_seq=2, batch_size=None if joint else 3, q_batch=3 if joint else None,
        N_bad=patience, abs_mode=absolute, merge_enabled=merge, seed=2021,
        ipt_solver="vanillabo" if backend == "bo" else "turbo",
        risk_tolerances=[.05], printer_on=False, print_tree_each_iter=False, verbose=False)
    reference.optimize()
    basat.set_seed(2021, threads=1)
    algorithm = basat.AG_VS(
        bounds=basat.Bounds([0.] * dimension, [1.] * dimension), dtype="float64",
        config=basat.AG_VSConfig(
            effective_dimension=2, split_groups=3, initial_points=6,
            sequence_length=2, batch_size=None if joint else 3, q_batch=3 if joint else None,
            stagnation_patience=patience, abs_mode=absolute, merge_enabled=merge,
            inner_optimizer=backend, seed=2021))
    algorithm.initialize_from_environment(basat.EnvironmentData(
        actions=torch.empty((0, dimension), dtype=torch.float64),
        observations=torch.empty(0, dtype=torch.float64)))
    while algorithm.evaluations < budget:
        action = algorithm.ask()
        rewards = objective_torch(action)
        algorithm.tell(action=action,
                       feedback={"evaluated_action": action, "observation": rewards})
    np.testing.assert_allclose(algorithm.observed_actions.cpu().numpy(),
                               np.asarray(reference.X), rtol=0, atol=1e-9)
    np.testing.assert_allclose(algorithm.observed_values.cpu().numpy(),
                               np.asarray(reference.y), rtol=0, atol=1e-10)
    actual = sorted((node.depth, node.node_id, tuple(node.active_indices.tolist()))
                    for node in algorithm.state.nodes.values() if not node.archived)
    expected = sorted((node.h, node.idx, tuple(node.active_dimensions))
                      for node in reference._all_nodes if not node.plot_archived)
    assert actual == expected


def test_large_turbo_candidate_pool_matches_isolated_zip_runtime(reference_optimizer):
    """The source runs separately so importing Basat cannot alter its defaults."""
    module = sys.modules[reference_optimizer.__module__]
    source_path = module.__file__.split("/ag_vs/")[0]
    code = """
import json, sys
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])
from ag_vs.optimizer import AGVSOptimizer
optimizer = AGVSOptimizer(
    objective=lambda x: -float(np.sum((x[:2]-.25)**2)),
    bounds=np.tile([0.,1.], (30,1)), eff_dim=2, n_split=3,
    n_init=6, max_iter=12, t_seq=2, batch_size=3, N_bad=12,
    seed=2021, ipt_solver='turbo', risk_tolerances=[.05],
    printer_on=False, print_tree_each_iter=False, verbose=False)
optimizer.optimize()
print(json.dumps({'actions': np.asarray(optimizer.X).tolist(), 'reward': optimizer.y}))
"""
    process = subprocess.run([sys.executable, "-c", code, source_path],
                             check=True, text=True, capture_output=True)
    expected = json.loads(process.stdout)
    basat.set_seed(2021, threads=1)
    algorithm = basat.AG_VS(
        bounds=basat.Bounds([0.] * 30, [1.] * 30), dtype="float64",
        config=basat.AG_VSConfig(
            effective_dimension=2, split_groups=3, initial_points=6,
            sequence_length=2, batch_size=3, stagnation_patience=12,
            inner_optimizer="turbo", seed=2021))
    algorithm.initialize_from_environment(basat.EnvironmentData(
        actions=torch.empty((0, 30), dtype=torch.float64),
        observations=torch.empty(0, dtype=torch.float64)))
    while algorithm.evaluations < 12:
        action = algorithm.ask()
        algorithm.tell(action=action, feedback={
            "evaluated_action": action, "observation": objective_torch(action)})
    np.testing.assert_allclose(algorithm.observed_actions.numpy(), expected["actions"],
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(algorithm.observed_values.numpy(), expected["reward"],
                               rtol=0, atol=1e-10)
