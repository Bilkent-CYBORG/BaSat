"""Example script for running APREMBO on a simple synthetic objective.

Put this file in the same project root where the `APREMBO/` package is
importable, then run:

    python example_aprembo.py

The objective is a 25-dimensional function with only the first two dimensions
active:

    f(x) = -(x_0^2 + x_1^2)

APREMBO maximizes the objective. The global maximum is 0 at x = 0.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from APREMBO import APREMBO


class ActiveSphere:
    """Maximization benchmark: negative Sphere on a low-dimensional active subspace."""

    def __init__(self, active_indices=(0, 1)):
        self.active_indices = np.asarray(active_indices, dtype=int)
        self.noisy_observations = []

    def __call__(self, x_real: np.ndarray) -> float:
        """Evaluate f(x). APREMBO passes x with shape (1, D) or (D,)."""
        x = np.asarray(x_real, dtype=float).reshape(-1)
        x_active = x[self.active_indices]
        return -float(np.sum(x_active ** 2))

    def record_noisy_observation(self, x_real: np.ndarray, y_raw: float, y_noisy: float) -> None:
        """Optional hook used by APREMBO to store noisy observations."""
        self.noisy_observations.append(
            {
                "x": np.asarray(x_real, dtype=float).reshape(-1).copy(),
                "y_raw": float(y_raw),
                "y_noisy": float(y_noisy),
            }
        )


def flatten_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    """Make APREMBO's nested threshold-trace rows CSV-friendly."""
    flat = dict(row)
    for key in [
        "x",
        "z",
        "recommended_x",
        "recommended_z",
        "current_stoppable_threshold_by_delta",
        "best_stoppable_threshold_by_delta",
    ]:
        if key in flat:
            flat[key] = repr(flat[key])
    return flat


def save_threshold_trace(optimizer: APREMBO, output_path: Path) -> None:
    trace = getattr(optimizer, "threshold_trace", [])
    if not trace:
        return

    rows = [flatten_trace_row(row) for row in trace]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_optimizer(args: argparse.Namespace) -> APREMBO:
    D = args.dimension
    d_embedding = args.d_embedding

    # Real high-dimensional box. APREMBO internally normalizes this to [-1, 1]^D.
    original_boundaries = np.array([[-2.0, 2.0]] * D, dtype=float)

    # For f(x)=-(x0^2+x1^2) on [-2,2]^2, values are in approximately [-8, 0].
    objective_lower_bound = -8.0
    objective_upper_bound = 0.0

    optimum_x = np.zeros(D, dtype=float)
    active_indices = np.arange(d_embedding, dtype=int)

    return APREMBO(
        original_boundaries=original_boundaries,
        d_embedding=d_embedding,
        max_iter=args.max_iter,
        tau=args.tau,
        N=args.children,
        delta=args.delta,
        batch_size=1,
        n_init=args.n_init,
        show=False,
        verbose=True,
        split=True,
        embedding_radius=np.sqrt(d_embedding),
        a=objective_lower_bound,
        b=objective_upper_bound,
        R=args.noise_std,
        optimum_x=optimum_x,
        kernel_name=args.kernel,
        auto_select_kernel=args.auto_select_kernel,
        kernel_candidates=("rbf", "matern12", "matern32", "matern52"),
        lipschitz_mode=args.lipschitz_mode,
        L=args.fixed_lipschitz if args.lipschitz_mode == "user" else None,
        lengthscale_lower_bound=0.01,
        lengthscale_upper_bound=50.0,
        lengthscale_reoptimize_interval=20,
        t_sigma=0.002,
        exploitation_patience=5,
        D_min=1,
        effective_dim=d_embedding,
        active_indices=active_indices,
        threshold_trace_risk_tolerances=[args.delta],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run APREMBO on a simple active-subspace Sphere benchmark.")
    parser.add_argument("--dimension", type=int, default=25, help="Original ambient dimension D.")
    parser.add_argument("--d-embedding", type=int, default=2, help="REMBO embedded dimension.")
    parser.add_argument("--max-iter", type=int, default=60, help="Total objective-evaluation budget.")
    parser.add_argument("--n-init", type=int, default=4, help="Initial random/LHS evaluations.")
    parser.add_argument("--children", type=int, default=2, help="Number of child cells per split.")
    parser.add_argument("--tau", type=float, default=-1e-3, help="Posterior stopping threshold.")
    parser.add_argument("--delta", type=float, default=1.0 / 16.0, help="Stopping risk tolerance.")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Gaussian observation-noise standard deviation.")
    parser.add_argument("--kernel", type=str, default="matern52", help="Kernel used when auto selection is disabled.")
    parser.add_argument("--auto-select-kernel", action="store_true", help="Try multiple kernels and select by marginal likelihood.")
    parser.add_argument(
        "--lipschitz-mode",
        choices=("predict", "user"),
        default="predict",
        help="Use GP mean-gradient L estimate or a fixed user L.",
    )
    parser.add_argument("--fixed-lipschitz", type=float, default=1.0, help="Used only with --lipschitz-mode user.")
    parser.add_argument("--trace-csv", type=Path, default=Path("aprembo_threshold_trace.csv"))
    parser.add_argument("--verbose", action="store_true", help="Print per-iteration APREMBO progress.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(0)

    objective = ActiveSphere(active_indices=tuple(range(args.d_embedding)))
    optimizer = build_optimizer(args)

    result = optimizer.optimize(objective)

    print("\n=== APREMBO result ===")
    print(f"raw result tuple: {result}")
    print(f"best true score: {optimizer.best_score:.6f}")
    print(f"best observed score: {optimizer.best_observed_score:.6f}")
    print(f"selected kernel: {getattr(optimizer, 'selected_kernel_name', None)}")
    print(f"estimated / current embedded Lipschitz L_emb: {getattr(optimizer, 'L_emb', None)}")
    print(f"best x first {args.d_embedding} dims: {optimizer.best_x[:args.d_embedding] if optimizer.best_x is not None else None}")
    print(f"best z: {optimizer.best_z}")

    save_threshold_trace(optimizer, args.trace_csv)
    if args.trace_csv.exists():
        print(f"threshold trace saved to: {args.trace_csv.resolve()}")


if __name__ == "__main__":
    main()
