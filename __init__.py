"""Reusable Bayesian-optimization helpers.

This package is intentionally not tied to REMBO-UCB.  It provides kernels,
stopping rules, sampling, domain transforms, embedding projections, and
discrete partition utilities that optimizers can import and re-export.
"""

from .kernel import (
    SUPPORTED_BASE_KERNELS,
    SUPPORTED_EMBEDDED_KERNELS,
    InducedEmbeddingKernel,
    build_base_kernel,
    build_embedded_kernel,
    build_induced_kernel,
    build_normal_kernel,
    build_rembo_ucb_kernel,
)
from .Stopping_rules import (
    epsilon_delta_pac_bound,
    posterior_stoppable_threshold,
    posterior_stoppable_thresholds,
    posterior_success_probability,
    posterior_summary,
    risk_tolerance_label,
    should_stop_epsilon_delta_pac,
    should_stop_posterior_probability,
)
from .utils import (
    embed_original_to_embedded,
    estimate_lipschitz_from_gp_mean,
    make_random_embedding_matrix,
    project_embedded_to_original,
    project_original_to_embedded,
    select_best_kernel_by_mll,
)
from .APREMBO import APREMBO, GaussianProcess

__all__ = [
    "APREMBO",
    "GaussianProcess",
    "SUPPORTED_BASE_KERNELS",
    "SUPPORTED_EMBEDDED_KERNELS",
    "InducedEmbeddingKernel",
    "build_base_kernel",
    "build_embedded_kernel",
    "build_induced_kernel",
    "build_normal_kernel",
    "build_rembo_ucb_kernel",
    "epsilon_delta_pac_bound",
    "posterior_stoppable_threshold",
    "posterior_stoppable_thresholds",
    "posterior_success_probability",
    "posterior_summary",
    "risk_tolerance_label",
    "should_stop_epsilon_delta_pac",
    "should_stop_posterior_probability",
    "embed_original_to_embedded",
    "estimate_lipschitz_from_gp_mean",
    "make_random_embedding_matrix",
    "project_embedded_to_original",
    "project_original_to_embedded",
    "select_best_kernel_by_mll",
]
