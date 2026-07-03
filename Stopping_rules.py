"""Stopping rules for Bayesian-optimization algorithms.

This module keeps stopping logic separate from optimizer classes.  It contains
posterior-probability stopping and an epsilon-delta PAC stopping rule for
high-confidence incumbent certification.
"""

import math

import numpy as np
import torch


def risk_tolerance_label(value):
    """Return a stable label such as ``1/256`` for a risk tolerance."""
    value = float(value)
    if value <= 0:
        return str(value)
    denominator = int(round(1.0 / value))
    if denominator > 0 and math.isclose(value, 1.0 / denominator, rel_tol=1e-12, abs_tol=1e-15):
        return f"1/{denominator}"
    return f"{value:.12g}"


def posterior_summary(model, x_one):
    """Return posterior mean, standard deviation, and variance at one point."""
    if model is None:
        return float("nan"), float("nan"), float("nan")

    with torch.no_grad():
        posterior = model.posterior(torch.as_tensor(x_one, dtype=torch.double))
        mean = posterior.mean.reshape(-1)[0]
        variance = torch.clamp(posterior.variance.reshape(-1)[0], min=1e-12)
        std = torch.sqrt(variance)
    return (
        float(mean.detach().cpu().item()),
        float(std.detach().cpu().item()),
        float(variance.detach().cpu().item()),
    )


def posterior_success_probability(model, x, thresholds):
    """Compute P(f(x) >= threshold) for each threshold and candidate point."""
    thresholds = [float(value) for value in thresholds]
    with torch.no_grad():
        posterior = model.posterior(torch.as_tensor(x, dtype=torch.double))
        mu = posterior.mean.reshape(-1)
        latent_std = torch.sqrt(torch.clamp(posterior.variance.reshape(-1), min=1e-12))
        threshold_tensor = torch.tensor(thresholds, dtype=mu.dtype, device=mu.device).view(-1, 1)
        z_score = (threshold_tensor - mu.view(1, -1)) / latent_std.view(1, -1)
        probabilities = 1.0 - torch.distributions.Normal(0, 1).cdf(z_score)
    return probabilities.detach().cpu().numpy()


def should_stop_posterior_probability(model, x, tau, delta, min_iterations_met=True):
    """Return whether posterior probability stopping accepts the candidate."""
    if model is None or not min_iterations_met:
        return False, 0.0
    probabilities = np.asarray(posterior_success_probability(model, x, [tau]), dtype=float).reshape(-1)
    probability = float(np.max(probabilities)) if probabilities.size else 0.0
    return probability >= 1.0 - float(delta), probability


def posterior_stoppable_threshold(mean, std, delta):
    """Return tau such that P(f(x) >= tau) >= 1-delta under a Gaussian posterior."""
    mean_tensor = torch.as_tensor(mean, dtype=torch.double)
    std_tensor = torch.as_tensor(std, dtype=torch.double)
    delta_tensor = torch.as_tensor(
        min(max(float(delta), 1e-12), 1.0 - 1e-12),
        dtype=mean_tensor.dtype,
        device=mean_tensor.device,
    )
    normal = torch.distributions.Normal(
        torch.zeros((), dtype=mean_tensor.dtype, device=mean_tensor.device),
        torch.ones((), dtype=mean_tensor.dtype, device=mean_tensor.device),
    )
    threshold = mean_tensor + std_tensor * normal.icdf(delta_tensor)
    return float(threshold.detach().cpu().item())


def posterior_stoppable_thresholds(model, x_one, risk_tolerances):
    """Return stoppable thresholds for multiple risk tolerances."""
    labels_to_thresholds = {}
    if model is None:
        for delta in risk_tolerances:
            labels_to_thresholds[risk_tolerance_label(delta)] = -float("inf")
        return labels_to_thresholds

    mean, std, _variance = posterior_summary(model, x_one)
    for delta in risk_tolerances:
        labels_to_thresholds[risk_tolerance_label(delta)] = posterior_stoppable_threshold(
            mean,
            std,
            delta,
        )
    return labels_to_thresholds


def epsilon_delta_pac_bound(std, delta, union_count=1):
    """Return a Gaussian high-confidence radius for epsilon-delta PAC stopping."""
    delta = min(max(float(delta), 1e-12), 1.0 - 1e-12)
    union_count = max(int(union_count), 1)
    adjusted_delta = delta / union_count
    normal = torch.distributions.Normal(
        torch.zeros((), dtype=torch.double),
        torch.ones((), dtype=torch.double),
    )
    beta = float(normal.icdf(torch.tensor(1.0 - adjusted_delta, dtype=torch.double)).item())
    return float(std) * beta


def should_stop_epsilon_delta_pac(
    model,
    candidate_points,
    best_observed,
    epsilon,
    delta,
    min_iterations_met=True,
):
    """Return whether the posterior upper bound is within epsilon of the incumbent."""
    if model is None or not min_iterations_met:
        return False, {
            "gap": float("inf"),
            "upper_bound": float("inf"),
            "best_observed": float(best_observed),
        }

    with torch.no_grad():
        posterior = model.posterior(torch.as_tensor(candidate_points, dtype=torch.double))
        means = posterior.mean.reshape(-1)
        stds = torch.sqrt(torch.clamp(posterior.variance.reshape(-1), min=1e-12))

    bounds = []
    for mean, std in zip(means, stds):
        radius = epsilon_delta_pac_bound(
            float(std.detach().cpu().item()),
            delta,
            union_count=max(int(means.numel()), 1),
        )
        bounds.append(float(mean.detach().cpu().item()) + radius)

    upper_bound = max(bounds) if bounds else float("inf")
    gap = float(upper_bound) - float(best_observed)
    return gap <= float(epsilon), {
        "gap": float(gap),
        "upper_bound": float(upper_bound),
        "best_observed": float(best_observed),
        "epsilon": float(epsilon),
        "delta": float(delta),
    }
