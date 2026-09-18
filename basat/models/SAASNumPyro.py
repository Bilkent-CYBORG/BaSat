"""Basat-owned NumPyro inference for the released MCTS-VS SAAS model.

This is a numerical surrogate, not an experiment runner: it receives only arrays
of observations. The statistical specification follows Eriksson and Jankowiak
(2021) and the MCTS-VS ``inner_optimizer/saasgp.py`` / ``saasbo_component.py``
release (commit 29d32b6977d37f5e257d359373497e375befa96b). No external optimizer
implementation or objective callback is imported. JAX and NumPyro are optional.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _weighted_rbf(
    X: Any, Z: Any, variance: Any, inverse_length_squared: Any, noise: Any, diagonal: bool
) -> Any:
    """Source reduction and dynamic-noise arguments, compiled by each backend instance."""
    import jax.numpy as jnp

    distances = jnp.square(X[:, None, :] - Z) * inverse_length_squared
    covariance = variance * jnp.exp(-0.5 * jnp.sum(distances, axis=-1))
    if diagonal:
        covariance = covariance + (noise + 1e-6) * jnp.eye(X.shape[-2])
    return covariance


class SAASNumPyro:
    """Fit the source RBF SAAS posterior and evaluate pointwise expected improvement.

    Parameters
    ----------
    shrinkage_scale : float
        Half-Cauchy scale of the global inverse-squared-lengthscale multiplier.
    warmup_steps, posterior_samples, thinning, num_chains, max_tree_depth : int
        NUTS controls. Thinning is applied after inference, not to the sampler.
    target_accept_prob : float
        NUTS acceptance target; the released default is 0.8.
    verbose : bool
        Whether NumPyro displays sampling progress.

    Notes
    -----
    Inputs must already be unit scaled; outcomes must use population
    standardization in the minimization orientation. The source uses an RBF
    kernel, zero mean, LogNormal(0, 10) amplitude and total diagonal variance
    2e-6 (observation variance 1e-6 plus the source's 1e-6 nugget). Each fit
    starts from the first split of PRNGKey(0), independently of the trial label.
    Prediction includes the same 2e-6 diagonal term. The class owns no benchmark.
    """

    def __init__(
        self,
        *,
        shrinkage_scale: float,
        warmup_steps: int,
        posterior_samples: int,
        thinning: int,
        num_chains: int,
        max_tree_depth: int,
        target_accept_prob: float,
        verbose: bool,
    ) -> None:
        try:
            import jax
            import numpyro
        except ImportError as error:
            raise ImportError(
                "NumPyro SAAS inference requires the 'basat[numpyro]' optional extra."
            ) from error
        self.jax = jax
        self.numpyro = numpyro
        self.shrinkage_scale = shrinkage_scale
        self.warmup_steps = warmup_steps
        self.posterior_samples = posterior_samples
        self.thinning = thinning
        self.num_chains = num_chains
        self.max_tree_depth = max_tree_depth
        self.target_accept_prob = target_accept_prob
        self.verbose = verbose
        self._kernel = jax.jit(_weighted_rbf, static_argnums=(5,))
        self.X: Any = None
        self.Y: Any = None
        self.samples: dict[str, Any] = {}
        self.diagnostics: dict[str, Any] = {}
        self._cholesky: Any = None

    def kernel(
        self, X: Any, Z: Any, variance: Any, inverse_length_squared: Any, diagonal: bool
    ) -> Any:
        """Evaluate the source weighted RBF kernel in its original reduction order."""
        return self._kernel(X, Z, variance, inverse_length_squared, 1e-6, diagonal)

    def model(self, X: Any, Y: Any) -> None:
        """Define the Basat-owned SAAS prior and Gaussian observation likelihood."""
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        variance = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
        global_scale = numpyro.sample("kernel_tausq", dist.HalfCauchy(self.shrinkage_scale))
        local_scale = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(X.shape[1])))
        inverse_length_squared = numpyro.deterministic(
            "kernel_inv_length_sq", global_scale * local_scale
        )
        covariance = self.kernel(X, X, variance, inverse_length_squared, True)
        numpyro.sample(
            "Y", dist.MultivariateNormal(jnp.zeros(X.shape[0]), covariance_matrix=covariance), obs=Y
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> SAASNumPyro:
        """Run fresh NUTS from the source's fixed per-fit PRNG key.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Unit-cube training inputs.
        Y : ndarray, shape (n,)
            Population-standardized minimization outcomes.

        Returns
        -------
        SAASNumPyro
            This fitted numerical surrogate. Posterior draws remain in memory.
        """
        import jax.numpy as jnp
        from jax import random
        from numpyro.diagnostics import summary
        from numpyro.infer import MCMC, NUTS

        X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or Y.shape != (len(X),) or not len(X):
            raise ValueError("NumPyro SAAS needs nonempty aligned (n,d) inputs and (n,) outcomes.")
        if not np.isfinite(X).all() or not np.isfinite(Y).all() or np.any((X < 0) | (X > 1)):
            raise ValueError("NumPyro SAAS training values must be finite and inputs unit scaled.")
        with self.jax.enable_x64():
            self.X, self.Y = jnp.array(X), jnp.array(Y)
            sampler = NUTS(
                self.model,
                max_tree_depth=self.max_tree_depth,
                target_accept_prob=self.target_accept_prob,
                dense_mass=False,
            )
            mcmc = MCMC(
                sampler,
                num_warmup=self.warmup_steps,
                num_samples=self.posterior_samples,
                num_chains=self.num_chains,
                # One CPU source chain is usual; explicit sequential chains
                # avoid silently requesting unavailable accelerator devices.
                chain_method="sequential",
                progress_bar=self.verbose,
            )
            fit_key, _ = random.split(random.PRNGKey(0), 2)
            mcmc.run(fit_key, self.X, self.Y)
            raw = mcmc.get_samples(group_by_chain=False)
            self.samples = {name: value[:: self.thinning] for name, value in raw.items()}
            chain_divergences = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"])
            self.diagnostics = {
                "backend": "numpyro",
                "fit_seed": 0,
                "fit_key_convention": "jax.random.split(PRNGKey(0),2)[0]",
                "divergences": {
                    f"chain {i}": np.flatnonzero(row).tolist()
                    for i, row in enumerate(chain_divergences)
                },
            }
            if self.posterior_samples >= 4:
                for name, values in summary(raw, prob=0.9, group_by_chain=False).items():
                    self.diagnostics[name] = {
                        key: np.asarray(value).copy() for key, value in values.items()
                    }
            else:
                self.diagnostics["unavailable"] = "At least four draws are needed for diagnostics."
        self._cholesky = None
        return self

    def posterior(self, candidates: np.ndarray) -> tuple[Any, Any]:
        """Return per-draw predictive means and variances, shaped (draws, points)."""
        import jax.numpy as jnp
        from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

        if not self.samples:
            raise RuntimeError("Fit NumPyro SAAS before requesting its posterior.")
        with self.jax.enable_x64():
            candidate_array = jnp.asarray(candidates, dtype=jnp.float64)
            variances = self.samples["kernel_var"]
            inverse_lengths = self.samples["kernel_inv_length_sq"]
            if self._cholesky is None:
                self._cholesky = [
                    cho_factor(self.kernel(self.X, self.X, amplitude, inverse, True), lower=True)[0]
                    for amplitude, inverse in zip(variances, inverse_lengths, strict=True)
                ]
            means, variances_out = [], []
            for amplitude, inverse, chol in zip(
                variances, inverse_lengths, self._cholesky, strict=True
            ):
                cross = self.kernel(candidate_array, self.X, amplitude, inverse, False)
                means.append(jnp.matmul(cross, cho_solve((chol, True), self.Y)))
                solved = solve_triangular(chol, cross.T, lower=True)
                variances_out.append(amplitude + 2e-6 - (solved * solved).sum(axis=0))
            return jnp.stack(means), jnp.stack(variances_out)

    def expected_improvement(self, candidates: np.ndarray) -> Any:
        """Average the released zero-offset pointwise EI across retained draws."""
        import jax.numpy as jnp
        from jax.scipy.stats import norm

        with self.jax.enable_x64():
            mean, variance = self.posterior(candidates)
            sigma = jnp.maximum(jnp.sqrt(variance), 1e-6)
            improvement = self.Y.min() - mean
            ratio = improvement / sigma
            values = improvement * norm.cdf(ratio) + sigma * norm.pdf(ratio)
            return jnp.nan_to_num(values, nan=0.0).mean(axis=0)

    def select_candidates(self, candidates: np.ndarray, count: int = 3) -> np.ndarray:
        """Select the released top-k EI points, including JAX's tie ordering."""
        from jax import lax

        with self.jax.enable_x64():
            _, indices = lax.top_k(self.expected_improvement(candidates), count)
            return np.asarray(candidates)[np.asarray(indices)].copy()
